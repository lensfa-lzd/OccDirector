import concurrent.futures
import os
import random

import dotenv
from tqdm import tqdm

from data_engine.agent_generator.plan_scenes import text_to_scene

dotenv.load_dotenv()


def batch_run(workers=4, save_path='output/ttsg_auto', count=1600):
    # 定义环境池
    environments = [
        "Urban Canyon",
        "Highway Ramp",
        "Rural Roundabout",
        "Construction Zone",
        "Busy Intersection",
        "Tunnel"
    ]

    # 定义事件类型池 (这里可以用字典来包含子类别描述，给 LLM 更多灵感)
    events = {
        "VRU Interaction": [
            "Jaywalking pedestrian",
            "cyclist swerving",
            "child chasing a ball"
        ],
        "Vehicle Conflict": [
            "Aggressive cut-in",
            "emergency braking ahead",
            "blind-spot merge",
            "oncoming traffic in wrong lane",
            "vehicle running a red light/stop sign",  # 闯红灯/不停
            "sudden U-turn",  # 突然掉头
            "tailgating vehicle",  # 后车紧跟
        ],
        "Occlusion": [
            "Bus blocking view of intersection",
            "large truck obscuring traffic signs",
            "buildings blocking corners"
        ],
        "Special/Emergency": [
            "Ambulance/Police car passing",
            "road debris, animal crossing",
            "malfunctioning traffic light"
        ],
    }

    tasks = []
    model_name = os.getenv('OPENAI_MODEL')
    for i in range(count):
        # 1. 随机选环境
        selected_env = random.choice(environments)

        # 2. 随机选大类事件
        selected_event_type, selected_event_context_list = random.choice(list(events.items()))

        # 3. 获取该事件的子描述 (作为 Context 给 LLM)
        selected_event_context = random.choice(selected_event_context_list)

        input_prompt = f"""
        Generate ONE unique driving scenario based on the STRICT constraints below.

        **Constraints (Fixed by System)**
        The scenario MUST take place in the following setting:
        1.  **Environment:** {selected_env}
        2.  **Event Category:** {selected_event_type}
            *   *Examples/Inspiration for this category:* {selected_event_context}
        
        seed: {random.randint(100_0000, 999_9999)}
        """

        save_dir = f"{save_path}/auto/{i + 1}"
        tasks.append((
            input_prompt,
            model_name,
            save_dir
        ))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(text_to_scene, *task): task
            for task in tasks
        }

        with tqdm(total=len(tasks)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Randomized prompt synthesis feeding plan_scenes.text_to_scene')
    parser.add_argument('--save-path', default='output/ttsg_auto')
    parser.add_argument('--count', type=int, default=1600)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    batch_run(args.workers, args.save_path, args.count)
