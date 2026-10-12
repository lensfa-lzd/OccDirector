# Derived from https://github.com/basiclab/TTSG (Text-to-Traffic-Scene Generation, arXiv 2409.09575),
# modified for OccInteract-Agent occupancy collection.
import json
import os
import time
from glob import glob
from pathlib import Path

from tqdm import tqdm
from timeout_decorator import timeout
from data_engine.agent_generator.misc.create_scene_from_json import scene_generation


def demo():
    with open(
            'output/scene_config/case1/agent_output.json',
            "r"
    ) as f:
        data = json.load(f)

    # 具有随机性，有可能会生成失败，重新运行即可
    scene_generation(
        data["retreival"],
        data["planning"],
        save_dir='output/case1',
        max_save_time=10,
    )

    # 未能解决车辆碰撞消失问题


# use_signals=False 使用子进程，确保没有内存溢出
@timeout(600, use_signals=False)
def scene_generation_timeout_warper(*args, **kwargs):
    return scene_generation(*args, **kwargs)


def check_finished(save_dir):
    def _check_file(file_path):
        if os.path.exists(file_path):
            return True
        else:
            return False

    files = [
        'agent_output.json',
        'prompt.txt',
        'road_info.json',

        'bev.mp4',
        'front.mp4',
    ]
    for file in files:
        if not _check_file(os.path.join(save_dir, file)):
            return False

    return True


def batch_run(args):
    base_path = args.source_path
    save_path = args.save_path

    num_finished = 0
    tasks = []
    for prompt_type_path in glob(f'{base_path}/*'):
        p = Path(prompt_type_path)
        prompt_type = p.name
        for try_instance_path in glob(f'{prompt_type_path}/*'):
            pp = Path(try_instance_path)
            try_instance = pp.name

            config_file = f'{try_instance_path}/agent_output.json'
            prompt_file = f'{try_instance_path}/prompt.txt'

            with open(config_file, 'r') as f:
                data = json.load(f)

            retreival = data['retreival']
            planning = data['planning']

            # 每个视频生成多次
            for i in range(args.videos_per_task):
                save_dir = f'{save_path}/{prompt_type}/{try_instance}/video_{i}'
                if check_finished(save_dir):
                    num_finished += 1
                else:
                    tasks.append({
                        'retreival': retreival,
                        'planning': planning,
                        'save_dir': save_dir,
                        'config_file': config_file,
                        'prompt_file': prompt_file
                    })

    print(f'检测到已经完成任务 {num_finished}')
    print(f'还有 {len(tasks)} 个任务待完成')
    p_bar = tqdm(total=len(tasks), desc='任务执行进度')
    for i, task in enumerate(tasks):
        save_dir = task['save_dir']
        print(f'正在处理 {save_dir}')

        try_count = 1
        while try_count <= 3:
            save_dir = task['save_dir']

            # 清理可能存在的文件
            if os.path.exists(f'{save_dir}/occupancy'):
                os.system(f"rm -r {save_dir}/occupancy")

            try:
                scene_generation_timeout_warper(
                    task["retreival"],
                    task["planning"],
                    save_dir=save_dir,
                    max_save_time=args.max_save_time,
                    make_done_tag=True,
                    device=args.device,
                    batch_generation=False
                )
            except Exception as e:
                print(e)

            # scene_generation 完成了生成任务
            if os.path.exists(f'{save_dir}/done'):
                # 复制生成信息文件
                os.system(f"cp {task['config_file']} {save_dir}")
                os.system(f"cp {task['prompt_file']} {save_dir}")

                # 移除 done tag
                os.system(f"rm {save_dir}/done")
                print(f'{i}: {save_dir} 已经完成')
                break

            # 需要等待时间避免服务崩溃
            time.sleep(5)
            try_count += 1

        p_bar.update(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Single-host rendering of planned scenes')
    parser.add_argument('--source-path', required=True,
                        help='planned-scene tree containing agent_output.json per scene')
    parser.add_argument('--save-path', required=True, help='output root for rendered videos')
    parser.add_argument('--videos-per-task', type=int, default=5)
    parser.add_argument('--max-save-time', type=float, default=10.0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    batch_run(args)
