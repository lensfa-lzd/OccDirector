# Derived from https://github.com/basiclab/TTSG (Text-to-Traffic-Scene Generation, arXiv 2409.09575),
# modified for OccInteract-Agent occupancy collection.
import json
import os
import random
import time

import imageio
import numpy as np
from colorama import Fore, Style
from tqdm import tqdm

from data_engine.agent_generator.scene_utils.scene_client import CarlaClient

EGO_SETUP = {
    "type": "car",
    "is_ego": True,
    "relative_to_ego": "none",
    "road_type": "driving",
    "behavior": "cautious",
}


def create_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def scene_generation(
        retreival,
        planning,
        save_dir,
        use_cache=True,
        cache_dir=None,
        ip_address="localhost",
        port=2000,
        map_folder=None,
        return_ego=True,
        override_option=False,
        max_save_time=15,  # 秒
        occ_save_fps=2,
        device='cuda',
        make_done_tag=False,
        batch_generation=False,
        rpc_timeout=120.0,
):
    if map_folder is None:
        map_folder = os.path.join(os.path.dirname(__file__), '..', 'maps')
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'graph_cache')

    carla_client = CarlaClient(
        input_folder=map_folder,
        host=ip_address,
        port=port,
        use_cache=use_cache,
        cache_dir=cache_dir,
        override_option=override_option,
        device=device,
        rpc_timeout=rpc_timeout,
    )
    agent_planning = planning["agents"] + ([EGO_SETUP] if not return_ego else [])
    action_list = set()
    road_type_list = set()
    agent_type_list = set()
    for agent in agent_planning:
        action_list.add((agent["action"], agent["relative_to_ego"]))
        road_type_list.add(agent["road_type"])
        agent_type_list.add(agent["type"])
    town_name, road_id, direction, road_info = carla_client.get_valid_road(
        retreival,
        list(agent_type_list),
        list(action_list),
        list(road_type_list),
    )

    if town_name is None:
        print(f"{Fore.RED}No valid road found{Style.RESET_ALL}")
        return

    print(
        f"{Style.BRIGHT}{Fore.YELLOW}Spawn information{Style.RESET_ALL}: {town_name}, {road_id}, {direction}"
    )
    print(f"{Style.BRIGHT}{Fore.YELLOW}Road information{Style.RESET_ALL}: {road_info}")

    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/road_info.json", "w") as f:
        json.dump(
            {
                "town": town_name,
                "road_id": road_id,
                "direction": direction,
                "road_info": road_info,
            },
            f,
            indent=2,
        )

    carla_client.load_map(town_name, weather=planning["env"]["weather"])
    carla_client.spawn_all_agent(
        road_id[0],
        agent_planning,
        direction=direction,
        at_junction=planning["env"]["at_junction"],
    )

    delta = carla_client.get_world_delta()
    fps = int(1.0 / delta)
    max_save_frames = max_save_time // delta

    occ_save_interval = fps / occ_save_fps
    if int(occ_save_interval) != occ_save_interval:
        print(f'请让 Occ 存储帧率 {occ_save_fps} 能被真实帧率 {fps} 整除')
        raise ValueError
    occ_save_interval = int(occ_save_interval)

    if batch_generation:
        p_bar = None
    else:
        p_bar = tqdm(total=int(max_save_frames), desc='场景采集进度')

    os.makedirs(f"{save_dir}/occupancy", exist_ok=True)
    bev_video_writer = imageio.get_writer(f"{save_dir}/bev.mp4", fps=fps, macro_block_size=None)
    front_video_writer = imageio.get_writer(f"{save_dir}/front.mp4", fps=fps, macro_block_size=None)

    print(f"{Style.BRIGHT}{Fore.YELLOW} >>> 物理帧率: {fps} Hz, Occupancy 存储帧率: {occ_save_fps} Hz{Style.RESET_ALL}")
    try:
        occ_save_count = 0
        save_frame = 0
        count_frame = 1
        while True:
            done, data = carla_client.check_finish()
            if count_frame > 10:
                for sensor_name, sensor_data in data.items():
                    if sensor_data is not None:
                        try:
                            if sensor_name == 'front_image':
                                front_video_writer.append_data(sensor_data)

                            if sensor_name == 'bev_image':
                                bev_video_writer.append_data(sensor_data)

                            if sensor_name == 'lidar_data':
                                if count_frame % occ_save_interval == 0:
                                    # 量化 lidar 点云的过程比较占用资源
                                    # 仅在需要保存的时候执行量化
                                    occupancy = carla_client.semantic_lidar_to_occupancy(sensor_data)
                                    occupancy = np.flip(occupancy, axis=1)
                                    np.save(f"{save_dir}/occupancy/{save_frame}.npy", occupancy)
                                    occ_save_count += 1

                        except Exception as e:
                            print(str(e))

                save_frame += 1
                if p_bar is not None:
                    p_bar.update(1)

            if done:
                if make_done_tag:
                    f = open(f'{save_dir}/done', 'w')
                    f.close()
                break

            if save_frame >= max_save_frames:
                if occ_save_count > 0 and make_done_tag:
                    f = open(f'{save_dir}/done', 'w')
                    f.close()
                break

            count_frame += 1
            time.sleep(0.02)

    finally:
        print(f'{Style.BRIGHT}{Fore.YELLOW} 清理资源...{Style.RESET_ALL}')
        bev_video_writer.close()
        front_video_writer.close()
        carla_client.destroy()
