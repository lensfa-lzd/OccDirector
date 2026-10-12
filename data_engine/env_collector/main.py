"""Batch CARLA scene collection for OccInteract-Env.

Spawns ego vehicles with background traffic on remote CARLA 0.9.15 servers,
records front/BEV videos and converts the semantic-lidar grid into UniOcc-format
200x200x16 occupancy at 2 Hz.

Usage (from the repo root, after starting CARLA servers and writing hosts.json):

    python -m data_engine.env_collector.main \
        --save-dir output/batch_output --hosts data_engine/env_collector/hosts.json
"""
import argparse
import glob
import multiprocessing
import os
import time

import imageio
import numpy as np
import torch
from timeout_decorator import timeout
from tqdm import tqdm

from data_engine.env_collector.carla_utils import CarlaClient
from data_engine.env_collector.config import MAPS
from data_engine.hosts import load_host_resources


def scene_generation(arg, batch_generation=False):
    timeout = arg['timeout']

    delta = 0.1
    world_fps = 1 / delta
    save_dir = arg['save_dir']
    os.makedirs(f"{save_dir}/occupancy", exist_ok=True)
    bev_video_writer = imageio.get_writer(f"{save_dir}/bev.mp4", fps=world_fps, macro_block_size=None)
    front_video_writer = imageio.get_writer(f"{save_dir}/front.mp4", fps=world_fps, macro_block_size=None)

    client = None
    try:
        client = CarlaClient(arg, delta)
        save_time = arg['save_time']
        occ_save_fps = arg['occ_save_fps']
        total_occ_save_count = int(save_time * occ_save_fps)

        occ_save_interval = world_fps / occ_save_fps
        if int(occ_save_interval) != occ_save_interval:
            print(f'请让 Occ 存储帧率 {occ_save_fps} 能被真实帧率 {world_fps} 整除')
            raise ValueError
        occ_save_interval = int(occ_save_interval)

        if batch_generation:
            p_bar = None
        else:
            p_bar = tqdm(total=int(total_occ_save_count), desc='场景采集进度')

        cur_occ_save_count = 0
        count_frame = 0
        save_frame = 0
        while cur_occ_save_count < total_occ_save_count:
            # 使用 client 中的 world 进行 tick
            client.world.tick()

            # 从 client 的队列中获取数据
            if not client.front_camera_queue.empty():
                front_image = client.front_camera_queue.get(True, timeout)
            else:
                front_image = None

            if not client.bev_camera_queue.empty():
                bev_image = client.bev_camera_queue.get(True, timeout)
            else:
                bev_image = None

            if bev_image is not None and front_image is not None:
                # 使用 client 封装的方法获取 lidar 数据列表
                lidar_data = client.get_lidar_data()

                if len(lidar_data) == 0:
                    lidar_data = None
            else:
                lidar_data = None

            data = {
                "front_image": front_image,
                "bev_image": bev_image,
                'lidar_data': lidar_data
            }

            # 跳过开始的 5 秒
            if count_frame > 5 * world_fps:
                save_frame += 1
                for sensor_name, sensor_data in data.items():
                    if sensor_data is not None:
                        try:
                            if sensor_name == 'front_image':
                                # 调用 CarlaClient 的静态方法
                                front_video_writer.append_data(CarlaClient.parse_image(sensor_data))

                            if sensor_name == 'bev_image':
                                bev_video_writer.append_data(CarlaClient.parse_image(sensor_data))

                            if sensor_name == 'lidar_data':
                                if count_frame % occ_save_interval == 0:
                                    # 量化 lidar 点云的过程比较占用资源
                                    # 仅在需要保存的时候执行量化
                                    # 使用 client 的方法直接计算，参数已在内部维护
                                    occupancy = client.semantic_lidar_to_occupancy(sensor_data)
                                    occupancy = np.flip(occupancy, axis=1)
                                    np.save(f"{save_dir}/occupancy/{save_frame}.npy", occupancy)
                                    cur_occ_save_count += 1
                                    if p_bar:
                                        p_bar.update(1)

                        except Exception as e:
                            print(str(e))

            count_frame += 1
            time.sleep(0.02)

    except Exception as e:
        print('scene_generation: ', e)

    finally:
        bev_video_writer.close()
        front_video_writer.close()

        # 使用 client 的 destroy 方法统一清理
        if client:
            client.destroy()
            time.sleep(3)


# use_signals=False 使用子进程，确保没有内存溢出
@timeout(600, use_signals=False)
def scene_generation_timeout_warper(*args, **kwargs):
    return scene_generation(*args, **kwargs)


def check_finished(save_dir, occ_save_count):
    occ_files = glob.glob(f'{save_dir}/occupancy/*.npy')
    if len(occ_files) >= occ_save_count:
        return True
    else:
        return False


def worker_run_tasks(worker_id, tasks, progress_queue, start_delay=0):
    """
    每个 Worker 进程执行的函数
    :param worker_id: 进程 ID
    :param tasks: 该进程分配到的任务列表
    :param progress_queue: 用于向主进程汇报进度的队列
    :param start_delay: 启动前的等待秒数
    """

    # --- 延迟启动逻辑 ---
    if start_delay > 0:
        print(f'[Worker {worker_id}] 将延迟 {start_delay} 秒后启动...')
        time.sleep(start_delay)
    # -----------------------

    for task in tasks:
        save_dir = task['save_dir']
        # 为了避免多进程 print 混乱，可以加个前缀或直接注释掉
        print(f'[Worker {worker_id}] 正在处理 {task}')

        try_count = 1
        while try_count <= 3:
            # 清理可能存在的文件
            if os.path.exists(f'{save_dir}/occupancy'):
                os.system(f"rm -r {save_dir}/occupancy")

            try:
                # 执行生成任务
                scene_generation_timeout_warper(
                    task,
                    batch_generation=True
                )

                # 检查结果
                occ_save_count = task['save_time'] * task['occ_save_fps']
                if not check_finished(save_dir, occ_save_count):
                    raise ValueError(f'生成失败 {task}')

                # 如果成功，跳出重试循环
                break

            except Exception as e:
                # 打印错误信息
                print(f'[Worker {worker_id}] Error in {save_dir}: {e}')

            # 需要等待时间避免服务崩溃
            time.sleep(5)
            try_count += 1
            print(f'[Worker {worker_id}] 重新执行 {task}')

        # 无论成功还是失败（超过重试次数），都向队列发送完成信号，以便更新进度条
        # 如果你想只统计成功的，可以加判断
        progress_queue.put(1)


def batch_run(args):
    save_path = args.save_dir

    # 一个视频 1 分钟
    # 一个地图生成 600 个视频

    # --- 自动检测 CUDA ---
    cuda_count = torch.cuda.device_count()
    if cuda_count > 0:
        print(f"检测到 {cuda_count} 个可用 CUDA 设备，将自动均衡分配。")
    else:
        print("未检测到 CUDA 设备，将使用 CPU。")

    host_resources = load_host_resources(args.hosts)

    # --- 资源解析与 Worker 生成 ---
    # 根据 host_resources 生成所有可用的 worker slots (每个 slot 对应一个 Host 和 Port)
    worker_slots = []
    for h_res in host_resources:
        h_ip = h_res['ip']
        h_workers = h_res['workers']
        h_base_port = h_res['base_port']

        for i in range(h_workers):
            worker_slots.append({
                "host": h_ip,
                # 注意：这里计算端口是基于该 Host 内部的索引 i，而不是全局 worker_id
                "port": int(h_base_port[0] + 100 * i),
                "tm_port": int(h_base_port[1] + 10 * i)
            })

    # 动态计算总 Worker 数量
    num_works = len(worker_slots)
    if num_works == 0:
        print("错误：hosts 文件中没有配置有效的 Host 资源。")
        return

    print(f"根据配置，共初始化 {num_works} 个并行 Worker，分布在 {len(host_resources)} 台 Host 上。")

    all_tasks = []
    for map_path in MAPS:
        for i in range(args.scenes_per_map):
            map_name = map_path.split('/')[-1]
            # 在这里我们暂时不指定 device，等到分配给具体 worker 时再指定
            all_tasks.append(
                {
                    "save_dir": f"{save_path}/{map_name}/{i}",

                    "timeout": args.rpc_timeout,
                    "new_world": map_path,
                    "save_time": args.save_time,
                    "occ_save_fps": args.occ_save_fps,
                }
            )

            # 分配 job
    num_todo = 0
    num_finished = 0
    tasks_by_workers = [[] for _ in range(num_works)]

    for i, task in enumerate(all_tasks):
        if check_finished(task['save_dir'], task['save_time'] * task['occ_save_fps']):
            num_finished += 1
        else:
            # 轮询分配给 Worker
            worker_id = i % num_works

            # 获取该 Worker 对应的资源槽位
            slot = worker_slots[worker_id]

            # 自动分配 Device: 根据 worker_id 对 GPU 数量取模
            # 这样同一个 Worker 进程始终使用同一块 GPU，避免上下文切换开销
            if cuda_count > 0:
                cur_device = f'cuda:{worker_id % cuda_count}'
            else:
                cur_device = 'cpu'

            # 更新任务信息
            task.update({
                "host": slot['host'],
                "port": slot['port'],
                "tm_port": slot['tm_port'],
                "device": cur_device
            })

            tasks_by_workers[worker_id].append(task)
            num_todo += 1

    print(f'检测到已经完成任务 {num_finished}')
    print(f'还有 {num_todo} 个任务待完成')

    if num_todo == 0:
        print("所有任务已完成。")
        return

    # --- 并行化 ---

    # 使用 Manager Queue 来处理跨进程的进度条通信
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()

    processes = []

    # === 定义间隔时间 ===
    startup_interval = args.startup_interval  # 每个进程间隔几秒启动
    # =========================

    # 启动子进程
    print(f"启动 {num_works} 个 Worker 并行处理 (启动间隔: {startup_interval}s)...")
    for worker_id in range(num_works):
        if not tasks_by_workers[worker_id]:
            continue

        # === 计算延迟并传参 ===
        delay = worker_id * startup_interval

        p = multiprocessing.Process(
            target=worker_run_tasks,
            # 传入 delay
            args=(worker_id, tasks_by_workers[worker_id], progress_queue, delay)
        )
        # ==========================

        p.start()
        processes.append(p)

        # 主进程负责更新进度条
    # 只要队列里有东西，就更新进度条。一旦处理的数量达到 num_todo，就停止。
    with tqdm(total=num_todo, desc='任务执行进度') as p_bar:
        finished_count = 0
        while finished_count < num_todo:
            # 阻塞等待子进程的消息
            _ = progress_queue.get()
            p_bar.update(1)
            finished_count += 1

            # (可选) 检查子进程是否还存活，防止严重的 Crash 导致死锁
            # 如果所有进程都死了但任务还没做完，需要退出循环
            if finished_count < num_todo and all(not p.is_alive() for p in processes):
                print("检测到所有子进程已退出，但任务未完成，可能发生了严重错误。")
                break

    # 等待所有子进程优雅结束
    for p in processes:
        p.join()

    print("\n所有并行任务处理完毕。")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--save-dir', type=str, default='output/batch_output',
                        help='root directory for collected scenes')
    parser.add_argument('--hosts', type=str, default='data_engine/env_collector/hosts.json',
                        help='path to the CARLA host pool JSON (see hosts.example.json)')
    parser.add_argument('--scenes-per-map', type=int, default=600,
                        help='number of 1-minute scenes to collect per map')
    parser.add_argument('--save-time', type=float, default=60.0,
                        help='scene duration in seconds')
    parser.add_argument('--occ-save-fps', type=int, default=2,
                        help='occupancy save rate (must divide the 10 Hz world fps)')
    parser.add_argument('--startup-interval', type=float, default=3.0,
                        help='seconds between worker startups')
    parser.add_argument('--rpc-timeout', type=float, default=120.0,
                        help='CARLA client RPC timeout in seconds; cold-start map '
                             'loading easily exceeds 10s on a single machine')
    return parser.parse_args()


if __name__ == '__main__':
    batch_run(parse_args())
