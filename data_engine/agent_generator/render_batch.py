import json
import os
import time
import multiprocessing
import torch

from data_engine.hosts import load_host_resources
from pathlib import Path
from tqdm import tqdm

from data_engine.agent_generator.render import check_finished, scene_generation_timeout_warper


def worker_run_tasks(worker_id, tasks, progress_queue, start_delay=0):
    """
    Worker 进程：负责执行分配到的任务列表
    """
    # --- 1. 延迟启动逻辑 ---
    if start_delay > 0:
        print(f'[Worker {worker_id}] 将延迟 {start_delay:.1f} 秒后启动...')
        time.sleep(start_delay)
    # -----------------------

    print(f'[Worker {worker_id}] 启动，共分配到 {len(tasks)} 个任务。')

    for task in tasks:
        save_dir = task['save_dir']
        # 为了日志清晰，可包含 Worker ID
        print(f'[Worker {worker_id}] 正在处理 {save_dir}')

        try_count = 1
        while try_count <= 5:
            # 清理可能存在的文件
            if os.path.exists(f'{save_dir}/occupancy'):
                os.system(f"rm -r {save_dir}/occupancy")

            try:
                # --- 执行生成任务 (参数动态化) ---
                scene_generation_timeout_warper(
                    task["retreival"],
                    task["planning"],
                    ip_address=task["host"],  # <--- 自动分配的 Host
                    port=task["port"],  # <--- 自动分配的 Port
                    save_dir=save_dir,
                    max_save_time=args.max_save_time,
                    make_done_tag=True,
                    device=task["device"],  # <--- 自动分配的 Device
                    batch_generation=True
                )
            except Exception as e:
                print(f'[Worker {worker_id}] Error in {save_dir}: {e}')

            # scene_generation 完成了生成任务的后处理
            if os.path.exists(f'{save_dir}/done'):
                # 复制生成信息文件
                os.system(f"cp {task['config_file']} {save_dir}")
                os.system(f"cp {task['prompt_file']} {save_dir}")

                # 移除 done tag
                os.system(f"rm {save_dir}/done")

            if check_finished(save_dir):
                print(f'[Worker {worker_id}] {save_dir} 已经完成')
                break

            # 失败重试等待
            time.sleep(5)
            try_count += 1
            print(f'[Worker {worker_id}] 第 {try_count} 次重试: {save_dir}')

        # 无论成功与否，向主进程汇报进度
        progress_queue.put(1)


def batch_run(args):
    from data_engine.agent_generator.curate import get_task_v3
    all_tasks = get_task_v3(args.task_root)
    total_tasks = len(all_tasks)

    if total_tasks == 0:
        print("没有任务需要处理。")
        return

    # --- 2. 自动检测 CUDA ---
    cuda_count = torch.cuda.device_count()
    if cuda_count > 0:
        print(f"检测到 {cuda_count} 个可用 CUDA 设备，将自动均衡分配。")
    else:
        print("未检测到 CUDA 设备，将使用 CPU。")

    host_resources = load_host_resources(args.hosts)

    # --- 3. 资源解析 (计算 IP/Port Slot) ---
    worker_slots = []
    for h_res in host_resources:
        h_ip = h_res['ip']
        h_workers = h_res['workers']
        h_base_port = h_res['base_port']  # 假设格式为 [RPC_PORT, TM_PORT] 或类似

        for i in range(h_workers):
            worker_slots.append({
                "host": h_ip,
                "port": int(h_base_port[0] + 100 * i),
                # "tm_port": int(h_base_port[1] + 10 * i)
            })

    num_works = len(worker_slots)
    if num_works == 0:
        print("错误：Config 中没有配置有效的 Host 资源。")
        return

    print(f"根据配置，共初始化 {num_works} 个并行 Worker，分布在 {len(host_resources)} 台 Host 上。")

    # --- 4. 任务分配 (Round-Robin) ---
    tasks_by_workers = [[] for _ in range(num_works)]

    for i, task in enumerate(all_tasks):
        worker_id = i % num_works
        slot = worker_slots[worker_id]

        # 自动分配 Device: 同一个 Worker 始终绑定同一块 GPU
        if cuda_count > 0:
            cur_device = f'cuda:{worker_id % cuda_count}'
        else:
            cur_device = 'cpu'

        # 将资源信息注入 Task
        task.update({
            "host": slot['host'],
            "port": slot['port'],
            "device": cur_device
        })

        tasks_by_workers[worker_id].append(task)

    # --- 5. 并行启动 ---
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    processes = []

    # === 定义启动间隔 ===
    STARTUP_INTERVAL = 3.0
    # =================

    print(f"启动 {num_works} 个 Worker 并行处理 (启动间隔: {STARTUP_INTERVAL}s)...")

    for worker_id in range(num_works):
        if not tasks_by_workers[worker_id]:
            continue

        # 计算延迟
        delay = worker_id * STARTUP_INTERVAL

        p = multiprocessing.Process(
            target=worker_run_tasks,
            args=(worker_id, tasks_by_workers[worker_id], progress_queue, delay)
        )
        p.start()
        processes.append(p)

    # --- 6. 进度监控 ---
    with tqdm(total=total_tasks, desc='总任务进度') as p_bar:
        finished_count = 0
        while finished_count < total_tasks:
            try:
                # 阻塞等待消息，设定超时以便检查进程存活状态
                _ = progress_queue.get(timeout=2)
                p_bar.update(1)
                finished_count += 1
            except:
                # 队列获取超时，检查子进程状态
                if all(not p.is_alive() for p in processes):
                    print("\nWARNING: 所有子进程已退出，但任务未完全结束。")
                    break

    # 等待即使失败的进程也优雅退出
    for p in processes:
        p.join()

    print("\n所有并行任务处理完毕。")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Multi-host batch rendering of planned scenes')
    parser.add_argument('--task-root', required=True,
                        help='rendered-output tree to scan for incomplete videos (see curate.py)')
    parser.add_argument('--hosts', default='data_engine/agent_generator/hosts.json',
                        help='CARLA host pool JSON (see data_engine/hosts.example.json)')
    parser.add_argument('--max-save-time', type=float, default=10.0)
    parser.add_argument('--startup-interval', type=float, default=3.0)
    args = parser.parse_args()
    batch_run(args)
