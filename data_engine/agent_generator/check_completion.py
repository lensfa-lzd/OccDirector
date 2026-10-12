import json
import os
import time
from glob import glob
from pathlib import Path

from tqdm import tqdm
# 1. 引入线程池和任务完成生成器
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_engine.agent_generator.render import check_finished


def check_occ_file(video_path):
    occ_files = glob(f'{video_path}/occupancy/*npy')
    if len(occ_files) > 0:
        return True
    return False


def get_finish_task(save_path):
    finish_task = []
    for try_instance_path in glob(f'{save_path}/*'):
        p = Path(try_instance_path)
        for video_path in glob(f'{try_instance_path}/*'):
            pp = Path(video_path)
            if check_finished(video_path) and check_occ_file(video_path):
                finish_task.append((p.name, pp.name, video_path))

    print(f'检测到已经完成任务 {len(finish_task)}')
    return finish_task


def copy_task_file(task, dst_path):
    try_name, video_name, src_path = task

    save_path = f'{dst_path}/{try_name}/{video_name}'
    cmd = f'cp -r {src_path} {save_path}'

    # print(task)
    # print(cmd)
    # time.sleep(3600)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.system(cmd)
    time.sleep(0.2)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Copy finished videos (occupancy + videos + json) out of a render tree')
    parser.add_argument('--check-path', required=True, help='render tree to scan')
    parser.add_argument('--save-path', required=True, help='destination root')
    args = parser.parse_args()

    check_path = args.check_path
    save_path = args.save_path
    finish_task = get_finish_task(check_path)

    with open(f'finish_task.json', 'w') as f:
        json.dump(finish_task, f, indent=2)

    finish_task = finish_task
    max_workers = 8

    print(f"开始并行复制，线程数: {max_workers}，待处理任务: {len(finish_task)}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_task_file, task, save_path) for task in finish_task]

        # 4. 使用 as_completed 配合 tqdm 显示进度
        # as_completed 会在某个任务完成时 yield 出来，这样进度条是真实的完成进度
        for _ in tqdm(as_completed(futures), total=len(finish_task)):
            pass


if __name__ == '__main__':
    main()
