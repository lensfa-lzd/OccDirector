import json
import pickle
from glob import glob
from pathlib import Path

from data_engine.agent_generator.render import check_finished
from data_engine.agent_generator.check_completion import check_occ_file


def get_task_v3(task_path):
    num_finished = 0
    finish_task = {}
    for try_instance_path in glob(f'{task_path}/*'):
        try_name = Path(try_instance_path).name
        finish_task[try_name] = []
        for video_path in glob(f'{try_instance_path}/*'):
            video_name = Path(video_path).name
            if check_finished(video_path) and check_occ_file(video_path):
                finish_task[try_name].append((try_name, video_name, video_path))
                num_finished += 1

    video_try_num = 10
    todo_tasks = []
    for try_name, finish_videos in finish_task.items():
        if len(finish_videos) < video_try_num:
            finish_video_name = {}
            for _, video_name, video_path in finish_videos:
                finish_video_name[video_name] = video_path

            all_video_name = [f'video_{i}' for i in range(video_try_num)]
            for video_name in all_video_name:
                if video_name not in finish_video_name:
                    save_dir = f'{task_path}/{try_name}/{video_name}'

                    template_video_path = finish_videos[0][2]

                    config_file = f'{template_video_path}/agent_output.json'
                    prompt_file = f'{template_video_path}/prompt.txt'

                    with open(config_file, 'r') as f:
                        data = json.load(f)

                    retreival = data.get('retreival')
                    planning = data.get('planning')

                    todo_tasks.append({
                        'retreival': retreival,
                        'planning': planning,
                        'save_dir': save_dir,
                        'config_file': config_file,
                        'prompt_file': prompt_file
                    })

    print(f'检测到已经完成任务 {num_finished}')
    print(f'还有 {len(todo_tasks)} 个任务待完成')
    return todo_tasks


def get_occ_file_number(file_path):
    name = Path(file_path).name
    return int(name.split('.')[0])


def get_occ_file(video_path, prefix):
    result = []
    occ_files = sorted(glob(f'{video_path}/occupancy/*npy'), key=get_occ_file_number)
    for occ_file in occ_files:
        # keep the 'occupancy/' segment: files live at {prefix}/occupancy/{frame}.npy
        # and the loaders join paths as f'{dataset_path}/{file}'
        result.append(f'{prefix}/occupancy/{Path(occ_file).name}')

    return result


def save_finished_pkl(save_path, save_file):
    scene_info = []
    for try_instance_path in glob(f'{save_path}/*'):
        p = Path(try_instance_path).name
        for video_path in glob(f'{try_instance_path}/*'):
            pp = Path(video_path).name
            if check_finished(video_path) and check_occ_file(video_path):
                scene_info.append(
                    {
                        'scene_name': f'{p}/{pp}',
                        'occ_in_scene_paths': get_occ_file(video_path, f'{p}/{pp}'),
                    }
                )

    with open(f'{save_path}/{save_file}', 'wb') as f:
        pickle.dump(scene_info, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Build scene_infos.pkl from finished videos, or list remaining tasks')
    parser.add_argument('--save-path', help='rendered-output tree to index')
    parser.add_argument('--save-file', default='scene_infos.pkl')
    parser.add_argument('--list-tasks', action='store_true',
                        help='list incomplete tasks (feed --task-root of render_batch.py)')
    parser.add_argument('--task-root', help='tree to scan for incomplete videos')
    args = parser.parse_args()

    if args.list_tasks:
        assert args.task_root, '--task-root is required with --list-tasks'
        tasks = get_task_v3(args.task_root)
        print(f'{len(tasks)} tasks remaining')
    elif args.save_path:
        save_finished_pkl(args.save_path, args.save_file)
    else:
        parser.error('need either --save-path or --list-tasks --task-root')
