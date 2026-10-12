import json
import os.path

from tqdm import tqdm
from pathlib import Path
from glob import glob

from occinteract.loaders.supervised.core import OccCaptionOneFileDataset


def get_process_task():
    result_path = os.getenv('GEN_RESULT_PATH')
    npy_files = glob(f'{result_path}/*npy')

    errors = []
    result = []
    for npy_file in tqdm(npy_files):
        file_name = Path(npy_file).stem

        if not os.path.isfile(f"{result_path}/{file_name}.txt"):
            errors.append(file_name)
        else:
            with open(f"{result_path}/{file_name}.txt", 'r') as f:
                prompt = f.read()

        result.append({
            'scene_name': file_name,
            'scene_files': npy_file,
            'caption': prompt,
        })

    print(f'成功加载了 {len(result)} 个数据, 有 {len(errors)} 个失败数据')

    dataset = OccCaptionOneFileDataset(
        data_root=None,
        resize_cfg={
            'origin': 200,
            'after': 224,
        },
        load_occ=True,
        instance=result,
    )

    try_count = int(os.getenv("NUM_TEST_COUNT", 5))  # 计算5次取平均值
    dataset_name = os.getenv("MODEL_NAME", 'occ_gen')
    dataset_dict = {
        f'{dataset_name}': dataset,
    }

    tasks = []
    for i in range(len(dataset)):
        for j in range(try_count):
            task_cfg = {
                'key': f'{dataset_name}__{i}__{j}',
            }
            tasks.append(json.dumps(task_cfg))

    return tasks, dataset_dict
