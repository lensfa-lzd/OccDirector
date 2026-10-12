"""将 UniOcc 原始数据集转换为 8-bit 帧格式（OccInteract-Static 数据链路入口）。

从原始 UniOcc 发行树（.npz，含 occ_label/flow/masks）读取 occupancy，
逐帧保存为 int8 的 .npy，并生成 full_scene_infos.pkl 帧索引。
产出即 loader 消费的 `{dataset}-{split}` 目录树。

注意：读取 raw .npz 需要环境变量 USE_8_BIT_DATA=0（BaseDataset 默认
优先读取 8bit .npy，转换 raw 时必须关闭）。

断点续跑：已存在且形状校验通过 (200, 200, 16) 的输出文件会跳过。

Usage:
    USE_8_BIT_DATA=0 python -m occinteract.processing.convert_uniocc_8bit \
        --source /path/uniocc-full/v1-0 --save-path /path/uniocc-8bit
"""
import argparse
import os
import pickle
from os.path import dirname, isfile

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from occinteract.loaders.uniocc_raw.core import BaseDataset
from occinteract.loaders.uniocc_raw.get_dataset import get_full_uniocc_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description='UniOcc npz -> 8-bit int8 npy converter')
    parser.add_argument('--source', required=True,
                        help='raw UniOcc root, e.g. uniocc-full/v1-0')
    parser.add_argument('--save-path', required=True,
                        help='output root for the 8-bit tree')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    dataset = ConcatDataset(get_full_uniocc_dataset(args.source))

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        drop_last=False,
        batch_size=args.batch_size,
        collate_fn=BaseDataset.collate_fn,
        num_workers=args.workers
    )

    info_dict = {}
    for batch_data in tqdm(dataloader):
        for i in range(len(batch_data['data_root'])):
            data_root = batch_data['data_root'][i]
            occ = batch_data['occ_data'][i]

            dataset_name = data_root.split('/')[-1]
            if dataset_name not in info_dict:
                info_dict[dataset_name] = []

            sub_instance_path = batch_data['instance_path'][i].replace('.npz', '')

            if 'OpV2V' not in dataset_name:
                scene_name = batch_data['scene_name'][i]
            else:
                scene_name, ego_name, _ = sub_instance_path.split('\\')
                sub_instance_path = sub_instance_path.replace('\\', '/')
                scene_name = f'{scene_name}-{ego_name}'

            occ_save_path = f'{args.save_path}/{dataset_name}/{sub_instance_path}.npy'
            info_dict[dataset_name].append({
                'dataset_name': dataset_name,
                'scene_name': scene_name,
                'frame_file': f'{dataset_name}/{sub_instance_path}.npy',
            })

            finished = False
            if isfile(occ_save_path):
                try:
                    try_occ = np.load(occ_save_path)
                    assert len(try_occ.shape) == 3
                    assert try_occ.shape[0] == try_occ.shape[1] == 200
                    assert try_occ.shape[2] == 16
                    finished = True
                except Exception as e:
                    print(occ_save_path, e)

            if not finished:
                os.makedirs(dirname(occ_save_path), exist_ok=True)
                np.save(occ_save_path, occ.numpy().astype(np.int8))

    for dataset_name in info_dict:
        info_dict[dataset_name] = sorted(info_dict[dataset_name], key=lambda x: x['frame_file'])

    with open(f'{args.save_path}/full_scene_infos.pkl', 'wb') as f:
        pickle.dump(info_dict, f)

    # 同时在每个 split 目录内写 list 格式的 scene_infos.pkl（帧级索引），
    # 这是 UniOcc 上游发行树的约定，uniocc_raw / uniocc_seq 读取器按此入口加载。
    for dataset_name, entries in info_dict.items():
        scenes = {}
        for e in entries:
            rel = e['frame_file'].split('/', 1)[1]  # 去掉 '{dataset_name}/' 前缀
            scenes.setdefault(e['scene_name'], []).append(rel)

        def frame_key(path):
            stem = path.rsplit('/', 1)[-1].split('.')[0]
            return int(stem) if stem.isdigit() else path

        split_index = [
            {'scene_name': scene,
             'occ_in_scene_paths': sorted(paths, key=frame_key)}
            for scene, paths in scenes.items()
        ]
        with open(f'{args.save_path}/{dataset_name}/scene_infos.pkl', 'wb') as f:
            pickle.dump(split_index, f)

    total = sum(len(v) for v in info_dict.values())
    print(f'完成: {total} 帧, {len(info_dict)} 个子数据集 -> {args.save_path}/full_scene_infos.pkl + 各 split scene_infos.pkl')


if __name__ == '__main__':
    main()
