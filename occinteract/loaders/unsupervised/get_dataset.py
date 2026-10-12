import json
import logging
import os
import pickle
from typing import List, Dict, Tuple, Optional

import numpy as np

from occinteract.type_models.dataset import OccInstance
from occinteract.loaders.unsupervised.core import OccReconstructionDataset


def get_uniocc_dataset_info(uni_occ_path: str) -> Dict[str, List[OccInstance]]:
    """从 UniOcc 数据集获取场景信息。

    根据 DATASET_SETTING 环境变量决定是否包含 OpenOccupancy 和
    SurroundOcc 数据集。

    Args:
        uni_occ_path: UniOcc 数据集路径。

    Returns:
        Dict[str, List[OccInstance]]: 包含 'train', 'val', 'test' 键的字典，
            每个键对应的值为场景信息列表。
    """
    with open(f'{uni_occ_path}/scene_infos.pkl', 'rb') as f:
        full_scene_infos = pickle.load(f)

    result = {
        'train': [],
        'val': [],
        'test': [],
    }
    if os.getenv('DATASET_SETTING') == '1':
        # 排除 SurroundOcc、OpenOccupancy 数据集
        dataset_setting = [
            ('NuScenes-via-Occ3D-2Hz', False),
            ('Waymo-via-Occ3D-2Hz', False),
            ('Carla-2Hz', False),
            ('OpV2V-10Hz', True),
        ]
    else:
        dataset_setting = [
            ('NuScenes-via-Occ3D-2Hz', False),
            ('NuScenes-via-OpenOccupancy-2Hz', False),
            ('NuScenes-via-SurroundOcc-2Hz', False),
            ('Waymo-via-Occ3D-2Hz', False),
            ('Carla-2Hz', False),
            ('OpV2V-10Hz', True),
        ]
    for subset in result.keys():
        for dataset, has_test_set in dataset_setting:
            if subset == 'test':
                if not has_test_set:
                    use_subset = 'val'
                else:
                    use_subset = 'test'
            else:
                use_subset = subset

            dataset_key = f'{dataset}-{use_subset}'
            scene_infos = full_scene_infos[dataset_key]

            new_scene_infos = []
            for it in scene_infos:
                occ_in_scene_paths = it['occ_in_scene_paths']

                if '10Hz' in dataset_key:
                    occ_in_scene_paths = occ_in_scene_paths[::5]

                # 转换为绝对地址
                new_occ_in_scene_paths = []
                for file in occ_in_scene_paths:
                    new_occ_in_scene_paths.append(f'{uni_occ_path}/{file}')

                new_scene_infos.append({
                    'scene_name': it['scene_name'],
                    'occ_in_scene_paths': new_occ_in_scene_paths
                })

            result[subset].extend(new_scene_infos)

            logging.info(
                f'unsupervised_seq_v2/get_dataset.py -> get_uniocc_dataset_info: '
                f'{subset} 加载数据 {dataset_key} {len(scene_infos)} 序列'
            )

    return result


def systematic_split(x: np.ndarray, test_ratio: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """等间距分割数据集。

    使用等间距抽样方法将数据分割为训练集和测试集。

    Args:
        x: 待分割的数据数组。
        test_ratio: 测试集比例，例如 0.3 表示每 10 个样本取 3 个。

    Returns:
        Tuple[np.ndarray, np.ndarray]: (train_set, test_set)，分别为训练集和测试集。
    """
    # 计算步长 k，例如 0.3 就是每 3-4 个取 1 个
    step = int(1 / test_ratio)

    n_samples = len(x)
    indices = np.arange(n_samples)

    # 定义测试集索引规则：每隔 step 取一个 (例如索引 0, 5, 10...)
    # 也可以使用 (indices + offset) % step == 0 来改变起始位置
    test_mask = (indices % step) == 0

    # 训练集就是测试集的反面
    train_mask = ~test_mask

    return x[train_mask], x[test_mask]


def get_collect_dataset_info(dataset_path: str, scene_file: str = 'scene_infos.pkl') -> Dict[str, List[OccInstance]]:
    """从 OccCollect 数据集获取场景信息。

    Args:
        dataset_path: 数据集路径。
        scene_file: 场景信息文件名。

    Returns:
        Dict[str, List[OccInstance]]: 包含 'train', 'val', 'test' 键的字典。
            注意：val 和 test 共享相同的数据。
    """
    with open(f'{dataset_path}/{scene_file}', 'rb') as f:
        scene_infos = pickle.load(f)

    new_scene_infos = []
    for it in scene_infos:
        occ_in_scene_paths = it['occ_in_scene_paths']

        # 转换为绝对地址
        new_occ_in_scene_paths = []
        for file in occ_in_scene_paths:
            new_occ_in_scene_paths.append(f'{dataset_path}/{file}')

        new_scene_infos.append({
            'scene_name': it['scene_name'],
            'occ_in_scene_paths': new_occ_in_scene_paths
        })

    train, val = systematic_split(np.array(new_scene_infos), 0.3)
    train = list(train)
    val = list(val)

    logging.info(
        f'unsupervised_seq_v2/get_dataset.py -> get_collect_dataset_info: '
        f'{dataset_path} 加载数据 {len(train):,} 训练序列, {len(val):,} 验证/测试序列'
    )

    result = {
        'train': train,
        'val': val,
        'test': val,
    }
    return result


def merge_dataset_info(dataset_info_list: List[Dict], factor_list: List[Dict]) -> Dict[str, List[OccInstance]]:
    """合并多个数据集的信息并进行抽样。

    Args:
        dataset_info_list: 数据集信息列表。
        factor_list: 对应每个数据集的抽样比例，
            1 表示全部使用，0.x 表示保留 x 比例。

    Returns:
        Dict[str, List[OccInstance]]: 合并后的数据集信息。
    """
    result = {
        'train': [],
        'val': [],
        'test': [],
    }
    logging.warning(
        f'unsupervised_seq_v2/get_dataset.py -> merge_dataset_info: '
        f'数据集抽样比例 {factor_list}'
    )
    for i, it in enumerate(dataset_info_list):
        for subset in result.keys():
            factor = factor_list[i][subset]
            if factor == 1:
                result[subset].extend(it[subset])
            else:
                # 进行等间距抽样
                _, mini_set = systematic_split(np.array(it[subset]), factor)
                result[subset].extend(list(mini_set))

    return result


def get_seq(
        seq_len: int,
        scene_files: List[str],
) -> List[List[str]]:
    """从场景文件列表中提取固定长度的序列。

    Args:
        seq_len: 要提取的序列长度。
        scene_files: 场景文件路径列表。

    Returns:
        List[List[str]]: 包含所有可能序列的列表。
    """
    result = []
    limit = len(scene_files) - seq_len + 1
    for i in range(limit):
        sub_seq = scene_files[i: i + seq_len]
        result.append(sub_seq)
    return result


def split_dataset_to_small_seq(
        len_config: Dict[int, int],
        dataset_items: List[OccInstance],
        subset: str,
) -> List[OccInstance]:
    """将数据集分割为不同长度的小序列。

    Args:
        len_config: 序列长度配置，格式为 {长度: 权重}。
        dataset_items: 原始数据集项列表。
        subset: 数据子集名称 ('train', 'val', 或 'test')。

    Returns:
        List[OccInstance]: 分割后的数据集项列表。
    """
    result = []
    for occ_len, weight in len_config.items():
        sub_result = []
        for scene_info in dataset_items:
            scene_name = scene_info['scene_name']
            occ_files = scene_info['occ_in_scene_paths']

            if len(occ_files) >= occ_len:
                for i, sequence_paths in enumerate(get_seq(occ_len, occ_files)):
                    sub_result.append({
                        'scene_name': f'{scene_name}__split_len_{occ_len}_item_{i}',
                        'occ_in_scene_paths': sequence_paths,
                    })

        result.extend(sub_result * weight)
        logging.warning(
            f'unsupervised_seq_v2/get_dataset.py -> split_dataset_to_small_seq: '
            f'{subset} 加载长度 {occ_len} 的序列 {len(sub_result):,} * {weight} = {len(sub_result) * weight:,} 个'
        )

    logging.warning(
        f'unsupervised_seq_v2/get_dataset.py -> split_dataset_to_small_seq: '
        f'{subset} 一共加载序列 {len(result):,} 个'
    )
    return result


def get_full_dataset(
        uniocc_dataset_path: str,
        collect_dataset_path: str,
        ttsg_dataset_path: str,
        resize_cfg: Optional[Dict[str, int]] = None,
        load_occ: bool = True,
) -> Tuple[OccReconstructionDataset, OccReconstructionDataset, OccReconstructionDataset]:
    """获取完整的自监督重建数据集。

    从 UniOcc、OccCollect 和 TTSG 数据集中加载数据，合并并分割
    为不同长度的序列。

    Args:
        uniocc_dataset_path: UniOcc 数据集路径。
        collect_dataset_path: OccCollect 数据集路径。
        ttsg_dataset_path: TTSG 数据集路径。
        resize_cfg: 数据缩放配置，包含 'origin' 和 'after' 键。
        load_occ: 是否加载实际的 Occupancy 数据。

    Returns:
        Tuple[OccReconstructionDataset, OccReconstructionDataset, OccReconstructionDataset]:
            (train_dataset, val_dataset, test_dataset) 三个 OccReconstructionDataset 实例。
    """
    raw_dataset_info = merge_dataset_info(
        [
            get_uniocc_dataset_info(uniocc_dataset_path),
            get_collect_dataset_info(collect_dataset_path, 'scene_infos_20260120.pkl'),
            get_collect_dataset_info(ttsg_dataset_path)
        ],
        [
            {
                'train': 1,
                'val': 1,
                'test': 1,
            },
            {
                'train': 0.01,
                'val': 0.01,
                'test': 0.01,
            },
            {
                'train': 0.01,
                'val': 0.01,
                'test': 0.01,
            }
        ]

    )

    for subset, items in raw_dataset_info.items():
        logging.warning(
            f'unsupervised_seq_v2/get_dataset.py -> get_full_dataset: '
            f'{subset} 加载数据 {len(items):,} 序列'
        )

    result = {
        'train': [],
        'val': [],
        'test': [],
    }

    len_config = {
        32: 1,
        16: 1, 8: 1,
        4: 1, 2: 1, 1: 1
    }
    logging.warning(
        f'unsupervised_seq_v2/get_dataset.py -> get_full_dataset: 序列长度比例设置, '
        f'{json.dumps(len_config)}'
    )

    for subset in result.keys():
        result[subset] = split_dataset_to_small_seq(len_config, raw_dataset_info[subset], subset)

    # return result

    train = OccReconstructionDataset(
        resize_cfg=resize_cfg,
        load_occ=load_occ,
        instance=result['train'],
    )

    val = OccReconstructionDataset(
        resize_cfg=resize_cfg,
        load_occ=load_occ,
        instance=result['val'],
    )

    test = OccReconstructionDataset(
        resize_cfg=resize_cfg,
        load_occ=load_occ,
        instance=result['test'],
    )

    return train, val, test
