import os
from typing import Tuple

from torch.utils.data import ConcatDataset


from occinteract.loaders.uniocc_raw.core import BaseDataset


def get_train_val_test_dataset(base_path: str, test_dataset_exists: bool) -> Tuple[
    BaseDataset, BaseDataset, BaseDataset]:
    """获取训练集、验证集和测试集的 Dataset 对象。

    Args:
        base_path (str): 数据集基础路径，不包含后缀（如 '-train'）。
        test_dataset_exists (bool): 是否存在独立的测试集。如果为 False，
            测试集将使用验证集。

    Returns:
        Tuple[BaseDataset, BaseDataset, BaseDataset]: 包含训练集、验证集和测试集的元组。
    """
    train_dataset = BaseDataset(
        data_root=base_path + '-train',
    )

    val_dataset = BaseDataset(
        data_root=base_path + '-val',
    )

    if test_dataset_exists:
        test_dataset = BaseDataset(
            data_root=base_path + '-test',
        )
    else:
        test_dataset = BaseDataset(
            data_root=base_path + '-val',
        )

    return train_dataset, val_dataset, test_dataset


def get_full_uniocc_dataset(uni_occ_path: str) -> Tuple[ConcatDataset, ConcatDataset, ConcatDataset]:
    """获取完整的 UniOcc 数据集（包含所有子数据集）。

    该函数会合并多个数据集（NuScenes、Waymo、Carla、OpV2V）的训练集、
    验证集和测试集。

    Args:
        uni_occ_path (str): UniOcc 数据集的根路径。

    Returns:
        Tuple[ConcatDataset, ConcatDataset, ConcatDataset]: 包含合并后的训练集、
            验证集和测试集的元组。
    """
    all_dataset = [
        ('NuScenes-via-Occ3D-2Hz', False),
        ('NuScenes-via-OpenOccupancy-2Hz', False),
        ('NuScenes-via-SurroundOcc-2Hz', False),

        # ('Waymo-via-Occ3D-2Hz', False), # 10hz 的版本应该涵盖 2Hz
        ('Waymo-via-Occ3D-10Hz', False),

        # ('Carla-2Hz', False),
        ('Carla-10Hz', False),

        ('OpV2V-10Hz', True),
    ]
    datasets = [[], [], []]
    for dataset_name, test_dataset_exists in all_dataset:
        train, val, test = get_train_val_test_dataset(os.path.join(uni_occ_path, dataset_name), test_dataset_exists)
        datasets[0].append(train)
        datasets[1].append(val)
        datasets[2].append(test)

    train = ConcatDataset(datasets[0])
    val = ConcatDataset(datasets[1])
    test = ConcatDataset(datasets[2])

    print('Load full uniocc dataset')
    print(f'train {len(train)} instance, val {len(val)} instance, test {len(test)} instance')
    return train, val, test
