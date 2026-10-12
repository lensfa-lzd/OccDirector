import os
from typing import Tuple, Optional, Dict

from torch.utils.data import ConcatDataset

from occinteract.loaders.uniocc_seq.core import BaseDataset


def get_train_val_test_dataset(
        seq_len: int,
        base_path: str,
        test_dataset_exists: bool,
        resize_cfg: Optional[Dict] = None,
) -> Tuple[BaseDataset, BaseDataset, BaseDataset]:
    """获取指定数据集的训练集、验证集和测试集。

    Args:
        seq_len: 加载的序列长度。如果为 -1，则加载完整场景。
        base_path: 数据集的基础路径（不含 -train/-val/-test 后缀）。
        test_dataset_exists: 是否存在独立的测试集。如果为 False，则使用验证集作为测试集。
        resize_cfg: 用于调整数据大小的配置字典，包含 'after' 键指定目标尺寸。

    Returns:
        包含三个 BaseDataset 对象的元组：(train_dataset, val_dataset, test_dataset)。
    """
    train_dataset = BaseDataset(
        data_root=base_path + '-train',
        seq_len=seq_len,
        resize_cfg=resize_cfg,
    )

    val_dataset = BaseDataset(
        data_root=base_path + '-val',
        seq_len=seq_len,
        resize_cfg=resize_cfg,
    )

    if test_dataset_exists:
        test_dataset = BaseDataset(
            data_root=base_path + '-test',
            seq_len=seq_len,
            resize_cfg=resize_cfg,
        )
    else:
        test_dataset = BaseDataset(
            data_root=base_path + '-val',
            seq_len=seq_len,
            resize_cfg=resize_cfg,
        )

    return train_dataset, val_dataset, test_dataset


def get_full_uniocc_dataset(
        seq_len: int,
        uni_occ_path: str,
        resize_cfg: Optional[Dict] = None,
) -> Tuple[ConcatDataset, ConcatDataset, ConcatDataset]:
    """获取完整的 UniOcc 数据集（包含多个源数据集）。

    根据 DATASET_SETTING 环境变量选择使用的数据集：
        - "1": 仅使用 Carla-2Hz
        - "2": 仅使用 NuScenes-via-Occ3D-2Hz
        - 其他: 使用完整数据集（包含 NuScenes、Waymo、Carla、OpV2V）

    Args:
        seq_len: 加载的序列长度。如果为 -1，则加载完整场景。
        uni_occ_path: UniOcc 数据集的根路径。
        resize_cfg: 用于调整数据大小的配置字典，包含 'after' 键指定目标尺寸。

    Returns:
        包含三个 ConcatDataset 对象的元组：(train_dataset, val_dataset, test_dataset)。
    """
    if os.getenv('DATASET_SETTING') == '1':
        all_dataset = [('Carla-2Hz', False), ]
    elif os.getenv('DATASET_SETTING') == '2':
        all_dataset = [('NuScenes-via-Occ3D-2Hz', False), ]
    else:
        all_dataset = [
            ('NuScenes-via-Occ3D-2Hz', False),
            ('NuScenes-via-OpenOccupancy-2Hz', False),
            ('NuScenes-via-SurroundOcc-2Hz', False),

            ('Waymo-via-Occ3D-2Hz', False),  # 统一视频的帧率
            # ('Waymo-via-Occ3D-10Hz', False),

            ('Carla-2Hz', False),
            # ('Carla-10Hz', False),

            ('OpV2V-10Hz', True),
        ]
    datasets = [[], [], []]
    for dataset_name, test_dataset_exists in all_dataset:
        train, val, test = get_train_val_test_dataset(
            seq_len,
            os.path.join(uni_occ_path, dataset_name),
            test_dataset_exists,
            resize_cfg=resize_cfg,
        )
        datasets[0].append(train)
        datasets[1].append(val)
        datasets[2].append(test)

    train = ConcatDataset(datasets[0])
    val = ConcatDataset(datasets[1])
    test = ConcatDataset(datasets[2])

    print('Load full uniocc dataset with seq')
    print(f'train {len(train):,} instance, val {len(val):,} instance, test {len(test):,} instance')
    return train, val, test
