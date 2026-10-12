"""
加载带注释的 4D Occupancy 数据集。

该模块提供了加载 UniOcc、OccCollect 和 TTSG 数据集的接口，
支持实例级注释和序列级注释两种方式。
"""

import logging
import os
from typing import List, Dict, Optional, Tuple

import numpy as np

from occinteract.loaders.supervised.core import OccCaptionDataset
from occinteract.loaders.supervised.get_per_instance_caption_train_data import load_instance_caption_dataset_info
from occinteract.loaders.supervised.get_per_seq_caption_train_data import get_occ_collect_dataset_info, get_ttsg_dataset_info
from occinteract.type_models.dataset import CaptionInstance


def get_instance_caption_dataset_info(dataset_path: str, min_gap: int = 8) -> List[CaptionInstance]:
    """获取基于实例级注释的训练数据信息。

    将序列的第一帧的 caption 作为整个序列的 caption，
    按照固定的长度比例输出不同比例的序列数据。

    Args:
        dataset_path (str): 数据集根目录路径。
        min_gap (int): 序列中帧之间的最小间隔，默认为 8。

    Returns:
        List[CaptionInstance]: 训练实例列表，每个实例包含：
            - caption (str): 文本描述
            - scene_name (str): 场景标识符
            - scene_files (List[str]): occupancy 文件路径列表
    """
    len_config = {
        16: 1, 8: 1, 4: 1, 2: 1, 1: 1
    }

    instance = load_instance_caption_dataset_info(
        dataset_path,
        len_config,
        min_gap=min_gap,
    )

    logging.warning(
        f'supervised_seq_v3/get_dataset.py -> get_instance_caption_dataset_info: '
        f'生成 {len(instance):,} 个训练实例.'
    )
    return instance


def get_seq_caption_dataset_info(
        occ_collect_dataset_path: str,
        ttsg_dataset_path: str,
) -> List[CaptionInstance]:
    """获取基于序列级注释的训练数据信息。

    从 OccCollect 和 TTSG 数据集中加载带注释的序列数据。

    Args:
        occ_collect_dataset_path (str): OccCollect 数据集根目录路径。
        ttsg_dataset_path (str): TTSG 数据集根目录路径。

    Returns:
        List[CaptionInstance]: 训练实例列表，每个实例包含：
            - caption (str): 文本描述
            - scene_name (str): 场景标识符
            - scene_files (List[str]): occupancy 文件路径列表
    """
    instances = []
    if not os.getenv("DATA_NO_INTERACT_ENV"):
        instances.extend([
            *get_occ_collect_dataset_info(occ_collect_dataset_path)
        ])

    if not os.getenv("DATA_NO_INTERACT_AGENT"):
        instances.extend([
            *get_ttsg_dataset_info(ttsg_dataset_path)
        ])

    logging.warning(
        f'supervised_seq_v3/get_dataset.py -> load_seq_caption_dataset_info: '
        f'生成 {len(instances):,} 个训练实例.'
    )
    return instances


def get_full_dataset(
        uni_occ_dataset_path: str,
        occ_collect_dataset_path: str,
        ttsg_dataset_path: str,
        resize_cfg: Optional[Dict[str, int]],
        load_occ: bool,
) -> OccCaptionDataset:
    """获取完整的带注释训练数据集。

    整合 UniOcc、OccCollect 和 TTSG 数据集，创建统一的训练数据集对象。

    Args:
        uni_occ_dataset_path (str): UniOcc 数据集根目录路径。
        occ_collect_dataset_path (str): OccCollect 数据集根目录路径。
        ttsg_dataset_path (str): TTSG 数据集根目录路径。
        resize_cfg (Optional[Dict[str, int]]): 数据集重置配置字典。
        load_occ (bool): 是否加载 occupancy 数据。

    Returns:
        OccCaptionDataset: 完整的训练数据集对象。
    """
    uniocc_mini_gap = os.getenv("UNIOCC_MINI_GAP", 16)

    instances = []
    if not os.getenv("DATA_NO_STATIC"):
        instances.extend([
            *get_instance_caption_dataset_info(uni_occ_dataset_path, uniocc_mini_gap)
        ])

    if not os.getenv("DATA_NO_INTERACTION"):
        instances.extend([
            *get_seq_caption_dataset_info(occ_collect_dataset_path, ttsg_dataset_path)
        ])

    logging.warning(
        f'supervised_seq_v3/get_dataset.py -> get_full_dataset: '
        f'生成 {len(instances):,} 个训练实例.'
    )

    dataset = OccCaptionDataset(
        data_root=None,
        resize_cfg=resize_cfg,
        load_occ=load_occ,
        instance=instances,
    )
    return dataset


def get_test_dataset(
        samples: int,
        uni_occ_dataset_path: str,
        occ_collect_dataset_path: str,
        ttsg_dataset_path: str,
        resize_cfg: Optional[Dict[str, int]],
        load_occ: bool,
) -> Tuple[OccCaptionDataset, OccCaptionDataset, OccCaptionDataset]:
    """等间距抽样获取测试集
    """
    uniocc_mini_gap = os.getenv("UNIOCC_MINI_GAP", 16)

    uniocc_instances = get_instance_caption_dataset_info(uni_occ_dataset_path, uniocc_mini_gap)
    occ_collect_instances = get_occ_collect_dataset_info(occ_collect_dataset_path)
    ttsg_instances = get_ttsg_dataset_info(ttsg_dataset_path)

    def filter_instances(instances):
        if len(instances) == 0:
            return []
        rate = samples / len(instances)
        step = int(1 // rate)
        return instances[::step]

    uniocc_dataset = OccCaptionDataset(
        data_root=None,
        resize_cfg=resize_cfg,
        load_occ=load_occ,
        instance=filter_instances(uniocc_instances),
    )

    occ_collect_dataset = OccCaptionDataset(
        data_root=None,
        resize_cfg=resize_cfg,
        load_occ=load_occ,
        instance=filter_instances(occ_collect_instances),
    )

    ttsg_dataset = OccCaptionDataset(
        data_root=None,
        resize_cfg=resize_cfg,
        load_occ=load_occ,
        instance=filter_instances(ttsg_instances),
    )

    return uniocc_dataset, occ_collect_dataset, ttsg_dataset
