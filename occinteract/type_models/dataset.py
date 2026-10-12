"""Dataset（数据集）相关的类型定义。"""

from typing import List, TypedDict

import numpy as np
import torch


class CaptionInstance(TypedDict, total=False):
    """类型别名，用于表示 supervised_seq* 的训练实例数据结构。

    每个 instance 字典包含:
        - caption (str): 场景描述
        - scene_name (str): 场景标识符
        - scene_files (List[str]): occupancy 文件路径列表
    """
    caption: str
    scene_name: str
    scene_files: List[str]


class OccCaptionDatasetInstance(TypedDict, total=False):
    """类型别名，用于表示 OccCaptionDataset 返回的数据结构。

    每个 instance 字典包含:
        - data_root (str): 数据根目录路径
        - seq_len (int): 序列长度
        - scene_name (str): 场景名称
        - instance_path (List[str]): occupancy 文件路径列表
        - occ_label (np.ndarray) : occupancy grid 数组，形状 (F, H, W, D),
            (List[torch.Tensor]) 如果为 dataloader 返回的batch 数据，则用列表存储不同的序列 [(F, H, W, D)]
        - scene_caption (str): 场景描述文本
    """
    data_root: str
    seq_len: int
    scene_name: str
    instance_path: List[str]
    occ_label: np.ndarray | List[torch.Tensor]
    scene_caption: str


class OccInstance(TypedDict, total=False):
    """类型别名，主要用于表示 unsupervised_seq* 下训练实例数据结构。

    每个 instance 字典包含:
        - scene_name (str): 场景名称
        - occ_in_scene_paths (List[str]): 场景中的 occupancy 文件路径列表

    注意：这不是训练实例，而是场景级别的元数据结构。
    """
    scene_name: str
    occ_in_scene_paths: List[str]


class OccDatasetInstance(TypedDict, total=False):
    """用于表示 unsupervised_seq* 中 dataset 的返回值。

    使用 total=False 允许字段可选，以兼容不同实现。

    每个 instance 字典包含:
        - data_root (str, 可选): 数据根目录路径
        - seq_len (int): 序列长度
        - scene_name (str): 场景名称
        - instance_path (List[str]): occupancy 文件路径列表
        - occ_data (np.ndarray | torch.Tensor): occupancy grid 数组，形状 (F, H, W, D) | 如果以列表存储单帧，则为 [(H, W, D)]
        - occ_label (np.ndarray | torch.Tensor): occupancy grid 数组，形状 (F, H, W, D) | 如果以列表存储单帧，则为 [(H, W, D)]
    """
    data_root: str
    seq_len: int
    scene_name: str
    instance_path: List[str]
    occ_data: np.ndarray | List[np.ndarray] | torch.Tensor | List[torch.Tensor]
    occ_label: np.ndarray | List[np.ndarray] | torch.Tensor | List[torch.Tensor]


__all__ = [
    "CaptionInstance",
    "OccCaptionDatasetInstance",
    "OccInstance",
    "OccDatasetInstance",
]
