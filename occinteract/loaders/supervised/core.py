import os
from functools import lru_cache
from typing import List, Dict, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from occinteract.tools.occ_resizer import occ_resize
from occinteract.type_models.dataset import (
    CaptionInstance,
    OccCaptionDatasetInstance,
)


class OccCaptionDataset(Dataset):
    """占用栅格（Occupancy Grid）序列数据集类。

    用于加载带有 Captions 的 4D Occupancy Grid 序列数据，支持通过 data_root 加载文件
    或直接加载预构建的 instance 数据。
    """

    def __init__(
            self,
            data_root: str,
            resize_cfg: Optional[Dict[str, int]] = None,
            load_occ: bool = True,
            instance: Optional[List[CaptionInstance]] = None,
    ) -> None:
        """初始化 Occupancy Grid 序列数据集。

        Args:
            data_root: 数据集根目录路径。
            resize_cfg: 调整 Occupancy Grid 大小的配置，包含 'origin' 和 'after' 键。默认为 None。
            load_occ: 是否加载 Occupancy Grid 数据。默认为 True。
            instance: 直接在外部构建好所需的数据实例列表。每个实例是包含
                'scene_files'、'scene_name' 和 'caption' 的字典。默认为 None。
        """
        super().__init__()
        self.data_root = data_root
        self.resize_cfg = resize_cfg
        self.load_occ = load_occ
        self.instances = instance

    def __len__(self) -> int:
        """返回数据集中的实例数量。

        Returns:
            int: 数据集中实例的总数。
        """
        return len(self.instances)

    @lru_cache(maxsize=16)
    def __load_file__(self, path: str) -> np.ndarray:
        """加载并缓存 NumPy 文件。

        使用 LRU 缓存避免重复的 I/O 开销。

        Args:
            path: NumPy 文件路径。

        Returns:
            np.ndarray: 加载的 NumPy 数组。

        Raises:
            FileNotFoundError: 如果文件不存在或为空时抛出。
            Exception: 如果加载数据失败时抛出。
        """
        try:
            # 最好先检查文件是否存在且不为空
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                print(f"Error: File does not exist or is empty: {path}")
                # 你可以在这里决定是抛出异常还是返回一个默认值
                raise FileNotFoundError(f"File not found or is empty: {path}")

            return np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"Failed to load file: {path}")
            print(f"Original error: {e}")
            # 重新抛出异常，让程序像原来一样停止，但你已经看到了文件名
            raise e

    def __getitem__(self, index: int) -> OccCaptionDatasetInstance:
        """获取指定索引的数据实例。

        Args:
            index: 数据实例的索引。

        Returns:
            OccCaptionDatasetInstance: 包含以下键的字典：
                - data_root (str): 数据根目录路径。
                - seq_len (int): 序列长度（帧数）。
                - scene_name (str): 场景名称。
                - instance_path (List[str]): Occupancy 文件路径列表。
                - occ_label (np.ndarray): Occupancy Grid 标签，形状为 (F, H, W, D)，
                  其中 F 为帧数，H/W 为高度/宽度，D 为深度。
                  形状示例: [frame, 200, 200, 16]
                - scene_caption (str): 场景描述文本。
        """
        data_dict = self.instances[index]
        occ_paths = data_dict['scene_files']

        # 传入的 occ_paths 是不定长的视频
        if self.load_occ:
            if self.data_root:
                occ_labels = [
                    self.__load_file__(os.path.join(self.data_root, path))
                    for path in occ_paths
                ]
            else:
                occ_labels = [
                    self.__load_file__(path)
                    for path in occ_paths
                ]
            occ_labels = np.stack(occ_labels, axis=0)
            if self.resize_cfg:
                occ_labels = occ_resize(occ_labels, (self.resize_cfg['after'], self.resize_cfg['after']))
        else:
            # 伪数据
            occ_labels = np.zeros((1, 1, 1, 1))

        return {
            "data_root": self.data_root,
            "seq_len": len(occ_paths),
            "scene_name": data_dict['scene_name'],  # str
            "instance_path": occ_paths,  # List[str]
            "occ_label": occ_labels,  # (F, H, W, D)  seq_len*高*宽*深 [frame, 200, 200, 16]
            "scene_caption": data_dict['caption'],
        }

    @staticmethod
    def collate_fn(batch: Any) -> OccCaptionDatasetInstance:
        """将一个批次的数据实例合并为字典。

        将批次中每个实例的对应键收集到一起：对于 NumPy 数组，转换为 PyTorch 张量；
        对于其他类型，收集为列表。

        Args:
            batch: 一个批次的数据实例列表，每个实例是包含相同键的字典。

        Returns:
            Dict[str, Any]: 碎片化后的批次数据字典。每个键对应的值是列表：
                - 对于原为 np.ndarray 的键，列表包含 torch.Tensor
                - 对于其他键，列表包含原始类型
        """
        collated_dict = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], np.ndarray):
                collated_dict[key] = [torch.from_numpy(inst[key]) for inst in batch]
            else:  # List, Int
                collated_dict[key] = [inst[key] for inst in batch]
        return collated_dict


class OccCaptionOneFileDataset(OccCaptionDataset):
    """占用栅格（Occupancy Grid）序列数据集类。

    用于加载带有 Captions 的 4D Occupancy Grid 序列数据，支持通过 data_root 加载文件
    或直接加载预构建的 instance 数据。
    """

    def __init__(
            self,
            *args,
            **kwargs
    ) -> None:
        """初始化 Occupancy Grid 序列数据集。

        Args:
            data_root: 数据集根目录路径。
            resize_cfg: 调整 Occupancy Grid 大小的配置，包含 'origin' 和 'after' 键。默认为 None。
            load_occ: 是否加载 Occupancy Grid 数据。默认为 True。
            instance: 直接在外部构建好所需的数据实例列表。每个实例是包含
                'scene_files'、'scene_name' 和 'caption' 的字典。默认为 None。
        """
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> OccCaptionDatasetInstance:
        """获取指定索引的数据实例。

        Args:
            index: 数据实例的索引。

        Returns:
            OccCaptionDatasetInstance: 包含以下键的字典：
                - data_root (str): 数据根目录路径。
                - seq_len (int): 序列长度（帧数）。
                - scene_name (str): 场景名称。
                - instance_path (List[str]): Occupancy 文件路径列表。
                - occ_label (np.ndarray): Occupancy Grid 标签，形状为 (F, H, W, D)，
                  其中 F 为帧数，H/W 为高度/宽度，D 为深度。
                  形状示例: [frame, 200, 200, 16]
                - scene_caption (str): 场景描述文本。
        """
        data_dict = self.instances[index]
        occ_path = data_dict['scene_files']  # 一个文件存储完整的序列

        # 传入的 occ_paths 是不定长的视频
        if self.load_occ:
            if self.data_root:
                occ_labels = self.__load_file__(os.path.join(self.data_root, occ_path))
            else:
                occ_labels = self.__load_file__(occ_path)

            occ_labels = np.stack(occ_labels, axis=0)
            if self.resize_cfg:
                occ_labels = occ_resize(occ_labels, (self.resize_cfg['after'], self.resize_cfg['after']))
        else:
            # 伪数据
            occ_labels = np.zeros((1, 1, 1, 1))

        return {
            "data_root": self.data_root,
            "seq_len": occ_labels.shape[0],
            "scene_name": data_dict['scene_name'],  # str
            "instance_path": occ_path,  # str
            "occ_label": occ_labels,  # (F, H, W, D)  seq_len*高*宽*深 [frame, 200, 200, 16]
            "scene_caption": data_dict['caption'],
        }
