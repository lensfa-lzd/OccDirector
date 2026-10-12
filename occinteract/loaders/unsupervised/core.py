import os
from functools import lru_cache
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from occinteract.tools.occ_resizer import occ_resize
from occinteract.type_models.dataset import OccDatasetInstance, OccInstance


class OccReconstructionDataset(Dataset):
    """用于自监督 Occupancy 序列重建的 Dataset。

    Dataset 加载序列化的 Occupancy grid 数据，不使用任何文本标注。
    """

    def __init__(
            self,
            resize_cfg: Optional[Dict[str, int]] = None,
            load_occ: bool = True,
            instance: Optional[List[OccInstance]] = None,
    ) -> None:
        """初始化 Occupancy 重建 Dataset。

        Args:
            resize_cfg: 包含 'origin' 和 'after' 键的字典，用于数据缩放。
            load_occ: 是否加载实际的 Occupancy 数据。
            instance: 外部构建好的数据实例列表，每个实例包含
                'scene_name' 和 'occ_in_scene_paths' 键。
        """
        super().__init__()
        self.resize_cfg = resize_cfg
        self.load_occ = load_occ
        self.instances = instance

    def __len__(self) -> int:
        """返回 Dataset 中实例的数量。

        Returns:
            int: 实例的总数。
        """
        return len(self.instances)

    @lru_cache(maxsize=16)
    def __load_file__(self, path: str) -> np.ndarray:
        """加载并缓存 NumPy 文件以避免重复 I/O 开销。

        Args:
            path: NumPy 文件路径。

        Returns:
            np.ndarray: 加载的数组数据。

        Raises:
            FileNotFoundError: 如果文件不存在或为空。
            Exception: 其他加载异常。
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

    def __getitem__(self, index: int) -> OccDatasetInstance:
        """获取指定索引的数据样本。

        Args:
            index: 数据样本的索引。

        Returns:
            OccDatasetInstance: 包含以下键的字典：
                - "seq_len": 序列长度。
                - "scene_name": 场景名称。
                - "instance_path": occupancy 文件路径列表。
                - "occ_data": Occupancy 数据，形状为 (F, H, W, D)，其中 F 为帧数，H 为高度，
                             W 为宽度，D 为深度。
                - "occ_label": Occupancy 标签，与 occ_data 相同，形状为 (F, H, W, D)。
        """
        data_dict = self.instances[index]
        occ_paths = data_dict['occ_in_scene_paths']

        # 传入的 occ_paths 是不定长的视频
        if self.load_occ:
            occ_data = [
                self.__load_file__(path)
                for path in occ_paths
            ]
            occ_data = np.stack(occ_data, axis=0)
            if self.resize_cfg:
                occ_data = occ_resize(occ_data, (self.resize_cfg['after'], self.resize_cfg['after']))
        else:
            # 伪数据
            occ_data = np.zeros((1, 1, 1, 1))

        return {
            "seq_len": len(occ_paths),
            "scene_name": data_dict['scene_name'],  # str
            "instance_path": occ_paths,  # List[str]
            "occ_data": occ_data,  # (F, H, W, D) - [frame, height, width, depth], e.g., [F, 200, 200, 16]
            "occ_label": occ_data,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> OccDatasetInstance:
        """将 batch 中的数据整理为列表形式。

        Args:
            batch: 包含多个数据样本的列表。

        Returns:
            OccDatasetInstance: 整理后的 batch 数据。
        """
        collated_dict = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], np.ndarray):
                collated_dict[key] = [torch.from_numpy(inst[key]) for inst in batch]
            else:  # List, Int
                collated_dict[key] = [inst[key] for inst in batch]
        return collated_dict
