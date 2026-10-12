import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from occinteract.tools.occ_resizer import occ_resize
from occinteract.tools.uniocc_dataset import UniOcc
from occinteract.type_models.dataset import OccDatasetInstance


class BaseDataset(UniOcc):
    """用于加载不附带标签的 Occupancy 序列数据集的基类。

    继承自 UniOcc 类，支持从指定路径加载 Occupancy 数据。
    可以加载固定长度的序列或完整场景，支持降采样和数据调整大小。

    Attributes:
        instances: 存储所有数据实例的列表，每个元素为 (scene_name, occ_paths) 元组。
        seq_len: 加载的序列长度。如果为正数，则使用滑动窗口截取固定长度序列；
                 如果为 -1，则加载完整场景。
        resize_cfg: 用于调整数据大小的配置字典，包含 'after' 键指定目标尺寸。
        load_occ: 是否加载 Occupancy 数据。如果为 False，则返回伪数据（仅用于测试）。
        use_8bit_data: 是否使用 8 位格式的数据。如果为 True，则加载 .npy 文件；
                       否则从 .npz 文件中提取 'occ_label'。
        resample: 是否需要对数据进行重新采样（如从 10Hz 降采样到 2Hz）。
                 处理逻辑在 get_seq_instance 中。
    """

    def __init__(
            self,
            data_root: str,
            seq_len: int,
            resize_cfg: Optional[Dict] = None,
            load_occ: bool = True,
            use_8bit_data: bool = True
    ):
        """初始化 BaseDataset。

        Args:
            data_root: 数据集根路径。
            seq_len: 加载的序列长度。如果为 -1，则加载完整场景。
            resize_cfg: 用于调整数据大小的配置字典，包含 'after' 键指定目标尺寸。
            load_occ: 是否加载 Occupancy 数据。如果为 False，则返回伪数据。
            use_8bit_data: 是否使用 8 位格式的数据。如果为 True，则加载 .npy 文件；
                           否则从 .npz 文件中提取。
        """
        super().__init__(data_root, -1, -1, construct_instances=False)

        self.instances = []
        self.seq_len = seq_len
        self.resize_cfg = resize_cfg
        self.load_occ = load_occ
        self.use_8bit_data = use_8bit_data

        if '10Hz' in data_root:
            # 统一帧率，重新采样为 2Hz
            self.resample = True
        else:
            self.resample = False

        for scene in self.scene_infos:
            scene_name = scene["scene_name"]
            if 'OpV2V' in data_root:
                for ego in scene['egos']:
                    # Broken file uniocc/OpV2V-10Hz-train/2021_08_16_22_26_54/650/000443.npz
                    if scene_name == '2021_08_16_22_26_54' and ego['ego_name'] == '650':
                        # 丢弃这个场景 OpV2V-10Hz-train/2021_08_16_22_26_54/650
                        continue

                    self.instances += self.get_seq_instance(scene_name, ego["occ_in_scene_paths"])
            else:
                self.instances += self.get_seq_instance(scene_name, scene["occ_in_scene_paths"])

        logging.warning(f"Loaded {len(self.instances)} instances from {data_root}.")

    def get_seq_instance(self, scene_name: str, raw_scene_files: List[str]) -> List[Tuple[str, List[str]]]:
        """从场景文件列表中生成序列实例。

        根据指定的序列长度，使用滑动窗口从场景文件中生成多个序列实例。
        如果 seq_len 为 -1，则直接返回整个场景作为单个实例。

        Args:
            scene_name: 场景名称。
            raw_scene_files: 场景中所有文件的路径列表。

        Returns:
            包含 (scene_name, file_list) 元组的列表，每个元组代表一个序列实例。
        """
        instances_subset = []

        if len(raw_scene_files) == 0:
            logging.warning(
                f"Scene {scene_name} - has no frames."
            )
            return instances_subset
        if self.resample:
            # 将 10Hz 降采样到 2Hz
            scene_files = raw_scene_files[::5]
        else:
            scene_files = raw_scene_files

        if self.seq_len > 0:
            if len(scene_files) < self.seq_len:
                logging.warning(
                    f"Skipping scene {scene_name} - insufficient frames ({len(scene_files)}) for seq_len {self.seq_len}."
                )
                return instances_subset

            # Construct sliding-window segments.
            limit = len(scene_files) - self.seq_len + 1
            for i in range(limit):
                sub_seq = scene_files[i: i + self.seq_len]
                instances_subset.append((scene_name, sub_seq))
        else:
            # 直接加载整个场景 / 最终返回的长度不定
            instances_subset.append((scene_name, scene_files))

        return instances_subset

    def __getitem__(self, index: int) -> OccDatasetInstance:
        """获取指定索引的数据样本。

        Args:
            index: 数据样本的索引。

        Returns:
            包含以下键的字典：
                - "data_root": 数据集根路径。
                - "seq_len": 序列长度。
                - "scene_name": 场景名称。
                - "instance_path": 实例路径列表。
                - "occ_data": Occupancy 数据，形状为 (F, H, W, D)，其中 F 为帧数，H 为高度，
                             W 为宽度，D 为深度。
                - "occ_label": Occupancy 标签，与 occ_data 相同，形状为 (F, H, W, D)。
        """
        scene_name, occ_paths = self.instances[index]

        if self.load_occ:
            if self.use_8bit_data:
                occ_labels = [
                    self.__load_file__(os.path.join(self.data_root, path.replace('npz', 'npy')))
                    for path in occ_paths
                ]
            else:
                occ_labels = [
                    self.__load_file__(os.path.join(self.data_root, path))["occ_label"]
                    for path in occ_paths
                ]

            occ_labels = np.stack(occ_labels, axis=0)
            if self.resize_cfg:
                occ_labels = occ_resize(occ_labels, (self.resize_cfg['after'], self.resize_cfg['after']))
        else:
            # 伪数据（仅用于测试）
            if self.seq_len > 0:
                occ_labels = np.zeros((self.seq_len, 1, 1, 1))
            else:
                occ_labels = np.zeros((1, 1, 1, 1))

        return {
            "data_root": self.data_root,
            "seq_len": self.seq_len,
            "scene_name": scene_name,  # str
            "instance_path": occ_paths,  # List[str]
            "occ_data": occ_labels,  # (F, H, W, D)  seq_len*高*宽*深 [frame, 200, 200, 16]
            "occ_label": occ_labels,  # (F, H, W, D)
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """将一批数据样本整理成批次。

        如果 seq_len > 0，则使用父类 UniOcc 的 collate_fn 进行批量拼接。
        如果 seq_len == -1（可变长度序列），则返回列表形式的批次。

        Args:
            batch: 数据样本的列表，每个样本是一个字典。

        Returns:
            如果 seq_len > 0，返回拼接后的批次字典。
            如果 seq_len == -1，返回字典，其中每个键对应的值为样本列表。
        """
        seq_len = batch[0]['seq_len']
        if seq_len > 0:
            return UniOcc.collate_fn(batch)
        else:
            # 在 frame 长度不一致的情况下不能直接拼接
            collated_dict = {}
            for key in batch[0].keys():
                if isinstance(batch[0][key], np.ndarray):
                    collated_dict[key] = [torch.from_numpy(inst[key]) for inst in batch]
                else:  # List, Int
                    collated_dict[key] = [inst[key] for inst in batch]
            return collated_dict
