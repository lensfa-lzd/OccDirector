"""用于无监督训练的数据集模块。

该模块提供基于 UniOcc 原始 npz 数据的无监督数据集类，用于重建训练
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from occinteract.tools.occ_resizer import occ_resize
from occinteract.tools.uniocc_dataset import UniOcc


class BaseDataset(UniOcc):
    """基础无监督数据集类，继承自 UniOcc。

    用于重建训练，数据本身即标签（occ_data 和 occ_label 相同）。
    支持加载 8bit 格式数据、重采样和数据尺寸调整。

    Attributes:
        load_occ (bool): 是否加载 Occupancy 数据。
        resize_cfg (Optional[dict]): 数据尺寸调整配置。
        use_8bit_data (bool): 是否优先使用 8bit 格式数据。
        instances (List[Tuple[str, str]]): 实例列表，每个元素为 (scene_name, occ_path)。
    """

    def __init__(
            self,
            data_root: str,
            resample_to_2hz: bool = False,
            resize_cfg: Optional[dict] = None,
            load_occ: bool = True,
            use_8bit_data: bool = True
    ):
        """初始化 BaseDataset。

        Args:
            data_root (str): 数据集根路径。
            resample_to_2hz (bool, optional): 是否将 10Hz 数据降采样到 2Hz。默认为 False。
            resize_cfg (Optional[dict], optional): 数据尺寸调整配置。默认为 None。
            load_occ (bool, optional): 是否加载 Occupancy 数据。默认为 True。
            use_8bit_data (bool, optional): 是否优先使用 8bit 格式数据。默认为 True。
                实际行为可通过环境变量 USE_8_BIT_DATA 覆盖。
        """
        super().__init__(data_root, -1, -1, construct_instances=False)
        self.load_occ = load_occ
        self.resize_cfg = resize_cfg
        self.use_8bit_data = use_8bit_data

        self.instances = []
        for scene in self.scene_infos:
            scene_name = scene["scene_name"]
            if 'OpV2V' in data_root:
                scene_files = []
                for ego in scene['egos']:
                    scene_files += ego["occ_in_scene_paths"]

                    # Broken file uniocc/OpV2V-10Hz-train/2021_08_16_22_26_54/650/000443.npz
                    if scene_name == '2021_08_16_22_26_54' and ego['ego_name'] == '650':
                        scene_files.remove('2021_08_16_22_26_54\\650\\000443.npz')
            else:
                scene_files = scene["occ_in_scene_paths"]

            if len(scene_files) == 0:
                logging.warning(f"Scene {scene_name} - has no frames.")

            if resample_to_2hz and '10Hz' in data_root:
                # 将 10Hz 降采样到 2Hz
                scene_files = scene_files[::5]

            # 重建训练，数据本身即为标签
            for file in scene_files:
                self.instances.append((scene_name, file))

        logging.warning(f"Loaded {len(self.instances)} instances from {data_root}.")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """获取指定索引的数据样本。

        Args:
            index (int): 数据样本的索引。

        Returns:
            Dict[str, Any]: 包含以下键的字典:
                - data_root (str): 数据根路径。
                - scene_name (str): 场景名称。
                - instance_path (str): 实例文件相对路径。
                - abs_instance_path (str): 实例文件绝对路径。
                - occ_data (np.ndarray): Occupancy 数据，形状为 (H, W, D)，如 (200, 200, 16)。
                - occ_label (np.ndarray): 标签，与 occ_data 相同。
        """
        scene_name, occ_path = self.instances[index]

        use_8bit_data = False
        if self.use_8bit_data:
            use_8bit_data = True

        if os.getenv('USE_8_BIT_DATA') == '1':
            use_8bit_data = True
        elif os.getenv('USE_8_BIT_DATA') == '0':
            use_8bit_data = False

        if self.load_occ:
            if use_8bit_data:
                occ = self.__load_file__(os.path.join(self.data_root, occ_path.replace('npz', 'npy')))
                if self.resize_cfg:
                    occ = occ_resize(occ, (self.resize_cfg['after'], self.resize_cfg['after']))
            else:
                occ = np.array(self.__load_file__(os.path.join(self.data_root, occ_path))["occ_label"])
        else:
            # 伪数据
            occ = np.zeros((1, 1, 1))

        # 重建，数据本身即标签
        return {
            "data_root": self.data_root,
            "scene_name": scene_name,  # str
            "instance_path": occ_path,  # str
            "abs_instance_path": os.path.join(self.data_root, occ_path),
            "occ_data": occ,  # (H, W, D)  高*宽*深 [200, 200, 16]
            "occ_label": occ,  # (H, W, D)
        }
