# Copyright (c) 2025. All rights reserved.
# Licensed under the MIT License.
# Absorb from https://github.com/tasl-lab/uniocc
# uniocc_dataset.py

"""UniOcc 统一占据数据集加载器,支持多源数据集(nuScenes, Waymo, Carla)。"""

import logging
import os
import pickle
from functools import lru_cache
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class UniOcc(Dataset):
    """UniOcc 统一占据预测数据集,支持历史观测和未来预测帧。

    数据集里面的 Occ 场景大小全部统一为 200*200*16。
    类别标签统一为 11 个。

    类别映射表(不同数据源到 UniOcc 格式):
    | ID | nuScenes             | Waymo             | Carla        | UniOcc (Ours)    |
    |:---|:---------------------|:------------------|:-------------|:-----------------|
    | 0  | general_object       | general_object    | free         | general_object   |
    | 1  | barrier              | vehicle           | buildings    | vehicle          |
    | 2  | bicycle              | pedestrian        | fences       | bicycle          |
    | 3  | bus                  | sign              | other        | motorcycle       |
    | 4  | car                  | cyclist           | pedestrians  | pedestrian       |
    | 5  | construction_vehicle | traffic_light     | poles        | traffic_cone     |
    | 6  | motorcycle           | pole              | roadlines    | vegetation       |
    | 7  | pedestrian           | construction_cone | roads        | road             |
    | 8  | traffic_cone         | bicycle           | sidewalks    | walkable/terrain |
    | 9  | trailer              | motorcycle        | vegetation   | building         |
    | 10 | truck                | building          | vehicles     | free             |
    | 11 | drivable_surface     | vegetation        | walls        | -                |
    | 12 | other_flat           | tree_trunk        | trafficSigns | -                |
    | 13 | sidewalk             | road              | sky          | -                |
    | 14 | terrain              | walkable          | ground       | -                |
    | 15 | manmade              | -                 | -            | -                |
    | 16 | vegetation           | -                 | -            | -                |
    | 17 | free                 | -                 | -            | -                |
    | 23 | -                    | free              | -            | -                |

    Attributes:
        label_range (List[int]): 有效标签范围 [min_id, max_id],通常为 [0, 10]
        instances (List[Tuple]): 数据实例列表,每个元素为 (scene_name, occ_paths)
    """

    def __init__(
            self,
            data_root: str,
            obs_len: int,
            fut_len: int,
            datasource_root: Optional[str] = None,
            construct_instances: bool = True
    ) -> None:
        """初始化 UniOcc 数据集。

        Args:
            data_root (str): 占据数据根目录,如 "unified_datasets/NuScenes-via-Occ3D-2Hz-mini"
            obs_len (int): 用作输入的历史观测帧数
            fut_len (int): 需要预测的未来帧数
            datasource_root (Optional[str]): [仅用于相机图像] nuScenes, Waymo, OpV2V 等
                数据集的根目录。如果为 None 则不加载相机数据。默认: None
            construct_instances (bool): 是否自动构建数据实例列表。
                对于自定义数据集可设为 False。默认: True
        """
        self.data_root = data_root
        self.obs_len = obs_len
        self.fut_len = fut_len
        self.provide_camera = datasource_root is not None
        self.datasource_root = datasource_root
        self.label_range = [0, 10]

        # Load scene meta-info.
        if os.path.isfile(os.path.join(data_root, 'scene_infos.pkl')):
            with open(os.path.join(data_root, 'scene_infos.pkl'), 'rb') as f:
                self.scene_infos = pickle.load(f)
        else:
            self.scene_infos = []
            logging.warning(f"Scene_infos file not found at {os.path.join(data_root, 'scene_infos.pkl')}.")

        # 非的情况用于自定义数据集
        self.instances = []
        if construct_instances:
            for scene in self.scene_infos:
                scene_name = scene["scene_name"]
                scene_files = scene["occ_in_scene_paths"]

                # Verify scene length sufficiency.
                if len(scene_files) < (self.obs_len + self.fut_len):
                    logging.warning(
                        f"Skipping scene {scene_name} - insufficient frames ({len(scene_files)})."
                    )
                    continue

                # Construct sliding-window segments.
                limit = len(scene_files) - (self.obs_len + self.fut_len) + 1
                for i in range(limit):
                    sub_seq = scene_files[i: i + self.obs_len + self.fut_len]
                    self.instances.append((scene_name, sub_seq))

            logging.info(f"Loaded {len(self.instances)} instances from {data_root}.")

    def __len__(self):
        return len(self.instances)

    def __pad_on_dim__(self, arr: np.ndarray, pad_to: int, axis: int) -> np.ndarray:
        """沿指定轴将数组填充到特定长度。

        Args:
            arr (np.ndarray): 输入数组
            pad_to (int): 目标长度
            axis (int): 填充的轴

        Returns:
            np.ndarray: 填充后的数组,使用 0 填充
        """
        pad_len = pad_to - arr.shape[axis]
        padding = [(0, 0)] * len(arr.shape)
        padding[axis] = (0, pad_len)
        return np.pad(arr, padding, mode="constant", constant_values=0)

    @lru_cache(maxsize=16)
    def __load_file__(self, path: str) -> np.lib.npyio.NpzFile:
        """加载并缓存 NumPy 文件以避免重复 I/O 开销。

        使用 LRU 缓存机制,最多缓存 16 个文件。

        Args:
            path (str): .npz 文件的路径

        Returns:
            np.lib.npyio.NpzFile: 加载的 npz 文件对象

        Raises:
            FileNotFoundError: 如果文件不存在或为空
            Exception: 其他加载错误
        """

        # OpV2V-10Hz-train/2021_08_16_22_26_54\\659\\000795.npz
        path = path.replace('\\', '/')
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

    @lru_cache(maxsize=16)
    def __load_image__(self, path: str) -> np.ndarray:
        """加载并缓存图像文件以避免重复 I/O 开销。

        使用 LRU 缓存机制,最多缓存 16 个图像。

        Args:
            path (str): 图像文件的路径

        Returns:
            np.ndarray: 图像数组。形状: (H, W, 3),dtype: uint8
        """
        return np.array(Image.open(path))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """获取单个训练实例的观测序列和未来序列。

        Args:
            index (int): 数据实例索引

        Returns:
            Dict[str, Any]: 包含以下键的字典:
                - scene_name (str): 场景名称
                - scene_token (str): 当前帧的 token
                - instance_path (str): 当前帧的路径
                - obs_occ_labels (np.ndarray): 观测帧占据标签。形状: (obs_len, 200, 200, 16)
                - fut_occ_labels (np.ndarray): 未来帧占据标签。形状: (fut_len, 200, 200, 16)
                - obs_flows_forward (np.ndarray): 观测帧前向光流。形状: (obs_len, 200, 200, 16, 3)
                - fut_flows_forward (np.ndarray): 未来帧前向光流。形状: (fut_len, 200, 200, 16, 3)
                - obs_flows_backward (np.ndarray): 观测帧后向光流。形状: (obs_len, 200, 200, 16, 3)
                - fut_flows_backward (np.ndarray): 未来帧后向光流。形状: (fut_len, 200, 200, 16, 3)
                - obs_ego_to_worlds (np.ndarray): 观测帧 ego 到世界坐标变换。形状: (obs_len, 4, 4)
                - fut_ego_to_worlds (np.ndarray): 未来帧 ego 到世界坐标变换。形状: (fut_len, 4, 4)
                - num_cameras (int): 相机数量
                - obs_cameras_image (np.ndarray): 观测帧相机图像。形状: (obs_len, MAX_CAM, H, W, 3)
                - fut_cameras_image (np.ndarray): 未来帧相机图像。形状: (fut_len, MAX_CAM, H, W, 3)
                - obs_cameras_fov_mask (np.ndarray): 观测帧相机 FOV 掩码。形状: (obs_len, 200, 200, 16)
                - fut_cameras_fov_mask (np.ndarray): 未来帧相机 FOV 掩码。形状: (fut_len, 200, 200, 16)
                - obs_cameras_intrinsics (np.ndarray): 观测帧相机内参。形状: (obs_len, MAX_CAM, 3, 3)
                - fut_cameras_intrinsics (np.ndarray): 未来帧相机内参。形状: (fut_len, MAX_CAM, 3, 3)
                - obs_cameras_extrinsics (np.ndarray): 观测帧相机外参。形状: (obs_len, MAX_CAM, 4, 4)
                - fut_cameras_extrinsics (np.ndarray): 未来帧相机外参。形状: (fut_len, MAX_CAM, 4, 4)
                - annotations (List): 标注字典列表。长度: obs_len + fut_len
        """
        scene_name, occ_paths = self.instances[index]
        scene_tokens = [
            os.path.splitext(os.path.basename(path))[0] for path in occ_paths
        ]

        # Load occupancy labels
        occ_labels = [
            self.__load_file__(os.path.join(self.data_root, path))["occ_label"]
            for path in occ_paths
        ]
        occ_labels = np.array(occ_labels)

        # Load forward flows
        flows_forward = [
            self.__load_file__(os.path.join(self.data_root, path))["occ_flow_forward"]
            for path in occ_paths
        ]
        flows_forward = np.array(flows_forward)

        # Load backward flows
        flows_backward = [
            self.__load_file__(os.path.join(self.data_root, path))["occ_flow_backward"]
            for path in occ_paths
        ]
        flows_backward = np.array(flows_backward)

        # Load ego-to-world transformations
        ego_to_worlds = [
            self.__load_file__(os.path.join(self.data_root, path))[
                "ego_to_world_transformation"
            ]
            for path in occ_paths
        ]
        ego_to_worlds = np.array(ego_to_worlds)

        # Load annotations
        ann_dicts = [
            self.__load_file__(os.path.join(self.data_root, path))["annotations"]
            for path in occ_paths
        ]

        # Load camera FOV Mask
        cameras_fov_masks = [
            self.__load_file__(os.path.join(self.data_root, path))["occ_mask_camera"]
            for path in occ_paths
        ]

        cameras_image = []
        cameras_intrinsics = []
        cameras_extrinsics = []
        num_cameras = 0
        if self.provide_camera:
            # Load per-camera data
            cameras_infos = [
                self.__load_file__(os.path.join(self.data_root, path))["cameras"]
                for path in occ_paths
            ]
            num_cameras = len(cameras_infos[0])

            for t in range(len(occ_paths)):
                cameras_image.append([self.__load_image__(os.path.join(self.datasource_root, camera_info['filename']))
                                      for camera_info in cameras_infos[t]])
                cameras_intrinsics.append([camera_info['intrinsics'] for camera_info in cameras_infos[t]])
                cameras_extrinsics.append([camera_info['extrinsics'] for camera_info in cameras_infos[t]])

            # Stack over time
            cameras_image = np.array(cameras_image)
            cameras_intrinsics = np.array(cameras_intrinsics)
            cameras_extrinsics = np.array(cameras_extrinsics)

            # Pad to the max number of cameras (for example we assume 6 here).
            cameras_image = self.__pad_on_dim__(cameras_image, 6, axis=1)
            cameras_intrinsics = self.__pad_on_dim__(cameras_intrinsics, 6, axis=1)
            cameras_extrinsics = self.__pad_on_dim__(cameras_extrinsics, 6, axis=1)

        return {
            "scene_name": scene_name,  # str
            "scene_token": scene_tokens[self.obs_len],  # str
            "instance_path": occ_paths[self.obs_len],  # str
            "obs_occ_labels": occ_labels[:self.obs_len],  # (obs_len, L, W, H)
            "fut_occ_labels": occ_labels[self.obs_len:],  # (fut_len, L, W, H)
            "obs_flows_forward": flows_forward[:self.obs_len],  # (obs_len, L, W, H, 3)
            "fut_flows_forward": flows_forward[self.obs_len:],  # (fut_len, L, W, H, 3)
            "obs_flows_backward": flows_backward[:self.obs_len],  # (obs_len, L, W, H, 3)
            "fut_flows_backward": flows_backward[self.obs_len:],  # (fut_len, L, W, H, 3)
            "obs_ego_to_worlds": ego_to_worlds[:self.obs_len],  # (obs_len, 4, 4)
            "fut_ego_to_worlds": ego_to_worlds[self.obs_len:],  # (fut_len, 4, 4)
            "num_cameras": num_cameras,  # int
            "obs_cameras_image": cameras_image[:self.obs_len],  # (obs_len, MAX_CAM, H, W, 3)  [0 ~ 255]
            "fut_cameras_image": cameras_image[self.obs_len:],  # (fut_len, MAX_CAM, H, W, 3)  [0 ~ 255]
            "obs_cameras_fov_mask": cameras_fov_masks[:self.obs_len],  # (obs_len, L, W, H)         Binary mask
            "fut_cameras_fov_mask": cameras_fov_masks[self.obs_len:],  # (fut_len, L, W, H)         Binary mask
            "obs_cameras_intrinsics": cameras_intrinsics[:self.obs_len],  # (obs_len, MAX_CAM, 3, 3)
            "fut_cameras_intrinsics": cameras_intrinsics[self.obs_len:],  # (fut_len, MAX_CAM, 3, 3)
            "obs_cameras_extrinsics": cameras_extrinsics[:self.obs_len],  # (obs_len, MAX_CAM, 4, 4)
            "fut_cameras_extrinsics": cameras_extrinsics[self.obs_len:],  # (fut_len, MAX_CAM, 4, 4)
            "annotations": ann_dicts  # (obs_fut_len) of (num_annotations) dicts
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """将样本列表合并为统一的 batch 以进行高效处理。

        Args:
            batch (List[Dict[str, Any]]): 样本字典的列表

        Returns:
            Dict[str, Any]: 合并后的 batch 字典。
                对于 np.ndarray 类型,沿 batch 维度堆叠并转换为 torch.Tensor。
                对于其他类型(List, Int 等),保持为列表形式。
        """
        collated_dict = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], np.ndarray):
                collated_dict[key] = torch.tensor(np.stack([inst[key] for inst in batch]))
            else:  # List, Int
                collated_dict[key] = [inst[key] for inst in batch]
        return collated_dict


def check_label(occ_data: torch.Tensor) -> Tuple[int, int]:
    """检查占据数据中的标签范围。

    Args:
        occ_data (torch.Tensor): 占据数据张量

    Returns:
        Tuple[int, int]: (最大标签值, 最小标签值)
    """
    return int(torch.max(occ_data)), int(torch.min(occ_data))
