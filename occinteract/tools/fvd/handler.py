import os
import pickle
import random
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from einops import repeat, rearrange

from occinteract.tools.fvd.fvd import load_i3d_pretrained, get_fvd_feats, frechet_distance
from occinteract.core.base_handler import BaseVisualizationHandler


class OccFVDHandler(pl.LightningModule):
    """Occupancy 序列的 FVD（Fréchet Video Distance）计算处理器。

    使用 I3D 模型提取视频特征，然后计算生成数据与真实数据之间的 FVD 距离。
    支持多进程分布式计算，每个进程处理部分数据后在 Rank 0 合并结果。
    """

    def __init__(
            self,
            output_file: Optional[str] = None,
            data_warper_func: Optional[Callable] = None
    ) -> None:
        """初始化 FVD 处理器。

        Args:
            output_file: 最终结果保存的文件路径。
            data_warper_func: 数据转换函数，用于将 batch 转换为可处理的数据列表。
        """
        super().__init__()
        self.output_file = output_file
        self.data_warper_func = data_warper_func

        self.process_result = []
        self.i3d = load_i3d_pretrained(device='cpu')

        self.vis_handler = BaseVisualizationHandler()
        self.occ_render_kwargs = {
            'elev': 90,
            'radius_factor': 1.5,
        }
        self.image_size = (512, 512)

    @staticmethod
    def load_data_file(file_path: str) -> Any:
        """从指定路径加载 pickle 数据文件。

        Args:
            file_path: 数据文件的路径。

        Returns:
            Any: 加载的数据对象。
        """
        with open(file_path, 'rb') as f:
            result = pickle.load(f)
        return result

    def _save_process_result(self, file_path: str) -> None:
        """将处理结果保存到 pickle 文件。

        Args:
            file_path: 保存文件的路径。
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self.process_result, f)

    def render_occ_video(self, occ_seq):
        video_frame = []
        for i in range(occ_seq.shape[0]):
            img_bgr = self.vis_handler.render_bev_image(
                -1,
                occ_seq[i].to('cpu').numpy(),
                image_size=self.image_size,
                **self.occ_render_kwargs,
            )
            video_frame.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        return video_frame

    def process_single_occ_seq(self, occ_seq: torch.Tensor) -> Optional[torch.Tensor]:
        """处理单个 occupancy 序列，提取其 FVD 特征。

        将 occupancy 序列渲染为视频帧，然后使用 I3D 模型提取视频特征向量。
        如果序列长度小于 10 帧，返回 None。

        Args:
            occ_seq: Occupancy 序列张量，形状为 [F, H, W, D]，其中 F 为帧数，H/W/D 为空间维度。

        Returns:
            Optional[torch.Tensor]: FVD 特征向量，形状为 [1, 400]。如果序列长度不足 10 则返回 None。
        """
        if occ_seq.shape[0] < 10:
            return None

        device = occ_seq.device
        video_frame = self.render_occ_video(occ_seq)
        video = torch.from_numpy(np.stack(video_frame, axis=0)).to(device)
        video = repeat(video, 'f h w c -> bs c f h w', bs=1)
        video = video / 255.  # [0, 255] -> [0, 1]
        feats = get_fvd_feats(video, i3d=self.i3d, device=device, bs=1)  # [1, 400] 的向量，代表视频特征
        return feats

    def process_batch_occ_seq(self, occ_seq: torch.Tensor) -> Optional[torch.Tensor]:
        """处理多个 occupancy 序列，提取其 FVD 特征。 适用于长度一致的序列。

        将 occupancy 序列渲染为视频帧，然后使用 I3D 模型提取视频特征向量。
        如果序列长度小于 10 帧，返回 None。

        Args:
            occ_seq: Occupancy 序列张量，形状为 [Bs, F, H, W, D]，其中 F 为帧数，H/W/D 为空间维度。

        Returns:
            Optional[torch.Tensor]: FVD 特征向量，形状为 [Bs, 400]。如果序列长度不足 10 则返回 None。
        """
        if occ_seq.shape[1] < 10:
            return None

        device = occ_seq.device

        videos = []
        for i in range(occ_seq.shape[0]):
            sub_seq = occ_seq[i]
            video_frame = self.render_occ_video(sub_seq)
            videos.append(
                np.stack(video_frame, axis=0)  # [F, H, W, C]
            )

        videos = torch.from_numpy(np.stack(videos, axis=0)).to(device)
        videos = rearrange(videos, 'bs f h w c -> bs c f h w')
        videos = videos / 255.  # [0, 255] -> [0, 1]
        feats = get_fvd_feats(videos, i3d=self.i3d, device=device, bs=1)  # [bs, 400] 的向量，代表视频特征
        return feats

    def single_step(self, batch: torch.Tensor) -> None:
        """处理单个 batch 的数据。

        因为 occupancy 序列长度不定，每个序列需要单独处理。

        Args:
            batch: 输入的 batch 数据张量。
        """
        # 因为 occ 序列不定长，所以每个序列单独处理
        item_list: list = self.data_warper_func(batch)
        for item in item_list:
            key: str = item[0]
            occ_seq: torch.Tensor = item[1]

            feats = self.process_single_occ_seq(occ_seq)
            self.process_result.append((key, feats))

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        """PyTorch Lightning 的测试步骤。

        Args:
            batch: 输入的 batch 数据张量。
            batch_idx: 当前 batch 的索引。

        Returns:
            float: 占位符返回值，实际数据在 on_test_epoch_end 中处理。
        """
        self.single_step(batch)
        return random.random()  # 占位符

    def on_test_epoch_end(self) -> None:
        """测试 epoch 结束时的回调方法。

        在多进程分布式环境下，每个进程保存自己的处理结果，并在 Rank 0 处合并所有结果并保存到最终文件。
        """
        # 获取当前进程的 rank
        rank = self.trainer.global_rank

        # 每个进程保存自己的文件部分，例如 result_part_0.pkl, result_part_1.pkl
        file_name = f"result_part_{rank}.pkl"
        self._save_process_result(file_name)

        # 等待所有进程保存完毕
        self.trainer.strategy.barrier()

        # 只在 Rank 0 运行合并脚本
        if self.trainer.is_global_zero:
            full_result = []
            for r in range(self.trainer.world_size):
                part_path = f"result_part_{r}.pkl"
                part_data = self.load_data_file(part_path)
                full_result.extend(part_data)
                # 可选：删除分片文件
                # os.remove(part_path)

            self.process_result = full_result

            # 保存最终文件（裸文件名没有目录部分，makedirs('') 会崩溃）
            out_dir = os.path.dirname(self.output_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(self.output_file, 'wb') as f:
                pickle.dump(full_result, f)

            print(f'Result save to file: {self.output_file}')

    @staticmethod
    def filter_result(data: List[Tuple[str, Optional[torch.Tensor]]]) -> List[torch.Tensor]:
        """过滤掉结果为 None 的数据。

        Args:
            data: 数据列表，每个元素为 (key, feats) 元组，feats 可能为 None。

        Returns:
            List[torch.Tensor]: 过滤后的特征向量列表，只保留非 None 的值。
        """
        result = []
        for k, v in data:
            if v is not None:
                result.append(v)

        return result

    @staticmethod
    def get_fvd(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """计算两组特征之间的 FVD 距离。

        Args:
            feat1: 第一组特征向量，形状为 [bs, 400]。
            feat2: 第二组特征向量，形状为 [bs, 400]。

        Returns:
            float: 两组特征之间的 FVD 距离值。
        """
        # feat [bs, 400]
        return frechet_distance(feat1, feat2)
