import json
import os
import pickle
import random
import sys
from os.path import abspath, dirname
from typing import Optional

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from prdc import compute_prdc
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance

project_root = dirname(dirname(abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from occinteract.core.base_handler import BaseVisualizationHandler


class Occ2DMetricHandler(pl.LightningModule):
    """仅支持单卡运行
    """

    def __init__(
            self,
            batch_size,
            output_file: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.output_file = output_file

        self.process_result = []
        self.vis_handler = BaseVisualizationHandler()
        self.occ_render_kwargs = {
            'elev': 90,
            'radius_factor': 1.5,
        }
        self.image_size = (299, 299)

        self.is_handler = InceptionScore(splits=10)
        self.is_handler_real = InceptionScore(splits=10)
        self.fid_handler = FrechetInceptionDistance(feature=2048)
        self.kid_handler = KernelInceptionDistance(subset_size=batch_size)

    def render_occ_frame(self, occ_frame: np.ndarray):
        img_bgr = self.vis_handler.render_bev_image(
            -1,
            occ_frame,
            image_size=self.image_size,
            **self.occ_render_kwargs,
        )
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def batch_occ_to_image_tensor(self, batch_occ):
        images = []
        for i in range(batch_occ.shape[0]):
            images.append(self.render_occ_frame(batch_occ[i]))

        image_tensor = torch.stack([torch.from_numpy(image) for image in images])
        image_tensor = rearrange(image_tensor, 'b h w c -> b c h w')
        return image_tensor.to(device=self.device, dtype=torch.uint8)

    def single_step(self, batch) -> None:
        """处理单个 batch 的数据。

        因为 occupancy 序列长度不定，每个序列需要单独处理。

        Args:
            batch: 输入的 batch 数据张量。
        """
        real_occ, fake_occ = batch  # Bs H W D

        real_images = self.batch_occ_to_image_tensor(real_occ)
        fake_images = self.batch_occ_to_image_tensor(fake_occ)

        # ==========================================
        # 1. 计算 Inception Score (IS)
        # 注意：IS 只需要生成(fake)的图片
        # ==========================================
        self.is_handler.update(fake_images)
        self.is_handler_real.update(real_images)

        # ==========================================
        # 2. 计算 Fréchet Inception Distance (FID)
        # 注意：FID 需要真实(real)和生成(fake)的图片
        # ==========================================
        self.fid_handler.update(real_images, real=True)
        self.fid_handler.update(fake_images, real=False)

        # ==========================================
        # 3. 计算 Kernel Inception Distance (KID)
        # ==========================================
        self.kid_handler.update(real_images, real=True)
        self.kid_handler.update(fake_images, real=False)

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
        is_mean, is_std = self.is_handler.compute()
        fid_score = self.fid_handler.compute()
        kid_mean, kid_std = self.kid_handler.compute()

        # 计算 Precision 和 Recall
        real_features = torch.concatenate(self.is_handler_real.features, dim=0).detach().cpu().float().numpy()
        fake_features = torch.concatenate(self.is_handler.features, dim=0).detach().cpu().float().numpy()
        metrics = compute_prdc(
            real_features=real_features,
            fake_features=fake_features,
            nearest_k=5
        )

        result = {
            'Inception Score': {
                'mean': float(is_mean),
                'std': float(is_std),
            },
            'FID Score': float(fid_score),
            'KID Score': {
                'mean': float(kid_mean),
                'std': float(kid_std),
            },
            'Precision': float(metrics['precision']),
            'Recall': float(metrics['recall']),
        }

        # 保存最终文件
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(result, f, indent=2)
