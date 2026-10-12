import math
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from occinteract.tools.pyvista_vis import render_semantic_scene
from occ_vae.loss.iou import logist_to_label


class ReconstructionRenderer:
    """
    一个独立的渲染器，用于将 3D 语义体素的真值和预测结果可视化。

    它接收 PyTorch 张量形式的真值标签和模型输出的 logits，
    使用 `render_semantic_scene` 函数进行 3D 渲染，并将结果拼接成一个网格图。
    """

    def __init__(self, elev: int = 30, azim: int = 30, max_size: int = 1080):
        """
        初始化渲染器。

        Args:
            elev (int): 渲染视角的仰角。
            azim (int): 渲染视角的方位角。
            max_size (int): 输出网格图的最大尺寸（宽度或高度），用于缩放以避免图像过大。
        """
        self.render_kwargs = {'elev': elev, 'azim': azim, 'show': False}
        self.max_size = max_size

    def render_grid(self, labels: torch.Tensor, logits: torch.Tensor) -> Optional[np.ndarray]:
        """
        根据给定的标签和 logits 生成可视化网格图。

        Args:
            labels (torch.Tensor): 真值标签张量，形状为 (B, H, W, D)。
            logits (torch.Tensor): 模型输出的 logits 张量，形状为 (B, C, H, W, D)。

        Returns:
            np.ndarray | None: 一个表示拼接后图像的 NumPy 数组 (H, W, C)，
                              如果无法生成图像则返回 None。
        """
        num_samples = labels.shape[0]
        if num_samples == 0:
            return None

        all_images = []
        for i in range(num_samples):
            gt_label = labels[i].numpy()  # Shape: (H, W, D)
            pred_logits = logits[i]  # Shape: (C, H, W, D)

            # 将模型的 logits 输出转换为预测标签
            pred_label = logist_to_label(pred_logits.unsqueeze(0)).squeeze(0).numpy()

            # 分别渲染 GT 和 Prediction，确保使用相同的视角
            gt_img = render_semantic_scene(grid_data=gt_label, **self.render_kwargs)
            pred_img = render_semantic_scene(grid_data=pred_label, **self.render_kwargs)

            # 确保图像是 uint8 类型
            all_images.extend([gt_img.astype(np.uint8), pred_img.astype(np.uint8)])

        if not all_images:
            return None

        # 拼接所有图像成一个网格
        grid_image = self._cat_images(num_samples, all_images)

        # 对图像重新采样，防止图像过大
        if max(grid_image.shape) > self.max_size:
            h, w, c = grid_image.shape
            scale_factor = self.max_size / max(h, w)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)

            pil_img = Image.fromarray(grid_image)
            resized_pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            grid_image = np.array(resized_pil_img)

        return grid_image

    @staticmethod
    def _cat_images(num_samples: int, all_images: list[np.ndarray], gap: int = 5) -> np.ndarray:
        """将一系列图像拼接成一个网格图。"""
        # 1. 计算网格布局
        num_pairs = num_samples
        n_rows = max(1, int(math.sqrt(num_pairs)))
        n_cols_pairs = math.ceil(num_pairs / n_rows)
        n_cols = n_cols_pairs * 2

        # 2. 创建一个大的画布
        img_h, img_w, img_c = all_images[0].shape
        grid_h = img_h * n_rows + (n_rows - 1) * gap
        grid_w = img_w * n_cols + (n_cols - 1) * gap
        grid_image = np.full((grid_h, grid_w, img_c), 0, dtype=np.uint8)

        # 3. 将所有小图拼接到大图上
        for idx, img in enumerate(all_images):
            row_idx = idx // n_cols
            col_idx = idx % n_cols
            row_start = row_idx * (img_h + gap)
            col_start = col_idx * (img_w + gap)
            grid_image[row_start: row_start + img_h, col_start: col_start + img_w, :] = img

        return grid_image


class ReconstructionVisualizer(pl.Callback):
    """
    一个 PyTorch Lightning Callback，用于在验证周期结束后，
    将模型的 3D 重构结果与真值进行可视化，并记录到 TensorBoard。

    它使用 `ReconstructionRenderer` 类来执行实际的渲染工作。
    """

    def __init__(
            self,
            every_n_epochs: int = 10,
            num_samples: int = 4,
            elev: int = 30,
            azim: int = 180,
            max_size: int = 4000,
    ):
        """
        初始化回调。

        Args:
            every_n_epochs (int): 每隔多少个 epoch 执行一次可视化。
            num_samples (int): 要从验证批次中可视化多少个样本。
            elev (int): 渲染视角的仰角 (传递给渲染器)。
            azim (int): 渲染视角的方位角 (传递给渲染器)。
            max_size (int): 输出网格图的最大尺寸 (传递给渲染器)。
        """
        super().__init__()
        assert num_samples >= 0, "num_samples must be non-negative."
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self._validation_batch = None

        # 实例化独立的渲染器
        self.renderer = ReconstructionRenderer(elev=elev, azim=azim, max_size=max_size)

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: torch.Tensor,
            batch: dict,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        """在每个验证批次结束后，保存所需数量的样本数据以供后续可视化。"""
        if self.num_samples == 0 or (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        if trainer.is_global_zero:
            # 初始化或补充  _validation_batch
            if self._validation_batch is None:
                self._validation_batch = {k: v.cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                # 仅保留需要的键以节省内存
                self._validation_batch = {k: self._validation_batch[k] for k in ['occ_data', 'occ_label'] if
                                          k in self._validation_batch}

            current_samples = next(iter(self._validation_batch.values())).shape[0]
            if current_samples < self.num_samples:
                samples_to_add = 1  # 从不同 batch 中采样
                for key in self._validation_batch.keys():
                    if key in batch and isinstance(batch[key], torch.Tensor):
                        self._validation_batch[key] = torch.cat(
                            (self._validation_batch[key], batch[key][:samples_to_add].cpu()), dim=0
                        )

    @rank_zero_only
    def on_validation_epoch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """在验证周期结束后，执行3D可视化并将结果记录到 TensorBoard。"""
        if trainer.sanity_checking or self._validation_batch is None:
            return

        if not isinstance(trainer.logger, TensorBoardLogger):
            print("Warning: ReconstructionVisualizer Callback requires a TensorBoardLogger.")
            self._validation_batch = None
            return

        num_samples_to_log = min(self.num_samples, next(iter(self._validation_batch.values())).shape[0])
        if num_samples_to_log == 0:
            self._validation_batch = None
            return

        # 使用保存的批次进行推理，获取最新模型输出
        with torch.no_grad():
            pl_module.eval()
            device_batch = {k: v.to(pl_module.device) for k, v in self._validation_batch.items()}
            output_dict = pl_module.single_step(device_batch)
            logits = output_dict['logits'][:num_samples_to_log].cpu()
            labels = output_dict['labels'][:num_samples_to_log].cpu()
            pl_module.train()

        # 调用渲染器生成图像
        grid_image = self.renderer.render_grid(labels=labels, logits=logits)

        # 记录到 TensorBoard
        if grid_image is not None:
            writer = trainer.logger.experiment
            writer.add_image(
                'val_reconstruction_grid',
                grid_image,
                global_step=trainer.current_epoch,
                dataformats='HWC'
            )

        # 清理保存的批次数据
        self._validation_batch = None
