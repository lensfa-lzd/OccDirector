import logging
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from occinteract.tools.pyvista_vis import render_semantic_scene
from occ_vae.loss.iou import logist_to_label
from occ_vae.trainer.callback.visualizer import ReconstructionVisualizer


class ReconstructionRendererSeq:
    """
    一个独立的渲染器，用于将 3D 语义体素序列的真值和预测结果可视化。

    它可以将结果渲染为两种格式：
    1. 'image': 将序列帧拼接成一个大的网格图。
       每个样本的排版为：[重建帧1, ..., 重建帧N, <大间隙>, 真值帧1, ..., 真值帧N]
    2. 'video': 将预测序列和真值序列并排拼接成一个视频。
       视频的每一帧左边是预测，右边是真值。
    """

    def __init__(
            self,
            elev: int = 30,
            azim: int = 180,
            max_size: int = 1080,
            pred_gt_gap: int = 20
    ):
        """
        初始化渲染器。

        Args:
            elev (int): 渲染视角的仰角。
            azim (int): 渲染视角的方位角。
            max_size (int): 'image'模式下输出网格图的最大尺寸；'video'模式下为视频帧的最大宽度。
            pred_gt_gap (int): 在预测和真值之间的水平间距宽度。
        """
        self.render_kwargs = {'elev': elev, 'azim': azim, 'show': False}
        self.max_size = max_size
        self.pred_gt_gap = pred_gt_gap

    def render_image(
            self,
            labels: torch.Tensor,
            logits: torch.Tensor,
            gap: int = 5,
    ):
        num_samples, num_frames = labels.shape[:2]
        if num_samples == 0:
            return None

        all_row_images = []
        for i in range(num_samples):  # 遍历每个样本
            pred_images, gt_images = [], []

            # 1. 渲染预测序列
            for j in range(num_frames):
                pred_logits_frame = logits[i, :, j]
                pred_label_frame = logist_to_label(pred_logits_frame.unsqueeze(0)).squeeze(0).numpy()
                pred_img = render_semantic_scene(grid_data=pred_label_frame, **self.render_kwargs)
                pred_images.append(pred_img.astype(np.uint8))

            # 2. 渲染真值序列
            for j in range(num_frames):
                gt_label_frame = labels[i, j].numpy()
                gt_img = render_semantic_scene(grid_data=gt_label_frame, **self.render_kwargs)
                gt_images.append(gt_img.astype(np.uint8))

            if not pred_images or not gt_images:
                continue

            # 3. 将预测序列和真值序列分别水平拼接
            pred_with_gaps = [np.pad(img, ((0, 0), (0, gap), (0, 0)), 'constant') for img in pred_images[:-1]] + [
                pred_images[-1]]
            pred_row = np.concatenate(pred_with_gaps, axis=1)

            gt_with_gaps = [np.pad(img, ((0, 0), (0, gap), (0, 0)), 'constant') for img in gt_images[:-1]] + [
                gt_images[-1]]
            gt_row = np.concatenate(gt_with_gaps, axis=1)

            # 4. 创建间隔并将预测行、间隔、真值行拼接
            h, _, c = pred_row.shape
            spacer = np.zeros((h, self.pred_gt_gap, c), dtype=np.uint8)
            sample_row_image = np.concatenate([pred_row, spacer, gt_row], axis=1)

            all_row_images.append(sample_row_image)

        if not all_row_images:
            return None

        # 5. 将所有样本行垂直拼接
        grid_with_gaps = [np.pad(row_img, ((0, gap), (0, 0), (0, 0)), 'constant') for row_img in
                          all_row_images[:-1]] + [all_row_images[-1]]
        grid_image = np.concatenate(grid_with_gaps, axis=0)

        # 6. 重新采样以防图像过大
        if max(grid_image.shape) > self.max_size:
            h, w, _ = grid_image.shape
            scale_factor = self.max_size / max(h, w)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)

            pil_img = Image.fromarray(grid_image)
            resized_pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            grid_image = np.array(resized_pil_img)

        return grid_image

    def render_video(
            self,
            labels: torch.Tensor,
            logits: torch.Tensor,
            gap: int = 5,
    ):
        num_samples, num_frames = labels.shape[:2]
        if num_samples == 0:
            return None

        all_video_frames = []

        # 循环遍历时间帧，为每个时间点创建一个组合帧
        for j in range(num_frames):
            rows_for_this_timestep = []

            # 在当前时间点，为每个样本创建一行 [pred | gt]
            for i in range(num_samples):
                # 1. 渲染预测
                pred_logits_frame = logits[i, :, j]
                pred_label_frame = logist_to_label(pred_logits_frame.unsqueeze(0)).squeeze(0).numpy()
                pred_img = render_semantic_scene(grid_data=pred_label_frame, **self.render_kwargs).astype(np.uint8)

                # 2. 渲染真值
                gt_label_frame = labels[i, j].numpy()
                gt_img = render_semantic_scene(grid_data=gt_label_frame, **self.render_kwargs).astype(np.uint8)

                # 确保高度一致
                h_pred, _, c_pred = pred_img.shape
                h_gt, _, _ = gt_img.shape
                target_h = max(h_pred, h_gt)
                if h_pred < target_h:
                    pred_img = np.pad(pred_img, ((0, target_h - h_pred), (0, 0), (0, 0)), 'constant')
                if h_gt < target_h:
                    gt_img = np.pad(gt_img, ((0, target_h - h_gt), (0, 0), (0, 0)), 'constant')

                # 3. 水平拼接成一行
                spacer = np.zeros((target_h, self.pred_gt_gap, c_pred), dtype=np.uint8)
                sample_row = np.concatenate([pred_img, spacer, gt_img], axis=1)
                rows_for_this_timestep.append(sample_row)

            if not rows_for_this_timestep:
                continue

            # 4. 将当前时间点的所有行垂直拼接起来，形成一帧完整的视频帧
            rows_with_gaps = [np.pad(row, ((0, gap), (0, 0), (0, 0)), 'constant') for row in
                              rows_for_this_timestep[:-1]] + [rows_for_this_timestep[-1]]
            composite_frame = np.concatenate(rows_with_gaps, axis=0)

            # 5. （缩放该帧以避免视频过大
            if max(composite_frame.shape) > self.max_size:
                h, w, _ = composite_frame.shape
                scale_factor = self.max_size / max(h, w)
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                pil_img = Image.fromarray(composite_frame)
                resized_pil = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                composite_frame = np.array(resized_pil)

            all_video_frames.append(composite_frame)

        if not all_video_frames:
            return None

        return np.stack(all_video_frames, axis=0)

    def render_grid(
            self,
            labels: torch.Tensor,
            logits: torch.Tensor,
            gap: int = 5,
            output_format: str = 'image',
    ) -> Optional[np.ndarray]:
        """
        根据给定的序列标签和 logits 生成可视化网格图或视频。

        Args:
            labels (torch.Tensor): 真值标签张量，形状为 (B, F, H, W, D)。
            logits (torch.Tensor): 模型输出的 logits 张量，形状为 (B, num_cls, F, H, W, D)。
            gap (int): 图像模式下，序列内图像之间的间距。
            output_format (str): 输出格式，'image' 或 'video'。

        Returns:
            - 如果 output_format='image': 一个表示拼接后图像的 NumPy 数组 (H, W, C)。
            - 如果 output_format='video': 一个列表，每个元素是代表一个样本视频的字节串 (bytes)。
            - 如果无法生成，则返回 None。
        """
        if output_format == 'video':
            return self.render_video(
                labels=labels,
                logits=logits,
                gap=gap,
            )

        elif output_format == 'image':
            return self.render_image(
                labels=labels,
                logits=logits,
                gap=gap,
            )

        else:
            raise NotImplementedError("output_format 必须是 'image' 或 'video'")


class ReconstructionVisualizerSeq(ReconstructionVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 覆写父类中的渲染器为我们新的序列渲染器
        # 我们从父类已创建的渲染器中获取参数，以保持一致性
        self.renderer = ReconstructionRendererSeq(
            elev=self.renderer.render_kwargs['elev'],
            azim=self.renderer.render_kwargs['azim'],
            max_size=self.renderer.max_size
        )

    @rank_zero_only
    def on_validation_epoch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """
        在验证周期结束后，为不同长度的序列创建可视化结果，并记录到 TensorBoard。
        """
        if trainer.sanity_checking or self._validation_batch is None:
            return

        if not isinstance(trainer.logger, TensorBoardLogger):
            print("Warning: ReconstructionVisualizerSeq Callback requires a TensorBoardLogger.")
            self._validation_batch = None
            return

        num_samples_to_log = min(self.num_samples, next(iter(self._validation_batch.values())).shape[0])
        if num_samples_to_log == 0:
            self._validation_batch = None
            return

        pl_module.eval()
        writer = trainer.logger.experiment

        # 定义要可视化的不同序列长度
        # split_frames_to_vis = (1, 2, 4, 8)
        split_frames_to_vis = (1,)

        pl_module.split_frame = False
        # 循环遍历每个定义的序列长度
        for split_frame in split_frames_to_vis:
            # 从保存的完整序列批次中，截取当前循环所需的长度
            # 注意：我们假设_validation_batch中的数据有足够的帧数（例如8帧）
            batch_for_split = {
                k: v[:num_samples_to_log, :split_frame, ...].clone()
                for k, v in self._validation_batch.items()
                if isinstance(v, torch.Tensor) and v.ndim > 1
            }

            # 如果截取后序列长度为0，则跳过（例如，原始数据不足8帧）
            if batch_for_split['occ_data'].shape[1] == 0:
                continue

            # 使用截取后的批次进行推理
            # 假设 pl_module.single_step 在 eval 模式下不会进行随机截取，而是处理传入的全部帧
            with torch.no_grad():
                device_batch = {k: v.to(pl_module.device) for k, v in batch_for_split.items()}
                output_dict = pl_module.single_step(device_batch)

                # single_step的输出应该与输入有相同的序列长度
                logits = output_dict['logits'].cpu()
                labels = output_dict['labels'].cpu()

            # 调用新的序列渲染器生成图像
            grid_image = self.renderer.render_grid(labels=labels, logits=logits)

            # 记录到 TensorBoard，每个长度的序列使用一个独立的tag
            if grid_image is not None:
                tag = f'val_reconstruction_grid_{split_frame}'
                writer.add_image(
                    tag,
                    grid_image,
                    global_step=trainer.current_epoch,
                    dataformats='HWC'
                )

        # 恢复模型状态并清理保存的批次数据
        pl_module.split_frame = True
        pl_module.train()
        self._validation_batch = None


class ReconstructionVisualizerSeqByStep(pl.Callback):
    def __init__(
            self,
            test_samples: int,
            every_n_steps: int,
            dataset,
    ):
        self.test_sample_len = 1
        self.test_samples = test_samples
        self.every_n_steps = every_n_steps
        self.dataset = dataset

        self.renderer = ReconstructionRendererSeq(
            elev=30,
            azim=180,
            max_size=1080
        )

    def get_test_sample(self):
        result = []
        for i in range(0, len(self.dataset), 100):
            data_dict = self.dataset[i]
            occ = data_dict['occ_data']
            if occ.shape[0] < self.test_sample_len:
                continue

            result.append(torch.from_numpy(occ[:self.test_sample_len]))

            if len(result) >= self.test_samples:
                break

        # 返回 [b frame h w d]
        return torch.stack(result, dim=0)

    @rank_zero_only
    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
            outputs, batch, batch_idx
    ) -> None:
        """在训练批次结束后，执行可视化并将结果记录到 TensorBoard。"""
        if trainer.sanity_checking or batch_idx == 0 or batch_idx % self.every_n_steps != 0:
            return

        if not isinstance(trainer.logger, TensorBoardLogger):
            logging.warning("ReconstructionVisualizerSeqByStep Callback requires a TensorBoardLogger.")
            return

        pl_module.eval()
        writer = trainer.logger.experiment

        with torch.no_grad():
            occ = self.get_test_sample().to(pl_module.device)
            output_dict = pl_module.single_step({'occ_data': occ})

            # single_step的输出应该与输入有相同的序列长度
            logits = output_dict['logits'].cpu()
            labels = output_dict['labels'].cpu()

        # 调用新的序列渲染器生成图像
        grid_image = self.renderer.render_grid(labels=labels, logits=logits)

        # 记录到 TensorBoard，每个长度的序列使用一个独立的tag
        if grid_image is not None:
            tag = f'val_reconstruction_grid'
            writer.add_image(
                tag,
                grid_image,
                global_step=trainer.global_step,
                dataformats='HWC'
            )
        pl_module.train()


if __name__ == '__main__':
    handler = ReconstructionRendererSeq()
    # labels (torch.Tensor): 真值标签张量，形状为 (B, F, H, W, D)。
    # logits (torch.Tensor): 模型输出的 logits 张量，形状为 (B, num_cls, F, H, W, D)。

    labels = torch.ones((1, 8, 224, 224, 16))
    logits = torch.rand((1, 11, 8, 224, 224, 16))
    image = handler.render_grid(labels, logits, output_format='video')
    print()
