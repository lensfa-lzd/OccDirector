import random
from os.path import abspath, dirname

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from bitsandbytes.optim import PagedAdamW8bit
from torch.optim.lr_scheduler import LambdaLR

project_root = dirname(dirname(abspath(__file__)))


class RectifiedFlowTrainer(pl.LightningModule):
    """
    使用 PyTorch Lightning 训练 Flow Matching 模型的 Trainer。

    这个类封装了 Flow Matching 的训练循环、损失计算、优化器设置。
    它遵循了 Conditional Flow Matching (CFM) 框架，使用简单的线性路径
    连接噪声和数据。
    """

    def __init__(self, lr=1e-3, cfg_prob: float = 0.15, **kwargs):
        super().__init__()
        self.cfg_prob = cfg_prob
        self.warmup_steps = 1e3
        self.lr = lr
        self.trainable_params = None
        self.history_conditioning_config = {
            'prob': 0.5,
            'sub_config_by_frame': {
                2: {
                    'max': 1,
                    'min': 1
                },
                4: {
                    'max': 2,
                    'min': 1
                },
                8: {
                    'max': 4,
                    'min': 2
                },
                16: {
                    'max': 8,
                    'min': 4
                }
            }
        }

        self.save_hyperparameters()

        # 用于记录“每个 optimizer step（考虑梯度累加）”的 mean loss
        self._micro_losses: list[torch.Tensor] = []

    def configure_optimizers(self):
        trainable_params = self.trainable_params
        if trainable_params is None:
            trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = PagedAdamW8bit(
            trainable_params,
            lr=self.lr,
            betas=(0.9, 0.95),
        )

        warmup_steps = self.warmup_steps

        def lr_lambda(current_step: int):
            # 在 warmup_steps 之前，学习率呈线性增长
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def single_step(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def get_history_conditioning_length(self, occ_len):
        frame_len = 0
        for frame_threshold, config in self.history_conditioning_config['sub_config_by_frame'].items():
            if occ_len > frame_threshold:
                frame_len = random.randint(config['min'], config['max'])
        return frame_len

    def training_step(self, batch, batch_idx):
        """
        执行单步训练，并以一定概率进行无条件训练以支持 CFG。

        Flow Matching 训练的核心步骤：
        1. 从数据集中获取目标图像 (x_1) 和 prompt。
        2. (CFG 修改) 以 self.cfg_prob 的概率，将 batch 中每个样本的 prompt 替换为空字符串 ""。
        3. 从先验分布（如标准正态分布）中采样得到源图像 (x_0)。
        4. 为 batch 中的每个图像随机采样一个时间步 t ~ U[0, 1]。
        5. 根据路径公式计算中间点 x_t。
        6. 根据路径公式计算目标向量场 u_t。
        7. 将 x_t, t, 和 (可能为空的) prompt embedding 输入 U-Net 模型，预测向量场 v(x_t, t)。
        8. 计算模型预测的向量场 v 和目标向量场 u_t 之间的损失。
        9. 返回损失。
        """
        target_occupancy = batch['occ_label']
        prompts = batch['scene_caption']

        prompts = list(prompts)
        # 以 self.cfg_prob 的概率随机将 prompt 设置为空字符串
        for i in range(len(prompts)):
            if random.random() < self.cfg_prob:
                prompts[i] = ""

        # 随机历史帧控制
        history_num_frames = []
        for i in range(len(target_occupancy)):
            his_frame_len = 0
            if random.random() < self.history_conditioning_config['prob']:
                occ_len = target_occupancy[i].shape[0]
                his_frame_len = self.get_history_conditioning_length(occ_len)

            history_num_frames.append(his_frame_len)

        # 使用可能被修改过的 prompts 创建新的 batch
        processed_batch = {
            'target_occupancy': target_occupancy,
            'prompts': prompts,
            'history_num_frames': history_num_frames,
        }

        loss = self.single_step(processed_batch)

        # A) epoch 聚合（平滑趋势）
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_loss_epoch", loss, prog_bar=False, on_step=False, on_epoch=True)

        # B) optimizer step 聚合（严格对应梯度累加后的更新）
        self._micro_losses.append(loss.detach())
        acc = int(getattr(self.trainer, "accumulate_grad_batches", 1) or 1)
        is_acc_boundary = ((batch_idx + 1) % acc == 0)
        if is_acc_boundary:
            step_loss = torch.stack(self._micro_losses, dim=0).mean()
            self.log("train_loss_accumulate_grad", step_loss, prog_bar=True)
            self._micro_losses.clear()

        return loss

    @staticmethod
    def loss_function(pred_vector_field, target_vector_field):
        # L2 正则取平均
        loss = F.mse_loss(pred_vector_field, target_vector_field)
        return loss
