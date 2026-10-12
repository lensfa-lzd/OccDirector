import os
import random

import pytorch_lightning as pl
import torch
from bitsandbytes.optim import AdamW8bit
from einops import repeat, rearrange
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from occinteract.tools.occ_resizer import occ_resize
from occ_vae.loss.iou import ComputeIoU, ComputeMeanIoU, logist_to_label


class Trainer(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            loss_cfg: list,
            resize_cfg: dict = None,
            use_4d_occ=False,
            warmup_steps=1000,
            *args,
            **kwargs
    ):
        super().__init__()
        """
        loss_cfg = [{
            'name', 'weight', 'function'
        }]
        """
        self.save_hyperparameters(ignore=['model'])  # 保存超參數
        self.model = model
        self.loss_cfg = loss_cfg
        self.resize_cfg = resize_cfg
        self.use_4d_occ = use_4d_occ
        self.warmup_steps = warmup_steps

    def forward(self, occ):
        # occ 的形狀: [bs, H, W, D]
        if self.use_4d_occ:
            occ = repeat(occ, 'b h w d -> b frame h w d', frame=1)

        output_dict = self.model(occ)
        z_mu = output_dict['z_mu']
        z_logvar = output_dict['z_logvar']
        logits = output_dict['logits']

        if self.use_4d_occ:
            z_mu = rearrange(z_mu, 'b frame dim h w -> (b frame) dim h w')
            z_logvar = rearrange(z_logvar, 'b frame dim h w -> (b frame) dim h w')
            logits = rearrange(logits, 'bs num_cls F H W D -> (bs F) num_cls H W D')

        # z_mu, z_logvar shape -> [bs, Dim, H, W]
        # logits 最终返回的张量形状是 (bs, H, W, D, num_cls)
        return {
            'z_mu': z_mu,
            'z_logvar': z_logvar,
            'logits': logits,
        }

    def log_loss(self, output_dict, stage: str):
        total_loss = 0.0
        for cfg in self.loss_cfg:
            loss = cfg['function'](**output_dict)
            self.log(f"{stage}_{cfg['name']}", loss, prog_bar=False)

            loss = loss * cfg['weight']
            if cfg['name'] == 'KLLoss':
                loss = loss * getattr(self, 'current_beta', 1)  # kl 退火

            total_loss += loss
        self.log(f"{stage}_loss", total_loss, prog_bar=True)
        return total_loss

    def log_iou(self, output_dict, stage: str):
        logits = output_dict['logits']
        real_label = output_dict['labels']

        pred_label = logist_to_label(logits)
        IoU = ComputeIoU(real_label, pred_label)
        mIoU = ComputeMeanIoU(real_label, pred_label)
        self.log(f"{stage}_IoU", IoU * 100, prog_bar=False)  # 计算百分比
        self.log(f"{stage}_mIoU", mIoU * 100, prog_bar=False)
        return mIoU, IoU

    def single_step(self, batch):
        occ = batch['occ_data']
        label = batch['occ_label']
        if self.resize_cfg:
            occ = occ_resize(occ, (self.resize_cfg['after'], self.resize_cfg['after']))
            label = occ_resize(label, (self.resize_cfg['after'], self.resize_cfg['after']))

        output_dict = self.forward(occ)
        output_dict.update({'labels': label})
        return output_dict

    def training_step(self, batch, batch_idx):
        output_dict = self.single_step(batch)
        loss = self.log_loss(output_dict, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        output_dict = self.single_step(batch)
        loss = self.log_loss(output_dict, 'val')
        mIoU, IoU = self.log_iou(output_dict, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        output_dict = self.single_step(batch)
        loss = self.log_loss(output_dict, 'test')
        mIoU, IoU = self.log_iou(output_dict, 'test')
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # 仅用于编码
        occ = batch['occ_data']
        z, mean, logvar = self.model.encode(occ)
        return z

    def configure_optimizers(self):
        optimizer = AdamW8bit(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.learning_rate
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


class TrainerSeq(Trainer):
    def __init__(
            self,
            model: nn.Module,
            loss_cfg: list,
            split_frame=True,
            *args,
            **kwargs,
    ):
        super().__init__(
            model=model,
            loss_cfg=loss_cfg,
            *args,
            **kwargs,
        )
        self.split_frame = split_frame

    def resize_4d_occ(self, occ):
        b, frame = occ.shape[:2]
        occ = rearrange(occ, 'b frame h w d -> (b frame) h w d')
        occ = occ_resize(occ, (self.resize_cfg['after'], self.resize_cfg['after']))
        occ = rearrange(occ, '(b frame) h w d -> b frame h w d', b=b)
        return occ

    def single_step(self, batch):
        # 这里应该输入 [b frame h w d] 的数据
        occ = batch['occ_data']
        label = occ
        if isinstance(occ, list):
            occ, label, occ_len = self.process_list_input(occ)

        if self.resize_cfg:
            occ = self.resize_4d_occ(occ)
            label = self.resize_4d_occ(label)

        # frame 应该等于 8
        # 对 frame 进行随机截取，实现不同长度的序列/单场景的混合训练
        if self.split_frame:
            if os.getenv('SPLIT_FRAMES_SETTING') == '1':
                occ = occ[:, :1]
                label = label[:, :1]
            else:
                split_frame = random.choice((1, 2, 4, 8))
                occ = occ[:, :split_frame]
                label = label[:, :split_frame]

        occ = occ.contiguous()
        label = label.contiguous()
        output_dict = self.model(occ)
        output_dict.update({'labels': label})
        return output_dict

    @staticmethod
    def pad_occ_seq_with_last_frame(occ_seq, max_frame):
        if occ_seq.shape[0] == max_frame:
            return occ_seq

        last_frame = occ_seq[-1]
        num_pad_frame = max_frame - occ_seq.shape[0]
        pad_frames = torch.stack([last_frame] * num_pad_frame, dim=0)
        pad_seq = torch.cat([occ_seq, pad_frames], dim=0)
        return pad_seq

    def process_list_input(self, occ_list):
        result = []

        occ_len = []
        for occ in occ_list:
            occ_len.append(occ.shape[0])

        max_len = max(occ_len)
        for i, occ in enumerate(occ_list):
            result.append(
                self.pad_occ_seq_with_last_frame(occ, max_len)
            )

        occ_label = torch.stack(result, dim=0).long()
        labels_masked = self.mask_occ_label(occ_label, occ_len)
        return occ_label, labels_masked, occ_len

    @staticmethod
    def mask_occ_label(occ_label, occ_len):
        bs, total_frame, _, _, _ = occ_label.shape
        valid_lengths = torch.tensor(occ_len, device=occ_label.device)

        # --- 步骤 1: 创建填充掩码 (Mask) ---
        # 生成序列索引 (0, 1, ..., Total_F-1)
        seq_range = torch.arange(total_frame, device=occ_label.device)
        # 利用广播机制生成 Mask: (bs, F)
        # 如果索引 >= valid_lengths，则为 True (表示是 Padding)
        pad_mask = seq_range.unsqueeze(0) >= valid_lengths.unsqueeze(1)

        # 将 mask 扩展到与 labels 相同的形状 (bs, F, H, W, D)
        # 以此确保不仅仅是帧维度，而是该帧下的所有像素都被 mask 覆盖
        pad_mask_expanded = pad_mask.view(bs, total_frame, 1, 1, 1).expand_as(occ_label)

        # --- 步骤 2: 修改标签 ---
        # 将 labels 中填充位置的值设为 -100 (PyTorch ignore_index 的默认值)
        # 注意：一定要 clone，避免修改原始数据
        labels_masked = occ_label.clone()
        labels_masked[pad_mask_expanded] = -100
        return labels_masked
