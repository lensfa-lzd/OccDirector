"""
Absorb from https://github.com/gusongen/DOME
/loss/recon_loss.py
"""
import sys
import torch.distributed as dist
import torch
import torch.nn.functional as F
from torch import nn


class ReconLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def forward(logits, labels, **kwargs):
        # logits 张量形状是 (bs, num_cls, H, W, D)
        # labels 张量形状是 (bs, H, W, D)

        # 或者
        # logits 张量形状是 (bs, num_cls, F, H, W, D)
        # labels 张量形状是 (bs, F, H, W, D)
        # PyTorch 会自动忽略值为 -100 的位置
        # check_tensors(logits, labels, ignore=-100, num_classes=11)
        rec_loss = F.cross_entropy(logits, labels.long(), ignore_index=-100)
        return rec_loss
