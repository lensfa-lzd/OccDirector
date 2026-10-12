"""
Absorb from https://github.com/gusongen/DOME
/loss/recon_loss.py
"""
import torch
from torch import nn


class KLLoss(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def forward(z_mu, z_logvar, **kwargs):
        """
        DOME 在个函数的实现上好像有问题
        其输入的是 z_sigma 标准差，但是用了 z_logvar ($\log \sigma^2$, 方差的对数) 的计算公式
        """
        # z_mu, z_logvar 张量形状是 [bs, Dim, H, W]

        # 1. 逐元素计算 KL 散度的项
        # formula: -0.5 * (1 + logvar - mu^2 - exp(logvar))
        kld_element = 1 + z_logvar - z_mu ** 2 - z_logvar.exp()

        # 2. 对每个样本的所有潜在维度求和
        kld_per_sample = -0.5 * torch.sum(kld_element, dim=[1, 2, 3])

        # 3. 对整个批次取平均值
        kld_loss = torch.mean(kld_per_sample)
        return kld_loss
