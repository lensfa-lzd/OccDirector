"""
Modified from https://github.com/gusongen/DOME/tree/da390aeeab1990fa06e037ad4d486729ae4fe712/model/VAE
"""
from typing import Union, Type

import torch
from einops import repeat, rearrange
from torch import nn


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class Normalize(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if channels <= 32:
            num_groups = channels // 4
        else:
            num_groups = 32
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.norm(x)


class ResnetBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            conv_module: Union[Type[nn.Conv2d], Type[nn.Conv3d]],
            dropout=0.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = conv_module(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv_module(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        if self.in_channels != self.out_channels:
            self.shortcut = conv_module(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(
            self,
            channels,
            conv_module: Union[Type[nn.Conv2d], Type[nn.Conv3d]],
    ):
        super().__init__()
        self.in_channels = channels

        self.norm = Normalize(channels)
        self.to_qkv = conv_module(
            channels,
            channels * 3,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.proj_out = conv_module(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        h = self.norm(x)
        qkv = self.to_qkv(h)

        # 仅有一个头
        if len(x.shape) == 4:
            qkv = repeat(qkv, "b c h w -> b 1 (h w) c").contiguous()
        elif len(x.shape) == 5:
            # 这里的注意力并不 catch 时间维度的 feature
            qkv = repeat(qkv, "b c f h w -> (b f) 1 (h w) c").contiguous()
        else:
            raise NotImplementedError

        q, k, v = torch.chunk(qkv, 3, dim=-1)
        h = nn.functional.scaled_dot_product_attention(q, k, v)

        if len(x.shape) == 4:
            _, _, height, weight = x.shape
            h = rearrange(
                h,
                " b 1 (h w) c -> b c h w",
                h=height,
                w=weight
            )
        elif len(x.shape) == 5:
            _, _, frame, height, weight = x.shape
            h = rearrange(
                h,
                "(b f) 1 (h w) c -> b c f h w",
                f=frame,
                h=height,
                w=weight
            )
        else:
            raise NotImplementedError

        return x + self.proj_out(h)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 1)

    def forward(self, x):
        x_shape = x.shape
        if len(x_shape) == 5:
            x = rearrange(x, "b c f h w -> (b f) c h w")

        x = self.conv(x)

        if len(x_shape) == 5:
            b, _, f, _, _ = x_shape
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)

        return x


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x_shape = x.shape
        if len(x_shape) == 5:
            x = rearrange(x, "b c f h w -> (b f) c h w")

        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)

        if len(x_shape) == 5:
            b, _, f, _, _ = x_shape
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)

        return x
