import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# This repo originally used `flash_attn` for attention.
# Some environments cannot install flash_attn, so we implement a compatible
# PyTorch fallback using scaled_dot_product_attention.


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        proj_drop = attn_drop if proj_drop == 0 else proj_drop

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: [B, N, C]
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # -> [B, H, N, D]
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)

        # PyTorch SDPA handles scaling internally.
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )

        x = rearrange(attn, 'b h n d -> b n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        proj_drop = attn_drop if proj_drop == 0 else proj_drop

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        # x: queries  [B, Nq, C]
        # y: context  [B, Nk, C]
        q = self.q(x)
        kv = self.kv(y)
        k, v = kv.chunk(2, dim=-1)

        # -> [B, H, N, D]
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )

        x = rearrange(attn, 'b h n d -> b n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
