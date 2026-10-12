import math
import os
from typing import List, Tuple

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from torch import nn
from timm.models.vision_transformer import Mlp, PatchEmbed

from model.dynamic_city_dit.utils.attention_utils import Attention, CrossAttention
from model.dynamic_city_dit.utils.dit_utils import get_2d_sincos_pos_embed, modulate


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, x_attn=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.use_x_attn = x_attn
        if self.use_x_attn:
            self.normx = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.x_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, y):
        # x = block(x, c, hexplane + layout)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.use_x_attn:
            x = x + self.x_attn(self.normx(x), y)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            # config args
            depth=16,
            hidden_size=384,
            patch_size=2,
            num_heads=8,

            # main args
            in_channels=16,
            x_attn=False,

            # other args
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            learn_sigma=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.learn_sigma = learn_sigma
        self.out_channels = in_channels * 2 if learn_sigma else in_channels

        self.occ_in = nn.Linear(256, hidden_size)
        self.text_in = nn.Linear(int(os.getenv('ENCODER_TEXT_DIM')), hidden_size)

        # self.x_embedder = PatchEmbed(im_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, 196 * 32, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio, x_attn) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_config({
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "_diffusers_version": "0.34.0.dev0",

            # 单帧长度
            "base_image_seq_len": 196,
            "base_shift": 0.5,

            "invert_sigmas": False,

            # 最长的帧数
            "max_image_seq_len": 196 * 32,
            # 增加 max_shift 以应对巨大的序列长度
            "max_shift": 1.15,  # 或者尝试更高，如 2.0

            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": 0.02,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False
        })

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int((196 * 40) ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed[:196 * 32]).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpack_latents(self, x, frames):
        height = width = 28
        x = rearrange(
            x,
            "b (f h w) (c p1 p2) -> b c f (h p1) (w p2)",
            f=frames,
            h=height // self.patch_size,
            w=width // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
        )

        return x

    def pack_latents(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.x_embedder(x) + self.pos_embed  # B, N, D; N = X * Y / patch_size ** 2
        x = rearrange(
            x,
            "b c f (h p1) (w p2) -> b (f h w) (c p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return x

    def forward(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states,
            *args,
            **kwargs,
    ):
        x = hidden_states
        t = timestep
        text_token = encoder_hidden_states

        x = self.occ_in(x)
        _, seq_len, _ = x.shape
        x = x + self.pos_embed[:, :seq_len, :]
        t = self.t_embedder(t)  # B, N

        c = t.unsqueeze(1) + self.text_in(text_token.to(x))  # B, N, D
        for block in self.blocks:
            x = block(x, c, None)  # B, N, D
        x = self.final_layer(x, c)  # B, N, patch_size ** 2 * out_channels

        # x = self.unpatchify(x)  # B, out_channels, X, Y

        return x

    def forward_with_cfg(self, x, t, cfg_scale, *args, **kwargs):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, *args, **kwargs)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @staticmethod
    def pad_occ_seq_with_last_frame(occ_seq: torch.Tensor, max_frame: int) -> torch.Tensor:
        """使用序列最后一帧对 occupancy 序列进行 padding。

        Args:
            occ_seq (torch.Tensor): occupancy 序列，形状为 `[F, ...]`。
            max_frame (int): 目标帧数。

        Returns:
            torch.Tensor: padding 后的序列，形状为 `[max_frame, ...]`。
        """

        if occ_seq.shape[0] == max_frame:
            return occ_seq

        last_frame = occ_seq[-1]
        num_pad_frame = max_frame - occ_seq.shape[0]
        pad_frames = torch.stack([last_frame] * num_pad_frame, dim=0)
        pad_seq = torch.cat([occ_seq, pad_frames], dim=0)
        return pad_seq

    def pad_occ_seq_list(self, occ_list: List[torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
        """将不定长 occupancy 序列列表 padding 并 stack。

        Args:
            occ_list (List[torch.Tensor]): 长度为 `B` 的序列列表。
                每个元素形状为 `[F_i, ...]`。

        Returns:
            Tuple[torch.Tensor, List[int]]:
                - padded: `[B, F_max, ...]`
                - occ_len: 原始长度列表 `[F_1, ..., F_B]`
        """

        result = []

        occ_len = []
        for occ in occ_list:
            occ_len.append(occ.shape[0])

        max_len = max(occ_len)
        for i, occ in enumerate(occ_list):
            result.append(
                self.pad_occ_seq_with_last_frame(occ, max_len)
            )

        occ_label = torch.stack(result, dim=0)
        return occ_label, occ_len


def get_dit() -> DiT:
    model = DiT()
    return model
