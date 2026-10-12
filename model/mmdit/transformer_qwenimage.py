# Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified from diffusers==0.35.0
# diffusers/models/transformers/transformer_qwenimage.py
#
# ==============================================================================


import functools
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin, CacheMixin
from diffusers.configuration_utils import register_to_config
from diffusers.loaders import PeftAdapterMixin, FromOriginalModelMixin
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import RMSNorm, AdaLayerNormContinuous
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
from diffusers.utils import logging
from einops import rearrange, repeat

from model.mmdit.att import SingleStreamSelfAttnProcessor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class QwenTimestepProjEmbeddings(nn.Module):
    """timestep -> embedding 的投影模块。

    输入 `timestep` 与 `hidden_states`，输出与 batch 对齐的 timestep embedding：`[B, embedding_dim]`。
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """将 timestep 投影为与 batch 对齐的 embedding。

        Args:
            timestep (torch.Tensor): timestep，shape `[B]`。
            hidden_states (torch.Tensor): image stream token，用于提供 dtype，shape `[B, S_img, D]`。

        Returns:
            torch.Tensor: timestep embedding，shape `[B, embedding_dim]`。
        """
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))  # (N, D)

        conditioning = timesteps_emb

        return conditioning


class QwenEmbedRope(nn.Module):
    """Qwen-Image 的 RoPE positional embedding 生成器。

    `forward` 返回：
    - `vid_freqs`: `[S_vid, rope_dim]`，其中 `S_vid = sum_i(F_i*H_i*W_i)`
    - `txt_freqs`: `[T_max, rope_dim]`

    rope_dim 的精确定义取决于 `axes_dim` 的拆分。
    由于 QwenEmbedRope 仅继承自 nn.Module，其运算逻辑均定义在类内部，rope_dim 可由本类实现直接推导/验证（见 `rope_params/_compute_video_freqs`）。
    """

    def __init__(self, theta: int, axes_dim: List[int], scale_rope: bool = False) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.rope_cache = {}

        # DO NOT USING REGISTER BUFFER HERE, IT WILL CAUSE COMPLEX NUMBERS LOSE ITS IMAGINARY PART
        self.scale_rope = scale_rope

    def rope_params(self, index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        """生成某一轴的 RoPE 复数频率表。

        Args:
            index (torch.Tensor): 位置索引，1D 张量，形状 `[S]`（例如 `torch.arange(S)`）。
            dim (int): 该轴分配到的 RoPE 维度（要求为偶数）。
            theta (int): RoPE 的 base。

        Returns:
            torch.Tensor: 复数频率表，形状 `[S, dim/2]`（complex dtype）。
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(
            self,
            video_fhw: Union[Tuple[int, int, int], List[Tuple[int, int, int]]],
            txt_seq_lens: List[int],
            device: Union[str, torch.device],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成 video/text 的 RoPE 频率。

        Args:
            video_fhw (Union[Tuple[int, int, int], List[Tuple[int, int, int]]]): 视频 token 的形状描述。
                - 若为 `Tuple[int, int, int]`：表示单个视频 `(F, H, W)`。
                - 若为 `List[Tuple[int, int, int]]`：表示 batch 内各样本的 `(F, H, W)`。

                TODO: 当前实现对 `List[Tuple[int, int, int]]` 会取 `video_fhw[0]`（仅使用 batch 内第一个样本的 shape）。
                      请确认这是“假设 batch 内 shape 一致”的设计，还是逻辑需要改为逐样本计算。
                      这里使用 video_fhw 为 List 是为了兼容 batch 的设置，List 中的每个元素为单个 video
                      直接使用 `video_fhw[0]` 是因为 batch 中 video 的形状已经对齐

                TODO: 总的来说，这是一个代码的历史遗留问题，有待后序修改
            txt_seq_lens (List[int]): batch 内每个样本的文本有效长度（token 数），长度为 `B`。
            device (Union[str, torch.device]): 目标设备。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - vid_freqs (torch.Tensor): 视频侧 RoPE 频率，形状 `[S_vid, rope_dim]`。
                - txt_freqs (torch.Tensor): 文本侧 RoPE 频率，形状 `[T_max, rope_dim]`，其中 `T_max = max(txt_seq_lens)`。
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{frame}_{height}_{width}"  # 为这个 key 添加 frame, 以适配视频数据

            if not torch.compiler.is_compiling():
                if rope_key not in self.rope_cache:
                    self.rope_cache[rope_key] = self._compute_video_freqs(frame, height, width, idx)
                video_freq = self.rope_cache[rope_key]
            else:
                video_freq = self._compute_video_freqs(frame, height, width, idx)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index: max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @functools.lru_cache(maxsize=None)
    def _compute_video_freqs(self, frame: int, height: int, width: int, idx: int = 0) -> torch.Tensor:
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx: idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2):], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2):], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class QwenImageTransformerBlockFiXMask(QwenImageTransformerBlock):
    """修复 attention mask 没有正确使用的问题的 block。

    该 block 主要提供：
    - `get_rope`: 生成 image/text 的 rotary embedding。
    - `get_attention_mask`: 生成 joint attention mask（text + image tokens）。
    - `preprocess/postprocess`: 对 img/text 两个 stream 做 AdaLayerNorm + 门控调制。

    形状约定（高频）：
    - `hidden_states`（image stream tokens）: `[B, S_img, dim]`
    - `encoder_hidden_states`（text tokens）: `[B, S_txt, dim]`
    - `img_shapes[i] = (F_i, H_grid, W_grid)`，`S_img` 通常为 `F_max * H_grid * W_grid`（按 batch padding）。

    `img_shapes` 的具体含义与 token 排布强相关；若更改 token 排布，需要同步确认 mask/rope 逻辑（已在其他注释中说明）。
    """

    # 修复 attention mask 没有正确使用的问题
    # https://github.com/huggingface/diffusers/issues/12294
    def __init__(
            self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm",
            eps: float = 1e-6
    ) -> None:
        super().__init__(
            dim,
            num_attention_heads,
            attention_head_dim,
            qk_norm,
            eps
        )

    @staticmethod
    def get_rope(
            img_shapes: List[Tuple[int, int, int]],  # [(F, H_grid, W_grid), ...]
            txt_seq_lens: List[int],
            device: Union[str, torch.device],
            pos_embed: "QwenEmbedRope",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成 image/text 的 rotary embedding。

        Args:
            img_shapes (List[Tuple[int, int, int]]): 每个样本的 `(F, H_grid, W_grid)`。
            txt_seq_lens (List[int]): 每个样本文本 token 有效长度。
            device (Union[str, torch.device]): 目标设备。
            pos_embed (QwenEmbedRope): RoPE 生成器。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - vid_freqs: `[S_vid, rope_dim]`
                - txt_freqs: `[T_max, rope_dim]`
        """
        max_len = 0
        for frame, height, width in img_shapes:
            if max_len < frame:
                max_len = frame

        pos_embed_shape = []
        for frame, height, width in img_shapes:
            pos_embed_shape.append((max_len, height, width))

        # 计算位置编码使用填充后的形状（最长的帧数）
        image_rotary_emb = pos_embed(pos_embed_shape, txt_seq_lens, device=device)

        return image_rotary_emb

    @staticmethod
    def get_attention_mask(
            img_shapes: List[Tuple[int, int, int]],  # [(F, H_grid, W_grid), ...]
            txt_seq_lens: List[int],
            device: Union[str, torch.device],
    ) -> torch.Tensor:
        """生成 joint attention mask（text + image）。

        Returns 的 mask 形状为 broadcastable 到 SDPA 的 4D mask：`[B, 1, 1, S_txt + S_img]`。

        Args:
            img_shapes (List[Tuple[int, int, int]]): `(F, H_grid, W_grid)` 列表。
            txt_seq_lens (List[int]): 文本有效长度列表。
            device (Union[str, torch.device]): 目标设备。

        Returns:
            torch.Tensor: bool mask。
        """
        # 这里计算两个模态统一的 mask
        img_attn_mask_list = [
            torch.ones(img_shape[0] * img_shape[1] * img_shape[2], device=device)
            for img_shape in img_shapes
        ]
        img_attention_mask = torch.nn.utils.rnn.pad_sequence(img_attn_mask_list, batch_first=True, padding_value=0)

        text_attn_mask_list = [
            torch.ones(txt_seq_len, device=device)
            for txt_seq_len in txt_seq_lens
        ]
        text_attention_mask = torch.nn.utils.rnn.pad_sequence(text_attn_mask_list, batch_first=True, padding_value=0)

        text_attention_mask = text_attention_mask.bool()
        img_attention_mask = img_attention_mask.bool()

        joint_attention_mask_1d = torch.cat([text_attention_mask, img_attention_mask], dim=1)
        # broadcastable shape for SDPA
        attention_mask = joint_attention_mask_1d[:, None, None, :]

        return attention_mask

    def _modulate(
            self,
            x: torch.Tensor,
            mod_params: torch.Tensor,
            mod_params_tr: Optional[torch.Tensor] = None,
            num_replace_tokens: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对输入做门控调制（Hunyuan token_replace 风格）。

        该函数等价于 HunyuanVideo-I2V 中的：
        - `hyvideo/modules/modulate_layers.py:modulate`
        - `hyvideo/modules/modulate_layers.py:apply_gate`

        核心语义：
        - `mod_params` 产生一套 (shift/scale/gate)，用于「非替换 token」区域。
        - 当提供 `mod_params_tr` + `num_replace_tokens` 时：
          - token 前缀 `[:num_replace_tokens[i]]` 使用 (tr_shift/tr_scale/tr_gate)
          - 其余 token 使用 (shift/scale/gate)
        - 返回的 `gate_map` 已经在 token 维做了广播/拼接，可直接与残差分支输出逐元素相乘。

        Args:
            x (torch.Tensor): 归一化后的 token，形状 `[B, S, D]`。
            mod_params (torch.Tensor): 调制参数，形状 `[B, 3*D]`，按最后一维切分为 `(shift, scale, gate)`。
            mod_params_tr (Optional[torch.Tensor]): token_replace 分支调制参数，形状 `[B, 3*D]`，切分为 `(tr_shift, tr_scale, tr_gate)`。
            num_replace_tokens (Optional[List[int]]): batch 内每个样本的替换前缀长度（token 数），长度为 `B`。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x_modulated: 调制后的 token，形状 `[B, S, D]`。
                - gate_map: 门控系数（已按 token 维广播/拼接），形状 `[B, S, D]`。
        """

        shift, scale, gate = mod_params.chunk(3, dim=-1)  # [B, D]

        # default branch: no token_replace
        if mod_params_tr is None or num_replace_tokens is None:
            x_mod = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            gate_map = gate.unsqueeze(1).expand_as(x)
            return x_mod, gate_map

        tr_shift, tr_scale, tr_gate = mod_params_tr.chunk(3, dim=-1)  # [B, D]

        # per-sample prefix lengths
        b, s, _d = x.shape
        x_mod = torch.empty_like(x)
        gate_map = torch.empty_like(x)
        for i, n in enumerate(num_replace_tokens):
            n = int(n)
            n = max(0, min(n, s))

            if n > 0:
                x_zero = x[i:i + 1, :n] * (1 + tr_scale[i:i + 1].unsqueeze(1)) + tr_shift[i:i + 1].unsqueeze(1)
                g_zero = tr_gate[i:i + 1].unsqueeze(1).expand(1, n, -1)
                x_mod[i:i + 1, :n] = x_zero
                gate_map[i:i + 1, :n] = g_zero

            if n < s:
                x_orig = x[i:i + 1, n:] * (1 + scale[i:i + 1].unsqueeze(1)) + shift[i:i + 1].unsqueeze(1)
                g_orig = gate[i:i + 1].unsqueeze(1).expand(1, s - n, -1)
                x_mod[i:i + 1, n:] = x_orig
                gate_map[i:i + 1, n:] = g_orig

        return x_mod, gate_map

    def preprocess(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            temb: torch.Tensor,
            temb_history: Optional[torch.Tensor] = None,
            num_history_tokens: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """对 img/text stream 做 norm1 + 调制（含 token_replace 支持）。

        说明：
        - 该函数会先用 `temb` 计算常规调制参数（img/text 各一套）。
        - 当提供 `temb_history` 与 `num_history_tokens` 时，会额外计算一套「history 分支」的 image 调制参数，
          并在 `_modulate(...)` 内部对 token 前缀 `[:num_history_tokens[i]]` 应用该分支（等价 Hunyuan 的 token_replace）。
        - 当前实现只对 image stream 做 token_replace 风格调制；text stream 保持单分支调制。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - img_modulated: image stream 经 norm1+调制后的 token，`[B, S_img, D]`
                - img_gate1: image stream attention residual 的 gate_map，`[B, S_img, D]`
                - img_mod2: image stream 的第二段调制参数（供 norm2/MLP 使用），`[B, 3*D]`
                - txt_modulated: text stream 经 norm1+调制后的 token，`[B, S_txt, D]`
                - txt_gate1: text stream attention residual 的 gate_map，`[B, S_txt, D]`
                - txt_mod2: text stream 的第二段调制参数（供 norm2/MLP 使用），`[B, 3*D]`
        """

        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]
        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        img_mod1_tr = None
        if temb_history is not None:
            img_mod_params_hist = self.img_mod(temb_history)  # [B, 6*dim]
            img_mod1_tr, _img_mod2_tr = img_mod_params_hist.chunk(2, dim=-1)

        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(
            img_normed,
            mod_params=img_mod1,
            mod_params_tr=img_mod1_tr,
            num_replace_tokens=num_history_tokens,
        )

        # Process text stream - norm1 + modulation (no token_replace)
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, mod_params=txt_mod1)
        return img_modulated, img_gate1, img_mod2, txt_modulated, txt_gate1, txt_mod2

    def postprocess(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            img_mod2: torch.Tensor,
            txt_mod2: torch.Tensor,
            temb_history: Optional[torch.Tensor] = None,
            num_history_tokens: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对 img/text stream 做 norm2 + MLP + residual（含 token_replace 支持）。

        说明：
        - image stream：会对 `hidden_states` 做 norm2，再通过 `_modulate(...)` 产生 `img_modulated2` 与 gate_map，
          gate_map 会逐 token 乘到 MLP 输出上再回写残差。
        - 当提供 `temb_history` 与 `num_history_tokens` 时：image stream 的 token 前缀 `[:num_history_tokens[i]]`
          会使用 history 分支调制参数（等价 Hunyuan token_replace 的 tr_* 分支）。
        - text stream：保持单分支调制。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - hidden_states: 更新后的 image stream token，形状 `[B, S_img, D]`
                - encoder_hidden_states: 更新后的 text stream token，形状 `[B, S_txt, D]`
        """

        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)

        img_mod2_tr = None
        if temb_history is not None:
            img_mod_params_hist = self.img_mod(temb_history)  # [B, 6*dim]
            _img_mod1_tr, img_mod2_tr = img_mod_params_hist.chunk(2, dim=-1)

        img_modulated2, img_gate2 = self._modulate(
            img_normed2,
            mod_params=img_mod2,
            mod_params_tr=img_mod2_tr,
            num_replace_tokens=num_history_tokens,
        )

        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, mod_params=txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states, encoder_hidden_states

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,  # 使用统一的 attention_mask
            temb: torch.Tensor,
            image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,

            history_conditioning_states: Optional[torch.Tensor] = None,
            num_history_tokens: Optional[List[int]] = None,
            temb_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行一次双流（text/image）joint attention + MLP 的 block 前向。

        说明（token_replace / history prefix）：
        - 当提供 `num_history_tokens` 与 `temb_history` 时：
          - 在 `preprocess/postprocess` 内部会使用 `_modulate(...)` 对 image stream 的 token 前缀
            `[:num_history_tokens[i]]` 应用 history 分支调制参数（等价 Hunyuan token_replace 的 tr_* 分支）。
          - `temb_history` 通常由上层使用 `t=0` 计算得到（对齐 Hunyuan 的 `token_replace_t = zeros_like(t)`）。
        - `history_conditioning_states` 当前仅作为预留接口透传（用于未来更复杂的 history 注入）。

        Args:
            hidden_states (torch.Tensor): image stream token，shape `[B, S_img, dim]`。
            encoder_hidden_states (torch.Tensor): text stream token，shape `[B, S_txt, dim]`。
            attention_mask (torch.Tensor): joint attention mask，shape `[B, 1, 1, S_txt + S_img]`。
            temb (torch.Tensor): timestep embedding，shape `[B, dim]`。
            image_rotary_emb (Optional[Tuple[torch.Tensor, torch.Tensor]]): (vid_freqs, txt_freqs)。
            joint_attention_kwargs (Optional[Dict[str, Any]]): 传给 attention processor 的额外参数。
            history_conditioning_states (Optional[torch.Tensor]): history latents 的 token 序列（packed），当前不参与计算，仅透传。
            num_history_tokens (Optional[List[int]]): batch 内每个样本的 history token 前缀长度（token 数）。
            temb_history (Optional[torch.Tensor]): history 分支 timestep embedding，shape `[B, dim]`（通常对应 `t=0`）。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - encoder_hidden_states: updated text stream token，shape `[B, S_txt, dim]`。
                - hidden_states: updated image stream token，shape `[B, S_img, dim]`。
        """

        # Hunyuan-style token_replace: history token prefix can use a separate modulation branch
        # (D1) reuse the same modulation MLP but feed it `temb_history` (typically t=0 embedding).
        img_modulated, img_gate1, img_mod2, txt_modulated, txt_gate1, txt_mod2 = self.preprocess(
            hidden_states,
            encoder_hidden_states,
            temb,
            temb_history=temb_history,
            num_history_tokens=num_history_tokens,
        )

        # Use QwenAttnProcessor2_0 for joint attention computation
        # This directly implements the DoubleStreamLayerMegatron logic:
        # 1. Computes QKV for both streams
        # 2. Applies QK normalization and RoPE
        # 3. Concatenates and runs joint attention
        # 4. Splits results back to separate streams
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            attention_mask=attention_mask,  # 使用统一的 attention_mask，同时处理文本和图像序列
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        hidden_states, encoder_hidden_states = self.postprocess(
            hidden_states,
            encoder_hidden_states,
            img_mod2,
            txt_mod2,
            temb_history=temb_history,
            num_history_tokens=num_history_tokens,
        )

        return encoder_hidden_states, hidden_states


class QwenImageTransformerBlockFiXMaskST(QwenImageTransformerBlockFiXMask):
    """分离时空注意力（Spatial + Temporal）的 block。

    该版本在原双流（text/image）结构基础上，额外增加：
    - `temporal_attn`: 仅对 image stream 做时间分支的 attention。
    - `get_rope/get_attention_mask`: 生成 spatial 与 temporal 两套 RoPE/mask。

    temporal 分支的 token 排布依赖 `img_shapes` 的定义；如修改 token 排布需同步确认（在注释中说明即可）。
    """

    # 分离时空注意力
    def __init__(
            self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm",
            eps: float = 1e-6
    ) -> None:
        super().__init__(
            dim,
            num_attention_heads,
            attention_head_dim,
            qk_norm,
            eps
        )

        self.temporal_attn = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=SingleStreamSelfAttnProcessor(),
            qk_norm=qk_norm,
            eps=eps,
        )

        # 类似于 img_gate1，为时间分支添加一个可学习的 gate。
        self.temporal_gate = nn.Parameter(torch.zeros(1))

    @staticmethod
    def get_hw_and_max_frame(
            img_shapes: List[Tuple[int, int, int]],  # [(F, H_grid, W_grid), ...]
    ) -> Tuple[int, int, int]:
        """从 img_shapes 中提取 H/W 以及 batch 内最大帧数。"""
        # batch 内所有视频 H, W 相同。取第一个样本的 H, W
        h, w = img_shapes[0][1], img_shapes[0][2]
        max_frame = max([img_shape[0] for img_shape in img_shapes])
        return h, w, max_frame

    def get_rope(
            self,
            img_shapes: List[Tuple[int, int, int]],
            txt_seq_lens: List[int],
            device: Union[str, torch.device],
            pos_embed: "QwenEmbedRope",
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """生成 spatial 与 temporal 两套 RoPE。

        Returns:
            Tuple:
                - spatial_rope: `(vid_freqs, txt_freqs)`
                - temporal_rope: `vid_freqs`（temporal attention 仅使用 image stream）
        """

        h, w, max_frame = self.get_hw_and_max_frame(img_shapes)

        # --- A. 生成 Spatial RoPE (单帧画面) ---
        # 构造一个虚拟的 shape [(1, H, W)]，代表单帧
        # 这样生成的 RoPE 所有 t 都是 0，只有 h, w 变化
        spatial_rope_shape = [(1, h, w)]
        spatial_freqs_vid, spatial_freqs_txt = pos_embed(
            spatial_rope_shape,
            txt_seq_lens,
            device
        )

        # 构造 Tuple 供 Attention 使用
        spatial_rope = (spatial_freqs_vid, spatial_freqs_txt)

        # --- B. 生成 Temporal RoPE (纯时间序列) ---
        # 构造一个虚拟的 shape [(max_F, 1, 1)]，代表 1x1 像素的视频，长度为 F
        # 这样生成的 RoPE 只有 t 变化，h=0, w=0
        temporal_rope_shape = [(max_frame, 1, 1)]
        # 文本长度这里不重要，因为 Temporal Attention 不看文本，随便传 1
        temporal_freqs_vid, _ = pos_embed(temporal_rope_shape, [1], device)
        # Temporal Attention 只有 Image branch，所以只需要 vid freqs
        temporal_rope = temporal_freqs_vid

        return spatial_rope, temporal_rope

    def get_attention_mask(
            self,
            img_shapes: List[Tuple[int, int, int]],
            txt_seq_lens: List[int],
            device: Union[str, torch.device],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成 spatial 与 temporal 的 attention mask。

        Args:
            img_shapes (List[Tuple[int, int, int]]): batch 内每个样本的 `(F, H_grid, W_grid)`。
            txt_seq_lens (List[int]): batch 内每个样本的文本有效长度（token 数）。
            device (Union[str, torch.device]): 目标设备。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - spatial_mask (torch.Tensor): shape `[B*F, 1, 1, S]` 的 bool mask，`S = L_txt + H_grid*W_grid`。
                - temporal_mask (torch.Tensor): shape `[B*HW, 1, 1, F]` 的 bool mask，`HW = H_grid*W_grid`。
        """
        h, w, max_frame = self.get_hw_and_max_frame(img_shapes)

        # 处理 Spatial Mask
        # img 仅考虑单帧的情况 h*w
        img_attn_mask_list = [  # 理论上每个img大小一致
            torch.ones(img_shape[1] * img_shape[2], device=device)
            for img_shape in img_shapes
        ]
        img_attention_mask = torch.nn.utils.rnn.pad_sequence(img_attn_mask_list, batch_first=True, padding_value=0)

        text_attn_mask_list = [
            torch.ones(txt_seq_len, device=device)
            for txt_seq_len in txt_seq_lens
        ]
        text_attention_mask = torch.nn.utils.rnn.pad_sequence(text_attn_mask_list, batch_first=True, padding_value=0)

        text_attention_mask = text_attention_mask.bool()
        img_attention_mask = img_attention_mask.bool()

        joint_attention_mask_1d = torch.cat([text_attention_mask, img_attention_mask], dim=1)

        # spatial_mask = joint_attention_mask_1d[:, None, None, :]
        # 注：此处以 `spatial_mask = repeat(...)` 得到的形状为准。
        # spatial_mask 形状为 `(B*F, head, query, S)`，可以对齐到 4D mask 记法 `(N, Hq, L, S)`：
        # - N = B*F
        # - Hq = head
        # - L = query（这里为 1）
        # - S = hw_l（text + image token 总长度）
        # L -> Query; S -> Key and Value

        # img [B*F, HW, D] text [B*F, L, D]
        spatial_mask = repeat(
            joint_attention_mask_1d,
            'bs hw_l -> (bs f) head query hw_l',
            f=max_frame,
            head=1,
            query=1
        )

        # 处理 temporal_mask

        # 目标: 将 [B, F*HW, D] -> [B*HW, F, D]
        # 先恢复得到 (B, F, HW, D)，然后把 HW 移到 batch 维

        frame_mask_list = [
            torch.ones(img_shape[0], device=device)
            for img_shape in img_shapes
        ]
        frame_mask = torch.nn.utils.rnn.pad_sequence(frame_mask_list, batch_first=True, padding_value=0)
        frame_mask = frame_mask.bool()

        # frame_mask = frame_mask[:, None, None, :]
        temporal_mask = repeat(frame_mask, 'bs f -> (bs hw) head query f', hw=h * w, head=1, query=1)

        return spatial_mask, temporal_mask

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            attention_mask: Tuple[torch.Tensor, torch.Tensor],
            temb: torch.Tensor,
            rotary_emb: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],

            img_shapes: List[Tuple[int, int, int]],
            txt_seq_lens: List[int],

            history_conditioning_states: Optional[torch.Tensor] = None,
            num_history_tokens: Optional[List[int]] = None,
            temb_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行 Spatial + Temporal 前向。

        说明（token_replace / history prefix）：
        - 当提供 `num_history_tokens` 与 `temb_history` 时：
          - 本 block 会在 `preprocess/postprocess` 内部通过 `_modulate(...)` 对 image stream 的 token 前缀
            `[:num_history_tokens[i]]` 应用 history 分支调制参数（等价 Hunyuan token_replace 的 tr_* 分支）。
          - `temb_history` 通常由上层使用 `t=0` 计算得到（对齐 Hunyuan 的 `token_replace_t = zeros_like(t)`）。
        - `history_conditioning_states` 当前仍为预留接口透传（未来可用于更复杂的 history 注入/替换策略）。

        Args:
            hidden_states (torch.Tensor): image stream token。
            encoder_hidden_states (torch.Tensor): text stream token。
            attention_mask (Tuple[torch.Tensor, torch.Tensor]): (spatial_mask, temporal_mask)。
            temb (torch.Tensor): timestep embedding。
            rotary_emb (Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]): (spatial_rope, temporal_rope)。
            img_shapes (List[Tuple[int, int, int]]): 每个样本的 (F, H_grid, W_grid)。
            txt_seq_lens (List[int]): 每个样本文本有效长度。
            history_conditioning_states (Optional[torch.Tensor]): history latents 的 token 序列（packed），当前不参与计算，仅透传。
            num_history_tokens (Optional[List[int]]): batch 内每个样本的 history token 前缀长度（token 数）。
            temb_history (Optional[torch.Tensor]): history 分支 timestep embedding，shape `[B, dim]`（通常对应 `t=0`）。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - encoder_hidden_states: updated text stream token
                - hidden_states: updated image stream token
        """
        batch_size = hidden_states.shape[0]
        h, w, max_frame = self.get_hw_and_max_frame(img_shapes)
        hw = h * w

        # ==========================
        # 1. 调制与归一化 (Modulation & Norm)
        # ==========================
        # Hunyuan-style token_replace: history token prefix can use a separate modulation branch
        img_modulated, img_gate1, img_mod2, txt_modulated, txt_gate1, txt_mod2 = self.preprocess(
            hidden_states,
            encoder_hidden_states,
            temb,
            temb_history=temb_history,
            num_history_tokens=num_history_tokens,
        )

        spatial_mask, temporal_mask = attention_mask
        spatial_rope, temporal_rope = rotary_emb

        # ==========================
        # 3. 空间注意力 (Spatial Attention, Joint with Text)
        # ==========================

        # 目标: 将 [B, F*HW, D] -> [B*F, HW, D]
        input_spatial = rearrange(img_modulated, 'b (f hw) d -> (b f) hw d', f=max_frame, hw=hw)

        # 文本需要复制以匹配 batch 维度: [B, L, D] -> [B*F, L, D]
        txt_spatial = repeat(txt_modulated, 'b l d -> (b f) l d', f=max_frame)

        # 计算空间注意力
        img_spatial_out, txt_spatial_out = self.attn(
            hidden_states=input_spatial,  # [B*F, HW, D]
            encoder_hidden_states=txt_spatial,  # [B*F, L, D]
            attention_mask=spatial_mask,
            image_rotary_emb=spatial_rope,
        )

        # 还原并添加残差
        # Image: [B*F, HW, D] -> [B, F*HW, D]
        img_spatial_res = rearrange(img_spatial_out, '(b f) hw d -> b (f hw) d', b=batch_size, f=max_frame)

        # Text: [B*F, L, D] -> [B, L, D]。取帧平均
        txt_spatial_res = rearrange(txt_spatial_out, '(b f) l d -> b f l d', f=max_frame)
        txt_spatial_res = txt_spatial_res.mean(dim=1)

        hidden_states = hidden_states + img_gate1 * img_spatial_res
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_spatial_res

        # ==========================
        # 4. 时间注意力 (Temporal Attention, Image Only)
        # ==========================

        # 目标: 将 [B, F*HW, D] -> [B*HW, F, D]
        # 先恢复得到 (B, F, HW, D)，然后把 HW 移到 batch 维
        input_temporal = rearrange(hidden_states, 'b (f hw) d -> (b hw) f d', f=max_frame, hw=hw)

        # 计算时间注意力 (纯 Self-Attention)
        temporal_out = self.temporal_attn(
            hidden_states=input_temporal,  # [B*HW, F, D]
            attention_mask=temporal_mask,
            rotary_emb=temporal_rope,
        )

        # 还原并添加残差
        # [B*HW, F, D] -> [B, F*HW, D]
        img_temporal_res = rearrange(temporal_out, '(b hw) f d -> b (f hw) d', b=batch_size, hw=hw)
        hidden_states = hidden_states + self.temporal_gate * img_temporal_res

        hidden_states, encoder_hidden_states = self.postprocess(
            hidden_states,
            encoder_hidden_states,
            img_mod2,
            txt_mod2,
            temb_history=temb_history,
            num_history_tokens=num_history_tokens,
        )

        return encoder_hidden_states, hidden_states


class QwenImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    """Qwen-Image Transformer2DModel（本仓库修改版）。

    该模型接收两路输入：
    - image tokens: `hidden_states`，形状 `[B, S_img, D_in]`
    - text tokens: `encoder_hidden_states`，形状 `[B, S_txt, D_txt]`

    以及额外的 batch-level 描述信息：
    - `img_shapes`: List[(F, H_grid, W_grid)]，用于生成 RoPE / mask。
    - `txt_seq_lens`: List[int]，用于生成文本侧 mask。

    输出通常为 `Transformer2DModelOutput(sample=...)`。

    该类源自 diffusers 的 Qwen-Image Transformer 实现（本仓库做了适配与修复）。

    Args:
        patch_size (int, 默认 2): 将输入切分为 patch 的大小。
        in_channels (int, 默认 64): 输入通道数。
        out_channels (Optional[int], 默认 16): 输出通道数；若为 None 则默认与 `in_channels` 相同。
        num_layers (int, 默认 60): 使用的双流 DiT block 层数。
        attention_head_dim (int, 默认 128): 每个 attention head 的维度。
        num_attention_heads (int, 默认 24): attention head 数量。
        joint_attention_dim (int, 默认 3584): joint attention 的维度（即 `encoder_hidden_states` 的 embedding/channel 维度）。
        guidance_embeds (bool, 默认 False): 是否使用 guidance embeddings（用于 guidance-distilled 变体）。
        axes_dims_rope (Tuple[int, int, int], 默认 (16, 56, 56)): RoPE 在各轴上分配的维度。
    """

    # 修复 attention mask 没有正确使用的问题
    # https://github.com/huggingface/diffusers/issues/12294
    # +
    # 同时支持对 img 模态使用 mask
    # 在序列生成时支持不同长度的序列

    _supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImageTransformerBlockFiXMask"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["QwenImageTransformerBlockFiXMask"]

    @register_to_config
    def __init__(
            self,
            patch_size: int = 2,
            in_channels: int = 64,
            out_channels: Optional[int] = 16,
            num_layers: int = 60,
            attention_head_dim: int = 128,
            num_attention_heads: int = 24,
            joint_attention_dim: int = 3584,
            axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
    ) -> None:
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.use_st_att = bool(os.getenv('USE_ST_ATT'))

        if not self.use_st_att:
            self.transformer_blocks = nn.ModuleList(
                [
                    QwenImageTransformerBlockFiXMask(
                        dim=self.inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [
                    QwenImageTransformerBlockFiXMaskST(
                        dim=self.inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                    )
                    for _ in range(num_layers)
                ]
            )
        self.transformer_block = self.transformer_blocks[0]

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            timestep: torch.Tensor,

            # 实际上每个 Tuple 为代表 video 的 [frame, height, width] 大小
            img_shapes: List[Tuple[int, int, int]],
            txt_seq_lens: List[int],

            history_conditioning_states: Optional[torch.Tensor] = None,
            num_history_tokens: Optional[List[int]] = None,

            attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
    ) -> Union[Transformer2DModelOutput, Tuple[torch.Tensor]]:
        """前向传播。

        Args:
            hidden_states (torch.Tensor): image stream token，形状 `[B, S_img, in_channels]`。
            encoder_hidden_states (torch.Tensor): text token，形状 `[B, S_txt, joint_attention_dim]`。
            timestep (torch.Tensor): timestep，形状 `[B]`。
                说明：本项目中常将 timestep 以 float dtype 输入（例如在上游除以 1000 做归一化），
                因此不强制要求为 LongTensor。
            img_shapes (List[Tuple[int, int, int]]): 每个样本的 `(F, H_grid, W_grid)`。
            txt_seq_lens (List[int]): 每个样本文本有效长度。
            history_conditioning_states (Optional[torch.Tensor]): history latents 的 token 序列（packed）。当前主要用于与采样侧对齐的数据流透传（预留接口）。
            num_history_tokens (Optional[List[int]]): batch 内每个样本的 history token 前缀长度（token 数）。当不为 None 时，会启用 history/token_replace 风格调制。
            attention_kwargs (Optional[Dict[str, Any]]): 额外 attention 参数。
            return_dict (bool): 是否返回 `Transformer2DModelOutput`。

        Returns:
            Union[Transformer2DModelOutput, Tuple[torch.Tensor]]:
                - 若 `return_dict=False`：返回 1-tuple：`(sample,)`
                - 否则返回 `Transformer2DModelOutput(sample=...)`

        该函数对 `img_shapes/txt_seq_lens` 的要求与 token padding/pack 方式强相关；
        如修改 pack/token 排布，请同步确认 mask/rope 的生成逻辑。
        mask/rope 的生成逻辑随 Att Block 定义，详见 QwenImageTransformerBlockFiXMask 以及 QwenImageTransformerBlockFiXMaskST。
        """

        device = hidden_states.device
        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        temb = self.time_text_embed(timestep, hidden_states)

        # token_replace-style modulation: use t=0 embedding for history tokens
        temb_history = None
        if num_history_tokens is not None:
            temb_history = self.time_text_embed(torch.zeros_like(timestep), hidden_states)

        rotary_emb = self.transformer_block.get_rope(
            img_shapes,
            txt_seq_lens,
            device,
            self.pos_embed
        )
        attention_mask = self.transformer_block.get_attention_mask(
            img_shapes,
            txt_seq_lens,
            device,
        )

        for index_block, block in enumerate(self.transformer_blocks):
            if not self.use_st_att:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    temb=temb,
                    image_rotary_emb=rotary_emb,
                    joint_attention_kwargs=attention_kwargs,

                    history_conditioning_states=history_conditioning_states,
                    num_history_tokens=num_history_tokens,
                    temb_history=temb_history,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    temb=temb,
                    rotary_emb=rotary_emb,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,

                    history_conditioning_states=history_conditioning_states,
                    num_history_tokens=num_history_tokens,
                    temb_history=temb_history,
                )

        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
