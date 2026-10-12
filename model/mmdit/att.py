# Derived from the diffusers AttentionProcessor design
# (https://github.com/huggingface/diffusers), adapted as a single-stream
# self-attention processor with RoPE for the OccDirector MMDiT.
"""diffusers 单流 self-attention processor。

该文件提供 `SingleStreamSelfAttnProcessor`，用于替换 diffusers 的 AttentionProcessor，
实现单流 self-attention 的 Q/K/V 投影、RoPE 以及 attention mask 传递。
"""

from typing import Any, Optional

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen


class SingleStreamSelfAttnProcessor:
    """单流（single-stream）Self-Attention Processor。

    该类用于替换 diffusers 的 attention processor，实现单流 self-attn 的 Q/K/V 投影、RoPE 以及
    attention mask 的传递。

    形状约定（以 diffusers/Attention 为准）：
    - `hidden_states`: `[B, S, D]`
    - `attention_mask`: broadcastable 到 SDPA 的 mask（常见为 `[B, 1, 1, S_total]`）

    当前关于 `attention_mask` 的注释是准确的（以 diffusers/Attention 的约定为准）。
    """

    _attention_backend: Any = None  # comment 说明从库文件里面之过来就是这样的，不用管

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,  # data stream
            attention_mask: Optional[torch.Tensor] = None,
            rotary_emb: Optional[torch.Tensor] = None,
            **kwargs: Any,
    ) -> torch.Tensor:
        """执行单流 self-attention。

        Args:
            attn (Attention): diffusers 的 Attention 模块实例（包含 to_q/to_k/to_v 等投影层）。
            hidden_states (torch.Tensor): 输入 token，形状 `[B, S, D]`。
                dtype 由上游决定（可能为 bf16/fp16/fp32）。
            attention_mask (Optional[torch.Tensor]): 可 broadcast 到 SDPA 的 mask（常见为 `[B, 1, 1, S]`）。
            rotary_emb (Optional[torch.Tensor]): RoPE 频率张量（由上游 block 生成），用于对 Q/K 做旋转位置编码。
            **kwargs: 兼容 diffusers AttentionProcessor 的额外参数（当前实现不使用）。

        Returns:
            torch.Tensor: 输出 token，形状 `[B, S, D]`。
        """
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Reshape for multi-head attention
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE
        if rotary_emb is not None:
            query = apply_rotary_emb_qwen(query, rotary_emb, use_real=False)
            key = apply_rotary_emb_qwen(key, rotary_emb, use_real=False)

        # Compute attention
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Apply output projections
        hidden_states = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            hidden_states = attn.to_out[1](hidden_states)  # dropout

        return hidden_states
