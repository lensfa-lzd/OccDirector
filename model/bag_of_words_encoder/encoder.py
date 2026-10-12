import os
from typing import Union, List, Tuple

import torch
from einops import repeat
from torch import nn
from transformers import BitsAndBytesConfig


class TEXTEncoder(nn.Module):
    """

    主要接口：
    - `encode_prompt`: 将 `Union[str, List[str]]` prompt 编码为 padding 后的 embedding + mask。

    返回张量形状：
    - `prompt_embeds`: `[B, T, text_dim]`
    - `prompt_embeds_mask`: `[B, T]`（通常为 long/int mask，1 表示有效 token，0 表示 padding）
    """

    def __init__(
            self,
            text_encoder_path: str,
            tokenizer_path: str,
    ) -> None:
        super().__init__()
        if os.getenv('VLM') == 'CLIP-VIT-LARGE':
            self.max_sequence_length = 77
            self.text_dim = 768

            from transformers import CLIPTextModel, CLIPTokenizer

            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
            self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)

        elif os.getenv('VLM') == 'T5-XL':
            self.max_sequence_length = 256
            self.text_dim = 2048

            from transformers import T5Tokenizer, T5EncoderModel

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            self.tokenizer = T5Tokenizer.from_pretrained(
                tokenizer_path,
                quantization_config=quantization_config,
                device_map={"": torch.cuda.current_device()},
            )
            self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_path)
        else:
            raise NotImplementedError

    def freeze_grad(self) -> None:
        """冻结该编码器的所有参数梯度。"""
        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def unpad_and_listify_sequences(
            padded_states: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        将一个经过填充（padded）的批次张量，根据其注意力掩码，转换回一个包含可变长度序列的 tuple。

        说明：当前实现使用 `torch.split`，因此返回类型为 tuple（而非 list）。

        Args:
            padded_states (torch.Tensor): 形状为 `[B, T, hidden_size]` 的填充后张量。
            attention_mask (torch.Tensor): 形状为 `[B, T]` 的注意力掩码，1 表示有效 token，0 表示 padding token。

        Returns:
            Tuple[torch.Tensor, ...]: 一个 tuple，其中每个元素都是一个未填充的、可变长度的张量。
        """

        # 等价代码
        # unpadded_list = []
        # # 遍历批次中的每一个序列
        # for i in range(padded_states.shape[0]):
        #     # 计算当前序列的实际（有效）长度
        #     actual_length = int(attention_mask[i].sum())
        #     # 根据实际长度，从填充张量中切片出有效部分
        #     sequence = padded_states[i, :actual_length]
        #     unpadded_list.append(sequence)

        # 说明：原始实现 `hidden_states[mask.bool()]` 和 `torch.split` 是一种更高效的向量化操作，
        # 在生产环境中性能更佳，但这里的循环版本更易于理解其逻辑。

        # 原始实现
        bool_mask = attention_mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = padded_states[bool_mask]
        unpadded_list = torch.split(selected, valid_lengths.tolist(), dim=0)

        return unpadded_list

    def get_rich_prompt_embeddings(
            self,
            prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            prompts (List[str]): 用户输入的原始提示词列表。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - prompt_embeds (torch.Tensor): 形状为 `(batch_size, sequence_length, hidden_size)` 的嵌入向量。
                - attention_mask (torch.Tensor): 形状为 `(batch_size, sequence_length)` 的注意力掩码。
        """
        if not hasattr(self.text_encoder, 'hf_device_map'):
            device = self.text_encoder.device
        else:
            device = next(self.text_encoder.parameters()).device

        dtype = self.text_encoder.dtype

        tokenized_output = self.tokenizer(
            prompts,
            max_length=self.max_sequence_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        last_hidden_states = self.text_encoder(tokenized_output.input_ids).last_hidden_state

        # ---- 步骤 4: 移除填充并裁剪掉模板部分 ----
        # 4.1. 将填充后的批次张量转换回可变长度的序列列表。
        unpadded_states_list = self.unpad_and_listify_sequences(last_hidden_states, tokenized_output.attention_mask)

        # 4.2. 从每个序列中裁剪掉前面属于模板（System指令和User标识）的部分。
        # self.prompt_template_encode_start_idx 是预先计算好的模板部分的token长度。
        # 我们保留的是用户提示词和 'assistant' 标识符对应的嵌入，但它们已吸收了模板指令的上下文。
        # crop_index = self.prompt_template_encode_start_idx
        # cropped_embeds_list = [seq_embeds[crop_index:] for seq_embeds in unpadded_states_list]

        # ---- 步骤 5: 将裁剪后的序列重新填充为规整的张量 ----
        # 5.1. 创建新的注意力掩码（此时所有token都是有效的，所以全是1）。
        new_attn_mask_list = [
            torch.ones(embeds.size(0), device=device, dtype=torch.long) for embeds in unpadded_states_list
        ]

        # 5.2. 找到裁剪后序列的最大长度，作为新张量的标准长度。
        # max_len_after_crop = max(embeds.size(0) for embeds in cropped_embeds_list)

        # 5.3. 统一填充（padding）所有序列到 max_len_after_crop。
        # `torch.nn.utils.rnn.pad_sequence` 是一个更标准、更易读的填充工具。
        # `batch_first=True` 使输出形状为 (batch_size, max_len, hidden_size)。
        # `padding_value=0` 用0填充。
        prompt_embeds = torch.nn.utils.rnn.pad_sequence(unpadded_states_list, batch_first=True, padding_value=0.)
        encoder_attention_mask = torch.nn.utils.rnn.pad_sequence(new_attn_mask_list, batch_first=True, padding_value=0)

        # 确保数据类型正确
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
            self,
            prompt: Union[str, List[str]],
            num_images_per_prompt: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将文本 prompt 编码为 embedding 与 attention mask。

        Args:
            prompt (Union[str, List[str]]): 待编码的 prompt。
                - 若为 `str`：视为单条 prompt。
                - 若为 `List[str]`：长度为 `B`。
            num_images_per_prompt (int): 每个 prompt 需要生成的样本数。
                该参数会在 batch 维复制文本 token：
                - `prompt_embeds`: `[B, T, D] -> [B*num_images_per_prompt, T, D]`
                - `prompt_embeds_mask`: `[B, T] -> [B*num_images_per_prompt, T]`

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - prompt_embeds (torch.Tensor): 文本 embedding，形状 `[B', T, text_dim]`，其中 `B' = B*num_images_per_prompt`。
                  当 `prompt` 为 `List[str]` 时，`B = len(prompt)`。
                - prompt_embeds_mask (torch.Tensor): token mask，形状 `[B', T]`。
                  值为 1 表示有效 token，0 表示 padding token；dtype 通常为 `torch.long`。
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds, prompt_embeds_mask = self.get_rich_prompt_embeddings(prompt)

        prompt_embeds = prompt_embeds[:, :self.max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :self.max_sequence_length]

        # 为多图生成复制数据: 这一部分是为了处理 num_images_per_prompt > 1 的情况。
        # _, seq_len, _ = prompt_embeds.shape
        # prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        # prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        # prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        # 使用 einops.repeat 替换原始代码
        # 'b' 代表 batch_size, 's' 代表 seq_len, 'h' 代表 hidden_dim
        # 'b -> (n b)' 表示将批次维度 b 重复 n 次并展平
        prompt_embeds = repeat(prompt_embeds, 'b s h -> (n b) s h', n=num_images_per_prompt)
        prompt_embeds_mask = repeat(prompt_embeds_mask, 'b s -> (n b) s', n=num_images_per_prompt)

        return prompt_embeds, prompt_embeds_mask
