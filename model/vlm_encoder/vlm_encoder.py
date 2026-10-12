import os
from typing import Union, List, Tuple

import torch
from einops import repeat
from torch import nn
from transformers import Qwen3VLForConditionalGeneration, AutoModel, AutoTokenizer, BitsAndBytesConfig


class VLMEncoder(nn.Module):
    """VLM 文本编码器（冻结使用）。

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
            max_text_prompt_length: int = 1024,
    ) -> None:
        super().__init__()
        if os.getenv('VLM') == 'Qwen2.5-VL-3B':
            self.text_dim = 2048
        elif os.getenv('VLM') == 'Qwen3-VL-2B':
            self.text_dim = 2048
        else:
            # Qwen2.5-VL-7B
            self.text_dim = 3584

        # --- QLoRA 配置 ---
        # 以 4-bit 量化加载模型
        # 无论是纯推理或者微调，都可以这样设置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        if os.getenv('VLM') == 'Qwen3-VL-2B':
            self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                text_encoder_path,
                quantization_config=quantization_config,
                device_map={"": torch.cuda.current_device()},
            )
        else:
            self.text_encoder = AutoModel.from_pretrained(
                text_encoder_path,
                quantization_config=quantization_config,
                device_map={"": torch.cuda.current_device()},
            )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # <|im_start|> 和 <|im_end|> 定义了一个“消息块”, 而不是图片
        self.max_sequence_length = max_text_prompt_length
        self.prompt_template_encode = \
            ("<|im_start|>system\nDescribe an image by detailing the scene, structure, "
             "and spatial relationships of the objects and background, based on the user's text input."
             "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")
        # 注：该 start_idx 属于硬编码机制（模板 token 长度）。
        # 更换 tokenizer / 模型版本 / prompt_template_encode 内容后，该索引可能不再正确，需要重新计算。
        self.prompt_template_encode_start_idx = 35

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
        利用 Qwen-VL 模型将简单的文本提示词（prompt）转换为内容丰富的嵌入向量（embeddings）。

        这个过程的核心思想是“隐式提示词扩展”：
        1. 使用一个包含详细描述指令的模板来包装用户的简单提示词。
        2. 让 Qwen-VL 模型在理解了这些指令的上下文中，去处理用户的提示词。
        3. 提取出在这种丰富上下文中生成的、代表用户提示词部分的嵌入向量。

        最终得到的嵌入向量会蕴含比原始文本更多的视觉细节信息。

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

        # ---- 步骤 1: 使用模板构建完整的输入 ----
        # 这个模板指示Qwen模型扮演一个详细的图像描述者。
        template = self.prompt_template_encode
        full_prompts = [template.format(p) for p in prompts]

        # ---- 步骤 2: 分词（Tokenization） ----
        # 将格式化后的完整提示词转换为模型可处理的 token ID。
        # 注意：max_length 需要足够长，以容纳模板和用户输入。
        tokenizer_max_len = self.max_sequence_length + self.prompt_template_encode_start_idx
        tokenized_output = self.tokenizer(
            full_prompts,
            max_length=tokenizer_max_len,
            padding=True,  # 对批次中的短序列进行填充
            truncation=True,  # 对超长序列进行截断
            return_tensors="pt"
        ).to(device)

        # ---- 步骤 3: 获取模型最后一层的隐藏状态（Embeddings） ----
        # 将 token 输入 Qwen-VL 模型，获取其最后一层 Transformer 的输出。
        # 这些隐藏状态是在模型充分理解了 System 指令后生成的。
        encoder_outputs = self.text_encoder(
            input_ids=tokenized_output.input_ids,
            attention_mask=tokenized_output.attention_mask,
            output_hidden_states=True,
        )
        last_hidden_states = encoder_outputs.hidden_states[-1]

        # ---- 步骤 4: 移除填充并裁剪掉模板部分 ----
        # 4.1. 将填充后的批次张量转换回可变长度的序列列表。
        unpadded_states_list = self.unpad_and_listify_sequences(last_hidden_states, tokenized_output.attention_mask)

        # 4.2. 从每个序列中裁剪掉前面属于模板（System指令和User标识）的部分。
        # self.prompt_template_encode_start_idx 是预先计算好的模板部分的token长度。
        # 我们保留的是用户提示词和 'assistant' 标识符对应的嵌入，但它们已吸收了模板指令的上下文。
        crop_index = self.prompt_template_encode_start_idx
        cropped_embeds_list = [seq_embeds[crop_index:] for seq_embeds in unpadded_states_list]

        # ---- 步骤 5: 将裁剪后的序列重新填充为规整的张量 ----
        # 5.1. 创建新的注意力掩码（此时所有token都是有效的，所以全是1）。
        new_attn_mask_list = [torch.ones(embeds.size(0), device=device, dtype=torch.long) for embeds in
                              cropped_embeds_list]

        # 5.2. 找到裁剪后序列的最大长度，作为新张量的标准长度。
        # max_len_after_crop = max(embeds.size(0) for embeds in cropped_embeds_list)

        # 5.3. 统一填充（padding）所有序列到 max_len_after_crop。
        # `torch.nn.utils.rnn.pad_sequence` 是一个更标准、更易读的填充工具。
        # `batch_first=True` 使输出形状为 (batch_size, max_len, hidden_size)。
        # `padding_value=0` 用0填充。
        prompt_embeds = torch.nn.utils.rnn.pad_sequence(cropped_embeds_list, batch_first=True, padding_value=0.)
        encoder_attention_mask = torch.nn.utils.rnn.pad_sequence(new_attn_mask_list, batch_first=True, padding_value=0)

        # 确保数据类型正确
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        # ---- 最终结果注释 ----
        # prompt_embeds:
        # 这是一个形状为 (batch_size, max_len_after_crop, hidden_size) 的张量。
        # 它包含了经过Qwen-VL模型“意图增强”后的提示词嵌入。
        # 即使原始提示词很简单，这里的嵌入也蕴含了颜色、形状、背景等丰富细节的语义信息。

        # encoder_attention_mask:
        # 这是一个形状为 (batch_size, max_len_after_crop) 的张量。
        # 它标记了 prompt_embeds 中哪些是有效部分（值为1），哪些是填充部分（值为0）。

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


if __name__ == '__main__':
    from os.path import dirname, abspath

    __project_root__ = dirname(dirname(dirname(abspath(__file__))))

    os.environ['VLM'] = 'Qwen3-VL-2B'
    text_encoder_qlora = VLMEncoder(
        text_encoder_path=f'{__project_root__}/ckpt/Qwen3-VL-2B-Instruct',
        tokenizer_path=f'{__project_root__}/ckpt/Qwen3-VL-2B-Instruct',
        max_text_prompt_length=256,
    )
    prompt_embeds, prompt_embeds_mask = text_encoder_qlora.encode_prompt(['', 'prompt 1', 'some prompt 2'])
    print()
