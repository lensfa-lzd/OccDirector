import logging
import os
from os.path import dirname, abspath

__project_root__ = dirname(dirname(abspath(__file__)))

from typing import Optional, Union, Tuple, List
import torch
import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange, repeat
from tqdm import tqdm

from model.mmdit.transformer_qwenimage import QwenImageTransformer2DModel


class MMDiT(QwenImageTransformer2DModel):
    """基于 QwenImageTransformer2DModel 的视频/序列 MMDiT。

    该实现将 VAE latent（`[B, C, F, H, W]`）按 patch 打包成 token 序列：
    - `pack_latents`: `[B, C, F, H, W] -> [B, S, C*p*p]`
      其中 `S = F * (H/p) * (W/p)`。
    - `unpack_latents`: 逆操作。

    采样时（`sample`）会在 token 维度上进行 denoising，并支持 CFG 以及历史 token 注入。

    注意：
    - 需要保证 `H % patch_size == 0` 且 `W % patch_size == 0`。
    - history conditioning 逻辑要求 `num_history_frames[i] <= min(frames, history_F)`，否则语义可能不一致。
      这里不需要显式 assert。
    """

    def __init__(
            self,
            patch_size: int,
            num_layers: int,
            max_frames: int,
            latent_occ_frame_shape: Tuple[int, int, int],  # vae 的 latent_occ_frame_shape [H_lat, W_lat, C_lat]
            joint_attention_dim: int = 3584,  # 文本的输入维度，即 VLM 的文本编码维度；3584 为 Qwen 2.5 VL 7B 的隐藏维度
    ) -> None:
        # 参考 SD3 的 MMDiT 结构进行放缩
        hidden_size = 64 * num_layers
        number_of_att_head = num_layers
        attention_head_dim = hidden_size // number_of_att_head  # 64

        self.patch_size = patch_size
        self.latent_occ_channel = latent_occ_frame_shape[-1]
        self.latent_occ_frame_shape = latent_occ_frame_shape

        if os.getenv("SAMPLE_NO_TQDM"):
            self.sample_tqdm = False
        else:
            self.sample_tqdm = True

        super().__init__(
            patch_size=patch_size,  # MMdit 本身并不会将输入再次分块，需要在输入前处理完毕
            in_channels=self.latent_occ_channel * patch_size * patch_size,
            out_channels=self.latent_occ_channel,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=number_of_att_head,
            joint_attention_dim=joint_attention_dim,  # 文本的输入维度，即 VLM 的文本编码维度；3584 为 Qwen 2.5 VL 7B 的隐藏维度
            axes_dims_rope=(8, 28, 28),  # 控制 3D RoPE 编码大小，总和需要等于 attention_head_dim
        )

        # 针对视频生成进行更新
        height, width, _ = self.latent_occ_frame_shape
        token_per_frame = (height // self.patch_size) * (width // self.patch_size)
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_config({
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "_diffusers_version": "0.34.0.dev0",

            # 单帧长度
            "base_image_seq_len": token_per_frame,
            "base_shift": 0.5,

            "invert_sigmas": False,

            # 最长的帧数
            "max_image_seq_len": token_per_frame * max_frames,
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

    def pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """将 VAE latent 打包为 token 序列。

        Args:
            latents (torch.Tensor): 形状为 `[B, C_lat, F, H_lat, W_lat]`。

                - `C_lat`: VAE latent channel 数。
                - `F`: 帧数。
                - `H_lat, W_lat`: latent 平面分辨率。

                需要满足 `H_lat % patch_size == 0` 且 `W_lat % patch_size == 0`。

        Returns:
            torch.Tensor: 打包后的 token 序列，形状为 `[B, S, C_lat * patch_size * patch_size]`。
            其中 `S = F * (H_lat / patch_size) * (W_lat / patch_size)`。
        """

        # latents [bs, c, f, h, w] -> [bs, seq_len, c_stack]

        latents = rearrange(
            latents,
            "b c f (h p1) (w p2) -> b (f h w) (c p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )

        return latents

    def unpack_latents(self, latents: torch.Tensor, frames: int) -> torch.Tensor:
        """将 token 序列还原为 VAE latent。

        Args:
            latents (torch.Tensor): 形状为 `[B, S, C_lat * patch_size * patch_size]`。
            frames (int): 目标帧数 `F`。

                该参数需要与 `S` 保持一致（`S == F * H_grid * W_grid`），否则 `einops.rearrange` 会报错。
                若不一致，`einops.rearrange` 会报错。

        Returns:
            torch.Tensor: 形状为 `[B, C_lat, F, H_lat, W_lat]`。
        """

        # latents [bs, seq_len, c]
        height, width, _ = self.latent_occ_frame_shape
        latents = rearrange(
            latents,
            "b (f h w) (c p1 p2) -> b c f (h p1) (w p2)",
            f=frames,
            h=height // self.patch_size,
            w=width // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
        )

        return latents

    def prepare_latents(
            self,
            batch_size: int,
            frames: int,
            generator: Optional[torch.Generator],
            device: Union[str, torch.device],
    ) -> torch.Tensor:
        """初始化采样用的噪声 latent，并打包为 token 序列。

        Args:
            batch_size (int): 采样 batch 大小（若 CFG 生效，则是原始 prompt batch_size）。
            frames (int): 采样帧数 `F`。
            generator (Optional[torch.Generator]): 随机数生成器（可用于复现）。
            device (Union[str, torch.device]): 目标设备。

        Returns:
            torch.Tensor: 打包后的 latent token，形状为 `[B, S, C_lat*patch_size*patch_size]`。
            其中 `S = F * (H_lat/patch_size) * (W_lat/patch_size)`。
        """

        height, width, _ = self.latent_occ_frame_shape

        if generator and os.getenv('BATCH_SAMPLE_FROM_SAME_NOISE'):
            shape = (self.latent_occ_channel, frames, height, width)
            latent = torch.randn(shape, generator=generator, device=device)
            latents = repeat(latent, 'c f h w -> b c f h w', b=batch_size)
        else:
            shape = (batch_size, self.latent_occ_channel, frames, height, width)
            latents = torch.randn(shape, generator=generator, device=device)
        latents = self.pack_latents(latents)
        return latents

    def calculate_shift(self, image_seq_len: int) -> float:
        """根据序列长度计算 dynamic shifting 的 mu。

        Args:
            image_seq_len (int): 图像/视频 token 序列长度（即 `S`）。

        Returns:
            float: 调度器使用的 mu。
        """

        base_seq_len = self.noise_scheduler.config.get("base_image_seq_len")
        max_seq_len = self.noise_scheduler.config.get("max_image_seq_len")
        base_shift = self.noise_scheduler.config.get("base_shift")
        max_shift = self.noise_scheduler.config.get("max_shift")

        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def retrieve_timesteps(
            self,
            sigmas: np.ndarray,
            mu: float,
            device: Optional[Union[str, torch.device]],
    ) -> torch.Tensor:
        """调用 scheduler 并返回 timesteps。

        该方法会调用 `self.noise_scheduler.set_timesteps(sigmas=..., mu=..., device=...)`，并读取 scheduler
        内部生成的 `timesteps`。

        Args:
            sigmas (np.ndarray): 采样用的 sigma 序列，形状通常为 `[num_inference_steps]`。
            mu (float): dynamic shifting 的参数（由 `calculate_shift` 计算得到）。
            device (Optional[Union[str, torch.device]]): 目标设备。

        Returns:
            torch.Tensor: scheduler 的 timesteps，形状为 `[num_inference_steps]`。
        """
        self.noise_scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=device)
        timesteps = self.noise_scheduler.timesteps
        return timesteps

    def get_sample_time_steps(
            self,
            image_seq_len: int,
            num_inference_steps: int,
            device: Union[str, torch.device],
    ) -> torch.Tensor:
        """生成采样过程使用的 timesteps。

        Args:
            image_seq_len (int): token 序列长度 `S`。
            num_inference_steps (int): 采样步数。
            device (Union[str, torch.device]): 目标设备。

        Returns:
            torch.Tensor: scheduler timesteps，形状为 `[num_inference_steps]`。
        """

        sigmas_np = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = self.calculate_shift(image_seq_len)
        timesteps = self.retrieve_timesteps(sigmas=sigmas_np, mu=mu, device=device)
        return timesteps

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

    @staticmethod
    def replace_history_tokens(
            latents: torch.Tensor,
            num_history_tokens: List[int],
            history_latents: torch.Tensor,
    ) -> torch.Tensor:
        """在 token 维度上用 history token 前缀替换当前 latents 的前缀部分。

        该函数用于实现 HunyuanVideo-I2V 数学形式等价的“token_replace（输入级替换/硬条件）”行为：
        - 对每个样本 i，将 `latents[i, :num_history_tokens[i]]` 替换为 `history_latents[i, :num_history_tokens[i]]`。
        - 该替换不依赖 timestep，也不对 history 做额外加噪；history 是否带噪由传入的 `history_latents` 决定。

        Args:
            latents (torch.Tensor): 当前采样 latents（token 序列），形状 `[B, S, D]`。
            num_history_tokens (List[int]): 每个样本需要替换的 history token 前缀长度。
            history_latents (torch.Tensor): 用于替换的 history token 序列，形状 `[B, S_hist, D]`。

        Returns:
            torch.Tensor: 替换后的 latents，形状 `[B, S, D]`。
        """

        for i, n in enumerate(num_history_tokens):
            if n > 0:
                latents[i, :n, :] = history_latents[i, :n, :]
        return latents

    @staticmethod
    def get_num_history_tokens(
            padded_frame: int,
            num_history_frames: List[int],
            padded_history_latents: torch.Tensor,
    ) -> List[int]:
        """计算 batch 内每个样本的 history token 前缀长度。

        说明：
        - 该函数假设 `pack_latents` 的 token 排布是 frame-major（`(f h w)`），因此前
          `num_history_frames[i]` 帧对应的 token 在序列中天然构成一个 prefix。

        Args:
            padded_frame (int): `padded_history_latents` 中的（padding 后）帧数。
            num_history_frames (List[int]): batch 内每个样本的 history 帧数，长度为 `B`。
            padded_history_latents (torch.Tensor): 打包后的 history latents，形状 `[B, S, D]`。

        Returns:
            List[int]: batch 内每个样本的 history token 数（prefix 长度），长度为 `B`。
        """

        _, num_token, _ = padded_history_latents.shape
        hw = num_token // padded_frame

        num_history_tokens = [hw * num_history_frame for num_history_frame in num_history_frames]
        return num_history_tokens

    def sample(
            self,
            prompt_embeds: torch.Tensor,
            prompt_embeds_mask: torch.Tensor,

            history_conditioning_latent: Optional[torch.Tensor] = None,  # [B, C_lat, F_hist, H_lat, W_lat]
            num_history_frames: Optional[List[int]] = None,

            frames: int = 1,  # 总生成帧数（包含历史帧在内，或者根据你的逻辑定义）
            num_inference_steps: int = 50,
            generator: Optional[torch.Generator] = None,
            guidance_scale: float = 1.0,

            show_process_bar: bool = True,
    ) -> torch.Tensor:
        """进行采样生成（Euler-style denoising loop）。

        Args:
            prompt_embeds (torch.Tensor): 文本 embedding。
                - 无 CFG 时形状 `[B, T, D_txt]`
                - CFG 时形状 `[2B, T, D_txt]`（uncond+cond 拼接）
            prompt_embeds_mask (torch.Tensor): 文本 mask，形状 `[B, T]` 或 `[2B, T]`。
            history_conditioning_latent (Optional[torch.Tensor]): 历史 latent，形状 `[B, C_lat, F_hist, H_lat, W_lat]`。
            num_history_frames (Optional[List[int]]): 每个样本的历史帧数，长度 `B`。
            frames (int): 总生成帧数（包含历史帧在内）。
            num_inference_steps (int): 采样步数。
            generator (Optional[torch.Generator]): 随机数生成器。
            guidance_scale (float): CFG scale；>1 时启用 CFG。
            show_process_bar (bool): 是否显示 tqdm。
                注意：实际是否显示还受环境变量 `SAMPLE_NO_TQDM` 控制（若设置则强制关闭）。

        Returns:
            torch.Tensor: 采样得到的 latent，形状 `[B, C_lat, frames, H_lat, W_lat]`。

        history conditioning 的 token 替换逻辑默认把历史部分放在 token 序列前缀；
        把历史部分放在前面是因为 “video frame” 的 token 顺序是按时间顺序增加的。
        若后续修改 pack 方式或 token 顺序，需要同步确认替换逻辑。

        备注：历史版本变量名曾误写为 `condiction`，此处语义为 conditioning（条件信息）。
        """
        padded_history_latents = None
        num_history_tokens: Optional[List[int]] = None
        # history_conditioning_states 为打包后的 history latents token 序列。
        # 当前仅用于在 transformer 调用链中透传，供后续 token_replace 风格的 modulation/gating 使用。
        history_conditioning_states: Optional[torch.Tensor] = None
        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            batch_size = prompt_embeds.shape[0] // 2
        else:
            batch_size = prompt_embeds.shape[0]

        device = prompt_embeds.device

        # 4. Prepare latent variables
        # latents [b c f (h p1) (w p2) -> b (f h w) (c p1 p2)]
        latents = self.prepare_latents(batch_size, frames, generator, device)

        # 5. Prepare timesteps and sigmas
        timesteps = self.get_sample_time_steps(latents.shape[1], num_inference_steps, device)

        # ==========================================
        # 4. 初始替换 (t=T)
        # ==========================================
        # 在开始去噪之前，将 latents 中的历史部分替换为加了最大噪声的历史帧
        # 这样保证起点分布的一致性
        if history_conditioning_latent is not None:
            # padded_history_latents [b c f (h p1) (w p2) -> b (f h w) (c p1 p2)]
            _, _, frame, _, _ = history_conditioning_latent.shape
            padded_history_latents = self.pack_latents(history_conditioning_latent)

            # history_conditioning_states 即打包后的 history latents（token 序列）。
            history_conditioning_states = padded_history_latents

            num_history_tokens = self.get_num_history_tokens(frame, num_history_frames, padded_history_latents)

            # # Hunyuan-style token_replace: 直接把 history token 前缀拼回输入（硬条件）。
            # latents = self.replace_history_tokens(
            #     latents=latents,
            #     num_history_tokens=num_history_tokens,
            #     history_latents=padded_history_latents,
            # )

            if do_classifier_free_guidance:
                num_history_tokens += num_history_tokens
                history_conditioning_states = torch.cat([history_conditioning_states] * 2, dim=0)

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        height, width, _ = self.latent_occ_frame_shape
        gen_occ_shapes = [(frames, height // self.patch_size, width // self.patch_size)] * prompt_embeds.shape[0]

        if self.sample_tqdm and show_process_bar:
            process_bar = tqdm(total=len(timesteps))
        else:
            process_bar = None

        # ==========================================
        # 5. Denoising loop
        # ==========================================
        for i, t in enumerate(timesteps):
            # 扩展 timestep 用于 CFG
            if do_classifier_free_guidance:
                timestep_expanded = t.expand(latents.shape[0] * 2).to(latents.dtype)
            else:
                timestep_expanded = t.expand(latents.shape[0]).to(latents.dtype)

            # 准备模型输入
            if do_classifier_free_guidance:
                model_input = torch.cat([latents] * 2)
            else:
                model_input = latents

            pred_vector = self.forward(
                hidden_states=model_input,
                timestep=timestep_expanded / 1000.0,
                encoder_hidden_states=prompt_embeds,
                img_shapes=gen_occ_shapes,
                txt_seq_lens=txt_seq_lens,
                history_conditioning_states=history_conditioning_states,
                num_history_tokens=num_history_tokens,
            ).sample

            # CFG 引导
            if do_classifier_free_guidance:
                pred_uncond, pred_cond = pred_vector.chunk(2)
                pred_vector = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # Scheduler Step: x_t -> x_{t-1}
            # latents 更新为去噪后的结果 (对应下一个时间步)
            latents = self.noise_scheduler.step(pred_vector, t, latents, return_dict=False)[0]

            # ==========================================
            # 6. Replacement Method (核心逻辑)
            # ==========================================
            # if history_conditioning_latent is not None:
            #     # 我们刚刚完成了从 t 到 t_next 的去噪
            #     # 现在 latents 处于 t_next 时刻
            #     # 我们需要将历史部分强制替换为：Ground Truth（history latent token 前缀）
            #
            #     latents = self.replace_history_tokens(
            #         latents=latents,
            #         num_history_tokens=num_history_tokens,
            #         history_latents=padded_history_latents,
            #     )

            if process_bar:
                process_bar.update(1)

        # 替换为真实的历史帧
        if padded_history_latents is not None:
            if do_classifier_free_guidance:
                num_history_tokens = num_history_tokens[:len(num_history_tokens) // 2]

            latents = self.replace_history_tokens(
                latents=latents,
                num_history_tokens=num_history_tokens,
                history_latents=padded_history_latents,
            )

        unpack_latents = self.unpack_latents(latents, frames)

        return unpack_latents

    def freeze_grad(self) -> None:
        """根据环境变量选择性冻结参数。"""
        # 主要是冻结文本相关模态
        mmdit_freeze = os.getenv("MMDIT_FREEZE")
        if mmdit_freeze == '1':
            # 输出层可训练
            params_to_freeze_keywords = [
                'text', 'txt'
            ]
        elif mmdit_freeze == '2':
            params_to_freeze_keywords = [
                'text', 'txt', 'out'
            ]
        else:
            # 正常情况，不冻结任何参数
            params_to_freeze_keywords = []

        for name, param in self.named_parameters():
            # 在这里选择性地冻结
            if any(keyword in name for keyword in params_to_freeze_keywords):
                param.requires_grad = False

        if len(params_to_freeze_keywords) > 0:
            logging.warning(f'冻结 MMDiT 的部分参数: {params_to_freeze_keywords}')


def format_params(num: float) -> str:
    """将参数量格式化为可读字符串。

    Args:
        num (float): 参数量。

    Returns:
        str: 带单位的字符串（K/M/B）。
    """
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"  # 十亿 (Billion)
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"  # 百万 (Million)
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"  # 千 (Thousand)
    else:
        return str(num)


def get_mmdit(
        max_frames: int,
        patch_size: int,
        depth: int,
        joint_attention_dim: int,
        latent_occ_frame_shape: Tuple[int, int, int] = (28, 28, 16),
) -> MMDiT:
    """构建 MMDiT 模型实例。

    Args:
        max_frames (int): 训练/采样支持的最大帧数。
        patch_size (int): latent patch 大小。
        depth (int): transformer 层数。
        joint_attention_dim (int): 文本 embedding 维度。
        latent_occ_frame_shape (Tuple[int, int, int]): VAE latent 单帧形状 `(H_lat, W_lat, C_lat)`。

    Returns:
        MMDiT: 模型实例。
    """

    model = MMDiT(
        patch_size=patch_size,
        num_layers=depth,
        max_frames=max_frames,
        latent_occ_frame_shape=latent_occ_frame_shape,
        joint_attention_dim=joint_attention_dim,
    )
    return model


if __name__ == '__main__':
    import torch

    # os.environ['MMDIT_FREEZE'] = '1'
    ps = 2
    d = 9
    h = w = 32
    model = get_mmdit(8, ps, d, 3584)
    model.freeze_grad()

    train_params = 0
    total_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            train_params += p.numel()

    print(f"Train params: {format_params(train_params)};Total params: {format_params(total_params)}")
    print()

    encoder_hidden_states = torch.rand(4, 256, 3584)  # [bs, len, c]
    encoder_hidden_states_mask = torch.ones(4, 256).int()

    res = model.sample(
        frames=8,
        prompt_embeds=encoder_hidden_states,
        prompt_embeds_mask=encoder_hidden_states_mask,
        num_inference_steps=10,
        guidance_scale=7.5
    )
    print()
