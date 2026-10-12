import logging
import os
from os.path import dirname, abspath

__project_root__ = dirname(dirname(abspath(__file__)))

from typing import Optional, Mapping, Any, List
import copy

import torch
from einops import repeat

from model.mmdit.mmdit import get_mmdit
from occ_vae.vae import get_vae
from model.vlm_encoder.vlm_encoder import VLMEncoder
from model.bag_of_words_encoder.encoder import TEXTEncoder

from model.trainer.trainer_module import RectifiedFlowTrainer
from model.loss.iou import logits_to_label


class OccGen(RectifiedFlowTrainer):
    """OccGen 生成模型（LightningModule）。

    该模块组合了：
    - VAE（冻结）：将 occupancy label 序列编码到 latent 空间，再从 latent 解码回类别 logits。
    - VLMEncoder（冻结）：将文本 prompt 编码为文本 token embedding。
    - MMDiT（可训练）：在 latent token 序列上学习向量场（Flow Matching）并支持采样生成。

    关键张量形状约定（高频）：
    - 训练数据的 occupancy label 序列：`occ_seq` 形状为 `[F, H, W, D]`
      - `F`：帧数
      - `H, W`：BEV 平面分辨率
      - `D`：垂直方向 depth bins（例如 16）
      - 值域通常为语义类别 id（int），而非 one-hot / logits。

    - VAE latent：`latent` 形状为 `[B, C_lat, F, H_lat, W_lat]`

    - VAE 解码 logits：`logits` 形状为 `[B, num_classes, F, H, W, D]`
      通过 `loss.iou.logits_to_label` argmax 后得到 label：`[B, F, H, W, D]`。

    注意：`H/W/D` 的定义在本仓库当前实现中如上所述。
    """

    # 同时支持单帧和序列生成
    def __init__(
            self,
            lr: float,
            max_frames: int,
            seed: Optional[int] = None,
            num_inference_steps: int = 20,
            training_stage: str = 'stage1',
            use_cfg: bool = False,
            cfg_scale: float = 1.0,
            **kwargs: Any,
    ) -> None:
        """初始化 OccGen 模型。

        Args:
            lr (float): 学习率。
            max_frames (int): MMDiT 支持的最大帧数（也用于 scheduler 的 max_image_seq_len 配置）。
            seed (Optional[int]): 采样时随机种子；若同时设置环境变量 `SEED`，则以环境变量为准。
            num_inference_steps (int): 采样步数默认值（当前 `predict_step` 内部仍显式传 20）。
            training_stage (str): 训练阶段标识（用于日志/ckpt 保存策略等）。
            use_cfg (bool): 是否启用 Classifier-Free Guidance。
                注意：若设置环境变量 `CFG_SCALE`，则会在 `predict_step` 中强制启用 CFG。
            cfg_scale (float): CFG 的 guidance scale（当启用 CFG 时生效）。
            **kwargs (Any): 预留参数（当前实现未使用，保留用于与外部构造接口兼容）。
        """
        super().__init__(lr=lr)
        self.mmdit_patch_size = 1  # int: 在输入 mmdit 对 vae 的结果进行分块 (折叠) 以节省 token
        if os.getenv('MMDIT_PATCH_SIZE'):
            self.mmdit_patch_size = int(os.getenv('MMDIT_PATCH_SIZE'))

        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.training_stage = training_stage
        self.use_cfg = use_cfg
        self.cfg_scale = cfg_scale

        logging.warning(f"Initializing OccGen for [Training Stage: {self.training_stage.upper()}]")

        self.vae = get_vae(
            use_scale=True,
            ckpt_path=f'{__project_root__}/ckpt/vae/best-IoU-epoch=41-val_IoU=84.07-val_mIoU=70.61.ckpt',
        )

        if os.getenv('VLM') == 'Qwen2.5-VL-3B':
            self.text_encoder = VLMEncoder(
                text_encoder_path=f'{__project_root__}/ckpt/Qwen2.5-VL-3B-Instruct',
                tokenizer_path=f'{__project_root__}/ckpt/Qwen2.5-VL-3B-Instruct',
                max_text_prompt_length=2048,
            )
        elif os.getenv('VLM') == 'Qwen3-VL-2B':
            self.text_encoder = VLMEncoder(
                text_encoder_path=f'{__project_root__}/ckpt/Qwen3-VL-2B-Instruct',
                tokenizer_path=f'{__project_root__}/ckpt/Qwen3-VL-2B-Instruct',
                max_text_prompt_length=2048,
            )
        elif os.getenv('VLM') == 'CLIP-VIT-LARGE':
            self.text_encoder = TEXTEncoder(
                text_encoder_path=f'{__project_root__}/ckpt/clip-vit-large-patch14',
                tokenizer_path=f'{__project_root__}/ckpt/clip-vit-large-patch14',
            )
        elif os.getenv('VLM') == 'T5-XL':
            self.text_encoder = TEXTEncoder(
                text_encoder_path=f'{__project_root__}/ckpt/flan-t5-xl',
                tokenizer_path=f'{__project_root__}/ckpt/flan-t5-xl',
            )
        else:
            raise NotImplementedError

        if os.getenv('USE_DYNAMIC_CITY_DIT'):
            from model.dynamic_city_dit.dit import get_dit

            self.mmdit_weight_prefix = 'dit.'
            os.environ['ENCODER_TEXT_DIM'] = f'{self.text_encoder.text_dim}'
            self.mmdit = get_dit()
        else:
            if os.getenv('MMDIT_DEPTH'):
                depth = int(os.getenv('MMDIT_DEPTH'))
            else:
                depth = 18

            self.mmdit_weight_prefix = 'mmdit.'
            self.mmdit = get_mmdit(
                max_frames=max_frames,
                patch_size=self.mmdit_patch_size,
                depth=depth,
                joint_attention_dim=self.text_encoder.text_dim,
                latent_occ_frame_shape=self.vae.latent_occ_frame_shape
            )

        # --- 模型冻结逻辑 ---
        # VAE 始终冻结
        self.vae.freeze_grad()
        self.vae.eval()

        self.text_encoder.freeze_grad()
        self.text_encoder.eval()

        self.noise_scheduler = copy.deepcopy(self.mmdit.noise_scheduler)

    def configure_optimizers(self) -> Any:
        """根据训练阶段配置需要优化的参数。

        Returns:
            Any: Lightning 期望的 optimizer / lr_scheduler 配置。
        """
        # 无需优化 vae
        trainable_params = []

        # 设置 mmdit 的训练权重
        for param in self.mmdit.parameters():
            if param.requires_grad:
                trainable_params.append(param)

        self.trainable_params = trainable_params
        return super().configure_optimizers()

    def single_step(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """执行一次训练 step（Flow Matching loss）。

        Args:
            batch (Mapping[str, Any]): 数据集的 batch。

                期望至少包含：
                - `target_occupancy`: `List[torch.Tensor]`，长度为 `B`。

                  每个元素是一个不定长的 occupancy label 序列：
                  - `occ_seq_i`: `[F_i, H, W, D]`

                  其中 `D` 表示垂直方向 depth bins（例如 16）。
                  这里的最后一维在当前数据集中代表 depth bins（无歧义）。

                - `prompts`: `List[str]`，长度为 `B`。

                可选包含：
                - `history_num_frames`: `List[int]`，长度为 `B`。
                  用于在计算 loss 时屏蔽前若干帧。

        Returns:
            torch.Tensor: 标量 loss（0 维张量）。
        """

        # 支持在 prompt 中使用空字符串，用于 Classifier-Free Guidance (CFG) 生成
        # ex. prompt: ['prompt 1', '', 'prompt 2']
        # 对应的 target_occupancy 中的第二个为 CFG 生成

        # mmdit 内部支持处理不同长度的序列。
        # 在数据集传入的是不定长的 list of target_occupancy。
        # target_occupancy: [[F1, H, W, D], [F2, H, W, D], ...]
        # 单场景生成时 F1=F2=...=1
        # 当前数据中最后一维为 depth bins（D），无需修正维度语义说明。
        target_occupancy_list = batch['target_occupancy']
        prompt = batch['prompts']
        history_num_frames = batch.get('history_num_frames', None)

        # 使用训练中的方式, last frame 填充 Occ
        device = target_occupancy_list[0].device
        padded_occupancy, _ = self.mmdit.pad_occ_seq_list(target_occupancy_list)

        with torch.no_grad():
            prompt_embeds, prompt_embeds_mask = self.text_encoder.encode_prompt(prompt=prompt)

            # latent_occ [bs, c, f, h, w]
            latent_occ, _, _ = self.vae.encode(padded_occupancy)

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        source_noise = torch.randn_like(latent_occ, device=device)
        bsz, channel, max_frame, height, width = latent_occ.shape
        occ_shapes = [
            (occ_frame.shape[0], height // self.mmdit_patch_size, width // self.mmdit_patch_size)
            for occ_frame in target_occupancy_list
        ]

        # logit_normal
        # SD3 paper reference: https://huggingface.co/papers/2403.03206v1.
        u = torch.normal(mean=0.0, std=1.0, size=(bsz,), device='cpu')
        u = torch.nn.functional.sigmoid(u)

        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device)

        # 计算路径上的点 x_t
        noisy_model_input = self.noise_scheduler.scale_noise(
            latent_occ,
            timesteps,
            source_noise
        )  # latent_occ & noisy_model_input -> [bsz, channel, max_frame, height, width]

        # 计算目标向量场 u_t, 认为路径为直线 (反向, 同理预测时由 1 -> 0)
        target_vector_field = source_noise - latent_occ

        packed_noisy_model_input = self.mmdit.pack_latents(noisy_model_input)

        # 让模型预测向量场
        # mmdit 接受 0-1 时间索引输入，这里需要除以 1000
        timesteps = timesteps / 1000.0

        # 训练侧启用 token_replace 风格的调制（history prefix）：
        # - 只要提供 `num_history_tokens`，MMDiT 内部就会用 t=0 生成 `temb_history`，
        #   并对 image token 的前缀 `[:num_history_tokens[i]]` 使用单独的调制/门控分支（等价 Hunyuan 的 tr_*）。
        num_history_tokens = None
        if history_num_frames is not None:
            num_history_tokens = self.mmdit.get_num_history_tokens(
                padded_frame=max_frame,
                num_history_frames=[int(f) for f in history_num_frames],
                padded_history_latents=packed_noisy_model_input,
            )

        packed_pred_vector_field = self.mmdit(
            hidden_states=packed_noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            img_shapes=occ_shapes,
            txt_seq_lens=txt_seq_lens,
            num_history_tokens=num_history_tokens,
        ).sample

        pred_vector_field = self.mmdit.unpack_latents(packed_pred_vector_field, max_frame)

        seq_mask_list = [
            torch.ones(occ_shape[0], device=target_vector_field.device)
            for occ_shape in occ_shapes
        ]

        # 也要排除历史帧的部分
        if history_num_frames is not None:
            for i, frames in enumerate(history_num_frames):
                if frames > 0:
                    seq_mask_list[i][:frames] = 0

        seq_mask_mask = torch.nn.utils.rnn.pad_sequence(seq_mask_list, batch_first=True, padding_value=0).bool()
        seq_mask_mask = repeat(
            seq_mask_mask,
            'bs seq_len -> bs channel seq_len height width',
            channel=channel,
            height=height,
            width=width
        )
        # pred_vector_field[seq_mask_mask] 会将张量展平
        loss = self.loss_function(pred_vector_field[seq_mask_mask], target_vector_field[seq_mask_mask])
        return loss

    # 推理绘图
    def predict_step(
            self,
            occ_gen_frames: int,
            prompt: List[str],
            history_occ_list: Optional[List[torch.Tensor]] = None,
            cut_history: bool = False,
            restore: bool = True,
    ) -> torch.Tensor:
        """根据文本 prompt 生成 occupancy label 序列。

        Args:
            occ_gen_frames (int): 总生成帧数（包含历史帧在内）。
            prompt (List[str]): 文本 prompt 列表，长度为 `B`。
            history_occ_list (Optional[List[torch.Tensor]]): 历史 occupancy label 序列列表。
                长度为 `B`，每个元素形状为 `[F_hist_i, H, W, D]`。
                这里的 `D` 为 depth bins（例如 16）。
            cut_history (bool): 是否在返回结果中裁剪掉历史帧。
                注意：该参数为保留接口，当前版本不生效（实现为 `pass`）。如需裁剪请在外部对返回结果自行处理。
            restore (bool): 是否在推理后将 mmdit 恢复到 train()。

        Returns:
            torch.Tensor: 生成的 occupancy label。

            - 形状：`[B, F, H, W, D]`，其中 `F == occ_gen_frames`。
              注意：当前实现不会在返回中裁剪历史帧（`cut_history` 为占位参数，逻辑为 `pass`）。
            - dtype：通常为 `torch.long`（来自 `loss.iou.logits_to_label` 的 argmax）。
            - 语义：每个体素位置的类别 id（取值范围通常为 `[0, num_classes-1]`，具体由 VAE 的类别数决定）。
        """

        self.mmdit.eval()
        self.text_encoder.eval()

        with torch.no_grad():
            # 设置可复现的随机种子
            if self.seed is not None or os.getenv('SEED') is not None:
                seed = self.seed
                if os.getenv('SEED'):
                    seed = int(os.getenv('SEED'))
                generator = torch.Generator(device=self.mmdit.device).manual_seed(seed)
            else:
                generator = None

            batch_size = len(prompt)
            if self.use_cfg or os.getenv('CFG_SCALE'):
                null_prompt = [""] * batch_size
                prompt = null_prompt + prompt

            prompt_embeds, prompt_embeds_mask = self.text_encoder.encode_prompt(
                prompt=prompt,
            )

            if history_occ_list is not None:
                padded_occupancy, num_history_frames = self.mmdit.pad_occ_seq_list(history_occ_list)

                # history_conditioning_latent [bs, c, f, h, w]
                history_conditioning_latent, _, _ = self.vae.encode(padded_occupancy.to(self.mmdit.device))
            else:
                history_conditioning_latent = None
                num_history_frames = None

            # 2. 实现 Classifier-Free Guidance
            if self.use_cfg or os.getenv('CFG_SCALE'):
                # 设置 guidance_scale
                guidance_scale = self.cfg_scale
                if os.getenv('CFG_SCALE'):
                    guidance_scale = float(os.getenv('CFG_SCALE'))
            else:
                # 如果不使用 CFG，guidance_scale 设为 1.0 或 1，表示完全不引导
                guidance_scale = 1.0

            # 3. 调用 sample 函数，传入 guidance_scale
            # 注意：此处假设你的 self.mmdit.sample 函数支持 guidance_scale 参数
            latent = self.mmdit.sample(
                frames=occ_gen_frames,  # 包含历史帧在内的总长度
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                history_conditioning_latent=history_conditioning_latent,
                num_history_frames=num_history_frames,
                num_inference_steps=20,
                generator=generator,
                guidance_scale=guidance_scale,
            )

            # 4. 解码 VAE 生成图像
            logits = self.vae.decode(latent)

        # 恢复模型训练状态
        if restore:
            self.mmdit.train()

        if cut_history:
            pass

        return logits_to_label(logits)

    def on_save_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        """根据训练阶段，保存相应的模型权重。"""
        keys_to_keep = (self.mmdit_weight_prefix,)
        state_dict = checkpoint['state_dict']
        # 移除不需要保存的模块
        checkpoint['state_dict'] = {
            k: v for k, v in state_dict.items()
            if k.startswith(keys_to_keep)
        }
        logging.warning(f"Saving checkpoint for Stage {self.training_stage}. Kept keys starting with: {keys_to_keep}")

    def load_state_dict(
            self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> None:
        """加载 state_dict。

        注意：为兼容 stage1 -> stage2 的 checkpoint（例如可能缺少 text_encoder 的 LoRA 权重等），
        当前实现**固定使用** `strict=False` 调用父类方法，因此传入的 `strict` 参数会被忽略。

        Args:
            state_dict (Mapping[str, Any]): 待加载的权重字典。
            strict (bool): 保留参数，当前版本不生效。
            assign (bool): 保留参数（与上游接口对齐），当前版本不使用。
        """
        super().load_state_dict(state_dict, strict=False)


if __name__ == '__main__':
    device = 'cuda'
    # os.environ['CFG_SCALE'] = '7.5'

    # os.environ['USE_ST_ATT'] = '1'
    # os.environ['VLM'] = 'Qwen3-VL-2B'
    os.environ['MMDIT_DEPTH'] = '14'
    os.environ['MMDIT_PATCH_SIZE'] = '2'
    # os.environ['VLM'] = 'CLIP-VIT-LARGE'
    os.environ['VLM'] = 'T5-XL'
    # os.environ['USE_DYNAMIC_CITY_DIT'] = '1'

    # Stage 1 test
    model = OccGen(lr=1e-3, max_frames=32, training_stage='stage1').to(device)
    batch = {
        'target_occupancy': [
            torch.ones(6, 224, 224, 16, device=device),
            torch.ones(8, 224, 224, 16, device=device)
        ],
        'prompts': ['some prompt 111', "222"],
        # 'history_num_frames': [0, 2]
    }
    loss = model.single_step(batch)
    print()

    os.environ['CFG_SCALE'] = '5.0'
    occ = model.predict_step(
        16,
        ['some prompt 111', "222"],
        [
            torch.ones(2, 224, 224, 16, device=device),
            torch.ones(4, 224, 224, 16, device=device)
        ]
    )
    print(occ.shape)
