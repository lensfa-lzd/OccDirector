from os.path import dirname, abspath

__project_root__ = dirname(dirname(dirname(abspath(__file__))))

from typing import Optional, List, Literal

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning.utilities import rank_zero_only
from occinteract.tools.pyvista_vis import render_semantic_scene


class Occ4DRenderer:
    """
    一个独立的渲染器，用于将 4D 语义体素序列可视化。

    它接收 PyTorch 张量形式的模型输出（一个序列），
    使用 `render_semantic_scene` 函数对序列中的每一帧进行 3D 渲染，
    并返回一个包含所有渲染结果的嵌套列表。
    """

    def __init__(
            self, elev: int = 30, azim: int = 30,
            draw_ego_car=False,
    ):
        """
        初始化渲染器。

        Args:
            elev (int): 渲染视角的仰角。
            azim (int): 渲染视角的方位角。
            draw_ego_car (bool): 是否在场景中绘制自车。
        """
        self.render_kwargs = {
            'elev': elev,
            'azim': azim,
            'show': False,
            'draw_ego_car': draw_ego_car
        }

    def render_grid(self, occ: torch.Tensor) -> Optional[List[List[np.ndarray]]]:
        """
        为每个样本的每一帧生成渲染图像。

        Args:
            occ (torch.Tensor): 标签张量，形状为 (B, F, H, W, D)，
                                B 是批量大小, F 是序列帧数。

        Returns:
            Optional[List[List[np.ndarray]]]: 一个嵌套列表，结构为
                                             [[seq1_frame1, seq1_frame2, ...],
                                              [seq2_frame1, seq2_frame2, ...], ...]。
                                             如果无法生成图像则返回 None。
        """
        num_samples, num_frames = occ.shape[0], occ.shape[1]
        if num_samples == 0 or num_frames == 0:
            return None

        all_sequences = []
        for i in range(num_samples):
            sequence_images = []
            for j in range(num_frames):
                # 渲染第 i 个样本的第 j 帧
                img = render_semantic_scene(grid_data=occ[i, j].cpu().numpy(), **self.render_kwargs)
                sequence_images.append(img.astype(np.uint8))

            if sequence_images:
                all_sequences.append(sequence_images)

        if not all_sequences:
            return None

        return all_sequences


class Occ4DGenVisualizer(pl.Callback):
    """
    一个 PyTorch Lightning Callback，用于在训练过程中，
    将模型的 4D 生成结果进行可视化，并记录到 TensorBoard。
    支持图像网格和视频两种记录方式。
    """

    def __init__(
            self,
            gen_frames: int,
            sample_prompts: List[str],
            every_n_steps: int = 10,
            elev: int = 35,
            azim: int = 180,
            plot_mode: Literal['grid', 'video'] = 'grid',
            fps: int = 2,
            tag: str = ''
    ):
        """
        初始化回调。

        Args:
            sample_prompts: 用来测试的提示词。
            every_n_steps (int): 每隔多少个 step 执行一次可视化。
            elev (int): 渲染视角的仰角 (传递给渲染器)。
            azim (int): 渲染视角的方位角 (传递给渲染器)。
            plot_mode (str): 绘图方式, 'grid' 或 'video'。
            fps (int): 当 plot_mode 为 'video' 时，视频的帧率。
        """
        super().__init__()
        self.gen_frames = gen_frames
        self.every_n_steps = every_n_steps
        self.sample_prompts = sample_prompts
        self.plot_mode = plot_mode
        self.fps = fps
        self.tag = tag

        # 实例化独立的渲染器
        self.renderer = Occ4DRenderer(elev=elev, azim=azim)

    @rank_zero_only
    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
            outputs, batch, batch_idx
    ) -> None:
        """在训练批次结束后，执行4D可视化并将结果记录到 TensorBoard。"""
        if (
                trainer.sanity_checking or
                batch_idx == 0 or
                trainer.global_step == 0 or
                trainer.global_step % self.every_n_steps != 0
        ):
            return

        # 使用保存的提示词进行推理，获取最新模型输出的序列
        # 假设 predict_step 输出 (B, F, H, W, D)
        occ_sequence = pl_module.predict_step(self.gen_frames, self.sample_prompts)

        # 调用渲染器为序列的每一帧生成图像
        image_sequences = self.renderer.render_grid(occ_sequence)

        if image_sequences is None:
            return

        # 根据选择的模式生成可视化结果
        output = plot_4d_visualizations(
            self.sample_prompts,
            image_sequences,
            mode=self.plot_mode
        )

        # 记录到 TensorBoard
        writer = trainer.logger.experiment
        if self.plot_mode == 'grid' and output is not None:
            writer.add_image(
                f'Generation_occ_scenes_grid{self.tag}',
                output,
                global_step=trainer.global_step,
                dataformats='HWC'
            )
        elif self.plot_mode == 'video' and output is not None:
            # add_video 期望的格式是 (T, C, H, W)
            writer.add_video(
                f'Generation_occ_scenes_video{self.tag}',
                output[None, ...],
                global_step=trainer.global_step,
                fps=self.fps
            )

    @rank_zero_only
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """在训练 epoch 结束时被调用，执行一次可视化。"""
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        # 复用 batch_end 的逻辑
        # self.on_train_batch_end(trainer, pl_module, {}, {}, self.every_n_steps)
        pass


def _wrap_text_by_pixel(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    """
    一个辅助函数，根据渲染后的像素宽度来换行文本，完美支持中文。

    Args:
        text (str): 需要换行的原始文本。
        font (ImageFont.FreeTypeFont): Pillow 字体对象。
        max_width (int): 允许的最大像素宽度。

    Returns:
        str: 已经插入换行符'\n'的文本字符串。
    """
    lines = []

    # 如果文本本身为空或不含任何字符，直接返回
    if not text:
        return ""

    current_line = ""
    for char in text:
        if hasattr(font, 'getbbox'):  # Pillow >= 10.0.0
            test_line_bbox = font.getbbox(current_line + char)
            test_width = test_line_bbox[2] - test_line_bbox[0]
        else:
            test_width = font.getsize(current_line + char)[0]

        if test_width <= max_width:
            # 如果没超过最大宽度，就将字符添加到当前行
            current_line += char
        else:
            # 如果超过了，就将之前的行存起来，并用当前字符开始一个新行
            lines.append(current_line)
            current_line = char

    lines.append(current_line)
    return "\n".join(lines)


def _create_text_image(prompt: str, width: int, height: int, bg_color='white', font_color='black') -> np.ndarray:
    """
    为给定的提示词创建一个居中、自动换行的文本图像。
    """
    try:
        font_path = f'{__project_root__}/font/msyh.ttf'
        # 如果字体不存在，使用默认字体以避免崩溃
        font_obj = ImageFont.truetype(font_path, size=int(height / 5))
    except IOError:
        print(f"警告: 字体文件未找到 at '{font_path}'. 使用默认字体。")
        font_obj = ImageFont.load_default()

    text_canvas = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(text_canvas)

    max_pixel_width = int(0.8 * width)
    max_pixel_height = int(0.8 * height)

    font_size = int(height / 5)
    wrapped_text = ""

    # 动态调整字体大小以适应画布
    while font_size > 5:
        try:
            font_obj = ImageFont.truetype(font_path, size=font_size)
        except IOError:
            font_obj = ImageFont.load_default()

        wrapped_text = _wrap_text_by_pixel(prompt, font_obj, max_pixel_width)

        if hasattr(draw, 'textbbox'):
            bbox = draw.textbbox((0, 0), wrapped_text, font=font_obj, align='center')
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            text_w, text_h = draw.textsize(wrapped_text, font=font_obj)

        if text_w <= max_pixel_width and text_h <= max_pixel_height:
            break
        font_size -= 1

    # 居中绘制文本
    if hasattr(draw, 'textbbox'):
        bbox = draw.textbbox((0, 0), wrapped_text, font=font_obj, align='center')
        total_text_width, total_text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        total_text_width, total_text_height = draw.textsize(wrapped_text, font=font_obj)

    text_x = (width - total_text_width) / 2
    text_y = (height - total_text_height) / 2
    draw.text((text_x, text_y), wrapped_text, font=font_obj, fill=font_color, align='center')

    return np.array(text_canvas)


def plot_4d_visualizations(
        prompts: List[str],
        image_sequences: List[List[np.ndarray]],
        mode: Literal['grid', 'video'] = 'grid',
        bg_color='white',
        font_color='black',
) -> Optional[np.ndarray]:
    """
    将由提示词生成的 4D 结果（图像序列）进行可视化。

    Args:
        prompts (List[str]): 提示词列表。
        image_sequences (List[List[np.ndarray]]): 图像序列的嵌套列表。
                                                  e.g., [[s1_f1, s1_f2], [s2_f1, s2_f2]]
        mode (str): 'grid' 或 'video'。
                    'grid': 将所有帧平铺成一个大图像。
                    'video': 将结果合成为一个视频张量 (T, C, H, W)。
        bg_color (str): 文本背景色。
        font_color (str): 文本颜色。

    Returns:
        np.ndarray | None: 如果是 'grid' 模式，返回拼接后的图像 (H, W, C)；
                           如果是 'video' 模式，返回视频张量 (T, C, H, W)。
                           如果输入无效，则返回 None。
    """
    if len(prompts) != len(image_sequences) or not prompts:
        print("警告：输入列表为空或长度不匹配，无法生成可视化。")
        return None

    # 获取图像尺寸
    sample_img = image_sequences[0][0]
    h, w, c = sample_img.shape

    # 预先生成所有提示词的图像
    text_images = [_create_text_image(p, w, h, bg_color, font_color) for p in prompts]

    # --- 模式 1: 多列图像网格 ---
    if mode == 'grid':
        combined_rows = []
        for text_img, img_seq in zip(text_images, image_sequences):
            # 拼接一行：[prompt, frame1, frame2, ...]
            row_parts = [text_img] + img_seq
            row = np.hstack(row_parts)
            combined_rows.append(row)

            # 添加行间距
            pad_h, pad_w, pad_c = row.shape
            pad_row = np.zeros((5, pad_w, pad_c), dtype=row.dtype)  # 黑色间隔
            combined_rows.append(pad_row)

        if not combined_rows:
            return None

        final_image = np.vstack(combined_rows[:-1])  # 移除最后一个间隔
        return final_image

    # --- 模式 2: 合并成视频 ---
    elif mode == 'video':
        num_frames = len(image_sequences[0])
        video_frames = []

        for frame_idx in range(num_frames):
            frame_rows = []
            for sample_idx in range(len(prompts)):
                text_img = text_images[sample_idx]
                render_img = image_sequences[sample_idx][frame_idx]

                # 拼接一行：[prompt, current_frame]
                row = np.hstack([text_img, render_img])
                frame_rows.append(row)

                # 添加行间距
                pad_h, pad_w, pad_c = row.shape
                pad_row = np.zeros((5, pad_w, pad_c), dtype=row.dtype)
                frame_rows.append(pad_row)

            # 将当前视频帧的所有行拼接起来
            if not frame_rows: continue
            video_frame = np.vstack(frame_rows[:-1])
            video_frames.append(video_frame)

        if not video_frames:
            return None

        # 堆叠成 (T, H, W, C) 的张量
        video_tensor_hwc = np.stack(video_frames, axis=0)
        # 转换为 TensorBoard 期望的 (T, C, H, W)
        video_tensor_chw = video_tensor_hwc.transpose(0, 3, 1, 2)
        return video_tensor_chw

    else:
        raise ValueError(f"未知的模式: '{mode}'. 请选择 'grid' 或 'video'.")


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    def create_dummy_4d_data(num_samples: int, num_frames: int, size: int = 256):
        prompts = [
            '这是一个中文提示词，用来测试自动换行和字体显示效果。',
            '场景中有多辆车，一辆在正前方，一辆在左前方。',
            '道路上车辆密度很高，交通繁忙。',
            '一辆公交车停在道路的左侧。',
            'A simple English prompt.',
            'Another English prompt to test alignment and wrapping.'
        ]
        sequences = []
        for i in range(num_samples):
            frames = []
            base_color = np.random.randint(50, 200, size=3, dtype=np.uint8)
            for j in range(num_frames):
                # 颜色随时间变化
                color_offset = j * 20
                color = np.clip(base_color + color_offset, 0, 255).astype(np.uint8)
                image = np.full((size, size, 3), color, dtype=np.uint8)
                frames.append(image)
            sequences.append(frames)

        return prompts[:num_samples], sequences


    # --- 准备 4D 模拟数据 ---
    NUM_SAMPLES = 4
    NUM_FRAMES = 5
    IMAGE_SIZE = 256
    dummy_prompts, dummy_image_sequences = create_dummy_4d_data(
        num_samples=NUM_SAMPLES, num_frames=NUM_FRAMES, size=IMAGE_SIZE
    )

    # --- 1. 测试 'grid' 模式 ---
    print("--- 正在生成 'grid' 模式图像... ---")
    grid_image_np = plot_4d_visualizations(
        dummy_prompts,
        dummy_image_sequences,
        mode='grid'
    )
    if grid_image_np is not None:
        print(f"Grid 图像生成完毕！形状: {grid_image_np.shape}")
        # 使用 Matplotlib 显示
        plt.figure(figsize=(15, 10))
        plt.imshow(grid_image_np)
        plt.title("'grid' mode output")
        plt.axis('off')
        plt.show()
        # 保存图像
        # Image.fromarray(grid_image_np).save("result_4d_grid.png")
        # print("Grid 图像已保存为 'result_4d_grid.png'")

    print("\n" + "=" * 50 + "\n")

    # --- 2. 测试 'video' 模式 ---
    print("--- 正在生成 'video' 模式张量... ---")
    video_tensor_np = plot_4d_visualizations(
        dummy_prompts,
        dummy_image_sequences,
        mode='video'
    )
    if video_tensor_np is not None:
        # (T, C, H, W)
        print(f"Video 张量生成完毕！形状: {video_tensor_np.shape}")
        print("这个张量可以直接传递给 TensorBoard 的 add_video() 方法。")
        # 验证第一帧
        first_frame_chw = video_tensor_np[0]
        first_frame_hwc = first_frame_chw.transpose(1, 2, 0)
        plt.figure(figsize=(8, 10))
        plt.imshow(first_frame_hwc)
        plt.title("First frame of 'video' mode output")
        plt.axis('off')
        plt.show()
