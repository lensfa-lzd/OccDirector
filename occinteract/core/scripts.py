import base64
import json
import os
from typing import List, Literal, Optional, Union

import cv2
import numpy as np
from json_repair import repair_json
from matplotlib import pyplot as plt


def show_cv2_img(img_bgr: np.ndarray, axis: Optional[Literal['on', 'off']] = None) -> None:
    """显示 BGR 格式的图像。

    使用 matplotlib 显示图像，会自动转换为 RGB 格式。

    Args:
        img_bgr: BGR 格式的图像数组，形状为 [H, W, 3]。
        axis: 坐标轴显示模式，可选 'on' 或 'off'。None 表示使用默认行为。
            如果设置环境变量 MAX_SIZE，会自动调整显示尺寸。
    """
    if os.getenv('MAX_SIZE'):
        h, w = img_bgr.shape[:2]
        factor = int(os.getenv('MAX_SIZE')) / max(h, w)
        plt.figure(figsize=(int(w * factor), int(h * factor)))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    if axis:
        plt.axis(axis)
    plt.show()


def encode_image_to_base64(image: np.ndarray) -> str:
    """将图像编码为 base64 字符串。

    Args:
        image: BGR 格式的图像数组。

    Returns:
        str: JPEG 格式的 base64 编码字符串。
    """
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


def create_grid_image(
        images_list: List[np.ndarray],
        rows: int = 3,
        cols: int = 3
) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
    """将一组图片拼成一个网格大图。

    如果图片不够填满网格，用黑色填充。如果图片超过网格容量，
    则返回多个网格图像。

    Args:
        images_list: BGR 格式图像列表，每张图像形状为 [H, W, C]。
        rows: 网格的行数。
        cols: 网格的列数。

    Returns:
        Optional[Union[np.ndarray, List[np.ndarray]]]: 单个网格图像或网格图像列表。
            单个网格形状为 [H * rows, W * cols, C]。如果输入为空返回 None。
    """

    def create_single_grid(images: List[np.ndarray]) -> np.ndarray:
        """将一组图片拼成单个网格。

        Args:
            images: 图片列表，每张形状为 [H, W, C]。

        Returns:
            np.ndarray: 拼接后的网格图像，形状为 [H * rows, W * cols, C]。
        """
        # 创建黑色背景大图
        h, w, c = images[0].shape
        grid_h = h * rows
        grid_w = w * cols

        grid_canvas = np.zeros((grid_h, grid_w, c), dtype=np.uint8)

        for idx, img in enumerate(images):
            if idx >= rows * cols:
                break

            r = idx // cols
            c_idx = idx % cols

            y_start = r * h
            y_end = y_start + h
            x_start = c_idx * w
            x_end = x_start + w

            grid_canvas[y_start:y_end, x_start:x_end] = img

        return grid_canvas

    if not images_list:
        return None

    if len(images_list) <= rows * cols:
        return create_single_grid(images_list)
    else:
        grids = []
        for index in range(len(images_list))[::rows * cols]:
            img_split = images_list[index: index + rows * cols]
            grids.append(create_single_grid(img_split))

        return grids


def cover_model_output(output: str) -> dict:
    """解析模型输出，分离思考内容和结果。

    支持 Qwen 等模型带思考标签的输出格式。

    Args:
        output: 模型输出的原始字符串。

    Returns:
        dict: 解析后的结果字典，包含 '_model_thinking_content' 键存储思考内容。
    """
    # Qwen3
    if '</think>' in output:
        thinking_content, result = output.split('</think>', maxsplit=1)
    else:
        thinking_content = None
        result = output

    result_dict = json.loads(repair_json(result))
    result_dict['_model_thinking_content'] = thinking_content
    return result_dict



def extract_frames_to_numpy(
        video_path: str,
        start_frame: int = 0,
        end_frame: int = -1
) -> List[np.ndarray]:
    """读取视频文件并将指定范围的帧转换为 Numpy 数组。

    Args:
        video_path: 视频文件的路径。
        start_frame: 起始帧索引（包含）。
        end_frame: 结束帧索引（不包含），-1 表示到视频末尾。

    Returns:
        List[np.ndarray]: BGR 格式的帧列表，每帧形状为 [H, W, 3]。

    Raises:
        FileNotFoundError: 当文件不存在时。
        IOError: 当无法打开视频文件时。
    """

    # 检查文件是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"找不到文件: {video_path}")

    # 打开视频
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")

    # 获取视频总帧数以进行基本的边界检查
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame >= total_frames:
        cap.release()
        print(f"警告: 起始帧 {start_frame} 超出了视频总帧数 {total_frames}")
        return np.array([])

    if end_frame == -1:
        end_frame = total_frames + 1

    # 设置读取位置到起始帧 (Seek操作，速度快)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_list = []
    current_frame_idx = start_frame

    try:
        while current_frame_idx < end_frame:
            ret, frame = cap.read()

            # ret 为 False 表示无法读取帧（例如视频已结束）
            if not ret:
                break

            frames_list.append(frame)

            current_frame_idx += 1

    finally:
        # 确保资源被释放
        cap.release()

    return frames_list


