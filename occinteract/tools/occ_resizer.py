"""2D 占据网格缩放工具,支持 PyTorch 和 NumPy。"""
from typing import Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def occ_resize(
        tensor: Union[torch.Tensor, np.ndarray],
        output_size: Tuple[int, int]
) -> Union[torch.Tensor, np.ndarray]:
    """通用 2D 图像/特征图缩放函数,支持 PyTorch 和 NumPy。

    使用最近邻插值进行缩放,适用于语义标签等离散值数据。
    支持 [bs, H, W, C] 和 [H, W, C] 两种输入形状,自动处理维度转换。

    Args:
        tensor (Union[torch.Tensor, np.ndarray]): 输入张量/数组。
            形状可以是 [bs, H, W, C] 或 [H, W, C]
        output_size (Tuple[int, int]): 目标输出尺寸 (H, W)

    Returns:
        Union[torch.Tensor, np.ndarray]: 缩放后的张量/数组。
            输出类型和输入类型保持一致。
            如果输入是 [H, W, C],输出也是 [H, W, C]

    Raises:
        ValueError: 如果输入张量/数组的维度不是 3 或 4
    """
    # 1. 自动判断输入类型，并统一转换为 PyTorch 张量
    is_numpy = isinstance(tensor, np.ndarray)
    if is_numpy:
        # from_numpy 共享内存，效率高
        input_tensor = torch.from_numpy(tensor)
    else:
        input_tensor = tensor

    # 2. 自动判断输入形状，统一为 4 维 [bs, H, W, C]
    if input_tensor.ndim == 3:
        # 输入是 [H, W, C]，增加一个临时的批次维度
        was_3d = True
        input_tensor_4d = input_tensor.unsqueeze(0)
    elif input_tensor.ndim == 4:
        # 输入已经是 [bs, H, W, C]
        was_3d = False
        input_tensor_4d = input_tensor
    else:
        raise ValueError(
            f"输入张量/数组需要 3 或 4 个维度，但得到了 {input_tensor.ndim} 维。"
        )

    # 3. 核心缩放逻辑 (与原函数基本一致)
    # 3.1. 保存原始数据类型
    original_dtype = input_tensor_4d.dtype

    # 3.2. 从 [bs, H, W, C] (channels-last) 变换到 [bs, C, H, W] (channels-first)
    tensor_chw = rearrange(input_tensor_4d, 'b h w c -> b c h w').float()

    # 3.3. 使用最近邻插值进行缩放
    resized_tensor_chw = F.interpolate(
        tensor_chw,
        size=output_size,
        mode='nearest'
    )

    # 3.4. 变换回 [bs, H, W, C] (channels-last) 的原始格式
    resized_tensor = rearrange(resized_tensor_chw, 'b c h w -> b h w c')

    # 3.5. 转换回原始数据类型
    resized_tensor = resized_tensor.to(original_dtype)

    # 4. 根据输入形状恢复输出形状
    if was_3d:
        # 如果原始输入是 3D，则移除临时的批次维度
        output_tensor = resized_tensor.squeeze(0)
    else:
        output_tensor = resized_tensor

    # 5. 如果原始输入是 NumPy，将结果转换回 NumPy 数组
    if is_numpy:
        # .cpu() 是为了防止张量在GPU上，导致无法转换为numpy
        return output_tensor.cpu().numpy()
    else:
        return output_tensor
