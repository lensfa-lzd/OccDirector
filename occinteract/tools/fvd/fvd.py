# Derived from https://github.com/universome/fvd-comparison (I3D TorchScript FVD
# implementation) and https://github.com/cvpr2022-stylegan-v/stylegan-v
# (frechet_video_distance.py), modified for OccDirector occupancy-sequence FVD.
import math
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm


# https://github.com/universome/fvd-comparison


def load_i3d_pretrained(device: Union[str, torch.device] = torch.device('cpu')) -> torch.jit.ScriptModule:
    """加载预训练的 I3D 模型。

    Args:
        device: 模型加载的目标设备，可以是字符串或 torch.device 对象。

    Returns:
        torch.jit.ScriptModule: 加载的 I3D 模型，设置为 eval 模式。
    """
    i3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'i3d_torchscript.pt')
    print('Loading model from: %s' % filepath)
    if not os.path.exists(filepath):
        print(f"preparing for download {i3D_WEIGHTS_URL}, you can download it by yourself.")
        os.system(f"wget {i3D_WEIGHTS_URL} -O {filepath}")
    i3d = torch.jit.load(filepath).eval().to(device)
    # i3d = torch.nn.DataParallel(i3d)
    return i3d


def get_feats(
        videos: torch.Tensor,
        detector: torch.jit.ScriptModule,
        device: Union[str, torch.device],
        bs: int = 10
) -> np.ndarray:
    """使用检测器提取视频特征。

    Args:
        videos: 视频数据张量，形状为 [B, C, T, H, W]，范围为 [0, 1]。
            B 为 batch size，C 为通道数，T 为帧数，H/W 为高度/宽度。
        detector: I3D 检测器模型。
        device: 模型运行的目标设备。
        bs: 处理时的 batch size。

    Returns:
        np.ndarray: 提取的特征向量数组，形状为 [N, 400]，其中 N 为视频总数。
    """
    # videos : torch.tensor BCTHW [0, 1]
    detector_kwargs = dict(rescale=False, resize=False,
                           return_features=True)  # Return raw features before the softmax layer.
    feats = np.empty((0, 400))
    with torch.no_grad():
        for i in range((len(videos) - 1) // bs + 1):
            # feats = np.vstack([feats, detector(
            #     x=torch.stack([preprocess_single(video) for video in videos[i * bs:(i + 1) * bs]]).to(device),
            #     **detector_kwargs).detach().cpu().numpy()])

            # 1. 取出一个 Batch 的视频数据
            batch_data = videos[i * bs: (i + 1) * bs]

            # 2. 预处理并堆叠成 Tensor
            input_tensor = torch.stack([preprocess_single(v) for v in batch_data])

            # 3. 放到 GPU 上
            input_tensor = input_tensor.to(device)

            # 4. 放入模型预测
            output_tensor = detector(x=input_tensor, **detector_kwargs)

            # 5. 转回 CPU 并变成 Numpy 数组
            new_feats = output_tensor.detach().float().cpu().numpy()

            # 6. 拼接到总特征库中
            feats = np.vstack([feats, new_feats])

    return feats


def get_fvd_feats(
        videos: torch.Tensor,
        i3d: torch.jit.ScriptModule,
        device: Union[str, torch.device],
        bs: int = 10
) -> np.ndarray:
    """使用 I3D 模型提取视频的 FVD 特征。

    Args:
        videos: 视频数据张量，形状为 [B, C, T, H, W]，范围为 [0, 1]。
        i3d: I3D 预训练模型。
        device: 模型运行的目标设备。
        bs: 处理时的 batch size。

    Returns:
        np.ndarray: 提取的特征向量数组，形状为 [N, 400]，其中 N 为视频总数。
    """
    # videos in [0, 1] as torch tensor BCTHW
    # videos = [preprocess_single(video) for video in videos]
    embeddings = get_feats(videos, i3d, device, bs)
    return embeddings


def preprocess_single(
        video: torch.Tensor,
        resolution: int = 224,
        sequence_length: Optional[int] = None
) -> torch.Tensor:
    """预处理单个视频张量。

    执行时间裁剪、缩放、中心裁剪和值域归一化等操作。

    Args:
        video: 视频张量，形状为 [C, T, H, W]，范围为 [0, 1]。
            C 为通道数，T 为帧数，H/W 为高度/宽度。
        resolution: 目标分辨率，默认 224。
        sequence_length: 时间裁剪的目标帧数，None 表示不裁剪。

    Returns:
        torch.Tensor: 预处理后的视频张量，形状为 [C, T', resolution, resolution]，范围为 [-1, 1]。
            其中 T' 为裁剪后的帧数。
    """
    # video: CTHW, [0, 1]
    c, t, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:, :sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)

    # center crop
    c, t, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]

    # [0, 1] -> [-1, 1]
    video = (video - 0.5) * 2

    return video.contiguous()


"""
Copy-pasted from https://github.com/cvpr2022-stylegan-v/stylegan-v/blob/main/src/metrics/frechet_video_distance.py
"""


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算特征向量的均值和协方差矩阵。

    Args:
        feats: 特征向量数组，形状为 [N, d]，其中 N 为样本数，d 为特征维度。

    Returns:
        Tuple[np.ndarray, np.ndarray]: (mu, sigma) 二元组。
            - mu: 均值向量，形状为 [d]。
            - sigma: 协方差矩阵，形状为 [d, d]。
    """
    mu = feats.mean(axis=0)  # [d]
    sigma = np.cov(feats, rowvar=False)  # [d, d]
    return mu, sigma


def frechet_distance(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    """计算两组特征之间的 Fréchet 距离。

    该距离用于衡量生成数据与真实数据之间的相似程度。

    Args:
        feats_fake: 生成数据的特征向量数组，形状为 [N, d]。
        feats_real: 真实数据的特征向量数组，形状为 [N, d]。

    Returns:
        float: Fréchet 距离值。距离越小表示生成数据与真实数据越相似。
    """
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)
    m = np.square(mu_gen - mu_real).sum()
    if feats_fake.shape[0] > 1:
        s, _ = sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    else:
        fid = np.real(m)
    return float(fid)
