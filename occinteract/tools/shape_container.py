# Modified from https://github.com/UMich-CURLY/3DMapping/blob/main/Data/Generation/ShapeContainer.py
import numpy as np
import torch


class ShapeContainerGPU:
    def __init__(
            self,
            grid_size,
            min_bound=np.array([0, -40, -3]),
            max_bound=np.array([70, 40, 3]),
            num_classes=25,
            device='cuda'
    ):
        """
        初始化 GPU 占据栅格处理器 (Cartesian 坐标系)
        """
        self.device = torch.device(device)
        self.num_classes = num_classes

        # 将配置参数转为 GPU Tensor，避免后续计算反复传输
        self.grid_size = torch.tensor(grid_size, device=self.device, dtype=torch.long)
        self.min_bound = torch.tensor(min_bound, device=self.device, dtype=torch.float32)
        self.max_bound = torch.tensor(max_bound, device=self.device, dtype=torch.float32)

        # 预计算网格分辨率
        self.intervals = (self.max_bound - self.min_bound) / self.grid_size.float()

        # 定义 Voxel Grid Tensor
        # Shape: (X, Y, Z, Num_Classes)
        self.grid_dims = list(grid_size) + [num_classes]
        self.voxels = None  # 在每一帧开始处理时重置

    def reset_grid(self):
        """重置网格，显存清零"""
        self.voxels = torch.zeros(self.grid_dims, device=self.device, dtype=torch.float32)

    def ray_trace_batch(self, points: torch.Tensor, labels: torch.Tensor, sample_spacing: float = 1.5):
        """
        使用 PyTorch CUDA 重构的 ray_trace_batch 函数。

        Args:
            points: (N, 3) FloatTensor, 位于 CUDA 设备上
            labels: (N, ) LongTensor, 位于 CUDA 设备上
            sample_spacing: 采样间距

        Returns:
            good_samples: (M, 3) FloatTensor
            good_labels: (M, 1) LongTensor
        """

        # --- 1. 数据过滤 (Filtering) ---
        # 移除 label 为 0 的点
        valid_mask = labels != 0
        points = points[valid_mask]  # (N', 3)
        labels = labels[valid_mask]  # (N',)

        # --- 2. 几何计算 (Geometry) ---
        # 计算每个点到原点的欧几里得距离 (N', 1)
        # 对应原代码: vec_norms
        r = torch.norm(points, dim=1, keepdim=True)

        # 计算归一化方向向量 (N', 3)
        # 添加 eps 防止除以 0 (虽然理论上 r>0，但为了数值稳定性)
        vec_angles = points / (r + 1e-8)

        # --- 3. 生成采样步长 (Step Generation) ---
        # 生成从 0 到 100 的标量步长，形状调整为 (1, S) 以便广播
        # 对应原代码: difs (部分逻辑)
        steps = torch.arange(0.0, 100.0, sample_spacing, device=self.device, dtype=points.dtype)
        steps = steps.view(1, -1)  # (1, S)

        # --- 4. 生成新样本 (Sample Computation) ---
        # 计算新的采样点坐标
        # 公式: Sample = Point - (Direction * Step)
        # 形状变化:
        # points:       (N', 1, 3)
        # vec_angles:   (N', 1, 3)
        # steps:        (1,  S, 1)
        # result:       (N', S, 3)

        # 计算所需的位移向量 (N', S, 3)
        difs = vec_angles.unsqueeze(1) * steps.unsqueeze(2)

        # 得到所有潜在的采样点 (N', S, 3)
        new_samples = points.unsqueeze(1) - difs

        # --- 5. 标签生成 (Label Creation) ---
        # 初始化标签为 0 (Free space)
        # 形状: (N', S)
        new_labels = torch.zeros((new_samples.shape[0], new_samples.shape[1]), device=self.device, dtype=labels.dtype)

        # 将 step=0 (即索引 0) 的位置设为原始标签
        # 因为 step=0 时，new_sample 就是原始点本身
        new_labels[:, 0] = labels

        # --- 6. 有效性筛选 (Validity Masking) ---
        # 原代码逻辑是检查生成的点是否还在原点和原始点之间。
        # 简单的做法是：只要 步长(step) 小于 原始点距离(r) 即可。
        # 这种比较比原代码的坐标除法更稳定且快。
        # Mask shape: (N', S)
        mask_valid_dist = steps < r

        # --- 7. 扁平化与输出 (Flatten & Return) ---
        # 利用 mask 提取有效点，自动 Flatten
        good_samples = new_samples[mask_valid_dist]  # (M, 3)
        good_labels = new_labels[mask_valid_dist]  # (M, )

        return good_samples, good_labels.unsqueeze(1)

    def add_points_to_grid(self, points, labels):
        """
        将点云累加到 Voxel Grid (GPU操作)
        """
        # 将点和标签组合
        pointlabels = torch.cat((points, labels), dim=1)

        # 核心优化：直接调用内部的累加方法，而不是 getitem+setitem
        # 这样做可以利用 CUDA 的原子操作避免多点落入同一体素时的冲突，也更快
        self.accumulate_voxels(pointlabels, 1.0)

    def accumulate_voxels(self, input_xyzl, value=1.0):
        """
        专门用于累加计数的优化方法。
        使用 index_put_(accumulate=True) 来处理并行写入冲突。
        """
        # 获取合法索引 (N, 4) -> [x_idx, y_idx, z_idx, ch_idx]
        grid_indices = self.grid_ind(input_xyzl)

        if grid_indices.shape[0] == 0:
            return

        # 准备高级索引 (Advaced Indexing)
        # index_put 需要索引是 tuple of tensors: (Tensor_dim0, Tensor_dim1, ...)
        indices = (
            grid_indices[:, 0],  # dim X
            grid_indices[:, 1],  # dim Y
            grid_indices[:, 2],  # dim Z
            grid_indices[:, 3]  # dim Channel
        )

        # 如果 value 是标量，扩展成 tensor
        values = torch.tensor(value, device=self.device, dtype=self.voxels.dtype)

        # 核心 CUDA 加速操作：
        # accumulate=True 保证如果多个索引指向同一个位置，值会累加而不是覆盖
        self.voxels.index_put_(indices, values, accumulate=True)

    def grid_ind(self, input_xyzl):
        """
        计算每个坐标在 grid 中的索引
        Returns: (M, 4) tensor (indices, label), M <= N (过滤了越界点)
        """
        # 确保输入维度
        if input_xyzl.dim() == 1:
            input_xyzl = input_xyzl.unsqueeze(0)

        input_xyz = input_xyzl[:, 0:3]
        labels = input_xyzl[:, 3:4]  # Keep dim as (N, 1)

        # 坐标转换：笛卡尔 -> 柱面
        xyz_pol = input_xyz

        # 过滤：检查是否在边界内
        # mask shape: (N,)
        valid_input_mask = torch.all(
            (xyz_pol < self.max_bound) & (xyz_pol >= self.min_bound), dim=1)

        valid_xyz_pol = xyz_pol[valid_input_mask]
        valid_labels = labels[valid_input_mask]

        # 计算索引： (value - min) / interval
        # 使用 floor 向下取整
        grid_ind = torch.floor((valid_xyz_pol - self.min_bound) / self.intervals).long()

        # Clipping (防止浮点精度误差导致的越界)
        max_limit = (self.grid_size - 1).long()
        min_limit = torch.zeros_like(max_limit)

        grid_ind = torch.max(torch.min(grid_ind, max_limit), min_limit)

        # 拼接标签索引
        # valid_labels 需要转换为 long 以作为索引
        grid_ind = torch.cat((grid_ind, valid_labels.long()), dim=1)

        return grid_ind

    def get_final_labels(self):
        """
        获取 Argmax 后的 Label 索引
        """
        # Argmax along channel dimension (3)
        # Shape: (X, Y, Z, C) -> (X, Y, Z)
        return torch.argmax(self.voxels, dim=3)
