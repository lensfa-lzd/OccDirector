import numpy as np
import torch


##############################################
#  SIMPLE BINARY IOU FOR MONOLABEL GRIDS
# Absorb from https://github.com/tasl-lab/uniocc
# uniocc_eval.py
##############################################

def ComputeIoU(occ_gt, occ_pred, free_label=10):
    """
    Compute IoU for two occupancy grids using PyTorch tensors.

    Parameters
    ----------
    occ_gt : torch.Tensor     Ground truth occupancy.
    occ_pred : torch.Tensor   Predicted occupancy.
    free_label: int           Label of the voxels that should not be counted in the IoU. (usually 10 for free space)

    Returns
    -------
    float
        Intersection-over-union for the single-class occupancy.
    """
    # 使用 torch.logical_and/or 替代 np.logical_and/or
    # 使用 torch.sum 替代 np.sum
    intersection = torch.sum(torch.logical_and(occ_gt != free_label, occ_pred != free_label))
    union = torch.sum(torch.logical_or(occ_gt != free_label, occ_pred != free_label))

    # torch.sum 返回的是一个0维张量，需要用 .item() 提取其值
    # 或者直接在张量上进行运算，最后再 .item()
    if union == 0:
        return 1.0
    else:
        return (intersection.float() / union.float()).item()


def ComputeIoUForCategory(occ_gt, occ_pred, category_id):
    """
    Similar to above but just compute IoU for a specific category in the occupancy grid using PyTorch tensors.

    Parameters
    ----------
    occ_gt : torch.Tensor     Ground truth occupancy.
    occ_pred : torch.Tensor   Predicted occupancy.
    category_id: int          The category ID to compute IoU for.

    Returns
    -------
    float
        Intersection-over-union for the specified category.
    """
    intersection = torch.sum(torch.logical_and(occ_gt == category_id, occ_pred == category_id))
    union = torch.sum(torch.logical_or(occ_gt == category_id, occ_pred == category_id))

    if union == 0:
        return 1.0
    else:
        return (intersection.float() / union.float()).item()


##############################################

def ComputeMeanIoU(occ_gt, occ_pred, ignore_labels=None):
    """
    Computes the mean Intersection-over-Union (mIoU) for semantic occupancy grids using PyTorch tensors.

    The mIoU is the average of the IoU scores for each semantic class present in
    either the ground truth or the prediction, ignoring specified labels.

    Parameters
    ----------
    occ_gt : torch.Tensor
        Ground truth semantic occupancy grid.
    occ_pred : torch.Tensor
        Predicted semantic occupancy grid.
    ignore_labels : list[int], optional
        A list of class IDs to ignore in the mIoU calculation.
        Defaults to [10], which often represents free space.

    Returns
    -------
    float
        The mean IoU score.
    """
    if ignore_labels is None:
        ignore_labels = [10]

    # 1. 找出唯一类别标签
    # 使用 torch.cat 和 torch.unique 替代 numpy 的版本
    unique_labels_tensor = torch.unique(torch.cat((occ_gt.flatten(), occ_pred.flatten())))
    # 将张量中的标签转换为 Python int 列表
    unique_labels = [label.item() for label in unique_labels_tensor]

    # 2. 遍历每个类别，计算其 IoU
    iou_scores = []
    for label in unique_labels:
        if label in ignore_labels:
            continue

        # 调用基于 PyTorch 的 IoU 计算函数
        iou = ComputeIoUForCategory(occ_gt, occ_pred, label)
        iou_scores.append(iou)

    # 3. 计算所有有效类别 IoU 分数的平均值
    if not iou_scores:
        return 1.0

    # iou_scores 是一个 float 列表, 可以用 np.mean 或纯 python 计算
    # return sum(iou_scores) / len(iou_scores)
    return np.mean(iou_scores)


def logist_to_label(logist):
    """
    将模型的 logit 输出 (PyTorch Tensor) 转换为类别标签。

    参数:
    logist (torch.Tensor): 一个 torch 张量，形状为 (bs, num_cls, d1, d2, ..., dn)。

    返回:
    torch.Tensor: 一个形状为 (bs, d1, d2, ..., dn) 的长整型张量 (LongTensor)，
                  其中每个元素的值是在 [0, num_valid_cls-1] 范围内预测的类别索引。
    """
    # 使用 torch.argmax，注意参数是 dim 而不是 axis
    labels = torch.argmax(logist, dim=1)

    return labels


# --- 示例用法 ---
if __name__ == '__main__':
    data = torch.rand([1, 16, 256, 256, 16])
    logist_to_label(data)

    # 创建一个简单的 3x3 的 ground truth 和 prediction
    # 类别 0, 1 是物体, 10 是 free space
    occ_gt = torch.tensor([
        [0, 0, 10],
        [1, 1, 10],
        [10, 10, 10]
    ])

    occ_pred = torch.tensor([
        [0, 10, 10],
        [1, 1, 10],
        [0, 10, 10]  # 预测错误，将 free space 预测为 class 0
    ])

    # 评估的类别将是 0 和 1 (因为 10 会被忽略)

    # IoU for class 0:
    #   Intersection = 1 (左上角)
    #   Union = 3 (左上角, 左下角, 中上角)
    #   IoU_0 = 1 / 3
    iou_0 = ComputeIoUForCategory(occ_gt, occ_pred, 0)
    print(f"IoU for class 0: {iou_0:.4f}")

    # IoU for class 1:
    #   Intersection = 2 (中间行)
    #   Union = 2 (中间行)
    #   IoU_1 = 2 / 2 = 1.0
    iou_1 = ComputeIoUForCategory(occ_gt, occ_pred, 1)
    print(f"IoU for class 1: {iou_1:.4f}")

    # Mean IoU (mIoU)
    # mIoU = (IoU_0 + IoU_1) / 2
    # mIoU = (1/3 + 1.0) / 2 = (4/3) / 2 = 2/3 ≈ 0.6667
    mean_iou = ComputeMeanIoU(occ_gt, occ_pred)
    print(f"Mean IoU (mIoU): {mean_iou:.4f}")

    # 与手动计算结果进行比较
    expected_miou = (1 / 3 + 1.0) / 2
    print(f"Expected mIoU: {expected_miou:.4f}")
    assert np.isclose(mean_iou, expected_miou)

    # 设定参数
    bs = 2  # 批处理大小
    num_cls = 4  # 类别数量
    H, W = 3, 3  # 图像高度和宽度

    # 随机生成一个模拟的 logit 输出
    # 形状为 (2, 4, 3, 3)
    mock_logits = torch.rand(bs, num_cls, H, W)

    print("输入的 Logits 形状:", mock_logits.shape)
    # print("第一个图像的 Logits:\n", mock_logits[0])

    # 调用函数转换
    predicted_labels = logist_to_label(mock_logits)

    print("\n输出的 Labels 形状:", predicted_labels.shape)
    print("第一个图像的预测标签图:\n", predicted_labels[0])
    print("\n第二个图像的预测标签图:\n", predicted_labels[1])

    # 验证一下第一个像素 (0,0) 的结果
    # 获取第一个图像 (0,0) 像素在所有类别上的分数
    pixel_0_0_scores = mock_logits[0, :, 0, 0]
    print(f"\n第一个图像(0,0)像素在4个类别上的分数: {pixel_0_0_scores}")

    # 手动计算 argmax
    manual_argmax = np.argmax(pixel_0_0_scores)
    print(f"手动计算的(0,0)像素的类别: {manual_argmax}")

    # 从函数输出中获取(0,0)像素的类别
    function_output_label = predicted_labels[0, 0, 0]
    print(f"函数输出的(0,0)像素的类别: {function_output_label}")

    assert manual_argmax == function_output_label
    print("\n验证成功！")
