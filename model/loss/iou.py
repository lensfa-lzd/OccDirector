from typing import Optional, Sequence

import torch


##############################################
#  SIMPLE BINARY IOU FOR MONOLABEL GRIDS
# Absorb from https://github.com/tasl-lab/uniocc
# uniocc_eval.py
##############################################

def compute_iou(occ_gt: torch.Tensor, occ_pred: torch.Tensor, free_label: int = 10) -> float:
    """计算两张 occupancy grid 的 IoU。

    Args:
        occ_gt (torch.Tensor): Ground truth occupancy。
            张量形状不做限制，但需要与 ``occ_pred`` 形状一致（或可 broadcast），以保证逐元素比较含义正确。
        occ_pred (torch.Tensor): Predicted occupancy。
            张量形状不做限制，但需要与 ``occ_gt`` 形状一致（或可 broadcast），以保证逐元素比较含义正确。
        free_label (int): 不计入 IoU 的 voxel 标签（通常 10 表示 free space）。
            ``occ_gt`` / ``occ_pred`` 通常是语义类别 ID（整型标签），需要能与 ``free_label`` 做逐元素比较。

    Returns:
        float: 单类别 occupancy 的 Intersection-over-Union。
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


def compute_iou_for_category(
        occ_gt: torch.Tensor,
        occ_pred: torch.Tensor,
        category_id: int,
) -> float:
    """计算某个语义类别在 occupancy grid 上的 IoU。

    Args:
        occ_gt (torch.Tensor): Ground truth occupancy。
            张量形状不做限制，但需要与 ``occ_pred`` 形状一致（或可 broadcast），以保证逐元素比较含义正确。
        occ_pred (torch.Tensor): Predicted occupancy。
            张量形状不做限制，但需要与 ``occ_gt`` 形状一致（或可 broadcast），以保证逐元素比较含义正确。
        category_id (int): 需要计算 IoU 的语义类别 ID。
            ``occ_gt`` / ``occ_pred`` 通常是语义类别 ID（整型标签），需要能与 ``category_id`` 做逐元素比较。

    Returns:
        float: 指定类别的 Intersection-over-Union。
    """
    intersection = torch.sum(torch.logical_and(occ_gt == category_id, occ_pred == category_id))
    union = torch.sum(torch.logical_or(occ_gt == category_id, occ_pred == category_id))

    if union == 0:
        return 1.0
    else:
        return (intersection.float() / union.float()).item()


##############################################

def compute_mean_iou(
        occ_gt: torch.Tensor,
        occ_pred: torch.Tensor,
        ignore_labels: Optional[Sequence[int]] = None,
) -> float:
    """计算语义 occupancy 的 mean IoU（mIoU）。

    mIoU 是对所有有效语义类别（在 ground truth 或 prediction 中出现）分别计算 IoU 后取平均，
    并忽略指定标签（例如 free space）。

    Args:
        occ_gt (torch.Tensor): Ground truth semantic occupancy。
            张量形状不做限制，但需要与 ``occ_pred`` 形状一致（或可 broadcast），以保证逐元素比较含义正确。
        occ_pred (torch.Tensor): Predicted semantic occupancy。
            张量形状不做限制，但需要与 ``occ_gt`` 形状一致（或可 broadcast），以保证逐元素比较含义正确。
        ignore_labels (Optional[Sequence[int]]): mIoU 计算时需要忽略的类别 ID 列表。
            ``occ_gt`` / ``occ_pred`` 通常是语义类别 ID（整型标签）。默认（None）时等价于 [10]
            （常用来表示 free space）。

    Returns:
        float: mean IoU 分数。
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
        iou = compute_iou_for_category(occ_gt, occ_pred, label)
        iou_scores.append(iou)

    # 3. 计算所有有效类别 IoU 分数的平均值
    if not iou_scores:
        return 1.0

    # iou_scores 是一个 float 列表，这里使用纯 Python 计算均值，保证返回类型为 float。
    return sum(iou_scores) / len(iou_scores)


def logits_to_label(logits: torch.Tensor) -> torch.Tensor:
    """将模型输出的 logit 张量转换为类别标签。

    Args:
        logits (torch.Tensor): 模型的 logit 输出。
            形状为 (bs, num_cls, d1, d2, ..., dn)。

    Returns:
        torch.Tensor: 形状为 (bs, d1, d2, ..., dn) 的 LongTensor，元素取值范围为
            [0, num_cls - 1]。
    """
    # 使用 torch.argmax，注意参数是 dim 而不是 axis
    labels = torch.argmax(logits, dim=1)

    return labels


# --- 示例用法 ---
if __name__ == '__main__':
    data = torch.rand([1, 16, 256, 256, 16])
    logits_to_label(data)

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
    iou_0 = compute_iou_for_category(occ_gt, occ_pred, 0)
    print(f"IoU for class 0: {iou_0:.4f}")

    # IoU for class 1:
    #   Intersection = 2 (中间行)
    #   Union = 2 (中间行)
    #   IoU_1 = 2 / 2 = 1.0
    iou_1 = compute_iou_for_category(occ_gt, occ_pred, 1)
    print(f"IoU for class 1: {iou_1:.4f}")

    # Mean IoU (mIoU)
    # mIoU = (IoU_0 + IoU_1) / 2
    # mIoU = (1/3 + 1.0) / 2 = (4/3) / 2 = 2/3 ≈ 0.6667
    mean_iou = compute_mean_iou(occ_gt, occ_pred)
    print(f"Mean IoU (mIoU): {mean_iou:.4f}")

    # 与手动计算结果进行比较
    expected_miou = (1 / 3 + 1.0) / 2
    print(f"Expected mIoU: {expected_miou:.4f}")
    assert abs(mean_iou - expected_miou) < 1e-6

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
    predicted_labels = logits_to_label(mock_logits)

    print("\n输出的 Labels 形状:", predicted_labels.shape)
    print("第一个图像的预测标签图:\n", predicted_labels[0])
    print("\n第二个图像的预测标签图:\n", predicted_labels[1])

    # 验证一下第一个像素 (0,0) 的结果
    # 获取第一个图像 (0,0) 像素在所有类别上的分数
    pixel_0_0_scores = mock_logits[0, :, 0, 0]
    print(f"\n第一个图像(0,0)像素在4个类别上的分数: {pixel_0_0_scores}")

    # 手动计算 argmax
    manual_argmax = torch.argmax(pixel_0_0_scores).item()
    print(f"手动计算的(0,0)像素的类别: {manual_argmax}")

    # 从函数输出中获取(0,0)像素的类别
    function_output_label = predicted_labels[0, 0, 0]
    print(f"函数输出的(0,0)像素的类别: {function_output_label}")

    assert manual_argmax == function_output_label
    print("\n验证成功！")
