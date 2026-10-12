import logging
import os
import pickle
from typing import List, Dict, Tuple, Optional

from occinteract.type_models import CaptionInstance


def load_bev_instance_caption(
        dataset_path: str,
        dataset_setting: List[Tuple[str, bool]],
) -> Dict[str, Dict[str, List[str]]]:
    """加载 BEV（Bird's Eye View）视角的实例级 Captions。

    从 bev_caption.pkl 文件中提取不同类型的 Caption，包括：
    - 非结构化内容描述
    - 第一段内容（场景全局描述）
    - 第一段首句（场景全局摘要）
    - 最后一段内容（车流量描述）
    - 最后一段首句（车流量摘要）

    Args:
        dataset_path: 数据集根目录路径。
        dataset_setting: 数据集设置列表，每个元素是 (数据集名称, 是否有测试集) 的元组。

    Returns:
        Dict[str, Dict[str, List[str]]]: 按照 {数据集名: {场景名: [captions]}} 嵌套组织的 Caption 字典。
    """
    load_keys = [
        # ("full", "content"),  # 由非结构化的版本替代
        ('unstructured', "content"),  # 由非结构化的版本替代
        ("first_paragraph", "content"),  # 场景的全局描述
        ("first_paragraph", "first_sentence"),  # 一句话摘要, 场景的全局描述
        ("last_paragraph", "content"),  # 场景的车流量描述
        ("last_paragraph", "first_sentence"),  # 一句话摘要, 场景的车流量描述
    ]
    with open(f'{dataset_path}/bev_caption.pkl', 'rb') as f:
        bev_caption = pickle.load(f)

    parsed_bev_caption = {}
    for dataset, has_test_set in dataset_setting:
        parsed_bev_caption[dataset] = {}
        for l1_key, l2_key in load_keys:
            caption_dict = bev_caption[dataset][l1_key][l2_key]
            for scene_name, caption in caption_dict.items():
                if scene_name not in parsed_bev_caption[dataset]:
                    parsed_bev_caption[dataset][scene_name] = []
                parsed_bev_caption[dataset][scene_name].append(caption)

    return parsed_bev_caption


def load_nuscene_extra_caption(dataset_path: str) -> Dict[str, Dict[str, List[str]]]:
    """加载 NuScenes 数据集的额外 Captions。

    Args:
        dataset_path: 数据集根目录路径。

    Returns:
        Dict[str, Dict[str, List[str]]]: 额外 Caption 字典，按照 {数据集名: {场景名: [captions]}} 组织。
    """
    with open(f'{dataset_path}/extra_caption.pkl', 'rb') as f:
        extra_caption = pickle.load(f)
    return extra_caption


def load_nuscene_location_caption(dataset_path: str) -> Dict[str, Dict[str, List[str]]]:
    """加载 NuScenes 数据集的位置 Captions。

    Args:
        dataset_path: 数据集根目录路径。

    Returns:
        Dict[str, Dict[str, List[str]]]: 位置 Caption 字典，按照 {数据集名: {场景名: [captions]}} 组织。
    """
    with open(f'{dataset_path}/location_caption.pkl', 'rb') as f:
        location_caption = pickle.load(f)
    return location_caption


def load_merge_captions(
        dataset_path: str,
        dataset_setting: List[Tuple[str, bool]],
) -> Dict[str, Dict[str, List[str]]]:
    """合并加载所有类型的 Captions。

    合并以下三种 Caption：
    - BEV 实例级 Captions
    - NuScenes 额外 Captions
    - NuScenes 位置 Captions

    Args:
        dataset_path: 数据集根目录路径。
        dataset_setting: 数据集设置列表，每个元素是 (数据集名称, 是否有测试集) 的元组。

    Returns:
        Dict[str, Dict[str, List[str]]]: 合并后的 Caption 字典，按照 {数据集名: {场景名: [captions]}} 组织。
    """
    loaded_caption = [
        load_bev_instance_caption(dataset_path, dataset_setting),
        load_nuscene_extra_caption(dataset_path),
        load_nuscene_location_caption(dataset_path),
    ]

    final_caption = {}
    for caption_dict in loaded_caption:
        for dataset, mappings in caption_dict.items():
            if dataset not in final_caption:
                final_caption[dataset] = {}

            for scene, caption in mappings.items():
                if scene not in final_caption[dataset]:
                    final_caption[dataset][scene] = []

                if isinstance(caption, list):
                    final_caption[dataset][scene] += caption
                else:
                    final_caption[dataset][scene].append(caption)

    # 输出统计信息
    for dataset in final_caption:
        count_caption = 0
        for scene in final_caption[dataset]:
            count_caption += len(final_caption[dataset][scene])

        logging.warning(f"load_merge_captions: 数据集 {dataset} 加载了 {count_caption:,} 个独立的 caption")

    return final_caption


def get_seq_instance(
        seq_len: int,
        scene_name: str,
        raw_scene_files: List[str],
        resample_to_2hz: bool = False,
        min_gap: Optional[int] = None,
) -> List[CaptionInstance]:
    """从原始场景文件中提取指定长度的序列实例。

    支持使用滑动窗口提取固定长度的子序列。

    Args:
        seq_len: 提取的序列长度。如果为 0，则返回整个场景。
        scene_name: 场景名称。
        raw_scene_files: 场景中所有文件的路径列表。
        resample_to_2hz: 是否将 10Hz 数据降采样到 2Hz。默认为 False。
        min_gap: 序列片段之间的最小间隔（帧数）。如果为 None，则无间隔（密集滑动窗口）。
            默认为 None。

    Returns:
        List[CaptionInstance]: 提取出的序列实例列表，每个元素是包含 'scene_name' 和
            'scene_files' 键的字典。

    Note:
        当 seq_len > 0 时：
        - 如果不指定 min_gap，使用步长为 1 的滑动窗口
        - 如果指定 min_gap，使用 max(min_gap, seq_len // 2) 作为步长
    """
    instances_subset = []

    if len(raw_scene_files) == 0:
        if os.getenv('DATASET_DEBUG'):
            logging.info(
                f"Scene {scene_name} - has no frames."
            )
        return instances_subset
    if resample_to_2hz:
        # 将 10Hz 降采样到 2Hz
        scene_files = raw_scene_files[::5]
    else:
        scene_files = raw_scene_files

    if seq_len > 0:
        if len(scene_files) < seq_len:
            if os.getenv('DATASET_DEBUG'):
                logging.info(
                    f"Skipping scene {scene_name} - insufficient frames ({len(scene_files)}) for seq_len {seq_len}."
                )
            return instances_subset

        # Construct sliding-window segments.
        if not min_gap:
            limit = len(scene_files) - seq_len + 1
            for i in range(limit):
                sub_seq = scene_files[i: i + seq_len]
                instances_subset.append({
                    'scene_name': f'{scene_name}__sub_seq_{i}',
                    'scene_files': sub_seq,
                })
        else:
            gap = max(min_gap, seq_len // 2)
            limit = len(scene_files) - seq_len + 1
            for i in range(0, limit, gap):
                sub_seq = scene_files[i: i + seq_len]
                instances_subset.append({
                    'scene_name': f'{scene_name}__seq_len_{seq_len}__gap_{gap}__sub_seq_{i}',
                    'scene_files': sub_seq,
                })
    else:
        # 直接加载整个场景 / 最终返回的长度不定
        instances_subset.append({
            'scene_name': scene_name,
            'scene_files': scene_files,
        })

    return instances_subset
