"""加载基于序列级注释的训练数据。

该模块从 OccCollect 和 TTSG 数据集中加载带注释的 occupancy 序列数据，
支持滑动窗口采样以生成不同长度的训练序列。
"""

import logging
import pickle
from typing import List

from occinteract.type_models.dataset import CaptionInstance


def get_slide_seq_list(
        occ_file_list: List[str],
        center_index: int,
        sub_seq_len: int,
        slide_count: int,
        slide_gap: int = 1,
) -> List[List[str]]:
    """生成滑动窗口子序列列表。

    以 center_index 为中心，向两侧滑动生成多个指定长度的子序列。

    Args:
        occ_file_list (List[str]): 原始 occupancy 文件路径列表。
        center_index (int): 中心帧的索引。
        sub_seq_len (int): 子序列的长度（帧数）。
        slide_count (int): 滑动的次数。
        slide_gap (int): 每次滑动的间隔帧数，默认为 1。

    Returns:
        List[List[str]]: 滑动窗口子序列列表，每个元素是一个子序列的文件路径列表。
    """
    if len(occ_file_list) < sub_seq_len:
        return [occ_file_list]

    half_window = sub_seq_len // 2
    slide_seq_list = []
    for i in range(-slide_count, slide_count):
        start = center_index - half_window - i * slide_gap
        end = start + 2 * half_window
        if start < 0 or end >= len(occ_file_list):
            continue

        slide_seq_list.append(occ_file_list[start:end])

    if len(slide_seq_list) == 0:
        slide_seq_list.append(occ_file_list)

    return slide_seq_list


def get_occ_collect_dataset_info(dataset_path: str) -> List[CaptionInstance]:
    """获取 OccCollect 数据集的训练数据信息。

    从 OccCollect 数据集中加载 turn caption 并使用滑动窗口生成训练序列。

    Args:
        dataset_path (str): OccCollect 数据集根目录路径。

    Returns:
        List[CaptionInstance]: 训练实例列表，每个实例包含：
            - caption (str): 文本描述，支持 'short'（简短描述）和 'full'（完整描述）两种类型
            - scene_name (str): 场景标识符
            - scene_files (List[str]): occupancy 文件路径列表
    """
    with open(f'{dataset_path}/scene_infos_20260120.pkl', 'rb') as f:
        occ_collect_info = pickle.load(f)

    with open(f'{dataset_path}/turn_caption_20260120.pkl', 'rb') as f:
        occ_collect_caption = pickle.load(f)

    logging.warning(
        f'supervised_seq_v3/get_per_seq_caption_train_data.py -> get_occ_collect_dataset_info: '
        f'加载到 {len(occ_collect_caption):,} 个独立的注释序列'
    )

    instance = []

    # 长度与滑动数量设置
    config = [
        (32, 1, 8),
        (16, 1, 4),
    ]

    logging.warning(
        f'supervised_seq_v3/get_per_seq_caption_train_data.py -> get_occ_collect_dataset_info: '
        f'长度配置信息 {config}'
    )

    occ_collect_info_map = {}
    for it in occ_collect_info:
        occ_collect_info_map[it['scene_name']] = it['occ_in_scene_paths']

    for sub_seq_len, slide_count, slide_gap in config:
        sub_instance = []
        for it in occ_collect_caption:
            scene_name = it['scene_name']
            key_frame_id = int(it['key_frame_id'])
            caption_list = [it['caption']['description_short'], it['caption']['description']]
            sub_seq_list = get_slide_seq_list(
                occ_file_list=occ_collect_info_map[scene_name],
                center_index=key_frame_id,
                sub_seq_len=sub_seq_len,
                slide_count=slide_count,
                slide_gap=slide_gap,
            )
            for i, sub_seq in enumerate(sub_seq_list):
                # 转换为绝对路径
                train_seq = []
                for file in sub_seq:
                    train_seq.append(f'{dataset_path}/{file}')

                for caption_type, caption in zip(['short', 'full'], caption_list):
                    sub_instance.append({
                        'caption': caption,
                        'scene_name': f'occ_collect/{scene_name}__sub_seq_len_{sub_seq_len}__'
                                      f'slide_count_{slide_count}__slide_gap_{slide_gap}__'
                                      f'seq_{i}__caption_{caption_type}',
                        'scene_files': train_seq,
                    })

        instance.extend(sub_instance)
        logging.warning(
            f'supervised_seq_v3/get_per_seq_caption_train_data.py -> get_occ_collect_dataset_info: '
            f'加载长度 {sub_seq_len} 共 {len(sub_instance):,} 个训练实例'
        )

    logging.warning(
        f'supervised_seq_v3/get_per_seq_caption_train_data.py -> get_occ_collect_dataset_info: '
        f'加载了 {len(instance):,} 个训练实例'
    )
    return instance


def get_ttsg_dataset_info(dataset_path: str) -> List[CaptionInstance]:
    """获取 TTSG 数据集的训练数据信息。

    从 TTSG 数据集中加载 caption 并根据序列长度使用滑动窗口或预设帧范围生成训练序列。

    对于超过 16 帧的序列，使用滑动窗口采样；对于不超过 16 帧的序列，
    使用预设的帧范围组合。

    Args:
        dataset_path (str): TTSG 数据集根目录路径。

    Returns:
        List[CaptionInstance]: 训练实例列表，每个实例包含：
            - caption (str): 文本描述，支持 'short'（简短描述）和 'full'（完整描述）两种类型
            - scene_name (str): 场景标识符
            - scene_files (List[str]): occupancy 文件路径列表
    """
    with open(f'{dataset_path}/scene_infos.pkl', 'rb') as f:
        ttsg_info = pickle.load(f)

    with open(f'{dataset_path}/scene_with_critical_score.pkl', 'rb') as f:
        ttsg_caption = pickle.load(f)

    logging.warning(
        f'supervised_seq_v3/get_per_seq_caption_train_data.py -> get_ttsg_dataset_info: '
        f'加载到 {len(ttsg_caption):,} 个独立的注释序列'
    )

    instance = []

    ttsg_info_map = {}
    for it in ttsg_info:
        ttsg_info_map[it['scene_name']] = it['occ_in_scene_paths']

    for it in ttsg_caption:
        scene_name = it['scene_name']
        caption_list = [it['caption']['description_short'], it['caption']['description']]
        occ_file_list = ttsg_info_map[scene_name]

        max_frame = len(occ_file_list)
        if max_frame > 16:
            slide_gap = 1
            sub_seq_len = 16
            slide_count = max(1, (max_frame - sub_seq_len) // slide_gap)
            sub_seq_list = get_slide_seq_list(
                occ_file_list=occ_file_list,
                center_index=max_frame // 2,
                sub_seq_len=sub_seq_len,
                slide_count=slide_count,
                slide_gap=slide_gap,
            )
        else:
            frame_setting = [
                (0, max_frame - 1),
                (0, max_frame - 2),
                (1, max_frame),
                (2, max_frame)
            ]
            sub_seq_list = []
            for start, end in frame_setting:
                if end - start < 8:
                    continue
                sub_seq_list.append(occ_file_list[start:end])

            if len(sub_seq_list) == 0:
                sub_seq_list.append(occ_file_list)

        for i, sub_seq in enumerate(sub_seq_list):
            # 转换为绝对路径
            train_seq = []
            for file in sub_seq:
                train_seq.append(f'{dataset_path}/{file}')

            for caption_type, caption in zip(['short', 'full'], caption_list):
                instance.append({
                    'caption': caption,
                    'scene_name': f'ttsg/{scene_name}__seq_{i}__caption_{caption_type}',
                    'scene_files': train_seq,
                })

    logging.warning(
        f'supervised_seq_v3/get_per_seq_caption_train_data.py -> get_ttsg_dataset_info: '
        f'加载了 {len(instance):,} 个训练实例'
    )
    return instance
