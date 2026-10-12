"""加载基于实例级注释的训练数据。

该模块从 UniOcc 数据集中加载带注释的 occupancy 序列数据，
使用序列第一帧的 caption 作为整个序列的注释。
"""

import json
import logging
import os
import pickle
from typing import Dict, List, Optional

from occinteract.loaders.supervised.caption_sources import load_merge_captions, get_seq_instance
from occinteract.type_models.dataset import CaptionInstance


def load_instance_caption_dataset_info(
        dataset_path: str,
        len_config: Dict[int, int],
        min_gap: Optional[int] = None
) -> List[CaptionInstance]:
    """加载基于实例级注释的训练数据信息。

    将序列的第一帧的 caption 作为整个序列的 caption，
    按照 len_config 中给定的长度比例输出不同比例的序列数据。

    dataset_setting 配置格式：
        - 第一个元素：数据集标识符字符串
        - 第二个元素：bool，是否包含 test 集

    如果某序列的第一帧缺失 caption，则跳过该序列。

    Args:
        dataset_path (str): 数据集根目录路径。
        len_config (Dict[int, int]): 序列长度到重复次数的映射字典，
            例如 {16: 1, 8: 1} 表示长度为 16 和 8 的序列各重复 1 次。
        min_gap (Optional[int]): 序列中帧之间的最小间隔，默认为 None。

    Returns:
        List[CaptionInstance]: 训练实例列表，每个实例包含：
            - caption (str): 文本描述
            - scene_name (str): 场景标识符
            - scene_files (List[str]): occupancy 文件路径列表
    """
    # 按照 len_config 中给定的长度比例输出不同比例的序列数据
    logging.warning(
        f'supervised_seq_v3/get_per_instance_caption_train_data.py -> load_instance_caption: '
        f'训练序列长度比例设置, {json.dumps(len_config)}'
    )

    dataset_setting = [
        ('NuScenes-via-Occ3D-2Hz', False),
        ('Waymo-via-Occ3D-2Hz', False),
        ('Carla-2Hz', False),
        ('OpV2V-10Hz', True),
    ]
    with open(f'{dataset_path}/scene_infos.pkl', 'rb') as f:
        all_scene_infos = pickle.load(f)

    caption_dict = load_merge_captions(dataset_path, dataset_setting)

    train_instances = []
    for dataset, has_test_set in dataset_setting:
        count_seq = 0
        count_train_instance = 0

        if '10Hz' in dataset:
            resample_to_2hz = True
        else:
            resample_to_2hz = False

        # 加载 caption 数据
        if has_test_set:
            sub_set = ['train', 'val', 'test']
        else:
            sub_set = ['train', 'val']

        for sub in sub_set:
            dataset_key = f'{dataset}-{sub}'
            for instance_len, repeat_time in len_config.items():
                sub_train_instances = []
                for scene_data in all_scene_infos[dataset_key]:
                    seq_list = get_seq_instance(
                        instance_len,
                        scene_data['scene_name'],
                        scene_data['occ_in_scene_paths'],
                        resample_to_2hz,
                        min_gap
                    )

                    count_seq += len(seq_list)
                    for seq_data in seq_list:
                        first_occ_file = seq_data['scene_files'][0]
                        caption_key = first_occ_file.replace(f'{dataset_key}/', '').replace('.npy', '.npz')
                        if caption_key in caption_dict[dataset]:
                            caption_list = caption_dict[dataset][caption_key]
                        else:
                            if os.getenv('DATASET_DEBUG'):
                                logging.info(
                                    f'缺失 caption 的文件 {caption_key}, 跳过对应序列 {seq_data["scene_name"]}'
                                )
                            continue

                        for caption in caption_list:
                            for _ in range(repeat_time):
                                # 转换为绝对路径
                                train_seq = []
                                for file in seq_data['scene_files']:
                                    train_seq.append(f'{dataset_path}/{file}')

                                sub_train_instances.append({
                                    'caption': caption,
                                    'scene_name': seq_data['scene_name'],
                                    'scene_files': train_seq,
                                })

                                count_train_instance += 1

                train_instances.extend(sub_train_instances)
                logging.info(
                    f'supervised_seq_v3/get_per_instance_caption_train_data.py -> load_instance_caption: '
                    f' 数据集 {dataset_key} 加载长度 {instance_len} 共 {len(sub_train_instances):,} 个训练实例.'
                )

        logging.warning(
            f'supervised_seq_v3/get_per_instance_caption_train_data.py -> load_instance_caption: '
            f'数据集 {dataset} 一共生成 {count_train_instance:,} 个训练实例.'
        )

    logging.warning(
        f'supervised_seq_v3/get_per_instance_caption_train_data.py -> load_instance_caption: '
        f'生成 {len(train_instances):,} 个训练实例.'
    )
    return train_instances
