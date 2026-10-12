import json
import logging
import os
import pickle
from glob import glob
from pathlib import Path

from tqdm import tqdm

from occinteract.core.base_handler import BaseVisualizationHandler, BaseHandler
from .road_geometry import get_bev_road_map_branch_distance


class PipeLine:
    def __init__(self):
        self.vis_handler = BaseVisualizationHandler()
        self.max_frame_idx = 120
        self.min_action_duration_frames = 16
        self.occ_render_kwargs = {
            'elev': 90,
            'radius_factor': 1.5,
            'extra_ignore_class': [1, 2, 3, 4, 5, 6, 8, 9]  # 只渲染路面， 排除 7
        }
        self.frame_id_args = {
            'org': (0, 80),
            'front_scale': 2.5,
            'thickness': 5,
        }
        self.lateral_threshold = 20  # 横向偏离的阈值（像素）
        self.lookahead_distance = 200  # ego 前方的探测距离

    def analyze_turn_point(self, occ_seq):
        # 解析序列中车辆行驶到转弯点（岔路口，交叉口）的帧时刻
        result = {
            'occ_seq': occ_seq,
            'graph': [],
            'distance': [],  # 存储提取到距离的点
            'm_points': {}
        }
        # for idx, occ in enumerate(tqdm(occ_seq)):
        for idx, occ in enumerate(occ_seq):
            road_img = self.vis_handler.render_bev_image(
                frame_index=-1,  # 无意义
                semantic_grid=occ,
                **self.occ_render_kwargs
            )
            G, dist, curve_point = get_bev_road_map_branch_distance(
                road_img,
                self.lateral_threshold,
                self.lookahead_distance
            )
            result['graph'].append((G, dist, curve_point))

            if dist < self.lookahead_distance:
                result['distance'].append((idx, dist))

        # 获取“极值点”; 路口最近的点; 在路口行为的中间时间点
        m_points, kept_trace = self.extract_minima_by_strict_descent(
            result['distance'],
            reject_limit=5
        )
        result['m_points'].update({
            'm_points': m_points,
            'kept_trace': kept_trace
        })

        return result

    @staticmethod
    def filter_isolated_points(data, min_count_threshold, max_gap=1):
        """
        过滤掉 x 轴上孤立的点集。

        Args:
            data: 格式为 [(x, y), ...] 的列表，必须按 x 从小到大排好序。
            min_count_threshold: 数量阈值。如果一个连贯点集的点数小于此值，则被过滤。
            max_gap: 距离阈值。如果两个相邻点的 x 差值大于此值，则断开链接。默认为 1。

        Returns:
            过滤后的 [(x, y), ...] 列表
        """
        if not data:
            return []

        filtered_data = []

        # 初始化当前分组
        current_cluster = [data[0]]
        for i in range(1, len(data)):
            prev_point = data[i - 1]
            curr_point = data[i]

            # 判断当前点是否与前一个点“连续”
            # 如果差值 <= max_gap，视为同一组
            if (curr_point[0] - prev_point[0]) <= max_gap:
                current_cluster.append(curr_point)
            else:
                # 距离过大，断开。结算上一组
                if len(current_cluster) >= min_count_threshold:
                    filtered_data.extend(current_cluster)

                # 开始新的分组
                current_cluster = [curr_point]

        # 循环结束后，不要忘记处理最后一组
        if len(current_cluster) >= min_count_threshold:
            filtered_data.extend(current_cluster)

        return filtered_data

    # 辅助函数：检查是否满足严格单调下降的
    @staticmethod
    def _check_look_back_valid(value, kept_points, window):
        # kept_points 包含现在的点，需要移除
        check_list = kept_points[:-1]

        # 1. 边界检查：如果前面数据不够窗口长度，视为无效（或者可视情况改为放行）
        if len(check_list) < window:
            return False

        # 2. 遍历回看：要求 list[i-1] > list[i]
        curr_d = value
        for k in range(1, window + 1):
            prev_d = check_list[-k][1]
            # 如果前一个点 小于等于 后一个点，说明不是严格单调下降
            if prev_d <= curr_d:
                return False

            curr_d = prev_d
        return True

    def extract_minima_by_strict_descent(
            self,
            data,
            reject_limit=5,
            diff_threshold=50,
            look_back_window=5,
            max_distance=50,
    ):
        """
        通过提取严格下降序列来寻找最小值。
        :param diff_threshold: 相邻的点的最大间隔
        :param data: 按照时间排序好的点 [(x, y), ...]
        :param reject_limit: 连续丢弃多少个点后，认为是一个新的下降序列开始了
        :param look_back_window: 回看窗口大小，要求最小值前的 N 个点必须是严格单调递减的。
        :param max_distance: minima (turn point) 的最小距离
        """
        # 过滤掉离群的点
        data = self.filter_isolated_points(data, min_count_threshold=5, max_gap=1)

        if len(data) == 0:
            return [], data

        # 存储最终结果（每个下降序列的最后一个点）
        minima_points = []

        # 状态初始化
        current_min_point = data[0]
        reject_count = 0  # 连续剔除计数器

        # 记录这一轮连续剔除是从哪个索引开始的
        first_reject_idx = -1

        # 记录所有保留下来的点
        kept_points = [data[0]]

        # 改用 while 循环，以便能够手动修改 i (实现回滚)
        i = 1
        while i < len(data):
            curr_t, curr_d = data[i]

            # === 核心逻辑 ===
            # 比较 当前点 vs 当前序列的最小值（上一个保留点）
            # 并且差值小于变化阈值
            if curr_d < current_min_point[1] and (current_min_point[1] - curr_d) < diff_threshold:
                # 情况A: 这是一个更小的点 -> 保留
                current_min_point = (curr_t, curr_d)

                # 既然找到了更好的点，之前的剔除都不算数了，重置状态
                reject_count = 0
                first_reject_idx = -1

                kept_points.append((curr_t, curr_d))

                # 继续向后遍历
                i += 1

            else:
                # 情况B: 这是一个更大的点 (或下降太剧烈被认为是噪声)

                # 如果这是当前下降序列后的第一个坏点，记录它的位置
                if reject_count == 0:
                    first_reject_idx = i

                reject_count += 1

                # 我们先继续向后看，直到忍无可忍
                i += 1

                # 检查是否达到“连续剔除阈值”
                if reject_count >= reject_limit:
                    # --- 触发结算 ---

                    # 1. 既然已经连续大了很多次，说明上一个序列肯定结束了
                    if self._check_look_back_valid(current_min_point[1], kept_points, look_back_window):
                        if current_min_point[1] < max_distance:
                            minima_points.append(current_min_point)

                    # 2. 回滚逻辑：
                    #    现在的序列已经彻底判定结束了，
                    #    我们要回到由于数值变大而被我们“暂时忍受”的第一个点 (first_reject_idx)，
                    #    把那个点当作新一轮爬坡/下坡的起点。

                    # 重置有效序列表，并将起点设为 first_reject_idx (第一个被剔除的点)
                    current_min_point = data[first_reject_idx]
                    kept_points.append(current_min_point)  # 记录下来方便调试

                    # 3. 调整循环指针 i，从回滚点的下一个点开始
                    i = first_reject_idx + 1

                    # 4. 重置计数器
                    reject_count = 0
                    first_reject_idx = -1

        # 循环结束后，处理最后一个序列
        # 如果最后一个点太靠近结尾则过滤掉，（认为不足以完成完整的通过动作）
        t_idx = current_min_point[0]
        if t_idx < self.max_frame_idx - self.min_action_duration_frames / 2:
            if self._check_look_back_valid(current_min_point[1], kept_points, look_back_window):
                if current_min_point[1] < max_distance:
                    minima_points.append(current_min_point)

        return minima_points, kept_points

class Handler(BaseHandler):
    def __init__(self, data_path, dataset):
        super().__init__(data_path, dataset)
        self.data_path = data_path
        self._pipeline = PipeLine()

    def pipeline(self, index):
        try:
            # print(index)
            data = self.dataset[index]
            occ_seq = self.load_occ_seq(data['occ_in_scene_paths'])

            turn_point_info = self._pipeline.analyze_turn_point(occ_seq)
            result = turn_point_info['m_points']['m_points']
            return True, result

        except Exception as e:
            logging.warning(f'caption_pipeline_geometry.py -> Handler -> pipeline: {e}')
            return False, None


def sort_occ_key(occ_file):
    file_name = Path(occ_file).name
    number = file_name.split('.')[0]
    return int(number)


def get_process_task(base_path, info_file='scene_infos.pkl'):
    p_bar = tqdm()
    folder_count = 0
    count = 0
    tasks = []
    dataset = {}

    if not os.path.isfile(f'{base_path}/{info_file}'):
        file_info = []
        for map_path in glob(f'{base_path}/Town*'):
            p = Path(map_path)
            map_name = p.name
            for try_instance_path in glob(f'{map_path}/*'):
                pp = Path(try_instance_path)
                video_try = pp.name
                folder_count += 1

                key = f'{map_name}/{video_try}'
                occ_path = os.path.join(try_instance_path, 'occupancy')
                raw_occ_files = sorted(glob(os.path.join(occ_path, '*.npy')), key=sort_occ_key)
                if len(raw_occ_files) > 0:
                    occ_files = []
                    for it in raw_occ_files:
                        occ_name = Path(it).name
                        occ_files.append(f'{map_name}/{video_try}/occupancy/{occ_name}')

                    file_info.append({
                        'scene_name': key,
                        'occ_in_scene_paths': occ_files,
                    })

                    task_cfg = {
                        'key': key,
                    }
                    tasks.append(json.dumps(task_cfg))
                    dataset[key] = {
                        'scene_name': key,
                        'occ_in_scene_paths': occ_files,
                    }
                    count += 1

                p_bar.update(1)

        print(f'找到文件夹 {folder_count} 个, 有效数据 {count} 个')
        file_info = sorted(file_info, key=lambda x: x['scene_name'])
    else:
        with open(f'{base_path}/{info_file}', 'rb') as f:
            file_info = pickle.load(f)

        for it in file_info:
            key = it['scene_name']
            task_cfg = {
                'key': key,
            }
            tasks.append(json.dumps(task_cfg))
            dataset[key] = {
                'scene_name': key,
                'occ_in_scene_paths': it['occ_in_scene_paths'],
            }
            p_bar.update(1)

        tasks = sorted(tasks)
        print(f'找到文件夹 {len(tasks)} 个')

    return tasks, dataset, file_info
