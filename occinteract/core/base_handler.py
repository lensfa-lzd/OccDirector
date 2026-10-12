import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from timeout_decorator import timeout

from occinteract.core.scripts import show_cv2_img, create_grid_image

from occinteract.tools.pyvista_vis import render_semantic_scene


class BaseHandler:
    """数据处理的基础处理器，提供通用的处理管道和可视化功能。

    支持批量处理任务，具备超时控制和错误处理机制。
    可通过继承重写 pipeline 方法实现具体的数据处理逻辑。
    """

    def __init__(self, data_path: str, dataset: Any = None) -> None:
        """初始化基础处理器。

        Args:
            data_path: 数据文件所在的基础路径。
            dataset: 可选的数据集对象，用于额外的数据加载支持。
        """
        self.dataset = dataset
        self.data_path = data_path

        self.vis_handler = BaseVisualizationHandler()

    @timeout(10 * 600, use_signals=False)
    def process(self, task_cfg_str: str) -> Dict[str, Optional[str]]:
        """处理指定索引的数据，生成场景的评分。

        该方法支持超时控制（默认 10 分钟超时），处理失败时返回错误信息。

        Args:
            task_cfg_str: JSON 格式的任务配置字符串，格式为:
                {
                    'key': str,  # 用于生成后查找数据
                    'index': int,
                }

        Returns:
            dict: 包含 task_cfg_str、处理结果的字典。成功时包含 'result' 字段，
                失败时包含 'error' 字段.
        """
        task_cfg = json.loads(task_cfg_str)
        key = task_cfg['key']
        try:
            success, result = self.pipeline(key)

            if success:
                return {
                    "task_cfg_str": task_cfg_str,
                    "result": json.dumps(result),
                }
            else:
                return {
                    "task_cfg_str": task_cfg_str,
                    "result": None,
                    "error": f'处理失败，中间结果{json.dumps(result)}'
                }

        except Exception as e:
            # 捕获可能出现的 API 错误或解析错误
            logging.warning(f"处理任务{task_cfg_str} 时发生错误:{e}")
            return {
                "task_cfg_str": task_cfg_str,
                "result": None,
                "error": str(e)
            }

    def pipeline(self, key: str) -> Tuple[bool, Any]:
        """处理单个任务的核心管道，需由子类实现。

        Args:
            key: 任务的唯一标识符，用于查找和加载数据。

        Returns:
            tuple[bool, Any]: (success, result) 二元组，success 表示是否成功处理。
                result 为可被 json.dumps 序列化的任意类型。
        """
        raise NotImplementedError

    def load_occ_seq(self, files: List[str]) -> List[np.ndarray]:
        """加载 occupancy 序列数据。

        Args:
            files: 数据文件名的列表，相对于 self.data_path。

        Returns:
            List[np.ndarray]: 加载的 occupancy 数据列表。
                每个数组一般为 [H, W, D] 形状，如 [200, 200, 16]（height, width, depth）。
        """
        occ = [np.load(f'{self.data_path}/{file}') for file in files]
        return occ


class BaseVisualizationHandler:
    """ occupancy 场景可视化处理器。

    提供 BEV（鸟瞰图）和后视角的渲染功能，支持箭头、线条、帧 ID 等标注。
    """

    def __init__(self, color: Tuple[int, int, int] = (0, 0, 0)) -> None:
        """初始化可视化处理器。

        Args:
            color: 可视化的颜色（线条、帧颜色），默认为黑色。
        """
        self.color = color  # 可视化的颜色（线条，帧颜色） 默认为黑色

    @staticmethod
    def add_center_point(
            img_cv2: np.ndarray,
            color: Tuple[int, int, int],
            radius: int,
    ):
        height, width = img_cv2.shape[:2]
        center_x = width // 2
        center_y = height // 2
        center_coordinates = (center_x, center_y)

        cv2.circle(img_cv2, center_coordinates, radius, color, -1)

    def add_arrow(
            self,
            img_cv2: np.ndarray,
            color: Tuple[int, int, int],
            arrow_args: Dict[str, Any]
    ) -> None:
        """在图像中心绘制代表车头方向的箭头。

        Args:
            img_cv2: BGR 格式的图像数组，形状为 [H, W, 3]。
            color: 箭头颜色，BGR 元组。
            arrow_args: 箭头参数字典，包含:
                - length: 箭头长度
                - radius: 中心点半径
                - thickness: 箭头线条宽度
        """
        height, width = img_cv2.shape[:2]
        center_x = width // 2
        center_y = height // 2
        center_coordinates = (center_x, center_y)

        arrow_length = arrow_args['length']
        point_radius = arrow_args['radius']
        arrow_thickness = arrow_args['thickness']

        end_point = (center_x, center_y - arrow_length)
        self.add_center_point(img_cv2, color, point_radius)
        cv2.arrowedLine(img_cv2, center_coordinates, end_point, color, arrow_thickness, tipLength=0.3)

    @staticmethod
    def add_line(
            img_cv2: np.ndarray,
            color: Tuple[int, int, int],
            line_args: Dict[str, Any]
    ) -> None:
        """在图像中心绘制垂直和水平交叉线。

        Args:
            img_cv2: BGR 格式的图像数组，形状为 [H, W, 3]。
            color: 线条颜色，BGR 元组。
            line_args: 线条参数字典，包含:
                - thickness: 线条宽度
        """
        height, width = img_cv2.shape[:2]
        center_x = width // 2
        center_y = height // 2

        line_thickness = line_args['thickness']
        cv2.line(img_cv2, (center_x, 0), (center_x, height), color, line_thickness)
        cv2.line(img_cv2, (0, center_y), (width, center_y), color, line_thickness)

    @staticmethod
    def add_frame_id(
            img_cv2: np.ndarray,
            frame_index: int,
            color: Tuple[int, int, int],
            frame_id_args: Dict[str, Any]
    ) -> None:
        """在图像上绘制帧编号。

        Args:
            img_cv2: BGR 格式的图像数组，形状为 [H, W, 3]。
            frame_index: 要绘制的帧编号。
            color: 文本颜色，BGR 元组。
            frame_id_args: 帧 ID 参数字典，包含:
                - org: 文本起始坐标 (x, y)
                - front_scale: 字体缩放比例
                - thickness: 文本线条宽度
        """
        frame_id_org = frame_id_args['org']
        frame_id_front_scale = frame_id_args['front_scale']
        frame_id_thickness = frame_id_args['thickness']

        cv2.putText(
            img_cv2, f"#{frame_index}", frame_id_org,
            cv2.FONT_HERSHEY_SIMPLEX, frame_id_front_scale, color, frame_id_thickness
        )

    def render_bev_image(
            self,
            frame_index: int,
            semantic_grid: np.ndarray,
            radius_factor: float,
            elev: int,
            center_point_args: Optional[Dict] = None,
            arrow_args: Optional[Dict[str, Any]] = None,
            frame_id_args: Optional[Dict[str, Any]] = None,
            line_args: Optional[Dict[str, Any]] = None,
            extra_ignore_class: Optional[List[int]] = None,
            image_size: Tuple[int, int] = (512, 512),
            azim: int = 180,
    ) -> np.ndarray:
        """渲染 BEV（鸟瞰图）或后视角的 occupancy 场景图像。

        Args:
            frame_index: 当前帧的索引，用于显示帧编号。
            semantic_grid: 语义网格数据，一般为 [H, W, D] 形状，如 [200, 200, 16]。
            radius_factor: 渲染半径的缩放因子。
            elev: 俯仰角（度），90 为正上方 BEV 视角。
            center_point_args: 车辆位置的中心点, radius
            arrow_args: 车头方向箭头参数，包含 length, radius, thickness。
            frame_id_args: 帧编号显示参数，包含 org, front_scale, thickness。
            line_args: 交叉线参数，包含 thickness。
            extra_ignore_class: 需要忽略不渲染的类别列表。
            image_size: 输出图像的大小 (width, height)。
            azim: 方向角角（度），180 图片上方为车辆正前方。

        Returns:
            np.ndarray: BGR 格式的渲染图像，形状为 [image_size[1], image_size[0], 3]。
        """
        img = render_semantic_scene(
            grid_data=semantic_grid,
            radius_factor=radius_factor,
            elev=elev,  # 俯视角度
            azim=azim,  # 方位角  180° 方向角渲染出来的结果才是前方在图片的上面
            output_path='',
            show=False,
            show_axis=False,
            draw_ego_car=False,
            extra_ignore_class=extra_ignore_class,  # 只渲染路面， 排除 7
            window_size=image_size,
        )
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 为 bev 中的 ego car 添加标识点以及代表车头方向的箭头
        color = (0, 0, 0)

        if center_point_args:
            self.add_center_point(
                img_cv2,
                color,
                **center_point_args,
            )

        if arrow_args:
            self.add_arrow(
                img_cv2,
                color,
                arrow_args
            )

        # 绘制垂直线+水平线
        if line_args:
            self.add_line(
                img_cv2,
                color,
                line_args
            )

        if frame_id_args:
            self.add_frame_id(
                img_cv2,
                frame_index,
                color,
                frame_id_args
            )

        return img_cv2

    @staticmethod
    def show_cv2_image_list(
            img_bgr_list: List[np.ndarray],
            axis: Optional[str] = None,
            make_grid: bool = False
    ) -> None:
        """显示一系列 BGR 图像。

        Args:
            img_bgr_list: BGR 格式图像列表，每张图像形状为 [H, W, 3]。
            axis: matplotlib 坐标轴显示模式，可选 'on'、'off' 或 None。
            make_grid: 是否将图像拼成网格显示。
        """
        if make_grid:
            grids = create_grid_image(img_bgr_list)
            if isinstance(grids, list):
                for grid in grids:
                    show_cv2_img(grid, axis)
            else:
                show_cv2_img(grids, axis)
        else:
            for img_bgr in img_bgr_list:
                show_cv2_img(img_bgr, axis)
