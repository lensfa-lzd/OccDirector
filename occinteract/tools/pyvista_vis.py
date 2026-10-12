"""使用 PyVista 渲染 3D 语义占据网格的高性能可视化工具。"""
import os
from typing import Tuple, Optional, Union, Set, List

import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap

if os.getenv('COLOR_MAP_V2'):
    COLOR_MAP_UINT8 = np.array([
        [255, 255, 255, 255],  # 0 undefined
        [232, 139, 196, 255],  # 1 car
        [141, 160, 204, 255],  # 2 bicycle
        [255, 217, 46, 255],  # 3 motorcycle
        [252, 142, 98, 255],  # 4 pedestrian
        [227, 25, 28, 255],  # 5 traffic_cone
        [43, 161, 43, 255],  # 6 vegetation
        [179, 179, 179, 255],  # 7 road
        [102, 102, 102, 255],  # 8 terrain
        [242, 228, 206, 255],  # 9 building
        [0, 0, 0, 0],  # 10 free
    ], dtype=np.uint8)
else:
    # Uniocc format from: https://github.com/tasl-lab/UniOcc/blob/main/uniocc_viz.py
    # 1. 定义颜色映射和类别 (与 matplotlib 版本保持一致)
    # -----------------------------------------------------------------------------
    COLOR_MAP_UINT8 = np.array([
        [255, 255, 255, 255],  # 0 undefined       white
        [0, 150, 245, 255],  # 1 car             blue
        [255, 192, 203, 255],  # 2 bicycle         pink
        [200, 180, 0, 255],  # 3 motorcycle      dark orange
        [255, 255, 0, 255],  # 4 pedestrian      yellow 将行人渲染为黄色，在红色的路面上更为突出
        [255, 240, 150, 255],  # 5 traffic_cone    light yellow
        [0, 175, 0, 255],  # 6 vegetation      green
        [255, 0, 255, 255],  # 7 road            dark pink
        [100, 100, 100, 255],  # 8 terrain         dark gray (为了清晰区分，颜色与matplotlib不同)
        [230, 230, 250, 255],  # 9 building        lavender/light gray
        [0, 0, 0, 0],  # 10 free           transparent black
        # --- 为坐标轴保留的标签 ---
        [255, 0, 0, 255],  # 11 reserved for X-axis (红色)
        [0, 255, 0, 255],  # 12 reserved for Y-axis (绿色)
        [0, 0, 255, 255],  # 13 reserved for Z-axis (蓝色)
        # --- ego-car保留标签 ---
        [0, 0, 139, 255],  # 14 for ego-car  dark blue
        # --- 其他保留标签 ---
        [120, 200, 255, 255],  # 15 reserved       light orange (reused color)
        [255, 100, 100, 255],  # 16 reserved       light red
        [255, 245, 190, 255],  # 17 reserved       light yellow
        [100, 220, 100, 255],  # 18 reserved       light green
        [255, 100, 255, 255],  # 19 reserved       light magenta
    ], dtype=np.uint8)

# 将颜色图归一化到 [0, 1] 范围
COLOR_MAP_FLOAT = COLOR_MAP_UINT8.astype(np.float32) / 255.0

# 定义不应被渲染的类别 (空闲和未定义)
EMPTY_CLASSES = {0, 10}


def render_semantic_scene(
        grid_data: np.ndarray,
        elev: int = 30,
        azim: int = 45,
        output_path: str = '',
        show: bool = False,
        window_size: Tuple[int, int] = (512, 512),
        show_axis: bool = False,
        draw_ego_car: bool = False,
        radius_factor: float = 1.5,
        extra_ignore_class: Optional[Union[List[int], Set[int]]] = None,
) -> np.ndarray:
    """使用 PyVista 从特定角度渲染语义占据网格。

    基于 GPU 加速的高性能渲染工具,相比 matplotlib 版本速度更快。
    支持坐标轴显示和 ego-car 标记,适合生成高质量的 BEV 可视化。

    Args:
        grid_data (np.ndarray): 3D 整数数组,表示语义标签。形状: (H, W, D)
        elev (int): 摄像机仰角(degrees)。默认: 30
        azim (int): 摄像机方位角(degrees)。默认: 45
        output_path (str): 图像保存路径。为空则不保存。默认: ''
        show (bool): 是否在交互式窗口中显示场景。默认: False
        window_size (Tuple[int, int]): 渲染窗口尺寸 (width, height)。默认: (512, 512)
        show_axis (bool): 是否在场景中用红绿蓝三色线显示坐标轴。默认: False
        draw_ego_car (bool): 是否在场景中心绘制表示 ego-car 的方块。
            主要用于在 BEV 图中展示自车位置。由于无法确定地面位置,
            自车会被绘制在场景的最顶端。仅保证在 BEV 视角中的正确性 (elev=90)。
            默认: False
        radius_factor (float): 控制相机距离,为场景对角线长度的 n 倍。默认: 1.5
        extra_ignore_class (Optional[Union[List[int], Set[int]]]):
            额外指定的不渲染的类别 ID 列表。默认会忽略类别 0 (undefined) 和 10 (free)。
            默认: None

    Returns:
        np.ndarray: 渲染后的 RGB 图像数组。形状: (H, W, 3),dtype: uint8
    """
    # 根据 show_axis 选项准备渲染数据和颜色图
    render_grid = grid_data.copy()
    active_colormap = COLOR_MAP_FLOAT.copy()

    if show_axis:
        # 定义坐标轴的标签和颜色
        AXIS_X_LABEL, AXIS_Y_LABEL, AXIS_Z_LABEL = 11, 12, 13
        active_colormap[AXIS_X_LABEL] = [1.0, 0.0, 0.0, 1.0]  # Red
        active_colormap[AXIS_Y_LABEL] = [0.0, 1.0, 0.0, 1.0]  # Green
        active_colormap[AXIS_Z_LABEL] = [0.0, 0.0, 1.0, 1.0]  # Blue

        # 在数据中"绘制"坐标轴 (此处的render_grid已经是副本)
        render_grid[:, 0, 0] = AXIS_X_LABEL
        render_grid[0, :, 0] = AXIS_Y_LABEL
        render_grid[0, 0, :] = AXIS_Z_LABEL

    if draw_ego_car:
        # 定义 ego-car 的标签和颜色
        EGO_CAR_LABEL = 14
        # 使用深蓝色 (R=0, G=0, B=139)
        EGO_CAR_COLOR = np.array([0, 0, 139, 255], dtype=np.float32) / 255.0
        active_colormap[EGO_CAR_LABEL] = EGO_CAR_COLOR

        # 获取场景尺寸
        h, w, d = render_grid.shape
        # 定义 ego-car 尺寸 (h, w, d 对应 x, y, z 轴)
        # 仅是可视化需要，并不代表真是尺寸
        ego_h, ego_w, ego_d = 10, 6, 1

        # 计算场景中心点
        center_x, center_y, center_z = h // 2, w // 2, d // 2

        # 计算 ego-car 的切片范围，使其位于中心
        start_x = center_x - ego_h // 2
        end_x = start_x + ego_h
        start_y = center_y - ego_w // 2
        end_y = start_y + ego_w

        # 在 render_grid 中填充 ego-car 标签
        render_grid[start_x:end_x, start_y:end_y, -ego_d:] = EGO_CAR_LABEL

    h, w, d = render_grid.shape

    # **关键步骤 1: 将 NumPy 数组转换为 PyVista StructuredGrid**
    x = np.arange(h + 1, dtype=np.float32)
    y = np.arange(w + 1, dtype=np.float32)
    z = np.arange(d + 1, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid = pv.StructuredGrid(xx, yy, zz)

    # 将语义标签作为 "cell data" 添加到网格中
    grid.cell_data['semantics'] = render_grid.flatten(order='F')

    # **关键步骤 2: 过滤掉空闲/未定义的体素**
    ignore_class = list(EMPTY_CLASSES)
    if extra_ignore_class:
        ignore_class.extend(list(extra_ignore_class))
    non_empty_mask = np.isin(grid.cell_data['semantics'], ignore_class, invert=True)
    visible_voxels = grid.extract_cells(non_empty_mask)

    # **关键步骤 3: 设置渲染器 (Plotter)**
    plotter = pv.Plotter(off_screen=not show, window_size=window_size)

    # 从可能被修改过的颜色图数组创建自定义颜色图
    custom_cmap = ListedColormap(active_colormap)

    # **关键步骤 4: 添加网格到场景并设置着色**
    # 只有在 visible_voxels 不为空的情况下才将其添加到渲染器
    if visible_voxels.n_cells > 0:
        plotter.add_mesh(
            visible_voxels,
            scalars='semantics',
            cmap=custom_cmap,
            show_edges=False,
            clim=[0, len(active_colormap) - 1],
            show_scalar_bar=False
        )

    # 美化场景
    plotter.set_background('white')
    plotter.enable_anti_aliasing('ssaa')

    # **关键步骤 5: 将 elev/azim 转换为 PyVista 相机位置**
    focal_point = (h / 2, w / 2, d / 2)
    radius = np.linalg.norm((h, w, d)) * radius_factor
    elev_rad = np.deg2rad(elev)
    azim_rad = np.deg2rad(azim)
    x_offset = radius * np.cos(elev_rad) * np.cos(azim_rad)
    y_offset = radius * np.cos(elev_rad) * np.sin(azim_rad)
    z_offset = radius * np.sin(elev_rad)
    camera_pos = (focal_point[0] + x_offset, focal_point[1] + y_offset, focal_point[2] + z_offset)
    view_up = (0, 0, 1)
    plotter.camera_position = [camera_pos, focal_point, view_up]

    # **关键步骤 6: 返回图像、保存和显示**
    img_array = plotter.screenshot(transparent_background=False, return_img=True)

    if output_path:
        plotter.screenshot(output_path)

    if show:
        plotter.show()

    plotter.close()

    return img_array


# --- Demo (synthetic data, no external files needed) ---
if __name__ == "__main__":
    # On a headless server, install vtk-osmesa first:
    # pip install --extra-index-url https://wheels.vtk.org vtk-osmesa
    semantic_grid = np.zeros((200, 200, 16), dtype=np.uint8)
    semantic_grid[90:110, 90:110, 8:] = 7  # road surface
    semantic_grid[95:105, 95:105, 10:] = 1  # car on the road
    semantic_grid[10:40, 150:180, 8:] = 9  # building

    print(f"grid shape: {semantic_grid.shape}")
    img = render_semantic_scene(grid_data=semantic_grid, elev=90, azim=180, show=False)
    print(f"rendered image: {img.shape}")
