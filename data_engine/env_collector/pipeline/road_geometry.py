import logging

import cv2
import networkx
import numpy as np
from scipy import interpolate
from skimage.morphology import skeletonize


def prune_skeleton(skeleton, min_length=20):
    """
    完善版骨架裁剪函数。
    逻辑：
    1. 识别交叉点区域，并分别标记。
    2. 提取线段。
    3. 对于短线段，检查其连通性：
       - 如果是"桥梁"（连接两个不同ID的交叉区域），保留。
       - 如果是"刺"（只连接一个交叉区域），删除。
    """
    # 0. 复制并确保输入格式正确
    skel = skeleton.copy()
    skel[skel > 0] = 1  # 归一化为 0/1

    # 1. 找到交叉点 (Junctions)
    # 卷积法：邻居数 >= 3 为交叉点
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    # img 这里需要是 0 和 1
    neighbor_count = cv2.filter2D(skel, -1, kernel)

    # 获取交叉点掩膜 (值 >= 13)
    # 为了防止路口像素断裂，稍微膨胀一下路口区域
    junctions_mask = np.where(neighbor_count >= 13, 1, 0).astype(np.uint8)
    junctions_mask = cv2.dilate(junctions_mask, np.ones((3, 3), np.uint8))

    # 关键步骤：给每个独立的交叉路口打上不同的标签 (Labeling)
    # num_junc: 路口数量, junc_labels: 标记图 (0是背景, 1是路口A, 2是路口B...)
    num_junc, junc_labels = cv2.connectedComponents(junctions_mask)

    # 2. 提取路径线段 (Segments)
    # 骨架 - 交叉点 = 独立的线段
    # 注意：为了让线段还能"够得着"路口，我们在逻辑判断时不直接减去，而是提取非路口区域
    # 但直接减去最简单，为了检测连接性，后面会对线段进行膨胀
    segments_mask = (skel == 1) & (junctions_mask == 0)
    segments_mask = segments_mask.astype(np.uint8)

    # 标记每一条线段
    num_seg, seg_labels, stats, centroids = cv2.connectedComponentsWithStats(segments_mask)

    # 创建一个画布用于重建
    # 先把所有的交叉点画回去 (路口是肯定要保留的)
    final_skeleton = junctions_mask.copy()

    # 3. 遍历分析每一条线段
    for i in range(1, num_seg):  # 0 是背景，跳过
        length = stats[i, cv2.CC_STAT_AREA]

        # 获取当前线段的掩膜
        current_seg_mask = (seg_labels == i).astype(np.uint8)

        # 判定逻辑 A: 长度足够长，直接保留
        if length >= min_length:
            final_skeleton = np.logical_or(final_skeleton, current_seg_mask)
            continue

        # 判定逻辑 B: 长度很短，进行拓扑检查
        # 膨胀当前线段，看它碰到了几个"不同的"路口
        dilated_seg = cv2.dilate(current_seg_mask, np.ones((3, 3), np.uint8))

        # 获取膨胀区域内覆盖到的路口标签
        # 这里的 junc_labels 包含了 0(背景) 和 1,2,3...(路口ID)
        touched_labels = junc_labels[dilated_seg > 0]
        # 去重，并移除 0 (背景)
        unique_touched = np.unique(touched_labels)
        unique_touched = unique_touched[unique_touched != 0]

        num_touched_junctions = len(unique_touched)

        # 情况 1: 孤立噪点 (没碰到任何路口) -> 删除 (不添加到 mask)
        if num_touched_junctions == 0:
            pass

        # 情况 2: 毛刺/末梢 (只碰到 1 个路口) -> 删除
        elif num_touched_junctions == 1:
            pass

        # 情况 3: 桥梁 (碰到了 2 个或更多不同的路口) -> 保留
        # 即使它很短，它也是连接路口的关键，不能删
        else:
            final_skeleton = np.logical_or(final_skeleton, current_seg_mask)

    # 4. 最终整理
    # 因为是拼凑的 mask，可能会变粗，重新细化一次保证单像素宽
    final_skeleton = final_skeleton.astype(np.uint8)
    refined_skeleton = skeletonize(final_skeleton).astype(np.uint8)

    return refined_skeleton


def smooth_geometry(path_coords, s=5.0, k=3):
    """
    使用 B样条 (B-Spline) 对路径坐标进行平滑。
    path_coords: shape (N, 2) 的列表或数组，格式为 [(y, x), ...]
    s: 平滑因子
    k: 样条的阶数 (3表示三次样条)
    """
    if len(path_coords) < k + 1:
        # 点太少，不够拟合样条，直接返回原点集
        return path_coords

    # 将坐标分离为 y 和 x
    coords = np.array(path_coords)
    y, x = coords[:, 0], coords[:, 1]

    # 移除重复点（样条拟合不允许相邻点完全重合）
    ok = np.r_[True, np.any(np.diff(coords, axis=0) != 0, axis=1)]
    y, x = y[ok], x[ok]

    if len(y) < k + 1:
        return list(zip(y, x))

    try:
        # tck 是样条参数，u 是参数化变量
        tck, u = interpolate.splprep([x, y], s=s, k=k)
        # 生成新的平滑点，数量跟原长度保持一致或更多
        u_new = np.linspace(0, 1, len(x))
        x_new, y_new = interpolate.splev(u_new, tck)

        # 返回 (y, x) 格式，保持整数以便画图，甚至可以保持 float 用于高精度计算
        return list(zip(y_new, x_new))
    except Exception as e:
        # 极端情况（如直线或回环检测失败），降级回原路径
        print(f"Smoothing failed: {e}")
        return path_coords


def build_graph_from_skeleton(img, smoothing_factor=5.0):
    h, w = img.shape[:2]

    # 1. 预处理
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img, 10, 1, cv2.THRESH_BINARY)
    skeleton = binary.astype(np.uint8)

    # 2. 计算度数
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    degrees_map = cv2.filter2D(skeleton, -1, kernel) * skeleton

    # 3. 识别关键节点
    junctions = np.column_stack(np.where(degrees_map >= 3))
    endpoints = np.column_stack(np.where(degrees_map == 1))

    key_nodes = set(map(tuple, np.vstack((junctions, endpoints)))) if (
            len(junctions) > 0 or len(endpoints) > 0) else set()

    # --- 新增功能：添加 Ego Car 位置 ---
    # 计算图像几何中心
    center_y, center_x = h // 2, w // 2

    # 获取骨架所有像素点
    skel_y, skel_x = np.where(skeleton == 1)

    ego_node = None
    if len(skel_y) > 0:
        # 计算距离中心最近的骨架点
        dists = (skel_y - center_y) ** 2 + (skel_x - center_x) ** 2
        min_idx = np.argmin(dists)
        ego_node = (skel_y[min_idx], skel_x[min_idx])

        # 将 Ego 位置强制加入关键节点集合
        # 这样路径追踪时，如果经过该点会停止并形成节点，保证 Ego 在图上
        key_nodes.add(ego_node)
    # --------------------------------

    # 建立图结构 (此时仍为图像坐标)
    G = networkx.Graph()
    for node in key_nodes:
        # 初始 pos 存 (x, y) 用于 plt.imshow 对应，逻辑坐标用 (y, x)
        attr = {'pos': (node[1], node[0])}
        if node == ego_node:
            attr['type'] = 'ego'
        G.add_node(node, **attr)

    # 4. 路径追踪 (逻辑保持不变)
    visited_mask = np.zeros_like(skeleton, dtype=np.bool_)
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1), (0, 1),
               (1, -1), (1, 0), (1, 1)]

    for start_node in key_nodes:
        sy, sx = start_node
        for dy, dx in offsets:
            ny, nx = sy + dy, sx + dx
            if not (0 <= ny < h and 0 <= nx < w): continue

            if skeleton[ny, nx] == 1 and not visited_mask[ny, nx]:
                path = [start_node, (ny, nx)]
                if (ny, nx) not in key_nodes:
                    visited_mask[ny, nx] = True

                curr = (ny, nx)
                prev = start_node  # 为了不走回头路

                while True:
                    cy, cx = curr
                    # 如果撞到了另一个关键节点（包括我们强制加入的 ego_node）
                    if curr in key_nodes and curr != (ny, nx):
                        end_node = curr
                        if not G.has_edge(start_node, end_node):
                            # 平滑处理 (得到的是图像坐标下的 (y, x) 序列)
                            smooth_path = smooth_geometry(path, s=smoothing_factor)
                            length = len(path)
                            G.add_edge(start_node, end_node, geometry=smooth_path, weight=length)
                        break

                    # 寻找下一个邻居
                    found_next = False
                    for ody, odx in offsets:
                        nny, nnx = cy + ody, cx + odx
                        if nny == prev[0] and nnx == prev[1]: continue  # 不回头
                        if not (0 <= nny < h and 0 <= nnx < w): continue

                        if skeleton[nny, nnx] == 1:
                            path.append((nny, nnx))
                            prev = curr
                            curr = (nny, nnx)
                            if curr not in key_nodes:
                                visited_mask[nny, nnx] = True
                            found_next = True
                            break

                    if not found_next:
                        # 到了死胡同（通常是图像边界或断路，且因为某种原因没被识别为 endpoint）
                        if curr in key_nodes:
                            end_node = curr
                            if start_node != end_node and not G.has_edge(start_node, end_node):
                                smooth_path = smooth_geometry(path, s=smoothing_factor)
                                G.add_edge(start_node, end_node, geometry=smooth_path, weight=len(path))
                        break

    return G, ego_node


def transform_graph_to_ego_frame(G, ego_node_img_coord):
    """
    1. 转换 pos 和 geometry 属性。
    2. 重命名节点 ID (u, v) 为新的坐标。
    """
    if ego_node_img_coord is None:
        return G

    ego_y, ego_x = ego_node_img_coord

    # --- 第一步：转换属性 (同之前的代码) ---
    for node in G.nodes():
        node_y, node_x = node
        trans_x = node_x - ego_x
        trans_y = ego_y - node_y
        G.nodes[node]['pos'] = (trans_x, trans_y)  # 更新 pos

    for u, v, data in G.edges(data=True):
        if 'geometry' in data:
            old_path = data['geometry']
            new_path = []
            for (py, px) in old_path:
                tx = px - ego_x
                ty = ego_y - py
                new_path.append((tx, ty))
            data['geometry'] = new_path

    # --- 第二步：重命名节点 ID (u, v) ---
    mapping = {}
    for node in G.nodes():
        # node 是旧的 (y, x)
        # 我们希望新的 ID 是 (x, y) 对应 Ego 坐标
        # 注意：这里我们使用的是 update 后的 pos 里的值，
        # 或者重新计算一遍 trans_x, trans_y 都可以

        # 直接拿第一步算好的 pos
        # (注意：pos 已经是 (x, y) 格式了)
        new_id = G.nodes[node]['pos']
        mapping[node] = new_id

    # copy=False 表示在原图上修改，不再返回新对象
    # 这一步之后，所有的 u, v 都变成了 (x, y) 格式
    networkx.relabel_nodes(G, mapping, copy=False)

    return G


def get_distance_to_nearest_curve(G, lateral_threshold=2.0, lookahead_distance=100):
    """
    计算车辆到最近弯道入口的距离。

    Args:
        G: NetworkX 图，由边属性 'geometry' 包含路径点。
        lateral_threshold: float, 横向偏离多少单位被视为弯道 (默认 2.0，可视作半个车道宽)。
        lookahead_distance: float, 向前搜索的最大纵向距离 (默认 100 单位)。

    Returns:
        float: 到弯道入口的距离。如果没有检测到弯道，返回 None。
        tuple: 弯道入口点的坐标 (x, y)，用于可视化调试。如果没有，返回 None。
    """

    # 1. 寻找 Ego 节点
    ego_node = None
    for node, data in G.nodes(data=True):
        if data.get('type') == 'ego':
            ego_node = node
            break
    if ego_node is None:
        pos = networkx.get_node_attributes(G, 'pos')
        nodes = list(G.nodes())
        dists = [np.linalg.norm(np.array(pos[n]) - np.array([0, 0])) for n in nodes]
        ego_node = nodes[np.argmin(dists)]

    # 2. 收集前方路径点
    path_points = []

    # 获取从 ego 出发的 BFS 边序列
    try:
        bfs_edges = list(networkx.bfs_edges(G, source=ego_node, depth_limit=10))
    except:
        bfs_edges = []
    initial_edges = list(G.out_edges(ego_node)) if G.is_directed() else list(G.edges(ego_node))
    all_edges = initial_edges + bfs_edges

    for u, v in all_edges:
        data = G.get_edge_data(u, v)
        if 'geometry' in data:
            pts = np.array(data['geometry'])
            # 过滤掉 y < 0 (车后) 和 太远的点
            valid_mask = (pts[:, 1] > 0) & (pts[:, 1] < lookahead_distance)
            valid_pts = pts[valid_mask]

            if len(valid_pts) > 0:
                path_points.append(valid_pts)

    if not path_points:
        return float('inf'), None

    # 合并所有点
    all_points = np.vstack(path_points)

    # 3. 找出所有“横向偏离过大”的候选点 (弯道上的点)
    # 计算每个点的 x 绝对值
    abs_x = np.abs(all_points[:, 0])

    # 这是一个布尔掩码，标记出哪些点在弯道区域里
    curve_mask = abs_x > lateral_threshold

    # 提取出所有弯道候选点
    curve_candidates = all_points[curve_mask]

    if len(curve_candidates) == 0:
        # 没有点超过阈值，全是直道
        return float('inf'), None

    # 4. 在候选点中，找到离原点 (0,0) 欧几里得距离最近的那个点
    # 计算每个候选点的距离: sqrt(x^2 + y^2)
    distances = np.linalg.norm(curve_candidates, axis=1)

    # 找到最小距离的索引
    min_idx = np.argmin(distances)

    nearest_dist = distances[min_idx]
    nearest_point = curve_candidates[min_idx]

    return nearest_dist, nearest_point


def get_bev_road_map_branch_distance(
        road_img_purple_bgr,
        lateral_threshold=10,  # 横向偏离的阈值（像素）
        lookahead_distance=100,  # ego 前方的探测距离
        vis_preprocess=False,
):
    # 紫色的道路条带图 -> 转化为 graph -> 计算前方的岔路距离

    # --- 1. 预处理 ---
    img = cv2.cvtColor(road_img_purple_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    kernel_morph = np.ones((8, 8), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_morph)
    dilated_mask = cv2.dilate(closed_mask, kernel_morph, iterations=2)

    # === 边缘平滑处理 ===
    # 宽路产生的毛刺通常是因为边缘有直角或噪点。
    # 使用高斯模糊将边缘磨圆，原来的矩形直角会变圆弧，骨架就不会分叉了。

    # 31x31 是模糊核大小，路越宽、噪点越多，这个值可以越大（必须是奇数）
    blurred_mask = cv2.GaussianBlur(dilated_mask, (31, 31), 0)

    # 再次二值化，切回清晰的 0/255
    # 127 是阈值，大于它的变白
    _, smooth_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

    # --- 2. 初始骨架化 ---
    raw_skeleton = skeletonize(smooth_mask // 255, method='lee').astype(np.uint8)

    # --- 3. 骨架裁剪 (Pruning) ---
    # 阈值设为 (像素)，小于这个长度的分支会被切掉
    pruned_skeleton = prune_skeleton(raw_skeleton, min_length=50)

    preprocess_img = None
    if vis_preprocess:
        preprocess_img = draw_pre_process_image(
            road_img_purple_bgr,
            dilated_mask,
            smooth_mask,
            raw_skeleton,
            pruned_skeleton
        )

    skel_img = pruned_skeleton * 255

    # 2. 构建图并获取 Ego 位置
    G, ego_node_raw = build_graph_from_skeleton(skel_img, smoothing_factor=3.0)

    # 3. 坐标转换
    # 现在 G 中的 pos 和 geometry 都是以 car 为原点，单位为像素的直角坐标系 (x, y)
    transform_graph_to_ego_frame(G, ego_node_raw)

    try:
        dist, curve_point = get_distance_to_nearest_curve(
            G, lateral_threshold=lateral_threshold, lookahead_distance=lookahead_distance
        )
    except Exception as e:
        dist = float('inf')
        curve_point = None
        logging.warning(f'road_geometry.py -> get_bev_road_map_branch_distance :{e}')

    if not vis_preprocess:
        return G, dist, curve_point
    else:
        return G, dist, curve_point, preprocess_img

