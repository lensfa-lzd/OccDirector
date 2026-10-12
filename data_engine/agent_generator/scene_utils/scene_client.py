# Derived from https://github.com/basiclab/TTSG (Text-to-Traffic-Scene Generation, arXiv 2409.09575),
# modified for OccInteract-Agent occupancy collection.
import copy
import random
import secrets
import weakref
from collections import defaultdict
from queue import Queue

import carla
import numpy as np
import torch

from data_engine.agent_generator.graph.graph_manager import GraphManager
from data_engine.agent_generator.manager import (
    AgentModelManager,
    CyclistManager,
    PedestrianManager,
    VehicleManager,
    WorldManager,
)
from occinteract.tools.carla_sensing import (
    BEV_CAMERA, FRONT_CAMERA, LIDAR, OCCUPANCY, CARLA_TO_UNIOCC_MAP, get_lidar_lookup_table,
)
from data_engine.agent_generator.scene_utils.retreival import retrieve_roads
from occinteract.tools.shape_container import ShapeContainerGPU


def inverse_format_town_name(town_name):
    if "/" in town_name:
        town_name = town_name.rsplit("/", 1)[-1]
    return town_name[:4].lower() + str(int(town_name[4:]))


def format_town_name(town_name):
    town_num = town_name[4:]
    return f"Town{town_num.zfill(2)}"


class CarlaClient:
    def __init__(
            self,
            input_folder="maps",
            host="127.0.0.1",
            port=2000,
            use_cache=True,
            cache_dir="graph_cache",
            override_option=False,
            device='cuda',
            rpc_timeout=120.0,
    ):
        self.device = device
        self.client = carla.Client(host, port)
        # 冷启动时 load_world 可能远超 10 秒（图加载 + get_intersection_info 逐图扫描）
        self.client.set_timeout(rpc_timeout)
        self.graph_manager = GraphManager(
            input_folder,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )
        self.world_delta = None
        self.world = None
        self.agent_model_manager = AgentModelManager(self.client)
        self.vehicle_manager = VehicleManager(
            self.graph_manager, override_option=override_option
        )
        self.pedestrian_manager = PedestrianManager()
        self.cyclist_manager = CyclistManager(self.graph_manager)
        self.world_manager = WorldManager()

        self.front_camera = None
        self.front_image_queue = Queue()
        self.bev_camera = None
        self.bev_image_queue = Queue()
        self.lidars = {}
        self.lidar_queue = Queue()

        self.json_file = None

        # Check the left, right, and straight lanes
        self.graph_manager.get_intersection_info(self, self.vehicle_manager)

        # Occ 采集器
        self.voxel_grid = ShapeContainerGPU(
            grid_size=np.array(OCCUPANCY['grid_size']),
            min_bound=np.array(OCCUPANCY['min_bound'], dtype=np.float32),
            max_bound=np.array(OCCUPANCY['max_bound'], dtype=np.float32),
            num_classes=OCCUPANCY['num_classes'],
            device=self.device
        )
        self.label_lookup_table = torch.from_numpy(get_lidar_lookup_table()).to(self.device)

    def get_world_delta(self):
        if self.world_delta is None:
            settings = self.world.get_settings()
            self.world_delta = settings.fixed_delta_seconds if settings.fixed_delta_seconds else 0.1

        return self.world_delta

    def set_json_file(self, json_file):
        """JSON file with the following format:
        {
            "analysis": {...},
            "retreival": {...},
            "planning": {...},
        }
        """
        self.json_file = json_file

    def draw_point(self, location, color=(255, 0, 0)):
        self.world.debug.draw_string(
            location,
            "O",
            color=carla.Color(*color),
            life_time=10,
            draw_shadow=False,
            persistent_lines=True,
        )

    def load_map(self, map_name, weather="ClearNoon"):
        if self.world is not None:
            self.clean()
            self.set_sync_mode(False, set_tm=False)
        self.world = self.client.load_world(map_name)
        if hasattr(carla.WeatherParameters, weather):
            self.world.set_weather(getattr(carla.WeatherParameters, weather))
        self.set_sync_mode(True, set_tm=False)
        self.set_manager()
        self.world.tick()
        self.set_traffic_light_time(5)

    def set_world(self, world):
        self.world = world
        self.set_manager()

    def set_manager(self):
        self.agent_model_manager.set_blueprint(self.world.get_blueprint_library())
        self.world_manager.set_world(self.world)
        self.vehicle_manager.set_new_manager(
            self.agent_model_manager, self.world_manager
        )
        self.pedestrian_manager.set_new_manager(
            self.agent_model_manager, self.world_manager
        )
        self.cyclist_manager.set_new_manager(
            self.agent_model_manager, self.world_manager
        )
        self.graph_manager.set_town_name(
            inverse_format_town_name(self.world.get_map().name)
        )

    def set_traffic_light_time(self, duration=20):
        actor_list = self.world.get_actors()
        for actor_ in actor_list:
            if isinstance(actor_, carla.TrafficLight):
                actor_.set_red_time(duration)

    def set_sync_mode(self, sync, set_tm=False):
        settings = self.world.get_settings()
        settings.synchronous_mode = sync
        if sync:
            settings.fixed_delta_seconds = 0.1
        else:
            settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        if set_tm:
            self.tm.set_synchronous_mode(sync)

    @staticmethod
    def parse_image(weak_self, carla_image, name_to_put):
        self = weak_self()
        np_img = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
        np_img = copy.deepcopy(np_img)
        np_img = np.reshape(np_img, (carla_image.height, carla_image.width, 4))
        np_img = np_img[:, :, :3][:, :, ::-1]
        getattr(self, name_to_put).put((carla_image.frame, np_img))

    def clean(self):
        self.vehicle_manager.clean()
        self.pedestrian_manager.clean()
        self.cyclist_manager.clean()
        if self.front_camera is not None:
            self.front_camera.stop()
            self.front_camera.destroy()
        if self.bev_camera is not None:
            self.bev_camera.stop()
            self.bev_camera.destroy()

        try:
            for lidar in self.lidars:
                if lidar is not None:
                    lidar.stop()
                    lidar.destroy()
        except AttributeError:
            pass

        if self.front_camera is not None or self.bev_camera is not None:
            self.world.tick()

        # voxel_grid 在 __init__ 尾部才创建，而 get_intersection_info 可能在更早的
        # load_map -> clean() 路径中调用本方法（冷缓存时），需要守卫
        if getattr(self, 'voxel_grid', None) is not None:
            self.voxel_grid.voxels = None
        torch.cuda.empty_cache()

    def sort_road_target(
            self, road_info, required_num_of_waypoints, action_list, road_type_list
    ):
        score = 0

        for action, relative_position in action_list:
            if (
                    relative_position == "road_of_straight"
                    and road_info["have_straight_from"]
            ):
                if action == "go_straight" and road_info["have_opposite"]:
                    score += 1
                elif action == "turn_left" and road_info["can_turn_right"]:
                    score += 1
                elif action == "turn_right" and road_info["can_turn_left"]:
                    score += 1
            elif (
                    relative_position == "road_of_left_turn" and road_info["have_left_from"]
            ):
                if action == "go_straight" and road_info["can_turn_right"]:
                    score += 1
                elif action == "turn_left" and road_info["can_go_straight"]:
                    score += 1
                elif action == "turn_right" and road_info["have_opposite"]:
                    score += 1
            elif (
                    relative_position == "road_of_right_turn"
                    and road_info["have_right_from"]
            ):
                if action == "go_straight" and road_info["can_turn_left"]:
                    score += 1
                elif action == "turn_left" and road_info["have_opposite"]:
                    score += 1
                elif action == "turn_right" and road_info["can_go_straight"]:
                    score += 1
            elif not relative_position.startswith("road"):
                if action == "go_straight" and road_info["can_go_straight"]:
                    score += 1
                elif action == "turn_left" and road_info["can_turn_left"]:
                    score += 1
                elif action == "turn_right" and road_info["can_turn_right"]:
                    score += 1
        for road_type in road_type_list:
            if road_type == "sidewalk" and road_info["have_sidewalk"]:
                score += 1
            elif road_type == "shoulder" and road_info["have_shoulder"]:
                score += 1
        if road_info["num_of_waypoints"] < required_num_of_waypoints:
            score = 0
        return score

    def get_valid_road(
            self, road_condition, agent_type_list, action_list, road_type_list
    ):
        road = retrieve_roads(self.graph_manager.graph, road_condition)
        if len(road) == 0:
            return None, None, None, None

        required_num_of_waypoints = self.vehicle_manager.compute_approximate_num_points(
            agent_type_list, max(road_condition["number_of_lanes"], 1)
        )

        road_target = []
        check_set = defaultdict(set)
        for node_id, direction, node_info in road:
            town_name, road_id, _ = self.graph_manager.node_id_to_town_road_id(node_id)
            if road_id in check_set[town_name]:
                continue
            check_set[town_name].add(road_id)
            road_target.append(
                (format_town_name(town_name), [road_id], direction, node_info)
            )
        score_list = sorted(
            [
                (
                    self.sort_road_target(
                        x[3], required_num_of_waypoints, action_list, road_type_list
                    ),
                    idx,
                )
                for idx, x in enumerate(road_target)
            ],
            reverse=True,
        )
        max_score = score_list[0][0]

        length_having_same_score = sum([score == max_score for score, _ in score_list])
        road_target = [
            road_target[road_idx]
            for _, road_idx in score_list[:length_having_same_score]
        ]

        diff_number_of_lane = [
            4 - abs(road_info["number_of_lane"] - road_condition["number_of_lanes"])
            for *_, road_info in road_target
        ]
        total = sum(diff_number_of_lane)
        diff_number_of_lane = [diff / total for diff in diff_number_of_lane]
        select_target = np.random.choice(
            range(len(diff_number_of_lane)), 1, p=diff_number_of_lane
        ).squeeze()
        town_name, road_id, direction, road_info = road_target[select_target]
        return town_name, road_id, direction, road_info

    def spawn_ego_monitor(self):
        # Create sensor
        weak_self = weakref.ref(self)
        camera_rgb_bp = self.agent_model_manager.get_blueprint_from_name(
            "sensor.camera.rgb"
        )
        camera_rgb_bp.set_attribute("image_size_x", str(FRONT_CAMERA["image_size_x"]))
        camera_rgb_bp.set_attribute("image_size_y", str(FRONT_CAMERA["image_size_y"]))
        camera_rgb_bp.set_attribute("fov", str(FRONT_CAMERA["fov"]))
        front_transform = carla.Transform(
            carla.Location(
                x=FRONT_CAMERA["x"],
                y=FRONT_CAMERA["y"],
                z=FRONT_CAMERA["z"],
            ),
            carla.Rotation(
                roll=FRONT_CAMERA["roll"],
                pitch=FRONT_CAMERA["pitch"],
                yaw=FRONT_CAMERA["yaw"],
            ),
        )
        self.front_camera = self.world.spawn_actor(
            camera_rgb_bp, front_transform, attach_to=self.vehicle_manager.vehicles[0]
        )
        self.front_camera.listen(
            lambda image: self.parse_image(weak_self, image, "front_image_queue")
        )
        camera_rgb_bp.set_attribute("image_size_x", str(BEV_CAMERA["image_size_x"]))
        camera_rgb_bp.set_attribute("image_size_y", str(BEV_CAMERA["image_size_y"]))
        camera_rgb_bp.set_attribute("fov", str(BEV_CAMERA["fov"]))
        bev_transform = carla.Transform(
            carla.Location(
                x=BEV_CAMERA["x"],
                y=BEV_CAMERA["y"],
                z=BEV_CAMERA["z"],
            ),
            carla.Rotation(
                roll=BEV_CAMERA["roll"],
                pitch=BEV_CAMERA["pitch"],
                yaw=BEV_CAMERA["yaw"],
            ),
        )
        self.bev_camera = self.world.spawn_actor(
            camera_rgb_bp, bev_transform, attach_to=self.vehicle_manager.vehicles[0]
        )
        self.bev_camera.listen(
            lambda image: self.parse_image(weak_self, image, "bev_image_queue")
        )

        # 参考 carla sc 的方式，使用多个激光雷达同时采集数据
        # 这里使用均匀插值布置雷达
        lidar_gap = OCCUPANCY['range'] / (LIDAR['line_num'] + 1)
        offsets_dict = {}
        for x_idx in range(-LIDAR['line_num'], LIDAR['line_num'] + 1):
            for y_idx in range(-LIDAR['line_num'], LIDAR['line_num'] + 1):
                offsets_dict[f'{x_idx}_{y_idx}'] = (
                    x_idx * lidar_gap,
                    y_idx * lidar_gap,
                    LIDAR['z'],
                )

                # 在旁边随机再安装雷达
                for i in range(4):
                    offsets_dict[f'{x_idx}_{y_idx}_r{i}'] = (
                        x_idx * lidar_gap + 5 * np.random.randint(-1, 1),
                        y_idx * lidar_gap + 5 * np.random.randint(-1, 1),
                        LIDAR['z'] + np.random.randint(-1, 1),
                    )

        for key, offsets in offsets_dict.items():
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('upper_fov', str(LIDAR["upper_fov"]))
            lidar_bp.set_attribute('lower_fov', str(LIDAR["lower_fov"]))
            lidar_bp.set_attribute('channels', str(LIDAR["channels"]))
            lidar_bp.set_attribute('range', str(max(2 * lidar_gap, LIDAR['min_range'])))
            lidar_bp.set_attribute('rotation_frequency', str(1.0 / self.get_world_delta()))
            lidar_bp.set_attribute('points_per_second', str(LIDAR["points_per_second"]))

            lidar = self.world.spawn_actor(
                lidar_bp,
                carla.Transform(
                    carla.Location(
                        x=offsets[0],
                        y=offsets[1],
                        z=offsets[2],
                    )
                ),
                attach_to=self.vehicle_manager.vehicles[0]
            )
            lidar.listen(lambda data, map_key=key: self.lidar_queue.put((map_key, data)))
            self.lidars[key] = lidar

        self.world.tick()

    def set_seed(self, seed=None):
        if seed is None:
            seed = secrets.randbelow(1_000_000_000)
        np.random.seed(seed)
        random.seed(seed)

    def spawn_all_agent(
            self,
            road_id,
            agent_info,
            direction,
            at_junction=False,
            ego_agents=None,
            ego_destination=None,
    ):
        if ego_agents is None:
            ego_waypoint = self.vehicle_manager.spawn_ego_car(
                road_id, agent_info, direction, at_junction
            )
        else:
            ego_waypoint = self.world_manager.get_waypoint_from_location(
                ego_agents.get_location(), carla.LaneType.Driving
            )
            self.vehicle_manager.add_pre_spawn_ego(
                ego_agents,
                ego_waypoint,
                agent_info,
                destination=ego_destination,
            )

        pedestrian_agents = []
        cyclist_agents = []
        for agent in agent_info:
            if agent["type"] == "pedestrian":
                pedestrian_agents.append(agent)
            elif agent["type"] == "cyclist":
                cyclist_agents.append(agent)

        if ego_waypoint is not None:
            self.spawn_ego_monitor()
            self.vehicle_manager.spawn_other_cars(
                agent_info, self.vehicle_manager.vehicles[0]
            )

            self.pedestrian_manager.set_pos_id_to_waypoints(self.vehicle_manager)
            self.pedestrian_manager.spawn_pedestrians(
                pedestrian_agents, ego_agent=self.vehicle_manager.vehicle_agent[0]
            )

            self.cyclist_manager.set_pos_id_to_waypoints(self.vehicle_manager)
            self.cyclist_manager.spawn_cyclists(
                cyclist_agents, ego_agent=self.vehicle_manager.vehicle_agent[0]
            )
        return ego_waypoint

    def check_finish(self, timeout=10.0, tick_world=True):
        vehicle_done = self.vehicle_manager.run_step()
        self.cyclist_manager.run_step()
        self.pedestrian_manager.run_step()
        if tick_world:
            self.world.tick()
        if not self.front_image_queue.empty():
            _, front_image = self.front_image_queue.get(True, timeout)
        else:
            front_image = None

        if not self.bev_image_queue.empty():
            _, bev_image = self.bev_image_queue.get(True, timeout)
        else:
            bev_image = None

        if bev_image is not None and front_image is not None:
            lidar_data = self.get_lidar_data(timeout)
            if len(lidar_data) == 0:
                lidar_data = None
        else:
            lidar_data = None

        return vehicle_done, {"front_image": front_image, "bev_image": bev_image, 'lidar_data': lidar_data}

    def destroy(self, set_sync=True):
        try:
            self.clean()
            if set_sync:
                self.set_sync_mode(False)
        except Exception as e:
            print(e)

    def get_lidar_data(self, timeout):
        lidar_data = []
        for _ in range(len(self.lidars)):
            lidar_data.append(self.lidar_queue.get(True, timeout))
        return lidar_data

    ######### START
    # Modified from https://github.com/UMich-CURLY/3DMapping
    def semantic_lidar_to_occupancy(self, point_cloud_list):
        self.voxel_grid.reset_grid()
        actor_list = self.world.get_actors()
        vehicle_id = self.vehicle_manager.vehicles[0]

        def parse_point_cloud(idx, pts_raw):
            # --- 解析数据 (混合 CPU/GPU) ---
            # 利用 Numpy 快速读取 dtype 结构，然后转 Tensor
            raw_data = np.frombuffer(pts_raw.raw_data, dtype=np.dtype(
                [
                    ('x', np.float32), ('y', np.float32), ('z', np.float32),
                    ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)
                ]
            ))

            # 转为 Tensor (这是唯一的数据传输开销)
            # 构造点云矩阵 (N, 3)
            pts_np = np.stack([raw_data['x'], raw_data['y'], raw_data['z']], axis=1)
            pts = torch.from_numpy(pts_np).to(self.device)

            labels_np = raw_data['ObjTag']
            labels = torch.from_numpy(labels_np.copy()).to(self.device).long()  # Long required for indexing

            ids_np = raw_data['ObjIdx']
            # ids = torch.from_numpy(ids_np).to(self.device) # 如果不需要在 GPU 上判断 ID，可省

            # --- 预处理 ---
            # 1. 移除 Ego Vehicle 自身的点 (假设 vehicle_id 已知)
            # 注意：此处 ids_np 比较建议在 CPU 做，或者上传 GPU 做
            non_ego_mask = torch.from_numpy(np.array(ids_np != vehicle_id)).to(self.device)
            pts = pts[non_ego_mask]
            labels = labels[non_ego_mask]

            # 2. Add Noise (GPU 极其擅长)
            noise = torch.empty_like(pts).uniform_(-0.01, 0.01)
            pts += noise

            lidar_id = self.lidars[idx].id
            lidar_loc = actor_list.find(lidar_id).get_transform()
            location = np.array(lidar_loc.get_matrix())
            location = torch.from_numpy(location).to(self.device).float()

            return pts, labels, location

        data = {}
        for idx, point_cloud in point_cloud_list:
            data[idx] = parse_point_cloud(idx, point_cloud)

        # 1. 准备 Ego 变换矩阵
        # 假设 transforms_dict['ego'] 是当前帧主车的 global transform matrix
        ego_pose_np = data['0_0'][2].cpu().numpy()  # 假设 '0_0' 是主车/参考坐标系
        to_ego_np = np.linalg.inv(ego_pose_np)
        to_ego = torch.from_numpy(to_ego_np).float().to(self.device)

        for idx, point_cloud in point_cloud_list:
            points, labels, lidar_to_world = data[idx]

            # --- 计算变换矩阵 ---
            # 计算从 Lidar 坐标系 -> Ego 坐标系的矩阵
            # T_lidar_to_ego = inv(T_ego_to_world) * T_lidar_to_world
            if idx != '0_0':
                # 矩阵乘法: (4,4) @ (4,4)
                inv_transform = torch.matmul(to_ego, lidar_to_world)
            else:
                inv_transform = torch.eye(4, device=self.device)

            # 提取旋转和平移
            # inv_transform[:3, :3] 是旋转, inv_transform[:3, 3] 是平移
            R = inv_transform[:3, :3]  # (3, 3)
            # 注意: 常规变换是 R @ p + t (若 p 是列向量) 或者 p @ R.T + t (若 p 是行向量)
            # 原代码 np.matmul(points, inv_transform[:3, :3]) 对应 p @ R
            # 根据数学定义，如果要将 point (N,3) 旋转，应该是 points @ R.T
            # 这里需保持原代码逻辑: np.matmul(points, R) -> points @ R
            t = inv_transform[:3, 3]  # (3, )

            # --- Ray Tracing (GPU) ---
            # 这一步数据膨胀几十倍，必须在显存操作
            grid_points, grid_labels = self.voxel_grid.ray_trace_batch(points, labels)

            # --- 坐标变换 (转到 Ego 系) ---
            # (M, 3) @ (3, 3) + (3,)
            transformed_points = torch.matmul(grid_points, R) + t

            # --- 填入网格 (GPU) ---
            self.voxel_grid.add_points_to_grid(transformed_points, grid_labels)

        # --- 获取结果 ---
        result_labels = self.voxel_grid.get_final_labels()

        # 映射最终标签 (Uniocc 格式)
        final_output = self.label_lookup_table[result_labels.long()]
        return final_output.cpu().numpy().astype(np.uint8)
    ######### END
