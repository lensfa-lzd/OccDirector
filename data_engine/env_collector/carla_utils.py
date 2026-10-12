import copy
import logging
import random
from queue import Queue

import carla
import numpy as np
import torch
from carla import VehicleLightState as vls

from occinteract.tools.carla_sensing import (
    BEV_CAMERA, FRONT_CAMERA, LIDAR, OCCUPANCY, CARLA_TO_UNIOCC_MAP, get_lidar_lookup_table,
)
from occinteract.tools.shape_container import ShapeContainerGPU


class CarlaClient:
    def __init__(self, arg, delta):
        self.arg = arg
        self.delta = delta
        self.device = arg['device']
        self.timeout = arg['timeout']

        # 连接 Client
        self.client = carla.Client(arg["host"], arg["port"])
        self.client.set_timeout(self.timeout)

        # 加载世界
        if arg["new_world"]:
            self.world = self.client.load_world(arg["new_world"])
        else:
            self.world = self.client.get_world()

        # 设置世界参数
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = delta
        settings.synchronous_mode = True
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)

        # 初始化变量
        self.traffic_manager = None
        self.vehicle = None
        self.vehicles_list = []
        self.front_camera = None
        self.bev_camera = None
        self.lidars = {}

        # 队列
        self.lidar_queue = Queue()
        self.front_camera_queue = Queue()
        self.bev_camera_queue = Queue()

        # 初始化各个模块
        self._init_traffic_manager()
        self._init_traffic_light()
        self._spawn_ego_vehicle()
        self._create_traffic()
        self._spawn_monitor()

        # Occ 转换器初始化
        self.voxel_grid = ShapeContainerGPU(
            grid_size=np.array(OCCUPANCY['grid_size']),
            min_bound=np.array(OCCUPANCY['min_bound'], dtype=np.float32),
            max_bound=np.array(OCCUPANCY['max_bound'], dtype=np.float32),
            num_classes=OCCUPANCY['num_classes'],
            device=self.device
        )
        self.label_lookup_table = torch.from_numpy(get_lidar_lookup_table()).to(self.device)

    def _init_traffic_manager(self):
        # --- 配置 Traffic Manager ---
        tm_port = self.arg['tm_port']
        self.traffic_manager = self.client.get_trafficmanager(tm_port)

        # 关键配置：同步模式
        self.traffic_manager.set_synchronous_mode(True)

        # 关键配置：混合物理模式 (Hybrid Physics Mode)
        # 这允许在仅渲染主角周围车辆的物理，远处车辆只进行位移计算，极大减少CPU消耗
        self.traffic_manager.set_hybrid_physics_mode(True)
        self.traffic_manager.set_hybrid_physics_radius(70.0)  # 只有70米内的车辆有物理碰撞计算

        # 关键配置：自动重生
        # 如果车辆掉落地图外或者卡死，TM会自动重生它们，保证交通流持续
        self.traffic_manager.set_respawn_dormant_vehicles(True)

        # 设置所有车辆与前车的安全距离（米）
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        # 设置随机化种子以便复现
        if self.arg.get('seed', None):
            self.traffic_manager.set_random_device_seed(self.arg['seed'])
            random.seed(self.arg['seed'])

    def _init_traffic_light(self):
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        for traffic_light in traffic_lights:
            # 1. 强制设置为绿灯
            traffic_light.set_state(carla.TrafficLightState.Green)
            # 2. 冻结状态 (时间不再倒数，永久保持当前状态)
            traffic_light.freeze(True)

    def _spawn_ego_vehicle(self):
        # --- 生成 Ego Vehicle (主角) ---
        blueprint_library = self.world.get_blueprint_library()
        # 简单处理，假设第一个就是需要的
        vehicle_bp = blueprint_library.filter('model3')[0]
        vehicle_transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)
        # 别忘了让 TM 接管主角的 Autopilot
        self.vehicle.set_autopilot(True, self.arg['tm_port'])

    def get_actor_blueprints(self, filter_pattern, generation):
        bps = self.world.get_blueprint_library().filter(filter_pattern)

        if generation.lower() == "all":
            return bps

        if len(bps) == 1:
            return bps

        try:
            int_generation = int(generation)
            if int_generation in [1, 2]:
                bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
                return bps
            else:
                print("   Warning! Actor Generation is not valid. No actor will be spawned.")
                return []
        except:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []

    def _create_traffic(self):
        """
        生成背景交通流（车辆）
        """
        num_vehicle = self.arg.get('num_vehicles', np.random.randint(1, 10))

        # 1. 获取并筛选车辆蓝图
        blueprints = self.get_actor_blueprints("vehicle.*", "All")

        # 过滤掉不安全的车辆（两轮车、特殊车辆等）
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']  # 确保是汽车

        # 手动排除已知可能导致问题的特定车型
        excluded_models = ['microlino', 'carlacola', 'cybertruck', 't2', 'sprinter', 'firetruck', 'ambulance', 'isetta']
        blueprints = [x for x in blueprints if not any(model in x.id for model in excluded_models)]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        # 2. 获取生成点
        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        # 检查生成点数量是否足够
        if num_vehicle < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif num_vehicle > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, num_vehicle, number_of_spawn_points)
            num_vehicle = number_of_spawn_points

        # 3. 准备生成指令
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        batch = []

        for n, transform in enumerate(spawn_points):
            if n >= num_vehicle:
                break

            blueprint = random.choice(blueprints)

            # 随机化颜色和驾驶员
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)

            blueprint.set_attribute('role_name', 'autopilot')

            # 设置车灯状态
            light_state = vls.Position | vls.LowBeam

            # 构建 Batch 命令：生成 -> 设为自动驾驶 -> 设车灯
            batch.append(SpawnActor(blueprint, transform)
                         .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port()))
                         .then(SetVehicleLightState(FutureActor, light_state)))

        # 4. 执行生成
        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

    def _spawn_monitor(self):
        blueprint_library = self.world.get_blueprint_library()

        # Front Camera
        camera_rgb_bp = blueprint_library.find('sensor.camera.rgb')
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
            camera_rgb_bp, front_transform, attach_to=self.vehicle
        )
        self.front_camera.listen(lambda data: self.front_camera_queue.put(data))

        # BEV Camera
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
            camera_rgb_bp, bev_transform, attach_to=self.vehicle
        )
        self.bev_camera.listen(lambda data: self.bev_camera_queue.put(data))

        # Lidars
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
            lidar_bp.set_attribute('rotation_frequency', str(1.0 / self.world.get_settings().fixed_delta_seconds))
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
                attach_to=self.vehicle
            )
            lidar.listen(lambda data, map_key=key: self.lidar_queue.put((map_key, data)))
            self.lidars[key] = lidar

    @staticmethod
    def parse_image(carla_image):
        np_img = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
        np_img = copy.deepcopy(np_img)
        np_img = np.reshape(np_img, (carla_image.height, carla_image.width, 4))
        np_img = np_img[:, :, :3][:, :, ::-1]
        return np_img

    def get_lidar_data(self):
        """Helper to collect all lidar data from queue"""
        lidar_data = []
        for _ in range(len(self.lidars)):
            # 注意：这里的 timeout 需要与外部循环一致，或者作为一个小偏移
            # 但因为是同步模式，通常队列里应该已经有数据了，或者即将有
            lidar_data.append(self.lidar_queue.get(True, self.timeout))
        return lidar_data

    def semantic_lidar_to_occupancy(self, point_cloud_list):
        self.voxel_grid.reset_grid()
        actor_list = self.world.get_actors()

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
            # ids = torch.from_numpy(ids_np).to(device) # 如果不需要在 GPU 上判断 ID，可省

            # --- 预处理 ---
            # 1. 移除 Ego Vehicle 自身的点 (假设 vehicle_id 已知)
            # 注意：此处 ids_np 比较建议在 CPU 做，或者上传 GPU 做
            non_ego_mask = torch.from_numpy(np.array(ids_np != self.vehicle.id)).to(self.device)
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

    def destroy(self):
        # 恢复世界设置
        if self.world:
            self.world.apply_settings(self.original_settings)

        # # 停止并销毁传感器
        # if self.front_camera:
        #     self.front_camera.stop()
        #     self.front_camera.destroy()
        # if self.bev_camera:
        #     self.bev_camera.stop()
        #     self.bev_camera.destroy()
        #
        # for lidar in self.lidars.values():
        #     if lidar:
        #         lidar.stop()
        #         lidar.destroy()

        # 销毁车辆
        if self.vehicle:
            self.vehicle.destroy()

        if self.vehicles_list:
            _ = self.client.apply_batch_sync(
                [carla.command.DestroyActor(x) for x in self.vehicles_list],
                True
            )

        # 停止 recorder
        if self.client:
            self.client.stop_recorder()

        # 退出 TM 同步
        if self.traffic_manager:
            self.traffic_manager.set_synchronous_mode(False)

        self.world.tick()

        # 清理显存
        self.voxel_grid.voxels = None
        # torch.cuda.empty_cache()
