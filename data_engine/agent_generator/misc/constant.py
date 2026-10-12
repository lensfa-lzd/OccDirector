# Derived from https://github.com/basiclab/TTSG (Text-to-Traffic-Scene Generation, arXiv 2409.09575),
# modified for OccInteract-Agent occupancy collection.
OBJECT_UNUSED = [
    "DashedSingleYellow",
    "Stencil_ArrowType4L",
    "oneWayL_Assembly",
    "ChevronRegion",
    "LaneR_L2_Assembly",
    "Stencil_STOP",
    "NoTurnLeft_Round_Assembly",
    "DoNotEnter_Assembly",
    "LaneR_L3Assembly",
    "LaneR_L1_Assembly",
    "Stencil_ArrowType4R",
    "SignPost_10ft",
    "NoTurn_Assembly",
    "LaneR_L0_Assembly",
    "oneWayR_Assembly",
    "MichiganTurnAssembly",
]

OBJECT_SEARCH_DICT = {
    "speed_30": ["speed_30", "Speed_30", "Speedlimit30Assembly"],
    "speed_40": ["Speed_40", "Speedlimit40Assembly"],
    "speed_60": ["speed_60", "Speed_60", "Speedlimit60Assembly"],
    "speed_90": ["Speed_90", "Speedlimit90Assembly"],
    "parallel_open_crosswalk": ["SimpleCrosswalk"],
    "ladder_crosswalk": ["LadderCrosswalk"],
    "continental_crosswalk": ["ContinentalCrosswalk"],
    "dashed_single_white": ["DashedSingleWhite"],
    "solid_single_white": ["SolidSingleWhite"],
    "crosswalk": [
        "SimpleCrosswalk",
        "LadderCrosswalk",
        "ContinentalCrosswalk",
        "SolidSingleWhite",
        "DashedSingleWhite",
    ],
    "stop_line": ["StopLine"],
    "stop_sign_on_road": ["StopSign", "Stencil_STOP"],
}

SIGNAL_UNUSED = []

SIGNAL_SEARCH_DICT = {
    "traffic_light": ["Signal_3Light_Post01"],
    "stop": ["Sign_Stop"],
    "yield": ["Sign_Yield"],
}

ACTION = [
    "turn_left",
    "turn_right",
    "go_straight",
    "change_lane_to_left",
    "change_lane_to_right",
    "stop",
    "block_the_ego",
    "cross_the_road",
    "on_the_sidewalk",
]
SPEED_MOVENT = ["slow down", "speed up", "constant"]
AGENT_BEHAVIOR = ["cautious", "normal", "aggressive"]
AGENT_TYPE = [
    "ambulance",
    "police",
    "firetruck",
    "bus",
    "truck",
    "motorcycle",
    "car",
    "pedestrian",
    "cyclist",
]
WEATHER = [
    "ClearNight",
    "ClearNoon",
    "ClearSunset",
    "CloudyNight",
    "CloudyNoon",
    "CloudySunset",
    "DustStorm",
    "HardRainNight",
    "HardRainNoon",
    "HardRainSunset",
    "MidRainSunset",
    "MidRainyNight",
    "MidRainyNoon",
    "SoftRainNight",
    "SoftRainNoon",
    "SoftRainSunset",
    "WetCloudyNight",
    "WetCloudyNoon",
    "WetCloudySunset",
    "WetNight",
    "WetNoon",
    "WetSunset",
]

ROAD_TYPE = ["driving", "sidewalk", "shoulder"]

RELATIVE_POSITION = """
- front: The agent is in front of the ego car
- back: The agent is behind the ego car
- left: The agent is on the left side of the ego car
- right: The agent is on the right side of the ego car
- front_left: The agent is in front and on the left side of the ego car
- front_right: The agent is in front and on the right side of the ego car
- back_left: The agent is behind and on the left side of the ego car
- back_right: The agent is behind and on the right side of the ego car
- road_of_left_turn: The agent is on different roads that ego car should take a left turn to reach
- road_of_right_turn: The agent is on different roads that ego car should take a right turn to reach
- road_of_straight: The agent is on different roads that ego car should go straight to reach
- at_the_destination: The agent is near at the destination of the ego car
- near_the_crosswalk: The agent is near at the crosswalk, used for pedestrian
"""

CAR_TYPE = [
    "ambulance",
    "police",
    "firetruck",
    "car",
    "truck",
    "bus",
    "motorcycle",
]
NUM_POINT_PER_CAR = 3
NULL_SPACE = 2
DISTANCE_FOR_ROUTE = 2.0
DIR_TO_COLOR = {
    "left": (0, 255, 0),
    "right": (0, 0, 255),
    "straight": (255, 0, 0),
}
LANE_TO_WAYPOINT = {
    "Driving": "get_driving",
    "Shoulder": "get_shoulder",
    "Sidewalk": "get_side_walk",
}
FRONT_CAMERA = {
    "x": -1.5,
    "y": 0.0,
    "z": 2.0,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "image_size_x": 900,
    "image_size_y": 256,
    "fov": 100,
}
BEV_CAMERA = {
    "x": 0.0,
    "y": 0.0,
    "z": 50.0,
    "roll": 0.0,
    "pitch": -90.0,
    "yaw": 0.0,
    "image_size_x": 512,
    "image_size_y": 512,
    "fov": 50.0,
}
LIDAR = {
    "line_num": 2,  # 使用多个 lidar 同时收集数据
    # lidar 总数为 (2 * line_num + 1)^2

    # 作为基准的 ego car lidar
    "z": 2.0,

    # 采集 0.4m^3 大小的 [-40m, -40m, -2.5m, 40m, 40m, 3.9m]
    # [200, 200, 16] Occupancy

    # 探测距离 (Range)
    "min_range": 20.0,

    # 垂直视场角 (FOV)
    "upper_fov": 5.0,
    "lower_fov": -25.0,

    # 线数 (Channels - 垂直分辨率)
    "channels": 32,

    # 每秒点数 (Points Per Second - 水平及总密度)
    "points_per_second": 10_0000,
}
OCCUPANCY = {  # lidar to Occupancy 设置
    "range": 40,
    "grid_size": [200, 200, 16],
    "min_bound": [-40., -40., -2.5],
    "max_bound": [40., 40., 3.9],

    # # carla 雷达语义的类别数量
    # 在 carla 0.9.15 中有 29 个类别
    # https://github.com/carla-simulator/carla/blob/0.9.15/LibCarla/source/carla/rpc/ObjectLabel.h
    #
    # enum class CityObjectLabel : uint8_t {
    #   None         =    0u,
    #   // cityscape labels
    #   Roads        =    1u,
    #   Sidewalks    =    2u,
    #   Buildings    =    3u,
    #   Walls        =    4u,
    #   Fences       =    5u,
    #   Poles        =    6u,
    #   TrafficLight =    7u,
    #   TrafficSigns =    8u,
    #   Vegetation   =    9u,
    #   Terrain      =   10u,
    #   Sky          =   11u,
    #   Pedestrians  =   12u,
    #   Rider        =   13u,
    #   Car          =   14u,
    #   Truck        =   15u,
    #   Bus          =   16u,
    #   Train        =   17u,
    #   Motorcycle   =   18u,
    #   Bicycle      =   19u,
    #   // custom
    #   Static       =   20u,
    #   Dynamic      =   21u,
    #   Other        =   22u,
    #   Water        =   23u,
    #   RoadLines    =   24u,
    #   Ground       =   25u,
    #   Bridge       =   26u,
    #   RailTrack    =   27u,
    #   GuardRail    =   28u,
    #
    #   Any          =  0xFF
    # };
    "num_classes": 29  # carla 雷达语义的类别数量
}
# carla 到 UniOcc 格式的转换映射表
# | Carla ID | Carla Label (Source) | UniOcc ID | UniOcc Label (Target) | 备注 / 映射逻辑          |
# |:--------:|:---------------------|:---------:|:----------------------|:-------------------|
# |    0     | None                 |    10     | free                  | 空置区域/无效数据          |
# |    1     | Roads                |     7     | road                  | 道路表面               |
# |    2     | Sidewalks            |     8     | walkable/terrain      | 人行道属于可行走区域         |
# |    3     | Buildings            |     9     | building              | 建筑物                |
# |    4     | Walls                |     0     | general_object        | 墙壁视为通用障碍物          |
# |    5     | Fences               |     0     | general_object        | 栅栏视为通用障碍物          |
# |    6     | Poles                |     0     | general_object        | 杆状物视为通用障碍物         |
# |    7     | TrafficLight         |     0     | general_object        | 交通设施视为通用障碍物        |
# |    8     | TrafficSigns         |     0     | general_object        | 交通设施视为通用障碍物        |
# |    9     | Vegetation           |     6     | vegetation            | 植被                 |
# |    10    | Terrain              |     8     | walkable/terrain      | 地形                 |
# |    11    | Sky                  |    10     | free                  | 天空视为非占用空间          |
# |    12    | Pedestrians          |     4     | pedestrian            | 行人                 |
# |    13    | Rider                |     4     | pedestrian            | 骑行者归类为行人/弱势群体      |
# |    14    | Car                  |     1     | vehicle               | 汽车                 |
# |    15    | Truck                |     1     | vehicle               | 卡车                 |
# |    16    | Bus                  |     1     | vehicle               | 公交车                |
# |    17    | Train                |     1     | vehicle               | 火车/轨道车辆            |
# |    18    | Motorcycle           |     3     | motorcycle            | 摩托车                |
# |    19    | Bicycle              |     2     | bicycle               | 自行车                |
# |    20    | Static               |     0     | general_object        | 静态杂物 (包含交通锥)       |
# |    21    | Dynamic              |     0     | general_object        | 动态杂物               |
# |    22    | Other                |     0     | general_object        | 其他杂物               |
# |    23    | Water                |     0     | general_object        | 水体 (通常无法行驶，视为障碍)   |
# |    24    | RoadLines            |     7     | road                  | 道路标线属于路面           |
# |    25    | Ground               |     8     | walkable/terrain      | 土地/地面              |
# |    26    | Bridge               |     0     | general_object        | 桥梁结构 (路面部分由ID 1覆盖) |
# |    27    | RailTrack            |     7     | road                  | 轨道表面 (此处视为可行驶类)    |
# |    28    | GuardRail            |     0     | general_object        | 护栏                 |
CARLA_TO_UNIOCC_MAP = {
    # --- Empty / Void ---
    0: 10,  # None         -> free (或根据需求设为 0-ignore)

    # --- Surfaces ---
    1: 7,  # Roads        -> road
    2: 8,  # Sidewalks    -> walkable/terrain

    # --- Structures ---
    3: 9,  # Buildings    -> building
    4: 0,  # Walls        -> general_object (UniOcc没有专门的Wall，归为通用物体或Building)
    5: 0,  # Fences       -> general_object

    # --- Street Objects ---
    6: 0,  # Poles        -> general_object
    7: 0,  # TrafficLight -> general_object
    8: 0,  # TrafficSigns -> general_object

    # --- Nature ---
    9: 6,  # Vegetation   -> vegetation
    10: 8,  # Terrain      -> walkable/terrain

    # --- Sky ---
    11: 10,  # Sky          -> free

    # --- Humans ---
    12: 4,  # Pedestrians  -> pedestrian
    13: 4,  # Rider        -> pedestrian (骑行者通常归类为行人/弱势道路使用者)

    # --- Vehicles ---
    14: 1,  # Car          -> vehicle
    15: 1,  # Truck        -> vehicle
    16: 1,  # Bus          -> vehicle
    17: 1,  # Train        -> vehicle
    18: 3,  # Motorcycle   -> motorcycle
    19: 2,  # Bicycle      -> bicycle

    # --- Custom / Others ---
    20: 0,  # Static       -> general_object (交通锥可能包含在此类中)
    21: 0,  # Dynamic      -> general_object
    22: 0,  # Other        -> general_object
    23: 0,  # Water        -> general_object (作为障碍物处理，或者映射为10-free)
    24: 7,  # RoadLines    -> road (路面的一部分)
    25: 8,  # Ground       -> walkable/terrain (通常指土壤或非路面地面)
    26: 0,  # Bridge       -> general_object (指桥梁结构，若指路面则已被Roads覆盖)
    27: 7,  # RailTrack    -> road (作为可行驶表面处理，或归为general_object)
    28: 0,  # GuardRail    -> general_object
}
