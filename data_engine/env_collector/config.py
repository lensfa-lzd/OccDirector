"""Collection configuration: CARLA maps and host resource pool.

Map paths are for CARLA 0.9.15. ``Town03_Opt`` is excluded upstream because it
breaks actor spawning.

Host resources (CARLA server addresses) live in a gitignored ``hosts.json``
(see ``data_engine/hosts.example.json``).

    [
        {"ip": "10.0.0.1", "workers": 8, "base_port": [2000, 20000]}
    ]

Each host entry spawns ``workers`` collector slots; CARLA listens on
``base_port[0] + 100 * i`` and the Traffic Manager on ``base_port[1] + 10 * i``
for the i-th worker.
"""
MAPS = [
    "/Game/Carla/Maps/Town01",
    "/Game/Carla/Maps/Town01_Opt",
    "/Game/Carla/Maps/Town02",
    "/Game/Carla/Maps/Town02_Opt",

    "/Game/Carla/Maps/Town03",
    # "/Game/Carla/Maps/Town03_Opt",  # broken map: actor spawning fails
    "/Game/Carla/Maps/Town04",
    "/Game/Carla/Maps/Town04_Opt",

    "/Game/Carla/Maps/Town05",
    "/Game/Carla/Maps/Town05_Opt",
    "/Game/Carla/Maps/Town06",
    "/Game/Carla/Maps/Town07",

    "/Game/Carla/Maps/Town10HD",
    "/Game/Carla/Maps/Town10HD_Opt",
]


