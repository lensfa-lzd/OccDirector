"""Build the road-graph cache (graph_cache/graph.pkl) for the agent generator.

The cache stores per-road lane/turn information extracted by driving a scan over
every bundled map on a live CARLA server (GraphManager.get_intersection_info).
Without this cache the first render pays the full 7-map scan inline; build it
once per machine and every subsequent run loads graph.pkl directly.

Usage (from the repo root, with one CARLA server running):

    python -m data_engine.agent_generator.build_graph_cache \
        --host 127.0.0.1 --port 2000
"""
import argparse
import os
import time

from data_engine.agent_generator.scene_utils.scene_client import CarlaClient

_CACHE_FILE = 'graph.pkl'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--maps', default=None,
                        help='OpenDRIVE map folder (default: bundled maps/)')
    parser.add_argument('--cache-dir', default=None,
                        help='cache directory (default: bundled graph_cache/)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--rpc-timeout', type=float, default=300.0,
                        help='per-call CARLA RPC timeout; each of the 7 map loads '
                             'can take tens of seconds on a cold server')
    parser.add_argument('--device', default='cpu',
                        help='device for the (idle) voxelizer; the scan itself is CPU/RPC bound')
    parser.add_argument('--force', action='store_true', help='rebuild even if the cache exists')
    args = parser.parse_args()

    pkg_dir = os.path.dirname(__file__)
    maps = args.maps or os.path.join(pkg_dir, 'maps')
    cache_dir = args.cache_dir or os.path.join(pkg_dir, 'graph_cache')

    if os.path.exists(os.path.join(cache_dir, _CACHE_FILE)) and not args.force:
        print(f'{cache_dir}/{_CACHE_FILE} 已存在，跳过（--force 重建）')
        return

    print(f'开始构建路网缓存: maps={maps} -> {cache_dir}/{_CACHE_FILE}')
    print('（需要逐一加载全部地图做路口扫描，冷服务器约需数分钟）')
    t0 = time.time()

    client = CarlaClient(
        input_folder=maps,
        host=args.host,
        port=args.port,
        use_cache=True,
        cache_dir=cache_dir,
        device=args.device,
        rpc_timeout=args.rpc_timeout,
    )
    # get_intersection_info 已在 __init__ 内完成扫描并落盘
    try:
        client.destroy()
    except Exception as e:
        print(f'best-effort 清理: {e}')

    assert os.path.exists(os.path.join(cache_dir, _CACHE_FILE)), '缓存未生成'
    print(f'完成，耗时 {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
