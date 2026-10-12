"""Merge VLM caption JSON outputs into training-ready pkl files.

Converts the ParallelProcessor JSON output into the list-of-dict pkl format
consumed by the OccInteract loaders:

* ``turn_point``      — [{'scene_name', 'key_frame_id', 'caption'}, ...]
* ``critical_score``  — [{'scene_name', 'caption'}, ...] (sorted by scene_name)

Usage (from the repo root):

    python -m data_engine.env_collector.merge_captions \
        --mode turn_point \
        --input output/captions/turn_caption.json \
        --output output/captions/turn_caption.pkl

    python -m data_engine.env_collector.merge_captions \
        --mode critical_score \
        --input output/captions/scene_with_critical_score.json \
        --output output/captions/scene_with_critical_score.pkl
"""
import argparse
import json
import pickle


def merge_turn_point(input_json: str, output_pkl: str) -> int:
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    caption = []
    for it in data:
        task_cfg = json.loads(it['task_cfg_str'])
        scene_name = task_cfg['key']

        result = json.loads(it['result'])
        for key_frame_id, caption_detail in result.items():
            caption.append(
                {
                    'scene_name': scene_name,
                    'key_frame_id': key_frame_id,
                    'caption': caption_detail,
                }
            )

    with open(output_pkl, 'wb') as f:
        pickle.dump(caption, f)
    return len(caption)


def merge_turn_point_index(input_json: str, output_pkl: str) -> int:
    """Convert the geometry pipeline's turn_point.json into the pkl consumed by
    the turn_point VLM pipeline: {scene_name: [(frame_id, dist), ...]}."""
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    index = {}
    for it in data:
        task_cfg = json.loads(it['task_cfg_str'])
        scene_name = task_cfg['key']

        m_points = json.loads(it['result'])
        index[scene_name] = [(int(fid), dist) for fid, dist in m_points]

    with open(output_pkl, 'wb') as f:
        pickle.dump(index, f)
    return len(index)


def merge_critical_score(input_json: str, output_pkl: str) -> int:
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    caption = []
    for it in data:
        task_cfg = json.loads(it['task_cfg_str'])
        scene_name = task_cfg['key']

        result = json.loads(it['result'])
        caption.append(
            {
                'scene_name': scene_name,
                'caption': result,
            }
        )

    caption = sorted(caption, key=lambda x: x['scene_name'])

    with open(output_pkl, 'wb') as f:
        pickle.dump(caption, f)
    return len(caption)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--mode', required=True,
                        choices=['turn_point', 'turn_point_index', 'critical_score'])
    parser.add_argument('--input', required=True, help='caption output JSON from run_caption.py')
    parser.add_argument('--output', required=True, help='destination pkl path')
    args = parser.parse_args()

    if args.mode == 'turn_point':
        count = merge_turn_point(args.input, args.output)
    elif args.mode == 'turn_point_index':
        count = merge_turn_point_index(args.input, args.output)
    else:
        count = merge_critical_score(args.input, args.output)
    print(f'写入 {count:,} 条标注到 {args.output}')


if __name__ == '__main__':
    main()
