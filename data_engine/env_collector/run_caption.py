"""Run VLM captioning pipelines over collected OccInteract-Env scenes.

Three pipelines are available (run one at a time via --pipeline):

* ``geometry``           — non-LLM: extract turn points from occupancy BEV road
                           skeletons, optionally writing a scene index pkl.
* ``turn_point``         — VLM: classify the ego action at each geometry turn
                           point (STRAIGHT/LEFT/RIGHT) and describe the scene.
* ``critical_score``     — VLM: whole-scene caption with a 0-10 criticality
                           score (used for the OccInteract-Agent subset).

The OpenAI-compatible VLM endpoint is configured through the environment
(``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` / ``OPENAI_MODEL``); a ``.env`` file
in the repo root is loaded automatically.

Usage examples (from the repo root):

    python -m data_engine.env_collector.run_caption \
        --pipeline turn_point \
        --data-path /data/occ_collect \
        --info-file scene_infos.pkl --turn-point-file turn_point.pkl \
        --save-path output/captions --save-name turn_caption.json

    python -m data_engine.env_collector.run_caption \
        --pipeline critical_score \
        --data-path /data/occ_collect \
        --save-path output/captions --save-name scene_with_critical_score.json
"""
import argparse
import os

from dotenv import load_dotenv

load_dotenv()

from occinteract.processing.parallel_processor import ParallelProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--pipeline', required=True,
                        choices=['geometry', 'turn_point', 'critical_score'])
    parser.add_argument('--data-path', required=True,
                        help='root of the collected scenes')
    parser.add_argument('--save-path', required=True,
                        help='directory for the caption output JSON')
    parser.add_argument('--save-name', default=None,
                        help='output file name (default depends on the pipeline)')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--info-file', default='scene_infos.pkl',
                        help='scene index pkl inside --data-path (geometry/turn_point)')
    parser.add_argument('--turn-point-file', default=None,
                        help='turn point pkl inside --data-path (turn_point pipeline)')
    parser.add_argument('--write-index', action='store_true',
                        help='(geometry) write the scene index pkl next to the output')
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    if args.pipeline == 'geometry':
        from data_engine.env_collector.pipeline.caption_pipeline_geometry import (
            Handler, get_process_task,
        )

        tasks, dataset, file_info = get_process_task(args.data_path, args.info_file)
        if args.write_index:
            import pickle
            with open(os.path.join(args.save_path, 'scene_infos.pkl'), 'wb') as f:
                pickle.dump(file_info, f)

        save_name = args.save_name or 'turn_point.json'
        handler = Handler(args.data_path, dataset)

    elif args.pipeline == 'turn_point':
        if not args.turn_point_file:
            parser.error('--turn-point-file is required for the turn_point pipeline')

        from data_engine.env_collector.pipeline.caption_turn_point import (
            Handler, get_process_task,
        )

        tasks, dataset = get_process_task(
            os.path.join(args.data_path, args.info_file),
            os.path.join(args.data_path, args.turn_point_file),
        )
        save_name = args.save_name or 'turn_caption.json'
        handler = Handler(args.data_path, dataset)

    else:  # critical_score
        from data_engine.env_collector.pipeline.caption_scene_with_critical_score import (
            Handler, get_process_task,
        )

        tasks, dataset = get_process_task(
            os.path.join(args.data_path, args.info_file))
        save_name = args.save_name or 'scene_with_critical_score.json'
        handler = Handler(args.data_path, dataset)

    processor = ParallelProcessor(
        text_rewriter=handler,
        texts_to_process=tasks,
        output_path=os.path.join(args.save_path, save_name),
        max_retries=3,
        result_schema={
            'input': 'task_cfg_str',
            'output': 'result'
        },
        max_workers=args.workers,
    )
    processor.run()


if __name__ == '__main__':
    main()
