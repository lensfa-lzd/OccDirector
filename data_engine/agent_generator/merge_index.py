"""Merge per-prompt-type scene_infos.pkl files into a release-root index.

The published OccInteract-Agent layout groups renders by prompt type
(<root>/<type>/prompt_N_try_M/video_i). ``curate.py`` indexes one prompt-type
directory at a time (its scan is two levels deep), so each type ends up with
its own ``scene_infos.pkl`` holding type-relative paths. The loaders expect a
single index at the release root with the type name included — this script
concatenates the per-type indexes with ``<type>/`` prefixed onto
``scene_name`` and every ``occ_in_scene_paths`` entry.

Usage (from the repo root):

    python -m data_engine.agent_generator.merge_index --root <agent_root>
    # -> <agent_root>/scene_infos.pkl
"""
import argparse
import pickle
from glob import glob
from os.path import join
from pathlib import Path


def merge_index(root: str, save_file: str) -> int:
    merged = []
    for type_index in sorted(glob(f'{root}/*/scene_infos.pkl')):
        prompt_type = Path(Path(type_index).parent).name
        with open(type_index, 'rb') as f:
            entries = pickle.load(f)
        for it in entries:
            merged.append({
                'scene_name': f'{prompt_type}/{it["scene_name"]}',
                'occ_in_scene_paths': [f'{prompt_type}/{p}' for p in it['occ_in_scene_paths']],
            })
        print(f'{prompt_type}: {len(entries)} scenes')

    out = join(root, save_file)
    with open(out, 'wb') as f:
        pickle.dump(merged, f)
    print(f'merged {len(merged)} scenes -> {out}')
    return len(merged)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge per-type scene_infos.pkl into the release-root index')
    parser.add_argument('--root', required=True, help='agent dataset root containing <type>/scene_infos.pkl')
    parser.add_argument('--save-file', default='scene_infos.pkl')
    args = parser.parse_args()
    merge_index(args.root, args.save_file)
