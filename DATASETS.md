# OccInteract Data Pipeline

This document describes how the three OccInteract subsets are produced and how
to assemble them into the directory layout the training/evaluation loaders
consume. All commands were verified end-to-end on the released data (see
*Verification* at the end).

Common conventions across all subsets:

- occupancy grids: `200×200×16`, 0.4 m voxels, ±40 m XY, 2 Hz
- labels: UniOcc semantic IDs `0–10`
- every index pickle stores paths **relative** to its dataset root; loaders
  join them as `f'{dataset_path}/{file}'`

```
                 OccInteract-Static          OccInteract-Env         OccInteract-Agent
                 ─────────────────          ───────────────         ──────────────────
raw source       UniOcc release (.npz)      CARLA scenes            LLM-planned CARLA
                                            (env_collector)         scenes (agent_generator)
                     │                          │                        │
convert/index     convert_uniocc_8bit.py     run_caption.py          curate.py
                  (int8 npy + frame          --pipeline geometry     (+ merge prompt-type
                   inventory)                --write-index            indexes)
                     │                          │                        │
captions          bev/extra/location_        run_caption.py          run_caption.py
                  caption.pkl (released)     --pipeline turn_point   --pipeline critical_score
                                             (turn points from       (VLM criticality score)
                                              geometry, captions      │
                                              from VLM)              merge_captions.py
                     │                          │                     --mode critical_score
                     │                          │                        │
train index       scene_infos.pkl            scene_infos_DATE.pkl    scene_infos.pkl
                  + caption pkls             + turn_caption_DATE.pkl + scene_with_critical_
                                                                       score.pkl
                     └──────────────┬───────────┴────────────────────────┘
                                    │
                     occinteract.loaders.supervised / .unsupervised
```

## 0. VLM endpoint (caption stages)

The caption stages talk to any OpenAI-compatible VLM endpoint configured via
`.env` at the repo root:

```
OPENAI_API_KEY=...
OPENAI_BASE_URL=...        # e.g. https://api.xxx/v1
OPENAI_MODEL=...           # a vision-language model
```

Reasoning-style outputs (`<think>...</think>`) are tolerated; responses are
repaired with `json_repair`.

## 1. OccInteract-Static (UniOcc-derived)

**Assemble from the release.** The Static root is the 8-bit frame tree plus
the released caption pickles:

```
<static_root>/
├── NuScenes-via-Occ3D-2Hz-train/scene-XXXX/N.npy      # int8 [200,200,16]
├── ... (other {dataset}-{split} dirs)
├── scene_infos.pkl        # dict: {'{dataset}-{split}': [{scene_name, occ_in_scene_paths}]}
├── bev_caption.pkl        # VLM captions keyed {dataset: {section: {field: {scene/F.npz: str}}}}
├── extra_caption.pkl      # {dataset: {scene/F.npz: [str, ...]}}
└── location_caption.pkl   # {dataset: {scene/F.npz: str}}
```

A sequence is captioned by its **first frame**: the caption key is the first
occ path with the split prefix stripped and `.npy` → `.npz`. `full_scene_infos.pkl`
(a flat frame inventory) ships with the release but is not read by any loader.

**Regenerate the 8-bit tree from raw UniOcc** (resumable; per-frame int8 npy):

```bash
USE_8_BIT_DATA=0 python -m occinteract.processing.convert_uniocc_8bit \
    --source /path/uniocc-full/v1-0 --save-path /path/static_root \
    --batch-size 32 --workers 16
```

`USE_8_BIT_DATA=0` is required: the raw reader prefers existing 8-bit files,
which do not exist yet when converting. Raw trees use 10 Hz variants for
Waymo/Carla; the converter and its OpV2V path quirks are preserved from the
original pipeline.

The converter also writes a **list-format** `scene_infos.pkl` (frame-level
index) into each `{dataset}-{split}` directory — the UniOcc-release
convention, consumed by `occinteract.loaders.uniocc_seq` (VAE scale-factor
measurement).

The caption pickles are part of the dataset release; regenerating them means
re-running VLM captioning over the whole corpus and is not required to train.

## 2. OccInteract-Env (collected scenes)

Collect scenes first (see `data_engine/env_collector/README.md`), then build
the training index over the collection tree:

```bash
# 2.1 geometry pass: renders BEV road maps, detects turn points (no VLM),
#     and writes the scene index
python -m data_engine.env_collector.run_caption \
    --pipeline geometry --data-path <env_tree> \
    --save-path <out> --write-index

# 2.2 convert turn points into the per-scene index the VLM stage consumes
python -m data_engine.env_collector.merge_captions \
    --mode turn_point_index \
    --input <out>/turn_point.json --output <out>/turn_point_index.pkl

# 2.3 caption each turn point (VLM; 9-frame front window + BEV road map)
python -m data_engine.env_collector.run_caption \
    --pipeline turn_point --data-path <env_tree> --save-path <out2> \
    --info-file <out>/scene_infos.pkl \
    --turn-point-file <out>/turn_point_index.pkl

# 2.4 merge into the loader pkl
python -m data_engine.env_collector.merge_captions \
    --mode turn_point \
    --input <out2>/turn_caption.json --output <out2>/turn_caption.pkl
```

The loaders read these under **dated snapshot names** — `scene_infos_DATE.pkl`
and `turn_caption_DATE.pkl` (e.g. the released `*_20260120.pkl`); copy/rename
the outputs accordingly. Scenes whose geometry finds no turn points
legitimately contribute no captions.

## 3. OccInteract-Agent (LLM-planned scenes)

Plan and render scenes first (see `data_engine/agent_generator/README.md`),
then index and caption:

```bash
# 3.1 index finished videos. The published layout groups renders by prompt
#     type (<type>/prompt_N_try_M/video_i), so run curate per prompt-type dir:
python -m data_engine.agent_generator.curate --save-path <agent_root>/<type>

# 3.2 merge the per-type indexes into the release-root index
python -m data_engine.agent_generator.merge_index \
    --root <agent_root>            # -> <agent_root>/scene_infos.pkl
```

`merge_index` concatenates each `<type>/scene_infos.pkl` with the type name
prefixed onto `scene_name` and `occ_in_scene_paths` (a `heavy_traffic/...`
prefix), producing the single root index the loaders expect.

```bash
# 3.3 criticality captions (VLM; one pass per scene over the front video)
python -m data_engine.env_collector.run_caption \
    --pipeline critical_score --data-path <agent_root> \
    --save-path <out> --info-file scene_infos.pkl

# 3.4 merge into the loader pkl
python -m data_engine.env_collector.merge_captions \
    --mode critical_score \
    --input <out>/scene_with_critical_score.json \
    --output <agent_root>/scene_with_critical_score.pkl
```

## 4. Feeding the loaders

```python
from occinteract.loaders.supervised.get_dataset import get_full_dataset, get_test_dataset

train = get_full_dataset(static_root, env_root, agent_root,
                         resize_cfg={'origin': 200, 'after': 224}, load_occ=True)
uniocc_ds, env_ds, agent_ds = get_test_dataset(samples, static_root, env_root,
                                               agent_root, resize_cfg, load_occ)
```

Subset toggles (env vars): `DATA_NO_STATIC`, `DATA_NO_INTERACT_ENV`,
`DATA_NO_INTERACT_AGENT`, `DATA_NO_INTERACTION`, `UNIOCC_MINI_GAP` (default 16),
`DATASET_DEBUG`. The unsupervised (VAE) loader shares the same roots; see
`occinteract/README.md` for its extra `scene_infos.pkl` key requirements and
1 % env/agent subsampling.
