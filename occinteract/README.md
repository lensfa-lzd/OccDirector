# occinteract — shared data library

Loaders and tools shared by every OccDirector stage. All occupancy data uses the
UniOcc label scheme (classes 0–10), `uint8 [H, W, D]` grids (typically
`200×200×16`, 0.4 m voxels, 2 Hz).

## Layout

- `core/` — `BaseHandler` / `BaseVisualizationHandler`, script helpers
- `loaders/` — training/eval dataset builders (see below); `uniocc_raw/` reads
  the original UniOcc `.npz` release (list-format `scene_infos.pkl`)
- `processing/` — `ParallelProcessor` for VLM fan-out, Static-subset 8-bit
  conversion (`convert_uniocc_8bit.py`)
- `tools/` — `occ_resizer`, `carla_sensing`, `pyvista_vis`, `shape_container`,
  `uniocc_dataset` (upstream UniOcc loader, MIT), `fvd/` (I3D-based FVD)
- `type_models/` — `TypedDict` shapes shared between loaders and trainers

## Loaders

### `loaders.supervised` — caption-conditioned training (OccGen)

```python
from occinteract.loaders.supervised.get_dataset import get_full_dataset, get_test_dataset

train = get_full_dataset(uniocc_path, occ_collect_path, ttsg_path,
                         resize_cfg={'origin': 200, 'after': 224}, load_occ=True)
uniocc_ds, env_ds, agent_ds = get_test_dataset(samples, uniocc_path, occ_collect_path,
                                               ttsg_path, resize_cfg, load_occ)
```

Both index-building datasets and returned items are documented in
`type_models/dataset.py`. Env toggles: `DATA_NO_STATIC`,
`DATA_NO_INTERACT_ENV`, `DATA_NO_INTERACT_AGENT`, `DATA_NO_INTERACTION`,
`UNIOCC_MINI_GAP` (default 16), `DATASET_DEBUG` (log skipped scenes).

### `loaders.unsupervised` — VAE reconstruction training

```python
from occinteract.loaders.unsupervised.get_dataset import get_full_dataset

train, val, test = get_full_dataset(uniocc_path, occ_collect_path, ttsg_path,
                                    resize_cfg={'origin': 200, 'after': 224}, load_occ=True)
```

`load_occ=False` runs the same index building and returns dummy arrays
(index-only passes). `DATASET_SETTING=1` drops the OpenOccupancy/SurroundOcc
splits. Env/agent sources are subsampled to 1 % here (hard-coded factors).

### `loaders.uniocc_seq` — fixed-length sequence reader (VAE scale measurement)

Reads the same 8-bit tree in fixed-length sliding windows
(`get_full_uniocc_dataset(seq_len, path)`; `DATASET_SETTING=1/2` restrict to
Carla/NuScenes). Unlike the loaders above it expects a **list-format**
`scene_infos.pkl` (`[{scene_name, occ_in_scene_paths}]`) inside each
`{dataset}-{split}` directory — the frame-level index, not the dict-format
root index. `convert_uniocc_8bit.py` emits one per split alongside the frames.
Consumed by `occ_vae/measure_scale_factor.py`.

Also in `loaders.supervised`: `core.py` holds the dataset classes
(`OccCaptionDataset`, and `OccCaptionOneFileDataset` — one `.npy` per sequence,
used by the VLM evaluation); `caption_sources.py` reads the Static-subset
caption pickles (`bev/extra/location_caption.pkl`) and builds sliding-window
instances; the `get_per_*` modules build the per-source training indexes.

## Expected data layout (per source root)

End-to-end pipeline commands (raw sources → these indexes) are documented in
[DATASETS.md](../DATASETS.md).

Paths inside the index pickles are **relative** to each root and joined as
`f'{dataset_path}/{file}'`.

**Static (UniOcc-format)** — `scene_infos.pkl` keyed
`{dataset}-{split}` (`NuScenes-via-Occ3D-2Hz`, `NuScenes-via-OpenOccupancy-2Hz`,
`NuScenes-via-SurroundOcc-2Hz`, `Waymo-via-Occ3D-2Hz`, `Carla-2Hz`,
`OpV2V-10Hz`; the unsupervised loader expects all keys, supervised skips test
splits without one). Captions come from `bev_caption.pkl` +
`extra_caption.pkl` + `location_caption.pkl`; a sequence is captioned by its
**first frame**, looked up as the first occ path with the split prefix
stripped and `.npy` → `.npz` (e.g. `NuScenes-via-Occ3D-2Hz-train/s/5.npy` →
`s/5.npz`). Sequences whose first frame has no caption are skipped.
These pickles ship with the released OccInteract-Static index.

The 8-bit frame tree itself is produced from the raw UniOcc `.npz` release by
`occinteract/processing/convert_uniocc_8bit.py` (run with `USE_8_BIT_DATA=0`;
writes per-frame int8 `.npy` + `full_scene_infos.pkl`, resumable). The published
`full_scene_infos.pkl` frame inventory is not read by any loader — the loaders
consume the dict-format `scene_infos.pkl`.

**Env (OccInteract-Env)** — `scene_infos_20260120.pkl` +
`turn_caption_20260120.pkl` (dated snapshot names are part of the released
index; regenerate equivalents with
`data_engine/env_collector/merge_captions.py --mode turn_point_index`).
Each turn-caption entry is `{scene_name, key_frame_id,
caption: {description_short, description}}`; training windows slide around
`key_frame_id`.

**Agent (OccInteract-Agent)** — `scene_infos.pkl` (from
`data_engine/agent_generator/curate.py`) + `scene_with_critical_score.pkl`
(from `merge_captions.py --mode critical_score`). The published zip ships only
the render trees (`<prompt_type>/prompt_N_try_M/video_i/occupancy/*.npy`) —
run `curate.py` once per prompt-type directory and merge the pickles with the
prompt-type prefix (`scene_name`/paths prefixed `heavy_traffic/` etc.) to get
the release-root index; captions are then added by the critical-score
pipeline.

## FVD (`tools/fvd`)

`OccFVDHandler` renders occupancy sequences to BEV videos (via
`BaseVisualizationHandler.render_bev_image`) and extracts I3D features;
`frechet_distance` computes FVD from two feature sets. The I3D TorchScript
weight (`i3d_torchscript.pt`, ~51 MB) is **not** committed — it auto-downloads
from the [fvd-comparison](https://github.com/universome/fvd-comparison) Dropbox
link on first use, or drop it next to `fvd.py` manually.

Derived from `universome/fvd-comparison` and StyleGAN-V's
`frechet_video_distance.py` (see the header of `fvd.py`).
