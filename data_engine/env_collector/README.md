# OccInteract-Env Collector

Batch data engine that produces the **OccInteract-Env** subset: ego-centric driving
scenes collected from remote CARLA 0.9.15 servers, converted to UniOcc-format
semantic occupancy (200×200×16, 0.4 m/voxel, ±40 m XY, 2 Hz) plus front/BEV videos,
then captioned by a VLM with intersection turn-point analysis and a 0–10 criticality score.

## Pipeline overview

```
CARLA servers ──main.py──► scenes: occupancy/*.npy + front.mp4 + bev.mp4
                                 │
                 run_caption.py --pipeline geometry        (no LLM: road-skeleton turn points)
                                 │ merge_captions.py --mode turn_point_index
                                 ▼
                 run_caption.py --pipeline turn_point      (VLM: LEFT/RIGHT/STRAIGHT + description)
                                 │ merge_captions.py --mode turn_point
                                 ▼
                 turn_caption.pkl  ──► OccInteract loaders
                 scene_with_critical_score.pkl (from --pipeline critical_score, for the Agent subset)
```

## Setup

1. **CARLA 0.9.15 servers** — start one instance per GPU (each collector worker talks
   to its own RPC port; ports are derived as `base_port + 100 * worker_index`):

   ```bash
   cd /path/to/CARLA_0.9.15
   ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -graphicsadapter=0
   ```

2. **Host pool** — copy `hosts.example.json` to `hosts.json` (gitignored) and list your
   servers: `[{"ip": "...", "workers": N, "base_port": [carla_start, tm_start]}]`.
   The Traffic Manager port for worker *i* is `tm_start + 10 * i`.

3. **VLM endpoint** (captioning only) — a `.env` file at the repo root providing an
   OpenAI-compatible API: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`.

## Usage

All commands run from the repo root.

**1. Collect scenes** (one 60 s scene per task; ~600 scenes per map by default):

```bash
python -m data_engine.env_collector.main \
    --save-dir output/batch_output \
    --hosts data_engine/env_collector/hosts.json
```

Useful flags: `--scenes-per-map`, `--save-time`, `--occ-save-fps`, `--rpc-timeout`
(increase on cold servers — map loading can exceed 10 s), `--startup-interval`.
The batch runner is resumable: finished scenes (occ frame count reached) are skipped
on re-run, and failed tasks are retried up to 3 times per worker.

**2. Find turn points** (geometry only, no LLM; also builds the scene index):

```bash
python -m data_engine.env_collector.run_caption \
    --pipeline geometry \
    --data-path output/batch_output \
    --save-path output/captions \
    --write-index --workers 8

python -m data_engine.env_collector.merge_captions \
    --mode turn_point_index \
    --input output/captions/turn_point.json \
    --output output/batch_output/turn_point.pkl
```

**3. Caption turn points** (VLM classifies the ego action at each intersection):

```bash
python -m data_engine.env_collector.run_caption \
    --pipeline turn_point \
    --data-path output/batch_output \
    --info-file scene_infos.pkl \
    --turn-point-file turn_point.pkl \
    --save-path output/captions --workers 16

python -m data_engine.env_collector.merge_captions \
    --mode turn_point \
    --input output/captions/turn_caption.json \
    --output output/captions/turn_caption.pkl
```

**4. Scene captions with criticality score** (used for the OccInteract-Agent subset):

```bash
python -m data_engine.env_collector.run_caption \
    --pipeline critical_score \
    --data-path output/batch_output \
    --info-file scene_infos.pkl \
    --save-path output/captions --workers 16

python -m data_engine.env_collector.merge_captions \
    --mode critical_score \
    --input output/captions/scene_with_critical_score.json \
    --output output/captions/scene_with_critical_score.pkl
```

## Output format

- `<data-path>/<Town>/<scene_id>/occupancy/<frame>.npy` — `uint8 [200, 200, 16]`,
  UniOcc labels 0–10 (see `occinteract/tools/carla_sensing.py: CARLA_TO_UNIOCC_MAP` for the CARLA mapping)
- `<Town>/<scene_id>/front.mp4`, `bev.mp4` — 10 Hz ego/front and top-down videos
- `scene_infos.pkl` — `[{scene_name, occ_in_scene_paths}]` scene index
- `turn_caption.pkl` — `[{scene_name, key_frame_id, caption}]`
- `scene_with_critical_score.pkl` — `[{scene_name, caption}]` with a 0–10 `score`

## Notes

- The voxelizer runs on the collector's GPU (`cuda:<i>` assigned round-robin);
  one worker per CARLA instance, sensors are 25 base semantic lidars (5×5 grid)
  with 4 jittered companions each.
- Requires the `carla==0.9.15` client wheel (matching the server).
