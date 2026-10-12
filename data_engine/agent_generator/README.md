# OccInteract-Agent Generator

LLM-planned, safety-critical scene factory that produces the **OccInteract-Agent**
subset: text prompts are turned into structured CARLA scene plans (environment,
agents, behaviors), rendered on CARLA 0.9.15 servers, and converted to UniOcc-format
semantic occupancy (200×200×16, 2 Hz) plus front/BEV videos.

## Pipeline overview

```
prompt bank / synthesize_prompts.py   (prompt source: curated bank or randomized synthesis)
        │
        ▼
plan_scenes.py text_to_scene          (LLM 3-stage: analysis -> road retrieval -> planning)
        │  agent_output.json + prompt.txt per scene
        ▼
render.py (single host)  or  render_batch.py (multi-host, CARLA pool from hosts.json)
        │  <scene>/video_i/: occupancy/*.npy + front.mp4 + bev.mp4
        ▼
curate.py --save-path ...             (scene_infos.pkl index for the OccInteract loaders)
check_completion.py                   (copy finished videos out of a render tree)
misc/evaluate.py                      (agent/road diversity statistics)
```

## Setup

1. **CARLA 0.9.15 servers** and a **host pool** — same convention as the env
   collector: one instance per GPU, ports `base_port + 100 * worker_index`.
   Copy `data_engine/hosts.example.json` to `data_engine/agent_generator/hosts.json`
   (gitignored) or pass `--hosts` explicitly.
2. **Road-graph cache** (one-time, needs a single live CARLA server) — build it
   before the first render, otherwise the first render pays the full 7-map
   intersection scan inline:

   ```bash
   python -m data_engine.agent_generator.build_graph_cache --host 127.0.0.1 --port 2000
   # → data_engine/agent_generator/graph_cache/graph.pkl  (cold server: several minutes)
   ```

3. **LLM endpoint** — a `.env` at the repo root with an OpenAI-compatible API:
   `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` (a reasoning model that
   emits `</think>` blocks is supported; output is repaired with `json_repair`).
4. **Maps** — `maps/town1..7.xodr` are bundled for the road-retrieval stage.
   They are exports from CARLA 0.9.15; regenerate them from your own server with
   `scene_utils/get_map_xodr.py` if you prefer.

## Usage

All commands run from the repo root.

**1. Plan scenes** (from the bundled prompt bank, 5 planning tries per prompt):

```bash
python -m data_engine.agent_generator.plan_scenes --workers 4
# or randomized synthesis (environment × event pools):
python -m data_engine.agent_generator.synthesize_prompts --count 100 --save-path output/ttsg_auto
```

**2. Render** planned scenes (needs CARLA servers + `hosts.json`):

```bash
# single host:
python -m data_engine.agent_generator.render \
    --source-path output/ttsg_auto/auto --save-path output/rendered --max-save-time 10

# multi-host (resume from a partially rendered tree):
python -m data_engine.agent_generator.render_batch \
    --task-root output/rendered --hosts data_engine/agent_generator/hosts.json
```

**3. Curate** the finished videos into the training index:

```bash
python -m data_engine.agent_generator.curate --save-path output/rendered
# → output/rendered/scene_infos.pkl  ([{scene_name, occ_in_scene_paths}])
# list missing videos instead:
python -m data_engine.agent_generator.curate --list-tasks --task-root output/rendered
```

On the **published** dataset layout (`<prompt_type>/prompt_N_try_M/video_i/`),
run curate once per prompt-type directory, then merge the per-type indexes into
the single release-root index the loaders expect:

```bash
python -m data_engine.agent_generator.merge_index --root <agent_root>
```

Scene captions with criticality scores for this subset are produced by the
env collector's captioning pipeline (`data_engine/env_collector/run_caption.py
--pipeline critical_score`), sharing the same `scene_infos.pkl` format.

## Notes

- With `return_ego=True` (the default in `plan_scenes`), the LLM planning output
  must include the ego entry itself (`is_ego: true`); the validators skip ego
  agents, so custom planners should follow the bundled prompt templates.
- Cold CARLA servers need generous RPC timeouts: `rpc_timeout` defaults to 120 s
  (`scene_generation`/`CarlaClient`) and 300 s (`build_graph_cache`).

## Output format

Same conventions as OccInteract-Env: `uint8 [200, 200, 16]` occupancy with UniOcc
labels 0–10, plus `front.mp4`/`bev.mp4`, `agent_output.json` (LLM plan),
`road_info.json` (spawn road). `ENVIRONMENT.md` documents the agent JSON schema
and the weather/road-type vocabularies.

## Provenance & licensing

- This module is **derived from [basiclab/TTSG](https://github.com/basiclab/TTSG)**
  (Text-to-Traffic-Scene Generation, arXiv 2409.09575) and modified for
  OccInteract-Agent. **The TTSG upstream repository does not provide a license**;
  the derivation is stated here and in the header of every derived file for
  attribution. If you are a TTSG author with licensing concerns, please open an
  issue against this repository.
- `agents/` is vendored from the [CARLA PythonAPI](https://github.com/carla-simulator/carla/tree/dev/PythonAPI/carla/agents)
  (CARLA is distributed under the MIT License).
- The occupancy label scheme follows [UniOcc](https://github.com/tasl-lab/UniOcc)
  (see `occinteract/tools/carla_sensing.py`).
- `maps/*.xodr` are OpenDRIVE exports of CARLA Town01–07, bundled for
  reproducibility; they can be regenerated with `scene_utils/get_map_xodr.py`.
