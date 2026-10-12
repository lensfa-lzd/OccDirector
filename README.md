# OccDirector

**OccDirector: Language-Guided Behavior and Interaction Generation in 4D Occupancy Space**
(ECCV 2026)

OccDirector generates 4D semantic occupancy driving scenes from natural
language: an `OccSeqCausalVAE` compresses occupancy sequences into a compact
latent space, and a flow-matching MMDiT conditioned on a frozen VLM text
encoder learns to generate latents — and thus scenes — from text prompts.

| Path | Contents |
|---|---|
| `occinteract/` | shared data library: dataset loaders, Static 8-bit conversion ETL, parallel VLM fan-out, rendering & FVD tools |
| `data_engine/` | dataset construction: `env_collector` (OccInteract-Env) and `agent_generator` (OccInteract-Agent, CARLA-based scene synthesis) |
| `occ_vae/` | OccSeqCausalVAE and its Lightning trainer, latent scale-factor measurement, release-weight export |
| `model/` | OccGen: flow-matching MMDiT + frozen text encoders (Qwen-VL / CLIP / T5), trainer, sampling |
| `evaluation/` | VLM-as-judge scoring, FID/IS/KID/PRDC image metrics, occupancy FVD |

## Installation

Python 3.10, CUDA 12.4:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Checkpoints live under `ckpt/` (gitignored) — see `model/README.md` for the
expected layout. The released OccGen checkpoint uses `VLM=Qwen3-VL-2B`,
`MMDIT_DEPTH=14`, `MMDIT_PATCH_SIZE=2`, `USE_ST_ATT=1`.

## Dataset

OccInteract consists of three subsets (Static / Env / Agent) sharing one
loader surface. The full construction pipeline — raw UniOcc conversion,
CARLA-based collection, caption generation — is documented in
[DATASETS.md](DATASETS.md). Dataset download links will be announced with
the paper.

## Usage

All entry points are configured through environment variables read at import
time; a `.env` file at the repo root is loaded automatically.

**Train the VAE** (all variables in `occ_vae/README.md`):

```bash
EXP_NAME=vae python -m occ_vae.train
```

**Train OccGen** (all variables in `model/README.md`):

```bash
VLM=Qwen3-VL-2B USE_ST_ATT=1 MMDIT_DEPTH=14 MMDIT_PATCH_SIZE=2 \
EXP_NAME=occgen python -m model.train
```

**Sample from a trained model** (run with `VLM=Qwen3-VL-2B` in the
environment so the matching text encoder is built):

```python
import torch
from model.occ_gen import OccGen

pipe = OccGen(lr=1e-3, max_frames=32, training_stage='stage1').cuda()
state = torch.load('ckpt/occ_gen/model-ema.ckpt', map_location='cpu')['state_dict']
pipe.mmdit.load_state_dict(
    {k[6:]: v for k, v in state.items() if k.startswith('mmdit.')}, strict=True)
pipe = pipe.eval()

occ = pipe.predict_step(16, ['...your prompt...'])  # [1, 16, 224, 224, 16] int64
```

**Evaluate** — VLM-as-judge, FID/IS/KID/PRDC and occupancy FVD; see
`evaluation/README.md`.

## License & acknowledgments

Code is released under the Apache License 2.0 (see [LICENSE](LICENSE)).
Several components derive from upstream projects with their own licenses —
see [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).

## Citation

If you use OccDirector in your research, please cite the paper. (BibTeX
will be added upon publication.)
