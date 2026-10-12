# evaluation

Three evaluation suites for OccGen. All entry scripts read their configuration
from environment variables at import time (`.env` at the repo root is loaded):
the three dataset roots `DATA_PATH_1/2/3` (static / env / agent), the VAE and
OccGen checkpoints under `ckpt/`, and the `VLM` selector described in
`model/README.md`.

## vlm_eval — VLM-as-judge on generated scenes

Renders generated occupancy sequences (BEV + front view grids) and asks a
VLM to score/caption them through a LangChain + OpenAI-compatible endpoint
(`OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`).

```bash
python -m evaluation.vlm_eval.main            # evaluate render grids
python -m evaluation.vlm_eval.eval_model_outputs   # evaluate model-generated .npy outputs
```

`eval_model_outputs.py` reads one `.npy` per sequence
(`OccCaptionOneFileDataset` layout).

## image_metrics — FID / IS / KID / PRDC

Compares rendered occupancy frames between generated and real data
(`GEN_DATA_PATH` points at the generation output tree).

```bash
GEN_DATA_PATH=<gen root> OUTPUT_FILE=<metrics.pkl> python -m evaluation.image_metrics.main
```

Torchmetrics FID/IS/KID plus `prdc` (precision/recall/density/coverage).

## fvd — Fréchet Video Distance

Occupy-sequence FVD via the shared `occinteract.tools.fvd.OccFVDHandler`
(I3D features over rendered BEV videos; the I3D weight auto-downloads on
first use — see `occinteract/README.md`).

| script | purpose |
|--------|---------|
| `fvd_dataset.py` | real-data FVD features (sampling rate 0.1 by default) |
| `fvd_model.py` | unconditional generation → FVD |
| `fvd_model_history_cond.py` | history-conditioned generation → FVD |
| `fvd_from_files.py` | FVD features from saved `.npy` outputs (`GEN_DATA_PATH` tree, one `[1, F, H, W, D]` file per sequence; `SAVE_FILE` for the result) |
| `gen_test_set.py` | generate the test set used by `vlm_eval` |

```bash
DATA_SIZE=8 python -m evaluation.fvd.fvd_dataset
```
