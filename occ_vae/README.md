# occ_vae — OccSeqCausalVAE

Causal sequence VAE over 4D occupancy, derived from
[DOME](https://github.com/gusongen/DOME) and modified for occupancy sequences
(2D per-frame encoder → causal 3D decoder over the `D·expansion`-channel
embedding, UniOcc class logits output).

## Layout

- `vae.py` / `module.py` — the model (`OccSeqCausalVAE`, `get_vae`)
- `train.py` — training entry point (Lightning)
- `trainer/` — shared Lightning pieces: data module, `TrainerSeq` (IoU/mIoU,
  frame-split training), EMA / stats / reconstruction-visualizer callbacks
- `loss/` — recon / KL / Lovasz losses and IoU metrics
- `measure_scale_factor.py` — post-training latent statistics
- `export_weights.py` — release-checkpoint export

## Train

Configured via environment variables (`.env` at the repo root is loaded):

| var | meaning | default |
|-----|---------|---------|
| `EXP_NAME` | experiment name (required) | — |
| `DATA_PATH_1/2/3` | static / env / agent dataset roots | — |
| `BATCH_SIZE` | batch size | 4 |
| `TOTAL_STEPS` | training steps | 1e7 |
| `LR` | learning rate | 1e-4 |
| `LOG_DIR` | Lightning log dir | `../pl_logs` |
| `DATA_WORKER_NUM` | dataloader workers | 4 |
| `CHECK_EVERY_STEPS` | checkpoint/visualization cadence | 5000 |
| `LOSS_SETTING` | loss weights preset | — |
| `NOT_SPLIT_FRAMES` | disable random frame cropping | — |
| `LOAD_FROM_CKPT` | resume from a specific ckpt | auto (latest) |
| `TEST_ONLY` | skip fit, run test only | — |

```bash
python -m occ_vae.train   # with the env vars above
```

Datasets come from `occinteract.loaders.unsupervised` (see
`occinteract/README.md` for the expected roots). Checkpoints land under
`$LOG_DIR/$EXP_NAME/version_N/checkpoints/`; resume picks the latest
`last.ckpt` automatically.

## Measure scale factors & export release weights

The VAE latents are normalized inside OccGen; the normalization constants are
measured over the training set once training is done:

```bash
python -m occ_vae.measure_scale_factor \
    --data-path <static_root> --ckpt-path <pl ckpt> [--seq-len 8]

python -m occ_vae.export_weights \
    --ckpt <pl ckpt> \
    --mean <m> --std <s> --ema-mean <m> --ema-std <s> \
    --output dome_vae.ckpt
```

`measure_scale_factor` prints mean/std for both the raw and the EMA weights;
feed them to `export_weights`, which strips the `model.` prefix and attaches
the scale factors. The exported file is the only checkpoint format
`get_vae(use_scale=True, ckpt_path=...)` accepts (it loads the EMA weights and
applies the stored scale).

`measure_scale_factor` reads the static root through
`occinteract.loaders.uniocc_seq`, which expects a **list-format**
`scene_infos.pkl` inside each `{dataset}-{split}` directory (the frame-level
index; see `occinteract/README.md`).

## Provenance & licensing

- Model architecture derived from
  [gusongen/DOME](https://github.com/gusongen/DOME) (see the header of
  `vae.py`); DOME carries its own license — stated here and in the file header
  for attribution.
