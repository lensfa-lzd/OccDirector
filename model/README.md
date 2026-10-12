# model — OccGen

Language-conditioned occupancy generation: a frozen `OccSeqCausalVAE`
(`occ_vae/`) provides the latent space, a frozen VLM/CLIP/T5 encoder provides
text conditioning, and a QwenImage-derived MMDiT (flow matching) is trained on
latents and samples new scenes.

## Layout

- `occ_gen.py` — `OccGen` (Lightning): model assembly, CFM training step,
  CFG, history-frame conditioning, `predict_step` sampling
- `occ_gen_test_fvd.py` — generation wrappers used by the FVD evaluation
- `mmdit/` — MMDiT: `transformer_qwenimage.py` (derived from Qwen-Image,
  Apache-2.0), `att.py` (diffusers-derived attention processor), `mmdit.py`
  (token packing, sampling, scheduler)
- `vlm_encoder/` — Qwen2.5/3-VL text encoder (4-bit, frozen)
- `bag_of_words_encoder/` — CLIP / T5 text encoders (frozen)
- `dynamic_city_dit/` — experimental DiT variant, enabled via
  `USE_DYNAMIC_CITY_DIT` (dispatch branch of `occ_gen`)
- `trainer/` — `RectifiedFlowTrainer`, data modules, EMA/visualizer/weight
  averaging callbacks
- `loss/iou.py` — logits→label conversion and IoU metrics

## Checkpoints expected under `ckpt/`

```
ckpt/vae/<release vae>.ckpt                 # occ_vae export format (see occ_vae/README.md)
ckpt/Qwen3-VL-2B-Instruct/ | ckpt/Qwen2.5-VL-3B-Instruct/   # VLM env option
ckpt/clip-vit-large-patch14/  | ckpt/flan-t5-xl/            # TEXTEncoder env options
```

## Text encoder selection (`VLM` env)

`Qwen2.5-VL-3B` / `Qwen3-VL-2B` (VLMEncoder, text dim 2048) or
`CLIP-VIT-LARGE` / `T5-XL` (TEXTEncoder, 768 / 2048). The choice must match
the checkpoint being loaded — the released OccGen checkpoint uses
`VLM=Qwen3-VL-2B`.

## Train

Configured via environment variables (same set as `occ_vae/train.py`:
`EXP_NAME`, `DATA_PATH_1/2/3`, `BATCH_SIZE`, `TOTAL_STEPS`, `LR`, `LOG_DIR`,
`DATA_WORKER_NUM`, `CHECK_EVERY_STEPS`, `LOAD_FROM_CKPT`, `TEST_ONLY`), plus:

| var | meaning | default |
|-----|---------|---------|
| `VLM` | text encoder selection | required |
| `MMDIT_DEPTH` | transformer depth (hidden = 64·depth) | 18 |
| `MMDIT_PATCH_SIZE` | latent patch folding | 1 |
| `USE_ST_ATT` | spatio-temporal attention blocks | off |
| `USE_DYNAMIC_CITY_DIT` | experimental DiT variant | off |
| `CFG_SCALE` / `SEED` | sampling options | — |

The released model was trained with depth 14, patch size 2 and
`USE_ST_ATT=1` (`VLM=Qwen3-VL-2B`).

```bash
VLM=Qwen3-VL-2B USE_ST_ATT=1 MMDIT_DEPTH=14 MMDIT_PATCH_SIZE=2 \
EXP_NAME=occgen python -m model.train
```

## Sampling

`OccGen.predict_step(frames, prompts, history_occ_list=None)` returns
`[B, F, 224, 224, 16]` occupancy labels (UniOcc ids). The FVD evaluation
scripts in `evaluation/fvd/` wrap this for batch generation.

## Provenance & licensing

- `mmdit/transformer_qwenimage.py`: derived from
  [Qwen-Image](https://github.com/QwenLM/Qwen-Image) (Apache-2.0, header kept).
- `mmdit/att.py`: derived from the
  [diffusers](https://github.com/huggingface/diffusers) AttentionProcessor
  design (header added).
- VAE: `occ_vae/` (DOME lineage, see its README).
