# Acknowledgments

OccDirector builds on the following open-source projects. Every derived file
keeps its upstream provenance header; per-module notes live in the module
READMEs (`occ_vae/`, `occinteract/`, `model/`, `data_engine/agent_generator/`).

| Component | Derived from | Upstream license |
|---|---|---|
| `occ_vae/` (OccSeqCausalVAE architecture) | [DOME](https://github.com/gusongen/DOME) | see DOME repository |
| `model/mmdit/transformer_qwenimage.py` | [Qwen-Image](https://github.com/QwenLM/Qwen-Image) | Apache-2.0 |
| `model/mmdit/att.py` (attention processor design) | [diffusers](https://github.com/huggingface/diffusers) | Apache-2.0 |
| `occinteract/tools/fvd/` (I3D-based FVD) | [fvd-comparison](https://github.com/universome/fvd-comparison) / StyleGAN-V | MIT |
| `occinteract/tools/uniocc_dataset.py` | [UniOcc](https://github.com/tasl-lab/UniOcc) | MIT |
| `occinteract/tools/carla_sensing.py` (occupancy label scheme) | [UniOcc](https://github.com/tasl-lab/UniOcc) | MIT |
| `data_engine/agent_generator/agents/` (navigation agents) | [CARLA PythonAPI](https://github.com/carla-simulator/carla) | MIT |
| `data_engine/agent_generator/` (scene planning & rendering) | [TTSG](https://github.com/basiclab/TTSG) | **no license provided upstream** — see note below |

**Note on TTSG**: `data_engine/agent_generator` is derived from
[TTSG](https://github.com/basiclab/TTSG) (Text-to-Traffic-Scene Generation,
arXiv 2409.09575) and substantially modified for OccInteract-Agent. The TTSG
upstream repository does not provide a license file. The derivation is stated
here and in the header of the derived files for attribution; if you are a
TTSG author with licensing concerns, please open an issue.

We thank all these authors for releasing their work.
