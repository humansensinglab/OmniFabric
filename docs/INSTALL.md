# Installation details

## Environments

| Environment | Used for | Python / PyTorch |
|---|---|---|
| `omnifabric` | everything: ChatGarment, GarmentCode + Warp simulation, FLUX.2 texture transfer, Hunyuan3D-2 projection, OmniFabric normalization | Python 3.10, PyTorch 2.7.1 + CUDA 11.8 |
| `comfyui` | the ComfyUI server running MiniMax-H3 + the 360° orbit LoRA (multi-view generation) | Python 3.12, a recent PyTorch (MiniMax-H3 support in ComfyUI needs it) |

ChatGarment/GarmentCode and the OmniFabric normalization model were originally developed in two separate environments.
They are merged into one here:

- Both run on the same PyTorch 2.7.1 + cu118, `transformers` 4.57.3 and `peft` 0.17.0 builds.
- GarmentCode requires `numpy<2`, so the merged environment uses `numpy` 1.26 and Python 3.10. OmniFabric inference gives
  bit-identical outputs in both setups.
- ChatGarment's `pyproject.toml` pins old versions (PyTorch 2.1, `transformers` 4.37), but FLUX.2 needs a recent
  `transformers`. ChatGarment is installed with `--no-deps` and patched for `transformers` 4.57:
  - its LLaVA `forward()` accepts the `cache_position` argument;
  - `generate()` output is BOS-prefixed again when generating from `inputs_embeds`, as older versions did, so token and
    hidden-state indices line up.

  ChatGarment decodes greedily in bf16, so different kernels occasionally flip near-tie tokens. We compared six images
  against reference outputs from the original setup. The merged environment differed from the references about as
  much as an environment with ChatGarment's original pins (PyTorch 2.1.2, `transformers` 4.37.2, flash-attn 2.5.8) did
  on the same machine: identical predictions for some images, a few differing categorical fields for others.
- DeepSpeed is only imported by ChatGarment's code. The newer 0.16.9, which works with PyTorch 2.7, replaces 0.12.6, and
  the inference script runs without the DeepSpeed launcher.

The ComfyUI server cannot share this environment: MiniMax-H3 support in ComfyUI needs a much newer PyTorch,
CUDA and `transformers` 5.x. It only communicates with the pipeline over HTTP (`omnifabric/coarse/orbit_views.py`),
so it can run in its own environment, or even on another machine (pass `COMFY_SERVER`; then also make its
`input/`/`output/` folders reachable through `COMFYUI_DIR`).

## Building

`install.sh` compiles CUDA code (NVIDIA Warp for GarmentCode, Hunyuan3D-2's `custom_rasterizer`, `flash-attn`), which
needs a full [CUDA 11.8 toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive) matching the cu118 PyTorch
build. Point `CUDA_HOME` at it (default `/usr/local/cuda-11.8`). Conda's `cuda-toolkit` package does not work: it lacks
the static NVRTC libraries Warp links against. `environment.yml` provides `gcc` 11, since CUDA 11.8 does not support newer
compilers. Several CUDA toolkits can be installed side by side; only `CUDA_HOME` matters here.

- `flash-attn` is compiled from source (about 2 hours with the default `MAX_JOBS=8`; more jobs build faster but need
  more RAM). `bash install.sh --no-flash-attn` skips it, and ChatGarment then uses PyTorch SDPA attention.
- Warp downloads its LLVM toolchain while building (`git-lfs` is needed).

Third-party code is cloned into `third_party/` at fixed commits and patched:

| Repository | Commit | Patch |
|---|---|---|
| [ChatGarment](https://github.com/biansy000/ChatGarment) | `4403c36` | GarmentCodeRC location from `$GARMENTCODE_ROOT`; import fix; `transformers` 4.57 compatibility (see above) |
| [GarmentCodeRC](https://github.com/biansy000/GarmentCodeRC) | `e3d4476` | exact frontal render camera; `libigl` 2.6 API fix |
| [NvidiaWarp-GarmentCode](https://github.com/maria-korosteleva/NvidiaWarp-GarmentCode) | `63baf68` | none |
| [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) | `9bcf302` | optional projection edge erosion / boundary masking, texture sampling in `custom_rasterizer` mode, bbox-preserving recentering, `trust_remote_code` |

Simulation and render settings (2048×2048 renders, no seam lines in the UV texture) live in `configs/sim_props.yaml`
instead of GarmentCodeRC's defaults.

## ComfyUI (multi-view generation)

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git third_party/ComfyUI
git -C third_party/ComfyUI checkout fd734e95    # the ComfyUI commit OmniFabric was tested with
conda create -n comfyui python=3.12 -y && conda activate comfyui
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r third_party/ComfyUI/requirements.txt
```

ComfyUI is pinned to commit `fd734e95` (tested with PyTorch 2.14 + CUDA 13.0). The models go into
`third_party/ComfyUI/models/{diffusion_models,text_encoders,vae,loras}` (`scripts/download_checkpoints.sh`). Models stored
elsewhere can be registered through ComfyUI's `extra_model_paths.yaml`.

Start the server with `GPU=<id> bash scripts/start_comfyui.sh`. A 360° orbit takes about 6-12 minutes on an RTX A6000;
`orbit_views.py` asks the server to unload its models afterwards.

## Memory

All steps run one after another on a single 48 GB GPU: FLUX.2-dev (4-bit NF4) needs about 28 GB, FLUX.1-dev with the
OmniFabric LoRA about 35 GB, and MiniMax-H3 (int8) uses ComfyUI's weight offloading.
