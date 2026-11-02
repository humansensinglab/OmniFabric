#!/bin/bash
# Start the ComfyUI server that runs MiniMax-H3 + the 360-orbit LoRA (multi-view generation).
# Run it in the separate `comfyui` environment (see docs/INSTALL.md), e.g. in another terminal or tmux:
#
#   conda activate comfyui
#   GPU=0 bash scripts/start_comfyui.sh            # serves on http://127.0.0.1:8188
#
# Models are loaded on the first request and unloaded by omnifabric.coarse.orbit_views when it is done.

COMFYUI_DIR=${COMFYUI_DIR:-$(cd "$(dirname "$0")/.." && pwd)/third_party/ComfyUI}
cd "$COMFYUI_DIR"
CUDA_VISIBLE_DEVICES=${GPU:-0} exec python main.py --listen 127.0.0.1 --port "${PORT:-8188}" "$@"
