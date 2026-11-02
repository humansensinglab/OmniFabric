#!/bin/bash
# Download the checkpoints used by OmniFabric inference. The OmniFabric LoRA and ChatGarment weights are downloaded
# manually: the script prints their links if they are missing.
#
#   bash scripts/download_checkpoints.sh            # MiniMax-H3 models for ComfyUI; lists the manual downloads
#
# FLUX.1-dev, FLUX.2-dev, Hunyuan3D-2, LLaVA-1.5-7B and CLIP are fetched automatically into the Hugging Face
# cache ($HF_HOME) on first use, Real-ESRGAN into checkpoints/realesrgan/ (FLUX models are gated: accept their
# licenses and `huggingface-cli login`).

set -e
cd "$(dirname "$0")/.."
COMFYUI_DIR=${COMFYUI_DIR:-third_party/ComfyUI}

# ---- MiniMax-H3 (FL2VA) + 360-degree orbit LoRA for ComfyUI (~51 GB; see the MiniMax-H3 license first)
H3=Comfy-Org/MiniMax-H3
huggingface-cli download $H3 diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors --local-dir "$COMFYUI_DIR/models"
huggingface-cli download $H3 text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors --local-dir "$COMFYUI_DIR/models"
huggingface-cli download $H3 vae/minimax_h3_video_vae_int8_convrot.safetensors --local-dir "$COMFYUI_DIR/models"
huggingface-cli download pablodawson/MiniMax-H3-360-Orbit-LoRA minimax_h3_flf2v_lora_v1.safetensors --local-dir "$COMFYUI_DIR/models/loras"

# ---- OmniFabric texture normalization LoRA (FLUX.1-dev): download manually
if [[ ! -f checkpoints/omnifabric/default.safetensors ]]; then
    mkdir -p checkpoints/omnifabric
    echo "Download the OmniFabric LoRA from"
    echo "  https://drive.google.com/file/d/1qMdaFTy8Mbh4yGcm41FE4iMOqUrGIp9k/view?usp=sharing"
    echo "and save it as checkpoints/omnifabric/default.safetensors"
fi

# ---- ChatGarment (sewing pattern prediction): copy of the authors' weights (licenses in the same folder), download manually
if [[ ! -f checkpoints/chatgarment/pytorch_model.bin ]]; then
    mkdir -p checkpoints/chatgarment
    echo "Download pytorch_model.bin (ChatGarment weights) from"
    echo "  https://drive.google.com/drive/folders/17JVZ1HYgMMMxw_zWmDgiVQ_PkIN-Fznf?usp=sharing"
    echo "and save it as checkpoints/chatgarment/pytorch_model.bin"
fi
