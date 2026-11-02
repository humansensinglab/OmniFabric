#!/bin/bash
# Install OmniFabric's Python dependencies and third-party code into the active conda environment.
#
#   conda env create -f environment.yml
#   conda activate omnifabric
#   CUDA_HOME=/usr/local/cuda-11.8 bash install.sh   # add --no-flash-attn to skip building flash-attn (ChatGarment then uses SDPA)
#
# Third-party repositories are cloned into third_party/ at the commits OmniFabric was developed with,
# and the small patches in third_party/patches/ are applied on top.

set -e
cd "$(dirname "$0")"
REPO_ROOT=$(pwd)
TP="$REPO_ROOT/third_party"
WITH_FLASH_ATTN=1
[[ "$1" == "--no-flash-attn" ]] && WITH_FLASH_ATTN=0

# full CUDA 11.8 toolkit (nvcc, headers, static NVRTC libraries), matching the cu118 PyTorch build
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-11.8}
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]] || ! "$CUDA_HOME/bin/nvcc" --version | grep -q "release 11.8"; then
    echo "Set CUDA_HOME to a CUDA 11.8 toolkit (found: $CUDA_HOME)" && exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"

clone() {  # clone <url> <dir> <commit> [patch]
    if [[ ! -d "$TP/$2" ]]; then
        git clone "$1" "$TP/$2"
        git -C "$TP/$2" checkout "$3"
        if [[ -n "$4" ]]; then git -C "$TP/$2" apply --whitespace=nowarn "$TP/patches/$4"; fi
    fi
}

# ---- Python packages (PyTorch first: basicsr's setup.py imports it at build time)
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install --no-build-isolation basicsr==1.4.2
pip install -r requirements.txt
# basicsr 1.4.2 imports a module removed in torchvision >= 0.17
BASICSR=$(python -c "import importlib.util; print(importlib.util.find_spec('basicsr').submodule_search_locations[0])")
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    "$BASICSR/data/degradations.py"

# ---- ChatGarment (sewing pattern prediction); its pinned dependencies are superseded by requirements.txt
clone https://github.com/biansy000/ChatGarment.git ChatGarment 4403c362de24872af3d5c5d7636e4a594433ed4d ChatGarment.patch
pip install --no-deps -e "$TP/ChatGarment"

# ---- GarmentCodeRC (sewing patterns + simulation) and its NVIDIA Warp fork
clone https://github.com/biansy000/GarmentCodeRC.git GarmentCodeRC e3d4476d6f9bcb0af5ee0eccd18b0433e3df398a GarmentCodeRC.patch
[[ -f "$TP/GarmentCodeRC/system.json" ]] || cp "$TP/GarmentCodeRC/system.template.json" "$TP/GarmentCodeRC/system.json"
# ChatGarment runs from its own root and reads GarmentCode assets from there (as in its installation guide)
ln -sfn ../GarmentCodeRC/assets "$TP/ChatGarment/assets"
clone https://github.com/maria-korosteleva/NvidiaWarp-GarmentCode.git NvidiaWarp-GarmentCode 63baf6855efdd89b2834b74640f84b3bb0d86b50
chmod +x "$TP/NvidiaWarp-GarmentCode/tools/packman/packman"  # not executable in the upstream repo
(cd "$TP/NvidiaWarp-GarmentCode" && python build_lib.py --cuda_path "$CUDA_HOME" && pip install -e .)

# ---- Hunyuan3D-2 (renderer used for multi-view texture projection)
clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git Hunyuan3D-2 9bcf3028145311cbea5c127aed19717833b3a35f Hunyuan3D-2.patch
pip install --no-deps -e "$TP/Hunyuan3D-2"
(cd "$TP/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer" && pip install --no-build-isolation .)
(cd "$TP/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer" && pip install --no-build-isolation .)

# ---- flash-attn (optional, used by ChatGarment)
if [[ $WITH_FLASH_ATTN == 1 ]]; then
    MAX_JOBS=${MAX_JOBS:-8} pip install flash-attn==2.7.3 --no-build-isolation
fi

echo "OmniFabric installation finished."
