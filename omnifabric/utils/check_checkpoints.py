"""Check that the models the pipeline needs are in place before it starts, and list everything that is missing at once.

Models a re-run would reuse results for (see scripts/run_pipeline.sh) are not required again.

Usage:
    python -m omnifabric.utils.check_checkpoints --output_dir <output_dir> --texture_backend <ask|flux2|gemini> \
        --views_backend <minimax360|flux2> --chatgarment_ckpt <file> --omnifabric_ckpt <dir> --comfy_server <url>
"""

import argparse
import glob
import json
import os
import sys
import urllib.error
import urllib.request

from huggingface_hub import auth_check, constants
from huggingface_hub.errors import GatedRepoError

from omnifabric.coarse.orbit_views import LORA, TEXT_ENCODER, UNET, VIDEO_VAE

OMNIFABRIC_URL = "https://drive.google.com/file/d/1qMdaFTy8Mbh4yGcm41FE4iMOqUrGIp9k/view?usp=sharing"
CHATGARMENT_URL = "https://drive.google.com/drive/folders/17JVZ1HYgMMMxw_zWmDgiVQ_PkIN-Fznf?usp=sharing"
COMFY_MODELS = [("UNETLoader", "unet_name", UNET), ("CLIPLoader", "clip_name", TEXT_ENCODER),
                ("VAELoader", "vae_name", VIDEO_VAE), ("LoraLoaderModelOnly", "lora_name", LORA)]


def hf_status(repo, cache_dir=None):
    """'cached', 'download' (will be fetched on first use), 'gated' (no access) or 'unreachable'."""
    snapshots = os.path.join(cache_dir or constants.HF_HUB_CACHE, f"models--{repo.replace('/', '--')}", "snapshots")
    if os.path.isdir(snapshots) and os.listdir(snapshots):
        return "cached"
    try:
        auth_check(repo)
        return "download"
    except GatedRepoError:
        return "gated"
    except Exception:
        return "unreachable"


def missing_comfy_models(server):
    """Ask the server, so models registered through extra_model_paths.yaml or on another machine are found too."""
    missing = []
    for node, field, name in COMFY_MODELS:
        with urllib.request.urlopen(f"{server}/object_info/{node}", timeout=10) as response:
            spec = json.load(response).get(node, {}).get("input", {}).get("required", {}).get(field)
        if not spec:
            continue
        options = spec[0] if isinstance(spec[0], list) else spec[1].get("options", [])  # older / newer ComfyUI
        if name not in options:
            missing.append(name)
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--texture_backend", required=True)
    parser.add_argument("--views_backend", required=True)
    parser.add_argument("--chatgarment_ckpt", required=True)
    parser.add_argument("--omnifabric_ckpt", required=True)
    parser.add_argument("--comfy_server", required=True)
    args = parser.parse_args()

    specs_file = os.path.join(args.output_dir, "chatgarment", "vis_new", "all_json_spec_files.json")
    predicted = os.path.isfile(specs_file)
    garments = []
    if predicted:
        with open(specs_file) as f:
            garments = [os.path.basename(p).replace("_specification.json", "").replace("validate_garment", "valid_garment")
                        for p in json.load(f)]

    garments = [g for g in garments if os.path.isdir(os.path.join(args.output_dir, g))]  # the ones textured so far

    def reusable(pattern):
        return bool(garments) and all(glob.glob(os.path.join(args.output_dir, g, pattern)) for g in garments)

    problems = []

    if not predicted and not os.path.isfile(args.chatgarment_ckpt):
        problems.append(f"ChatGarment weights not found. Download pytorch_model.bin from\n      {CHATGARMENT_URL}\n"
                        f"    and save it as {args.chatgarment_ckpt}")

    lora = os.path.join(args.omnifabric_ckpt, "default.safetensors")
    if not os.path.isfile(lora):
        problems.append(f"OmniFabric LoRA not found. Download it from\n      {OMNIFABRIC_URL}\n    and save it as {lora}")

    if args.views_backend == "minimax360" and not reusable("views/orbit/*_back.png"):
        try:
            missing = missing_comfy_models(args.comfy_server)
            if missing:
                problems.append(f"ComfyUI ({args.comfy_server}) is missing: {', '.join(missing)}\n"
                                "    Run: bash scripts/download_checkpoints.sh")
        except (urllib.error.URLError, OSError):
            problems.append(f"Cannot reach the ComfyUI server at {args.comfy_server}. Start it in another terminal:\n"
                            "      conda activate comfyui && GPU=0 bash scripts/start_comfyui.sh\n"
                            "    (see the Inference section of README.md)")

    # models downloaded automatically on first use: what can fail is gated access and the connection
    needs_flux2 = ((args.texture_backend == "flux2" and not reusable("views/front_garment.png"))
                   or (args.views_backend == "flux2" and not reusable("views/back_garment.png")))
    repos = [("black-forest-labs/FLUX.1-dev", os.environ.get("FLUX1_CACHE_DIR"), True),
             ("tencent/Hunyuan3D-2", None, True)]
    if needs_flux2 or (args.texture_backend == "ask" and not reusable("views/front_garment.png")):
        repos.append(("black-forest-labs/FLUX.2-dev", os.environ.get("FLUX2_CACHE_DIR"), needs_flux2))
    if not predicted:
        repos += [("liuhaotian/llava-v1.5-7b", None, True), ("openai/clip-vit-large-patch14-336", None, True)]

    to_download = []
    for repo, cache_dir, required in repos:
        status = hf_status(repo, cache_dir)
        if status == "download":
            to_download.append(repo)
        elif status != "cached":
            reason = (f"accept its license at https://huggingface.co/{repo} and run `huggingface-cli login`"
                      if status == "gated" else "it is not cached and Hugging Face cannot be reached")
            if required:
                problems.append(f"Cannot download {repo}: {reason}")
            else:
                print(f"Note: the FLUX.2-dev option will not work: {reason}")

    if problems:
        print("Some checkpoints are not ready:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        sys.exit(1)

    print("Checkpoints are ready.")
    if to_download:
        print(f"Downloaded on first use into the Hugging Face cache: {', '.join(to_download)}")


if __name__ == "__main__":
    main()
