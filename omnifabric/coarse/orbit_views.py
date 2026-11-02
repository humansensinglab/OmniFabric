"""Generate 360-degree orbit videos with MiniMax-H3 FL2VA + pablodawson/MiniMax-H3-360-Orbit-LoRA.

Submits an API-format workflow to a running ComfyUI server (see scripts/start_comfyui.sh). Settings follow the LoRA model card:
same image as first and last keyframe, 768x768, 73 frames @ 24 fps, 28 steps, no CFG, LoRA strength 1.0.

Each generated video is copied to --out_dir together with its 4 orthogonal views
(<stem>_front/left/back/right.png, plus a <stem>_views.png strip in that order). "left"/"right" are the
subject's own left/right side, i.e. the camera standing on the subject's left/right.

Usage:
    python -m omnifabric.coarse.orbit_views front_garment.png --out_dir views/ [--lora 1.0] [--seed 42]
    python -m omnifabric.coarse.orbit_views views/front_garment_orbit_s42_00001_.mp4 --out_dir views/  # only extract views
"""
import argparse
import json
import os
import shutil
import time
import urllib.request
import uuid

import av
import cv2
import numpy as np
from PIL import Image

from omnifabric.paths import THIRD_PARTY

PROMPT = (
    "One frozen instant. Only the camera moves. In a continuous 360 orbit. Preserve every person and object "
    "in exactly the same world position, orientation, shape and pose throughout the shot. Airborne objects remain "
    "suspended at the captured height and angle: no wobbling, shaking, spinning, drifting, falling or continued "
    "action. Keep faces, hands, clothing, liquids and the background motionless while retaining their natural "
    "appearance. Camera parallax is the only source of apparent movement. No cuts, zoom, morphing or added objects."
)

COMFY_DIR = os.environ.get("COMFYUI_DIR", os.path.join(THIRD_PARTY, "ComfyUI"))
UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
LORA = "minimax_h3_flf2v_lora_v1.safetensors"


def build_workflow(image_name, prefix, seed, lora_strength, size, length, steps):
    wf = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "image": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "i2v": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["clip", 0], "vae": ["vae", 0], "prompt": PROMPT,
            "width": size, "height": size, "length": length,
            "first_frame": ["image", 0], "last_frame": ["image", 0]}},
        "guider": {"class_type": "BasicGuider", "inputs": {"model": ["model", 0], "conditioning": ["i2v", 0]}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "sigmas": {"class_type": "BasicScheduler", "inputs": {"model": ["model", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
            "sigmas": ["sigmas", 0], "latent_image": ["i2v", 1]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"vae": ["vae", 0], "samples": ["sample", 0]}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "fps": 24.0}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": prefix, "format": "auto", "format.codec": "auto"}},
    }
    if lora_strength > 0:
        wf["model"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": LORA, "strength_model": lora_strength}}
    else:
        wf["model"] = {"class_type": "ModelSamplingMiniMaxH3", "inputs": {"model": ["unet", 0], "shift_video": 12.0, "shift_audio": 3.0}}
    return wf


def _foreground_masks(frames, thresh=40.0):
    """Foreground masks against a plain backdrop estimated from the frame border; None if the backdrop isn't plain."""
    border = np.concatenate([frames[0][:8].reshape(-1, 3), frames[0][-8:].reshape(-1, 3),
                             frames[0][:, :8].reshape(-1, 3), frames[0][:, -8:].reshape(-1, 3)]).astype(np.float32)
    bg = np.median(border, axis=0)
    if np.median(np.linalg.norm(border - bg, axis=1)) > thresh / 2:
        return None
    return [np.linalg.norm(f.astype(np.float32) - bg, axis=-1) > thresh for f in frames]


def _orbit_direction(frames, masks):
    """+1 if the camera orbits toward the subject's left (front surface drifts image-left at the start), else -1."""
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    n = len(frames) - 1
    dx = []
    # loop start going forward and loop end going forward both show the same orbit direction
    for a in list(range(0, 8, 2)) + list(range(n - 8, n, 2)):
        flow = cv2.calcOpticalFlowFarneback(gray[a], gray[a + 2], None, 0.5, 4, 21, 3, 5, 1.1, 0)
        sel = masks[a] if masks is not None else np.ones(gray[a].shape, bool)
        dx.append(np.median(flow[..., 0][sel]))
    mean_dx = float(np.mean(dx))
    if abs(mean_dx) < 0.2:
        print(f"Warning: weak orbit-direction signal in the 360 video (dx={mean_dx:.2f}px); its left and right views "
              f"may be swapped")
    return (1 if mean_dx <= 0 else -1), mean_dx


def extract_orthogonal_views(video_path, out_dir=None, window=4):
    """Save the front/left/back/right frames of a closed-loop 360 orbit video; returns {view: (frame_idx, path)}.

    Views are taken at 0, 1/4, 1/2 and 3/4 of the loop (the last frame repeats the first). On a plain backdrop the
    side/back picks are refined within +-window frames to the narrowest/widest silhouette.
    """
    frames = [f.to_ndarray(format="rgb24") for f in av.open(video_path).decode(video=0)]
    n = len(frames) - 1
    masks = _foreground_masks(frames)
    picks = {"front": 0, "side_a": round(n / 4), "back": round(n / 2), "side_b": round(3 * n / 4)}
    if masks is not None:
        # horizontal spread of the silhouette: smallest at the profiles, largest at front/back
        spread = np.array([np.nonzero(m)[1].std() if m.any() else 0.0 for m in masks])
        for view, pick in (("side_a", np.argmin), ("back", np.argmax), ("side_b", np.argmin)):
            lo = picks[view] - window
            picks[view] = lo + int(pick(spread[lo:picks[view] + window + 1]))
    direction, _ = _orbit_direction(frames, masks)
    first, second = ("left", "right") if direction > 0 else ("right", "left")
    picks = {"front": picks["front"], first: picks["side_a"], "back": picks["back"], second: picks["side_b"]}

    out_dir = out_dir or os.path.dirname(video_path)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0].rstrip("_")  # ComfyUI names end in "_00001_"
    views = {}
    for view in ("front", "left", "back", "right"):
        path = os.path.join(out_dir, f"{stem}_{view}.png")
        Image.fromarray(frames[picks[view]]).save(path)
        views[view] = (picks[view], path)
    strip = np.concatenate([frames[views[v][0]] for v in ("front", "left", "back", "right")], axis=1)
    Image.fromarray(strip).save(os.path.join(out_dir, f"{stem}_views.png"))
    return views


def post(server, path, payload):
    req = urllib.request.Request(f"{server}{path}", json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def wait(server, prompt_id, label):
    """Wait for a ComfyUI job, showing the elapsed time."""
    t0 = time.time()
    while True:
        elapsed = int(time.time() - t0)
        print(f"\r{label} ({elapsed // 60}m {elapsed % 60:02d}s)", end="", flush=True)
        hist = json.load(urllib.request.urlopen(f"{server}/history/{prompt_id}"))
        if prompt_id in hist:
            status = hist[prompt_id]["status"]
            if status.get("completed") or status.get("status_str") == "error":
                print()
                return hist[prompt_id]
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--comfy_dir", default=COMFY_DIR, help="ComfyUI root (its input/ and output/ folders are used)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora", type=float, default=1.0, help="LoRA strength; 0 runs the base model for comparison")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--length", type=int, default=73)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--tag", default="orbit")
    ap.add_argument("--keep_loaded", action="store_true",
                    help="keep the models loaded on the ComfyUI server (by default they are unloaded afterwards "
                         "so later pipeline steps can use the GPU)")
    ap.add_argument("--out_dir", required=True,
                    help="where the video and its front/left/back/right views are saved")
    args = ap.parse_args()

    for path in args.images:
        if path.lower().endswith(".mp4"):
            extract_orthogonal_views(path, args.out_dir)
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        image_name = f"{stem}_rgb.png"
        Image.open(path).convert("RGB").save(os.path.join(args.comfy_dir, "input", image_name))
        prefix = f"h3_orbit/{stem}_{args.tag}_s{args.seed}"
        wf = build_workflow(image_name, prefix, args.seed, args.lora, args.size, args.length, args.steps)
        prompt_id = post(args.server, "/prompt", {"prompt": wf, "client_id": str(uuid.uuid4())})["prompt_id"]
        result = wait(args.server, prompt_id, "Generating the 360-degree orbit video with MiniMax-H3 (ComfyUI)")
        if result["status"]["status_str"] == "error":
            print(f"ComfyUI failed on {path}:")
            print(json.dumps(result["status"]["messages"], indent=1)[-3000:])
            continue
        for out in result["outputs"].values():
            for item in out.get("images", []) + out.get("videos", []):
                video = os.path.join(args.comfy_dir, "output", item.get("subfolder", ""), item["filename"])
                os.makedirs(args.out_dir, exist_ok=True)
                video = shutil.copy(video, args.out_dir)
                extract_orthogonal_views(video, args.out_dir)

    if not args.keep_loaded and any(not p.lower().endswith(".mp4") for p in args.images):
        req = urllib.request.Request(f"{args.server}/free", json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                     {"Content-Type": "application/json"})
        urllib.request.urlopen(req)


if __name__ == "__main__":
    main()
