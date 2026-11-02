#!/usr/bin/env python3
"""
OmniFabric texture normalization (FLUX.1-dev + OmniFabric LoRA, OminiControl conditioning): generate the normalized
texture T in the sewing-pattern UV space from the coarse texture T', the template, the position map and the frontal
rendering R_f.

Usage:
    python -m omnifabric.normalize.infer --checkpoint checkpoints/omnifabric \
        --mask_path editable_template.png --proj_mask_path coarse_texture.png --use_proj_mask \
        --pos_path garment_position_map.pt --ref_path frontal_rendering.png --prompt "A textured sewing pattern" \
        --output normalize/normalized.png --target_size 1024 --postprocess --use_sr --denoise_full

Writes <output> (the 1024 canvas) and, with --postprocess, <output>_fullres.png at the template's resolution.
"""

import torch
import argparse
import os
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Optional, Tuple
import torchvision.transforms as T
import cv2

from diffusers.pipelines import FluxPipeline
from omnifabric.paths import REPO_ROOT
from omnifabric.normalize.flux_omini import Condition, generate, seed_everything
from omnifabric.utils.quiet import quiet

# Global variable for RealESRGAN model
_realesrgan_model = None

# Per-channel normalization ranges computed across the entire dataset.
# Must match the values in GarmentDataset (train_garment.py).
POS_CHANNEL_MINS = torch.tensor([-74.62, -83.75, -41.48])  # X, Y, Z
POS_CHANNEL_MAXS = torch.tensor([ 73.44, 173.22,  27.02])  # X, Y, Z


def load_position_tensor(pos_path: str, target_size: int = 512) -> Image.Image:
    """
    Load and preprocess 3D position map from .pt file.

    Args:
        pos_path: Path to position map (.pt file)
        target_size: Target size for output image

    Returns:
        PIL Image of normalized position map
    """
    # Load position map (PyTorch tensor or dict containing tensor)
    pos_data = torch.load(pos_path, weights_only=False)
    if isinstance(pos_data, dict):
        pos_tensor = pos_data['position_map']  # Shape: (H, W, 3)
    else:
        pos_tensor = pos_data

    # Create mask for empty regions (where position is [0,0,0])
    is_empty = (pos_tensor.abs().sum(dim=2) < 1e-6)

    # Per-channel normalization: each axis mapped independently to [0, 1]
    channel_ranges = POS_CHANNEL_MAXS - POS_CHANNEL_MINS
    tensor_norm = (pos_tensor - POS_CHANNEL_MINS) / channel_ranges
    tensor_norm = torch.clamp(tensor_norm, 0.0, 1.0)

    # Set empty regions to white (to match mask's empty regions)
    tensor_norm[is_empty] = 1.0

    # Convert to PIL Image
    img = Image.fromarray((tensor_norm.numpy() * 255).astype(np.uint8))

    return img


def resize_and_pad(img: Image.Image, target_size: int = 512) -> tuple[Image.Image, Image.Image]:
    """
    Resize image preserving aspect ratio, then pad to square.

    Args:
        img: Input PIL Image
        target_size: Target size for output (default 512)

    Returns:
        Tuple of (padded_image, padding_mask)
    """
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Create padding mask (True = valid region, False = padded)
    pad_mask = Image.new('L', (target_size, target_size), 0)
    pad_mask.paste(255, (0, 0, new_w, new_h))

    # Pad image
    if img.mode == 'RGB':
        padded = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    elif img.mode == 'L':
        padded = Image.new('L', (target_size, target_size), 255)
    else:  # RGBA or other
        padded = Image.new(img.mode, (target_size, target_size), (255, 255, 255, 0))
    padded.paste(img_resized, (0, 0))

    return padded, pad_mask


def create_token_mask(img: Image.Image, pad_mask: Image.Image, target_size: int = 512) -> torch.Tensor:
    """
    Create token-level mask from image-level mask.

    Args:
        img: Mask image (black = valid pattern, white = empty)
        pad_mask: Padding mask from resize_and_pad
        target_size: Image size (default 512)

    Returns:
        Boolean tensor of shape (token_h * token_w,) where True = valid token
    """
    to_tensor = T.ToTensor()

    img_t = to_tensor(img.convert('L'))
    pad_t = to_tensor(pad_mask)

    # For mask: black regions (< 0.5) are valid sewing pattern
    valid_mask = (img_t < 0.5) & (pad_t > 0.5)

    # Downsample to token resolution (512 -> 64 -> 32 = 32x32 tokens)
    token_h = target_size // 16
    token_w = target_size // 16

    valid_mask_2d = valid_mask.squeeze(0)
    token_mask = torch.zeros(token_h, token_w, dtype=torch.bool)
    for i in range(token_h):
        for j in range(token_w):
            patch = valid_mask_2d[i*16:(i+1)*16, j*16:(j+1)*16]
            token_mask[i, j] = patch.any()

    return token_mask.flatten()


def create_ref_mask(img: Image.Image, target_size: int = 512) -> torch.Tensor:
    """
    Create mask from reference image transparency.

    Args:
        img: Reference image (potentially with alpha channel)
        target_size: Image size (default 512)

    Returns:
        Boolean tensor of shape (token_h * token_w,) where True = valid token
    """
    to_tensor = T.ToTensor()

    if img.mode == 'RGBA':
        alpha = np.array(img)[:, :, 3]
        alpha_mask = Image.fromarray((alpha > 0).astype(np.uint8) * 255)
    else:
        alpha_mask = Image.new('L', img.size, 255)

    alpha_t = to_tensor(alpha_mask)
    token_h = target_size // 16
    token_w = target_size // 16

    alpha_2d = alpha_t.squeeze(0)
    token_mask = torch.zeros(token_h, token_w, dtype=torch.bool)
    for i in range(token_h):
        for j in range(token_w):
            patch = alpha_2d[i*16:(i+1)*16, j*16:(j+1)*16]
            token_mask[i, j] = patch.any()

    return token_mask.flatten()


def remove_padding(img: Image.Image, original_size: Tuple[int, int], target_size: int = 512) -> Image.Image:
    """
    Remove padding from a padded image to restore aspect ratio.

    Args:
        img: Padded image (square, target_size x target_size)
        original_size: Original image size (w, h) before resize and pad
        target_size: Target size that was used for padding

    Returns:
        Image with padding removed (maintains aspect ratio of original)
    """
    orig_w, orig_h = original_size
    scale = target_size / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Crop to remove padding
    unpadded = img.crop((0, 0, new_w, new_h))
    return unpadded


def load_realesrgan_model(device: str = "cuda", model_name: str = "RealESRGAN_x4plus"):
    """RealESRGAN x4 upsampler (loaded once; the weights are downloaded on first use)."""
    global _realesrgan_model

    if _realesrgan_model is not None:
        return _realesrgan_model

    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    # Define model architecture
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    # downloaded on first use into checkpoints/realesrgan/
    model_dir = os.path.join(REPO_ROOT, 'checkpoints', 'realesrgan')
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    if not os.path.exists(model_path):
        from basicsr.utils.download_util import load_file_from_url
        model_path = load_file_from_url(
            f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth",
            model_dir=model_dir, progress=True)

    # Create upsampler
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=256,  # Tile size for processing large images
        tile_pad=10,
        pre_pad=0,
        half=True if device == "cuda" else False,
        device=device,
    )

    _realesrgan_model = upsampler
    return upsampler


def progressive_super_resolution(
    img: Image.Image,
    target_size: Tuple[int, int],
    device: str = "cuda",
    use_sr: bool = True,
) -> Image.Image:
    """
    Apply progressive super-resolution to upscale image to target size.

    Strategy:
    1. If scale <= 4x: Apply SR once, then bicubic downsample to exact target
    2. If scale > 4x: Apply SR (4x) -> bicubic downsample to 1/4 target -> SR again (4x) -> bicubic to exact target

    Args:
        img: Input image to upscale
        target_size: Target size (width, height)
        device: Device for SR model
        use_sr: Whether to use super-resolution (False = bicubic only)

    Returns:
        Upscaled image at target size
    """
    current_w, current_h = img.size
    target_w, target_h = target_size

    scale_w = target_w / current_w
    scale_h = target_h / current_h
    scale = max(scale_w, scale_h)

    if not use_sr or scale <= 1.0:
        return img, img.resize(target_size, Image.BICUBIC)

    # Convert PIL to CV2 format for RealESRGAN
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    original_lr = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    if scale <= 4.0:
        # Single SR pass
        upsampler = load_realesrgan_model(device)
        with quiet():  # RealESRGAN prints every tile
            sr_img, _ = upsampler.enhance(img_cv, outscale=4)

        # Convert back to PIL
        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)
        sr_pil = Image.fromarray(sr_img)

        # Downsample to exact target size
        final_img = sr_pil.resize(target_size, Image.BICUBIC)
        return original_lr, final_img

    else:
        # Progressive SR: two passes
        upsampler = load_realesrgan_model(device)

        # First pass: 4x upscale
        with quiet():
            sr_img1, _ = upsampler.enhance(img_cv, outscale=4)
        sr_img1 = cv2.cvtColor(sr_img1, cv2.COLOR_BGR2RGB)
        sr_pil1 = Image.fromarray(sr_img1)

        # Intermediate downsample to 1/4 of target size
        intermediate_w = target_w // 4
        intermediate_h = target_h // 4
        intermediate_img = sr_pil1.resize((intermediate_w, intermediate_h), Image.BICUBIC)

        # Second pass: 4x upscale again
        intermediate_cv = cv2.cvtColor(np.array(intermediate_img), cv2.COLOR_RGB2BGR)
        with quiet():
            sr_img2, _ = upsampler.enhance(intermediate_cv, outscale=4)
        sr_img2 = cv2.cvtColor(sr_img2, cv2.COLOR_BGR2RGB)
        sr_pil2 = Image.fromarray(sr_img2)

        # Final downsample to exact target size
        final_img = sr_pil2.resize(target_size, Image.BICUBIC)
        return original_lr, final_img


def postprocess_result(
    generated_img: Image.Image,
    original_mask: Image.Image,
    target_size: int = 512,
    device: str = "cuda",
    use_sr: bool = True,
) -> Image.Image:
    """
    Post-process generated result to match original resolution.

    Steps:
    1. Remove padding to restore aspect ratio
    2. Apply progressive super-resolution to original resolution

    Args:
        generated_img: Generated image (padded, target_size x target_size)
        original_mask: Original mask image (for size reference)
        target_size: Target size used during generation
        device: Device for SR model
        use_sr: Whether to use super-resolution

    Returns:
        Processed image at original resolution
    """
    # Step 1: Remove padding
    unpadded = remove_padding(generated_img, original_mask.size, target_size)

    # Step 2: Progressive super-resolution to original size
    original_lr, final_img = progressive_super_resolution(
        unpadded,
        original_mask.size,
        device=device,
        use_sr=use_sr,
    )

    return original_lr, final_img


def load_pipeline(
    checkpoint_path: str,
    base_model: str = "black-forest-labs/FLUX.1-dev",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: Optional[str] = None,
) -> FluxPipeline:
    """
    Load FLUX pipeline with trained LoRA weights.

    Args:
        checkpoint_path: Path to checkpoint directory (e.g., "runs/20251021-004031/ckpt/3000")
        base_model: Base FLUX model to use
        device: Device to load model on
        dtype: Data type for model
        cache_dir: Cache directory for HuggingFace models

    Returns:
        FluxPipeline with loaded LoRA weights
    """
    # Load base FLUX pipeline
    load_kwargs = {"torch_dtype": dtype}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    pipe = FluxPipeline.from_pretrained(base_model, **load_kwargs)
    pipe = pipe.to(device)

    # Load LoRA weights from checkpoint
    checkpoint_path = Path(checkpoint_path)
    lora_weight_path = checkpoint_path / "default.safetensors"

    if not lora_weight_path.exists():
        raise FileNotFoundError(
            f"LoRA weights not found at {lora_weight_path}. Download them from "
            "https://drive.google.com/file/d/1qMdaFTy8Mbh4yGcm41FE4iMOqUrGIp9k/view?usp=sharing"
        )

    pipe.load_lora_weights(
        str(checkpoint_path),
        weight_name="default.safetensors",
        adapter_name="default",
    )

    return pipe


def generate_garment_texture(
    pipe: FluxPipeline,
    original_mask_img: Image.Image,
    cond0_mask_img: Image.Image,
    pos_img: Optional[Image.Image],
    ref_img: Optional[Image.Image],
    prompt: str,
    target_size: int = 512,
    ref_size: int = 512,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    image_guidance_scale: float = 1.0,
    seed: Optional[int] = None,
    device: str = "cuda",
    mask_denoising: bool = True,
    use_ref_image: bool = True,
    use_pos_map: bool = True,
) -> Image.Image:
    """
    Generate textured sewing pattern from inputs.

    Args:
        pipe: Loaded FluxPipeline with LoRA weights
        original_mask_img: Original mask image (for token mask creation, black=pattern, white=empty)
        cond0_mask_img: Mask image for condition_0 input (can be original or projected)
        pos_img: 3D position map image
        ref_img: Reference image of person wearing garment
        prompt: Text description
        target_size: Output resolution for GT/mask/pos (default 512)
        ref_size: Reference image resolution (default 512)
        num_inference_steps: Number of denoising steps
        guidance_scale: Text guidance scale
        image_guidance_scale: Image condition guidance scale
        seed: Random seed (None for random)
        device: Device to generate on
        mask_denoising: If True (default), only denoise sewing pattern tokens (matches
            training). If False, denoise the full image including abundant regions.
        use_ref_image: If True (default), include reference image as a condition.
        use_pos_map: If True (default), include position map as a condition.

    Returns:
        Generated textured sewing pattern image
    """
    # Set random seed if provided
    if seed is not None:
        seed_everything(seed)

    # Resize and pad spatially-aligned maps (original_mask, cond0_mask)
    original_mask_padded, pad_mask = resize_and_pad(original_mask_img, target_size)
    cond0_mask_padded, _ = resize_and_pad(cond0_mask_img, target_size)

    # Conditionally preprocess position map
    if use_pos_map:
        pos_padded, _ = resize_and_pad(pos_img, target_size)

    # Conditionally preprocess reference image
    if use_ref_image:
        ref_resized = ref_img.resize((ref_size, ref_size), Image.LANCZOS)
        ref_rgb = ref_resized.convert('RGB')

    # Create token masks using ORIGINAL mask (defines spatial extent)
    uv_token_mask = create_token_mask(original_mask_padded, pad_mask, target_size)
    if use_ref_image:
        ref_token_mask = create_ref_mask(ref_resized, ref_size)

    # Build conditions — mask is always condition 0; ref and pos added if active
    conditions = [
        # Condition 0: Garment mask (aligned with output)
        # Uses cond0_mask (can be original or projected)
        # But token mask always from original mask
        Condition(
            cond0_mask_padded.convert('RGB'),
            adapter_setting="default",
            position_delta=(0, 0),
            latent_mask=uv_token_mask.unsqueeze(0).to(device),
        ),
    ]
    if use_ref_image:
        # Condition: Reference image (shifted left by ref_size//16 tokens)
        conditions.append(Condition(
            ref_rgb,
            adapter_setting="default",
            position_delta=(0, -ref_size // 16),
            latent_mask=ref_token_mask.unsqueeze(0).to(device),
        ))
    if use_pos_map:
        # Condition: Position map (aligned with output)
        conditions.append(Condition(
            pos_padded.convert('RGB'),
            adapter_setting="default",
            position_delta=(0, 0),
            latent_mask=uv_token_mask.unsqueeze(0).to(device),
        ))

    result = generate(
        pipe,
        prompt=prompt,
        conditions=conditions,
        height=target_size,
        width=target_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        latent_mask=uv_token_mask.to(device) if mask_denoising else None,
    )

    return result.images[0]


def main():
    parser = argparse.ArgumentParser(
        description="Inference for garment texture generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint", "-c", required=True,
        help="Path to checkpoint directory containing default.safetensors (e.g., checkpoints/omnifabric)"
    )
    parser.add_argument(
        "--base_model", default="black-forest-labs/FLUX.1-dev",
        help="Base FLUX model (default: FLUX.1-dev)"
    )
    parser.add_argument(
        "--cache_dir", default=os.environ.get("FLUX1_CACHE_DIR"),
        help="HuggingFace cache directory for FLUX.1-dev (default: $FLUX1_CACHE_DIR, else the standard $HF_HOME cache)"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device to run inference on (default: cuda)"
    )

    # Input arguments (single sample)
    parser.add_argument(
        "--mask_path", required=True, help="Path to the binary UV template (black = sewing pattern)"
    )
    parser.add_argument(
        "--proj_mask_path", help="Path to the coarse texture map (used with --use_proj_mask)"
    )
    parser.add_argument(
        "--pos_path", help="Path to position map (.pt file)"
    )
    parser.add_argument(
        "--ref_path", help="Path to reference image"
    )
    parser.add_argument(
        "--prompt", default="A textured sewing pattern", help="Text prompt"
    )

    # Output arguments
    parser.add_argument(
        "--output", "-o", required=True, help="Output path"
    )

    # Generation parameters
    parser.add_argument(
        "--target_size", type=int, default=512,
        help="Output resolution for GT/mask/pos (default: 512, use 1024 for higher res)"
    )
    parser.add_argument(
        "--ref_size", type=int, default=512,
        help="Reference image resolution (default: 512)"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=28,
        help="Number of denoising steps (default: 28)"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=3.5,
        help="Text guidance scale (default: 3.5)"
    )
    parser.add_argument(
        "--image_guidance_scale", type=float, default=1.0,
        help="Image condition guidance scale (default: 1.0)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: random)"
    )
    parser.add_argument(
        "--use_proj_mask", action="store_true",
        help="Condition on the coarse texture (--proj_mask_path) instead of the empty template"
    )
    parser.add_argument(
        "--no_ref_image", action="store_true",
        help="Disable reference image as a condition (ablation; default: enabled)"
    )
    parser.add_argument(
        "--no_pos_map", action="store_true",
        help="Disable position map as a condition (ablation; default: enabled)"
    )
    parser.add_argument(
        "--save_conditions", action="store_true",
        help="Save condition images alongside output"
    )
    parser.add_argument(
        "--denoise_full", action="store_true",
        help="Denoise the full image including abundant regions (default: only denoise sewing pattern)"
    )

    # Post-processing arguments
    parser.add_argument(
        "--postprocess", action="store_true",
        help="Apply post-processing to restore original resolution"
    )
    parser.add_argument(
        "--use_sr", action="store_true",
        help="Use RealESRGAN for super-resolution (requires --postprocess)"
    )

    args = parser.parse_args()

    use_ref_image = not args.no_ref_image
    use_pos_map = not args.no_pos_map
    if use_pos_map and not args.pos_path:
        parser.error("--pos_path is required (or --no_pos_map)")
    if use_ref_image and not args.ref_path:
        parser.error("--ref_path is required (or --no_ref_image)")
    if args.use_proj_mask and not args.proj_mask_path:
        parser.error("--use_proj_mask requires --proj_mask_path")

    print("Normalizing the texture with OmniFabric (FLUX.1-dev + LoRA)")
    pipe = load_pipeline(args.checkpoint, args.base_model, args.device, torch.bfloat16, args.cache_dir)

    # the template defines the token masks; the coarse texture (or the template) is condition 0
    original_mask_img = Image.open(args.mask_path)
    cond0_mask_img = Image.open(args.proj_mask_path) if args.use_proj_mask else original_mask_img
    pos_img = load_position_tensor(args.pos_path, args.target_size) if use_pos_map else None
    ref_img = Image.open(args.ref_path) if use_ref_image else None

    output_img = generate_garment_texture(
        pipe, original_mask_img, cond0_mask_img, pos_img, ref_img, args.prompt,
        args.target_size, args.ref_size, args.num_inference_steps,
        args.guidance_scale, args.image_guidance_scale,
        args.seed, args.device,
        mask_denoising=not args.denoise_full,
        use_ref_image=use_ref_image,
        use_pos_map=use_pos_map,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_img.save(args.output)
    base_path = Path(args.output).with_suffix("")

    if args.postprocess:
        original_lr, processed_img = postprocess_result(
            output_img, original_mask_img, target_size=args.target_size, device=args.device, use_sr=args.use_sr)
        processed_img.save(f"{base_path}_fullres.png")
        original_lr.save(f"{base_path}_original_lr.png")

    if args.save_conditions:
        resize_and_pad(cond0_mask_img, args.target_size)[0].save(f"{base_path}_cond0_mask.png")
        if use_pos_map:
            resize_and_pad(pos_img, args.target_size)[0].save(f"{base_path}_pos.png")
        if use_ref_image:
            ref_img.resize((args.ref_size, args.ref_size), Image.LANCZOS).save(f"{base_path}_ref.png")


if __name__ == "__main__":
    main()
