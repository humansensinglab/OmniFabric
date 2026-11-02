"""
Texture transfer with FLUX.2-dev: re-texture a grey garment render (image1, a frontal garment-only render on a green
screen) with the garment in a reference photo (image2), keeping image1's shape exactly.

    python -m omnifabric.coarse.texture_transfer --task texture-transfer --image1 grey_garment.png \
        --image2 photo.png -o textured.png [--steps 28 --guidance 4.0 --seed 42]

Mistral 3 first rewrites the prompt by looking at both images (--upsample, on for the task), and the output is
composited onto image1's own silhouette and background (--mask-composite, on for the task), so the outline always
matches image1 whatever the generation drift.

Memory: Mistral 3 (~47 GB in bf16) and the FLUX.2 transformer (~64 GB) only fit a 48 GB GPU in 4-bit NF4 (~28 GB).
"""

import argparse
import os
import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import BitsAndBytesConfig as TransformersNF4Config
from transformers import Mistral3ForConditionalGeneration, PixtralProcessor
from diffusers import BitsAndBytesConfig as DiffusersNF4Config
from diffusers import Flux2Transformer2DModel, Flux2Pipeline

REPO_ID  = "black-forest-labs/FLUX.2-dev"
HF_CACHE = os.environ.get("FLUX2_CACHE_DIR")  # None -> standard Hugging Face cache ($HF_HOME)
# must be set before any CUDA context is created (cuBLAS reads it at init time)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_deterministic(seed):
    """Seed every RNG the pipeline can touch and force deterministic GPU kernels.

    A same-seed run can still diverge run-to-run because (1) `random`/`numpy`
    were never seeded at all, and (2) PyTorch's default GPU math (TF32, cuDNN
    kernel autotuning) picks whichever kernel looks fastest for the current
    memory/context state, which can round differently. That drift then gets
    amplified by the temperature>0 Mistral caption resample in --upsample,
    turning a tiny float difference into a different rewritten prompt and a
    visibly different image. Calling this once at the top of main(), before
    any model loading, makes the whole run reproducible for a given seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # warn_only=True: some bitsandbytes NF4 / flash-attention kernels have no
    # deterministic implementation, so hard-erroring on them would be a false alarm.
    torch.use_deterministic_algorithms(True, warn_only=True)

NF4_KWARGS = dict(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

NF4_KWARGS = dict(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# image1 = grey textureless garment (shape), image2 = reference photo (texture); written so that Mistral 3's rewrite
# states both what changes and what stays the same
TASK_PROMPTS = {
    "texture-transfer": (
        "The first image is a frontal product photo of a single garment only, isolated on a "
        "plain green background, with grey textureless fabric. "
        "The second image is a separate reference photo that may include a person wearing a "
        "similarly patterned garment; that person is irrelevant to this task — their face, "
        "skin, hair, pose, body, and background must be completely ignored and must NOT appear "
        "anywhere in the output. From the second image, extract ONLY the fabric's texture, "
        "surface pattern, material appearance, and colors. "
        "Use that extracted material to replace the grey, textureless fabric of the garment in "
        "the first image. "
        "Keep unchanged from the first image: the garment's exact shape, silhouette, cut, fit, "
        "all folds, drape, seams, 3D structure, camera framing, scale, and the plain green "
        "background. "
        "The output must contain only the garment on the green background, matching the first "
        "image's outline pixel-for-pixel — no person, face, skin, hands, hair, body parts, or "
        "any new background elements. "
        "Result: the exact garment from the first image, re-fabricated with the material shown "
        "in the second image — identical shape and outline, new surface only."
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.2-dev texture transfer",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["texture-transfer"], default=None,
                        help="Preset task: sets the prompt and enables --upsample and --mask-composite.")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="Overrides the task preset prompt.")
    parser.add_argument("--image1", type=str, required=True, help="grey garment render (shape source)")
    parser.add_argument("--image2", type=str, required=True, help="reference photo (texture source)")
    parser.add_argument("-o", "--output", type=str, default="flux2_output.png")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upsample", action="store_true",
                        help="Let Mistral 3 rewrite the prompt by looking at the reference images.")
    parser.add_argument("--width", type=int, default=None, help="Output width (default: image1's width).")
    parser.add_argument("--height", type=int, default=None, help="Output height (default: image1's height).")
    parser.add_argument("--upsample-temperature", type=float, default=0.15,
                        help="Sampling temperature of the prompt rewrite (the library's recommendation; higher "
                             "values make the output less reproducible).")
    parser.add_argument("--mask-composite", action=argparse.BooleanOptionalAction, default=None,
                        help="Composite the output onto image1's own silhouette and background (chroma-keyed).")
    return parser.parse_args()


def green_screen_alpha(img, margin=30, feather=3, erode=1):
    """Garment-vs-background alpha mask for a green-screen product photo.

    Background pixels are ones where green clearly dominates red and blue —
    true for both green-screen variants seen in the mesh renders, e.g.
    (0,177,64) and (79,174,79). Eroding before feathering keeps a thin green
    fringe from surviving at the garment edge after compositing.
    """
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    is_background = (g > r + margin) & (g > b + margin)
    alpha = np.where(is_background, 0.0, 1.0).astype(np.float32)

    mask_img = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
    if erode > 0:
        mask_img = mask_img.filter(ImageFilter.MinFilter(2 * erode + 1))
    if feather > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(feather))
    return np.asarray(mask_img).astype(np.float32) / 255.0


def composite_onto_reference(generated, reference, margin=30, feather=3, erode=1):
    """Hard-guarantee the output's silhouette and background match `reference` exactly.

    `reference` (image1) is always a clean frontal garment-only render on a known
    green screen, so its silhouette is ground truth. Diffusion editing only
    *usually* preserves layout — this makes it exact regardless of any generation
    drift, which is the main "wrong silhouette" failure reported for this task.
    """
    if generated.size != reference.size:
        reference = reference.resize(generated.size, Image.LANCZOS)
    alpha = green_screen_alpha(reference, margin=margin, feather=feather, erode=erode)[..., None]

    gen_arr = np.asarray(generated.convert("RGB")).astype(np.float32)
    ref_arr = np.asarray(reference.convert("RGB")).astype(np.float32)
    out = gen_arr * alpha + ref_arr * (1 - alpha)
    return Image.fromarray(out.astype(np.uint8), mode="RGB")


def load_pipeline():
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        REPO_ID,
        subfolder="text_encoder",
        quantization_config=TransformersNF4Config(**NF4_KWARGS),
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
    )
    tokenizer = PixtralProcessor.from_pretrained(REPO_ID, subfolder="tokenizer", cache_dir=HF_CACHE)
    transformer = Flux2Transformer2DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        quantization_config=DiffusersNF4Config(**NF4_KWARGS),
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
    )
    pipe = Flux2Pipeline.from_pretrained(
        REPO_ID,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
    )
    # 4-bit models use accelerate dispatch hooks and cannot be moved: move only the non-quantized VAE
    pipe.vae = pipe.vae.to("cuda", dtype=torch.bfloat16)
    return pipe


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — this model requires a GPU.")
    # seed before any model loading, see set_deterministic()
    set_deterministic(args.seed)

    prompt = args.prompt or TASK_PROMPTS.get(args.task)
    if prompt is None:
        raise ValueError("Give a prompt (-p) or a task (--task texture-transfer)")
    upsample = args.upsample or args.task is not None
    mask_composite = args.mask_composite if args.mask_composite is not None else args.task == "texture-transfer"

    ref_images = []
    for path in (args.image1, args.image2):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Reference image not found: {path}")
        ref_images.append(Image.open(path).convert("RGB"))
    out_width, out_height = args.width, args.height
    if out_width is None and out_height is None:
        out_width, out_height = ref_images[0].size

    print("Generating the textured frontal view with FLUX.2-dev")
    pipe = load_pipeline()

    # Upsample the prompt here rather than through pipe(caption_upsample_temperature=...): pipe() passes the images
    # as a flat list, which upsample_prompt() turns into one sample per image and then fails (2 images != 1 prompt).
    # [[img1, img2]] is one sample with both references, shown side by side to Mistral 3.
    if upsample:
        set_deterministic(args.seed)
        prompt = pipe.upsample_prompt([prompt], images=[ref_images], temperature=args.upsample_temperature,
                                      device="cuda")[0]

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    image = pipe(prompt=prompt, image=ref_images, width=out_width, height=out_height,
                 num_inference_steps=args.steps, guidance_scale=args.guidance, generator=generator).images[0]
    if mask_composite:
        image = composite_onto_reference(image, ref_images[0])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()
