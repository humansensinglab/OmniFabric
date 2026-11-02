"""
FLUX.2-dev back-view generation (VIEWS_BACKEND=flux2): given the frontal view of a single garment on a chroma-key
background (no person or mannequin), render the same garment rotated 180 degrees, showing its back. Fabric, color,
pattern and construction stay consistent with the front; the output's silhouette is forced to the front view's own
(build_garment_silhouette / composite_silhouette), so the model only decides what is drawn inside that outline.

Usage:
    python -m omnifabric.coarse.back_view_flux2 --image garment_front.png -o back_view.png [--no-upsample]
"""

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers.pipelines.flux2.system_messages import SYSTEM_MESSAGE_UPSAMPLING_I2I

from omnifabric.coarse.texture_transfer import load_pipeline, set_deterministic

# Mistral 3's image-to-image rewrite states what changes and what stays the same; the prompt makes that explicit and
# keeps attention on the back's construction. Two failure modes it is worded against:
# 1. fabric-level attributes (base color, weave, allover patterns) continue onto the back of a real garment, but a
#    one-off placement graphic (chest logo, slogan, printed text) almost never does, so they are separate cases;
# 2. person/body wording made the model draw a human into the output, so the prompt only talks about an isolated
#    garment, and says so twice.
BACK_VIEW_PROMPT = (
    "This image shows a single garment only — no person, mannequin, or body parts "
    "are in the photo — laid out or shaped on an invisible form, isolated against "
    "a solid, uniform chroma-key background. "
    "Rotate the garment 180 degrees horizontally, as if you walked around it, so "
    "we now see its back side, the same way the front is currently framed. "
    "Transfer only the garment's fabric-level properties to the back: the same base "
    "color(s), material, and weave/knit texture, plus any allover pattern that repeats "
    "uniformly across the whole fabric (for example checks, tartan, stripes, "
    "houndstooth, polka dots, or a small motif tiled repeatedly as a continuous "
    "surface pattern). "
    "Do NOT transfer text, wordmarks, slogans, numbers, brand names, or any single "
    "one-off placement graphic (such as a chest logo or a large printed design) that "
    "appears only once on the front — real garments do not usually repeat that kind of "
    "print on the back. Render those areas on the back as the plain base fabric/color "
    "(or the continuing allover pattern, if any), with no text, letters, or logo. "
    "Extend the garment's construction logically to the back — matching side seams, "
    "hemline, and any collar/neckline, sleeve, waistband, or leg-opening construction, "
    "plus any closures such as zippers, buttons, or drawstrings, in a way that is "
    "consistent with how the front was constructed. "
    "Keep unchanged: the garment's base color palette, allover pattern scale, and "
    "overall silhouette, outline, proportions, fit, and length from the reference "
    "image — the back should occupy the exact same position, scale, and outline in "
    "the frame as the front does, just showing the opposite side. "
    "Keep unchanged: the background — the same solid, uniform chroma-key color, with "
    "no added shadow, floor, prop, or texture. "
    "The output must be the same kind of image as the input: the garment only, still "
    "isolated on the same plain background, with no person, mannequin, hand, arm, "
    "face, or any other body part visible anywhere in the image. "
    "The result should look like the same garment product photo, just rotated 180 "
    "degrees horizontally to reveal its back — with no text or logo carried over "
    "from the front unless it is part of a repeating allover fabric pattern."
)

# Added to the library's image-to-image rewrite instructions, which tell Mistral 3 to keep what it sees: without it,
# front-only text and logos were described and kept on the back
SYSTEM_MESSAGE_UPSAMPLING_I2I_BACK_VIEW = SYSTEM_MESSAGE_UPSAMPLING_I2I + """

Additional rules — this edit turns a garment product photo around to show its back:
- The input is a garment only: no person, mannequin, or body part is in frame. Do
  not describe or introduce one in the rewritten instruction, and explicitly state
  that none should appear in the output either.
- The output must keep the exact same silhouette, outline, position, and scale as
  the input — only the surface facing the camera changes from front to back.
- Classify anything printed/woven onto the garment as one of two kinds:
  (a) fabric-level / allover: base color, material, weave or knit texture, and any
      pattern that repeats uniformly across the whole surface (checks, tartan,
      stripes, houndstooth, polka dots, a small motif tiled as a continuous pattern).
  (b) one-off placement graphic: text, a wordmark, a slogan, a number, a brand name,
      or any logo/print that appears a single time in one spot (e.g. a chest logo).
- Instruct that kind (a) elements continue onto the back.
- Instruct that kind (b) elements do NOT appear on the back — the corresponding back
  area is the plain base fabric/color (or the continuing allover pattern, if any).
- Never quote or restate the specific text/wordmark/logo content seen on the front;
  do not ask for it to be reproduced anywhere in the output."""


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.2-dev back-view generation from a frontal garment view")
    parser.add_argument("--image", "-i", type=str, required=True,
                        help="frontal view of the garment (garment only, on a solid chroma-key background)")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="Override the default back-view prompt.")
    parser.add_argument("-o", "--output", type=str, default="flux2_back_view.png")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-upsample", action="store_true", help="Skip the Mistral 3 prompt rewrite.")
    parser.add_argument("--width", type=int, default=None,
                        help="Generation width (default: the reference's); the output keeps the reference's size.")
    parser.add_argument("--height", type=int, default=None, help="Generation height, see --width.")
    parser.add_argument("--no-mask-composite", action="store_true",
                        help="Save the raw generation instead of forcing the front view's silhouette.")
    parser.add_argument("--bg-thresh", type=float, default=40,
                        help="RGB distance from the background color above which a pixel is garment.")
    parser.add_argument("--min-area-frac", type=float, default=1e-3,
                        help="Smallest connected component kept in the silhouette (fraction of the image).")
    parser.add_argument("--open-ksize", type=int, default=3, help="Morphological opening kernel size.")
    parser.add_argument("--close-ksize", type=int, default=5, help="Morphological closing kernel size.")
    parser.add_argument("--erode-px", type=int, default=1,
                        help="Erode the silhouette by this many pixels (avoids a chroma-key fringe).")
    return parser.parse_args()


def upsample_prompt_back_view(pipe, prompt, images, temperature=0.5, device="cuda"):
    """Flux2Pipeline.upsample_prompt() with our system message (SYSTEM_MESSAGE_UPSAMPLING_I2I_BACK_VIEW) instead of
    the library's fixed one. Uses diffusers internals, imported lazily so that a diffusers update that renames them
    raises an error the caller handles (falling back to the stock method)."""
    from diffusers.pipelines.flux2.pipeline_flux2 import (
        _validate_and_process_images,
        format_input,
    )

    prompt = [prompt] if isinstance(prompt, str) else prompt
    device = pipe.text_encoder.device if device is None else device

    processed_images = _validate_and_process_images(
        images, pipe.image_processor, pipe.upsampling_max_image_size
    )
    messages_batch = format_input(
        prompts=prompt,
        system_message=SYSTEM_MESSAGE_UPSAMPLING_I2I_BACK_VIEW,
        images=processed_images,
    )

    inputs = pipe.tokenizer.apply_chat_template(
        messages_batch,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=2048,
    )
    inputs["input_ids"] = inputs["input_ids"].to(device)
    inputs["attention_mask"] = inputs["attention_mask"].to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(device, pipe.text_encoder.dtype)

    generated_ids = pipe.text_encoder.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=temperature,
        use_cache=True,
    )
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = generated_ids[:, input_length:]
    return pipe.tokenizer.tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )


def estimate_bg_color(rgb, patch=20):
    """Sample the four corners to get the actual chroma-key color, rather than
    assuming a fixed green — real captures vary a bit in exact hue/brightness."""
    h, w = rgb.shape[:2]
    patch = min(patch, h // 2, w // 2)
    corners = [
        rgb[:patch, :patch], rgb[:patch, w - patch:],
        rgb[h - patch:, :patch], rgb[h - patch:, w - patch:],
    ]
    return np.concatenate([c.reshape(-1, 3) for c in corners], axis=0).mean(axis=0)


def build_garment_silhouette(rgb, bg_color=None, bg_thresh=40, min_area_frac=1e-3,
                              open_ksize=3, close_ksize=5, erode_px=1):
    """
    Segment a flat/ghost-mannequin garment product photo (apparel only, no person,
    solid chroma-key background) into a foreground silhouette mask.

    Only removes small isolated noise blobs via connected components — it never
    fills contours solid, so a real hole in the garment (e.g. the collar cavity
    visible through the neckline on a ghost-mannequin shot, or the gap between
    pant legs) is correctly preserved as background rather than papered over.
    """
    h, w = rgb.shape[:2]
    if bg_color is None:
        bg_color = estimate_bg_color(rgb)

    dist = np.linalg.norm(rgb.astype(np.int16) - np.asarray(bg_color), axis=-1)
    fg = (dist > bg_thresh).astype(np.uint8) * 255

    opened = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    min_area = min_area_frac * h * w
    mask = np.zeros((h, w), bool)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            mask |= (labels == lbl)

    if erode_px > 0:
        mask = cv2.erode(
            mask.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)),
        ) > 0

    return mask, bg_color


def composite_silhouette(front_rgb, generated_rgb, mask):
    """
    Force the output's silhouette (outer outline and any interior holes) to
    exactly match the front reference's own garment mask: pixels inside the mask
    come from the generated back-view, everything else is pasted straight from
    the front photo's own background. This is what actually guarantees "same
    silhouette as the front" — we don't rely on the model drawing the outline in
    the right place — and it also wipes out any hallucinated human body part the
    model may have drawn outside the garment's own outline.
    """
    if generated_rgb.shape[:2] != mask.shape:
        raise ValueError(
            f"generated image {generated_rgb.shape[:2]} and mask {mask.shape} size mismatch"
        )
    final = front_rgb.copy()
    final[mask] = generated_rgb[mask]
    return final


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — this model requires a GPU.")
    # seed before any model loading, see set_deterministic()
    set_deterministic(args.seed)

    prompt = args.prompt or BACK_VIEW_PROMPT
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Reference image not found: {args.image}")
    ref_image = Image.open(args.image).convert("RGB")
    ref_rgb = np.array(ref_image)
    out_width = args.width or ref_image.width
    out_height = args.height or ref_image.height

    mask = None
    if not args.no_mask_composite:
        mask, _ = build_garment_silhouette(
            ref_rgb, bg_thresh=args.bg_thresh, min_area_frac=args.min_area_frac,
            open_ksize=args.open_ksize, close_ksize=args.close_ksize, erode_px=args.erode_px,
        )

    print("Generating the back view with FLUX.2-dev")
    pipe = load_pipeline()

    # let Mistral 3 describe the garment from the reference image ([[ref_image]]: one sample, one image)
    if not args.no_upsample:
        set_deterministic(args.seed)
        try:
            upsampled = upsample_prompt_back_view(pipe, prompt, images=[[ref_image]], temperature=0.5, device="cuda")
        except Exception as e:
            print(f"Warning: back-view prompt rewrite failed ({e!r}); using the stock rewrite, so front-only text or "
                  f"logos may appear on the back.")
            upsampled = pipe.upsample_prompt([prompt], images=[[ref_image]], temperature=0.5, device="cuda")
        prompt = upsampled[0]

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    image = pipe(prompt=prompt, image=[ref_image], num_inference_steps=args.steps, guidance_scale=args.guidance,
                 generator=generator, width=out_width, height=out_height).images[0]

    if mask is not None:
        if image.size != (ref_image.width, ref_image.height):
            image = image.resize((ref_image.width, ref_image.height), Image.LANCZOS)
        image = Image.fromarray(composite_silhouette(ref_rgb, np.array(image.convert("RGB")), mask))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()
