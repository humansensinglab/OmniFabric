#!/usr/bin/env python3
"""
UV Panel Dilation with Boundary Healing
========================================
Two-pass process for seam-free UV texture padding:

  Pass 1 — Boundary healing (inward):
    The 1-2 pixel ring at each panel edge is typically washed out — either
    semi-transparent (RGBA) or white-blended (RGB, from soft diffusion masks or
    anti-aliasing). These washed-out colors would propagate into the gap zone and
    produce faint/desaturated seams.
    Fix: replace those near-edge pixels with the color of their nearest
    "safe interior" pixel (far enough from the edge to be solid), making the
    entire panel boundary solid before dilation.

  Pass 2 — Gap dilation (outward):
    Expand each panel's solid colors into the inter-island white gap zone using
    INPAINT_TELEA, which continues local gradient/pattern structure coherently.

Usage:
    python -m omnifabric.normalize.dilate_uv_panels texture.png editable_template.png --N 10 --heal_radius 1 \
        --output normalized_texture.png [--save_diagnostics] [--no_heal]

Inputs:
    textured_template  : Projected / denoised textured UV map (RGB or RGBA PNG)
    reference_mask     : Original black panel template  (black=panel, white=background)
                         Defines true panel boundaries, independent of texture content.
Output:
    Textured UV map where:
      • Panel interiors       — unchanged
      • Panel boundary ring   — healed to solid (no more washed-out edges)
      • Inter-island gap zone — filled with pattern-coherent color (TELEA)
      • True background       — unchanged (white)
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def load_image(path: str):
    """
    Load image and return (rgb_uint8, alpha_uint8_or_None).
    If the file has an alpha channel it is returned separately; the RGB array
    is NOT pre-composited — the alpha is kept raw for the healing step.
    """
    img = Image.open(path)
    if img.mode == 'RGBA':
        arr = np.array(img)
        return arr[:, :, :3].copy(), arr[:, :, 3].copy()
    else:
        return np.array(img.convert('RGB')), None


def load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path)
    return np.array(img.convert('RGB'))

# ---------------------------------------------------------------------------
# Pre-step — Background masking
# ---------------------------------------------------------------------------


def premask_texture(
    rgb: np.ndarray,
    alpha,                  # np.ndarray or None
    panel_mask: np.ndarray,
) -> tuple:
    """
    Whiten all pixels that lie outside the UV panel regions.

    Noisy colors in the background (outside the sewing pattern islands) would
    otherwise be picked up by the outward dilation step and bleed into the gap
    zone, corrupting the seam fill.  Forcing the background to pure white before
    the heal+expand passes guarantees that only genuine panel colors are
    propagated.

    For RGBA images the alpha of background pixels is also set to 0 so that the
    background is unambiguously transparent.

    Returns (rgb_out, alpha_out_or_None).
    """
    bg = ~panel_mask
    out_rgb = rgb.copy()
    out_rgb[bg] = 255

    out_alpha = None
    if alpha is not None:
        out_alpha = alpha.copy()
        out_alpha[bg] = 0
    return out_rgb, out_alpha

# ---------------------------------------------------------------------------
# Pass 1 — Boundary healing
# ---------------------------------------------------------------------------


def heal_rgba_boundary(
    rgb: np.ndarray,
    alpha: np.ndarray,
    panel_mask: np.ndarray,
) -> tuple:
    """
    Recover the true (pre-composited) color of semi-transparent boundary pixels.

    When a PNG is rendered with anti-aliasing against a white background:
        C_composite = C_true * (A/255) + 255 * (1 - A/255)
    Inverting:
        C_true = (C_composite - 255) * (255 / A) + 255

    After healing, all panel pixels with A > 0 are set to A = 255.
    Fully transparent pixels (A == 0) inside the panel are left for the
    nearest-interior fallback in the RGB healing step.
    """
    semi = panel_mask & (alpha > 0) & (alpha < 255)
    if not np.any(semi):
        return rgb, alpha

    healed = rgb.astype(np.float32)
    A = alpha[semi].astype(np.float32)          # shape (n,)

    for c in range(3):
        C = rgb[semi, c].astype(np.float32)
        # C = C_true * (A/255) + 255 * (1 - A/255)
        # => C_true = (C - 255) * (255/A) + 255
        C_true = (C - 255.0) * (255.0 / A) + 255.0
        healed[semi, c] = np.clip(C_true, 0, 255)

    healed_alpha = alpha.copy()
    healed_alpha[semi] = 255

    return healed.astype(np.uint8), healed_alpha


def heal_rgb_boundary(
    rgb: np.ndarray,
    panel_mask: np.ndarray,
    heal_radius: int,
    safe_interior_radius: int,
) -> np.ndarray:
    """
    Replace washed-out near-edge panel pixels with the color of their nearest
    "safe interior" pixel (a panel pixel that is far enough from the edge to
    be guaranteed solid).

    Args:
        heal_radius          : pixels from edge to heal  (e.g. 3)
        safe_interior_radius : min distance from edge to be "safe interior"
                               (must be > heal_radius; default heal_radius+3)
    """
    # Distance of each panel pixel from the nearest panel boundary
    dist_inward = distance_transform_edt(panel_mask)   # 0 at edge, increases inward

    # "Safe interior": panel pixels sufficiently far from the edge
    interior_mask = panel_mask & (dist_inward > safe_interior_radius)

    # Thin boundary ring: panel pixels within heal_radius of edge
    thin_boundary = panel_mask & (dist_inward <= heal_radius)

    # nothing to heal, or no panel pixel far enough from the edge to copy from
    if not np.any(thin_boundary) or not np.any(interior_mask):
        return rgb

    # For every non-interior pixel, find the nearest safe-interior pixel
    non_interior = ~interior_mask
    _, (ny, nx) = distance_transform_edt(non_interior, return_indices=True)

    healed = rgb.copy()
    healed[thin_boundary] = rgb[ny[thin_boundary], nx[thin_boundary]]

    return healed


def heal_boundary(
    rgb: np.ndarray,
    alpha,                  # np.ndarray or None
    panel_mask: np.ndarray,
    heal_radius: int,
    safe_interior_radius: int,
) -> tuple:
    """
    Run the appropriate healing strategy depending on whether the image has alpha.

    For RGBA: alpha-uncomposite first, then fall through to RGB heal for any
              fully-transparent (A==0) pixels that may still be washed out.
    For RGB:  nearest-safe-interior-neighbor heal only.

    Returns (healed_rgb, healed_alpha_or_None).
    """
    if alpha is not None:
        rgb, alpha = heal_rgba_boundary(rgb, alpha, panel_mask)
        # After alpha-uncompositing, A==0 pixels inside the panel still have
        # undefined / white-blended RGB.  Treat them via the RGB path too.
        zero_alpha_in_panel = panel_mask & (alpha == 0)
        if np.any(zero_alpha_in_panel):
            # Temporarily mark those pixels as "thin boundary" so the RGB
            # healer replaces them with nearest interior colors.
            extended_panel = panel_mask & (alpha > 0)
            rgb = heal_rgb_boundary(rgb, extended_panel, heal_radius, safe_interior_radius)
    else:
        rgb = heal_rgb_boundary(rgb, panel_mask, heal_radius, safe_interior_radius)

    return rgb, alpha

# ---------------------------------------------------------------------------
# Pass 2 — Gap dilation
# ---------------------------------------------------------------------------


def build_gap_mask(
    textured: np.ndarray,
    panel_mask: np.ndarray,
    N: int,
    white_threshold: int,
) -> np.ndarray:
    """
    Return uint8 mask (255 = inpaint, 0 = keep) for the inter-island gap zone.

    Gap pixels: outside every UV island, within N px of the nearest island,
    AND currently white/unfilled in the texture.
    """
    non_panel = ~panel_mask
    dist = distance_transform_edt(non_panel)
    within_n   = non_panel & (dist <= N)
    is_unfilled = np.all(textured >= white_threshold, axis=2)
    return (within_n & is_unfilled).astype(np.uint8) * 255


def dilate_outward(
    textured: np.ndarray,
    panel_mask: np.ndarray,
    N: int,
    white_threshold: int,
    method: str,
) -> tuple:
    """
    Dilate each UV panel outward by N pixels using INPAINT_TELEA (or NS).

    Returns (result_rgb, gap_mask_uint8).
    """
    gap_mask = build_gap_mask(textured, panel_mask, N, white_threshold)
    if not np.any(gap_mask):
        return textured.copy(), gap_mask

    flag = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
    result = cv2.inpaint(textured, gap_mask, inpaintRadius=N, flags=flag)
    return result, gap_mask

# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    textured: np.ndarray,
    alpha,
    reference: np.ndarray,
    N: int,
    heal_radius: int,
    safe_interior_radius: int,
    white_threshold: int,
    panel_threshold: int,
    method: str,
    do_heal: bool,
    do_premask: bool,
) -> tuple:
    """
    Full pipeline: optional pre-masking, then two-pass heal + expand.

    Returns (result_rgb, healed_rgb, gap_mask, alpha, premasked_rgb_or_None).
      premasked_rgb_or_None — the background-cleared texture fed into Pass 1,
                              or None when do_premask=False.
    """
    if textured.shape[:2] != reference.shape[:2]:
        raise ValueError(
            f"Size mismatch: texture {textured.shape[:2]} vs "
            f"reference {reference.shape[:2]}"
        )

    panel_mask = np.all(reference <= panel_threshold, axis=2)

    # Pre-step: whiten background to remove noisy pixels outside panels
    premasked = None
    if do_premask:
        textured, alpha = premask_texture(textured, alpha, panel_mask)
        premasked = textured.copy()

    # Pass 1: boundary healing
    if do_heal:
        healed, alpha = heal_boundary(
            textured, alpha, panel_mask, heal_radius, safe_interior_radius
        )
    else:
        healed = textured.copy()

    # Pass 2: outward dilation
    result, gap_mask = dilate_outward(healed, panel_mask, N, white_threshold, method)

    return result, healed, gap_mask, alpha, premasked

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def make_output_path(input_path: str, suffix: str = '_dilated') -> str:
    base, ext = os.path.splitext(input_path)
    return base + suffix + (ext if ext else '.png')


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Two-pass UV panel dilation: '
            '(1) heal washed-out boundary pixels, '
            '(2) TELEA-inpaint outward into the inter-island gap.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        'textured_template',
        help='Projected / denoised textured UV map to process (RGB or RGBA PNG)',
    )
    parser.add_argument(
        'reference_mask',
        help='Original black panel template  (black = panel, white = background)',
    )
    parser.add_argument(
        '--N', type=int, default=16,
        help='Dilation radius into the gap zone, in pixels  (default: 16)',
    )
    parser.add_argument(
        '--output', default=None,
        help='Output path  (default: input + _dilated suffix)',
    )
    parser.add_argument(
        '--method', choices=['telea', 'ns'], default='telea',
        help='Inpainting method for gap dilation: telea (default) or ns',
    )
    parser.add_argument(
        '--heal_radius', type=int, default=3,
        help='How many pixels from the panel edge to heal  (default: 3)',
    )
    parser.add_argument(
        '--safe_interior_radius', type=int, default=None,
        help='Min distance from edge to be "safe interior"  '
             '(default: heal_radius + 3)',
    )
    parser.add_argument(
        '--no_heal', action='store_true',
        help='Skip boundary healing (Pass 1); only do gap dilation',
    )
    parser.add_argument(
        '--no_premask', action='store_true',
        help='Skip background masking (do not whiten pixels outside the panel '
             'mask before heal+expand). Pre-masking is ON by default.',
    )
    parser.add_argument(
        '--white_threshold', type=int, default=250,
        help='Channels >= this are "unfilled white"  (default: 250)',
    )
    parser.add_argument(
        '--panel_threshold', type=int, default=10,
        help='Channels <= this are "panel" in reference mask  (default: 10)',
    )
    parser.add_argument(
        '--save_diagnostics', action='store_true',
        help='Save healed texture and gap mask as extra PNGs for inspection',
    )

    args = parser.parse_args()
    output_path = args.output or make_output_path(args.textured_template)
    safe_r = args.safe_interior_radius or (args.heal_radius + 3)

    textured, alpha = load_image(args.textured_template)
    reference       = load_image_rgb(args.reference_mask)

    result, healed, gap_mask, alpha_out, premasked = run_pipeline(
        textured, alpha, reference,
        N=args.N,
        heal_radius=args.heal_radius,
        safe_interior_radius=safe_r,
        white_threshold=args.white_threshold,
        panel_threshold=args.panel_threshold,
        method=args.method,
        do_heal=not args.no_heal,
        do_premask=not args.no_premask,
    )

    # Save main output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if alpha_out is not None:
        # Restore alpha channel (all healed panel pixels now have A=255)
        out_rgba = np.dstack([result, np.full(result.shape[:2], 255, dtype=np.uint8)])
        Image.fromarray(out_rgba, 'RGBA').save(output_path)
    else:
        Image.fromarray(result).save(output_path)

    # Optional diagnostics
    if args.save_diagnostics:
        if premasked is not None:
            premasked_path = make_output_path(output_path, '_premasked')
            Image.fromarray(premasked).save(premasked_path)
        healed_path = make_output_path(output_path, '_healed_boundary')
        gap_path    = make_output_path(output_path, '_gap_mask')
        Image.fromarray(healed).save(healed_path)
        Image.fromarray(gap_mask).save(gap_path)


if __name__ == '__main__':
    main()
