"""
Paste the textured garment of image B onto the grey garment of image A (no generation): B's own foreground is cropped,
resized to the bounding box of A's garment mask and pasted there; everything else in A (background, mannequin, any
part of the box B's silhouette does not cover) is left unchanged.

Inputs:
  --mannequin : image A, the mannequin wearing the untextured grey garment (RGB)
  --mask      : binary mask of A's garment (white = garment)
  --reference : image B, the textured garment render whose outline matches A's garment (RGBA, or RGB on a solid
                background)

Usage:
    python -m omnifabric.utils.composite_garment_texture --mannequin A.png --mask A_mask.png --reference B.png -o out.png
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image


def crop_foreground(rgba, alpha_thresh=128, bg_dist_thresh=24):
    """Tightly crop `rgba` to its real foreground content and return (crop, fg_mask).

    Prefers the real alpha channel. Falls back to distance-from-corner-color
    thresholding if the image turns out to have no real transparency (alpha
    uniformly opaque) — e.g. a textured render supplied as flat RGB instead of
    RGBA with a transparent background.
    """
    alpha = rgba[..., 3]
    if alpha.min() >= alpha_thresh:
        rgb = rgba[..., :3].astype(np.int16)
        bg_color = rgb[0, 0]
        dist = np.linalg.norm(rgb - bg_color, axis=-1)
        fg = dist > bg_dist_thresh
    else:
        fg = alpha > alpha_thresh

    ys, xs = np.where(fg)
    if len(xs) == 0:
        raise RuntimeError("Reference apparel image has no distinguishable foreground content.")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return rgba[y0:y1, x0:x1], fg[y0:y1, x0:x1]


def composite(mannequin_rgb, apparel_mask, reference_rgba):
    """Resize+paste `reference_rgba`'s own foreground onto `mannequin_rgb` at the
    bounding box of `apparel_mask`. Returns the composited RGB array."""
    ys, xs = np.where(apparel_mask)
    if len(xs) == 0:
        raise RuntimeError("Mask has no foreground (apparel) pixels.")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    bh, bw = y1 - y0, x1 - x0

    crop_rgba, crop_fg = crop_foreground(reference_rgba)
    resized_rgb = cv2.resize(crop_rgba[..., :3], (bw, bh), interpolation=cv2.INTER_LANCZOS4)
    resized_fg = cv2.resize(crop_fg.astype(np.uint8) * 255, (bw, bh), interpolation=cv2.INTER_NEAREST) > 0

    result = mannequin_rgb.copy()
    region = result[y0:y1, x0:x1]
    region[resized_fg] = resized_rgb[resized_fg]
    result[y0:y1, x0:x1] = region
    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mannequin", required=True, help="Image A: mannequin wearing the untextured grey apparel.")
    p.add_argument("--mask", required=True, help="Binary mask of A's apparel region.")
    p.add_argument("--reference", required=True, help="Image B: textured apparel render (same outline as A's apparel).")
    p.add_argument("-o", "--output", required=True, help="Output path for the composite.")
    return p.parse_args()


def main():
    args = parse_args()

    mannequin_rgb = np.array(Image.open(args.mannequin).convert("RGB"))
    apparel_mask = np.array(Image.open(args.mask).convert("L")) > 127
    reference_rgba = np.array(Image.open(args.reference).convert("RGBA"))

    if apparel_mask.shape != mannequin_rgb.shape[:2]:
        raise RuntimeError(
            f"Mask shape {apparel_mask.shape} doesn't match mannequin image {mannequin_rgb.shape[:2]}"
        )

    result_rgb = composite(mannequin_rgb, apparel_mask, reference_rgba)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    Image.fromarray(result_rgb).save(args.output)


if __name__ == "__main__":
    main()
