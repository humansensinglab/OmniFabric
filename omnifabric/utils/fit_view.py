"""Place a generated garment view (garment on a green screen) onto a garment-only render's silhouette.

Unlike remove_background.py, which assumes the generated image is pixel-aligned with the render
(true for the FLUX.2 front view, which edits the render itself), views taken from the 360-degree
orbit video only roughly match the mesh silhouette, especially the side views. The orbit video
keeps the framing of its first frame (the front view, aligned with the front render), so the view
is resized to the render resolution without stretching and centered on the render's silhouette.
The render's alpha becomes the output alpha, and silhouette pixels the generated garment does not
cover take the nearest generated garment color, so the green background is never projected onto
the mesh. With non-overlapping projection, side views only texture what front/back views missed.

--aligned skips resizing/centering, for views that are already pixel-aligned with the render
(the texture-transferred front view); it then only removes green-screen pixels inside the silhouette.

Usage:
    python -m omnifabric.utils.fit_view <render_rgba.png> <generated_view.png> <output_rgba.png> [--aligned]
"""

import sys

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


def green_screen_mask(rgb, margin=30, strong_margin=70, fringe=6, erode=4):
    """Garment foreground mask: largest region that is not green screen.

    Green screen is where green clearly dominates red and blue. The green screens of the renders, the orbit views and
    Gemini all have G - max(R, B) >= ~90, while greenish fabric (e.g. a yellow-green check) stays well below 70, so
    only strongly green pixels (> strong_margin), plus weakly green ones (> margin: green spill, anti-aliased edges)
    within `fringe` px of them, are background. The mask is eroded so green-tinted pixels along the garment's edge
    are not used.
    """
    rgb = rgb.astype(np.int16)
    dominance = rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2])
    strong = (dominance > strong_margin).astype(np.uint8)
    near = cv2.dilate(strong, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * fringe + 1, 2 * fringe + 1))) > 0
    fg = ~((strong > 0) | ((dominance > margin) & near))
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num <= 1:
        raise RuntimeError("Generated view has no foreground garment.")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fg = (labels == largest).astype(np.uint8)
    return cv2.erode(fg, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)) > 0


def bbox_center(mask):
    ys, xs = np.where(mask)
    return (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2


def fit_view_to_silhouette(render_rgba, view_rgb, aligned=False, alpha_thresh=127):
    """Return an RGBA image with the render's silhouette, textured by the (centered) generated view."""
    silhouette = render_rgba[..., 3] > alpha_thresh
    if not silhouette.any():
        raise RuntimeError("Render has an empty silhouette.")
    h, w = silhouette.shape

    # same framing as the render: uniform resize to the render resolution
    view_rgb = cv2.resize(view_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
    fg = green_screen_mask(view_rgb)

    if aligned:
        rgb, covered = view_rgb, fg
    else:
        # center the generated garment on the silhouette
        (vy, vx), (sy, sx) = bbox_center(fg), bbox_center(silhouette)
        shift = np.float32([[1, 0, sx - vx], [0, 1, sy - vy]])
        rgb = cv2.warpAffine(view_rgb, shift, (w, h), flags=cv2.INTER_LINEAR)
        covered = cv2.warpAffine(fg.astype(np.uint8), shift, (w, h), flags=cv2.INTER_NEAREST) > 0

    if (silhouette & ~covered).any():
        # copy the nearest generated garment pixel into silhouette pixels it does not cover
        _, (iy, ix) = distance_transform_edt(~covered, return_indices=True)
        rgb = rgb[iy, ix]

    return np.dstack([rgb, np.where(silhouette, render_rgba[..., 3], 0).astype(np.uint8)])


def match_scale(fg, silhouette, quarter=4):
    """Uniform scale + translation (2x3 affine) that best overlays the garment mask fg on the silhouette.

    Starts from the bounding boxes (centers, geometric-mean size ratio), then searches scale and offset for the
    highest IoU on 1/4-resolution masks. Returns the affine and the IoU before (no transform) and after.
    """
    def iou(a, b):
        return (a & b).sum() / max((a | b).sum(), 1)

    def warp(mask, m, size):
        return cv2.warpAffine(mask.astype(np.uint8), m, size, flags=cv2.INTER_NEAREST) > 0

    def affine(s, dx=0.0, dy=0.0):
        return np.float32([[s, 0, sx - s * vx + dx], [0, s, sy - s * vy + dy]])

    ys, xs = np.where(fg)
    (vy, vx), (sy, sx) = bbox_center(fg), bbox_center(silhouette)
    sys_, sxs = np.where(silhouette)
    s0 = np.sqrt((np.ptp(sys_) + 1) / (np.ptp(ys) + 1) * (np.ptp(sxs) + 1) / (np.ptp(xs) + 1))

    h, w = silhouette.shape
    small_size = (w // quarter, h // quarter)
    fg_small = cv2.resize(fg.astype(np.uint8), small_size, interpolation=cv2.INTER_AREA) > 0
    sil_small = cv2.resize(silhouette.astype(np.uint8), small_size, interpolation=cv2.INTER_AREA) > 0

    def score(m):
        m = m.copy()
        m[:, 2] /= quarter  # same transform in 1/4-resolution coordinates
        return iou(warp(fg_small, m, small_size), sil_small)

    best_s = max(s0 * np.linspace(0.85, 1.15, 61), key=lambda s: score(affine(s)))
    offsets = np.arange(-6, 7) * quarter  # +-24 px
    best_dx, best_dy = max(((dx, dy) for dx in offsets for dy in offsets),
                           key=lambda d: score(affine(best_s, *d)))

    # refine at full resolution: 1-px offsets, then 0.1% scale steps
    full = lambda m: iou(warp(fg, m, (w, h)), silhouette)
    best_dx, best_dy = max(((best_dx + dx, best_dy + dy) for dx in range(-3, 4) for dy in range(-3, 4)),
                           key=lambda d: full(affine(best_s, *d)))
    best_s = max(best_s * np.linspace(0.99, 1.01, 21), key=lambda s: full(affine(s, best_dx, best_dy)))
    m = affine(best_s, best_dx, best_dy)
    return m, iou(fg, silhouette), full(m)


def fit_scaled_view(render_rgba, view_rgb, alpha_thresh=127):
    """Like fit_view_to_silhouette, for a frontal view whose garment has the render's shape but not its scale/position
    (e.g. Gemini often draws the garment larger): the generated garment is chroma-keyed, scaled and shifted onto the
    silhouette (match_scale), and resampled once. Returns the RGBA result, the scale and the IoU before/after."""
    silhouette = render_rgba[..., 3] > alpha_thresh
    h, w = silhouette.shape

    # pad to the render's aspect ratio with the background color (no stretching), then resize to its resolution
    vh, vw = view_rgb.shape[:2]
    ph, pw = max(vh, round(vw * h / w)), max(vw, round(vh * w / h))
    background = np.median(np.concatenate([view_rgb[0], view_rgb[-1], view_rgb[:, 0], view_rgb[:, -1]]), axis=0)
    padded = np.empty((ph, pw, 3), np.uint8)
    padded[:] = background
    top, left = (ph - vh) // 2, (pw - vw) // 2
    padded[top:top + vh, left:left + vw] = view_rgb
    view_rgb = cv2.resize(padded, (w, h), interpolation=cv2.INTER_LANCZOS4)

    m, iou_before, iou_after = match_scale(green_screen_mask(view_rgb, erode=0), silhouette)
    rgb = cv2.warpAffine(view_rgb, m, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    covered = cv2.warpAffine(green_screen_mask(view_rgb).astype(np.uint8), m, (w, h), flags=cv2.INTER_NEAREST) > 0
    if (silhouette & ~covered).any():
        _, (iy, ix) = distance_transform_edt(~covered, return_indices=True)
        rgb = rgb[iy, ix]
    out = np.dstack([rgb, np.where(silhouette, render_rgba[..., 3], 0).astype(np.uint8)])
    return out, float(m[0, 0]), iou_before, iou_after


def main():
    args = [a for a in sys.argv[1:] if a != '--aligned']
    if len(args) != 3:
        print("Usage: python -m omnifabric.utils.fit_view <render_rgba.png> <generated_view.png> <output_rgba.png> [--aligned]")
        sys.exit(1)
    render_rgba = np.array(Image.open(args[0]).convert('RGBA'))
    view_rgb = np.array(Image.open(args[1]).convert('RGB'))
    out = fit_view_to_silhouette(render_rgba, view_rgb, aligned='--aligned' in sys.argv)
    Image.fromarray(out, 'RGBA').save(args[2])


if __name__ == "__main__":
    main()
