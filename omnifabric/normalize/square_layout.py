"""
Rearrange an elongated sewing-pattern layout for texture normalization, and back.

OmniFabric works on a square canvas (--target_size 1024): the layout is resized so its longer side fits and the rest
is padded. A sewing pattern whose layout is much wider than tall (or taller than wide) then fills only a strip of the
canvas, so much of the coarse texture's detail is lost before normalization. Two rearrangements cut the layout along
its long side where no panel is cut (in the middle of a run of fully empty columns, or rows for a tall layout):

  stack  (ratio above --max_ratio, e.g. a 3819 x 913 mini dress): the second part is stacked below (right of, for a
         tall layout) the first, padding the narrower part with empty pixels, and normalized in one run.
  split  (ratio above --min_split_ratio, e.g. a 2344 x 901 tank top): the two parts are normalized in two separate
         runs, each filling the canvas better, and put side by side again afterwards. Less elongated layouts are kept,
         as the resolution gain would be small for twice the normalization time.

Of all possible cuts, the one closest to square (for split: the more elongated part closest to square) is used; if
none is squarer than the original, the layout is kept as is.

to_square writes one folder per normalization run (<out_dir>/part0/, and part1/ for split), each with the template,
the coarse texture and the position map (.pt) rearranged the same way, plus <out_dir>/layout.json. from_square takes
the normalized textures of the runs, in order, and restores the original layout, so the result matches the original
template and sewing pattern again.

Usage:
    python -m omnifabric.normalize.square_layout to_square --template editable_template.png \\
        --coarse coarse_texture.png --pos garment_position_map.pt --out_dir normalize/inputs [--max_ratio 3]
    python -m omnifabric.normalize.square_layout from_square normalize/part0/normalized_fullres.png \\
        [normalize/part1/normalized_fullres.png] --layout normalize/inputs/layout.json --output restored.png
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np
import torch
from PIL import Image


def pattern_mask(template_path):
    """Sewing-pattern panels of the binary template: black = panel, white = empty."""
    return np.array(Image.open(template_path).convert('L')) < 128


def aspect(rows, cols):
    return max(rows, cols) / min(rows, cols)


def plan_layout(mask, max_ratio=3.0, min_split_ratio=2.0, min_separation=16, allow_split=True):
    """Choose how to rearrange the layout (see the module docstring).

    The layout is handled as wide (cut along columns); a tall one is transposed first ('transposed'). When stacking,
    the two parts are kept at least min_separation empty pixels apart ('gap' adds rows if needed). 'parts' lists the
    (height, width) of the image(s) normalization runs on.
    """
    h, w = mask.shape
    ratio = aspect(h, w)
    layout = {'height': h, 'width': w, 'ratio': ratio, 'mode': 'keep', 'parts': [[h, w]]}

    transposed = h > w
    m = mask.T if transposed else mask
    rows, cols = m.shape
    empty = ~m.any(axis=0)
    edges = np.flatnonzero(np.diff(np.r_[0, empty.astype(np.int8), 0]))
    # middle of each run of empty columns between panels (the layout's own margins have nothing to cut apart)
    cuts = [int(start + end) // 2 for start, end in zip(edges[::2], edges[1::2]) if start > 0 and end < cols]

    def oriented(r, c):  # (height, width) in the original orientation
        return [int(c), int(r)] if transposed else [int(r), int(c)]

    if ratio > max_ratio:
        best = None
        for cut in cuts:
            first_rows = np.flatnonzero(m[:, :cut].any(axis=1))
            second_rows = np.flatnonzero(m[:, cut:].any(axis=1))
            gap = max(0, min_separation - ((rows - 1 - first_rows[-1]) + second_rows[0]))
            stacked = (2 * rows + gap, max(cut, cols - cut))
            if best is None or aspect(*stacked) < best[0]:
                best = (aspect(*stacked), cut, gap, stacked)
        if best is not None and best[0] < ratio:
            layout.update(mode='stack', transposed=bool(transposed), cut=best[1], gap=int(best[2]),
                          ratio_after=float(best[0]), parts=[oriented(*best[3])])
            return layout

    if allow_split and ratio > min_split_ratio:
        best = None
        for cut in cuts:
            worst = max(aspect(rows, cut), aspect(rows, cols - cut))
            if best is None or worst < best[0]:
                best = (worst, cut)
        if best is not None and best[0] < ratio:
            cut = best[1]
            layout.update(mode='split', transposed=bool(transposed), cut=cut, ratio_after=float(best[0]),
                          parts=[oriented(rows, cut), oriented(rows, cols - cut)])
    return layout


def to_parts(arr, layout, fill):
    """Rearrange an (H, W, ...) array from the original layout into the image(s) normalization runs on; pixels added
    by stacking get fill."""
    if layout['mode'] == 'keep':
        return [arr]
    a = arr.swapaxes(0, 1) if layout['transposed'] else arr
    rows, cols = a.shape[:2]
    cut = layout['cut']
    if layout['mode'] == 'split':
        parts = [a[:, :cut], a[:, cut:]]
    else:
        gap = layout['gap']
        out = np.empty((2 * rows + gap, max(cut, cols - cut)) + a.shape[2:], a.dtype)
        out[...] = fill
        out[:rows, :cut] = a[:, :cut]
        out[rows + gap:, :cols - cut] = a[:, cut:]
        parts = [out]
    return [np.ascontiguousarray(p.swapaxes(0, 1) if layout['transposed'] else p) for p in parts]


def from_parts(parts, layout):
    """Inverse of to_parts."""
    if layout['mode'] == 'keep':
        return parts[0]
    a = [p.swapaxes(0, 1) for p in parts] if layout['transposed'] else parts
    rows = layout['width'] if layout['transposed'] else layout['height']
    cols = layout['height'] if layout['transposed'] else layout['width']
    cut = layout['cut']
    if layout['mode'] == 'split':
        out = np.concatenate(a, axis=1)
    else:
        gap = layout['gap']
        out = np.concatenate([a[0][:rows, :cut], a[0][rows + gap:2 * rows + gap, :cols - cut]], axis=1)
    return out.swapaxes(0, 1) if layout['transposed'] else out


def image_fill(mode):
    """Empty pixels: white, as in the template (and transparent for RGBA), matching infer.resize_and_pad."""
    return {'L': 255, 'RGB': (255, 255, 255), 'RGBA': (255, 255, 255, 0)}.get(mode, 255)


def rearrange_image(src, dsts, layout):
    img = Image.open(src)
    if img.mode not in ('L', 'RGB', 'RGBA'):
        img = img.convert('RGB')
    if (img.height, img.width) != (layout['height'], layout['width']):
        raise ValueError(f"{src} is {img.width}x{img.height}, the template is {layout['width']}x{layout['height']}")
    for dst, part in zip(dsts, to_parts(np.array(img), layout, image_fill(img.mode))):
        Image.fromarray(part, img.mode).save(dst)


def rearrange_position_map(src, dsts, layout):
    """Position map: a (H, W, 3) tensor, or a dict of them (position_map, validity_mask) with metadata. Pixels added
    by stacking are position 0 (what infer.load_position_tensor treats as empty) and validity False."""
    def convert(t):
        if not torch.is_tensor(t) or tuple(t.shape[:2]) != (layout['height'], layout['width']):
            return [t] * len(dsts)
        fill = False if t.dtype == torch.bool else 0
        return [torch.from_numpy(p) for p in to_parts(t.numpy(), layout, fill)]

    data = torch.load(src, map_location='cpu', weights_only=False)
    if isinstance(data, dict):
        converted = {k: convert(v) for k, v in data.items()}
        parts = [{k: v[i] for k, v in converted.items()} for i in range(len(dsts))]
        for part in parts:
            if 'shape' in part and torch.is_tensor(part.get('position_map')):
                part['shape'] = part['position_map'].shape
    else:
        parts = convert(data)
    for dst, part in zip(dsts, parts):
        torch.save(part, dst)


def cmd_to_square(args):
    os.makedirs(args.out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(args.out_dir, 'part*')):  # from an earlier run, possibly another mode
        shutil.rmtree(old)
    layout = plan_layout(pattern_mask(args.template), args.max_ratio, args.min_split_ratio,
                         allow_split=not args.no_split)
    part_dirs = [os.path.join(args.out_dir, f'part{i}') for i in range(len(layout['parts']))]
    for d in part_dirs:
        os.makedirs(d)
    names = {'template': 'editable_template.png', 'coarse': 'coarse_texture.png', 'pos': 'position_map.pt'}
    dsts = {key: [os.path.join(d, name) for d in part_dirs] for key, name in names.items()}

    if layout['mode'] == 'keep':
        for key in names:
            shutil.copy(getattr(args, key), dsts[key][0])
    else:
        rearrange_image(args.template, dsts['template'], layout)
        rearrange_image(args.coarse, dsts['coarse'], layout)
        rearrange_position_map(args.pos, dsts['pos'], layout)

    if layout['mode'] != 'keep':
        size = f"{layout['width']}x{layout['height']}"
        sizes = " + ".join(f"{w}x{h}" for h, w in layout['parts'])
        how = {'stack': 'stacked into a squarer layout', 'split': 'split in two parts normalized separately'}
        print(f"Elongated sewing-pattern layout ({size}, ratio {layout['ratio']:.2f}): {how[layout['mode']]} "
              f"({sizes})")

    with open(os.path.join(args.out_dir, 'layout.json'), 'w') as f:
        json.dump(layout, f, indent=2)


def cmd_from_square(args):
    with open(args.layout) as f:
        layout = json.load(f)
    if len(args.images) != len(layout['parts']):
        raise ValueError(f"Expected {len(layout['parts'])} image(s) ({layout['mode']} layout), got {len(args.images)}")
    parts = []
    for path, (h, w) in zip(args.images, layout['parts']):
        img = Image.open(path)
        if img.size != (w, h):
            raise ValueError(f"{path} is {img.width}x{img.height}, expected {w}x{h}")
        parts.append(img)
    mode = parts[0].mode
    out = from_parts([np.array(p.convert(mode)) for p in parts], layout)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    Image.fromarray(out, mode).save(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('to_square', help='rearrange the normalization inputs (one folder per normalization run)')
    p.add_argument('--template', required=True, help='binary sewing-pattern template (editable_template.png)')
    p.add_argument('--coarse', required=True, help='coarse texture map in the template layout')
    p.add_argument('--pos', required=True, help='position map (.pt) in the template layout')
    p.add_argument('--out_dir', required=True, help='writes part0/[, part1/] (editable_template.png, '
                                                    'coarse_texture.png, position_map.pt) and layout.json here')
    p.add_argument('--max_ratio', type=float, default=3.0, help='stack layouts more elongated than this')
    p.add_argument('--min_split_ratio', type=float, default=2.0,
                   help='split layouts more elongated than this (and not stacked) into two normalization runs')
    p.add_argument('--no_split', action='store_true', help='never split into two normalization runs')
    p.set_defaults(func=cmd_to_square)

    p = sub.add_parser('from_square', help='restore the original layout from the normalized part(s)')
    p.add_argument('images', nargs='+', help='normalized texture of each part, in order (part0 [part1])')
    p.add_argument('--layout', required=True, help='layout.json written by to_square')
    p.add_argument('--output', required=True)
    p.set_defaults(func=cmd_from_square)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
