"""Replace the transparent background of an RGBA render with a chroma-key green (RGB output).

Usage:
    python -m omnifabric.utils.color_background <input_rgba.png> <output_rgb.png>
"""
import sys

import numpy as np
from PIL import Image


def convert_rgba_to_chromakey_bg(rgba_img_path, output_rgb_path, chromakey_color=(0, 177, 64)):
    """Fully transparent pixels become chromakey_color, all others keep their RGB."""
    rgba = np.array(Image.open(rgba_img_path).convert('RGBA'))
    rgb = rgba[..., :3].copy()
    rgb[rgba[..., 3] == 0] = chromakey_color
    Image.fromarray(rgb).save(output_rgb_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m omnifabric.utils.color_background <input_rgba_image> <output_rgb_image>")
        sys.exit(1)
    convert_rgba_to_chromakey_bg(sys.argv[1], sys.argv[2])
