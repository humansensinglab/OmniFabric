"""Apply the alpha channel of an RGBA image (A) to an RGB image (B, resized to A if needed).

Usage:
    python -m omnifabric.utils.remove_background <rgba_A.png> <rgb_B.png> <output_rgba.png>
"""
import sys

from PIL import Image


def apply_alpha_mask(rgba_img_path, rgb_img_path, output_path):
    """
    Given:
        rgba_img_path: path to image A (RGBA)
        rgb_img_path: path to image B (RGB)
        output_path: path to save RGBA version of B with A's alpha mask applied
    """
    img_a = Image.open(rgba_img_path).convert('RGBA')
    img_b = Image.open(rgb_img_path).convert('RGB')
    # Resize B to A if not same size
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.LANCZOS)

    # Extract alpha from A
    _, _, _, alpha = img_a.split()
    # Compose B with alpha
    img_b_rgba = img_b.copy()
    img_b_rgba.putalpha(alpha)
    img_b_rgba.save(output_path)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python -m omnifabric.utils.remove_background <rgba_img_A_path> <rgb_img_B_path> <output_path>")
        sys.exit(1)
    rgba_img_path = sys.argv[1]
    rgb_img_path = sys.argv[2]
    output_path = sys.argv[3]
    apply_alpha_mask(rgba_img_path, rgb_img_path, output_path)
