"""Square crop of a render (a ratio of its width), centered and shifted vertically.

Usage:
    python -m omnifabric.utils.crop_ref --input render.png --output cropped.png --ratio 0.85 --vertical_offset -140
"""
import argparse

from PIL import Image


def compute_crop_box(w, h, crop_ratio, vertical_offset=0):
    """
    Compute a centered, square crop box of crop_ratio of width, shifted
    vertically by vertical_offset. Shared by center_crop_rgba_image and
    any other code that needs to crop a same-resolution image (e.g. masks)
    identically.
    """
    crop_width = int(w * crop_ratio)
    crop_height = crop_width  # keep square

    # Calculate coordinates for crop box
    left = (w - crop_width) // 2
    # Start from center and apply vertical offset (positive moves up, negative moves down)
    center_top = (h - crop_height) // 2
    top = center_top - vertical_offset  # Subtract because positive offset moves up (decreases y)
    right = left + crop_width
    bottom = top + crop_height

    return (left, top, right, bottom)


def center_crop_rgba_image(input_path, output_path, crop_ratio, vertical_offset=0):
    """
    Crop an RGBA image based on crop_ratio of width, keeping aspect ratio square.
    The crop box can be shifted vertically using vertical_offset.

    Parameters:
        input_path: path to input RGBA image
        output_path: path to save cropped image
        crop_ratio: float, ratio of original width to crop (e.g., 0.68 for 68%)
        vertical_offset: int, offset to move crop box up (positive) or down (negative) from center
    """
    with Image.open(input_path) as img:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        w, h = img.size
        box = compute_crop_box(w, h, crop_ratio, vertical_offset)

        cropped = img.crop(box)
        cropped.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Center crop an RGBA image based on a ratio of the width.")
    parser.add_argument("--input", required=True, help="Path to input RGBA image")
    parser.add_argument("--output", required=True, help="Path to save cropped image")
    parser.add_argument("--ratio", type=float, default=0.68,
                        help="Ratio of original width to crop (default: 0.68)")
    parser.add_argument("--vertical_offset", type=int, default=0,
                        help="Vertical offset of the crop box (default: 0)")
    args = parser.parse_args()
    center_crop_rgba_image(args.input, args.output, args.ratio, args.vertical_offset)


if __name__ == "__main__":
    main()
