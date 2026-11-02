#!/usr/bin/env python3

"""
Create the binary UV template of a garment's sewing pattern (black panels on a white background) from the UV texture
GarmentCode renders for the simulated garment.

Usage:
    python -m omnifabric.sewing.uv_template <garment_specification.json> --output_dir <dir> [--template_name name.png]
"""

import argparse
import os

import numpy as np
from PIL import Image
from scipy import ndimage


def create_texture_template(garment_spec_path, output_dir="./templates", template_name="editable_template.png"):
    """Create the template next to the garment specification's UV texture; returns its path."""
    garment_dir = os.path.dirname(garment_spec_path)
    garment_name = os.path.basename(garment_dir)

    # UV texture map (*_texture.png), else the pattern drawing (*_pattern.png); ChatGarment keeps the simulated
    # garment's texture in a nested <garment>/<garment>/ folder, which takes precedence
    texture_files = [os.path.join(garment_dir, f) for f in os.listdir(garment_dir) if f.endswith('_texture.png')]
    pattern_files = [os.path.join(garment_dir, f) for f in os.listdir(garment_dir) if f.endswith('_pattern.png')]
    if not texture_files and not pattern_files:
        raise FileNotFoundError(f"No texture or pattern files found in {garment_dir}")
    pattern_path = (texture_files or pattern_files)[0]
    nested_garment_dir = os.path.join(garment_dir, garment_name)
    if os.path.exists(nested_garment_dir):
        nested = [f for f in os.listdir(nested_garment_dir) if f.endswith('_texture.png')]
        if nested:
            pattern_path = os.path.join(nested_garment_dir, nested[0])

    texture_array = np.array(Image.open(pattern_path).convert('RGB'))
    height, width = texture_array.shape[:2]

    # everything that is not (near-)white background is a panel; fill holes so panels are solid
    panel_mask = ndimage.binary_fill_holes(~np.all(texture_array > 240, axis=2))
    template_array = np.full((height, width, 3), 255, dtype=np.uint8)
    template_array[panel_mask] = 0

    os.makedirs(output_dir, exist_ok=True)
    template_path = os.path.join(output_dir, template_name)
    Image.fromarray(template_array).save(template_path)
    return template_path


def main():
    parser = argparse.ArgumentParser(description='Create the UV template of a garment sewing pattern')
    parser.add_argument('garment_spec', help='Path to garment specification JSON')
    parser.add_argument('--output_dir', default='./templates', help='Output directory for template')
    parser.add_argument('--template_name', default='editable_template.png', help='Template filename')
    args = parser.parse_args()
    create_texture_template(args.garment_spec, args.output_dir, args.template_name)


if __name__ == '__main__':
    main()
