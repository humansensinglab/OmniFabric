#!/usr/bin/env python3
"""
Simulate a garment with a given texture, then render garment-only views and garment masks.

Usage:
    python -m omnifabric.sewing.textured_sim <garment_specification.json> <texture.png> --output_dir <dir>
"""

import argparse
import os
import shutil
import subprocess
import sys

import yaml

from omnifabric.paths import REPO_ROOT


def run_textured_garment_sim(garment_spec_path, custom_texture_path, output_dir="./results"):
    """Simulate the garment with the texture; returns <output_dir>/<garment>_textured."""
    garment_spec_path = os.path.abspath(garment_spec_path)
    custom_texture_path = os.path.abspath(custom_texture_path)
    output_dir = os.path.abspath(output_dir)

    garment_dir = os.path.dirname(garment_spec_path)
    garment_name = os.path.basename(garment_dir)

    custom_output_dir = os.path.join(output_dir, f"{garment_name}_textured")
    os.makedirs(custom_output_dir, exist_ok=True)
    custom_spec_path = os.path.join(custom_output_dir, f"{garment_name}_specification.json")
    shutil.copy2(garment_spec_path, custom_spec_path)
    for file in os.listdir(garment_dir):
        if file.endswith(('.obj', '.mtl', '.yaml', '.txt')) and not file.startswith('.'):
            shutil.copy2(os.path.join(garment_dir, file), os.path.join(custom_output_dir, file))

    custom_texture_dest = os.path.join(custom_output_dir, f"{garment_name}_texture_custom.png")
    shutil.copy2(custom_texture_path, custom_texture_dest)

    sim_props_path = os.path.join(custom_output_dir, 'sim_props.yaml')
    if os.path.exists(sim_props_path):
        with open(sim_props_path, 'r') as f:
            sim_props = yaml.safe_load(f)
        if 'uv_texture' in sim_props.get('render', {}).get('config', {}):
            sim_props['render']['config']['uv_texture']['fabric_grain_texture_path'] = custom_texture_dest
            with open(sim_props_path, 'w') as f:
                yaml.dump(sim_props, f, default_flow_style=False)

    subprocess.run([sys.executable, "-m", "omnifabric.sewing.simulate", "--json_spec_file", custom_spec_path,
                    "--easy_texture_path", custom_texture_dest], check=True, timeout=600, cwd=REPO_ROOT)

    garment_mesh_path = None
    for folder in (custom_output_dir, os.path.join(custom_output_dir, garment_name)):
        if os.path.isdir(folder):
            meshes = [f for f in os.listdir(folder) if f.endswith('_sim.obj')]
            if meshes:
                garment_mesh_path = os.path.join(folder, meshes[0])
                break
    if garment_mesh_path is None:
        print(f"Warning: no simulated mesh (*_sim.obj) in {custom_output_dir}; garment-only views were not rendered")
        return custom_output_dir

    try:
        from omnifabric.sewing.render_garment_only import render_garment_only_views
        render_garment_only_views(garment_mesh_path=garment_mesh_path, output_dir=custom_output_dir,
                                  garment_name=garment_name, resolution=(1080, 1080))
    except Exception as e:
        print(f"Warning: garment-only rendering failed: {e}")

    try:
        from omnifabric.sewing.garment_mask import generate_garment_masks
        generate_garment_masks(garment_mesh_path=garment_mesh_path, output_dir=os.path.dirname(garment_mesh_path),
                               garment_name=os.path.basename(garment_mesh_path)[:-len('_sim.obj')])
    except Exception as e:
        print(f"Warning: garment mask rendering failed: {e}")

    return custom_output_dir


def main():
    parser = argparse.ArgumentParser(description='Simulate a garment with a given texture')
    parser.add_argument('garment_spec', help='Path to garment specification JSON')
    parser.add_argument('custom_texture', help='Path to the texture')
    parser.add_argument('--output_dir', default='./custom_textured_results', help='Output directory')
    args = parser.parse_args()
    run_textured_garment_sim(args.garment_spec, args.custom_texture, args.output_dir)


if __name__ == '__main__':
    main()
