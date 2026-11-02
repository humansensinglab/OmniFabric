#!/usr/bin/env python3
"""
Garment Mask Generation Module
Renders a binary garment-silhouette mask that is pixel-aligned with the
combined body+garment renders (*_render_front.png / *_render_back.png)
produced by GarmentCodeRC's pygarment/meshgen/render/pythonrender.py.

The alignment works because the body+garment renderer uses a fixed
perspective camera (location taken from sim_props.yaml's
render.config.front_camera_location) that does not depend on the garment
itself. Rendering only the garment mesh with that same camera and reading
back the alpha channel therefore yields a mask where white = garment,
black = everything else (background and body).
"""

import os
import platform
if platform.system() == 'Linux':
    os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import trimesh
import pyrender
import yaml
from PIL import Image

from omnifabric.utils.crop_ref import compute_crop_box

# Must match GarmentCodeRC/pygarment/meshgen/render/pythonrender.py defaults
DEFAULT_RESOLUTION = (2048, 2048)
DEFAULT_CAMERA_LOCATION = [0.0, 0.97, 4.15]
DEFAULT_SIDES = ['front', 'back']
DEFAULT_YFOV = np.pi / 6.

# Must match the crop applied to *_render_front.png / *_render_back.png
# in scripts/run_pipeline.sh (crop_ref --ratio 0.85 --vertical_offset -140)
CROP_RATIO = 0.85
CROP_VERTICAL_OFFSET = -140


def rotate_matrix_y(matrix, angle_deg):
    """Rotate matrix around Y-axis (matches pythonrender.py exactly)"""
    rotation_angle = angle_deg * (np.pi / 180)
    rotation_matrix = np.array([
        [np.cos(rotation_angle), 0, np.sin(rotation_angle), 0],
        [0, 1, 0, 0],
        [-np.sin(rotation_angle), 0, np.cos(rotation_angle), 0],
        [0, 0, 0, 1]
    ])
    return np.dot(rotation_matrix, matrix)


def rotate_matrix_x(matrix, angle_deg):
    """Rotate matrix around X-axis (matches pythonrender.py exactly)"""
    rotation_angle = angle_deg * (np.pi / 180)
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [0, np.sin(rotation_angle), np.cos(rotation_angle), 0],
        [0, 0, 0, 1]
    ])
    return np.dot(rotation_matrix, matrix)


def build_camera_pose(camera_location, side):
    """Build the camera pose exactly as pythonrender.py's create_camera does"""
    camera_pose = np.array([
        [1.0, 0.0, 0.0, camera_location[0]],
        [0.0, 1.0, 0.0, camera_location[1]],
        [0.0, 0.0, 1.0, camera_location[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])
    camera_pose = rotate_matrix_x(camera_pose, 0)
    camera_pose = rotate_matrix_y(camera_pose, 0)
    if side == 'back':
        camera_pose = rotate_matrix_y(camera_pose, 180)
    return camera_pose


def load_render_config(sim_props_path):
    """Read resolution / camera location / sides used for the body+garment render"""
    resolution = DEFAULT_RESOLUTION
    camera_location = DEFAULT_CAMERA_LOCATION
    sides = DEFAULT_SIDES

    if sim_props_path and os.path.exists(sim_props_path):
        with open(sim_props_path, 'r') as f:
            sim_props = yaml.safe_load(f)
        render_cfg = (sim_props or {}).get('render', {}).get('config', {})
        resolution = tuple(render_cfg.get('resolution', resolution))
        camera_location = render_cfg.get('front_camera_location', camera_location)
        sides = render_cfg.get('sides', sides)

    return resolution, camera_location, sides


def render_garment_mask(garment_mesh_path, side, camera_location, resolution, output_path, threshold=127):
    """Render the garment-only silhouette and save its alpha channel as a binary mask"""

    garment_mesh = trimesh.load_mesh(str(garment_mesh_path))
    garment_mesh.vertices = garment_mesh.vertices / 100.0  # cm -> m, matches pythonrender.py

    # Only the silhouette matters here, so a plain opaque double-sided
    # material is enough (no need to load/decode the garment texture).
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True,
    )
    pyrender_garm_mesh = pyrender.Mesh.from_trimesh(garment_mesh, material=material, smooth=True)

    scene = pyrender.Scene(bg_color=(0.0, 0.0, 0.0, 0.0))
    scene.add(pyrender_garm_mesh)

    camera = pyrender.PerspectiveCamera(yfov=DEFAULT_YFOV)
    camera_pose = build_camera_pose(camera_location, side)
    scene.add(camera, pose=camera_pose)

    width, height = resolution
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    alpha = color[..., 3]
    mask = np.where(alpha > threshold, 255, 0).astype(np.uint8)
    mask_image = Image.fromarray(mask, mode='L')
    mask_image.save(output_path, "PNG")

    return output_path, mask_image


def save_cropped_mask(mask_image, output_path):
    """Crop a mask with the exact same box used for *_render_{side}_cropped.png"""
    w, h = mask_image.size
    box = compute_crop_box(w, h, CROP_RATIO, CROP_VERTICAL_OFFSET)
    mask_image.crop(box).save(output_path, "PNG")
    return output_path


def generate_garment_masks(garment_mesh_path, output_dir, garment_name, sim_props_path=None):
    """Render front/back (or whatever sides were used) garment masks aligned to the body+garment
    renders, plus cropped versions aligned to *_render_{side}_cropped.png"""

    if sim_props_path is None:
        candidate = os.path.join(os.path.dirname(garment_mesh_path), 'sim_props.yaml')
        if os.path.exists(candidate):
            sim_props_path = candidate

    resolution, camera_location, sides = load_render_config(sim_props_path)

    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for side in sides:
        output_path = os.path.join(output_dir, f"{garment_name}_render_{side}_mask.png")
        output_path, mask_image = render_garment_mask(
            garment_mesh_path, side, camera_location, resolution, output_path
        )
        results[side] = output_path

        cropped_output_path = os.path.join(output_dir, f"{garment_name}_render_{side}_cropped_mask.png")
        results[f"{side}_cropped"] = save_cropped_mask(mask_image, cropped_output_path)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Render garment silhouette masks aligned with *_render_front.png / *_render_back.png'
    )
    parser.add_argument('garment_mesh', help='Path to simulated garment mesh (*_sim.obj)')
    parser.add_argument('output_dir', help='Directory to save mask PNGs (same dir as *_render_front.png)')
    parser.add_argument('garment_name', help='Garment name prefix used in output filenames')
    parser.add_argument('--sim_props', default=None, help='Path to sim_props.yaml (for camera/resolution)')
    args = parser.parse_args()

    results = generate_garment_masks(
        args.garment_mesh, args.output_dir, args.garment_name, sim_props_path=args.sim_props
    )
    for side, path in results.items():
        print(f"Saved {side} mask: {path}")


if __name__ == '__main__':
    main()
