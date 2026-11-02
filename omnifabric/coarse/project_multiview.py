"""Project generated garment views onto the simulated garment mesh with Hunyuan3D-2's renderer.

The example directory <input_root>/<example_name>/ holds mesh.obj and {front,back,left,right}_view.png (RGBA,
alpha = garment). Writes <output_root>/results_multiview_general_<example_name>_full/ with the textured mesh
(*_textured_mesh.glb), the projected and generated views (generated_views/), renders of the textured mesh
(rendered_6views/) and the mesh silhouettes used to size the views (object_masks/).

Usage:
    python -m omnifabric.coarse.project_multiview --input_root IN --output_root OUT --example_name NAME
"""
import argparse
import json
import os

import numpy as np
import torch
import trimesh
from PIL import Image

from diffusers.utils import logging as diffusers_logging
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap

# Hunyuan3D-2 cameras: candidate_camera_azims = [0, 90, 180, 270, 0, 180], candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
VIEW_TO_CAMERA = {
    "front": {"azim": 0, "elev": 0, "index": 0},
    "right": {"azim": 90, "elev": 0, "index": 1},
    "back": {"azim": 180, "elev": 0, "index": 2},
    "left": {"azim": 270, "elev": 0, "index": 3},
    "top": {"azim": 0, "elev": 90, "index": 4},
    "bottom": {"azim": 180, "elev": -90, "index": 5},
}
SIDE_VIEWS = ["front", "right", "back", "left"]
INPUT_VIEW_WEIGHT = 90.0  # weight of the input views when blending front/back only (generated views: 0.01/0.005)
RENDER_RESOLUTION = 1024


def detect_input_views(input_dir):
    """{view: {"path", "camera"}} for each {view}_view.png in input_dir."""
    view_files = {}
    for view_name, camera in VIEW_TO_CAMERA.items():
        image_path = os.path.join(input_dir, f"{view_name}_view.png")
        if os.path.exists(image_path):
            view_files[view_name] = {"path": image_path, "camera": camera}
    if not view_files:
        raise FileNotFoundError(f"No {{front,back,left,right,top,bottom}}_view.png found in {input_dir}")
    return view_files


def generate_object_masks_and_bbox(pipeline, mesh, output_dir, output_filename, mask_resolution=1024):
    """Render the mesh silhouette from the four side cameras and save the masks and their bounding boxes (JSON)."""
    pipeline.render.load_mesh(mesh_uv_wrap(mesh))
    texture_size = pipeline.render.texture_size
    tex_w, tex_h = texture_size if isinstance(texture_size, tuple) else (texture_size, texture_size)
    pipeline.render.set_texture(Image.new('RGB', (tex_w, tex_h), (255, 255, 255)))

    masks_dir = os.path.join(output_dir, 'object_masks')
    os.makedirs(masks_dir, exist_ok=True)
    bbox_data = {}
    for view_name in SIDE_VIEWS:
        azim, elev = VIEW_TO_CAMERA[view_name]["azim"], VIEW_TO_CAMERA[view_name]["elev"]
        rendered = pipeline.render.render(elev=elev, azim=azim, resolution=(mask_resolution, mask_resolution),
                                          keep_alpha=True, bgcolor=None, filter_mode='linear', return_type='pl')
        if rendered.mode == 'RGBA':
            alpha = np.array(rendered.split()[-1])
        else:
            alpha = np.full((mask_resolution, mask_resolution), 255, np.uint8)
        mask_array = np.where(alpha > 0, 255, 0).astype(np.uint8)
        mask_path = os.path.join(masks_dir, f'{output_filename}_mask_{view_name}_azim{azim:03d}_elev{elev:+03d}.png')
        Image.fromarray(mask_array, mode='L').save(mask_path)

        entry = {'azim': azim, 'elev': elev, 'bbox': None,
                 'image_resolution': {'width': mask_resolution, 'height': mask_resolution}, 'mask_path': mask_path}
        rows, cols = np.where(mask_array > 0)
        if len(rows):
            entry['bbox'] = {'x_min': int(cols.min()), 'y_min': int(rows.min()),
                             'x_max': int(cols.max()), 'y_max': int(rows.max()),
                             'width': int(cols.max() - cols.min() + 1), 'height': int(rows.max() - rows.min() + 1)}
        else:
            entry['error'] = 'No object region found in mask'
        bbox_data[view_name] = entry

    bbox_json_path = os.path.join(masks_dir, f'{output_filename}_bbox_info.json')
    with open(bbox_json_path, 'w') as f:
        json.dump(bbox_data, f, indent=2)
    return bbox_data


def load_and_process_input_views(view_files, pipeline, bbox_data):
    """Load the input views and scale each garment to the mesh silhouette's bounding box from the same camera
    (padding to the mask resolution), then recenter them as Hunyuan3D does."""
    processed_views = {}
    for view_name, view_info in view_files.items():
        image = Image.open(view_info["path"])
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        target = bbox_data.get(view_name) if view_name in SIDE_VIEWS else None
        if target and target.get('bbox') is not None:
            rows, cols = np.where(np.array(image.split()[-1]) > 220)
            if len(rows):
                scale_w = target['bbox']['width'] / (cols.max() - cols.min() + 1)
                scale_h = target['bbox']['height'] / (rows.max() - rows.min() + 1)
                new_width, new_height = int(image.size[0] * scale_w), int(image.size[1] * scale_h)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                target_w = target['image_resolution'].get('width', new_width)
                target_h = target['image_resolution'].get('height', new_height)
                if (target_w, target_h) != (new_width, new_height):
                    padded = Image.new('RGBA', (target_w, target_h), (0, 0, 0, 0))
                    padded.paste(image, ((target_w - new_width) // 2, (target_h - new_height) // 2), image)
                    image = padded
            else:
                print(f"Warning: the {view_name} view is empty (no garment in its alpha channel)")

        processed_views[view_name] = pipeline.recenter_image(image, border_ratio=0.0, is_resized=True)
    return processed_views


def save_multiviews_general(pipeline, mesh, input_views, view_files, filename, multiview_dir):
    """Generate all 6 views with Hunyuan3D's multi-view model (from the first input view), then replace the generated
    views by the input views where they exist. Returns the 6 views and {camera index: view info}."""
    pipeline.render.load_mesh(mesh)
    elevs = pipeline.config.candidate_camera_elevs
    azims = pipeline.config.candidate_camera_azims
    target_resolution = max(img.size[0] for img in input_views.values())
    input_view_indices = {info["camera"]["index"]: name for name, info in view_files.items()}

    primary_image = input_views[list(input_views.keys())[0]]
    normal_maps = pipeline.render_normal_multiview(elevs, azims, use_abs_coor=True)
    position_maps = pipeline.render_position_multiview(elevs, azims)
    camera_info = [(((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[elev] +
                   {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev]
                   for azim, elev in zip(azims, elevs)]
    multiviews = pipeline.models['multiview_model']([primary_image], normal_maps + position_maps, camera_info)

    for idx, (elev, azim) in enumerate(zip(elevs, azims)):
        if idx not in input_view_indices:
            view = multiviews[idx].copy()
            if target_resolution != 512:
                view = view.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)
            view.save(os.path.join(multiview_dir, f'{filename}_view_elev{elev}_azim{azim}_AI_GENERATED.png'))

    view_info_map = {}
    for camera_idx, view_name in input_view_indices.items():
        input_image = input_views[view_name]
        if target_resolution != input_image.size[0]:
            input_image = input_image.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)
        if input_image.mode == 'RGBA':  # white background
            rgb = Image.new('RGB', input_image.size, (255, 255, 255))
            rgb.paste(input_image, mask=input_image.split()[-1])
            multiviews[camera_idx] = rgb
        else:
            multiviews[camera_idx] = input_image.convert('RGB')
        elev, azim = elevs[camera_idx], azims[camera_idx]
        multiviews[camera_idx].save(os.path.join(
            multiview_dir, f'{filename}_view_elev{elev}_azim{azim}_INPUT_{view_name.upper()}.png'))
        view_info_map[camera_idx] = {"view_name": view_name, "is_input": True, "azim": azim, "elev": elev}

    for idx in range(6):
        if idx not in view_info_map:
            view_info_map[idx] = {"view_name": f"generated_{idx}", "is_input": False,
                                  "azim": azims[idx], "elev": elevs[idx]}
    return multiviews, view_info_map


def apply_texture_general(pipeline, mesh, multiviews, view_info_map, input_weight=INPUT_VIEW_WEIGHT):
    """Project the views into the mesh texture and inpaint what no view covers.

    With side views present, projection is non-overlapping: front, then back, then left/right only on texels the
    earlier views left unpainted, then generated views. With only front/back inputs, the input views are blended with
    equal weights and the generated views with a tiny weight."""
    elevs = pipeline.config.candidate_camera_elevs
    azims = pipeline.config.candidate_camera_azims
    input_view_names = {info["view_name"] for info in view_info_map.values() if info["is_input"]}

    if input_view_names & {"left", "right"}:
        tier_of = {"front": 0, "back": 1, "left": 2, "right": 2}
        tiers = {}
        for idx, info in view_info_map.items():
            # input views: forgiving exponent 1.5; generated views: strict exponent 6 to minimize their influence
            tier, exp = (tier_of.get(info["view_name"], 2), 1.5) if info["is_input"] else (3, 6)
            project_texture, project_cos_map, _ = pipeline.render.back_project(multiviews[idx], elevs[idx], azims[idx])
            tiers.setdefault(tier, ([], []))
            tiers[tier][0].append(project_texture)
            tiers[tier][1].append(project_cos_map ** exp)

        texture, ori_trust_map = None, None
        for tier in sorted(tiers):
            tier_texture, tier_mask = pipeline.render.fast_bake_texture(*tiers[tier])
            if texture is None:
                texture, ori_trust_map = tier_texture, tier_mask
            else:
                texture = torch.where(tier_mask & ~ori_trust_map, tier_texture, texture)
                ori_trust_map = ori_trust_map | tier_mask
    else:
        weights = [0.01, 0.01, 0.01, 0.01, 0.005, 0.005]
        for idx, info in view_info_map.items():
            if info["is_input"]:
                weights[idx] = input_weight
        project_textures, project_weighted_cos_maps = [], []
        for i, (view, elev, azim, weight) in enumerate(zip(multiviews, elevs, azims, weights)):
            project_texture, project_cos_map, _ = pipeline.render.back_project(view, elev, azim)
            exp = 1.5 if view_info_map[i]["is_input"] else 6
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(weight * (project_cos_map ** exp))
        texture, ori_trust_map = pipeline.render.fast_bake_texture(project_textures, project_weighted_cos_maps)

    mask_np = ((ori_trust_map > 1E-8).squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
    texture = pipeline.texture_inpaint(texture, mask_np)
    pipeline.render.set_texture(texture)
    return pipeline.render.save_mesh()


def render_comprehensive_6views(pipeline, textured_mesh, output_dir, filename, resolution=RENDER_RESOLUTION):
    """Render the textured mesh from the 6 projection cameras (white background) into rendered_6views/."""
    sixview_dir = os.path.join(output_dir, 'rendered_6views')
    os.makedirs(sixview_dir, exist_ok=True)
    pipeline.render.load_mesh(textured_mesh)
    view_names = ["front", "right", "back", "left", "top", "bottom"]
    for i, (azim, elev, view_name) in enumerate(zip(pipeline.config.candidate_camera_azims,
                                                    pipeline.config.candidate_camera_elevs, view_names)):
        rendered = pipeline.render.render(elev=elev, azim=azim, resolution=(resolution, resolution),
                                          keep_alpha=False, bgcolor=[1, 1, 1], filter_mode='linear',
                                          return_type='pl')
        rendered.save(os.path.join(sixview_dir, f'{filename}_view_{i:02d}_{view_name}_elev{elev:+03d}_azim{azim:03d}.png'))


def main():
    parser = argparse.ArgumentParser(description='Project garment views onto the simulated mesh (Hunyuan3D-2)')
    parser.add_argument('--input_root', type=str, required=True,
                        help='Directory containing the example directory (mesh.obj + *_view.png)')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Directory where results_multiview_general_<example>_full/ is written')
    parser.add_argument('--example_name', type=str, required=True, help='Name of the example directory')
    args = parser.parse_args()

    print("Projecting the views onto the mesh (Hunyuan3D-2)")
    # diffusers logs a harmless error while probing a VAE folder that Hunyuan3D-2's turbo model does not have
    diffusers_logging.set_verbosity(diffusers_logging.CRITICAL)
    pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
    diffusers_logging.set_verbosity(diffusers_logging.ERROR)
    pipeline.render.bake_angle_thres = 75
    pipeline.render.enable_edge_shrink = False
    pipeline.render.enable_boundary_mask = False

    input_dir = os.path.join(args.input_root, args.example_name)
    output_dir = os.path.join(args.output_root, f"results_multiview_general_{args.example_name}_full")
    multiview_dir = os.path.join(output_dir, 'generated_views')
    os.makedirs(multiview_dir, exist_ok=True)
    output_filename = f"mesh_multiview_full_{args.example_name}"

    view_files = detect_input_views(input_dir)
    mesh = trimesh.load(os.path.join(input_dir, 'mesh.obj'))
    if isinstance(mesh, trimesh.Scene):
        meshes = mesh.dump()
        mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)

    bbox_data = generate_object_masks_and_bbox(pipeline, mesh, output_dir, output_filename)
    input_views = load_and_process_input_views(view_files, pipeline, bbox_data)
    multiviews, view_info_map = save_multiviews_general(pipeline, mesh, input_views, view_files, output_filename,
                                                        multiview_dir)
    textured_mesh = apply_texture_general(pipeline, mesh, multiviews, view_info_map)
    textured_mesh.export(os.path.join(output_dir, f'{output_filename}_textured_mesh.glb'))
    render_comprehensive_6views(pipeline, textured_mesh, output_dir, output_filename)


if __name__ == "__main__":
    main()
