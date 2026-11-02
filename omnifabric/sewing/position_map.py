#!/usr/bin/env python3

"""
Create the position map of a simulated garment: a map in the sewing-pattern UV space (the template's resolution)
storing at each pixel the 3D position (x, y, z) of the corresponding point on the mesh.

Output: <garment>_position_map.pt, a dict with 'position_map' (H, W, 3) float32 and 'validity_mask' (H, W) bool, plus
<garment>_position_map_visualization.png (x, y, z as RGB).

Usage:
    python -m omnifabric.sewing.position_map <garment_specification.json> --output_dir <dir>
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import trimesh
from PIL import Image


def load_obj_with_uv_manual(obj_path):
    """Fallback OBJ parser for when trimesh does not return UV coordinates."""
    vertices, uv_coords, faces, face_uvs = [], [], [], []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('vt '):
                parts = line.split()
                uv_coords.append([float(parts[1]), float(parts[2])])
            elif line.startswith('f '):
                face_verts, face_uv_indices = [], []
                for part in line.split()[1:]:
                    indices = part.split('/')
                    face_verts.append(int(indices[0]) - 1)
                    if len(indices) > 1 and indices[1]:
                        face_uv_indices.append(int(indices[1]) - 1)
                # triangles, and quads split into two triangles
                triangles = {3: [[0, 1, 2]], 4: [[0, 1, 2], [0, 2, 3]]}.get(len(face_verts), [])
                for tri in triangles:
                    faces.append([face_verts[i] for i in tri])
                    if face_uv_indices:
                        face_uvs.append([face_uv_indices[i] for i in tri])

    return (np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32),
            np.array(uv_coords, dtype=np.float32) if uv_coords else None,
            np.array(face_uvs, dtype=np.int32) if face_uvs else None)


def load_mesh_with_uv(mesh_path):
    """Vertices (V, 3), faces (F, 3), UV coordinates (UV, 2) in [0, 1] and face UV indices (F, 3), 0-indexed."""
    mesh_path = str(mesh_path)
    mesh = trimesh.load(mesh_path, process=False)  # process=False preserves the UV coordinates
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    uv_coords = face_uvs = None
    if getattr(mesh.visual, 'uv', None) is not None:
        uv_coords = np.array(mesh.visual.uv, dtype=np.float32)
        uv_faces = getattr(mesh.visual, 'uv_faces', None)
        face_uvs = np.array(uv_faces, dtype=np.int32) if uv_faces is not None else faces.copy()  # per-vertex UVs
    elif mesh_path.lower().endswith('.obj'):
        vertices, faces, uv_coords, face_uvs = load_obj_with_uv_manual(mesh_path)

    if uv_coords is not None:
        uv_min, uv_max = uv_coords.min(axis=0), uv_coords.max(axis=0)
        if uv_min.min() < -0.01 or uv_max.max() > 1.01:  # normalize UVs stored in another range
            uv_range = uv_max - uv_min
            uv_range[uv_range < 1e-6] = 1.0
            uv_coords = (uv_coords - uv_min) / uv_range
    return vertices, faces, uv_coords, face_uvs


def compute_barycentric(p, v0, v1, v2):
    """
    Compute barycentric coordinates of point p in triangle (v0, v1, v2)

    Returns:
        bary: (3,) array with barycentric coordinates [w0, w1, w2]
              where p = w0*v0 + w1*v1 + w2*v2
    """
    # Vectors from v0 to v1 and v2
    e1 = v1 - v0
    e2 = v2 - v0

    # Vector from v0 to p
    e = p - v0

    # Compute barycentric coordinates using dot products
    d00 = np.dot(e1, e1)
    d01 = np.dot(e1, e2)
    d11 = np.dot(e2, e2)
    d20 = np.dot(e, e1)
    d21 = np.dot(e, e2)

    denom = d00 * d11 - d01 * d01

    if abs(denom) < 1e-10:
        # Degenerate triangle
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    w1 = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    w0 = 1.0 - w1 - w2

    return np.array([w0, w1, w2], dtype=np.float32)


def process_triangle_batch(face_indices, vertices, faces, pixel_uv_coords, face_uvs, width, height):
    """
    Process a batch of triangles in parallel

    Returns:
        List of (y_pixel, u_pixel, pos_3d) tuples for valid pixels
    """
    results = []

    for face_idx in face_indices:
        face = faces[face_idx]
        face_uv = face_uvs[face_idx]

        # Get 3D vertices of this triangle
        v0_3d = vertices[face[0]]
        v1_3d = vertices[face[1]]
        v2_3d = vertices[face[2]]

        # Get UV coordinates of this triangle (in pixel space)
        uv0 = pixel_uv_coords[face_uv[0]]
        uv1 = pixel_uv_coords[face_uv[1]]
        uv2 = pixel_uv_coords[face_uv[2]]

        # Find bounding box of triangle in UV space
        min_u = int(np.floor(min(uv0[0], uv1[0], uv2[0])))
        max_u = int(np.ceil(max(uv0[0], uv1[0], uv2[0])))
        min_v = int(np.floor(min(uv0[1], uv1[1], uv2[1])))
        max_v = int(np.ceil(max(uv0[1], uv1[1], uv2[1])))

        # Clamp to image bounds
        min_u = max(0, min_u)
        max_u = min(width - 1, max_u)
        min_v = max(0, min_v)
        max_v = min(height - 1, max_v)

        # Rasterize triangle: check each pixel in bounding box
        for v_pixel in range(min_v, max_v + 1):
            for u_pixel in range(min_u, max_u + 1):
                # Pixel center in UV space
                p = np.array([u_pixel + 0.5, v_pixel + 0.5], dtype=np.float32)

                # Compute barycentric coordinates
                bary = compute_barycentric(p, uv0, uv1, uv2)

                # Check if point is inside triangle (all barycentric coords >= 0)
                if bary[0] >= 0 and bary[1] >= 0 and bary[2] >= 0:
                    # Interpolate 3D position using barycentric coordinates
                    pos_3d = bary[0] * v0_3d + bary[1] * v1_3d + bary[2] * v2_3d

                    # Store coordinates (note: image coordinates are flipped)
                    # UV v-axis points up, image y-axis points down
                    y_pixel = height - 1 - v_pixel
                    results.append((y_pixel, u_pixel, pos_3d))

    return results


def create_position_map_from_mesh(vertices, faces, uv_coords, face_uvs, width, height, num_workers=None):
    """Rasterize the mesh in UV space (in parallel over batches of triangles); returns the position map (H, W, 3)
    and the validity mask (H, W)."""
    num_workers = num_workers or cpu_count()
    total_faces = len(faces)
    position_map = np.zeros((height, width, 3), dtype=np.float32)
    validity_mask = np.zeros((height, width), dtype=bool)

    pixel_uv_coords = uv_coords.copy()
    pixel_uv_coords[:, 0] *= width
    pixel_uv_coords[:, 1] *= height

    batch_size = max(1, total_faces // (num_workers * 4))  # 4 batches per worker for load balancing
    face_batches = [list(range(i, min(i + batch_size, total_faces))) for i in range(0, total_faces, batch_size)]
    process_func = partial(process_triangle_batch, vertices=vertices, faces=faces, pixel_uv_coords=pixel_uv_coords,
                           face_uvs=face_uvs, width=width, height=height)
    with Pool(processes=num_workers) as pool:
        batch_results = pool.map(process_func, face_batches)

    for batch_result in batch_results:
        for y_pixel, u_pixel, pos_3d in batch_result:
            position_map[y_pixel, u_pixel] = pos_3d
            validity_mask[y_pixel, u_pixel] = True
    return position_map, validity_mask


def save_position_map(position_map, validity_mask, output_path):
    position_tensor = torch.from_numpy(position_map)
    torch.save({
        'position_map': position_tensor,
        'validity_mask': torch.from_numpy(validity_mask),
        'shape': position_tensor.shape,
        'dtype': str(position_tensor.dtype),
        'coordinate_system': '3D mesh coordinates (x, y, z)'
    }, output_path)


def create_position_map_visualization(position_map, validity_mask, output_path):
    """x, y, z mapped to RGB (each normalized over the garment); background white."""
    vis_map = np.zeros(position_map.shape, dtype=np.uint8)
    if validity_mask.any():
        valid_positions = position_map[validity_mask]
        for channel in range(3):
            min_val, max_val = valid_positions[:, channel].min(), valid_positions[:, channel].max()
            if max_val > min_val:
                normalized = (position_map[:, :, channel] - min_val) / (max_val - min_val) * 255
                vis_map[:, :, channel] = np.clip(normalized, 0, 255).astype(np.uint8)
        vis_map[~validity_mask] = 255
    Image.fromarray(vis_map).save(output_path)


def find_mesh(garment_dir, garment_name):
    """The simulated mesh: *_sim.obj > *_sim.ply > *.obj > *.ply, also in ChatGarment's nested <garment>/ folder."""
    candidates = []
    for folder in (garment_dir, os.path.join(garment_dir, garment_name)):
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            if file.endswith('_sim.obj'):
                candidates.insert(0, path)
            elif file.endswith('_sim.ply'):
                candidates.insert(sum(1 for c in candidates if c.endswith('_sim.obj')), path)
            elif file.endswith(('.obj', '.ply')):
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No mesh (.obj or .ply) files found in {garment_dir}")
    return candidates[0]


def create_position_map(garment_spec_path, output_dir, mesh_filename=None, num_workers=None):
    """Create the position map at the resolution of output_dir/editable_template.png; returns the .pt path."""
    garment_dir = os.path.dirname(garment_spec_path)
    garment_name = os.path.basename(garment_dir)
    mesh_path = os.path.join(garment_dir, mesh_filename) if mesh_filename else find_mesh(garment_dir, garment_name)

    vertices, faces, uv_coords, face_uvs = load_mesh_with_uv(mesh_path)
    if uv_coords is None or face_uvs is None:
        raise ValueError(f"Mesh does not have UV coordinates: {mesh_path}")

    # resolution: the UV template's, else the mesh's UV texture's
    sizes = [os.path.join(d, 'editable_template.png') for d in (output_dir, garment_dir, os.path.dirname(garment_dir))]
    mesh_dir = os.path.dirname(mesh_path)
    sizes += [os.path.join(mesh_dir, f) for f in sorted(os.listdir(mesh_dir)) if f.endswith('_texture.png')]
    existing = [p for p in sizes if os.path.exists(p)]
    if not existing:
        raise FileNotFoundError(f"No editable_template.png in {output_dir} to take the position map's resolution from")
    width, height = Image.open(existing[0]).size

    position_map, validity_mask = create_position_map_from_mesh(
        vertices, faces, uv_coords, face_uvs, width, height, num_workers)

    os.makedirs(output_dir, exist_ok=True)
    pt_output_path = os.path.join(output_dir, f"{garment_name}_position_map.pt")
    save_position_map(position_map, validity_mask, pt_output_path)
    create_position_map_visualization(position_map, validity_mask,
                                      os.path.join(output_dir, f"{garment_name}_position_map_visualization.png"))
    return pt_output_path


def main():
    parser = argparse.ArgumentParser(description='Create the position map of a simulated garment')
    parser.add_argument('garment_spec', help='Path to garment specification JSON')
    parser.add_argument('--output_dir', required=True, help='Folder with editable_template.png; output folder')
    parser.add_argument('--mesh_file', default=None, help='Specific mesh filename to use (optional)')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers (default: CPU count)')
    args = parser.parse_args()
    create_position_map(args.garment_spec, args.output_dir, args.mesh_file, args.num_workers)


if __name__ == '__main__':
    main()
