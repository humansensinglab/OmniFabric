#!/usr/bin/env python3
"""
Extract the coarse texture map in the sewing-pattern UV space: the simulated mesh (OBJ, with sewing-pattern UVs) and the
textured mesh from the projection (GLB, same faces, Hunyuan3D UVs) are matched face by face, each face is rasterized in
the template's UV space sampling the GLB texture (barycentric interpolation), and white seams between panels' texels
are filled.

Usage:
    python -m omnifabric.coarse.extract_uv_texture --obj_path mesh.obj --glb_path textured_mesh.glb \
        --template_path editable_template.png --output_path coarse_texture.png
"""

import argparse
import multiprocessing as mp
import os
from typing import Dict, Tuple

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

WHITE_THRESHOLD = 250  # pixels brighter than this are background


def find_best_matches_batch(args):
    """For each OBJ face signature of the batch, the index of the closest GLB face signature."""
    start_idx, end_idx, obj_sigs_batch, glb_sigs_all = args
    batch_mapping = np.zeros(end_idx - start_idx, dtype=np.int32)
    for i in range(end_idx - start_idx):
        batch_mapping[i] = np.argmin(np.sum((glb_sigs_all - obj_sigs_batch[i]) ** 2, axis=1))
    return start_idx, batch_mapping


def load_meshes_and_extract_data(obj_path: str, glb_path: str) -> Dict:
    obj_mesh = trimesh.load(obj_path)
    glb_scene = trimesh.load(glb_path)
    glb_mesh = list(glb_scene.geometry.values())[0] if hasattr(glb_scene, 'geometry') else glb_scene

    material = getattr(glb_mesh.visual, 'material', None)
    if getattr(material, 'baseColorTexture', None) is None:
        raise ValueError(f"No texture found in {glb_path}")
    obj_uvs = getattr(obj_mesh.visual, 'uv', None)
    glb_uvs = getattr(glb_mesh.visual, 'uv', None)
    if obj_uvs is None or glb_uvs is None:
        raise ValueError(f"No UV coordinates in {obj_path if obj_uvs is None else glb_path}")

    return {
        'glb_texture': np.array(material.baseColorTexture),
        'obj_vertices': obj_mesh.vertices,
        'glb_vertices': glb_mesh.vertices,
        'obj_faces': obj_mesh.faces,
        'glb_faces': glb_mesh.faces,
        'obj_uvs': obj_uvs,
        'glb_uvs': glb_uvs
    }


def create_face_mapping(data: Dict, num_processes: int) -> np.ndarray:
    """Index of the GLB face matching each OBJ face (same vertex positions), computed in parallel."""
    def create_face_signature(vertices, faces, precision=6):
        sorted_positions = np.sort(vertices[faces].reshape(len(faces), -1), axis=1)
        return np.round(sorted_positions, decimals=precision)

    obj_faces = data['obj_faces']
    obj_signatures = create_face_signature(data['obj_vertices'], obj_faces)
    glb_signatures = create_face_signature(data['glb_vertices'], data['glb_faces'])

    batch_size = max(100, len(obj_faces) // (num_processes * 4))
    batches = [(i, min(i + batch_size, len(obj_faces)), obj_signatures[i:i + batch_size], glb_signatures)
               for i in range(0, len(obj_faces), batch_size)]
    with mp.Pool(num_processes) as pool:
        results = pool.map(find_best_matches_batch, batches)

    mapping = np.zeros(len(obj_faces), dtype=np.int32)
    for start_idx, batch_mapping in results:
        mapping[start_idx:start_idx + len(batch_mapping)] = batch_mapping
    return mapping


def compute_barycentric_coordinates(points: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Compute barycentric coordinates for points relative to triangle"""
    v0, v1, v2 = triangle[0], triangle[1], triangle[2]

    # Compute triangle area using cross product
    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])

    if abs(denom) < 1e-10:
        return np.zeros((len(points), 3))

    # Compute barycentric coordinates
    w0 = ((v1[1] - v2[1]) * (points[:, 0] - v2[0]) + (v2[0] - v1[0]) * (points[:, 1] - v2[1])) / denom
    w1 = ((v2[1] - v0[1]) * (points[:, 0] - v2[0]) + (v0[0] - v2[0]) * (points[:, 1] - v2[1])) / denom
    w2 = 1 - w0 - w1

    return np.column_stack([w0, w1, w2])


def rasterize_triangle_barycentric_fast(triangle_coords: np.ndarray, triangle_uvs: np.ndarray,
                                      glb_texture: np.ndarray, output: np.ndarray, template_size: Tuple[int, int]):
    """Fast barycentric rasterization with optimizations"""
    # Get triangle bounding box
    min_x = int(max(0, np.floor(np.min(triangle_coords[:, 0]))))
    max_x = int(min(template_size[0] - 1, np.ceil(np.max(triangle_coords[:, 0]))))
    min_y = int(max(0, np.floor(np.min(triangle_coords[:, 1]))))
    max_y = int(min(template_size[1] - 1, np.ceil(np.max(triangle_coords[:, 1]))))

    if min_x >= max_x or min_y >= max_y:
        return

    # Limit processing to reasonable triangle sizes
    if (max_x - min_x) * (max_y - min_y) > 10000:  # Skip very large triangles
        # Fall back to center point sampling
        center_x = int((min_x + max_x) / 2)
        center_y = int((min_y + max_y) / 2)
        center_uv = np.mean(triangle_uvs, axis=0)
        tex_x = int(np.clip(center_uv[0] * (glb_texture.shape[1] - 1), 0, glb_texture.shape[1] - 1))
        tex_y = int(np.clip((1 - center_uv[1]) * (glb_texture.shape[0] - 1), 0, glb_texture.shape[0] - 1))
        output[center_y, center_x] = glb_texture[tex_y, tex_x]
        return

    # Vectorized point generation
    y_range = np.arange(min_y, max_y + 1)
    x_range = np.arange(min_x, max_x + 1)
    xx, yy = np.meshgrid(x_range, y_range)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    # Compute barycentric coordinates
    bary_coords = compute_barycentric_coordinates(points, triangle_coords)

    # Find points inside triangle
    inside_mask = (bary_coords >= -1e-6).all(axis=1) & (bary_coords.sum(axis=1) <= 1 + 1e-6)

    if not inside_mask.any():
        return

    # Get valid points and coordinates
    valid_points = points[inside_mask]
    valid_bary = bary_coords[inside_mask]

    # Interpolate UV coordinates
    interp_uvs = np.sum(valid_bary[:, :, np.newaxis] * triangle_uvs[np.newaxis, :, :], axis=1)

    # Convert to texture coordinates
    tex_coords = interp_uvs.copy()
    tex_coords[:, 0] *= (glb_texture.shape[1] - 1)
    tex_coords[:, 1] = (1 - tex_coords[:, 1]) * (glb_texture.shape[0] - 1)
    tex_coords = np.clip(tex_coords, 0, [glb_texture.shape[1]-1, glb_texture.shape[0]-1]).astype(np.int32)

    # Sample colors and write to output
    colors = glb_texture[tex_coords[:, 1], tex_coords[:, 0]]
    valid_y = valid_points[:, 1].astype(np.int32)
    valid_x = valid_points[:, 0].astype(np.int32)
    output[valid_y, valid_x] = colors


def create_texture_mapping_barycentric(data: Dict, face_mapping: np.ndarray,
                                     template_size: Tuple[int, int]) -> np.ndarray:
    """Create texture mapping using barycentric coordinate sampling"""
    obj_faces = data['obj_faces']
    obj_uvs = data['obj_uvs']
    glb_faces = data['glb_faces']
    glb_uvs = data['glb_uvs']
    glb_texture = data['glb_texture']

    # Create output texture
    output_texture = np.ones((template_size[1], template_size[0], 3), dtype=np.uint8) * 255

    for i, obj_face in enumerate(obj_faces):
        # Get corresponding GLB face
        glb_face_idx = face_mapping[i]
        glb_face = glb_faces[glb_face_idx]

        # Get coordinates
        obj_face_uvs = obj_uvs[obj_face].astype(np.float32)
        triangle_coords = obj_face_uvs.copy()
        triangle_coords[:, 0] *= (template_size[0] - 1)
        triangle_coords[:, 1] = (1 - triangle_coords[:, 1]) * (template_size[1] - 1)

        glb_face_uvs = glb_uvs[glb_face].astype(np.float32)

        # Rasterize with barycentric sampling
        rasterize_triangle_barycentric_fast(triangle_coords, glb_face_uvs,
                                               glb_texture, output_texture, template_size)

    return output_texture

# ---- seam removal: fill the white gaps left between rasterized faces and along panel edges


def detect_texture_mask(image: np.ndarray) -> np.ndarray:
    """Create mask of non-white (textured) pixels"""
    # Convert to grayscale for easier processing
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    # Create mask: True for textured pixels, False for white background
    texture_mask = gray < WHITE_THRESHOLD
    return texture_mask.astype(np.uint8) * 255


def dilate_texture_edges(image: np.ndarray, mask: np.ndarray,
                       iterations: int = 3, kernel_size: int = 5) -> np.ndarray:
    """Dilate textured regions to fill small gaps"""
    # Create dilation kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Dilate the mask to expand textured regions
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)

    # For each color channel, extend colors into dilated regions
    result = image.copy()

    for channel in range(image.shape[2]):
        # Use inpainting to fill the expanded regions
        channel_data = image[:, :, channel].astype(np.uint8)

        # Create inpainting mask (areas to fill)
        inpaint_mask = (dilated_mask > 0) & (mask == 0)
        inpaint_mask = inpaint_mask.astype(np.uint8) * 255

        if np.any(inpaint_mask > 0):
            # Use OpenCV inpainting to fill gaps
            inpainted = cv2.inpaint(channel_data, inpaint_mask,
                                  inpaintRadius=kernel_size,
                                  flags=cv2.INPAINT_TELEA)
            result[:, :, channel] = inpainted

    return result


def smart_gap_filling(image: np.ndarray, mask: np.ndarray,
                     max_gap_size: int = 10) -> np.ndarray:
    """Fill small gaps between texture regions using intelligent sampling"""
    result = image.copy()

    # Find white/background regions
    white_mask = mask == 0

    # Create distance transform to find thin gaps
    dist_transform = cv2.distanceTransform(white_mask.astype(np.uint8),
                                         cv2.DIST_L2, 5)

    # Find thin gaps (small distance values)
    thin_gaps = (dist_transform > 0) & (dist_transform <= max_gap_size)

    if np.any(thin_gaps):
        # For each channel, fill thin gaps using nearby texture colors
        for channel in range(image.shape[2]):
            channel_data = image[:, :, channel].astype(np.float32)

            # Create Gaussian blur to propagate nearby colors
            blurred = cv2.GaussianBlur(channel_data, (21, 21), 0)

            # Replace thin gaps with blurred values
            result[thin_gaps, channel] = blurred[thin_gaps]

    return result.astype(np.uint8)


def edge_color_propagation(image: np.ndarray, mask: np.ndarray,
                         propagation_distance: int = 5) -> np.ndarray:
    """Propagate edge colors outward to eliminate seam visibility"""
    result = image.copy().astype(np.float32)

    # Find edges of textured regions
    edges = cv2.Canny(mask, 50, 150)

    # Dilate edges to create propagation zone
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (propagation_distance*2+1, propagation_distance*2+1))
    propagation_zone = cv2.dilate(edges, kernel, iterations=1)

    # Remove already textured areas from propagation zone
    propagation_zone = propagation_zone & (mask == 0)

    if np.any(propagation_zone > 0):
        # For each pixel in propagation zone, find nearest textured pixel
        prop_coords = np.where(propagation_zone > 0)
        texture_coords = np.where(mask > 0)

        if len(texture_coords[0]) > 0:
            # Use KD-tree for efficient nearest neighbor search

            # Build tree of textured pixel coordinates
            tree = cKDTree(np.column_stack(texture_coords))

            # Find nearest textured pixel for each propagation pixel
            distances, indices = tree.query(np.column_stack(prop_coords), k=3)

            # Weighted average of 3 nearest neighbors
            for i, (y, x) in enumerate(zip(prop_coords[0], prop_coords[1])):
                if distances[i, 0] <= propagation_distance:
                    weights = 1.0 / (distances[i] + 1e-6)  # Inverse distance weighting
                    weights /= weights.sum()

                    # Get colors from nearest neighbors
                    neighbor_colors = np.zeros((3, 3))  # 3 neighbors, 3 channels
                    for j, idx in enumerate(indices[i]):
                        ny, nx = texture_coords[0][idx], texture_coords[1][idx]
                        neighbor_colors[j] = result[ny, nx]

                    # Weighted average
                    propagated_color = np.sum(neighbor_colors * weights.reshape(-1, 1), axis=0)
                    result[y, x] = propagated_color

    return result.astype(np.uint8)


def gaussian_seam_smoothing(image: np.ndarray, mask: np.ndarray,
                          smooth_radius: int = 3) -> np.ndarray:
    """Apply Gaussian smoothing near seam edges"""
    result = image.copy()

    # Find edges of textured regions
    edges = cv2.Canny(mask, 50, 150)

    # Create smoothing zone around edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (smooth_radius*2+1, smooth_radius*2+1))
    smooth_zone = cv2.dilate(edges, kernel, iterations=1)

    # Apply Gaussian blur to the entire image
    blurred = cv2.GaussianBlur(image, (smooth_radius*2+1, smooth_radius*2+1), 0)

    # Blend original and blurred in smooth zone
    alpha = 0.6  # Blending factor
    smooth_mask = (smooth_zone > 0) & (mask > 0)

    if np.any(smooth_mask):
        result[smooth_mask] = (alpha * blurred[smooth_mask] +
                             (1-alpha) * result[smooth_mask]).astype(np.uint8)

    return result


def remove_white_seam_artifacts(image: np.ndarray) -> np.ndarray:
    """Dilation, thin-gap filling, edge color propagation and smoothing, each on the updated texture mask."""
    texture_mask = detect_texture_mask(image)
    result = dilate_texture_edges(image.copy(), texture_mask, iterations=2, kernel_size=3)
    texture_mask = detect_texture_mask(result)
    result = smart_gap_filling(result, texture_mask, max_gap_size=8)
    texture_mask = detect_texture_mask(result)
    result = edge_color_propagation(result, texture_mask, propagation_distance=4)
    texture_mask = detect_texture_mask(result)
    return gaussian_seam_smoothing(result, texture_mask, smooth_radius=2)


def main():
    parser = argparse.ArgumentParser(description='Extract the coarse texture map in the sewing-pattern UV space')
    parser.add_argument('--obj_path', required=True, help='simulated garment mesh with sewing-pattern UVs')
    parser.add_argument('--glb_path', required=True, help='textured mesh from project_multiview')
    parser.add_argument('--template_path', required=True, help='UV template (editable_template.png): output size')
    parser.add_argument('--output_path', required=True, help='Output path')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of processes (default: auto)')
    args = parser.parse_args()

    data = load_meshes_and_extract_data(args.obj_path, args.glb_path)
    template_size = Image.open(args.template_path).size
    face_mapping = create_face_mapping(data, args.num_processes or max(1, os.cpu_count() - 2))
    output_array = create_texture_mapping_barycentric(data, face_mapping, template_size)
    Image.fromarray(remove_white_seam_artifacts(output_array)).save(args.output_path)


if __name__ == '__main__':
    main()
