#!/usr/bin/env python3
"""
Render a simulated garment mesh without the body (front, back, right, left; orthographic, as in Hunyuan3D), for the
multi-view projection.
"""

import os
import platform
if platform.system() == 'Linux':
    os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import trimesh
import pyrender
from PIL import Image

def rotate_matrix_y(matrix, angle_deg):
    """Rotate matrix around Y-axis"""
    rotation_angle = angle_deg * (np.pi / 180)
    rotation_matrix = np.array([
        [np.cos(rotation_angle), 0, np.sin(rotation_angle), 0],
        [0, 1, 0, 0],
        [-np.sin(rotation_angle), 0, np.cos(rotation_angle), 0],
        [0, 0, 0, 1]
    ])
    return np.dot(rotation_matrix, matrix)

def rotate_matrix_x(matrix, angle_deg):
    """Rotate matrix around X-axis"""
    rotation_angle = angle_deg * (np.pi / 180)
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [0, np.sin(rotation_angle), np.cos(rotation_angle), 0],
        [0, 0, 0, 1]
    ])
    return np.dot(rotation_matrix, matrix)

def create_camera_hunyuan_orthographic(pyrender_body_mesh, pyrender_garment_mesh, scene, side, camera_location=None):
    """Create camera with Hunyuan3D orthographic projection"""

    # Hunyuan3D orthographic camera setup
    # ortho_scale = 1.2, bounds = [-0.6, 0.6] for left/right/bottom/top
    ortho_scale = 1.2
    xmag = ortho_scale * 0.5  # # 0.5 for all experiments
    ymag = ortho_scale * 0.5  # # 0.5 for all experiments
    
    # Get the size of the 3D bounding box (x, y, z directions)
    bbox_min = pyrender_garment_mesh.bounds[0]
    bbox_max = pyrender_garment_mesh.bounds[1]
    bbox_size = bbox_max - bbox_min  # This is the size along x, y, z
    
    # Move camera distance further back if y bound is larger than a threshold
    if bbox_size[1] > 1.4:
        xmag = ortho_scale * 0.75  # 0.75 for all experiments
        ymag = ortho_scale * 0.75  # 0.75 for all experiments
    elif bbox_size[1] > 1.19:
        xmag = ortho_scale * 0.6  # 0.6 for all experiments
        ymag = ortho_scale * 0.6  # 0.6 for all experiments

    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag, znear=0.1, zfar=100.0)

    if camera_location is None:
        # Hunyuan3D uses fixed camera distance of 1.45
        camera_distance = 1.45

        # Calculate the bounding box center of the mesh
        bounding_box_center = pyrender_body_mesh.bounds.mean(axis=0)

        camera_location = bounding_box_center.copy()
        camera_location[-1] += camera_distance  # Move camera back along Z-axis

    # Calculate the camera pose
    camera_pose = np.array([
        [1.0, 0.0, 0.0, camera_location[0]],
        [0.0, 1.0, 0.0, camera_location[1]],
        [0.0, 0.0, 1.0, camera_location[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Apply rotations (same as original)
    if side == 'back':
        camera_pose = rotate_matrix_y(camera_pose, 180)
    elif side == 'right':
        # Viewer's right (wearer's left): orbit +90° → camera at +X, looks −X
        camera_pose = rotate_matrix_y(camera_pose, 90)
    elif side == 'left':
        # Viewer's left (wearer's right): orbit −90° → camera at −X, looks +X
        camera_pose = rotate_matrix_y(camera_pose, -90)

    # Set camera's pose in the scene
    scene.add(camera, pose=camera_pose)

def create_lights_consistent(scene, intensity=80.0):
    """Same lights as the body+garment render (GarmentCodeRC pythonrender.py)"""
    light_positions = [
        np.array([1.60614, 1.5341, 1.23701]),
        np.array([1.31844, 1.92831, -2.52238]),
        np.array([-2.80522, 1.2594, 2.34624]),
        np.array([0.160261, 1.81789, 3.52215]),
        np.array([-2.65752, 1.41194, -1.26328])
    ]
    
    for position in light_positions:
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        light_pose = np.eye(4)
        light_pose[:3, 3] = position
        scene.add(light, pose=light_pose)

def create_scaled_body_mesh_for_positioning(garment_mesh):
    """A coarse body-shaped mesh matching the garment's scale and position (only used to place the camera)"""
    # Get garment bounds to understand the coordinate system
    garment_bounds = garment_mesh.bounds
    garment_center = garment_bounds.mean(axis=0)
    garment_height = garment_bounds[1][1] - garment_bounds[0][1]  # Y dimension (height)

    # Create body mesh that matches this scale and position
    # Typical human proportions scaled to match garment height
    body_height = garment_height * 1.1  # Body slightly taller than garment
    body_width = body_height * 0.25     # Typical body width ratio
    
    # Position body to match garment center
    body_center_y = garment_center[1]
    body_bottom = body_center_y - body_height/2
    body_top = body_center_y + body_height/2
    
    # Create body vertices that match garment scale and position
    body_vertices = np.array([
        # Lower body (around garment bottom)
        [-body_width, body_bottom, -body_width/3],
        [body_width, body_bottom, -body_width/3],
        [-body_width, body_bottom, body_width/3],
        [body_width, body_bottom, body_width/3],
        
        # Mid body (around garment center)
        [-body_width*0.7, body_center_y, -body_width/4],
        [body_width*0.7, body_center_y, -body_width/4],
        [-body_width*0.7, body_center_y, body_width/4],
        [body_width*0.7, body_center_y, body_width/4],
        
        # Upper body (around garment top)
        [-body_width*0.6, body_top*0.9, -body_width/5],
        [body_width*0.6, body_top*0.9, -body_width/5],
        [-body_width*0.6, body_top*0.9, body_width/5],
        [body_width*0.6, body_top*0.9, body_width/5],
        
        # Head
        [0.0, body_top, 0.0]
    ])
    
    # Create simple faces for the body mesh
    body_faces = np.array([
        # Bottom face
        [0, 1, 2], [1, 2, 3],
        # Connect bottom to mid
        [0, 1, 4], [1, 4, 5], [0, 2, 4], [2, 4, 6], 
        [1, 3, 5], [3, 5, 7], [2, 3, 6], [3, 6, 7],
        # Connect mid to top  
        [4, 5, 8], [5, 8, 9], [4, 6, 8], [6, 8, 10],
        [5, 7, 9], [7, 9, 11], [6, 7, 10], [7, 10, 11],
        # Connect top to head
        [8, 9, 12], [8, 10, 12], [9, 11, 12], [10, 11, 12]
    ])
    
    # Center the body on the garment's X and Z coordinates
    body_vertices[:, 0] += garment_center[0]  # X offset
    body_vertices[:, 2] += garment_center[2]  # Z offset

    body_mesh = trimesh.Trimesh(vertices=body_vertices, faces=body_faces)
    
    body_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.0, 0.0, 0.0, 1.0),
        metallicFactor=0.658,
        roughnessFactor=0.5
    )
    
    pyrender_body_mesh = pyrender.Mesh.from_trimesh(body_mesh, material=body_material)
    return pyrender_body_mesh, body_mesh

def load_garment_mesh_only(garment_mesh_path):
    """Load garment mesh with materials for rendering"""
    
    # Load garment mesh
    garment_mesh = trimesh.load_mesh(str(garment_mesh_path))
    garment_mesh.vertices = garment_mesh.vertices / 100  # Scale to meters
    
    # same material handling as the body+garment render
    if hasattr(garment_mesh.visual, 'material'):
        material = garment_mesh.visual.material.to_pbr()
        material.baseColorFactor = [1.0, 1.0, 1.0, 1.0]
        material.doubleSided = True
        if getattr(material, 'baseColorTexture', None) is not None:
            # remove transparency: white background behind the texture
            white_back = Image.new('RGBA', material.baseColorTexture.size, color=(255, 255, 255, 255))
            white_back.paste(material.baseColorTexture)
            material.baseColorTexture = white_back.convert('RGB')
        garment_mesh.visual.material = material
    else:
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.8, 0.8, 0.8, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.5,
            doubleSided=True
        )
        garment_mesh.visual.material = material
    
    pyrender_garment_mesh = pyrender.Mesh.from_trimesh(garment_mesh, smooth=True)
    return pyrender_garment_mesh, garment_mesh

def render_garment_only(garment_mesh_path, output_path, side='front', resolution=(1080, 1080)):
    """Render garment-only view with Hunyuan3D orthographic projection"""
    scene = pyrender.Scene(bg_color=(1., 1., 1., 0.))  # transparent background

    # Load garment mesh
    pyrender_garment_mesh, garment_trimesh = load_garment_mesh_only(garment_mesh_path)

    # the body mesh is not added to the scene, only used to place the camera
    pyrender_body_mesh, _ = create_scaled_body_mesh_for_positioning(garment_trimesh)
    scene.add(pyrender_garment_mesh)
    create_camera_hunyuan_orthographic(pyrender_body_mesh, pyrender_garment_mesh, scene, side, camera_location=None)
    create_lights_consistent(scene, intensity=80.0)

    width, height = resolution
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    Image.fromarray(color).save(output_path, "PNG")
    renderer.delete()
    return output_path

def render_garment_only_views(garment_mesh_path, output_dir, garment_name, resolution=(1080, 1080)):
    """Render front, back, right, and left garment-only views with Hunyuan3D orthographic projection"""

    os.makedirs(output_dir, exist_ok=True)
    # right = viewer's right (wearer's left), left = viewer's left (wearer's right)
    results = {}
    for side in ('front', 'back', 'right', 'left'):
        path = os.path.join(output_dir, f"{garment_name}_render_garment_only_{side}.png")
        results[side] = render_garment_only(garment_mesh_path, path, side, resolution)
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python render_garment_only.py <garment_mesh.obj> <output_dir> <garment_name>")
        sys.exit(1)
    
    garment_mesh_path = sys.argv[1]
    output_dir = sys.argv[2] 
    garment_name = sys.argv[3]
    
    render_garment_only_views(garment_mesh_path, output_dir, garment_name)