"""
Assemble a multi-object mesh scene from objects/ folder.

Each object_{i}.glb  — trimesh mesh with vertex colors (local y-up space)
Each object_{i}_pose.pt — rotation (wxyz quaternion), translation, scale

Produces scene_meshes.glb — import into Blender via File > Import > glTF 2.0
"""

import glob
import sys
import numpy as np
import torch
import trimesh

# Optional CLI args: python make_scene_mesh.py <objects_dir> <output.glb>
OBJECTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "objects"
OUTPUT_GLB  = sys.argv[2] if len(sys.argv) > 2 else "scene_meshes.glb"

# The z→y rotation baked into every glb by to_glb():
#   glb_verts = local_verts @ Z2Y
Z2Y = np.array([[1,  0,  0],
                [0,  0, -1],
                [0,  1,  0]], dtype=np.float64)


def quat_to_matrix(q):
    """
    q = [w, x, y, z]  (PyTorch3D convention)
    Returns the standard column-vector rotation matrix R.
    PyTorch3D compose_transform / transform_points does p @ R (right-multiply),
    which is what we want here.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [    2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def place_mesh(glb, rotation_q, translation, scale):
    """
    Applies the same transform as make_scene() does for Gaussians:
        p_scene = (p_local * scale) @ R(rotation) + translation

    glb.vertices are in local y-up space (Z2Y already applied by to_glb),
    so we undo Z2Y first, apply the pose, then re-apply Z2Y for export.
    """
    verts  = np.array(glb.vertices, dtype=np.float64)
    faces  = np.array(glb.faces)
    colors = np.array(glb.visual.vertex_colors)  # RGBA uint8

    # 1. Undo Z2Y → back to local z-up (the space the pose lives in)
    verts_local = verts @ Z2Y.T

    # 2. Apply pose: scale → rotate → translate  →  result is world y-up
    R = quat_to_matrix(rotation_q)
    verts_world = (verts_local * scale) @ R + translation

    # No second Z2Y: the pose already maps local z-up → world y-up (GLB native)
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
    mesh.visual.vertex_colors = colors
    return mesh


# ── Load and place each object ────────────────────────────────────────────────
glb_files = sorted(glob.glob(f"{OBJECTS_DIR}/object_*.glb"))
print(f"Found {len(glb_files)} objects in '{OBJECTS_DIR}'\n")

scene = trimesh.Scene()

for glb_path in glb_files:
    idx = glb_path.split("object_")[1].split(".")[0]
    pose_path = f"{OBJECTS_DIR}/object_{idx}_pose.pt"

    pose        = torch.load(pose_path, map_location="cpu", weights_only=False)
    rotation_q  = pose["rotation"].squeeze().numpy()
    translation = pose["translation"].squeeze().numpy()
    scale       = pose["scale"].squeeze().numpy()

    glb = trimesh.load(glb_path, force="mesh")

    print(f"object_{idx}")
    print(f"  verts / faces   : {np.array(glb.vertices).shape[0]} / {np.array(glb.faces).shape[0]}")
    print(f"  rotation (wxyz) : {rotation_q.round(4)}")
    print(f"  translation     : {translation.round(4)}")
    print(f"  scale           : {scale.round(4)}")

    mesh = place_mesh(glb, rotation_q, translation, scale)
    scene.add_geometry(mesh, node_name=f"object_{idx}")
    print(f"  → placed in scene\n")

out_path = OUTPUT_GLB
scene.export(out_path)
print(f"Saved: {out_path}")
print("Import into Blender: File > Import > glTF 2.0 (.glb/.gltf)")
