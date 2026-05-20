"""
Run with:
    blender --background --python make_blend.py

Imports scene_meshes.glb, snaps objects to a floor plane, saves as scene.blend.
"""

import bpy
import mathutils
import os

SCENE_GLB    = os.path.join(os.path.dirname(__file__), "scene_meshes.glb")
OUTPUT_BLEND = os.path.join(os.path.dirname(__file__), "scene.blend")

# ── Clear default scene ───────────────────────────────────────────────────────
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ── Import the scene GLB ──────────────────────────────────────────────────────
bpy.ops.import_scene.gltf(filepath=SCENE_GLB)
print(f"Imported: {SCENE_GLB}")

meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
print(f"Mesh objects: {[o.name for o in meshes]}")

# ── Per-object snap: slide each object until its bottom touches Z = 0 ─────────
for obj in meshes:
    corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    obj_min_z = min(v.z for v in corners)
    obj.location.z -= obj_min_z
    print(f"  {obj.name}: shifted by {-obj_min_z:.4f}")

# ── Move each object's origin to its geometry center ──────────────────────────
# All world positions are baked into mesh vertices; origin was sitting at (0,0,z).
# Physics and viewport tools expect the origin to be at the object's actual center.
bpy.ops.object.select_all(action='DESELECT')
bpy.context.view_layer.update()
for obj in meshes:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.select_set(False)
    print(f"  {obj.name}: origin → {tuple(round(v, 3) for v in obj.location)}")

# ── Keep Blender's auto-generated materials (they already have vertex colours) ─
# Nothing to do — GLB import wires vertex colours correctly.

# ── Add floor plane ───────────────────────────────────────────────────────────
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
floor = bpy.context.active_object
floor.name = "Floor"

floor_mat = bpy.data.materials.new(name="Floor_mat")
floor_mat.use_nodes = True
bsdf = floor_mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.38, 0.22, 0.08, 1.0)  # warm wood brown
bsdf.inputs["Roughness"].default_value = 0.65
bsdf.inputs["Specular IOR Level"].default_value = 0.3
floor.data.materials.append(floor_mat)

# ── Save ─────────────────────────────────────────────────────────────────────
bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND)
print(f"\nSaved: {OUTPUT_BLEND}")
