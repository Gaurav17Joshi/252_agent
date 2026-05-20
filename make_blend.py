"""
Run with:
    blender --background --python make_blend.py

Imports scene_meshes.glb, snaps objects to a floor plane, saves as scene.blend.
"""

import bpy
import mathutils
import os
import sys

# Optional CLI args after --: blender --background --python make_blend.py -- input.glb output.blend
_here = os.path.dirname(__file__)
_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
_pos  = [a for a in _args if not a.startswith("--")]
SCENE_GLB    = _pos[0] if len(_pos) > 0 else os.path.join(_here, "scene_meshes.glb")
OUTPUT_BLEND = _pos[1] if len(_pos) > 1 else os.path.join(_here, "scene.blend")
SNAP_TO_FLOOR = "--no-snap" not in _args
_z_arg   = next((a for a in _args if a.startswith("--z-offset=")), None)
Z_OFFSET = float(_z_arg.split("=")[1]) if _z_arg else 0.0
_sep_arg      = next((a for a in _args if a.startswith("--separate=")), None)
SEPARATE_ITERS = int(_sep_arg.split("=")[1]) if _sep_arg else 0  # 0 = disabled

# ── Clear default scene ───────────────────────────────────────────────────────
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ── Import the scene GLB ──────────────────────────────────────────────────────
bpy.ops.import_scene.gltf(filepath=SCENE_GLB)
print(f"Imported: {SCENE_GLB}")

meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
print(f"Mesh objects: {[o.name for o in meshes]}")

# ── Per-object snap: slide each object until its bottom touches Z = 0 ─────────
if SNAP_TO_FLOOR:
    for obj in meshes:
        corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
        obj_min_z = min(v.z for v in corners)
        obj.location.z -= obj_min_z
        print(f"  {obj.name}: shifted by {-obj_min_z:.4f}")
else:
    print("  Floor snap skipped (--no-snap)")

if Z_OFFSET != 0.0:
    for obj in meshes:
        obj.location.z += Z_OFFSET
    print(f"  Applied z-offset: {Z_OFFSET:+.4f} m to all objects")

# ── Bounding-box separation (--separate=N) ────────────────────────────────────
# For each overlapping pair, push them apart along the axis of minimum penetration
# (smallest overlap dimension). For a box sitting on top of another that's Z —
# the correct direction. Repeats up to N iterations until no overlaps remain.
if SEPARATE_ITERS > 0:
    bpy.context.view_layer.update()
    total_moved = 0
    for iteration in range(SEPARATE_ITERS):
        moved = False
        for i in range(len(meshes)):
            for j in range(i + 1, len(meshes)):
                a, b = meshes[i], meshes[j]
                ca = [a.matrix_world @ mathutils.Vector(c) for c in a.bound_box]
                cb = [b.matrix_world @ mathutils.Vector(c) for c in b.bound_box]
                mina = [min(v[k] for v in ca) for k in range(3)]
                maxa = [max(v[k] for v in ca) for k in range(3)]
                minb = [min(v[k] for v in cb) for k in range(3)]
                maxb = [max(v[k] for v in cb) for k in range(3)]
                overlaps = [min(maxa[k], maxb[k]) - max(mina[k], minb[k]) for k in range(3)]
                if all(o > 1e-4 for o in overlaps):
                    moved = True
                    total_moved += 1
                    ax = overlaps.index(min(overlaps))   # axis of least penetration
                    gap = overlaps[ax] / 2 + 0.001       # half overlap + 1 mm clearance
                    ca_k = (mina[ax] + maxa[ax]) / 2
                    cb_k = (minb[ax] + maxb[ax]) / 2
                    sign = 1.0 if cb_k >= ca_k else -1.0
                    if ax == 0:
                        a.location.x -= sign * gap;  b.location.x += sign * gap
                    elif ax == 1:
                        a.location.y -= sign * gap;  b.location.y += sign * gap
                    else:
                        a.location.z -= sign * gap;  b.location.z += sign * gap
                    bpy.context.view_layer.update()
        if not moved:
            print(f"  Separation converged in {iteration + 1} iterations ({total_moved} pair moves)")
            break
    else:
        print(f"  Separation: reached max {SEPARATE_ITERS} iterations ({total_moved} pair moves)")

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
