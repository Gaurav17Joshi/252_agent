"""
Run with:
    blender --background --python make_blend.py

Imports scene_meshes.glb, snaps objects to a floor plane, saves as scene.blend.

--config=scene3.txt  per-object rules file.  Format (one line per object):
    <index>. <label>   label may contain:
        "useless" / "removed"  → delete the object
        "z = 0"                → snap this object's bottom to z = 0
        "passive"              → tag object["passive"] = True (used by animation scripts)
    Objects not in the file follow the global --snap / --no-snap setting.
"""

import bpy
import mathutils
import os
import sys


def parse_config(path):
    """Return dict {index: {'skip': bool, 'snap': bool, 'passive': bool}}."""
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dot = line.find(".")
            if dot == -1:
                continue
            try:
                idx = int(line[:dot])
            except ValueError:
                continue
            desc = line[dot + 1:].lower()
            cfg[idx] = {
                "skip":    "useless" in desc or "removed" in desc,
                "snap":    "z = 0" in desc,
                "passive": "passive" in desc,
                "drop":    "drop" in desc,
            }
    return cfg


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
_cfg_arg = next((a for a in _args if a.startswith("--config=")), None)
OBJ_CONFIG = parse_config(_cfg_arg.split("=", 1)[1]) if _cfg_arg else {}  # {idx: rules}

# ── Clear default scene ───────────────────────────────────────────────────────
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ── Import the scene GLB ──────────────────────────────────────────────────────
bpy.ops.import_scene.gltf(filepath=SCENE_GLB)
print(f"Imported: {SCENE_GLB}")

meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
print(f"Mesh objects: {[o.name for o in meshes]}")

# ── Apply per-object config: delete useless, snap z=0 objects, tag passives ───
def _obj_index(obj):
    """Extract integer index from names like 'object_10' or 'object_10.001'."""
    base = obj.name.split(".")[0]          # strip '.001' duplicates
    parts = base.rsplit("_", 1)
    try:
        return int(parts[-1])
    except ValueError:
        return None

if OBJ_CONFIG:
    to_delete = []
    for obj in meshes:
        idx = _obj_index(obj)
        rules = OBJ_CONFIG.get(idx, {})
        if rules.get("skip", False):
            to_delete.append(obj)
        else:
            if rules.get("passive", False):
                obj["passive"] = True
                print(f"  {obj.name}: tagged passive")
    # delete in batch
    bpy.ops.object.select_all(action='DESELECT')
    for obj in to_delete:
        obj.select_set(True)
    bpy.ops.object.delete()
    print(f"  Deleted {len(to_delete)} useless objects")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

# ── Helpers: bounding-box Z queries ──────────────────────────────────────────
def _bbox_corners(obj):
    return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]

def _snap_bottom(obj):
    obj.location.z -= min(v.z for v in _bbox_corners(obj))

def _bbox_min_z(obj):
    return min(v.z for v in _bbox_corners(obj))

def _drop_object(obj, all_meshes):
    """Move obj straight down until its lowest point lands on the highest surface below it, or z=0."""
    bpy.context.view_layer.update()
    corners      = _bbox_corners(obj)
    obj_min_z    = min(v.z for v in corners)
    obj_center_z = sum(v.z for v in corners) / 8
    obj_min_x    = min(v.x for v in corners);  obj_max_x = max(v.x for v in corners)
    obj_min_y    = min(v.y for v in corners);  obj_max_y = max(v.y for v in corners)

    best_top = 0.0  # default: floor at z=0
    for other in all_meshes:
        if other is obj:
            continue
        oc = _bbox_corners(other)
        if max(v.x for v in oc) < obj_min_x or min(v.x for v in oc) > obj_max_x:
            continue
        if max(v.y for v in oc) < obj_min_y or min(v.y for v in oc) > obj_max_y:
            continue
        if sum(v.z for v in oc) / 8 >= obj_center_z:
            continue  # skip objects whose center is above or level with this one
        best_top = max(best_top, max(v.z for v in oc))

    obj.location.z += best_top - obj_min_z

# ── Global snap: shift entire scene so its lowest point is at z=0 ─────────────
if SNAP_TO_FLOOR and meshes:
    bpy.context.view_layer.update()
    global_min_z = min(_bbox_min_z(obj) for obj in meshes)
    if abs(global_min_z) > 1e-4:
        for obj in meshes:
            obj.location.z -= global_min_z
        print(f"Global snap: shifted scene by {-global_min_z:+.4f} m (lowest point was z={global_min_z:.4f})")
elif not SNAP_TO_FLOOR and Z_OFFSET != 0.0:
    for obj in meshes:
        obj.location.z += Z_OFFSET
    print(f"z-offset: {Z_OFFSET:+.4f} m applied")

# ── Per-object z=0 overrides (run after global snap) ─────────────────────────
bpy.context.view_layer.update()
for obj in meshes:
    if OBJ_CONFIG.get(_obj_index(obj), {}).get("snap", False):
        _snap_bottom(obj)
        print(f"  {obj.name}: snapped to floor (z=0 rule)")

# ── Per-object drop: fall onto the highest surface below (bottom-up order) ───
bpy.context.view_layer.update()
drop_objs = sorted(
    [obj for obj in meshes if OBJ_CONFIG.get(_obj_index(obj), {}).get("drop", False)],
    key=_bbox_min_z,
)
for obj in drop_objs:
    bpy.context.view_layer.update()
    _drop_object(obj, meshes)
    print(f"  {obj.name}: dropped to landing surface")

# ── Z-offset on top of global snap ───────────────────────────────────────────
if SNAP_TO_FLOOR and Z_OFFSET != 0.0:
    bpy.context.view_layer.update()
    for obj in meshes:
        obj.location.z += Z_OFFSET
    print(f"  z-offset: {Z_OFFSET:+.4f} m applied on top of global snap")

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

    # ── Re-drop after separation: objects pushed apart horizontally may now float ─
    bpy.context.view_layer.update()
    for obj in sorted(drop_objs, key=_bbox_min_z):
        bpy.context.view_layer.update()
        _drop_object(obj, meshes)
        print(f"  {obj.name}: re-dropped after separation")

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
