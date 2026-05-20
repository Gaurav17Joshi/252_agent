"""
Debug script: run with different collision margins, bakes only first 5 frames.

Usage:
    blender --background --python animation2_debug.py -- scene2.blend physics2.json --margin=0.04
    blender --background --python animation2_debug.py -- scene2.blend physics2.json --margin=0.01
    blender --background --python animation2_debug.py -- scene2.blend physics2.json --margin=0.001

Output: animation2_debug.blend  (open in Blender, scrub frames 1-5 to inspect pop)
"""

import bpy
import json
import os
import sys
import mathutils

_here = os.path.dirname(__file__)
_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
_pos  = [a for a in _args if not a.startswith("--")]
SCENE_BLEND  = _pos[0] if len(_pos) > 0 else os.path.join(_here, "scene2.blend")
PHYSICS_JSON = _pos[1] if len(_pos) > 1 else os.path.join(_here, "physics2.json")
OUTPUT_BLEND = os.path.join(_here, "animation2_debug.blend")

_m_arg = next((a for a in _args if a.startswith("--margin=")), None)
COLLISION_MARGIN = float(_m_arg.split("=")[1]) if _m_arg else 0.04   # default = Blender's 4 cm

FLOOR_CLEARANCE = 0.005
DEBUG_FRAMES    = 5      # only bake this many frames

print(f"\n=== DEBUG: collision_margin={COLLISION_MARGIN*100:.1f} cm ===\n")

bpy.ops.wm.open_mainfile(filepath=SCENE_BLEND)
scene = bpy.context.scene

with open(PHYSICS_JSON) as f:
    config = json.load(f)

sim = config["simulation"]
scene.frame_start = sim["frame_start"]
scene.frame_end   = scene.frame_start + DEBUG_FRAMES
scene.render.fps  = sim["fps"]

if scene.rigidbody_world is None:
    bpy.ops.rigidbody.world_add()
rbw = scene.rigidbody_world
rbw.time_scale         = 1.0
rbw.substeps_per_frame = sim.get("substeps_per_frame", 10)
rbw.solver_iterations  = sim.get("solver_iterations", 20)
rbw.point_cache.frame_start = scene.frame_start
rbw.point_cache.frame_end   = scene.frame_end

meshes = [o for o in scene.objects if o.type == 'MESH' and o.name != 'Floor']
for obj in meshes:
    obj.location.z += FLOOR_CLEARANCE

floor = bpy.data.objects.get("Floor")
if floor:
    bpy.context.view_layer.objects.active = floor
    floor.select_set(True)
    bpy.ops.rigidbody.object_add(type='PASSIVE')
    floor.select_set(False)
    rb = floor.rigid_body
    rb.collision_shape  = 'MESH'
    rb.friction         = config["ground"]["friction"]
    rb.restitution      = config["ground"]["restitution"]
    rb.use_margin       = True
    rb.collision_margin = COLLISION_MARGIN

bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'

for obj_name, props in config["objects"].items():
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        continue

    rb_type = 'PASSIVE' if props.get("fixed", False) else 'ACTIVE'

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.rigidbody.object_add(type=rb_type)
    obj.select_set(False)

    rb = obj.rigid_body
    rb.friction         = props["friction"]
    rb.restitution      = props["restitution"]
    rb.collision_shape  = props.get("collision_shape", "CONVEX_HULL")
    rb.use_margin       = True
    rb.collision_margin = COLLISION_MARGIN

    if rb_type == 'ACTIVE':
        rb.mass            = props["mass"]
        rb.linear_damping  = props["linear_damping"]
        rb.angular_damping = props["angular_damping"]

    print(f"{obj_name} ({props.get('label','?')}): type={rb_type}  margin={COLLISION_MARGIN*100:.1f}cm")

print("\nBaking 5 frames …")
try:
    with bpy.context.temp_override(scene=scene):
        bpy.ops.ptcache.bake_all(bake=True)
    print("Bake complete.")
except Exception as e:
    print(f"Bake failed: {e}")

bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND)
print(f"\nSaved: {OUTPUT_BLEND}  (margin={COLLISION_MARGIN*100:.1f} cm)")
