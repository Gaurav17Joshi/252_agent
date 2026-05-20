"""
Run with:
    blender --background --python animation3.py

Fixes over animation2.py:
  3. More substeps → substeps_per_frame raised to 20.
  4. Zero-gravity pre-settle with high damping → SETTLE_FRAMES frames before the
     animation where gravity=0 and linear_damping=0.99. Contact forces nudge
     intersecting objects apart a tiny bit each frame without them flying away.
     At frame 1 gravity and normal damping are restored.

Reads scene.blend + physics.json, saves animation3.blend.
"""

import bpy
import json
import os
import sys
import mathutils

# Optional CLI args after --: blender --background --python animation3.py -- scene.blend physics.json out.blend
_here = os.path.dirname(__file__)
_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
SCENE_BLEND  = _args[0] if len(_args) > 0 else os.path.join(_here, "scene.blend")
PHYSICS_JSON = _args[1] if len(_args) > 1 else os.path.join(_here, "physics.json")
OUTPUT_BLEND = _args[2] if len(_args) > 2 else os.path.join(_here, "animation3.blend")

FLOOR_CLEARANCE = 0.005   # metres — lifts objects off floor at frame 1
N_KIN_FRAMES    = 8       # kinematic frames used to seed initial velocity
SETTLE_FRAMES   = 30      # pre-animation frames: zero gravity + high damping

# ── Load scene ────────────────────────────────────────────────────────────────
bpy.ops.wm.open_mainfile(filepath=SCENE_BLEND)
scene = bpy.context.scene

# ── Physics config ────────────────────────────────────────────────────────────
with open(PHYSICS_JSON) as f:
    config = json.load(f)

sim = config["simulation"]
scene.frame_start = sim["frame_start"]
scene.frame_end   = sim["frame_end"]
scene.render.fps  = sim["fps"]
dt = 1.0 / scene.render.fps

# ── Rigid body world ──────────────────────────────────────────────────────────
if scene.rigidbody_world is None:
    bpy.ops.rigidbody.world_add()
rbw = scene.rigidbody_world
rbw.time_scale         = 1.0
rbw.substeps_per_frame = sim.get("substeps_per_frame", 20)
rbw.solver_iterations  = sim.get("solver_iterations", 20)
rbw.point_cache.frame_start = scene.frame_start - SETTLE_FRAMES
rbw.point_cache.frame_end   = scene.frame_end

# ── Raise all objects slightly off the floor ──────────────────────────────────
meshes = [o for o in scene.objects if o.type == 'MESH' and o.name != 'Floor']
for obj in meshes:
    obj.location.z += FLOOR_CLEARANCE
print(f"Raised {len(meshes)} objects by {FLOOR_CLEARANCE*1000:.0f} mm to avoid floor penetration")

# ── Floor — passive ───────────────────────────────────────────────────────────
floor = bpy.data.objects.get("Floor")
if floor:
    bpy.context.view_layer.objects.active = floor
    floor.select_set(True)
    bpy.ops.rigidbody.object_add(type='PASSIVE')
    floor.select_set(False)
    rb = floor.rigid_body
    rb.collision_shape = 'MESH'
    rb.friction        = config["ground"]["friction"]
    rb.restitution     = config["ground"]["restitution"]
    print(f"Floor: friction={config['ground']['friction']}, restitution={config['ground']['restitution']}")

# ── Objects ───────────────────────────────────────────────────────────────────
settle_start = scene.frame_start - SETTLE_FRAMES

for obj_name, props in config["objects"].items():
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        print(f"Warning: {obj_name} not found, skipping")
        continue

    rb_type = 'PASSIVE' if props.get("fixed", False) else 'ACTIVE'

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.rigidbody.object_add(type=rb_type)
    obj.select_set(False)

    rb = obj.rigid_body
    rb.friction        = props["friction"]
    rb.restitution     = props["restitution"]
    rb.collision_shape = props.get("collision_shape", "CONVEX_HULL")

    if rb_type == 'ACTIVE':
        rb.mass = props["mass"]

        # Settle: extreme damping so objects barely move while contact forces
        # nudge them apart. Switch to normal damping at frame_start.
        bpy.context.preferences.edit.keyframe_new_interpolation_type = 'CONSTANT'
        scene.frame_set(settle_start)
        rb.linear_damping  = 0.99
        rb.angular_damping = 0.99
        rb.keyframe_insert(data_path="linear_damping")
        rb.keyframe_insert(data_path="angular_damping")

        scene.frame_set(scene.frame_start)
        rb.linear_damping  = props["linear_damping"]
        rb.angular_damping = props["angular_damping"]
        rb.keyframe_insert(data_path="linear_damping")
        rb.keyframe_insert(data_path="angular_damping")
        bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'

        vel = props.get("initial_linear_velocity", [0.0, 0.0, 0.0])
        if any(v != 0.0 for v in vel):
            pos0 = obj.location.copy()
            for f in range(N_KIN_FRAMES + 1):
                scene.frame_set(scene.frame_start + f)
                obj.location = mathutils.Vector((
                    pos0.x + vel[0] * f * dt,
                    pos0.y + vel[1] * f * dt,
                    pos0.z + vel[2] * f * dt,
                ))
                obj.keyframe_insert(data_path="location")
                rb.kinematic = (f < N_KIN_FRAMES)
                rb.keyframe_insert(data_path="kinematic")
            scene.frame_set(scene.frame_start)

    print(f"{obj_name} ({props.get('label','?')}): "
          f"type={rb_type}  friction={props['friction']}  "
          f"restitution={props['restitution']}"
          + (f"  mass={props['mass']}kg" if rb_type == 'ACTIVE' else ""))

# ── Zero-gravity during settle, normal gravity at frame 1 ────────────────────
bpy.context.preferences.edit.keyframe_new_interpolation_type = 'CONSTANT'
scene.frame_set(settle_start)
scene.gravity = mathutils.Vector((0.0, 0.0, 0.0))
scene.keyframe_insert(data_path="gravity")

scene.frame_set(scene.frame_start)
scene.gravity = mathutils.Vector((0.0, 0.0, -9.81))
scene.keyframe_insert(data_path="gravity")
bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'

scene.frame_set(settle_start)
print(f"Pre-settle: {SETTLE_FRAMES} frames at zero gravity + damping=0.99 "
      f"(frames {settle_start} → {scene.frame_start})")

# ── Bake ──────────────────────────────────────────────────────────────────────
print("\nBaking …")
try:
    with bpy.context.temp_override(scene=scene):
        bpy.ops.ptcache.bake_all(bake=True)
    print("Bake complete.")
except Exception as e:
    print(f"Headless bake failed ({e}).")
    print("Open animation3.blend → Scene > Rigid Body Cache > Bake All Dynamics.")

# ── Save ──────────────────────────────────────────────────────────────────────
bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND)
print(f"\nSaved: {OUTPUT_BLEND}")
