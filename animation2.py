"""
Run with:
    blender --background --python animation2.py

Fixes over animation.py:
  1. Floor pop-up → raises all objects 5 mm at start so convex hulls don't
     penetrate the floor, eliminating the solver "pop" on frame 1.
  2. Initial velocity → uses N_KIN_FRAMES of kinematic motion with LINEAR
     interpolation before handing off to physics. Blender infers the velocity
     from the finite difference of those positions, which is reliable.

Reads scene.blend + physics.json, saves animation2.blend.
"""

import bpy
import json
import os
import mathutils

SCENE_BLEND  = os.path.join(os.path.dirname(__file__), "scene.blend")
PHYSICS_JSON = os.path.join(os.path.dirname(__file__), "physics.json")
OUTPUT_BLEND = os.path.join(os.path.dirname(__file__), "animation2.blend")

FLOOR_CLEARANCE = 0.005   # metres — lifts objects off floor at frame 1
N_KIN_FRAMES    = 8       # kinematic frames used to seed initial velocity

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
rbw.substeps_per_frame = sim.get("substeps_per_frame", 10)
rbw.solver_iterations  = sim.get("solver_iterations", 20)
rbw.point_cache.frame_start = scene.frame_start
rbw.point_cache.frame_end   = scene.frame_end

# ── Fix 1: raise all objects slightly off the floor ───────────────────────────
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

# ── Objects — active ──────────────────────────────────────────────────────────
# Set LINEAR as default so every keyframe inserted below uses it
bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'

for obj_name, props in config["objects"].items():
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        print(f"Warning: {obj_name} not found, skipping")
        continue

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.rigidbody.object_add(type='ACTIVE')
    obj.select_set(False)

    rb = obj.rigid_body
    rb.mass            = props["mass"]
    rb.friction        = props["friction"]
    rb.restitution     = props["restitution"]
    rb.linear_damping  = props["linear_damping"]
    rb.angular_damping = props["angular_damping"]
    rb.collision_shape = props.get("collision_shape", "CONVEX_HULL")

    vel = props.get("initial_linear_velocity", [0.0, 0.0, 0.0])

    if any(v != 0.0 for v in vel):
        # Fix 2: kinematic motion over N_KIN_FRAMES frames with LINEAR
        # interpolation. Blender reads the velocity at handoff from the
        # finite difference of the last few kinematic positions.
        pos0 = obj.location.copy()

        for f in range(N_KIN_FRAMES + 1):
            scene.frame_set(scene.frame_start + f)
            obj.location = mathutils.Vector((
                pos0.x + vel[0] * f * dt,
                pos0.y + vel[1] * f * dt,
                pos0.z + vel[2] * f * dt,
            ))
            obj.keyframe_insert(data_path="location")
            rb.kinematic = (f < N_KIN_FRAMES)   # False on the last frame = hand off
            rb.keyframe_insert(data_path="kinematic")

        # Reset to frame_start for clean bake
        scene.frame_set(scene.frame_start)

    print(f"{obj_name} ({props.get('label','?')}): "
          f"mass={props['mass']}kg  friction={props['friction']}  "
          f"restitution={props['restitution']}  vel={vel}")

# ── Bake ──────────────────────────────────────────────────────────────────────
print("\nBaking …")
try:
    with bpy.context.temp_override(scene=scene):
        bpy.ops.ptcache.bake_all(bake=True)
    print("Bake complete.")
except Exception as e:
    print(f"Headless bake failed ({e}).")
    print("Open animation2.blend → Scene > Rigid Body Cache > Bake All Dynamics.")

# ── Save ──────────────────────────────────────────────────────────────────────
bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND)
print(f"\nSaved: {OUTPUT_BLEND}")
