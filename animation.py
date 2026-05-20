"""
Run with:
    blender --background --python animation.py

Reads scene.blend + physics.json, sets up rigid body simulation, saves animation.blend.
Open animation.blend in Blender and press Space/Play to watch, or go to
Scene > Rigid Body Cache > Bake All Dynamics to bake before rendering.
"""

import bpy
import json
import os

SCENE_BLEND  = os.path.join(os.path.dirname(__file__), "scene.blend")
PHYSICS_JSON = os.path.join(os.path.dirname(__file__), "physics.json")
OUTPUT_BLEND = os.path.join(os.path.dirname(__file__), "animation.blend")

# ── Load scene ────────────────────────────────────────────────────────────────
bpy.ops.wm.open_mainfile(filepath=SCENE_BLEND)
scene = bpy.context.scene

# ── Read physics config ───────────────────────────────────────────────────────
with open(PHYSICS_JSON) as f:
    config = json.load(f)

sim = config["simulation"]
scene.frame_start   = sim["frame_start"]
scene.frame_end     = sim["frame_end"]
scene.render.fps    = sim["fps"]
dt = 1.0 / scene.render.fps

# ── Set up rigid body world ───────────────────────────────────────────────────
if scene.rigidbody_world is None:
    bpy.ops.rigidbody.world_add()

rbw = scene.rigidbody_world
rbw.time_scale          = 1.0
rbw.substeps_per_frame  = sim.get("substeps_per_frame", 10)
rbw.solver_iterations   = sim.get("solver_iterations", 20)
rbw.point_cache.frame_start = scene.frame_start
rbw.point_cache.frame_end   = scene.frame_end

# ── Floor — passive rigid body ────────────────────────────────────────────────
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
    print(f"Floor: friction={rb.friction}, restitution={rb.restitution}")

# ── Objects — active rigid bodies ─────────────────────────────────────────────
for obj_name, props in config["objects"].items():
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        print(f"Warning: {obj_name} not found in scene, skipping")
        continue

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.rigidbody.object_add(type='ACTIVE')
    obj.select_set(False)

    rb = obj.rigid_body
    rb.mass             = props["mass"]
    rb.friction         = props["friction"]
    rb.restitution      = props["restitution"]
    rb.linear_damping   = props["linear_damping"]
    rb.angular_damping  = props["angular_damping"]
    rb.collision_shape  = props.get("collision_shape", "CONVEX_HULL")

    vel = props.get("initial_linear_velocity", [0.0, 0.0, 0.0])

    if any(v != 0.0 for v in vel):
        # Blender infers initial velocity from the object's motion during
        # kinematic (animated) frames just before physics takes over.
        # Frame start: kinematic=True, object at rest position.
        # Frame start+1: kinematic=False, object shifted by vel*dt — Blender
        # reads the implied velocity and hands off to the physics solver.
        pos0 = obj.location.copy()

        scene.frame_set(scene.frame_start)
        obj.location = pos0
        obj.keyframe_insert(data_path="location")
        rb.kinematic = True
        rb.keyframe_insert(data_path="kinematic")

        scene.frame_set(scene.frame_start + 1)
        obj.location = (
            pos0.x + vel[0] * dt,
            pos0.y + vel[1] * dt,
            pos0.z + vel[2] * dt,
        )
        obj.keyframe_insert(data_path="location")
        rb.kinematic = False
        rb.keyframe_insert(data_path="kinematic")

        obj.location = pos0  # reset; simulation takes over from frame_start+1

    print(f"{obj_name} ({props.get('label','?')}): "
          f"mass={props['mass']}kg  friction={props['friction']}  "
          f"restitution={props['restitution']}  vel={vel}")

# ── Bake physics ──────────────────────────────────────────────────────────────
print("\nBaking rigid body simulation …")
try:
    with bpy.context.temp_override(scene=scene):
        bpy.ops.ptcache.bake_all(bake=True)
    print("Bake complete.")
except Exception as e:
    print(f"Headless bake failed ({e}).")
    print("Open animation.blend → Scene > Rigid Body Cache > Bake All Dynamics.")

# ── Save ──────────────────────────────────────────────────────────────────────
bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND)
print(f"\nSaved: {OUTPUT_BLEND}")
