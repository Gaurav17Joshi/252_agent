"""
Run with:
    blender --background --python render_mp4.py

Opens animation.blend, adds camera + lighting, renders frames 1-50 to animation.mp4 via EEVEE.
"""

import bpy
import mathutils
import os

ANIMATION_BLEND = os.path.join(os.path.dirname(__file__), "animation.blend")
OUTPUT_MP4      = os.path.join(os.path.dirname(__file__), "animation.mp4")

bpy.ops.wm.open_mainfile(filepath=ANIMATION_BLEND)
scene = bpy.context.scene

# ── Render settings ───────────────────────────────────────────────────────────
scene.render.engine       = 'BLENDER_EEVEE_NEXT'
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.frame_start         = 1
scene.frame_end           = 50

scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format              = 'MPEG4'
scene.render.ffmpeg.codec               = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'HIGH'
scene.render.filepath = OUTPUT_MP4

# ── Camera ────────────────────────────────────────────────────────────────────
cam_data = bpy.data.cameras.new("Camera")
cam_data.lens = 35
cam_obj = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

# Point camera at scene center (objects cluster around X≈0.4, Y≈-1.6, Z≈0.05)
cam_obj.location = mathutils.Vector((0.45, 0.5, 1.5))
target    = mathutils.Vector((0.4, -1.65, 0.05))
direction = target - cam_obj.location
cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

# ── Sun light ─────────────────────────────────────────────────────────────────
sun_data = bpy.data.lights.new("Sun", type='SUN')
sun_data.energy = 4.0
sun_obj = bpy.data.objects.new("Sun", sun_data)
scene.collection.objects.link(sun_obj)
sun_obj.rotation_euler = (0.6, 0.2, 0.8)

# ── World background ──────────────────────────────────────────────────────────
world = scene.world or bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value    = (0.8, 0.85, 0.9, 1.0)
bg.inputs["Strength"].default_value = 0.6

# ── Render ────────────────────────────────────────────────────────────────────
print(f"Rendering frames {scene.frame_start}–{scene.frame_end} → {OUTPUT_MP4}")
bpy.ops.render.render(animation=True)
print(f"Done. Saved: {OUTPUT_MP4}")
