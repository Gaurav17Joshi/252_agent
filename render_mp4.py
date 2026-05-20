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
scene.render.engine       = 'BLENDER_EEVEE'
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.frame_start         = 1
scene.frame_end           = 50

FRAMES_DIR = os.path.join(os.path.dirname(__file__), "frames", "")
os.makedirs(FRAMES_DIR, exist_ok=True)
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = FRAMES_DIR

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

# ── Render PNG sequence ───────────────────────────────────────────────────────
print(f"Rendering frames {scene.frame_start}–{scene.frame_end} → {FRAMES_DIR}")
bpy.ops.render.render(animation=True)
print("Frames done.")

# ── Stitch to MP4 with system ffmpeg ─────────────────────────────────────────
import subprocess
fps = scene.render.fps
cmd = [
    "ffmpeg", "-y",
    "-framerate", str(fps),
    "-i", os.path.join(FRAMES_DIR, "%04d.png"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "18",
    OUTPUT_MP4,
]
print("Stitching MP4 …", " ".join(cmd))
subprocess.run(cmd, check=True)
print(f"\nDone. Saved: {OUTPUT_MP4}")
