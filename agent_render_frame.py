"""
Blender helper — called by camera_agent.py.
Renders one Workbench frame with given camera params and saves to output_png.

Usage (internal):
    blender --background --python agent_render_frame.py -- \
        animation_scene2.blend '{"location":[0,0,1],"target":[0,-1,0],"lens":20}' \
        /tmp/out.png [frame]
"""

import bpy
import json
import sys
import mathutils

_args       = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
blend_file  = _args[0]
camera_json = _args[1]
output_png  = _args[2]
frame       = int(_args[3]) if len(_args) > 3 else 1
engine      = _args[4] if len(_args) > 4 else "BLENDER_EEVEE"

cam = json.loads(camera_json)

bpy.ops.wm.open_mainfile(filepath=blend_file)
scene = bpy.context.scene
scene.frame_set(frame)

scene.render.engine                     = engine
scene.render.resolution_x               = 1280
scene.render.resolution_y               = 720
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath                   = output_png

cam_data      = bpy.data.cameras.new("AgentCam")
cam_data.lens = cam["lens"]
cam_obj       = bpy.data.objects.new("AgentCam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

loc            = mathutils.Vector(cam["location"])
tgt            = mathutils.Vector(cam["target"])
cam_obj.location      = loc
cam_obj.rotation_euler = (tgt - loc).to_track_quat('-Z', 'Y').to_euler()

# Add sun + world background when using EEVEE (workbench has built-in lighting)
if engine != "BLENDER_WORKBENCH":
    sun_data        = bpy.data.lights.new("AgentSun", type='SUN')
    sun_data.energy = 4.0
    sun_obj         = bpy.data.objects.new("AgentSun", sun_data)
    scene.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (0.6, 0.2, 0.8)

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background") or world.node_tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value    = (0.8, 0.85, 0.9, 1.0)
    bg.inputs["Strength"].default_value = 0.5

bpy.ops.render.render(write_still=True)
print(f"RENDER_OK:{output_png}")
