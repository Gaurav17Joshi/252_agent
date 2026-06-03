"""
Blender helper — called by camera_agent.py.
Prints one line: SCENE_INFO:<json>

Usage (internal):
    blender --background --python agent_get_scene_info.py -- animation_scene2.blend [frame]
"""

import bpy
import json
import sys
import mathutils

_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
blend_file = _args[0]
frame      = int(_args[1]) if len(_args) > 1 else 1

bpy.ops.wm.open_mainfile(filepath=blend_file)
scene = bpy.context.scene
scene.frame_set(frame)
bpy.context.view_layer.update()

objects = []
for obj in bpy.context.scene.objects:
    if obj.type != 'MESH' or obj.name == 'Floor':
        continue
    corners  = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    center   = [round(sum(v[k] for v in corners) / 8, 3) for k in range(3)]
    bbox_min = [round(min(v[k] for v in corners), 3) for k in range(3)]
    bbox_max = [round(max(v[k] for v in corners), 3) for k in range(3)]
    objects.append({
        "name":     obj.name,
        "center":   center,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    })

# Compute overall scene bounds
all_min = [min(o["bbox_min"][k] for o in objects) for k in range(3)]
all_max = [max(o["bbox_max"][k] for o in objects) for k in range(3)]
scene_center = [round((all_min[k] + all_max[k]) / 2, 3) for k in range(3)]

# Compute pairwise bounding-box overlaps
overlaps = []
for i in range(len(objects)):
    for j in range(i + 1, len(objects)):
        a, b = objects[i], objects[j]
        axes = []
        for k in range(3):
            lo = max(a["bbox_min"][k], b["bbox_min"][k])
            hi = min(a["bbox_max"][k], b["bbox_max"][k])
            axes.append(round(hi - lo, 4))
        if all(o > 0 for o in axes):
            overlaps.append({
                "a":           a["name"],
                "b":           b["name"],
                "overlap_xyz": axes,           # penetration per axis (metres)
                "depth":       round(min(axes), 4),  # smallest axis = separation needed
            })

output = {"objects": objects, "scene_center": scene_center,
          "scene_min": all_min, "scene_max": all_max, "overlaps": overlaps}
print("SCENE_INFO:" + json.dumps(output))
