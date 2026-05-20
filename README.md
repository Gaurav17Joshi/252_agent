# 252_agent

Agentic pipeline that takes a room image, reconstructs 3D objects using SAM-3D Objects (Meta), assembles them into a Blender scene with correct poses, and runs a physics simulation driven by a JSON config. Intended as a base for LLM-controlled physics ("move the pouf toward the briefcase").

---

## Pipeline overview

```
Room image + masks
       │
       ▼
sam-3d-objects  (run once per scene, needs GPU)
       │  produces:  objects/object_{i}.glb
       │             objects/object_{i}_pose.pt
       ▼
make_scene_mesh.py   →   scene_meshes.glb
       ▼
make_blend.py        →   scene.blend        (floor-snapped, origins centred)
       ▼
animation.py  /  animation2.py   →   animation.blend / animation2.blend
       ▼
render_mp4.py        →   animation.mp4
```

---

## Scripts

### `make_scene_mesh.py`
Reads `objects/object_{i}.glb` and `objects/object_{i}_pose.pt`, applies each object's rotation/translation/scale (PyTorch3D convention), and exports the whole scene as `scene_meshes.glb`.

```bash
python make_scene_mesh.py
```

**Key detail:** the quaternion in `pose.pt` is `[w, x, y, z]` (PyTorch3D order). The pose transforms from local z-up object space to world y-up space using right-multiplication (`p @ R`), matching PyTorch3D's `compose_transform / transform_points`.

---

### `make_blend.py`
Imports `scene_meshes.glb` into Blender, snaps every object to the floor individually (per-object bounding-box bottom → Z=0), moves each object's origin to its geometry centre, adds a brown wood-coloured floor plane, and saves `scene.blend`.

```bash
blender --background --python make_blend.py
```

---

### `physics.json`
Defines physical properties for the simulation. Edit this to change object behaviour without touching the scripts.

```jsonc
{
  "simulation": { "fps": 24, "frame_start": 1, "frame_end": 200, ... },
  "ground":  { "friction": 0.6, "restitution": 0.05 },
  "objects": {
    "object_3": {
      "label": "pouf",
      "mass": 5.0,
      "friction": 0.5,
      "initial_linear_velocity": [2.5, 0.0, 0.0],  // m/s in Blender X
      ...
    }
  }
}
```

---

### `animation.py` (v1)
First version. Adds Blender rigid body physics from `physics.json` and bakes the simulation. Known issues: solver pop on frame 1 (convex hull penetrates floor), initial velocity not reliably applied.

```bash
blender --background --python animation.py
```

---

### `animation2.py` (v2 — recommended)
Fixes over v1:

| Issue | Fix |
|---|---|
| Objects pop upward on frame 1 | Raises all objects 5 mm before simulation starts, and sets collision margin to 1 mm to eliminate hull-floor overlap |
| Initial velocity not applied | Uses `N_KIN_FRAMES=8` kinematic frames with `LINEAR` interpolation; Blender infers velocity from finite difference at handoff |

```bash
blender --background --python animation2.py
# → saves animation2.blend
```

Open `animation2.blend` in Blender and press **Space** to preview, or go to **Scene > Rigid Body Cache > Bake All Dynamics** to re-bake.

---

### `render_mp4.py`
Opens `animation.blend` (edit `ANIMATION_BLEND` to point at `animation2.blend`), adds a camera and sun light, renders frames 1–50 with EEVEE to a PNG sequence, then stitches to `animation.mp4` using system `ffmpeg`.

```bash
blender --background --python render_mp4.py
```

---

## Prerequisites

| Tool | Purpose |
|---|---|
| Python 3.10+ | `make_scene_mesh.py` |
| `trimesh`, `torch`, `numpy` | scene assembly |
| [SAM-3D Objects](https://github.com/facebookresearch/sam-3d-objects) | generate per-object GLB + pose files |
| Blender 5.x | scene, physics, render |
| `ffmpeg` | stitch PNG frames into MP4 |

Install Python deps:
```bash
pip install trimesh torch numpy
```

---

## Coordinate system notes

- **SAM-3D local space**: z-up, y-forward (same convention used by the mesh decoder)
- **GLB / Blender import**: y-up (`to_glb()` applies `Z2Y` rotation)
- **Blender world**: z-up (Blender's GLB importer converts y-up → z-up on import)
- **Pose quaternion**: `[w, x, y, z]`, right-multiplies row vectors (`p @ R`), matching PyTorch3D

---

## Physics tuning notes

### Frame-1 pop artifact
Objects jump slightly on frame 1 because Blender wraps each convex hull in an invisible collision margin (default 4 cm). When meshes are placed close together or slightly intersecting (e.g. stacked objects from scan data), those margins overlap and the solver violently separates them on the first frame.

**Key finding — `substeps_per_frame` is the main lever:**
- Higher substeps (e.g. 20) = solver applies penetration correction more aggressively each substep → **bigger pop**
- Lower substeps (e.g. 5–10) = gentler correction → **smaller pop**
- `substeps_per_frame = 10` (default in `physics2.json`) gives an acceptable small pop for stacked-object scenes
- Do **not** raise substeps to fix the pop — it makes it worse

### Floor snapping vs. `--no-snap`
- `make_blend.py` default: snaps every object individually to z=0 (bottom of bounding box touches floor). Good for physics (no intersections), bad visually for stacked objects.
- `--no-snap --z-offset=<m>`: preserves relative positions from scan. Required when objects are stacked on top of each other. Objects may slightly intersect, causing a small frame-1 pop — this is acceptable.
- Find `z-offset` by trial: run `make_blend.py` with different values and open the `.blend` to check.

### Making an object fixed (not moved by physics)
Set `"fixed": true` in `physics.json` for that object. It becomes a `PASSIVE` rigid body — other objects collide with and bounce off it, but it never moves. Used for the kid in scene 2.

---

## Next steps / planned

- [ ] LLM agent layer: parse text commands → modify `physics.json` → re-run `animation2.py`
- [ ] Fix cushion clipping through floor (use `MESH` collision shape instead of `CONVEX_HULL`)
- [ ] Add collision margin fix to fully eliminate frame-1 pop
