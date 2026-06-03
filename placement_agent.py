"""
Iterative placement agent — uses GPT-5.4-mini vision to fix object placement.

Usage:
    python placement_agent.py
    python placement_agent.py --blend scene.blend --glb scene_meshes.glb --max-iter 3

The agent:
  1. Renders the current scene.blend with EEVEE.
  2. Sends rendered image + original reference photo + object bboxes + object2.json
     to GPT-5.4-mini.
  3. GPT returns per-object placement rules (z=0 / drop / passive / useless).
  4. Re-runs make_blend.py with those rules as a --config file → new scene.blend.
  5. Repeats until GPT is satisfied or max_iter (default 3) reached.

Prerequisites:
    pip install openai
    export OPENAI_API_KEY=sk-...
    BLENDER_BIN env var: path to blender executable (default: "blender")
"""

import argparse
import base64
import json
import os
import subprocess
import tempfile

from openai import OpenAI

HERE          = os.path.dirname(os.path.abspath(__file__))
BLENDER       = os.environ.get("BLENDER_BIN", "blender")
BLEND_FILE    = os.path.join(HERE, "scene.blend")
SCENE_GLB     = os.path.join(HERE, "scene_meshes.glb")
REF_IMAGE     = os.path.join(HERE, "objects_2_masks", "image.png")
OBJ_DESC_JSON = os.path.join(HERE, "object2.json")
INFO_SCRIPT   = os.path.join(HERE, "agent_get_scene_info.py")
RENDER_SCRIPT = os.path.join(HERE, "agent_render_frame.py")
MAKE_BLEND    = os.path.join(HERE, "make_blend.py")

MODEL = "gpt-5.4-mini"

# Fixed camera — same defaults as camera_agent.py, tuned for this scene
CAMERA = {
    "location": [0.0, -0.3, 0.4],
    "target":   [0.0, -1.15, 0.35],
    "lens":     15.0,
}

SYSTEM_PROMPT = """
You are a 3D scene placement assistant. A room was 3D-scanned and reconstructed in Blender.
You will receive:
  1. A rendered image of the current Blender scene (EEVEE).
  2. A reference photo of the original real-world room.
  3. A list of objects with their label, description, and 3D bounding boxes (metres; Z is up, floor = Z=0).

Your task: assign a placement rule to each object so the scene is physically correct —
objects rest on the floor or on each other, nothing floats in the air, nothing clips below the floor.

Available rules (assign one placement rule per object; passive is independent and can be combined):
  "z = 0"  → snap this object's lowest point to the floor. Use for objects that clearly sit on the floor.
  "drop"   → drop this object straight down until it lands on the highest surface below it (or the floor).
             Use for objects that should rest on top of another object.
  "useless"→ delete this object (scan noise, duplicate mesh, artifact).
  ""        → keep at current position (position already looks correct).

"passive" is a separate flag (true/false). Set it for objects that should not move during physics
simulation — heavy furniture, the floor, structural objects, human figures.

You will also receive an "overlaps" list — pairs of objects whose bounding boxes interpenetrate.
Each entry shows which two objects overlap and by how much (metres) on each axis.
Overlapping objects need their placement corrected: use "z = 0", "drop", or leave blank
so that after rules are applied the objects no longer intersect.

Decision hints:
  - bbox_min[2] < -0.05               → below the floor, must use "z = 0"
  - bbox_min[2] > 0.1, nothing below  → probably floating, use "z = 0" or "drop"
  - Object described as "topmost" / "on top" → use "drop"
  - Object described as "bottommost" / "bottom" → use "z = 0"
  - Human or child figures            → "z = 0" + passive: true
  - Appears in overlaps list          → needs a rule to resolve the interpenetration

Output valid JSON only — no markdown fences, no extra text:
{
  "objects": {
    "0": {"rule": "z = 0", "passive": false},
    "1": {"rule": "drop",  "passive": false},
    "2": {"rule": "z = 0", "passive": true},
    "3": {"rule": "",      "passive": false}
  },
  "satisfied": true,
  "reason": "one sentence explaining the decision"
}

"rule" must be exactly one of: "z = 0", "drop", "useless", "".

Be STRICT about "satisfied". Only set it to true if ALL of the following hold:
  - No object has bbox_min[2] < -0.02 (nothing below the floor)
  - No object is visibly floating (hanging in air with empty space below it)
  - No objects are interpenetrating (overlaps list is empty or trivially small < 0.01 m)
  - Stacked objects look physically stable in the rendered image
  - The layout matches the reference photo
If ANY of these conditions are violated, set satisfied: false.
When in doubt, set satisfied: false — it is better to iterate than to accept a broken scene.
""".strip()


def blender_run(script, extra_args):
    cmd = [BLENDER, "--background", "--python", script, "--"] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr


def get_scene_info(blend_file):
    stdout, stderr = blender_run(INFO_SCRIPT, [blend_file, "1"])
    for line in stdout.splitlines():
        if line.startswith("SCENE_INFO:"):
            return json.loads(line[len("SCENE_INFO:"):])
    raise RuntimeError(f"No SCENE_INFO from agent_get_scene_info.py.\nSTDERR:\n{stderr[-1000:]}")


def render_frame(blend_file, output_png):
    stdout, stderr = blender_run(
        RENDER_SCRIPT,
        [blend_file, json.dumps(CAMERA), output_png, "1", "BLENDER_EEVEE"],
    )
    if not os.path.exists(output_png):
        raise RuntimeError(f"Render produced no output.\nSTDOUT:{stdout}\nSTDERR:{stderr[-1000:]}")


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_config_text(llm_objects, obj_desc):
    """Convert LLM JSON output dict → make_blend.py --config file text."""
    lines = []
    for idx_str, props in sorted(llm_objects.items(), key=lambda x: int(x[0])):
        idx  = int(idx_str)
        key  = f"object_{idx}"
        name = obj_desc.get("objects", {}).get(key, {}).get("name", "object")
        parts = []
        rule  = props.get("rule", "")
        if rule:
            parts.append(rule)
        if props.get("passive"):
            parts.append("passive")
        lines.append(f"{idx}. {name}   {'   '.join(parts)}".rstrip())
    return "\n".join(lines)


def remake_blend(glb_file, blend_out, config_text):
    """Write config to a temp file and re-run make_blend.py."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(config_text)
        cfg_path = f.name
    try:
        stdout, stderr = blender_run(
            MAKE_BLEND,
            [glb_file, blend_out, f"--config={cfg_path}", "--separate=10"],
        )
        for line in stdout.splitlines():
            if line.strip():
                print(f"    {line}")
        if "Error" in stderr or "Traceback" in stderr:
            print(f"  STDERR (tail):\n{stderr[-500:]}")
    finally:
        os.unlink(cfg_path)


def ask_gpt(client, render_png, scene_info, obj_desc, iteration):
    """Send current render + reference image + scene data to GPT-5.4-mini."""
    # Build enriched object list (bbox + description)
    objects_payload = []
    for obj in scene_info["objects"]:
        blender_name = obj["name"]
        desc_entry   = obj_desc.get("objects", {}).get(blender_name, {})
        try:
            idx = int(blender_name.rsplit("_", 1)[-1])
        except ValueError:
            idx = -1
        objects_payload.append({
            "index":       idx,
            "id":          blender_name,
            "label":       desc_entry.get("name", "unknown"),
            "description": desc_entry.get("description", ""),
            "bbox_min":    obj["bbox_min"],
            "bbox_max":    obj["bbox_max"],
            "center":      obj["center"],
        })

    overlaps = scene_info.get("overlaps", [])
    overlap_text = (
        f"Overlapping object pairs ({len(overlaps)}):\n{json.dumps(overlaps, indent=2)}"
        if overlaps else "Overlapping object pairs: none detected."
    )

    user_text = (
        f"Iteration {iteration}.\n\n"
        f"Scene objects with bounding boxes:\n{json.dumps(objects_payload, indent=2)}\n\n"
        f"{overlap_text}\n\n"
        "Image 1: current Blender EEVEE render.\n"
        "Image 2: original reference photo of the real room.\n\n"
        "Assign placement rules so the scene matches the reference photo layout "
        "and no objects interpenetrate.\n\n"
        "Note: after your rules are applied, a horizontal separation pass automatically "
        "pushes apart any objects that still overlap in XY. So focus your rules on "
        "vertical placement (z=0, drop) — horizontal overlaps will be resolved automatically."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{encode_image(render_png)}",
                    "detail": "high",
                }},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{encode_image(REF_IMAGE)}",
                    "detail": "high",
                }},
            ]},
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=1024,
    )
    return json.loads(response.choices[0].message.content)


def run(blend_file, glb_file, max_iter):
    client = OpenAI()

    with open(OBJ_DESC_JSON) as f:
        obj_desc = json.load(f)

    print(f"Blend  : {blend_file}")
    print(f"GLB    : {glb_file}")
    print(f"Ref    : {REF_IMAGE}")
    print(f"Desc   : {OBJ_DESC_JSON}")
    print(f"Model  : {MODEL}")
    print(f"MaxIter: {max_iter}\n")

    for i in range(max_iter):
        print(f"── Iteration {i + 1}/{max_iter} " + "─" * 40)

        print("  Getting scene info …")
        scene_info = get_scene_info(blend_file)
        overlaps   = scene_info.get("overlaps", [])
        print(f"  {len(scene_info['objects'])} objects | scene center: {scene_info['scene_center']}")
        if overlaps:
            print(f"  Overlaps ({len(overlaps)}):")
            for ov in overlaps:
                print(f"    {ov['a']} ↔ {ov['b']}  depth={ov['depth']:.4f} m  xyz={ov['overlap_xyz']}")
        else:
            print("  Overlaps: none")

        render_png = os.path.join(HERE, f"_placement_render_{i}.png")
        print(f"  Rendering EEVEE frame → {render_png} …")
        render_frame(blend_file, render_png)

        print(f"  Querying {MODEL} …")
        result = ask_gpt(client, render_png, scene_info, obj_desc, i + 1)

        print(f"  Satisfied : {result['satisfied']}")
        print(f"  Reason    : {result['reason']}")
        print("  Rules:")
        for idx_str, props in sorted(result["objects"].items(), key=lambda x: int(x[0])):
            rule    = props.get("rule", "") or "(keep)"
            passive = " + passive" if props.get("passive") else ""
            print(f"    object_{idx_str}: {rule}{passive}")

        config_text = build_config_text(result["objects"], obj_desc)
        print(f"\n  Config written:\n{config_text}\n")

        print("  Re-running make_blend.py …")
        remake_blend(glb_file, blend_file, config_text)

        if result["satisfied"] and not overlaps:
            print(f"\nDone after {i + 1} iteration(s). scene.blend updated.")
            return
        elif result["satisfied"] and overlaps:
            print(f"  GPT satisfied but {len(overlaps)} overlap(s) remain — continuing.")

    print(f"\nMax iterations ({max_iter}) reached. Using last result.")
    print(f"Final blend: {blend_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Iteratively fix object placement in a Blender scene using GPT-5.4-mini."
    )
    parser.add_argument("--blend",    default=BLEND_FILE, help="Path to scene.blend (will be overwritten)")
    parser.add_argument("--glb",      default=SCENE_GLB,  help="Path to scene_meshes.glb (source of truth)")
    parser.add_argument("--max-iter", type=int, default=3, help="Max GPT iterations (default 3)")
    args = parser.parse_args()

    run(args.blend, args.glb, args.max_iter)
