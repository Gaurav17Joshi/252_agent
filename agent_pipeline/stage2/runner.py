"""
stage2/runner.py

Stage 2 Runner — Physical Reasoning & Simulation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Orchestrates Stage 2:

  For each scene object
        ↓
  MaterialClassificationAgent   (vision — one annotated image per object)
        ↓  (all results merged)
  material_map.json             (consolidated output written to run_directory)
        ↓
  ForceInferenceAgent
        ↓
  BlenderScriptExporter  →  simulation.py

Expected scene object schema (from Stage 1)
───────────────────────────────────────────
{
  "label":                 "glass_bottle",
  "annotated_image_path":  "run/<id>/annotated/glass_bottle.png",
  ...                      (other Stage-1 fields, e.g. mesh_path, bbox)
}

The field ``annotated_image_path`` must be present on every object.
It should point to an image where only that object is visually highlighted
(e.g. coloured bounding box, mask, or crop).

Output — material_map.json
──────────────────────────
{
  "material_map": {
    "glass_bottle": {
      "material_type": "rigid",
      "sub_type":      "glass",
      "params": {
        "mass":        0.5,
        "friction":    0.4,
        "restitution": 0.05
      }
    },
    ...
  }
}
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "..")

from utils.shared import get_logger, save_state
from stage2.material_agent import MaterialClassificationAgent
from stage2.force_agent import ForceInferenceAgent
from stage2.blender_exporter import BlenderScriptExporter
from stage2.extract_objects import build_scene_from_directory
from stage2.extract_objects import _build_test_scene

_log = get_logger("stage2.runner")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _classify_all_objects(
    objects: list[dict],
    prompt: str,
    run_directory: Path,
) -> dict[str, dict]:
    """
    Iterate over every scene object, call the MaterialClassificationAgent
    with its annotated image, and return a consolidated material_map dict.

    Each object dict must have:
      - "label"                (str)
      - "annotated_image_path" (str | Path)

    Failed classifications are logged and skipped so a single bad object
    does not abort the whole stage.
    """
    agent = MaterialClassificationAgent()
    material_map: dict[str, dict] = {}

    for idx, obj in enumerate(objects, start=1):
        label = obj.get("object_label", f"object_{idx}")
        img_path_raw = obj.get("path")

        if not img_path_raw:
            _log.warning(
                "Object '%s' has no annotated_image_path — skipping.",
                label,
            )
            continue

        img_path = Path(img_path_raw)

        if not img_path.exists():
            _log.warning(
                "Annotated image not found for '%s': %s — skipping.",
                label,
                img_path,
            )
            continue

        _log.info(
            "  [%d/%d] Classifying '%s' from %s",
            idx,
            len(objects),
            label,
            img_path.name,
        )

        try:
            spec = agent.classify_object(
                annotated_image_path=img_path,
                object_label=label,
                simulation_prompt=prompt,
            )
            material_map[label] = spec

        except Exception as exc:  # noqa: BLE001
            _log.error(
                "Failed to classify '%s': %s",
                label,
                exc,
            )

    return material_map


def _write_material_map(
    material_map: dict[str, dict],
    run_directory: Path,
) -> Path:
    """
    Persist the consolidated material map as ``material_map.json``
    inside *run_directory* and return the file path.

    Uses save_state from shared utils so the file lands in the same
    location as all other stage artefacts, but also writes a
    pretty-printed standalone copy for human inspection.
    """
    save_state(
        run_directory,
        "material_map",
        {"material_map": material_map},
    )

    # Pretty-printed standalone copy
    out_path = run_directory / "material_map.json"
    out_path.write_text(
        json.dumps({"material_map": material_map}, indent=2),
        encoding="utf-8",
    )
    _log.info("material_map.json written → %s", out_path)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Main stage entry-point
# ──────────────────────────────────────────────────────────────────────────────

def run_stage2(
    scene: dict,
    prompt: str,
    run_directory: Path,
    frame_end: int = 250,
    fps: int = 24,
) -> Path:
    """
    Execute Stage 2: material classification → force inference → Blender export.

    Parameters
    ----------
    scene:
        Validated scene dict from Stage 1.  Must contain an ``"objects"``
        list where each item has ``"label"`` and ``"annotated_image_path"``.
    prompt:
        Original user simulation prompt.
    run_directory:
        Shared workspace directory — all artefacts are written here.
    frame_end:
        Total simulation frames (passed to Blender exporter).
    fps:
        Blender scene frame rate.

    Returns
    -------
    Path to the generated ``simulation.py`` Blender script.
    """
    _log.info("═══ Stage 2 — Physical Reasoning & Simulation ═══")
    scene = {"objects" : [{"object_label" : "obj_0", "path" : "../utils/annotated.png"}]}

    objects = scene.get("objects", [])
    _log.info("Scene contains %d object(s).", len(objects))

    # ── Step 1: Material Classification (one object at a time via vision) ─────
    _log.info("Step 1/3 — Material Classification Agent")

    material_map = _classify_all_objects(
        objects=objects,
        prompt=prompt,
        run_directory=run_directory,
    )

    if not material_map:
        raise RuntimeError(
            "Material classification produced no results. "
            "Check that objects have valid annotated_image_path values."
        )

    mat_map_path = _write_material_map(material_map, run_directory)

    _log.info(
        "Classified %d/%d object(s): %s",
        len(material_map),
        len(objects),
        {k: f"{v['material_type']}/{v['sub_type']}" for k, v in material_map.items()},
    )
    print(material_map)
    # ── Step 2: Force Inference ───────────────────────────────────────────────
    _log.info("Step 2/3 — Force Inference Agent")

    # ── TEST SCENE — 2 objects loaded from PT + GLB ───────────────────────────
    object_position_directory = "/home/keshav06/sam3d/sam-3d-objects/notebook/objects"

    physics_scene, prompt = _build_test_scene(
        object_position_directory=object_position_directory,
        run_directory=run_directory,
    )

    # Auto-generate prompt from the two object labels
    objects = [
    {
        "object_label": obj.get("label", key),
        "path": f"../utils/annotated.png",  # placeholder until Stage 1 works
    }
    for key, obj in physics_scene["objects"].items()
    ]

    force_agent = ForceInferenceAgent(
        api_key=os.environ.get("ANTHROPIC_API_KEY", "dummy"),
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        model="qwen3-small",
    )
    force_spec = force_agent.infer_forces(
        prompt=prompt,
        scene=physics_scene,
        material_map=material_map,
        run_directory=str(run_directory),
    )


    # ── Step 3: Blender Export ────────────────────────────────────────────────
    _log.info("Step 3/3 — Blender Script Exporter")

    exporter = BlenderScriptExporter()
    script_path = exporter.export(
        scene=scene,
        material_map=material_map,
        force_spec=force_spec,
        run_directory=run_directory,
        prompt=prompt,
        frame_end=frame_end,
        fps=fps,
    )

    # ── Persist stage summary ─────────────────────────────────────────────────
    save_state(
        run_directory,
        "stage2_output",
        {
            "material_count":  len(material_map),
            "force_count":     len(force_spec.get("forces", [])),
            "material_map_path": str(mat_map_path),
            "blender_script":  str(script_path),
        },
    )

    _log.info("Stage 2 complete — Blender script: %s", script_path)
    return script_path

if __name__ == "__main__":
    # This file is not meant to be run directly — use main.py instead.
    _log.error(
        "This script is not meant to be run directly. "
        "Use main.py to execute the full pipeline."
    )
    run_stage2(
        scene={"objects" : [{"object_label" : "obj_0", "path" : "../utils/annotated.png"}]},
        prompt="...",
        run_directory=Path("./"),
    )