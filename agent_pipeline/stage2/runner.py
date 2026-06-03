"""
stage2/runner.py

Stage 2 Runner — Physical Reasoning & Simulation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Orchestrates Stage 2:

  For each scene object (from Stage 1)
        ↓
  MaterialClassificationAgent   (vision — overlay image per object)
        ↓  (all results merged)
  material_map.json
        ↓
  ForceInferenceAgent           (text — scene + material_map)
        ↓
  BlenderScriptExporter  →  simulation.py

Data flow from Stage 1
──────────────────────
  scene.json      — objects with label, mesh_path, bounding_box
  masks.json      — objects with label, mask_path (overlay PNG lives at
                    {mask_path.stem}_overlay.png in the same directory)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.shared import get_logger, save_state
from stage2.material_agent import MaterialClassificationAgent
from stage2.force_agent import ForceInferenceAgent
from stage2.blender_exporter import BlenderScriptExporter

_log = get_logger("stage2.runner")


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_overlay_lookup(run_directory: Path) -> dict[str, str]:
    """
    Read masks.json and return {label: overlay_png_path}.

    The semantic relevance agent saves an overlay PNG alongside each mask
    at ``{mask_path.stem}_overlay.png``.  If the overlay doesn't exist yet
    (e.g. mask was generated without evaluation), fall back to the raw mask.
    """
    masks_file = run_directory / "masks.json"
    if not masks_file.exists():
        _log.warning("masks.json not found in %s — material agent will have no images", run_directory)
        return {}

    with open(masks_file) as fh:
        data = json.load(fh)

    lookup: dict[str, str] = {}
    for mask in data.get("masks", []):
        label     = mask.get("label")
        mask_path = mask.get("mask_path")
        if not label or not mask_path:
            continue
        overlay = Path(mask_path).parent / f"{Path(mask_path).stem}_overlay.png"
        lookup[label] = str(overlay) if overlay.exists() else mask_path

    _log.info("Overlay lookup built: %d entries", len(lookup))
    return lookup


def _build_physics_scene(scene: dict) -> dict:
    """
    Convert Stage 1 scene (list-based objects with bounding_box) into the
    dict-based physics scene expected by ForceInferenceAgent.

    Each physics object gets:
      position        — centre of the Stage 1 bounding_box
      scale           — extents of the bounding_box
      rotation        — identity quaternion
      collision_shape — SPHERE when roughly spherical, else CONVEX_HULL
    """
    physics_objects: dict = {}

    for i, obj in enumerate(scene.get("objects", [])):
        label   = obj.get("label", f"object_{i}")
        obj_key = f"object_{i}"

        bbox    = obj.get("bounding_box", {})
        bmin    = bbox.get("min", [0.0, 0.0, 0.0])
        bmax    = bbox.get("max", [1.0, 1.0, 1.0])

        position = [(a + b) / 2.0 for a, b in zip(bmin, bmax)]
        scale    = [b - a         for a, b in zip(bmin, bmax)]

        # Simple heuristic: if all extents are within 20 % of each other → sphere
        s_max = max(scale) if max(scale) > 0 else 1.0
        collision_shape = "SPHERE" if (min(scale) / s_max) > 0.8 else "CONVEX_HULL"

        physics_objects[obj_key] = {
            "label":           label,
            "mesh_path":       obj.get("mesh_path", ""),
            "position":        position,
            "rotation":        [1.0, 0.0, 0.0, 0.0],
            "scale":           scale,
            "collision_shape": collision_shape,
        }

    return {"objects": physics_objects}


def _classify_all_objects(
    objects: list[dict],
    prompt: str,
    run_directory: Path,
) -> dict[str, dict]:
    """
    Run MaterialClassificationAgent on each object's overlay image.

    Each item in *objects* must have:
      "object_label"  (str)
      "path"          (str | Path)  path to the overlay / annotated image
    """
    agent        = MaterialClassificationAgent()
    material_map: dict[str, dict] = {}

    for idx, obj in enumerate(objects, start=1):
        label        = obj.get("object_label", f"object_{idx}")
        img_path_raw = obj.get("path")

        if not img_path_raw:
            _log.warning("Object '%s' has no image path — skipping.", label)
            continue

        img_path = Path(img_path_raw)
        if not img_path.exists():
            _log.warning("Image not found for '%s': %s — skipping.", label, img_path)
            continue

        _log.info("[%d/%d] Classifying '%s' from %s", idx, len(objects), label, img_path.name)

        try:
            spec = agent.classify_object(
                annotated_image_path=img_path,
                object_label=label,
                simulation_prompt=prompt,
            )
            material_map[label] = spec
        except Exception as exc:
            _log.error("Failed to classify '%s': %s", label, exc)

    return material_map


def _write_material_map(material_map: dict, run_directory: Path) -> Path:
    save_state(run_directory, "material_map", {"material_map": material_map})
    out_path = run_directory / "material_map.json"
    out_path.write_text(json.dumps({"material_map": material_map}, indent=2), encoding="utf-8")
    _log.info("material_map.json written → %s", out_path)
    return out_path


# ── main entry-point ──────────────────────────────────────────────────────────

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
    scene:          Validated scene dict from Stage 1 (contains ``"objects"`` list).
    prompt:         Original user simulation prompt.
    run_directory:  Shared workspace directory — all artefacts written here.
    frame_end:      Total simulation frames.
    fps:            Blender scene frame rate.

    Returns
    -------
    Path to the generated ``simulation.py`` Blender script.
    """
    _log.info("═══ Stage 2 — Physical Reasoning & Simulation ═══")

    scene_objects = scene.get("objects", [])
    if not scene_objects:
        raise RuntimeError("Stage 1 scene contains no objects — cannot run Stage 2.")

    _log.info("Scene contains %d object(s).", len(scene_objects))

    # ── Step 1: Material Classification ───────────────────────────────────────
    _log.info("Step 1/3 — Material Classification Agent")

    overlay_lookup = _build_overlay_lookup(run_directory)

    # Build list of {object_label, path} from Stage 1 objects + overlay images
    classified_objects = []
    for obj in scene_objects:
        label        = obj.get("label", "unknown")
        overlay_path = overlay_lookup.get(label)
        if overlay_path:
            classified_objects.append({"object_label": label, "path": overlay_path})
        else:
            _log.warning("No overlay image for '%s' — skipping material classification.", label)

    material_map = _classify_all_objects(classified_objects, prompt, run_directory)

    if not material_map:
        raise RuntimeError(
            "Material classification produced no results. "
            "Ensure masks.json exists in the run directory with valid mask_path entries."
        )

    mat_map_path = _write_material_map(material_map, run_directory)
    _log.info(
        "Classified %d/%d object(s): %s",
        len(material_map), len(scene_objects),
        {k: f"{v['material_type']}/{v['sub_type']}" for k, v in material_map.items()},
    )

    # ── Step 2: Force Inference ───────────────────────────────────────────────
    _log.info("Step 2/3 — Force Inference Agent")

    physics_scene = _build_physics_scene(scene)

    force_agent = ForceInferenceAgent(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        model=os.environ.get("FORCE_MODEL", os.environ.get("CLAUDE_MODEL", "qwen3-small")),
    )
    force_spec = force_agent.infer_forces(
        prompt=prompt,
        scene=physics_scene,
        material_map=material_map,
        run_directory=run_directory,
    )

    # ── Step 3: Blender Export ────────────────────────────────────────────────
    _log.info("Step 3/3 — Blender Script Exporter")

    exporter    = BlenderScriptExporter()
    script_path = exporter.export(
        scene=scene,
        material_map=material_map,
        force_spec=force_spec,
        run_directory=run_directory,
        prompt=prompt,
        frame_end=frame_end,
        fps=fps,
    )

    save_state(run_directory, "stage2_output", {
        "material_count":    len(material_map),
        "force_count":       len(force_spec.get("objects", {})),
        "material_map_path": str(mat_map_path),
        "blender_script":    str(script_path),
    })

    _log.info("Stage 2 complete — Blender script: %s", script_path)
    return script_path
