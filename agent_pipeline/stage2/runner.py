"""
stage2/runner.py

Stage 2 Runner — Physical Reasoning & Simulation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Orchestrates Stage 2:

  MaterialClassificationAgent
         ↓
  ForceInferenceAgent
         ↓
  BlenderScriptExporter  → simulation.py
"""

from __future__ import annotations

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.shared import get_logger, save_state
from stage2.material_agent import MaterialClassificationAgent
from stage2.force_agent import ForceInferenceAgent
from stage2.blender_exporter import BlenderScriptExporter

_log = get_logger("stage2.runner")


def run_stage2(
    scene: dict,
    prompt: str,
    run_directory: Path,
    frame_end: int = 250,
    fps: int = 24,
) -> Path:
    """
    Execute Stage 2.

    Parameters
    ----------
    scene:         Validated scene dict from Stage 1.
    prompt:        Original user simulation prompt.
    run_directory: Shared workspace directory.
    frame_end:     Total simulation frames.
    fps:           Blender scene frame rate.

    Returns
    -------
    Path to the generated ``simulation.py`` Blender script.
    """
    _log.info("═══ Stage 2 — Physical Reasoning & Simulation ═══")

    # ── Step 1: Material Classification ──────────────────────────────────────
    _log.info("Step 1/3 — Material Classification Agent")
    mat_agent = MaterialClassificationAgent()
    material_map = mat_agent.classify_scene(
        scene=scene,
        prompt=prompt,
        run_directory=run_directory,
    )

    # ── Step 2: Force Inference ───────────────────────────────────────────────
    _log.info("Step 2/3 — Force Inference Agent")
    force_agent = ForceInferenceAgent()
    force_spec = force_agent.infer_forces(
        prompt=prompt,
        scene=scene,
        material_map=material_map,
        run_directory=run_directory,
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

    # Persist stage 2 summary
    save_state(run_directory, "stage2_output", {
        "material_count": len(material_map),
        "force_count": len(force_spec.get("forces", [])),
        "blender_script": str(script_path),
    })

    _log.info("Stage 2 complete — Blender script: %s", script_path)
    return script_path