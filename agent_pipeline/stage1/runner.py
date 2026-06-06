"""
stage1/runner.py

Stage 1 Runner — Scene Understanding & 3D Reconstruction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Orchestrates the full Stage 1 pipeline:

  SAM3  (text-prompted segmentation — segments only objects in prompt)
       ↓
  ReconstructionAgent (SAM3D)  ←─────────────────────────┐
       ↓                                                   │  Iterative
  ValidationAgent                                         │  Refinement
       ↓ (failed)                                         │
  RefinementAgent ────────────────────────────────────────┘
       ↓ (passed OR max iterations reached)
  returns validated scene dict
"""

from __future__ import annotations

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from utils.shared import get_logger, load_state, save_state
from utils.sam3_wrapper import generate_masks_from_prompt
from stage1.reconstruction_agent import ReconstructionAgent
from stage1.validation_agent import ValidationAgent
from stage1.refinement_agent import RefinementAgent

_log = get_logger("stage1.runner")


def run_stage1(
    image_path: str,
    prompt: str,
    run_directory: Path,
) -> dict:
    """
    Execute the full Stage 1 pipeline.

    Parameters
    ----------
    image_path:   Absolute path to the input image.
    prompt:       User's physics simulation prompt.
    run_directory: Timestamped workspace directory for this run.

    Returns
    -------
    The validated scene dict (written to ``scene.json``).
    """
    _log.info("═══ Stage 1 — Scene Understanding & 3D Reconstruction ═══")

    # ── Step 1: SAM3 — text-prompted segmentation ──────────────────────────────
    # SAM3 segments only the objects mentioned in the prompt directly —
    # no need for uniform point sampling + LLM filtering.
    _log.info("Step 1/3 — SAM3 text-prompted segmentation (prompt: %r)", prompt)
    mask_dir = run_directory / "masks"
    stage1_result = generate_masks_from_prompt(
        image_path=image_path,
        prompt=prompt,
        mask_output_dir=mask_dir,
    )

    relevant_objects = stage1_result["relevant_objects"]
    masks            = stage1_result["masks"]

    # Persist for downstream agents
    save_state(run_directory, "relevant_objects", {"relevant_objects": relevant_objects})
    save_state(run_directory, "masks", {"masks": masks})

    if not relevant_objects:
        raise RuntimeError(
            "SAM3 found no objects matching the prompt. "
            "Try rephrasing with clearer noun phrases (e.g. 'ball, bottle')."
        )

    # ── Iterative Refinement Loop (Steps 2-4) ─────────────────────────────────
    reconstruction_agent = ReconstructionAgent()
    validation_agent = ValidationAgent()
    refinement_agent = RefinementAgent()

    refinement_hints: dict[str, dict] | None = None
    scene: dict = {}
    validation_report: dict = {}

    for iteration in range(1, cfg.MAX_REFINEMENT_ITERATIONS + 1):
        _log.info(
            "─── Refinement iteration %d / %d ───",
            iteration,
            cfg.MAX_REFINEMENT_ITERATIONS,
        )

        # Step 2: SAM3D Reconstruction
        _log.info("Step 2 — 3D Reconstruction Agent (SAM3D) [iter %d]", iteration)
        scene = reconstruction_agent.run_for_masks(
            image_path=image_path,
            masks=masks,
            run_directory=run_directory,
            refinement_hints=refinement_hints,
        )

        # Step 3: Validation
        _log.info("Step 3 — Validation Agent [iter %d]", iteration)
        validation_report = validation_agent.validate_scene(scene, run_directory)

        if validation_report.get("passed", False):
            _log.info("✓ Validation passed on iteration %d", iteration)
            break

        _log.warning(
            "✗ Validation failed — %d issue(s) detected",
            len(validation_report.get("issues", [])),
        )

        if iteration == cfg.MAX_REFINEMENT_ITERATIONS:
            _log.warning(
                "Max iterations (%d) reached. Proceeding with best available scene.",
                cfg.MAX_REFINEMENT_ITERATIONS,
            )
            break

        # Step 4: Refinement
        _log.info("Step 4 — Refinement Agent [iter %d]", iteration)
        refinement_hints = refinement_agent.refine(
            validation_report=validation_report,
            run_directory=run_directory,
            iteration=iteration,
        )

    # Persist final outputs
    save_state(run_directory, "stage1_output", {
        "scene": scene,
        "validation_report": validation_report,
        "passed": validation_report.get("passed", False),
    })

    _log.info("Stage 1 complete — %d object(s) in scene", scene.get("object_count", 0))
    return scene