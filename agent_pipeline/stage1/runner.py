"""
stage1/runner.py

Stage 1 Runner — Scene Understanding & 3D Reconstruction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Orchestrates the full Stage 1 pipeline:

  SemanticRelevanceAgent
       ↓
  SAM2Agent
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
from stage1.semantic_relevance_agent import SemanticRelevanceAgent
from stage1.sam2_agent import SAM2Agent
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

    # ── Step 1: Semantic Relevance ────────────────────────────────────────────
    _log.info("Step 1/5 — Semantic Relevance Agent")
    sem_agent = SemanticRelevanceAgent()
    relevant_objects = sem_agent.run_for_scene(image_path, prompt, run_directory)

    if not relevant_objects:
        raise RuntimeError(
            "Semantic Relevance Agent returned no relevant objects. "
            "Try a more descriptive prompt or a different image."
        )

    # ── Step 2: SAM2 Segmentation ─────────────────────────────────────────────
    _log.info("Step 2/5 — SAM2 Agent (%d objects)", len(relevant_objects))
    sam2_agent = SAM2Agent()
    masks = sam2_agent.run_for_objects(image_path, relevant_objects, run_directory)

    if not masks:
        raise RuntimeError("SAM2 Agent produced no masks.")

    # ── Iterative Refinement Loop (Steps 3-5) ─────────────────────────────────
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

        # Step 3: SAM3D Reconstruction
        _log.info("Step 3 — 3D Reconstruction Agent (SAM3D) [iter %d]", iteration)
        scene = reconstruction_agent.run_for_masks(
            image_path=image_path,
            masks=masks,
            run_directory=run_directory,
            refinement_hints=refinement_hints,
        )

        # Step 4: Validation
        _log.info("Step 4 — Validation Agent [iter %d]", iteration)
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

        # Step 5: Refinement
        _log.info("Step 5 — Refinement Agent [iter %d]", iteration)
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