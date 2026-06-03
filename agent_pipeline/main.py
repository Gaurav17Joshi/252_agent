"""
main.py — Physics Simulation Pipeline Entry Point
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage
─────
    python main.py --image path/to/image.jpg \\
                   --prompt "A heavy metal ball drops onto a glass table" \\
                   [--frames 250] [--fps 24] [--run-id my_run]

Environment variables
─────────────────────
    ANTHROPIC_API_KEY   (required)
    SAM2_CHECKPOINT     (optional, default in config.py)
    SAM3D_CHECKPOINT    (optional, default in config.py)
    LOG_LEVEL           (optional, default INFO)

Output
──────
    workspace/<run_id>/
        relevant_objects.json
        masks.json
        scene.json
        validation_report.json
        refinement_hints_iter*.json   (if refinement ran)
        material_map.json
        force_spec.json
        simulation.py                 ← Blender script
        *.obj                         ← reconstructed meshes
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from utils.shared import get_logger, run_dir
from stage1.runner import run_stage1
from stage2.runner import run_stage2

_log = get_logger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an image + text prompt into a Blender physics simulation."
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the input image (JPEG or PNG).",
    )
    parser.add_argument(
        "--prompt", required=True,
        help="Natural-language description of the physics simulation to create.",
    )
    parser.add_argument(
        "--frames", type=int, default=250,
        help="Number of simulation frames (default: 250).",
    )
    parser.add_argument(
        "--fps", type=int, default=24,
        help="Blender scene frame rate (default: 24).",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Optional run identifier (default: timestamped).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        _log.error("Image not found: %s", image_path)
        sys.exit(1)

    if not cfg.ANTHROPIC_API_KEY:
        _log.error(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Export it before running: export ANTHROPIC_API_KEY=sk-..."
        )
        sys.exit(1)

    # ── Create workspace ──────────────────────────────────────────────────────
    workspace = run_dir(args.run_id)
    _log.info("Run workspace: %s", workspace)
    _log.info("Image       : %s", image_path)
    _log.info("Prompt      : %s", args.prompt)

    start = time.monotonic()

    try:
        # ── Stage 1 ───────────────────────────────────────────────────────────
        # scene = run_stage1(
        #     image_path=str(image_path),
        #     prompt=args.prompt,
        #     run_directory=workspace,
        # )

        # ── Stage 2 ───────────────────────────────────────────────────────────
        blender_script = run_stage2(
            scene=scene,
            prompt=args.prompt,
            run_directory=workspace,
            frame_end=args.frames,
            fps=args.fps,
        )

    except Exception as exc:
        _log.exception("Pipeline failed: %s", exc)
        sys.exit(1)

    elapsed = time.monotonic() - start
    _log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    _log.info("Pipeline complete in %.1f s", elapsed)
    _log.info("Blender script : %s", blender_script)
    _log.info("Run workspace  : %s", workspace)
    _log.info("")
    _log.info("To simulate:")
    _log.info("  blender --background --python %s", blender_script)
    _log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()