"""
stage1/reconstruction_agent.py

3D Reconstruction Agent (SAM3D)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Drives coarse-to-fine mesh generation for every masked object using the
local SAM3D model.  Supports optional refinement hints from the Refinement
Agent so that repeated passes can focus on previously detected problems.

Tools exposed to Claude
───────────────────────
  reconstruct_object    — generate a mesh for one masked object
  summarise_scene       — aggregate per-object results into a scene dict
"""

from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.shared import ClaudeAgent, get_logger, save_state
from utils.sam3d_wrapper import generate_mesh

_log = get_logger("stage1.reconstruction")

SYSTEM_PROMPT = """
You are the 3D Reconstruction Agent (SAM3D) in a physics-simulation pipeline.

Your job:
1. Receive a list of segmented objects (masks + metadata).
2. Call reconstruct_object for each object to generate a 3-D mesh.
3. After all reconstructions, call summarise_scene to create the full scene
   representation.
4. Return the scene summary as JSON with key "scene".

If refinement_hints are provided for an object, pass them to reconstruct_object
so SAM3D can address known issues (holes, overlaps, misclassifications).

Always prefer "adaptive" quality unless explicitly told otherwise.
"""


class ReconstructionAgent(ClaudeAgent):
    name = "reconstruction_agent"

    def __init__(self) -> None:
        super().__init__()
        self._meshes: list[dict] = []
        self._run_dir: Path = Path(".")

        self.tools = [
            {
                "name": "reconstruct_object",
                "description": (
                    "Call SAM3D locally to generate a 3-D mesh for one masked "
                    "object. Returns mesh metadata including vertex/face counts "
                    "and the path to the .obj file."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to the original RGB image.",
                        },
                        "mask_data": {
                            "type": "object",
                            "description": "Mask dict for this object (from SAM2 Agent).",
                        },
                        "quality": {
                            "type": "string",
                            "enum": ["coarse", "fine", "adaptive"],
                            "description": "Mesh quality preset.",
                        },
                        "refinement_hints": {
                            "type": "object",
                            "description": (
                                "Optional hints from the Refinement Agent "
                                "(e.g. {'holes': [...], 'overlaps': [...]})."
                            ),
                        },
                    },
                    "required": ["image_path", "mask_data"],
                },
            },
            {
                "name": "summarise_scene",
                "description": (
                    "Aggregate all reconstructed meshes into a single scene "
                    "dictionary suitable for the Validation Agent."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "meshes": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of mesh dicts from reconstruct_object.",
                        },
                    },
                    "required": ["meshes"],
                },
            },
        ]

        self.tool_handlers = {
            "reconstruct_object": self._reconstruct_object,
            "summarise_scene": self._summarise_scene,
        }

    # ── public ────────────────────────────────────────────────────────────────

    def run_for_masks(
        self,
        image_path: str,
        masks: list[dict],
        run_directory: Path,
        refinement_hints: dict[str, dict] | None = None,
    ) -> dict:
        """
        Reconstruct all masked objects.

        Parameters
        ----------
        refinement_hints:
            Mapping of object label → hint dict from the Refinement Agent.
            Pass ``None`` on the first iteration.

        Returns the scene dict and writes ``scene.json``.
        """
        self._run_dir = run_directory
        self._meshes = []

        hints_str = ""
        if refinement_hints:
            hints_str = (
                "\n\nRefinement hints from previous validation:\n"
                + json.dumps(refinement_hints, indent=2)
            )

        user_msg = (
            f"Image path: {image_path}\n\n"
            f"Masks to reconstruct ({len(masks)}):\n"
            + json.dumps(
                [{k: v for k, v in m.items() if k != "mask_rle"} for m in masks],
                indent=2,
            )
            + hints_str
            + "\n\nReconstruct each object, then summarise the scene."
        )

        raw_response = self.run(system=SYSTEM_PROMPT, user_message=user_msg)
        scene = self._parse_scene(raw_response)

        if not scene and self._meshes:
            scene = self._build_scene(self._meshes)

        save_state(run_directory, "scene", scene)
        _log.info("Scene reconstructed: %d objects", len(scene.get("objects", [])))
        return scene

    # ── tool handlers ─────────────────────────────────────────────────────────

    def _reconstruct_object(
        self,
        image_path: str,
        mask_data: dict,
        quality: str = "adaptive",
        refinement_hints: dict | None = None,
    ) -> dict:
        label = mask_data.get("label", "unknown")
        _log.info("Reconstructing '%s' (quality=%s)", label, quality)

        mesh_info = generate_mesh(
            image_path=image_path,
            mask_data=mask_data,
            output_dir=self._run_dir,
            quality=quality,
            refinement_hints=refinement_hints,
        )
        self._meshes.append(mesh_info)
        return {"status": "ok", "mesh": mesh_info}

    def _summarise_scene(self, meshes: list[dict]) -> dict:
        scene = self._build_scene(meshes)
        return {"status": "ok", "scene": scene}

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_scene(meshes: list[dict]) -> dict:
        return {
            "object_count": len(meshes),
            "objects": meshes,
        }

    @staticmethod
    def _parse_scene(text: str) -> dict:
        for start in ["{", "["]:
            idx = text.find(start)
            if idx != -1:
                try:
                    blob = json.loads(text[idx:])
                    if isinstance(blob, dict):
                        for key in ("scene", "summary"):
                            if key in blob:
                                return blob[key]
                        if "objects" in blob:
                            return blob
                except json.JSONDecodeError:
                    pass
        return {}