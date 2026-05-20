"""
stage2/force_agent.py

Force Inference Agent
━━━━━━━━━━━━━━━━━━━━
Reads the user's simulation prompt and the material-classified scene to
infer all physical forces and initial conditions needed for simulation.

Per force the agent predicts:
  • direction        — 3-D unit vector [x, y, z]
  • magnitude        — scalar in Newtons (for impulse) or Pa (for pressure)
  • point_of_application — object label + local surface position
  • duration         — seconds (0 = instantaneous impulse)
  • force_type       — gravity | impulse | pressure | wind | explosion | custom

Tools exposed to Claude
───────────────────────
  parse_prompt_forces   — extract force mentions from the natural-language prompt
  resolve_force_targets — map force descriptions to specific scene objects
  compute_force_vectors — convert human descriptions to 3-D vectors + magnitudes
  compile_force_spec    — assemble the final force specification
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.shared import ClaudeAgent, get_logger, save_state

_log = get_logger("stage2.force")

SYSTEM_PROMPT = """
You are the Force Inference Agent in a physics-simulation pipeline.

Your job:
1. Read the simulation prompt carefully to identify all physical forces,
   interactions, and initial conditions described.
2. Call parse_prompt_forces to extract force descriptions.
3. Call resolve_force_targets to map each force to a scene object.
4. Call compute_force_vectors to convert descriptions into numeric 3-D vectors.
5. Call compile_force_spec to produce the final structured force specification.

Coordinate system (Blender convention):
  +X → right,  +Y → into screen (depth),  +Z → up

Force types:
  gravity      — constant downward field (already set by Blender; include for clarity)
  impulse      — instantaneous force applied at t=0 or t=start_time
  sustained    — constant force applied over [start_time, start_time + duration]
  pressure     — uniform surface pressure (e.g. wind, explosion)
  explosion    — radial outward force from a point origin
  constraint   — fixed / pinned points (duration = -1 means permanent)

Return a JSON object with key "force_spec":
{
  "gravity": {"enabled": true, "magnitude": 9.81, "direction": [0, 0, -1]},
  "forces": [
    {
      "force_id": "f1",
      "force_type": "impulse",
      "target_object": "<label>",
      "direction": [0.0, 0.0, -1.0],
      "magnitude": 500.0,
      "point_of_application": "center_of_mass",
      "start_time": 0.0,
      "duration": 0.0,
      "description": "<human-readable summary>"
    }
  ],
  "initial_conditions": []
}
"""


class ForceInferenceAgent(ClaudeAgent):
    name = "force_agent"

    def __init__(self) -> None:
        super().__init__()
        self._forces: list[dict] = []

        self.tools = [
            {
                "name": "parse_prompt_forces",
                "description": (
                    "Extract all force-related phrases and events from the "
                    "simulation prompt (e.g. 'drops', 'explosion', 'wind blowing')."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                    },
                    "required": ["prompt"],
                },
            },
            {
                "name": "resolve_force_targets",
                "description": (
                    "Map each extracted force description to the specific "
                    "scene object(s) it acts upon."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "force_descriptions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "scene_objects": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of object labels in the scene.",
                        },
                    },
                    "required": ["force_descriptions", "scene_objects"],
                },
            },
            {
                "name": "compute_force_vectors",
                "description": (
                    "Convert each force description into a numeric 3-D direction "
                    "vector, magnitude, and duration."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "force_targets": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"},
                                    "target_object": {"type": "string"},
                                    "force_type": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["force_targets"],
                },
            },
            {
                "name": "compile_force_spec",
                "description": "Assemble all forces into the final force specification.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "forces": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                        "include_gravity": {
                            "type": "boolean",
                            "default": True,
                        },
                        "gravity_magnitude": {
                            "type": "number",
                            "default": 9.81,
                        },
                    },
                    "required": ["forces"],
                },
            },
        ]

        self.tool_handlers = {
            "parse_prompt_forces": self._parse_prompt_forces,
            "resolve_force_targets": self._resolve_force_targets,
            "compute_force_vectors": self._compute_force_vectors,
            "compile_force_spec": self._compile_force_spec,
        }

    # ── public ────────────────────────────────────────────────────────────────

    def infer_forces(
        self,
        prompt: str,
        scene: dict,
        material_map: dict[str, dict],
        run_directory: Path,
    ) -> dict:
        """Infer forces and write ``force_spec.json``."""
        object_labels = [o.get("label", "") for o in scene.get("objects", [])]

        user_msg = (
            f"Simulation prompt: {prompt}\n\n"
            f"Scene objects: {object_labels}\n\n"
            "Material map:\n"
            + json.dumps(
                {k: v.get("material_type") for k, v in material_map.items()},
                indent=2,
            )
            + "\n\nInfer all forces and return the force_spec JSON."
        )

        raw_response = self.run(system=SYSTEM_PROMPT, user_message=user_msg)
        force_spec = self._parse_force_spec(raw_response)

        if not force_spec:
            force_spec = {
                "gravity": {"enabled": True, "magnitude": 9.81, "direction": [0, 0, -1]},
                "forces": self._forces,
                "initial_conditions": [],
            }

        save_state(run_directory, "force_spec", force_spec)
        _log.info(
            "Force spec: %d force(s) inferred",
            len(force_spec.get("forces", [])),
        )
        return force_spec

    # ── tool handlers ─────────────────────────────────────────────────────────

    def _parse_prompt_forces(self, prompt: str) -> dict:
        _log.info("Parsing force descriptions from prompt")
        return {
            "status": "ok",
            "prompt": prompt,
            "instruction": (
                "Identify all physical events in the prompt: impacts, drops, "
                "explosions, wind, gravity effects, sustained pushes, etc."
            ),
        }

    def _resolve_force_targets(
        self,
        force_descriptions: list[str],
        scene_objects: list[str],
    ) -> dict:
        _log.info(
            "Resolving %d force descriptions against %d objects",
            len(force_descriptions), len(scene_objects),
        )
        return {
            "status": "ok",
            "force_descriptions": force_descriptions,
            "scene_objects": scene_objects,
            "instruction": (
                "For each description, identify which scene object it primarily "
                "acts upon. If unclear, use the most physically plausible match."
            ),
        }

    def _compute_force_vectors(self, force_targets: list[dict]) -> dict:
        _log.info("Computing vectors for %d forces", len(force_targets))
        # Claude populates actual values; this confirms the call
        return {
            "status": "ok",
            "force_count": len(force_targets),
            "instruction": (
                "Convert each force to: direction [x,y,z] unit vector, "
                "magnitude (N or Pa), start_time (s), duration (s)."
            ),
        }

    def _compile_force_spec(
        self,
        forces: list[dict],
        include_gravity: bool = True,
        gravity_magnitude: float = 9.81,
    ) -> dict:
        _log.info("Compiling force spec: %d force(s)", len(forces))
        # Normalize directions to unit vectors
        for f in forces:
            d = f.get("direction", [0, 0, -1])
            mag = math.sqrt(sum(x ** 2 for x in d))
            if mag > 0:
                f["direction"] = [x / mag for x in d]

        self._forces = forces
        spec = {
            "gravity": {
                "enabled": include_gravity,
                "magnitude": gravity_magnitude,
                "direction": [0, 0, -1],
            },
            "forces": forces,
            "initial_conditions": [],
        }
        return spec

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_force_spec(text: str) -> dict:
        for start in ["{"]:
            idx = text.find(start)
            if idx != -1:
                try:
                    blob = json.loads(text[idx:])
                    if "force_spec" in blob:
                        return blob["force_spec"]
                    if "forces" in blob and "gravity" in blob:
                        return blob
                except json.JSONDecodeError:
                    pass
        return {}