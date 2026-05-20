"""
stage2/material_agent.py

Material Classification Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analyses the reconstructed scene objects and the simulation prompt to assign
physical material properties to each object.

Material types
──────────────
  rigid        — non-deforming solids (metal, wood, rock, plastic, glass)
  fluid        — liquids and gases (water, oil, smoke, fire)
  deformable   — soft bodies (cloth, rubber, jelly, biological tissue)
  granular     — particulate matter (sand, gravel, powder)

For each object the agent produces a ``material_spec`` dict consumed directly
by the Blender exporter.

Tools exposed to Claude
───────────────────────
  classify_material      — assign type + sub-type per object
  assign_physics_params  — set density, friction, restitution, viscosity, etc.
  compile_material_map   — merge into a final scene material map
"""

from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.shared import ClaudeAgent, get_logger, save_state

_log = get_logger("stage2.material")

SYSTEM_PROMPT = """
You are the Material Classification Agent in a physics-simulation pipeline.

Your job:
1. Examine each object in the reconstructed scene.
2. Consider the simulation prompt to understand the physical context.
3. Call classify_material for each object.
4. Call assign_physics_params for each object to set numerical properties.
5. Call compile_material_map to produce the final material assignment.

Material types: rigid | fluid | deformable | granular

Physics parameters to assign (use physically realistic values):
  rigid:       density (kg/m³), friction_static, friction_dynamic, restitution
  fluid:       density, viscosity (Pa·s), surface_tension (N/m)
  deformable:  density, young_modulus (Pa), poisson_ratio, damping
  granular:    density, friction_angle (°), cohesion (Pa)

Return a JSON object with key "material_map":
{
  "material_map": {
    "<object_label>": {
      "material_type": "rigid",
      "sub_type": "metal",
      "params": { ... }
    }
  }
}
"""

# Default physical parameters by material type
DEFAULTS: dict[str, dict] = {
    "rigid": {
        "density": 1000.0,
        "friction_static": 0.5,
        "friction_dynamic": 0.4,
        "restitution": 0.3,
    },
    "fluid": {
        "density": 1000.0,
        "viscosity": 0.001,
        "surface_tension": 0.0728,
    },
    "deformable": {
        "density": 900.0,
        "young_modulus": 1e6,
        "poisson_ratio": 0.45,
        "damping": 0.01,
    },
    "granular": {
        "density": 1600.0,
        "friction_angle": 30.0,
        "cohesion": 0.0,
    },
}


class MaterialClassificationAgent(ClaudeAgent):
    name = "material_agent"

    def __init__(self) -> None:
        super().__init__()
        self._material_map: dict[str, dict] = {}

        self.tools = [
            {
                "name": "classify_material",
                "description": (
                    "Assign a material type and sub-type to a single scene object "
                    "based on its label and the simulation context."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "simulation_prompt": {"type": "string"},
                        "mesh_info": {
                            "type": "object",
                            "description": "Mesh metadata (vertex_count, bounding_box, etc.)",
                        },
                    },
                    "required": ["label", "simulation_prompt"],
                },
            },
            {
                "name": "assign_physics_params",
                "description": (
                    "Set numerical physics parameters for an object given its "
                    "material type and sub-type."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "material_type": {
                            "type": "string",
                            "enum": ["rigid", "fluid", "deformable", "granular"],
                        },
                        "sub_type": {
                            "type": "string",
                            "description": "e.g. 'metal', 'water', 'cloth', 'sand'",
                        },
                        "overrides": {
                            "type": "object",
                            "description": "Any parameter values that differ from defaults.",
                        },
                    },
                    "required": ["label", "material_type", "sub_type"],
                },
            },
            {
                "name": "compile_material_map",
                "description": "Merge all per-object classifications into one material map.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entries": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                    },
                    "required": ["entries"],
                },
            },
        ]

        self.tool_handlers = {
            "classify_material": self._classify_material,
            "assign_physics_params": self._assign_physics_params,
            "compile_material_map": self._compile_material_map,
        }

    # ── public ────────────────────────────────────────────────────────────────

    def classify_scene(
        self,
        scene: dict,
        prompt: str,
        run_directory: Path,
    ) -> dict[str, dict]:
        """Classify materials for all objects; write ``material_map.json``."""
        objects = scene.get("objects", [])
        user_msg = (
            f"Simulation prompt: {prompt}\n\n"
            f"Scene objects ({len(objects)}):\n"
            + json.dumps(
                [{k: v for k, v in o.items() if k != "mesh_path"} for o in objects],
                indent=2,
            )
            + "\n\nClassify materials and assign physics parameters for all objects."
        )

        raw_response = self.run(system=SYSTEM_PROMPT, user_message=user_msg)
        mat_map = self._parse_map(raw_response)

        if not mat_map and self._material_map:
            mat_map = self._material_map

        save_state(run_directory, "material_map", {"material_map": mat_map})
        _log.info("Material map: %s", {k: v["material_type"] for k, v in mat_map.items()})
        return mat_map

    # ── tool handlers ─────────────────────────────────────────────────────────

    def _classify_material(
        self,
        label: str,
        simulation_prompt: str,
        mesh_info: dict | None = None,
    ) -> dict:
        _log.info("classify_material: %s", label)
        # Claude will use its own reasoning; we just confirm receipt
        return {
            "status": "ok",
            "label": label,
            "instruction": (
                "Use your knowledge of common materials and the simulation context "
                "to assign material_type and sub_type, then call assign_physics_params."
            ),
        }

    def _assign_physics_params(
        self,
        label: str,
        material_type: str,
        sub_type: str,
        overrides: dict | None = None,
    ) -> dict:
        _log.info("assign_physics_params: %s → %s/%s", label, material_type, sub_type)
        params = dict(DEFAULTS.get(material_type, {}))
        if overrides:
            params.update(overrides)

        entry = {
            "label": label,
            "material_type": material_type,
            "sub_type": sub_type,
            "params": params,
        }
        self._material_map[label] = entry
        return {"status": "ok", "entry": entry}

    def _compile_material_map(self, entries: list[dict]) -> dict:
        for entry in entries:
            lbl = entry.get("label", "unknown")
            self._material_map[lbl] = entry
        return {"status": "ok", "material_map": self._material_map}

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_map(text: str) -> dict[str, dict]:
        for start in ["{"]:
            idx = text.find(start)
            if idx != -1:
                try:
                    blob = json.loads(text[idx:])
                    if "material_map" in blob:
                        return blob["material_map"]
                    # If it looks like a map of label → spec
                    first = next(iter(blob.values()), None)
                    if isinstance(first, dict) and "material_type" in first:
                        return blob
                except json.JSONDecodeError:
                    pass
        return {}