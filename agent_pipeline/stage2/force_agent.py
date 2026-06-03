# """
# motion_initialization_agent.py

# ForceInferenceAgent
# ━━━━━━━━━━━━━━━━━━━

# This agent predicts:
# - initial linear velocity
# - initial angular velocity
# - linear damping
# - angular damping

# for Blender rigid-body simulation.

# Inputs:
# - natural-language simulation prompt
# - scene objects with:
#     - label
#     - mass
#     - friction
#     - restitution
#     - collision_shape
#     - position
#     - rotation
#     - scale

# Blender handles:
# - gravity
# - collisions
# - rigid-body simulation
# - momentum transfer
# """

# from __future__ import annotations

# import json
# import os
# import re
# from pathlib import Path
# from typing import Dict, Any, Optional

# from anthropic import Anthropic


# SYSTEM_PROMPT = """/no_think
# You are a JSON-only API endpoint for a Blender physics simulation pipeline.

# YOU MUST RESPOND WITH ONLY A JSON OBJECT.
# NO explanations. NO bullet points. NO markdown. NO text of any kind.
# YOUR ENTIRE RESPONSE MUST START WITH { AND END WITH }

# If you write any text outside the JSON object, the pipeline will crash.

# TASK:
# Given a simulation prompt and scene objects, predict:
# - initial_linear_velocity
# - initial_angular_velocity
# - linear_damping
# - angular_damping

# CRITICAL RULE — DIRECTION CALCULATION:
# Compute velocity direction mathematically from positions.

# Steps:
# 1. Identify the moving object and target object from the prompt
# 2. Compute direction vector:
#    dx = target.position[0] - source.position[0]
#    dy = target.position[1] - source.position[1]
#    dz = target.position[2] - source.position[2]
# 3. Normalize:
#    magnitude = sqrt(dx^2 + dy^2 + dz^2)
#    direction = [dx/magnitude, dy/magnitude, dz/magnitude]
# 4. Scale by realistic speed (3.0 - 6.0 for a ball hit)
# 5. Use as initial_linear_velocity

# CRITICAL RULE — STATIONARY OBJECTS:
# - Only the MOVING object gets non-zero velocity
# - ALL OTHER objects: initial_linear_velocity = [0.0, 0.0, 0.0]
# - NEVER assign post-collision velocity to stationary objects

# MATERIAL GUIDANCE — use sub_type to tune damping:
# - glass/ceramic  -> linear_damping: 0.01-0.03, angular_damping: 0.01-0.03
# - rubber/soft    -> linear_damping: 0.1-0.2,   angular_damping: 0.1-0.2
# - cardboard/wood -> linear_damping: 0.04-0.08, angular_damping: 0.05-0.1
# - metal          -> linear_damping: 0.02-0.05, angular_damping: 0.02-0.05

# Coordinate system: +X=right, +Y=depth, +Z=up
# Velocity limits: linear [-10, 10], angular [-20, 20]

# OUTPUT FORMAT — your response must be exactly this structure, nothing else:
# {"objects":{"object_0":{"label":"ball","initial_linear_velocity":[3.578,1.789,0.0],"initial_angular_velocity":[0.0,0.0,0.0],"linear_damping":0.1,"angular_damping":0.1},"object_1":{"label":"bottle","initial_linear_velocity":[0.0,0.0,0.0],"initial_angular_velocity":[0.0,0.0,0.0],"linear_damping":0.02,"angular_damping":0.02}}}
# """


# class ForceInferenceAgent:
#     def __init__(
#         self,
#         api_key: str,
#         model: str = "qwen3",
#         base_url: str = "http://localhost:11434",
#     ):
#         self.client = Anthropic(
#             api_key=api_key,
#             base_url=base_url,
#         )
#         self.model = model

#     def infer_forces(
#         self,
#         prompt: str,
#         scene: Dict[str, Any],
#         material_map: Optional[Dict[str, Any]] = None,
#         objects: Optional[str] = "/home/keshav06/sam3d/sam-3d-objects/notebook/objects",
#         run_directory: Optional[str] = "./",
#     ) -> Dict[str, Any]:
#         """
#         Predict initial motion states for Blender rigid-body simulation.
#         Saves results to force_spec.json in run_directory.
#         """

#         # Merge material_map into scene objects for richer context
#         enriched_scene = self._enrich_scene_with_materials(scene, material_map)

#         user_message = f"""/no_think
# Simulation Prompt:
# {prompt}

# Scene Objects (with material properties):
# {json.dumps(enriched_scene, indent=2)}

# Infer physically plausible motion initialization parameters.
# """
#         print(user_message)

#         response = self.client.messages.create(
#             model=self.model,
#             max_tokens=4096,
#             temperature=0.2,
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user",   "content": user_message},
#             ],
#             extra_body={
#                 "chat_template_kwargs": {"enable_thinking": False},
#             },
#         )

#         text = response.content[0].text
#         print("=== Raw model response ===")
#         print(text)
#         print("==========================")

#         motion_spec = self._parse_json(text)
#         motion_spec = self._clamp_values(motion_spec)

#         # Save force_spec.json with material properties merged in
#         self._save_force_spec(motion_spec, enriched_scene, run_directory)

#         return motion_spec

#     def _enrich_scene_with_materials(
#         self,
#         scene: Dict[str, Any],
#         material_map: Optional[Dict[str, Any]],
#     ) -> Dict[str, Any]:
#         """
#         Merge material_map properties into each scene object.
#         Returns enriched scene dict.
#         """
#         if not material_map:
#             return scene

#         enriched = json.loads(json.dumps(scene))  # deep copy

#         mat = material_map.get("material_map", material_map)

#         for obj_key, obj in enriched.get("objects", {}).items():
#             label = obj.get("label", obj_key)

#             # Try matching by label or obj_key
#             mat_entry = mat.get(label) or mat.get(obj_key)

#             if mat_entry:
#                 obj["material_type"] = mat_entry.get("material_type")
#                 obj["sub_type"]      = mat_entry.get("sub_type")

#                 # Overwrite physics params from material_map if present
#                 params = mat_entry.get("params", {})
#                 if "mass"        in params: obj["mass"]        = params["mass"]
#                 if "friction"    in params: obj["friction"]    = params["friction"]
#                 if "restitution" in params: obj["restitution"] = params["restitution"]

#         return enriched

#     def _save_force_spec(
#         self,
#         motion_spec: Dict[str, Any],
#         enriched_scene: Dict[str, Any],
#         run_directory: str,
#     ) -> Path:
#         """
#         Save force_spec.json to run_directory.
#         Includes both velocity parameters and material properties.
#         Separate from material_map.json.
#         """

#         # Merge material properties from enriched scene into motion_spec
#         for obj_key, obj in motion_spec.get("objects", {}).items():
#             scene_obj = enriched_scene.get("objects", {}).get(obj_key, {})
#             obj["material_type"]   = scene_obj.get("material_type")
#             obj["sub_type"]        = scene_obj.get("sub_type")
#             obj["mass"]            = scene_obj.get("mass")
#             obj["friction"]        = scene_obj.get("friction")
#             obj["restitution"]     = scene_obj.get("restitution")
#             obj["collision_shape"] = scene_obj.get("collision_shape")
#             obj["position"]        = scene_obj.get("position")
#             obj["rotation"]        = scene_obj.get("rotation")
#             obj["scale"]           = scene_obj.get("scale")

#         out_dir  = Path(run_directory)
#         out_dir.mkdir(parents=True, exist_ok=True)
#         out_path = out_dir / "force_spec.json"

#         out_path.write_text(
#             json.dumps(motion_spec, indent=2),
#             encoding="utf-8",
#         )
#         print(f"force_spec.json written -> {out_path}")
#         return out_path

#     def _strip_markdown(self, text: str) -> str:
#         """
#         Remove markdown code fences (```json ... ``` or ``` ... ```)
#         so JSON can be extracted cleanly.
#         """
#         text = re.sub(r'```json\s*', '', text)
#         text = re.sub(r'```\s*',     '', text)
#         return text.strip()

#     def _parse_json(self, text: str) -> Dict[str, Any]:
#         """
#         Extract and normalize JSON from model response.
#         Handles markdown fences and different key formats.
#         """
#         # Strip markdown backticks first
#         text = self._strip_markdown(text)

#         match = re.search(r'\{.*\}', text, re.DOTALL)

#         if not match:
#             raise ValueError(
#                 f"No JSON found in model response. "
#                 f"Response was: {repr(text[:300])}"
#             )

#         try:
#             raw = json.loads(match.group())
#         except json.JSONDecodeError as e:
#             raise ValueError(
#                 f"Found JSON block but failed to parse (likely truncated): {e}\n"
#                 f"JSON blob: {match.group()[:300]}"
#             )

#         return self._normalize_response(raw)

#     def _normalize_response(self, raw: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Normalize model response into standard format regardless of
#         what structure the model returns.

#         Handles:
#         - correct format:  {"objects": {"object_0": {"initial_linear_velocity": ...}}}
#         - short keys:      {"objects": {"object_0": {"linear_velocity": ...}}}
#         - label-keyed:     {"ball": {"linear_velocity": ...}, "bottle": {...}}
#         """

#         # Already has "objects" wrapper
#         if "objects" in raw:
#             for obj in raw["objects"].values():
#                 # Normalize short key names
#                 if "linear_velocity" in obj and "initial_linear_velocity" not in obj:
#                     obj["initial_linear_velocity"] = obj.pop("linear_velocity")
#                 if "angular_velocity" in obj and "initial_angular_velocity" not in obj:
#                     obj["initial_angular_velocity"] = obj.pop("angular_velocity")
#             return raw

#         # Model returned label-keyed dict: {"ball": {...}, "bottle": {...}}
#         normalized = {"objects": {}}
#         for idx, (label, props) in enumerate(raw.items()):
#             obj_key = f"object_{idx}"
#             normalized["objects"][obj_key] = {
#                 "label": label,
#                 "initial_linear_velocity":  props.get("linear_velocity",
#                                             props.get("initial_linear_velocity", [0.0, 0.0, 0.0])),
#                 "initial_angular_velocity": props.get("angular_velocity",
#                                             props.get("initial_angular_velocity", [0.0, 0.0, 0.0])),
#                 "linear_damping":           props.get("linear_damping",  0.04),
#                 "angular_damping":          props.get("angular_damping", 0.1),
#             }

#         return normalized

#     def _clamp_values(
#         self,
#         spec: Dict[str, Any],
#     ) -> Dict[str, Any]:
#         """
#         Clamp values to prevent unstable Blender physics.
#         """
#         if "objects" not in spec:
#             return spec

#         for obj in spec["objects"].values():

#             lv = obj.get("initial_linear_velocity", [0.0, 0.0, 0.0])
#             obj["initial_linear_velocity"] = [
#                 max(-10.0, min(10.0, float(v))) for v in lv
#             ]

#             av = obj.get("initial_angular_velocity", [0.0, 0.0, 0.0])
#             obj["initial_angular_velocity"] = [
#                 max(-20.0, min(20.0, float(v))) for v in av
#             ]

#             obj["linear_damping"] = max(
#                 0.0, min(0.3, float(obj.get("linear_damping", 0.04)))
#             )

#             obj["angular_damping"] = max(
#                 0.0, min(0.5, float(obj.get("angular_damping", 0.1)))
#             )

#         return spec


# if __name__ == "__main__":

#     prompt = "A ball hits a bottle from the left."

#     scene = {
#         "objects": {
#             "object_0": {
#                 "label": "ball",
#                 "mass": 0.5,
#                 "friction": 0.4,
#                 "restitution": 0.7,
#                 "collision_shape": "SPHERE",
#                 "position": [0.0, 0.0, 0.5],
#                 "rotation": [1.0, 0.0, 0.0, 0.0],
#                 "scale": [0.15, 0.15, 0.15],
#             },
#             "object_1": {
#                 "label": "bottle",
#                 "mass": 1.0,
#                 "friction": 0.6,
#                 "restitution": 0.2,
#                 "collision_shape": "CONVEX_HULL",
#                 "position": [2.0, 1.0, 0.5],
#                 "rotation": [1.0, 0.0, 0.0, 0.0],
#                 "scale": [0.2, 0.2, 0.5],
#             },
#         }
#     }

#     material_map = {
#         "material_map": {
#             "ball": {
#                 "material_type": "rigid",
#                 "sub_type": "rubber",
#                 "params": {
#                     "mass": 0.5,
#                     "friction": 0.4,
#                     "restitution": 0.7,
#                 },
#             },
#             "bottle": {
#                 "material_type": "rigid",
#                 "sub_type": "glass",
#                 "params": {
#                     "mass": 1.0,
#                     "friction": 0.6,
#                     "restitution": 0.2,
#                 },
#             },
#         }
#     }

#     agent = ForceInferenceAgent(
#         api_key=os.environ.get("ANTHROPIC_API_KEY", "dummy"),
#         base_url=os.environ.get("ANTHROPIC_BASE_URL"),  # your server URL
#         model="qwen3-small",
#     )

#     result = agent.infer_forces(
#         prompt=prompt,
#         scene=scene,
#         material_map=material_map,
#         run_directory="./run_output",
#     )

#     print(json.dumps(result, indent=2))

"""
stage2/force_agent.py

ForceInferenceAgent
━━━━━━━━━━━━━━━━━━━

This agent predicts:
- initial linear velocity
- initial angular velocity
- linear damping
- angular damping

for Blender rigid-body simulation.

Inputs:
- natural-language simulation prompt
- scene objects with:
    - label
    - mass
    - friction
    - restitution
    - collision_shape
    - position
    - rotation
    - scale

Blender handles:
- gravity
- collisions
- rigid-body simulation
- momentum transfer
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union

from anthropic import Anthropic


SYSTEM_PROMPT = """/no_think
You are a JSON-only API endpoint for a Blender physics simulation pipeline.

YOU MUST RESPOND WITH ONLY A JSON OBJECT.
NO explanations. NO bullet points. NO markdown. NO text of any kind.
YOUR ENTIRE RESPONSE MUST START WITH { AND END WITH }

If you write any text outside the JSON object, the pipeline will crash.

TASK:
Given a simulation prompt and scene objects, predict:
- initial_linear_velocity
- initial_angular_velocity
- linear_damping
- angular_damping

CRITICAL RULE — DIRECTION CALCULATION:
Compute velocity direction mathematically from positions.

Steps:
1. Identify the moving object and target object from the prompt
2. Compute direction vector:
   dx = target.position[0] - source.position[0]
   dy = target.position[1] - source.position[1]
   dz = target.position[2] - source.position[2]
3. Normalize:
   magnitude = sqrt(dx^2 + dy^2 + dz^2)
   direction = [dx/magnitude, dy/magnitude, dz/magnitude]
4. Scale by realistic speed (3.0 - 6.0 for a ball hit)
5. Use as initial_linear_velocity

CRITICAL RULE — STATIONARY OBJECTS:
- Only the MOVING object gets non-zero velocity
- ALL OTHER objects: initial_linear_velocity = [0.0, 0.0, 0.0]
- NEVER assign post-collision velocity to stationary objects

MATERIAL GUIDANCE — use sub_type to tune damping:
- glass/ceramic  -> linear_damping: 0.01-0.03, angular_damping: 0.01-0.03
- rubber/soft    -> linear_damping: 0.1-0.2,   angular_damping: 0.1-0.2
- cardboard/wood -> linear_damping: 0.04-0.08, angular_damping: 0.05-0.1
- metal          -> linear_damping: 0.02-0.05, angular_damping: 0.02-0.05

Coordinate system: +X=right, +Y=depth, +Z=up
Velocity limits: linear [-10, 10], angular [-20, 20]

OUTPUT FORMAT — your response must be exactly this structure, nothing else:
{"objects":{"object_0":{"label":"ball","initial_linear_velocity":[3.578,1.789,0.0],"initial_angular_velocity":[0.0,0.0,0.0],"linear_damping":0.1,"angular_damping":0.1},"object_1":{"label":"bottle","initial_linear_velocity":[0.0,0.0,0.0],"initial_angular_velocity":[0.0,0.0,0.0],"linear_damping":0.02,"angular_damping":0.02}}}
"""


class ForceInferenceAgent:
    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-small",
        base_url: Optional[str] = None,
    ):
        self.client = Anthropic(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model

    def infer_forces(
        self,
        prompt: str,
        scene: Dict[str, Any],
        material_map: Optional[Dict[str, Any]] = None,
        run_directory: Union[str, Path] = "./",
    ) -> Dict[str, Any]:
        """
        Predict initial motion states for Blender rigid-body simulation.
        Saves results to force_spec.json in run_directory.
        """

        # Merge material_map into scene objects for richer context
        enriched_scene = self._enrich_scene_with_materials(scene, material_map)

        user_message = f"""/no_think
Simulation Prompt:
{prompt}

Scene Objects (with material properties):
{json.dumps(enriched_scene, indent=2)}

Infer physically plausible motion initialization parameters.
"""
        print(user_message)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        text = response.content[0].text
        print("=== Raw model response ===")
        print(text)
        print("==========================")

        motion_spec = self._parse_json(text)
        motion_spec = self._clamp_values(motion_spec)

        # Save force_spec.json with material properties merged in
        self._save_force_spec(motion_spec, enriched_scene, run_directory)

        return motion_spec

    def _enrich_scene_with_materials(
        self,
        scene: Dict[str, Any],
        material_map: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Merge material_map properties into each scene object.
        Returns enriched scene dict.
        """
        if not material_map:
            return scene

        enriched = json.loads(json.dumps(scene))  # deep copy

        mat = material_map.get("material_map", material_map)

        for obj_key, obj in enriched.get("objects", {}).items():
            label = obj.get("label", obj_key)

            # Try matching by label or obj_key
            mat_entry = mat.get(label) or mat.get(obj_key)

            if mat_entry:
                obj["material_type"] = mat_entry.get("material_type")
                obj["sub_type"]      = mat_entry.get("sub_type")

                # Overwrite physics params from material_map if present
                params = mat_entry.get("params", {})
                if "mass"        in params: obj["mass"]        = params["mass"]
                if "friction"    in params: obj["friction"]    = params["friction"]
                if "restitution" in params: obj["restitution"] = params["restitution"]

        return enriched

    def _save_force_spec(
        self,
        motion_spec: Dict[str, Any],
        enriched_scene: Dict[str, Any],
        run_directory: Union[str, Path],
    ) -> Path:
        """
        Save force_spec.json to run_directory.
        Includes both velocity parameters and material properties.
        Separate from material_map.json.
        """

        # Merge material properties from enriched scene into motion_spec
        for obj_key, obj in motion_spec.get("objects", {}).items():
            scene_obj = enriched_scene.get("objects", {}).get(obj_key, {})
            obj["material_type"]   = scene_obj.get("material_type")
            obj["sub_type"]        = scene_obj.get("sub_type")
            obj["mass"]            = scene_obj.get("mass")
            obj["friction"]        = scene_obj.get("friction")
            obj["restitution"]     = scene_obj.get("restitution")
            obj["collision_shape"] = scene_obj.get("collision_shape")
            obj["position"]        = scene_obj.get("position")
            obj["rotation"]        = scene_obj.get("rotation")
            obj["scale"]           = scene_obj.get("scale")

        out_dir  = Path(run_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "force_spec.json"

        out_path.write_text(
            json.dumps(motion_spec, indent=2),
            encoding="utf-8",
        )
        print(f"force_spec.json written -> {out_path}")
        return out_path

    def _strip_markdown(self, text: str) -> str:
        """
        Remove markdown code fences (```json ... ``` or ``` ... ```)
        so JSON can be extracted cleanly.
        """
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*',     '', text)
        return text.strip()

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """
        Extract and normalize JSON from model response.
        Handles markdown fences and different key formats.
        """
        # Strip markdown backticks first
        text = self._strip_markdown(text)

        match = re.search(r'\{.*\}', text, re.DOTALL)

        if not match:
            raise ValueError(
                f"No JSON found in model response. "
                f"Response was: {repr(text[:300])}"
            )

        try:
            raw = json.loads(match.group())
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Found JSON block but failed to parse (likely truncated): {e}\n"
                f"JSON blob: {match.group()[:300]}"
            )

        return self._normalize_response(raw)

    def _normalize_response(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize model response into standard format regardless of
        what structure the model returns.

        Handles:
        - correct format:  {"objects": {"object_0": {"initial_linear_velocity": ...}}}
        - short keys:      {"objects": {"object_0": {"linear_velocity": ...}}}
        - label-keyed:     {"ball": {"linear_velocity": ...}, "bottle": {...}}
        """

        # Already has "objects" wrapper
        if "objects" in raw:
            for obj in raw["objects"].values():
                # Normalize short key names
                if "linear_velocity" in obj and "initial_linear_velocity" not in obj:
                    obj["initial_linear_velocity"] = obj.pop("linear_velocity")
                if "angular_velocity" in obj and "initial_angular_velocity" not in obj:
                    obj["initial_angular_velocity"] = obj.pop("angular_velocity")
            return raw

        # Model returned label-keyed dict: {"ball": {...}, "bottle": {...}}
        normalized = {"objects": {}}
        for idx, (label, props) in enumerate(raw.items()):
            obj_key = f"object_{idx}"
            normalized["objects"][obj_key] = {
                "label": label,
                "initial_linear_velocity":  props.get("linear_velocity",
                                            props.get("initial_linear_velocity", [0.0, 0.0, 0.0])),
                "initial_angular_velocity": props.get("angular_velocity",
                                            props.get("initial_angular_velocity", [0.0, 0.0, 0.0])),
                "linear_damping":           props.get("linear_damping",  0.04),
                "angular_damping":          props.get("angular_damping", 0.1),
            }

        return normalized

    def _clamp_values(
        self,
        spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Clamp values to prevent unstable Blender physics.
        """
        if "objects" not in spec:
            return spec

        for obj in spec["objects"].values():

            lv = obj.get("initial_linear_velocity", [0.0, 0.0, 0.0])
            obj["initial_linear_velocity"] = [
                max(-10.0, min(10.0, float(v))) for v in lv
            ]

            av = obj.get("initial_angular_velocity", [0.0, 0.0, 0.0])
            obj["initial_angular_velocity"] = [
                max(-20.0, min(20.0, float(v))) for v in av
            ]

            obj["linear_damping"] = max(
                0.0, min(0.3, float(obj.get("linear_damping", 0.04)))
            )

            obj["angular_damping"] = max(
                0.0, min(0.5, float(obj.get("angular_damping", 0.1)))
            )

        return spec


if __name__ == "__main__":

    prompt = "A ball hits a bottle from the left."

    scene = {
        "objects": {
            "object_0": {
                "label": "ball",
                "mass": 0.5,
                "friction": 0.4,
                "restitution": 0.7,
                "collision_shape": "SPHERE",
                "position": [0.0, 0.0, 0.5],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "scale": [0.15, 0.15, 0.15],
            },
            "object_1": {
                "label": "bottle",
                "mass": 1.0,
                "friction": 0.6,
                "restitution": 0.2,
                "collision_shape": "CONVEX_HULL",
                "position": [2.0, 1.0, 0.5],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "scale": [0.2, 0.2, 0.5],
            },
        }
    }

    material_map = {
        "material_map": {
            "ball": {
                "material_type": "rigid",
                "sub_type": "rubber",
                "params": {
                    "mass": 0.5,
                    "friction": 0.4,
                    "restitution": 0.7,
                },
            },
            "bottle": {
                "material_type": "rigid",
                "sub_type": "glass",
                "params": {
                    "mass": 1.0,
                    "friction": 0.6,
                    "restitution": 0.2,
                },
            },
        }
    }

    agent = ForceInferenceAgent(
        api_key=os.environ.get("ANTHROPIC_API_KEY", "dummy"),
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        model="qwen3-small",
    )

    result = agent.infer_forces(
        prompt=prompt,
        scene=scene,
        material_map=material_map,
        run_directory=Path("./run_output"),
    )

    print(json.dumps(result, indent=2))