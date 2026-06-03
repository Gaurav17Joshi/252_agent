"""
stage2/material_agent.py

Material Classification Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Classifies physical material properties for a single scene object by
sending its annotated image alongside the simulation prompt to a
vision-capable Claude model.

Inputs (per call)
─────────────────
  annotated_image_path  — path to the image with the object highlighted
  object_label          — string identifier for the object
  simulation_prompt     — user's original simulation description

Output (per call)
─────────────────
{
  "<object_label>": {
    "material_type": "rigid",
    "sub_type":      "metal",
    "params": {
      "mass":        2.0,   # kg
      "friction":    0.5,   # 0–1
      "restitution": 0.1    # 0–1
    }
  }
}

Material types
──────────────
  rigid      — non-deforming solids (metal, wood, rock, plastic, glass)
  fluid      — liquids and gases   (water, oil, smoke, fire)
  deformable — soft bodies         (cloth, rubber, jelly, tissue)
  granular   — particulate matter  (sand, gravel, powder)
"""
# ── Qwen3.5-4B via local vLLM / OpenAI-compatible endpoint ───────────────
# Server launched with:
#   vllm serve Qwen/Qwen3.5-4B-Instruct \
#     --tensor-parallel-size 1 \
#     --max-model-len 1048576 \
#     --trust-remote-code \
#     --reasoning-parser qwen          # separates <think> from final content
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Optional

import anthropic

import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.shared import get_logger

_log = get_logger("stage2.material")
_VISION_MODEL   = "gpt-5.1"
_VISION_MODEL   = "gpt-4o"
_VISION_MODEL   = "qwen3-small"
# _VISION_MODEL   = "gpt-oss"
# _VLLM_BASE_URL  = "http://localhost:8000/v1"   # adjust if remote
_VLLM_BASE_URL  = os.environ.get("OPENAI_BASE_URL")   # adjust if remote
_VLLM_API_KEY   = os.environ.get("OPENAI_API_KEY")   or "EMPTY"                  # vLLM ignores this value
SYSTEM_PROMPT = """
You are a material-classification expert embedded in a physics-simulation pipeline.
You will receive:
  1. An image of a scene with ONE object highlighted or annotated.
  2. A simulation prompt describing the physical scenario.
  3. The object's label.

Your task:
  - Identify the material of the highlighted object from its visual appearance,
    shape, context, and the simulation prompt.
  - Assign a material_type and a more specific sub_type.
  - Assign an object class (e.g. "glass_bottle", "rubber_ball", "wooden_table") based on what you see.
  - Predict physically realistic simulation parameters:
      mass        (kg)  — realistic absolute mass for this object
      friction    (0–1) — surface friction coefficient
      restitution (0–1) — bounciness (0 = no bounce, 1 = perfect elastic)

Material types:
  rigid      — non-deforming solids
  fluid      — liquids and gases
  deformable — soft bodies
  granular   — particulate matter

Guidelines:
  - Derive values from what you actually SEE in the image plus context.
  - Use specific, realistic numbers — never placeholder or default values.
  - Examples of good predictions:
      bowling ball  → rigid/ceramic,  mass=6.0,  friction=0.15, restitution=0.05
      rubber ball   → rigid/rubber,   mass=0.06, friction=0.8,  restitution=0.85
      throw pillow  → deformable/foam,mass=0.4,  friction=0.6,  restitution=0.05
      glass bottle  → rigid/glass,    mass=0.5,  friction=0.4,  restitution=0.05
      sand pile     → granular/sand,  mass=5.0,  friction=0.55, restitution=0.02

Return ONLY valid JSON — no prose, no markdown fences:
{
  "material_type": "rigid",
  "sub_type": "metal",
  "object_class": "shiny_metal_pan",
  "params": {
    "mass": 2.0,
    "friction": 0.5,
    "restitution": 0.1
  }
}
"""


def _call_qwen(user_content: list[dict]) -> tuple[str, str]:
    """
    Call the Qwen3.5-4B vision model with thinking enabled.

    Returns
    -------
    (reasoning_content, answer_content)
        reasoning_content — the model's internal <think> trace (may be "")
        answer_content    — the final JSON answer
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=_VLLM_BASE_URL,
        api_key=_VLLM_API_KEY,
    )

    # user_content = "hello"
    # print(user_content)

    response = client.chat.completions.create(
        model=_VISION_MODEL,
        messages=[
            # System role injected as first user turn because some
            # vLLM builds drop the system field in multimodal mode.
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,   # list with image_url + text blocks
            },
        ],
        max_tokens=8192,        # budget: up to ~6 k think + ~512 answer
        temperature=1.0,        # recommended for thinking mode (Qwen docs)
        top_p=0.95,
        presence_penalty=1.5,   # reduces repetition in long think traces
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {
                "enable_thinking": True,    # emit <think>…</think> block
            },
        },
    )

    msg = response.choices[0].message

    # reasoning_content is populated only when --reasoning-parser qwen
    # is active on the server AND enable_thinking=True was passed.
    reasoning = getattr(msg, "reasoning_content", "") or ""
    answer    = msg.content or ""

    return reasoning, answer



class MaterialClassificationAgent:
    """
    Classify one object at a time using vision inference.

    Usage
    -----
    agent = MaterialClassificationAgent()
    result = agent.classify_object(
        annotated_image_path=Path("run/annotated/bottle.png"),
        object_label="glass_bottle",
        simulation_prompt="A bottle falls off a table onto a wooden floor.",
    )
    """

    name = "material_agent"

    def __init__(self) -> None:
        self._client = anthropic.Anthropic()

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def classify_object(
        self,
        annotated_image_path: Path,
        object_label: str,
        simulation_prompt: str,
    ) -> dict:
        _log.info("Classifying material for '%s'", object_label)

        image_data, media_type = self._load_image(annotated_image_path)
        _log.info(
            "Image loaded: path=%s media_type=%s b64_len=%d",
            annotated_image_path, media_type, len(image_data)
        )
        if len(image_data) < 100:
            raise ValueError(f"Image data suspiciously small for {annotated_image_path}")
        # Build OpenAI-style multimodal content list
        user_content = [
            {
                "type": "image_url",
                "image_url": {
                    # vLLM accepts base64 data URIs
                    "url": f"data:{media_type};base64,{image_data}",
                },
            },
            {
                "type": "text",
                "text": (
                    f"Object label: {object_label}\n"
                    f"Simulation prompt: {simulation_prompt}\n\n"
                    "Classify the highlighted object and predict its "
                    "physics parameters."
                ),
            },
        ]

        reasoning, raw_text = _call_qwen(user_content)

        if reasoning:
            _log.debug(
                "Reasoning trace for '%s' (%d chars): %s…",
                object_label,
                len(reasoning),
                reasoning[:200],
            )

        _log.debug("Answer for '%s': %s", object_label, raw_text)

        spec = self._parse_spec(raw_text, object_label)

        _log.info(
            "  %s → %s/%s  mass=%.2f  friction=%.2f  restitution=%.2f",
            object_label,
            spec["material_type"],
            spec["sub_type"],
            spec["params"]["mass"],
            spec["params"]["friction"],
            spec["params"]["restitution"],
        )

        return spec

    # ─────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _load_image(path: Path) -> tuple[str, str]:
        """
        Load an image file and return (base64_data, media_type).
        Supports JPEG and PNG.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        media_type_map = {
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png":  "image/png",
            ".webp": "image/webp",
            ".gif":  "image/gif",
        }
        media_type = media_type_map.get(suffix, "image/jpeg")

        with open(path, "rb") as fh:
            data = base64.standard_b64encode(fh.read()).decode("utf-8")

        return data, media_type

    @staticmethod
    def _parse_spec(text: str, label: str) -> dict:
        """
        Extract and validate the material spec JSON from model output.
        Strips markdown fences if present, then parses JSON.
        """
        # Strip ```json ... ``` or ``` ... ``` fences
        text = re.sub(r"```[a-zA-Z]*\n?", "", text).strip()

        # Find the outermost JSON object
        start = text.find("{")
        end   = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(
                f"No JSON object found in model response for '{label}': {text!r}"
            )

        blob = json.loads(text[start : end + 1])

        # Validate required keys
        required_top   = {"material_type", "sub_type", "params"}
        required_params = {"mass", "friction", "restitution"}

        missing_top = required_top - blob.keys()
        if missing_top:
            raise ValueError(
                f"Model response for '{label}' missing keys: {missing_top}"
            )

        missing_params = required_params - blob["params"].keys()
        if missing_params:
            raise ValueError(
                f"Model response for '{label}' params missing: {missing_params}"
            )

        # Clamp friction and restitution to [0, 1]
        blob["params"]["friction"]    = max(0.0, min(1.0, float(blob["params"]["friction"])))
        blob["params"]["restitution"] = max(0.0, min(1.0, float(blob["params"]["restitution"])))
        blob["params"]["mass"]        = max(0.0, float(blob["params"]["mass"]))

        return blob