"""
stage1/semantic_relevance_agent.py

Semantic Relevance Agent
━━━━━━━━━━━━━━━━━━━━━━━
Receives candidate masks produced by SAM2 (via uniform point sampling) and
decides, for each one, whether the segmented region is relevant to the user's
physics simulation prompt.

Flow
────
  for each mask:
    1. Build an overlay image: original scene with the mask region tinted green.
    2. Send overlay + prompt to Claude (vision).
    3. Parse keep / label / reason / relevance_score from the response.
  Return the masks where keep == True and relevance_score >= threshold.
"""

from __future__ import annotations

import base64
import json
import re
import os
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from utils.shared import ClaudeAgent, get_logger, save_state

_log = get_logger("stage1.semantic_relevance")

_EVAL_SYSTEM = f"""
You are the Semantic Relevance Agent in a physics simulation pipeline.

You will receive an image where ONE region is highlighted with a green tint,
together with a physics simulation prompt.

Decide whether the highlighted region is relevant to the simulation.

Respond with ONLY a JSON object — no markdown, no explanation:
{{
  "keep": true or false,
  "label": "short generic type (e.g. box, person, bottle, table)",
  "object_name": "specific descriptive/positional name (e.g. topmost box, second box from left, person in foreground, red bottle on right)",
  "description": "one sentence describing what this object is and where it sits in the scene",
  "reason": "one sentence explaining the relevance decision",
  "relevance_score": <float 0.0–1.0>
}}

Guidelines for object_name:
  - Use position when objects are of the same type: "topmost box", "bottom-left box", "second box from left"
  - Use appearance when objects differ: "red ball", "tall bottle", "wooden table"
  - Use role when clear: "person holding box", "falling object"

Scoring guide:
  0.8–1.0  directly involved in the physics interaction
  0.4–0.7  indirectly affected (surface, container, environment)
  0.0–0.39 background / irrelevant clutter

Set "keep" to true when relevance_score >= {cfg.RELEVANCE_SCORE_THRESHOLD}.
""".strip()


def _encode_image(path: str | Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for a local image file."""
    path = Path(path)
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(path.suffix.lower(), "image/jpeg")
    with open(path, "rb") as fh:
        data = base64.standard_b64encode(fh.read()).decode()
    return data, media_type


def _build_overlay(image_path: str | Path, mask_path: str | Path) -> Path:
    """
    Tint the masked pixels green and save the result alongside the mask PNG.

    Returns the path to the overlay image.
    """
    image_path = Path(image_path)
    mask_path  = Path(mask_path)

    scene = PILImage.open(image_path).convert("RGB")
    scene_np = np.array(scene, dtype=np.float32)

    mask = PILImage.open(mask_path).convert("L")
    # Resize mask to scene dimensions if they differ
    if mask.size != scene.size:
        mask = mask.resize(scene.size, PILImage.NEAREST)
    mask_np = np.array(mask) > 128  # bool

    overlay = scene_np.copy()
    # Blend masked pixels toward green: keep 40 % original + 60 % green channel boost
    overlay[mask_np, 0] = overlay[mask_np, 0] * 0.4               # R ↓
    overlay[mask_np, 1] = np.clip(overlay[mask_np, 1] * 0.4 + 153, 0, 255)  # G ↑
    overlay[mask_np, 2] = overlay[mask_np, 2] * 0.4               # B ↓

    result_img = PILImage.fromarray(overlay.astype(np.uint8))
    out_path = mask_path.parent / f"{mask_path.stem}_overlay.png"
    result_img.save(out_path)
    return out_path


def _parse_decision(text: str) -> dict:
    """Extract the JSON decision dict from Claude's response."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    _log.warning("Could not parse decision JSON; defaulting to keep=False")
    return {
        "keep": False,
        "label": "unknown",
        "reason": "failed to parse model response",
        "relevance_score": 0.0,
    }


class SemanticRelevanceAgent(ClaudeAgent):
    """
    Evaluates each SAM2 mask for relevance to the simulation prompt.

    Call `run_for_scene(image_path, masks, prompt, run_directory)` to get
    the filtered list of relevant masks.
    """

    name = "semantic_relevance"

    def __init__(self) -> None:
        super().__init__()
        # No tools needed — each mask gets its own direct vision call.
        self.tools = []
        self.tool_handlers = {}

    # ── public ────────────────────────────────────────────────────────────────

    def run_for_scene(
        self,
        image_path: str,
        masks: list[dict],
        prompt: str,
        run_directory: Path,
    ) -> dict:
        """
        Evaluate every mask and keep the relevant ones.

        Parameters
        ----------
        image_path:     Path to the original scene image.
        masks:          List of mask dicts produced by SAM2.  Each must contain
                        at least ``mask_path`` (path to the PNG file).
        prompt:         User's physics simulation prompt.
        run_directory:  Workspace directory for this run.

        Returns
        -------
        {
          "relevant_objects": [...],   # kept masks with label / score / bbox
          "masks":            [...],   # same items (full mask dicts)
        }
        """
        if not masks:
            _log.warning("No masks provided to SemanticRelevanceAgent.")
            save_state(run_directory, "relevant_objects", {"relevant_objects": []})
            save_state(run_directory, "masks", {"masks": []})
            return {"relevant_objects": [], "masks": []}

        _log.info(
            "Evaluating %d candidate mask(s) against prompt: %r",
            len(masks), prompt,
        )

        relevant_masks: list[dict] = []

        for i, mask_info in enumerate(masks):
            mask_path = mask_info.get("mask_path")
            if not mask_path or not Path(mask_path).exists():
                _log.warning("Mask %d has no valid mask_path — skipping.", i)
                continue

            _log.info("  Evaluating mask %d / %d  (%s)", i + 1, len(masks), mask_path)
            decision = self._evaluate_mask(image_path, mask_info, prompt)

            _log.info(
                "    → keep=%s  label=%r  score=%.2f  reason=%s",
                decision.get("keep"),
                decision.get("label"),
                decision.get("relevance_score", 0.0),
                decision.get("reason", ""),
            )

            if decision.get("keep", False):
                relevant_masks.append({
                    **mask_info,
                    "label":           decision.get("label", f"object_{i}"),
                    "object_name":     decision.get("object_name", decision.get("label", f"object_{i}")),
                    "description":     decision.get("description", ""),
                    "reason":          decision.get("reason", ""),
                    "relevance_score": float(decision.get("relevance_score", 0.0)),
                })

        _log.info(
            "%d / %d masks kept as relevant.",
            len(relevant_masks), len(masks),
        )

        # Build relevant_objects list for downstream compatibility
        relevant_objects = [
            {
                "label":           m["label"],
                "object_name":     m.get("object_name", m["label"]),
                "description":     m.get("description", ""),
                "relevance_score": m["relevance_score"],
                "reason":          m.get("reason", ""),
                "bbox":            m.get("bbox"),
            }
            for m in relevant_masks
        ]

        save_state(run_directory, "relevant_objects", {"relevant_objects": relevant_objects})
        save_state(run_directory, "masks", {"masks": relevant_masks})

        return {
            "relevant_objects": relevant_objects,
            "masks":            relevant_masks,
        }

    # ── internal ──────────────────────────────────────────────────────────────

    def _evaluate_mask(
        self,
        image_path: str | Path,
        mask_info: dict,
        prompt: str,
    ) -> dict:
        """
        Build an overlay image for this mask and ask Claude whether to keep it.
        """
        mask_path = mask_info["mask_path"]

        # Build overlay (green tint on masked region)
        try:
            overlay_path = _build_overlay(image_path, mask_path)
        except Exception as exc:
            _log.error("Failed to build overlay for %s: %s", mask_path, exc)
            return {"keep": False, "label": "unknown", "reason": str(exc), "relevance_score": 0.0}

        # Encode overlay as base64
        overlay_data, overlay_media = _encode_image(overlay_path)

        area       = mask_info.get("area", "unknown")
        confidence = mask_info.get("confidence", 0.0)

        user_content = [
            {
                "type": "image",
                "source": {
                    "type":       "base64",
                    "media_type": overlay_media,
                    "data":       overlay_data,
                },
            },
            {
                "type": "text",
                "text": (
                    f"Physics simulation prompt: {prompt}\n\n"
                    f"The green-highlighted region covers {area} pixels "
                    f"(SAM2 confidence: {confidence:.2f}).\n\n"
                    "Is this highlighted region relevant to the simulation? "
                    "Reply with ONLY the JSON object described in your instructions."
                ),
            },
        ]

        response = self.client.messages.create(
            model=cfg.CLAUDE_MODEL,
            max_tokens=512,
            system=_EVAL_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )

        raw = response.content[0].text if response.content else "{}"
        return _parse_decision(raw)


# ── quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob as _glob

    image  = os.environ.get("TEST_IMAGE", "scene.jpg")
    prompt = os.environ.get(
        "TEST_PROMPT",
        "A white ball rolls from the left and knocks over a bottle on a red surface",
    )
    out_dir = Path("./test_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a synthetic mask list from any PNG files in ./masks/
    mask_pngs = sorted(_glob.glob("masks/*.png"))
    if not mask_pngs:
        print("No mask PNGs found — place mask files in ./masks/ to test.")
    else:
        fake_masks = [
            {
                "idx":        i,
                "mask_path":  p,
                "area":       10000,
                "confidence": 0.85,
                "bbox":       [0, 0, 100, 100],
            }
            for i, p in enumerate(mask_pngs)
        ]

        agent  = SemanticRelevanceAgent()
        result = agent.run_for_scene(
            image_path=image,
            masks=fake_masks,
            prompt=prompt,
            run_directory=out_dir,
        )

        print("\n=== Relevant Objects ===")
        for obj in result["relevant_objects"]:
            print(
                f"  {obj['label']:15s}"
                f"  name={obj.get('object_name',''):30s}"
                f"  score={obj['relevance_score']:.2f}"
                f"  desc={obj.get('description','')}"
            )
