"""
stage1/sam2_agent.py

SAM2 Agent
━━━━━━━━━━
Calls the local SAM2 wrapper to generate segmentation masks for every
object that passed the Semantic Relevance Agent's threshold.

Claude acts as the orchestration layer: it decides which prompts to
pass to SAM2, interprets low-confidence results, and can request a retry
with a tighter prompt if confidence is too low.

Tools exposed to Claude
───────────────────────
  run_sam2_segmentation   — call SAM2 on one or more objects
  validate_mask_quality   — flag masks whose confidence < 0.6 for retry
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.shared import ClaudeAgent, get_logger, save_state
from utils.sam2_wrapper import generate_masks

_log = get_logger("stage1.sam2_agent")

SYSTEM_PROMPT = """
You are the SAM2 Segmentation Agent in a physics-simulation pipeline.

Your job:
1. Receive a list of relevant scene objects (each with a label, relevance
   score, and optional bounding box).
2. Call run_sam2_segmentation to generate precise segmentation masks.
3. Call validate_mask_quality to check for low-confidence masks.
4. If any masks fail quality validation, retry run_sam2_segmentation ONCE
   for those failing objects with a tighter bounding-box prompt.
5. Return the final list of masks as a JSON object with key "masks".

Quality threshold: confidence >= 0.60. Masks below this threshold must be
retried once with a refined bounding-box prompt if possible.

Your final message MUST be a JSON object of the form:
{"masks": [ ... ]}
No explanation, no markdown — just the JSON object.
"""

MIN_CONFIDENCE = 0.60
MAX_RETRIES = 1


class SAM2Agent(ClaudeAgent):
    name = "sam2_agent"

    def __init__(self) -> None:
        super().__init__()
        self._masks: list[dict] = []

        self.tools = [
            {
                "name": "run_sam2_segmentation",
                "description": (
                    "Run SAM2 locally to generate segmentation masks for the "
                    "given objects. Returns a list of mask dicts."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "objects": {
                            "type": "array",
                            "description": (
                                "List of objects to segment. Each must have "
                                "'label' and 'relevance_score'; optionally 'bbox'."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "relevance_score": {"type": "number"},
                                    "bbox": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                    },
                                },
                                "required": ["label", "relevance_score"],
                            },
                        },
                    },
                    "required": ["image_path", "objects"],
                },
            },
            {
                "name": "validate_mask_quality",
                "description": (
                    "Check each mask's confidence score. "
                    "Returns lists of passing and failing masks."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "masks": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                        "min_confidence": {
                            "type": "number",
                            "description": "Minimum acceptable confidence (0–1).",
                        },
                    },
                    "required": ["masks"],
                },
            },
        ]

        self.tool_handlers = {
            "run_sam2_segmentation": self._run_sam2_segmentation,
            "validate_mask_quality": self._validate_mask_quality,
        }

    # ── public ────────────────────────────────────────────────────────────────

    def run_for_image(
        self,
        image_path: str,
        run_directory: Path,
    ) -> list[dict]:
        """
        Generate candidate masks for the whole image via uniform point sampling.

        Calls the SAM2 automatic mask generator (no prior object labels needed)
        and returns all raw masks for downstream filtering by the Semantic
        Relevance Agent.

        Writes ``candidate_masks.json`` to *run_directory*.
        """
        from utils.sam2_wrapper import generate_masks as _gen

        _log.info("SAM2: generating candidate masks for image: %s", image_path)

        # Pass a single sentinel object so generate_masks runs automatic mode;
        # the wrapper's automatic generator returns all masks it finds.
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            import torch
            from PIL import Image as _PILImage
            import numpy as _np
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            import config as _cfg
            from utils.sam2_wrapper import _save_mask, _mask_to_rle

            device     = "cuda" if torch.cuda.is_available() else "cpu"
            sam2_model = build_sam2(
                config_file=_cfg.SAM2_CONFIG,
                ckpt_path=str(_cfg.SAM2_CHECKPOINT),
                device=device,
            )
            mask_gen = SAM2AutomaticMaskGenerator(
                model=sam2_model,
                points_per_side=32,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.85,
                min_mask_region_area=500,
            )

            image_np  = _np.array(_PILImage.open(image_path).convert("RGB"))
            auto_masks = mask_gen.generate(image_np)

            mask_dir = run_directory / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)

            results: list[dict] = []
            for idx, am in enumerate(auto_masks):
                seg  = am["segmentation"].astype(bool)
                ys, xs = _np.where(seg)
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                label = f"region_{idx:03d}"
                mask_path = _save_mask(seg, label, mask_dir)
                results.append({
                    "idx":        idx,
                    "mask_path":  str(mask_path),
                    "confidence": float(am["predicted_iou"]),
                    "area":       int(am["area"]),
                    "bbox":       bbox,
                    "rle":        _mask_to_rle(seg),
                })

        except ImportError:
            _log.warning("SAM2 not available — returning empty candidate mask list.")
            results = []

        save_state(run_directory, "candidate_masks", {"candidate_masks": results})
        _log.info("SAM2 produced %d candidate masks.", len(results))
        return results

    def run_for_objects(
        self,
        image_path: str,
        relevant_objects: list[dict],
        run_directory: Path,
    ) -> list[dict]:
        """Generate masks and write ``masks.json`` to *run_directory*."""

        user_msg = (
            f"Image path: {image_path}\n\n"
            f"Objects to segment ({len(relevant_objects)}):\n"
            + json.dumps(relevant_objects, indent=2)
            + "\n\nGenerate masks, validate quality, retry any failures, "
            "and return the final masks as JSON."
        )

        raw_response = self.run(system=SYSTEM_PROMPT, user_message=user_msg)

        # Ensure we have a plain string to parse
        if not isinstance(raw_response, str):
            raw_response = str(raw_response)

        masks = self._parse_masks(raw_response)

        # Fallback: use internally stored masks accumulated via tool calls
        if not masks and self._masks:
            _log.warning(
                "Could not parse masks from Claude's final response — "
                "falling back to tool-call results."
            )
            masks = self._masks

        if not masks:
            _log.error("No masks produced for image: %s", image_path)

        save_state(run_directory, "masks", {"masks": masks})
        _log.info("Masks generated for %d objects", len(masks))
        return masks

    # ── tool handlers ─────────────────────────────────────────────────────────

    def _run_sam2_segmentation(
        self,
        image_path: str,
        objects: list[dict],
    ) -> dict:
        _log.info("SAM2 segmenting %d objects from %s", len(objects), image_path)
        masks = generate_masks(image_path=image_path, relevant_objects=objects)

        # Accumulate masks across calls (initial run + retries)
        existing_labels = {m.get("label") for m in self._masks}
        for m in masks:
            if m.get("label") in existing_labels:
                # Replace old low-confidence mask with the retry result
                self._masks = [
                    m if existing.get("label") == m.get("label") else existing
                    for existing in self._masks
                ]
            else:
                self._masks.append(m)

        return {"status": "ok", "mask_count": len(masks), "masks": masks}

    def _validate_mask_quality(
        self,
        masks: list[dict],
        min_confidence: float = MIN_CONFIDENCE,
    ) -> dict:
        passing = [m for m in masks if m.get("confidence", 0) >= min_confidence]
        failing = [m for m in masks if m.get("confidence", 0) < min_confidence]
        _log.info(
            "Mask quality: %d passing, %d failing (threshold=%.2f)",
            len(passing), len(failing), min_confidence,
        )
        return {
            "passing": passing,
            "failing": failing,
            "all_pass": len(failing) == 0,
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_masks(text: str) -> list[dict]:
        """
        Robustly extract a mask list from Claude's final response.

        Handles:
        - clean JSON:  {"masks": [...]}
        - bare list:   [{"label": ...}, ...]
        - trailing text after the JSON block
        """
        # Try to extract a JSON object first (most expected case)
        obj_match = re.search(r'\{.*\}', text, re.DOTALL)
        if obj_match:
            try:
                blob = json.loads(obj_match.group())
                for key in ("masks", "passing", "results"):
                    if key in blob and isinstance(blob[key], list):
                        return blob[key]
            except json.JSONDecodeError:
                pass

        # Fall back to a bare JSON array
        arr_match = re.search(r'\[.*\]', text, re.DOTALL)
        if arr_match:
            try:
                blob = json.loads(arr_match.group())
                if isinstance(blob, list):
                    return blob
            except json.JSONDecodeError:
                pass

        return []