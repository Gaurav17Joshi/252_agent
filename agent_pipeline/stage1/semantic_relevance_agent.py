"""
stage1/semantic_relevance_agent.py

Semantic Relevance Agent
━━━━━━━━━━━━━━━━━━━━━━━
Analyses the input image (via vision) and the text prompt to decide
which objects in the scene are relevant to the requested physics simulation.

Each object receives a relevance score in [0, 1].  Objects below
``cfg.RELEVANCE_SCORE_THRESHOLD`` are filtered out before masking.

Tools exposed to Claude
───────────────────────
  detect_objects        — run a vision detection pass on the image
  score_relevance       — assign relevance scores + bounding boxes given the prompt
  filter_by_threshold   — drop objects below the configured threshold
"""

from __future__ import annotations

import base64
from PIL import Image as PILImage
import json
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from utils.shared import ClaudeAgent, get_logger, save_state
from stage1.sam2_agent import SAM2Agent

_log = get_logger("stage1.semantic_relevance")

SYSTEM_PROMPT = """
You are the Semantic Relevance Agent in a physics-simulation pipeline.

Your job:
1. Examine the scene image provided.
2. Read the user's physics simulation prompt.
3. Detect every distinct physical object visible in the image.
4. For each object, provide a tight bounding box in pixel coordinates [x1, y1, x2, y2].
5. Score every detected object from 0.0 (irrelevant) to 1.0 (critical) based
   on how important each is to the requested simulation.
6. Filter out objects with a score below {threshold}.
7. Return a clean ranked list of relevant objects.

Use the tools provided in this exact order:
  detect_objects → score_relevance → filter_by_threshold

Scoring guidelines
──────────────────
- Objects directly involved in the physical interaction  → 0.8–1.0
- Background objects that may be indirectly affected    → 0.4–0.7
- Pure background / skybox / irrelevant clutter         → 0.0–0.39

Bounding box guidelines
───────────────────────
- bbox format: [x1, y1, x2, y2] in pixel coordinates
- x1, y1 = top-left corner of the object
- x2, y2 = bottom-right corner of the object
- Draw the box as TIGHTLY as possible around the visible object
- Use the ACTUAL pixel dimensions of the image — do NOT assume 1000x1000

Your final message MUST be a JSON object — no explanation, no markdown:
{{
  "relevant_objects": [
    {{
      "label":           "ball",
      "relevance_score": 0.95,
      "reason":          "directly involved as the projectile",
      "bbox":            [620, 460, 750, 590]
    }},
    ...
  ]
}}
""".format(threshold=cfg.RELEVANCE_SCORE_THRESHOLD)


def _encode_image(image_path: str) -> tuple[str, str]:
    """
    Read image from disk and return (base64_data, media_type).
    Supports jpg, jpeg, png, webp.
    """
    path = Path(image_path)
    ext  = path.suffix.lower()
    media_type = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
        ".gif":  "image/gif",
    }.get(ext, "image/jpeg")

    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return data, media_type


class SemanticRelevanceAgent(ClaudeAgent):
    name = "semantic_relevance"

    def __init__(self) -> None:
        super().__init__()
        # Internal state populated by tool calls
        self._detected_objects: list[dict] = []
        self._prompt: str = ""

        self.tools = [
            {
                "name": "detect_objects",
                "description": (
                    "Analyse the image already provided in the conversation and "
                    "return a list of every distinct physical object visible, "
                    "with tight bounding boxes in pixel coordinates [x1, y1, x2, y2]."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to the input image (for reference only).",
                        },
                        "candidate_labels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional list of label hints derived from the "
                                "simulation prompt to guide detection."
                            ),
                        },
                    },
                    "required": ["image_path"],
                },
            },
            {
                "name": "score_relevance",
                "description": (
                    "Assign a relevance score [0.0-1.0] and bounding box "
                    "[x1, y1, x2, y2] to each detected object based on the "
                    "physics simulation prompt."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "objects": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {
                                        "type": "string",
                                    },
                                    "relevance_score": {
                                        "type": "number",
                                        "description": "Score from 0.0 to 1.0.",
                                    },
                                    "reason": {
                                        "type": "string",
                                        "description": "Why this score was assigned.",
                                    },
                                    "bbox": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "description": (
                                            "Tight bounding box [x1, y1, x2, y2] "
                                            "in actual image pixel coordinates."
                                        ),
                                    },
                                },
                                "required": ["label", "relevance_score", "bbox"],
                            },
                        },
                        "simulation_prompt": {"type": "string"},
                    },
                    "required": ["objects", "simulation_prompt"],
                },
            },
            {
                "name": "filter_by_threshold",
                "description": (
                    "Remove objects whose relevance_score is below the "
                    "configured threshold and return the filtered list."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scored_objects": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label":           {"type": "string"},
                                    "relevance_score": {"type": "number"},
                                    "reason":          {"type": "string"},
                                    "bbox": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "description": "Bounding box [x1, y1, x2, y2].",
                                    },
                                },
                                "required": ["label", "relevance_score", "bbox"],
                            },
                        },
                    },
                    "required": ["scored_objects"],
                },
            },
        ]

        self.tool_handlers = {
            "detect_objects":      self._detect_objects,
            "score_relevance":     self._score_relevance,
            "filter_by_threshold": self._filter_by_threshold,
        }

    # ── public ────────────────────────────────────────────────────────────────

    def run_for_scene(
        self,
        image_path: str,
        prompt: str,
        run_directory: Path,
    ) -> dict:
        """
        Full entry point. Sends the actual image as base64 vision input,
        runs semantic relevance scoring, then calls SAM2Agent.

        Returns:
            {
                "relevant_objects": [...],  # scored + filtered with bboxes
                "masks":            [...],  # SAM2 masks for each object
            }
        """
        self._prompt = prompt

        # ── encode image as base64 for vision input ───────────────────────────
        _log.info("Encoding image for vision: %s", image_path)
        image_data, media_type = _encode_image(image_path)

        # ── get actual image dimensions ───────────────────────────────────────
        with PILImage.open(image_path) as _img:
            img_w, img_h = _img.size
        _log.info("Image dimensions: %dx%d px", img_w, img_h)

        # ── build multimodal user message ─────────────────────────────────────
        user_msg = [
            {
                "type": "image",
                "source": {
                    "type":       "base64",
                    "media_type": media_type,
                    "data":       image_data,
                },
            },
            {
                "type": "text",
                "text": (
                    f"Image path: {image_path}\n"
                    f"Image dimensions: {img_w}x{img_h} pixels (width x height)\n\n"
                    f"Simulation prompt: {prompt}\n\n"
                    f"The image is {img_w} pixels wide and {img_h} pixels tall. "
                    "Look at the image carefully and identify every physical object. "
                    "For each object, provide a TIGHT bounding box using the ACTUAL "
                    f"pixel coordinates within the {img_w}x{img_h} image. "
                    "x1 is distance from LEFT edge, y1 is distance from TOP edge. "
                    "Detect all objects with accurate bboxes, score relevance, "
                    "filter by threshold, and return the final relevant_objects JSON."
                ),
            },
        ]

        raw_response = self.run(system=SYSTEM_PROMPT, user_message=user_msg)

        # Ensure we have a plain string to parse
        if not isinstance(raw_response, str):
            raw_response = str(raw_response)

        relevant = self._parse_response(raw_response)

        # Fallback: use internally stored detected objects if parse fails
        if not relevant and self._detected_objects:
            _log.warning(
                "Could not parse relevant_objects from final response — "
                "falling back to tool-call results."
            )
            relevant = [
                o for o in self._detected_objects
                if o.get("relevance_score", 0) >= cfg.RELEVANCE_SCORE_THRESHOLD
            ]

        # Validate bboxes
        for obj in relevant:
            if not obj.get("bbox") or len(obj["bbox"]) != 4:
                _log.warning(
                    "Object '%s' has no valid bbox — SAM2 will use centre-crop fallback.",
                    obj.get("label", "?"),
                )

        save_state(run_directory, "relevant_objects", {"relevant_objects": relevant})
        _log.info(
            "Relevant objects (%d): %s",
            len(relevant),
            [
                f"{o['label']}(score={o.get('relevance_score', 0):.2f}, "
                f"bbox={o.get('bbox', 'missing')})"
                for o in relevant
            ],
        )

        # ── Stage 1b: SAM2 segmentation ───────────────────────────────────────
        masks: list[dict] = []
        if relevant:
            _log.info("Handing off to SAM2Agent (%d objects)...", len(relevant))
            sam2_agent = SAM2Agent()
            masks = sam2_agent.run_for_objects(
                image_path=image_path,
                relevant_objects=relevant,
                run_directory=run_directory,
            )
        else:
            _log.warning("No relevant objects — skipping SAM2.")

        return {
            "relevant_objects": relevant,
            "masks":            masks,
        }

    # ── tool handlers ─────────────────────────────────────────────────────────

    def _detect_objects(
        self,
        image_path: str,
        candidate_labels: list[str] | None = None,
    ) -> dict:
        """
        Object detection pass — the image is already in the conversation context
        as a vision input, so we just instruct the model to use what it sees.
        """
        _log.info("detect_objects: %s", image_path)

        # Get image dimensions to pass to the model
        try:
            with PILImage.open(image_path) as _img:
                img_w, img_h = _img.size
            dims = f"{img_w}x{img_h}"
        except Exception:
            img_w, img_h = None, None
            dims = "unknown"

        return {
            "status":    "ok",
            "image_path": image_path,
            "image_dimensions": dims,
            "note": (
                f"The image ({dims} pixels, width x height) has already been "
                "provided in this conversation. Look at it carefully and enumerate "
                "every distinct physical object. For each object provide a TIGHT "
                "bounding box [x1, y1, x2, y2] using ACTUAL pixel coordinates — "
                "x1=left edge, y1=top edge, x2=right edge, y2=bottom edge. "
                "Do NOT guess or use a 1000x1000 default — use the real image size. "
                "Pass all objects with their bboxes to score_relevance."
            ),
            "candidate_labels": candidate_labels or [],
        }

    def _score_relevance(
        self,
        objects: list[dict],
        simulation_prompt: str,
    ) -> dict:
        """Store scored objects (with bboxes) and return them for filtering."""
        _log.info("score_relevance: %d objects", len(objects))
        for obj in objects:
            _log.debug(
                "  %s — score=%.2f  bbox=%s",
                obj.get("label", "?"),
                obj.get("relevance_score", 0),
                obj.get("bbox"),
            )
        self._detected_objects = objects
        return {
            "status":        "ok",
            "scored_objects": objects,
            "threshold":     cfg.RELEVANCE_SCORE_THRESHOLD,
        }

    def _filter_by_threshold(self, scored_objects: list[dict]) -> dict:
        """Filter objects below the relevance threshold, preserving bboxes."""
        kept = [
            o for o in scored_objects
            if o.get("relevance_score", 0) >= cfg.RELEVANCE_SCORE_THRESHOLD
        ]
        dropped = len(scored_objects) - len(kept)
        _log.info(
            "filter_by_threshold: kept %d / %d (dropped %d below %.2f)",
            len(kept), len(scored_objects), dropped, cfg.RELEVANCE_SCORE_THRESHOLD,
        )
        return {"filtered_objects": kept, "dropped_count": dropped}

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(text: str) -> list[dict]:
        """
        Robustly extract the relevant_objects list from the model's final text.

        Handles:
        - clean JSON object:  {"relevant_objects": [...]}
        - alternate keys:     {"filtered_objects": [...], "objects": [...]}
        - bare JSON array:    [{...}, {...}]
        - trailing text after the JSON block
        """
        obj_match = re.search(r'\{.*\}', text, re.DOTALL)
        if obj_match:
            try:
                blob = json.loads(obj_match.group())
                for key in ("relevant_objects", "filtered_objects", "objects"):
                    if key in blob and isinstance(blob[key], list):
                        return blob[key]
            except json.JSONDecodeError:
                pass

        arr_match = re.search(r'\[.*\]', text, re.DOTALL)
        if arr_match:
            try:
                blob = json.loads(arr_match.group())
                if isinstance(blob, list):
                    return blob
            except json.JSONDecodeError:
                pass

        _log.warning(
            "Could not parse JSON from SemanticRelevanceAgent response; "
            "returning empty list"
        )
        return []


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    image  = os.environ.get("TEST_IMAGE",  "scene.jpg")
    prompt = os.environ.get(
        "TEST_PROMPT",
        "A white ball rolls from the left and knocks over a white square bottle on a red surface",
    )
    out_dir = Path("./test_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Image  : {image}")
    print(f"Prompt : {prompt}")
    print(f"Output : {out_dir}\n")

    agent  = SemanticRelevanceAgent()
    result = agent.run_for_scene(
        image_path=image,
        prompt=prompt,
        run_directory=out_dir,
    )

    relevant = result["relevant_objects"]
    masks    = result["masks"]

    print("\n=== Relevant Objects ===")
    for obj in relevant:
        print(
            f"  {obj['label']:20s}"
            f"  score={obj.get('relevance_score', 0):.2f}"
            f"  bbox={obj.get('bbox', 'MISSING')}"
            f"  reason={obj.get('reason', '')}"
        )

    print("\n=== Masks ===")
    for m in masks:
        print(
            f"  {m['label']:20s}"
            f"  confidence={m.get('confidence', 0):.2f}"
            f"  area={m.get('area', 0)}"
            f"  path={m.get('mask_path', 'MISSING')}"
        )