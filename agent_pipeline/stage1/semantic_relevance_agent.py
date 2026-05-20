"""
stage1/semantic_relevance_agent.py

Semantic Relevance Agent
━━━━━━━━━━━━━━━━━━━━━━━
Analyses the input image (via description) and the text prompt to decide
which objects in the scene are relevant to the requested physics simulation.

Each object receives a relevance score in [0, 1].  Objects below
``cfg.RELEVANCE_SCORE_THRESHOLD`` are filtered out before masking.

Tools exposed to Claude
───────────────────────
  detect_objects        — run a vision detection pass on the image
  score_relevance       — assign relevance scores given the prompt
  filter_by_threshold   — drop objects below the configured threshold
"""

from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from utils.shared import ClaudeAgent, get_logger, save_state

_log = get_logger("stage1.semantic_relevance")

SYSTEM_PROMPT = """
You are the Semantic Relevance Agent in a physics-simulation pipeline.

Your job:
1. Examine a description of the scene (objects detected in the image).
2. Read the user's physics simulation prompt.
3. Score every detected object from 0.0 (irrelevant) to 1.0 (critical) based
   on how important each is to the requested simulation.
4. Filter out objects with a score below {threshold}.
5. Return a clean ranked list of relevant objects.

Use the tools provided in this exact order:
  detect_objects → score_relevance → filter_by_threshold

Scoring guidelines
──────────────────
- Objects that will be directly involved in the physical interaction → 0.8–1.0
- Background objects that may be indirectly affected              → 0.4–0.7
- Pure background / skybox / irrelevant clutter                  → 0.0–0.39

Return your final answer as a JSON object with key "relevant_objects",
containing a list of {{label, relevance_score, reason}} dicts.
""".format(threshold=cfg.RELEVANCE_SCORE_THRESHOLD)


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
                    "Analyse the image and return a list of distinct objects "
                    "present in the scene. Uses vision understanding of the "
                    "provided image path."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Absolute path to the input image.",
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
                    "Assign a relevance score [0.0–1.0] to each detected object "
                    "based on the physics simulation prompt."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "objects": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "bbox": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "description": "[x1, y1, x2, y2]",
                                    },
                                },
                                "required": ["label"],
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
                                    "label": {"type": "string"},
                                    "relevance_score": {"type": "number"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["label", "relevance_score"],
                            },
                        },
                    },
                    "required": ["scored_objects"],
                },
            },
        ]

        self.tool_handlers = {
            "detect_objects": self._detect_objects,
            "score_relevance": self._score_relevance,
            "filter_by_threshold": self._filter_by_threshold,
        }

    # ── public ────────────────────────────────────────────────────────────────

    def run_for_scene(
        self,
        image_path: str,
        prompt: str,
        run_directory: Path,
    ) -> list[dict]:
        """
        Full entry point.  Returns the filtered relevant-objects list and
        writes ``relevant_objects.json`` to *run_directory*.
        """
        self._prompt = prompt
        user_msg = (
            f"Image path: {image_path}\n\n"
            f"Simulation prompt: {prompt}\n\n"
            "Please detect objects, score their relevance, filter by threshold, "
            "and return the final relevant_objects JSON."
        )
        raw_response = self.run(system=SYSTEM_PROMPT, user_message=user_msg)

        relevant = self._parse_response(raw_response)
        save_state(run_directory, "relevant_objects", {"relevant_objects": relevant})
        _log.info(
            "Relevant objects (%d): %s",
            len(relevant),
            [o["label"] for o in relevant],
        )
        return relevant

    # ── tool handlers ─────────────────────────────────────────────────────────

    def _detect_objects(
        self,
        image_path: str,
        candidate_labels: list[str] | None = None,
    ) -> dict:
        """
        Object detection pass.

        In production this calls a local vision model (e.g. GroundingDINO or
        OWLv2).  Here Claude itself acts as the detector using its vision
        understanding — we return a structured representation so Claude can
        score it in the next tool call.
        """
        _log.info("detect_objects: %s", image_path)
        # In real deployment: call GroundingDINO / OWLv2 / YOLO here
        # For now we return a sentinel that tells Claude to use its own
        # understanding of the image it has been shown.
        return {
            "status": "ok",
            "image_path": image_path,
            "note": (
                "Use your vision understanding of this image to enumerate "
                "the distinct physical objects present. Return them in the "
                "score_relevance call."
            ),
            "candidate_labels": candidate_labels or [],
        }

    def _score_relevance(
        self,
        objects: list[dict],
        simulation_prompt: str,
    ) -> dict:
        """Store scored objects and return them for threshold filtering."""
        _log.info("score_relevance: %d objects", len(objects))
        self._detected_objects = objects
        return {
            "status": "ok",
            "scored_objects": objects,
            "threshold": cfg.RELEVANCE_SCORE_THRESHOLD,
        }

    def _filter_by_threshold(self, scored_objects: list[dict]) -> dict:
        """Filter objects below the relevance threshold."""
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
        """Extract the relevant_objects list from Claude's final text."""
        # Try to find a JSON block
        for start_char in ["{", "["]:
            idx = text.find(start_char)
            if idx != -1:
                try:
                    blob = json.loads(text[idx:])
                    if isinstance(blob, list):
                        return blob
                    if isinstance(blob, dict):
                        for key in ("relevant_objects", "filtered_objects", "objects"):
                            if key in blob:
                                return blob[key]
                except json.JSONDecodeError:
                    pass
        _log.warning("Could not parse JSON from SemanticRelevanceAgent response; "
                     "returning empty list")
        return []