"""
stage1/refinement_agent.py

Refinement Agent
━━━━━━━━━━━━━━━
Takes a failed Validation Report and produces structured refinement hints
that the Reconstruction Agent (SAM3D) can consume on its next iteration.

The agent does NOT call SAM3D directly — it reasons about the failures and
annotates each object with actionable instructions (tighter crop, higher
resolution, hole-filling flags, etc.).

Tools exposed to Claude
───────────────────────
  analyse_failures      — categorise each validation issue by fix type
  generate_hints        — produce per-object SAM3D refinement hints
"""

from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.shared import ClaudeAgent, get_logger, save_state

_log = get_logger("stage1.refinement")

SYSTEM_PROMPT = """
You are the Refinement Agent in a physics-simulation pipeline.

You receive a validation report listing geometric problems in a 3-D scene.
Your job is to translate each problem into actionable reconstruction hints
that SAM3D can use on its next pass.

Use the tools in order:
  1. analyse_failures  — group issues by type and severity
  2. generate_hints    — produce per-object refinement instructions

Hint types you can specify:
  - quality           : "coarse" | "fine" | "adaptive"
  - enable_hole_fill  : true | false
  - tighter_crop      : [x1, y1, x2, y2] or null
  - separate_objects  : [list of labels that overlapped with this one]
  - notes             : any free-text instruction for the reconstruction pass

Return a JSON object with key "refinement_hints":
{
  "refinement_hints": {
    "<object_label>": {
      "quality": "fine",
      "enable_hole_fill": true,
      "tighter_crop": null,
      "separate_objects": [],
      "notes": "..."
    }
  }
}
"""


class RefinementAgent(ClaudeAgent):
    name = "refinement_agent"

    def __init__(self) -> None:
        super().__init__()

        self.tools = [
            {
                "name": "analyse_failures",
                "description": (
                    "Categorise each issue in the validation report by fix type "
                    "(hole_filling, re_segmentation, quality_upgrade, etc.)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "issues": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of issue dicts from the Validation Agent.",
                        },
                    },
                    "required": ["issues"],
                },
            },
            {
                "name": "generate_hints",
                "description": (
                    "Produce per-object reconstruction hints for the next SAM3D pass."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "categorised_issues": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                        "existing_refinement_hints": {
                            "type": "object",
                            "description": "Existing hints from the validation report (if any).",
                        },
                    },
                    "required": ["categorised_issues"],
                },
            },
        ]

        self.tool_handlers = {
            "analyse_failures": self._analyse_failures,
            "generate_hints": self._generate_hints,
        }

    # ── public ────────────────────────────────────────────────────────────────

    def refine(
        self,
        validation_report: dict,
        run_directory: Path,
        iteration: int,
    ) -> dict[str, dict]:
        """
        Generate refinement hints from *validation_report*.

        Returns a mapping of ``{label: hints_dict}`` and writes
        ``refinement_hints_iter<N>.json``.
        """
        user_msg = (
            f"Iteration {iteration} — validation failed.\n\n"
            "Validation report:\n"
            + json.dumps(validation_report, indent=2)
            + "\n\nAnalyse the failures and generate SAM3D refinement hints."
        )

        raw_response = self.run(system=SYSTEM_PROMPT, user_message=user_msg)
        hints = self._parse_hints(raw_response)

        filename = f"refinement_hints_iter{iteration}"
        save_state(run_directory, filename, {"refinement_hints": hints})
        _log.info(
            "Refinement hints generated for %d object(s) (iter=%d)",
            len(hints), iteration,
        )
        return hints

    # ── tool handlers ─────────────────────────────────────────────────────────

    def _analyse_failures(self, issues: list[dict]) -> dict:
        """Group issues by fix type."""
        _log.info("Analysing %d validation issues", len(issues))
        categories: dict[str, list[dict]] = {
            "hole_filling": [],
            "re_segmentation": [],
            "quality_upgrade": [],
            "misclassification_review": [],
            "overlap_separation": [],
        }

        for issue in issues:
            itype = issue.get("issue_type", "")
            severity = issue.get("severity", "warning")

            if itype == "hole":
                if severity == "critical":
                    categories["hole_filling"].append(issue)
                else:
                    categories["quality_upgrade"].append(issue)
            elif itype == "overlap":
                categories["overlap_separation"].append(issue)
                categories["re_segmentation"].append(issue)
            elif itype == "misclassification":
                categories["misclassification_review"].append(issue)
            elif itype == "non_manifold":
                categories["hole_filling"].append(issue)
                categories["quality_upgrade"].append(issue)

        _log.debug("Issue categories: %s", {k: len(v) for k, v in categories.items()})
        return {"categorised": categories, "total": len(issues)}

    def _generate_hints(
        self,
        categorised_issues: list[dict] | dict,
        existing_refinement_hints: dict | None = None,
    ) -> dict:
        """Build per-object hint dicts."""
        hints: dict[str, dict] = dict(existing_refinement_hints or {})

        # Handle both list and dict inputs gracefully
        if isinstance(categorised_issues, dict):
            cats = categorised_issues.get("categorised", {})
            all_issues: list[dict] = []
            for v in cats.values():
                all_issues.extend(v)
        else:
            all_issues = categorised_issues

        for issue in all_issues:
            label = issue.get("label") or (issue.get("objects") or ["unknown"])[0]
            if label not in hints:
                hints[label] = {
                    "quality": "adaptive",
                    "enable_hole_fill": False,
                    "tighter_crop": None,
                    "separate_objects": [],
                    "notes": "",
                }
            h = hints[label]
            itype = issue.get("issue_type", "")

            if itype in ("hole", "non_manifold"):
                h["enable_hole_fill"] = True
                h["quality"] = "fine"
                h["notes"] += f" Hole/non-manifold: {issue.get('details', '')}."

            elif itype == "overlap":
                partners = issue.get("objects", [])
                for p in partners:
                    if p != label and p not in h["separate_objects"]:
                        h["separate_objects"].append(p)
                h["notes"] += f" Overlaps with {partners}."

            elif itype == "misclassification":
                h["quality"] = "fine"
                h["notes"] += (
                    f" Possible misclassification: {issue.get('details', '')}. "
                    "Re-examine mask boundaries."
                )

        return {"refinement_hints": hints}

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_hints(text: str) -> dict[str, dict]:
        for start in ["{"]:
            idx = text.find(start)
            if idx != -1:
                try:
                    blob = json.loads(text[idx:])
                    if "refinement_hints" in blob:
                        return blob["refinement_hints"]
                    if isinstance(blob, dict):
                        # Check if it looks like a hints map
                        first_val = next(iter(blob.values()), None)
                        if isinstance(first_val, dict):
                            return blob
                except json.JSONDecodeError:
                    pass
        return {}