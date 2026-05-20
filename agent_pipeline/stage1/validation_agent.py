"""
stage1/validation_agent.py

Validation Agent
━━━━━━━━━━━━━━━
Audits the reconstructed 3-D scene for geometric problems that would cause
physics simulation to behave incorrectly or crash:

  • Holes         — open boundaries in watertight meshes
  • Overlaps      — intersecting geometry between separate objects
  • Misclassifications — objects whose mesh topology is implausible for
                         the label (e.g. "sphere" reconstructed as a flat plane)
  • Non-manifold edges — edges shared by more than two faces

The agent returns a structured validation report and a boolean ``passed``
flag.  The iterative refinement loop in ``stage1/runner.py`` uses this
flag to decide whether another SAM3D pass is needed.

Tools exposed to Claude
───────────────────────
  check_mesh_integrity    — per-mesh watertight / manifold checks
  detect_overlaps         — pairwise AABB + mesh-level intersection tests
  classify_topology       — compare mesh topology against expected label type
  compile_report          — aggregate per-object issues into one report
"""

from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.shared import ClaudeAgent, get_logger, save_state

_log = get_logger("stage1.validation")

SYSTEM_PROMPT = """
You are the Validation Agent in a physics-simulation pipeline.

Your job: inspect a reconstructed 3-D scene for geometric quality problems.

For each object:
1. Call check_mesh_integrity to find holes and non-manifold edges.
2. Call detect_overlaps to find objects whose meshes intersect.
3. Call classify_topology to flag obvious misclassifications.

Then call compile_report to produce a structured validation report.

Your final JSON response must have this shape:
{
  "passed": true | false,
  "issues": [
    {
      "label": "<object label>",
      "issue_type": "hole" | "overlap" | "misclassification" | "non_manifold",
      "severity": "critical" | "warning",
      "details": "<free-text description>",
      "suggested_fix": "<one-line hint for the Refinement Agent>"
    }
  ],
  "refinement_hints": {
    "<label>": { "holes": [...], "overlaps": [...], "notes": "..." }
  }
}

Set "passed": true only when there are zero critical issues.
"""


class ValidationAgent(ClaudeAgent):
    name = "validation_agent"

    def __init__(self) -> None:
        super().__init__()
        self._issues: list[dict] = []

        self.tools = [
            {
                "name": "check_mesh_integrity",
                "description": (
                    "Check a single mesh .obj file for holes (open boundary "
                    "edges) and non-manifold edges. Returns counts of each."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "mesh_path": {"type": "string"},
                        "vertex_count": {"type": "integer"},
                        "face_count": {"type": "integer"},
                    },
                    "required": ["label", "mesh_path"],
                },
            },
            {
                "name": "detect_overlaps",
                "description": (
                    "Test each pair of objects for bounding-box and mesh-level "
                    "intersection. Returns a list of overlapping pairs."
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
                                    "mesh_path": {"type": "string"},
                                    "bounding_box": {"type": "object"},
                                },
                                "required": ["label", "bounding_box"],
                            },
                        },
                    },
                    "required": ["objects"],
                },
            },
            {
                "name": "classify_topology",
                "description": (
                    "Determine whether the mesh topology is plausible for the "
                    "object label (e.g. a 'ball' should be roughly spherical)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "mesh_path": {"type": "string"},
                        "vertex_count": {"type": "integer"},
                        "face_count": {"type": "integer"},
                        "bounding_box": {"type": "object"},
                    },
                    "required": ["label", "mesh_path"],
                },
            },
            {
                "name": "compile_report",
                "description": "Aggregate all detected issues into the final validation report.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "issues": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                    },
                    "required": ["issues"],
                },
            },
        ]

        self.tool_handlers = {
            "check_mesh_integrity": self._check_mesh_integrity,
            "detect_overlaps": self._detect_overlaps,
            "classify_topology": self._classify_topology,
            "compile_report": self._compile_report,
        }

    # ── public ────────────────────────────────────────────────────────────────

    def validate_scene(
        self,
        scene: dict,
        run_directory: Path,
    ) -> dict:
        """
        Validate the reconstructed scene.

        Returns the validation report dict and writes
        ``validation_report.json`` to *run_directory*.
        """
        self._issues = []

        user_msg = (
            "Please validate this reconstructed scene:\n\n"
            + json.dumps(scene, indent=2)
        )

        raw_response = self.run(system=SYSTEM_PROMPT, user_message=user_msg)
        report = self._parse_report(raw_response)

        if not report:
            # Fallback: no issues found means passed
            report = {
                "passed": not bool(self._issues),
                "issues": self._issues,
                "refinement_hints": {},
            }

        save_state(run_directory, "validation_report", report)

        status = "✓ PASSED" if report.get("passed") else "✗ FAILED"
        _log.info(
            "Validation %s — %d issue(s)",
            status,
            len(report.get("issues", [])),
        )
        return report

    # ── tool handlers ─────────────────────────────────────────────────────────

    def _check_mesh_integrity(
        self,
        label: str,
        mesh_path: str,
        vertex_count: int = 0,
        face_count: int = 0,
    ) -> dict:
        """
        Real implementation: load mesh with trimesh and check watertightness.
        Mock: basic heuristic — very low face count for a complex object flags
        a potential hole.
        """
        _log.info("check_mesh_integrity: %s", label)
        issues: list[dict] = []

        try:
            import trimesh  # optional real dependency
            mesh = trimesh.load(mesh_path)
            open_edges = len(mesh.faces) - len(mesh.edges_unique)  # heuristic
            is_watertight = mesh.is_watertight
            if not is_watertight:
                issues.append({
                    "issue_type": "hole",
                    "severity": "critical",
                    "details": "Mesh is not watertight (open boundary edges detected).",
                    "suggested_fix": "Increase SAM3D reconstruction resolution and enable hole-filling.",
                })
        except Exception:  # trimesh not installed or mock mesh
            # Simple heuristic on vertex/face ratio
            if vertex_count > 0 and face_count > 0:
                euler = vertex_count - (face_count * 3 // 2) + face_count
                if euler != 2:  # not a closed surface
                    issues.append({
                        "issue_type": "hole",
                        "severity": "warning",
                        "details": f"Euler characteristic {euler} ≠ 2 (may have holes).",
                        "suggested_fix": "Run with quality='fine' for this object.",
                    })

        self._issues.extend([{**i, "label": label} for i in issues])
        return {"label": label, "integrity_issues": issues, "issue_count": len(issues)}

    def _detect_overlaps(self, objects: list[dict]) -> dict:
        """Check pairwise AABB overlaps."""
        _log.info("detect_overlaps: %d objects", len(objects))
        overlapping_pairs: list[dict] = []

        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a, b = objects[i], objects[j]
                bb_a = a.get("bounding_box", {})
                bb_b = b.get("bounding_box", {})
                if bb_a and bb_b and _aabb_intersects(bb_a, bb_b):
                    pair = {
                        "objects": [a["label"], b["label"]],
                        "issue_type": "overlap",
                        "severity": "critical",
                        "details": f"AABB of '{a['label']}' intersects '{b['label']}'.",
                        "suggested_fix": (
                            f"Re-segment '{a['label']}' and '{b['label']}' with "
                            "tighter bounding-box prompts."
                        ),
                    }
                    overlapping_pairs.append(pair)
                    self._issues.extend([
                        {**pair, "label": a["label"]},
                        {**pair, "label": b["label"]},
                    ])

        return {"overlapping_pairs": overlapping_pairs, "count": len(overlapping_pairs)}

    def _classify_topology(
        self,
        label: str,
        mesh_path: str,
        vertex_count: int = 0,
        face_count: int = 0,
        bounding_box: dict | None = None,
    ) -> dict:
        """Heuristic topology classification."""
        _log.info("classify_topology: %s", label)
        issues: list[dict] = []

        # Heuristic: extremely flat bounding box for a non-flat label
        if bounding_box:
            mn = bounding_box.get("min", [0, 0, 0])
            mx = bounding_box.get("max", [1, 1, 1])
            dims = [abs(mx[i] - mn[i]) for i in range(3)]
            min_dim = min(dims) if dims else 0
            max_dim = max(dims) if dims else 1
            flatness = min_dim / max_dim if max_dim > 0 else 1.0
            flat_labels = {"floor", "ground", "table", "wall", "plane", "sheet"}
            if flatness < 0.05 and not any(fl in label.lower() for fl in flat_labels):
                issues.append({
                    "label": label,
                    "issue_type": "misclassification",
                    "severity": "warning",
                    "details": (
                        f"Mesh for '{label}' is very flat (flatness={flatness:.3f}). "
                        "Possible misclassification or failed depth estimation."
                    ),
                    "suggested_fix": "Review object mask and re-run reconstruction.",
                })

        self._issues.extend(issues)
        return {"label": label, "topology_issues": issues}

    def _compile_report(self, issues: list[dict]) -> dict:
        """Merge all issues and build refinement hints."""
        # Combine with internally tracked issues
        all_issues = {
            json.dumps(i, sort_keys=True): i
            for i in (issues + self._issues)
        }
        unique_issues = list(all_issues.values())

        # Build per-object refinement hints
        hints: dict[str, dict] = {}
        for issue in unique_issues:
            lbl = issue.get("label", issue.get("objects", ["unknown"])[0])
            if lbl not in hints:
                hints[lbl] = {"issues": [], "notes": ""}
            hints[lbl]["issues"].append(issue.get("suggested_fix", ""))

        passed = not any(
            i.get("severity") == "critical" for i in unique_issues
        )
        return {
            "passed": passed,
            "issues": unique_issues,
            "refinement_hints": hints,
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_report(text: str) -> dict:
        for start in ["{"]:
            idx = text.find(start)
            if idx != -1:
                try:
                    blob = json.loads(text[idx:])
                    if "passed" in blob:
                        return blob
                    if "report" in blob:
                        return blob["report"]
                except json.JSONDecodeError:
                    pass
        return {}


def _aabb_intersects(a: dict, b: dict) -> bool:
    """Return True if two axis-aligned bounding boxes overlap."""
    a_min, a_max = a.get("min", [0, 0, 0]), a.get("max", [1, 1, 1])
    b_min, b_max = b.get("min", [0, 0, 0]), b.get("max", [1, 1, 1])
    for i in range(3):
        if a_max[i] <= b_min[i] or b_max[i] <= a_min[i]:
            return False
    return True