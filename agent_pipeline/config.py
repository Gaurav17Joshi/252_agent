"""
config.py — Central configuration for the physics simulation pipeline.
"""

import os
from pathlib import Path

# ── Anthropic ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = "kimi"
MAX_TOKENS: int = 4096

# ── Pipeline ─────────────────────────────────────────────────────────────────
MAX_REFINEMENT_ITERATIONS: int = 3   # Validation → Refinement → SAM3D cycles
RELEVANCE_SCORE_THRESHOLD: float = 0.4  # Objects below this are filtered out

# ── Workspace ─────────────────────────────────────────────────────────────────
# Each run gets its own timestamped subdirectory
WORKSPACE_ROOT: Path = Path(__file__).parent / "workspace"

# ── SAM2 (local) ─────────────────────────────────────────────────────────────
SAM2_CHECKPOINT: str = os.environ.get(
    "SAM2_CHECKPOINT",
    "sam2/checkpoints/sam2.1_hiera_large.pt",
)
SAM2_CONFIG: str = os.environ.get(
    "SAM2_CONFIG",
    "sam2_hiera_l.yaml",
)

# ── SAM3D (local) ─────────────────────────────────────────────────────────────
SAM3D_CHECKPOINT: str = os.environ.get(
    "SAM3D_CHECKPOINT",
    "checkpoints/sam3d.pt",
)
# Mesh quality: "coarse" | "fine" | "adaptive"
SAM3D_MESH_QUALITY: str = "adaptive"

# ── Blender output ────────────────────────────────────────────────────────────
BLENDER_SCRIPT_NAME: str = "simulation.py"

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")