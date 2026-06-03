"""
utils/sam2_wrapper.py

SAM2 Wrapper
━━━━━━━━━━━━
Thin wrapper around Facebook's SAM2 (Segment Anything Model 2).

Exposes a single public function:

    generate_masks(image_path, relevant_objects) -> list[dict]

Each returned mask dict has:
    {
        "label":      str,           # from relevant_objects
        "confidence": float,         # SAM2 IoU prediction score
        "bbox":       [x1,y1,x2,y2], # bounding box used as prompt
        "mask_path":  str,           # absolute path to saved PNG mask
        "area":       int,           # number of True pixels in mask
        "rle":        str | None,    # RLE-encoded mask string (optional)
    }

SAM2 is loaded once (lazy singleton) and reused across calls.
Checkpoint and config paths come from config.py / env vars.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

_log = logging.getLogger("utils.sam2_wrapper")

# ── lazy singleton ────────────────────────────────────────────────────────────
_predictor = None


def _load_predictor():
    """
    Load SAM2 predictor once and cache it.
    Raises a clear error if SAM2 is not installed or checkpoint is missing.
    """
    global _predictor
    if _predictor is not None:
        return _predictor

    # ── import guard ──────────────────────────────────────────────────────────
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError(
            "SAM2 is not installed.\n"
            "  git clone https://github.com/facebookresearch/sam2.git\n"
            "  cd sam2 && pip install -e ."
        )

    checkpoint = Path(cfg.SAM2_CHECKPOINT)
    config     = cfg.SAM2_CONFIG

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {checkpoint}\n"
            f"Download it with:\n"
            f"  cd checkpoints && ./download_ckpts.sh\n"
            f"Or set SAM2_CHECKPOINT env var to the correct path."
        )

    _log.info("Loading SAM2 — checkpoint: %s  config: %s", checkpoint, config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log.info("SAM2 running on: %s", device)

    sam2_model = build_sam2(
        config_file=config,
        ckpt_path=str(checkpoint),
        device=device,
    )
    _predictor = SAM2ImagePredictor(sam2_model)

    _log.info("SAM2 loaded successfully.")
    return _predictor


# ── internal helpers ──────────────────────────────────────────────────────────

def _bbox_from_object(obj: dict, img_w: int, img_h: int) -> list[float] | None:
    """
    Return [x1, y1, x2, y2] from the object dict.
    Falls back to a centred 50 % crop if no bbox is present.
    """
    bbox = obj.get("bbox")
    if bbox and len(bbox) == 4:
        return [float(v) for v in bbox]

    # Fallback: use 50 % centre crop as a weak prompt
    margin_x = img_w * 0.25
    margin_y = img_h * 0.25
    _log.debug(
        "No bbox for '%s' — using centre-crop fallback.", obj.get("label", "?")
    )
    return [margin_x, margin_y, img_w - margin_x, img_h - margin_y]


def _save_mask(
    mask: np.ndarray,
    label: str,
    run_mask_dir: Path,
) -> Path:
    """Save a boolean mask array as a grayscale PNG and return the path."""
    run_mask_dir.mkdir(parents=True, exist_ok=True)
    # Sanitise label for use as filename
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    out_path   = run_mask_dir / f"{safe_label}_mask.png"
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)
    return out_path


def _mask_to_rle(mask: np.ndarray) -> str:
    """
    Encode a boolean 2-D mask as a compact run-length encoded JSON string.
    Format: {"size": [H, W], "counts": [run0, run1, ...]}
    """
    flat   = mask.flatten(order="F").astype(np.uint8)
    counts: list[int] = []
    current_val = 0
    run = 0
    for px in flat:
        if px == current_val:
            run += 1
        else:
            counts.append(run)
            run = 1
            current_val = px
    counts.append(run)
    return json.dumps({"size": list(mask.shape), "counts": counts})


# ── public API ────────────────────────────────────────────────────────────────

def _get_mask_center(mask: np.ndarray) -> tuple[float, float]:
    """Return the (cx, cy) centroid of a boolean mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def _iou(boxA: list, boxB: list) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def _match_auto_masks_to_objects(
    auto_masks: list[dict],
    relevant_objects: list[dict],
    img_w: int,
    img_h: int,
) -> dict[str, dict]:
    """
    Match SAM2 auto-generated masks to labeled objects.

    Strategy:
      1. For objects WITH a bbox — pick the auto mask with highest IoU
         against the predicted bbox.
      2. For objects WITHOUT a bbox — pick the auto mask whose centroid
         is closest to the centre of the image quadrant implied by the label.

    Returns dict mapping label -> best auto mask dict.
    """
    matched: dict[str, dict] = {}
    used_indices: set[int] = set()

    # Sort objects: bbox ones first (more reliable matching)
    with_bbox    = [o for o in relevant_objects if o.get("bbox") and len(o["bbox"]) == 4]
    without_bbox = [o for o in relevant_objects if o not in with_bbox]

    for obj in with_bbox + without_bbox:
        label = obj.get("label", "unknown")
        bbox  = obj.get("bbox")

        best_idx   = -1
        best_score = -1.0

        for idx, am in enumerate(auto_masks):
            if idx in used_indices:
                continue

            if bbox:
                score = _iou(bbox, am["bbox"])
            else:
                # Fall back to centroid distance (normalised)
                cx, cy = _get_mask_center(am["segmentation"])
                dist   = ((cx - img_w / 2) ** 2 + (cy - img_h / 2) ** 2) ** 0.5
                score  = 1.0 / (1.0 + dist)

            if score > best_score:
                best_score = score
                best_idx   = idx

        if best_idx >= 0:
            matched[label] = auto_masks[best_idx]
            used_indices.add(best_idx)
            _log.info(
                "  Matched '%s' → auto mask #%d (score=%.3f, area=%d)",
                label, best_idx, best_score, auto_masks[best_idx]["area"],
            )
        else:
            _log.warning("  No auto mask found for '%s'", label)

    return matched


def generate_masks(
    image_path: str,
    relevant_objects: list[dict],
    mask_output_dir: str | Path | None = None,
) -> list[dict]:
    """
    Run SAM2 automatic mask generation on *image_path*, then match the
    resulting masks to each labeled object in *relevant_objects*.

    Uses SAM2AutomaticMaskGenerator (no bbox prompt needed) for accurate
    segmentation, then matches masks to objects via bbox IoU or centroid
    proximity.

    Parameters
    ----------
    image_path:
        Path to the scene image (jpg / png).
    relevant_objects:
        List of dicts with at minimum ``{"label": str}``.
        Optional ``"bbox": [x1, y1, x2, y2]`` improves matching accuracy.
    mask_output_dir:
        Directory where mask PNGs are saved.
        Defaults to ``<image_dir>/masks/``.

    Returns
    -------
    List of mask dicts — one per object — with keys:
        label, confidence, bbox, mask_path, area, rle
    """
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not relevant_objects:
        _log.warning("generate_masks called with empty relevant_objects list.")
        return []

    # ── output directory ──────────────────────────────────────────────────────
    if mask_output_dir is None:
        mask_output_dir = image_path.parent / "masks"
    mask_output_dir = Path(mask_output_dir)

    # ── load image ────────────────────────────────────────────────────────────
    image_pil = Image.open(image_path).convert("RGB")
    image_np  = np.array(image_pil)
    img_h, img_w = image_np.shape[:2]
    _log.info("Image loaded: %s (%dx%d)", image_path.name, img_w, img_h)

    # ── run SAM2 automatic mask generator ─────────────────────────────────────
    _log.info("Running SAM2 automatic mask generation...")
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        raise ImportError(
            "SAM2 is not installed.\n"
            "  git clone https://github.com/facebookresearch/sam2.git\n"
            "  cd sam2 && pip install -e ."
        )

    checkpoint  = Path(cfg.SAM2_CHECKPOINT)
    config      = cfg.SAM2_CONFIG
    device      = "cuda" if torch.cuda.is_available() else "cpu"

    sam2_model  = build_sam2(config_file=config, ckpt_path=str(checkpoint), device=device)
    mask_gen    = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        min_mask_region_area=500,
    )

    auto_masks = mask_gen.generate(image_np)
    _log.info("SAM2 generated %d candidate masks", len(auto_masks))

    # ── match auto masks to labeled objects ───────────────────────────────────
    matched = _match_auto_masks_to_objects(auto_masks, relevant_objects, img_w, img_h)

    # ── save results ──────────────────────────────────────────────────────────
    results: list[dict] = []
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    for obj in relevant_objects:
        label    = obj.get("label", "unknown")
        obj_bbox = obj.get("bbox")
        am       = matched.get(label)

        if am is None:
            _log.warning("No mask matched for '%s' — returning zero-confidence placeholder", label)
            results.append({
                "label":      label,
                "confidence": 0.0,
                "bbox":       obj_bbox,
                "mask_path":  None,
                "area":       0,
                "rle":        None,
            })
            continue

        best_mask  = am["segmentation"].astype(bool)
        confidence = float(am["predicted_iou"])
        area       = int(am["area"])

        # Compute tight bbox from actual mask pixels
        ys, xs     = np.where(best_mask)
        tight_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        mask_path = _save_mask(best_mask, label, mask_output_dir)
        rle       = _mask_to_rle(best_mask)

        _log.info(
            "  '%s' — iou=%.3f  area=%d px  bbox=%s  saved: %s",
            label, confidence, area, tight_bbox, mask_path.name,
        )

        results.append({
            "label":      label,
            "confidence": confidence,
            "bbox":       tight_bbox,
            "mask_path":  str(mask_path),
            "area":       area,
            "rle":        rle,
        })

    _log.info(
        "generate_masks complete: %d/%d succeeded",
        sum(1 for r in results if r["confidence"] > 0),
        len(results),
    )
    return results