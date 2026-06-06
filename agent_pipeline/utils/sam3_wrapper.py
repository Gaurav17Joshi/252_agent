"""
utils/sam3_wrapper.py

SAM3 Wrapper — Text-Prompted Segmentation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uses Meta's SAM3 (Segment Anything with Concepts) to segment ONLY the objects
mentioned in a physics simulation prompt — no LLM filtering needed.

SAM3 accepts open-vocabulary text prompts (noun phrases) and returns masks
for every matching instance in the image.

Public API
──────────
    generate_masks_from_prompt(image_path, prompt, mask_output_dir)
        → {"relevant_objects": [...], "masks": [...]}

Return format is identical to SemanticRelevanceAgent.run_for_scene(), so the
Stage 1 runner can swap between the two without downstream changes.

Install SAM3
────────────
    git clone https://github.com/facebookresearch/sam3.git
    cd sam3 && pip install -e .
    # Download checkpoints per the SAM3 README
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

_log = logging.getLogger("utils.sam3_wrapper")

# ── lazy singleton ────────────────────────────────────────────────────────────

_sam3_model     = None
_sam3_processor = None


def _load_sam3():
    global _sam3_model, _sam3_processor
    if _sam3_model is not None:
        return _sam3_model, _sam3_processor

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError:
        raise ImportError(
            "SAM3 is not installed.\n"
            "  git clone https://github.com/facebookresearch/sam3.git\n"
            "  cd sam3 && pip install -e ."
        )

    checkpoint = os.environ.get("SAM3_CHECKPOINT", "")
    _log.info("Loading SAM3 model%s", f" from {checkpoint}" if checkpoint else " (default weights)")

    _sam3_model = build_sam3_image_model(checkpoint=checkpoint or None)
    _sam3_processor = Sam3Processor(_sam3_model)

    _log.info("SAM3 loaded successfully.")
    return _sam3_model, _sam3_processor


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_noun_phrases(prompt: str) -> list[str]:
    """
    Extract candidate object nouns from a physics simulation prompt.

    Uses simple heuristics: split on physics verbs and prepositions,
    strip common filler words, return unique non-empty noun phrases.

    Examples
    ────────
    "A white ball rolls and knocks over a bottle on a red surface"
      → ["white ball", "bottle", "red surface"]

    "ball hitting bottle"
      → ["ball", "bottle"]
    """
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', prompt, flags=re.IGNORECASE)

    # Split on physics verbs, prepositions, conjunctions
    splitters = r'\b(hits?|rolls?|knocks?|falls?|slides?|thrown|bounces?|collides?|pushes?|pulls?|drops?|lands?|moves?|and|from|onto|into|over|under|toward|towards|against|with|on|off|at|to|by|in|out|up|down|left|right|then|while|as)\b'
    parts = re.split(splitters, text, flags=re.IGNORECASE)

    nouns: list[str] = []
    for part in parts:
        clean = part.strip().lower()
        # Skip single-char fragments and stop-words caught by the split
        if len(clean) <= 2:
            continue
        # Skip if it looks like a verb/prep we split on
        if re.fullmatch(splitters[3:-1], clean, re.IGNORECASE):
            continue
        nouns.append(clean)

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for n in nouns:
        if n not in seen:
            seen.add(n)
            result.append(n)

    _log.info("Extracted noun phrases from prompt: %s", result)
    return result or [prompt.lower().strip()]


def _save_mask(mask: np.ndarray, label: str, mask_dir: Path) -> Path:
    mask_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    out  = mask_dir / f"{safe}_mask.png"
    Image.fromarray((mask.astype(np.uint8)) * 255).save(out)
    return out


def _mask_to_rle(mask: np.ndarray) -> str:
    flat = mask.flatten(order="F").astype(np.uint8)
    counts: list[int] = []
    val, run = 0, 0
    for px in flat:
        if px == val:
            run += 1
        else:
            counts.append(run)
            run = 1
            val = px
    counts.append(run)
    return json.dumps({"size": list(mask.shape), "counts": counts})


# ── public API ────────────────────────────────────────────────────────────────

def generate_masks_from_prompt(
    image_path: str | Path,
    prompt: str,
    mask_output_dir: str | Path | None = None,
) -> dict:
    """
    Segment only the objects mentioned in *prompt* using SAM3.

    Parameters
    ----------
    image_path:       Path to the scene image.
    prompt:           Physics simulation prompt (e.g. "ball hitting bottle").
    mask_output_dir:  Where to save mask PNGs.  Defaults to <image_dir>/masks/.

    Returns
    -------
    {
      "relevant_objects": [
          {
            "label":           str,
            "object_name":     str,
            "description":     str,
            "relevance_score": float,   # SAM3 confidence score
            "bbox":            [x1,y1,x2,y2],
          }, ...
      ],
      "masks": [
          {
            "idx":        int,
            "mask_path":  str,
            "confidence": float,
            "area":       int,
            "bbox":       [x1,y1,x2,y2],
            "rle":        str,
            "label":      str,
          }, ...
      ],
    }

    This return structure is identical to SemanticRelevanceAgent.run_for_scene().
    """
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if mask_output_dir is None:
        mask_output_dir = image_path.parent / "masks"
    mask_dir = Path(mask_output_dir)

    # ── load SAM3 ──────────────────────────────────────────────────────────────
    _, processor = _load_sam3()

    # ── open image ─────────────────────────────────────────────────────────────
    pil_image = Image.open(image_path).convert("RGB")
    state     = processor.set_image(pil_image)

    # ── build text queries from prompt ─────────────────────────────────────────
    noun_phrases = _extract_noun_phrases(prompt)

    relevant_objects: list[dict] = []
    masks:            list[dict] = []
    global_idx = 0

    for noun in noun_phrases:
        _log.info("SAM3: querying for %r", noun)
        try:
            output = processor.set_text_prompt(state=state, prompt=noun)
        except Exception as exc:
            _log.warning("SAM3 query failed for %r: %s — skipping", noun, exc)
            continue

        raw_masks = output.get("masks", [])
        boxes     = output.get("boxes", [])
        scores    = output.get("scores", [])

        if not raw_masks:
            _log.info("  SAM3 found no instances of %r", noun)
            continue

        _log.info("  SAM3 found %d instance(s) of %r", len(raw_masks), noun)

        for inst_i, (mask_arr, score) in enumerate(zip(raw_masks, scores)):
            mask_bool = np.asarray(mask_arr).astype(bool)

            # Squeeze batch/channel dims if present
            while mask_bool.ndim > 2:
                mask_bool = mask_bool.squeeze(0)

            area = int(mask_bool.sum())
            if area == 0:
                continue

            # Bounding box from mask pixels
            ys, xs   = np.where(mask_bool)
            tight_box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

            # Label: noun phrase, disambiguate multiple instances
            label = noun if len(raw_masks) == 1 else f"{noun}_{inst_i}"
            label = label.replace(" ", "_")

            confidence = float(score) if score is not None else 0.0

            mask_path = _save_mask(mask_bool, f"{label}_{global_idx:03d}", mask_dir)
            rle       = _mask_to_rle(mask_bool)

            masks.append({
                "idx":        global_idx,
                "mask_path":  str(mask_path),
                "confidence": confidence,
                "area":       area,
                "bbox":       tight_box,
                "rle":        rle,
                "label":      label,
            })

            relevant_objects.append({
                "label":           label,
                "object_name":     label.replace("_", " "),
                "description":     f"{noun} detected in scene",
                "relevance_score": confidence,
                "bbox":            tight_box,
            })

            global_idx += 1

    _log.info(
        "SAM3 segmentation complete: %d object(s) found for prompt %r",
        len(masks), prompt,
    )
    return {"relevant_objects": relevant_objects, "masks": masks}


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

    image  = os.environ.get("TEST_IMAGE",  "scene.jpg")
    prompt = os.environ.get("TEST_PROMPT", "ball hitting bottle")
    outdir = os.environ.get("TEST_OUTDIR", "./test_sam3_run/masks")

    print(f"Image:  {image}")
    print(f"Prompt: {prompt}")
    print(f"Output: {outdir}")

    result = generate_masks_from_prompt(
        image_path=image,
        prompt=prompt,
        mask_output_dir=outdir,
    )

    print(f"\nFound {len(result['masks'])} mask(s):")
    for obj in result["relevant_objects"]:
        print(
            f"  {obj['label']:25s}  score={obj['relevance_score']:.2f}"
            f"  bbox={obj['bbox']}"
        )
