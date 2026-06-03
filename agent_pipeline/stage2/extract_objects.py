"""
build_scene.py

Reads object_i.glb and object_i_pose.pt files from a directory
and builds a scene.json file.

File naming convention:
    object_0.glb
    object_0_pose.pt
    object_1.glb
    object_1_pose.pt
    ...
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import random

import torch
import trimesh


def load_glb(glb_path: Path) -> dict:
    """
    Extract bounding box info from a GLB file.
    Returns bbox_min, bbox_max, center, extents.
    """
    mesh = trimesh.load(str(glb_path))

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()

    return {
        "bbox_min": mesh.bounds[0].tolist(),
        "bbox_max": mesh.bounds[1].tolist(),
        "center":   (( mesh.bounds[0] + mesh.bounds[1]) / 2).tolist(),
        "extents":  mesh.extents.tolist(),
    }


def load_pt(pt_path: Path) -> dict:
    """
    Extract translation, rotation, scale from a pose PT file.
    Expected keys: rotation, translation, scale
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=True)

    return {
        "position": data["translation"].tolist(),
        "rotation": data["rotation"].tolist(),
        "scale":    data["scale"].tolist(),
    }


def build_scene_from_directory(
    directory: str,
    output_path: str = None,
) -> dict:
    """
    Reads all object_i.glb and object_i_pose.pt files from directory.
    Builds and returns a scene dict. Optionally saves to output_path.

    Parameters
    ----------
    directory:
        Path to folder containing object_i.glb and object_i_pose.pt files.
    output_path:
        If provided, saves scene.json to this path.

    Returns
    -------
    scene dict with structure:
    {
      "objects": {
        "object_0": {
          "label":    "object_0",
          "position": [...],
          "rotation": [...],
          "scale":    [...],
          "bbox_min": [...],
          "bbox_max": [...],
          "extents":  [...],
          "center":   [...],
        },
        ...
      }
    }
    """
    base = Path(directory)

    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {base}")

    # Find all object indices from GLB and PT files
    glb_indices = {
        int(m.group(1))
        for f in base.glob("object_*.glb")
        if (m := re.match(r"object_(\d+)\.glb", f.name))
    }
    pt_indices = {
        int(m.group(1))
        for f in base.glob("object_*_pose.pt")
        if (m := re.match(r"object_(\d+)_pose\.pt", f.name))
    }

    all_indices = sorted(glb_indices | pt_indices)

    if not all_indices:
        raise FileNotFoundError(
            f"No object_i.glb or object_i_pose.pt files found in: {base}"
        )

    print(f"Found indices: {all_indices}")
    print(f"  GLB files: {sorted(glb_indices)}")
    print(f"  PT  files: {sorted(pt_indices)}")

    objects = {}

    for idx in all_indices:
        obj_key  = f"object_{idx}"
        glb_path = base / f"object_{idx}.glb"
        pt_path  = base / f"object_{idx}_pose.pt"

        obj = {"label": obj_key}

        # Load PT pose (position, rotation, scale)
        if pt_path.exists():
            try:
                pose = load_pt(pt_path)
                obj.update(pose)
                print(f"  [{obj_key}] PT  loaded: {pt_path.name}")
            except Exception as e:
                print(f"  [{obj_key}] PT  FAILED: {e}")
        else:
            print(f"  [{obj_key}] PT  not found: {pt_path.name}")

        # Load GLB bbox (bbox_min, bbox_max, extents, center)
        if glb_path.exists():
            try:
                bbox = load_glb(glb_path)
                obj.update(bbox)
                # Use extents as scale fallback if PT had no scale
                if "scale" not in obj:
                    obj["scale"] = bbox["extents"]
                print(f"  [{obj_key}] GLB loaded: {glb_path.name}")
            except Exception as e:
                print(f"  [{obj_key}] GLB FAILED: {e}")
        else:
            print(f"  [{obj_key}] GLB not found: {glb_path.name}")

        # Fallback defaults if neither file provided position/scale
        obj.setdefault("position", [0.0, 0.0, 0.0])
        obj.setdefault("rotation", [1.0, 0.0, 0.0, 0.0])
        obj.setdefault("scale",    [1.0, 1.0, 1.0])

        objects[obj_key] = obj

    scene = {"objects": objects}

    # Save to file if output_path provided
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(scene, indent=2), encoding="utf-8")
        print(f"\nscene.json written -> {out}")

    return scene


if __name__ == "__main__":

    DIRECTORY   = "/home/keshav06/sam3d/sam-3d-objects/notebook/objects"
    OUTPUT_PATH = "./run_output/scene.json"

    scene = build_scene_from_directory(
        directory=DIRECTORY,
        output_path=OUTPUT_PATH,
    )

    print("\nScene:")
    print(json.dumps(scene, indent=2))

def _build_test_scene(
    object_position_directory: str,
    run_directory: Path,
) -> tuple[dict, str]:
    physics_scene = build_scene_from_directory(
        directory=object_position_directory,
        output_path=str(run_directory / "scene.json"),
    )

    # Randomly select 2 objects
    all_keys = list(physics_scene["objects"].keys())
    selected = random.sample(all_keys, min(2, len(all_keys)))
    physics_scene["objects"] = {
        f"object_{i}": physics_scene["objects"][key]
        for i, key in enumerate(selected)
    }

    labels = [
        obj.get("label", f"object_{i}")
        for i, obj in enumerate(physics_scene["objects"].values())
    ]

    prompt_templates = [
        f"A {labels[0]} hits a {labels[1]} from the left.",
        f"A {labels[0]} slides into a {labels[1]}.",
        f"A {labels[0]} falls onto a {labels[1]}.",
        f"A {labels[0]} rolls toward a {labels[1]} and collides.",
        f"A {labels[0]} is thrown at a {labels[1]}.",
    ]
    prompt = random.choice(prompt_templates)
    return physics_scene, prompt