# Physics Simulation Pipeline

An agent-based pipeline that converts an input **image + text prompt** into a
**Blender physics simulation script** using Claude as the reasoning backbone
for each agent stage.

---

## Architecture

```
Input (image + prompt)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  Stage 1 — Scene Understanding & 3D Reconstruction            │
│                                                               │
│  SemanticRelevanceAgent  → filters objects, scores relevance  │
│           │                                                   │
│           ▼                                                   │
│       SAM2Agent          → generates segmentation masks       │
│           │                                                   │
│           ▼                                                   │
│  ReconstructionAgent  ◄──────────────────────────────┐        │
│  (SAM3D)                                             │        │
│           │                                          │        │
│           ▼                                Iterative │        │
│  ValidationAgent                           Refinement│        │
│  (holes/overlaps/misclassifications)                 │        │
│           │ failed                                   │        │
│           ▼                                          │        │
│  RefinementAgent  ───────────────────────────────────┘        │
│           │ passed (or max iterations reached)                │
└───────────┼───────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────┐
│  Stage 2 — Physical Reasoning & Simulation                    │
│                                                               │
│  MaterialClassificationAgent                                  │
│  (rigid / fluid / deformable / granular)                      │
│           │                                                   │
│           ▼                                                   │
│  ForceInferenceAgent                                          │
│  (direction, magnitude, point of application, duration)       │
│           │                                                   │
│           ▼                                                   │
│  BlenderScriptExporter  → simulation.py                       │
└───────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
git clone <this-repo>
cd physics_pipeline

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# For real SAM2 (optional):
#   pip install segment-anything-2
#   Download checkpoint to checkpoints/sam2_hiera_large.pt

# For real SAM3D (optional):
#   Follow the SAM-3D repository setup instructions
#   Set SAM3D_CHECKPOINT env var
```

---

## Configuration

Set environment variables before running:

| Variable            | Required | Description                             |
|---------------------|----------|-----------------------------------------|
| `ANTHROPIC_API_KEY` | ✓        | Your Anthropic API key (`sk-...`)       |
| `SAM2_CHECKPOINT`   |          | Path to SAM2 `.pt` checkpoint           |
| `SAM3D_CHECKPOINT`  |          | Path to SAM3D checkpoint                |
| `LOG_LEVEL`         |          | `DEBUG` / `INFO` / `WARNING` (default INFO) |

Or edit `config.py` directly.

---

## Usage

```bash
export ANTHROPIC_API_KEY=sk-...

python main.py \
    --image  path/to/scene.jpg \
    --prompt "A bowling ball rolls down a wooden ramp and strikes a stack of cans" \
    --frames 300 \
    --fps    24
```

### Run Blender simulation

After the pipeline completes:

```bash
blender --background --python workspace/<run_id>/simulation.py
```

This will:
1. Import all reconstructed meshes
2. Assign physics properties
3. Create force fields
4. Bake the simulation
5. Save `simulation.blend` to the workspace directory

Open `simulation.blend` in Blender to inspect, adjust, and render.

---

## Output files

All outputs land in `workspace/<run_id>/`:

| File | Description |
|------|-------------|
| `relevant_objects.json` | Objects filtered and scored by SemanticRelevanceAgent |
| `masks.json` | SAM2 segmentation masks (base64 RLE encoded) |
| `scene.json` | 3D scene with mesh paths and bounding boxes |
| `validation_report.json` | Geometry validation results |
| `refinement_hints_iter*.json` | Per-iteration SAM3D refinement hints |
| `material_map.json` | Physics material assignments |
| `force_spec.json` | Inferred forces (direction, magnitude, duration) |
| `simulation.py` | **Blender Python script — main output** |
| `*.obj` | Reconstructed mesh files |

---

## Project structure

```
physics_pipeline/
├── main.py                        # CLI entry point
├── config.py                      # All configuration
├── requirements.txt
│
├── utils/
│   ├── __init__.py
│   ├── shared.py                  # ClaudeAgent base, logging, JSON I/O
│   ├── sam2_wrapper.py            # SAM2 local interface
│   └── sam3d_wrapper.py           # SAM3D local interface
│
├── stage1/
│   ├── __init__.py
│   ├── runner.py                  # Stage 1 orchestrator
│   ├── semantic_relevance_agent.py
│   ├── sam2_agent.py
│   ├── reconstruction_agent.py
│   ├── validation_agent.py
│   └── refinement_agent.py
│
├── stage2/
│   ├── __init__.py
│   ├── runner.py                  # Stage 2 orchestrator
│   ├── material_agent.py
│   ├── force_agent.py
│   └── blender_exporter.py
│
└── workspace/                     # Auto-created; one dir per run
    └── <run_id>/
        └── ...
```

---

## Mock mode

When SAM2 / SAM3D are not installed the wrappers fall back to **mock mode**
automatically:

- **SAM2 mock**: generates synthetic striped masks as placeholders
- **SAM3D mock**: writes a unit-cube `.obj` for each object

All downstream agents (Validation, Material, Force, Blender) receive the
mock data and produce valid output — useful for end-to-end testing without
GPU models.

---

## Extending the pipeline

### Add a new agent
1. Subclass `ClaudeAgent` in `utils/shared.py`
2. Define `self.tools` and `self.tool_handlers`
3. Call `self.run(system=..., user_message=...)` and parse the response
4. Wire it into the relevant `runner.py`

### Swap in a different 3D model
Edit `utils/sam3d_wrapper.py` — implement `generate_mesh()` for your model.
The rest of the pipeline is model-agnostic.

### Add more material types
Add entries to `MATERIAL_TO_BLENDER_TYPE` and `_physics_block()` in
`stage2/blender_exporter.py`.