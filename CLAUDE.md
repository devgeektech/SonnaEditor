# CLAUDE.md — Sonna Editor

This file is auto-loaded by Claude Code at the start of every session in this project. It contains operational rules and pointers to deeper context.

## What this project is

Internal AI photo editing tool for Sonna Studios. Trains a personalised model on Sonna's existing Lightroom-edited photos, then applies that style to new shoots by predicting Lightroom slider values and outputting XMP sidecars. Local-only and cross-platform: macOS, Windows, and Linux.

Not for sale. Not competing with Imagen. Internal tooling for cost savings, IP control, and learning value.

## Required reading before execution

For any non-trivial task, read these before proposing or executing changes:

1. **HANDOVER.md** and **SESSION_STATE.md** - current project status, active risks, recent decisions, and next-step context
2. **project_knowledge.md** - source map and behavior notes for the current codebase
3. **SONNA_EDITOR_BUILD_SPEC.md** - task-by-task build spec with workflow guidance per task
4. **CLI_COMMANDS.md**, **FOUNDATION_TRAINING.md**, and **RUN.md** when touching setup, runtime, training, inference, frontend startup, foundation training, or operator commands

If you have not read the relevant files in the current session and the user asks you to implement anything beyond a trivial fix, read them now before executing commands or editing files.

## Documentation update rule

After every non-trivial execution, update the Markdown file that owns the changed truth:

- **SESSION_STATE.md**: always update after meaningful work so the next session starts from reality.
- **HANDOVER.md**: update when status, architecture, model lineage, training recipe, known risks, or environment assumptions change.
- **project_knowledge.md**: update when source files, source-map responsibilities, APIs, scripts, or behavior notes change.
- **CLI_COMMANDS.md**: update when dataset, Personal AI, Lite, inference, diagnostics, or fine-tune commands change.
- **FOUNDATION_TRAINING.md**: update when foundation dataset, training, resume, retrain, promotion, or foundation checkpoint assumptions change.
- **RUN.md** and **README.md**: update when setup, runtime verification, backend/frontend startup, or user-facing quick-start behavior changes.
- **SONNA_EDITOR_BUILD_SPEC.md**: update only when the planned build workflow itself changes. Do not rewrite historical task specs for minor implementation notes.

Do not modify generated diagnostic reports under `scripts/output/` unless the task is specifically to rerun or revise that report.

## Owner context

User is Darshil, founder of Sonna Studios. Hamilton, NZ. Direct, action-oriented working style. Prefers immediately usable outputs and clean prerequisite thinking. No em dashes, no corporate filler, casual warm professional tone.

## Operational rules

### Code

- Python 3.11, managed by `uv` (NOT pip, NOT poetry, NOT conda)
- Type hints on all public functions
- `pathlib.Path` for file paths, never `os.path` strings
- pytest for tests, ruff for linting, mypy for type checking
- Run tests after every change to non-trivial code
- Commit small: 1-3 commits per task, not 20

### ML-specific

- PyTorch with runtime device selection: CUDA first, Apple MPS second, CPU fallback
- fp32 precision ("32-true") by default for cross-platform reliability
- Batch size 16 for training default, 32 for inference
- Train/val/test split BY SHOOT, not by photo (data leakage prevention)
- Augment INPUT image, NEVER augment target slider values
- Model input resolution is config-driven (`config.IMAGE_RESOLUTION`): 384 for v1, 512 for v2, 768 for v3 (matches Imagen)
- Slider list is config-driven (`config.SLIDER_FIELDS`): 147 values, in the order specified

### Files and data

- Original RAW files: NEVER modify, NEVER move, only read
- Lightroom catalog (.lrcat): open READ-ONLY only, NEVER write
- Trained models: NEVER overwrite, always create new version (v1, v2, v3...)
- XMP sidecars: write next to RAWs, use proper namespace declarations and process version 15.4
- Training data Parquet: hash-keyed, deterministic IDs

### Workflow per task

Each task in SONNA_EDITOR_BUILD_SPEC.md specifies its own workflow at the top:
- Which model (Sonnet vs Opus)
- Whether to use `/plan`
- Whether to invoke multi-agent review

Follow the per-task guidance. Don't apply multi-agent to routine tasks (it's friction). Don't skip multi-agent on flagged high-stakes tasks (Phase 1 Task 1.4, Phase 3 Tasks 3.1/3.2, Phase 5 Task 5.2).

## Multi-agent invocation pattern

When a task specifies multi-agent, use this template with EXPLICIT pauses:

```
For this task, work through four roles in sequence, pausing between each:

1. ARCHITECT: Propose the design. Address [task-specific concerns]. Pause for review.
2. ENGINEER: Implement after architect is approved. Stick to the plan.
3. REVIEWER: Critically review the code. Find at least 3 things to flag.
4. QA: Write comprehensive tests covering edge cases.
```

Without the pauses, multi-agent collapses into rationalising the first answer. The pause is what makes it real review.

## Key gotchas

- GPU availability differs by OS and machine; use `sonna_editor.runtime` helpers instead of hardcoding `mps`, `cuda`, or `cpu`
- On Windows/Linux x86_64 this repo pins PyTorch/TorchVision to CUDA 12.8 wheels via `pyproject.toml` / `uv.lock`; `uv sync --extra dev` should preserve GPU support when an NVIDIA driver is present.
- MPS is not bit-reproducible across runs even with seeded RNG (acceptable for our use case, document if it becomes an issue)
- Lightroom .lrcat is SQLite but schema is partially undocumented; if reverse-engineering gets ugly, fall back to "Save Metadata to File" workflow and read XMPs directly
- Embedded RAW preview JPEG is sufficient for model input at 384-768px; we do NOT need full RAW decode
- DNG conversion at ingestion is for FORMAT NORMALISATION (one format to handle vs. 15), not for transport — we run locally
- Smart Previews are an Imagen transport-layer thing for cloud upload; we don't need them since we run locally

## Session opener template

When the user starts a new Claude Code session, expect this prompt:

```
Project context: Sonna Editor build.

Please read HANDOVER.md and SONNA_EDITOR_BUILD_SPEC.md before responding.

I'm working on [Phase X, Task Y] today. Confirm you've read both documents and summarise the relevant context for this task in 3-5 bullets before we start.
```

If they don't paste it, ask them which task they're working on and read both docs yourself.

## Do not

- Do not propose architectural changes without consulting HANDOVER.md first
- Do not switch from PyTorch to other frameworks (decision is locked)
- Do not suggest cloud GPUs unless local CUDA/MPS/CPU workflows prove inadequate
- Do not modify the user's Lightroom catalog or RAW files
- Do not overwrite trained model checkpoints; always version
- Do not skip tests for non-trivial logic
- Do not use Sonnet for tasks flagged as Opus-required (architecture, fine-tuning, catalog reader)
- Do not do "complete implementation" of whole phases; work task by task
- Do not respond to a task without reading the relevant SPEC section first

## Quick reference

| Phase | Deliverable | Realistic time |
|---|---|---|
| 0 | Working dev environment | 1-2 hrs |
| 1 | Data extraction pipeline | 6-10 hrs |
| 2 | Mode B (Lite preset) end-to-end | 4-6 hrs |
| 3 | Model architecture + first trained profile | 8-12 hrs + training time |
| 4 | Inference engine | 3-5 hrs |
| 5 | Continuous learning loop | 4-6 hrs |
| 6 | Profile management | 2-3 hrs |
| 7 | Electron desktop UI | 10-15 hrs |
| 8 | Team distribution | Deferred |

Total: ~40-60 hours focused work. Mode B usable by week 2 at 10 hrs/week pace.

Quality targets for v1 trained model:
- Median exposure error < 0.20 stops
- Median temperature error < 250K
- Median tint error < 5 units
- Median HSL error < 6 units

## When in doubt

Ask the user. Don't guess on architectural decisions. Don't expand scope without confirmation. The user prefers being asked one good question over receiving a wrong-direction implementation.
