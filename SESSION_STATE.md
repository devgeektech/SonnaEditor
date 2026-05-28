# Session State — Sonna Editor

**Saved:** 2026-05-14, night (NZT) — second session of the day
**Current phase/task:** Cross-platform environment/docs pass in progress. Saha-app UI rebuild P0 was previously complete; P1 profile-type backend prep remains next after this hygiene pass.

## Cross-platform update — 2026-05-28

- Project target is now explicit: macOS, Windows, and Linux.
- Python setup uses uv + Python 3.11 on every OS.
- Runtime device selection should use CUDA, then Apple MPS, then CPU fallback.
- Adobe DNG Converter is optional for non-DNG workflows and configurable with `SONNA_DNG_CONVERTER`.
- Electron backend launching should use `uv` directly from the repo root instead of shell-specific `bash -lc` assumptions.

## What was completed this session

Two-part session: full audit + plan for the Saha-app UI rebuild, then P0 (first commit of the implementation track).

### Audit + plan track
- Read entire repo state: CLAUDE.md, MEMORY.md, SESSION_STATE.md (previous save), HANDOVER.md (all 1229 lines), SONNA_EDITOR_BUILD_SPEC.md (headings + Phase 5 section in detail).
- Audited the Saha-app codebase in depth: `saha-app/src/{App.jsx, components/*, hooks/*, api/*, tokens.js}`, `saha-app/electron/{main.js, preload.js}`, and the FastAPI backend under `src/sonna_editor/api/{server.py, models.py, jobs.py, callbacks.py, routes/*}`. Confirmed stack: Electron 33 + React 18 + Vite 6 + JSX inline styles, FastAPI backend, HTTP+WS IPC at 127.0.0.1:8765, no global store (React useState lifted in App.jsx).
- Produced the full 7-section audit + phased build plan: 12-13 commits across 7 phases, ~14-17 hrs estimate. **P6 (Full Personal AI training UI) deferred**; training stays a CLI operation for now.
- Surfaced 7 open questions; user approved with refinements (see memory `project_saha_ui_rebuild.md` for the canonical record).
- Identified the live-progress bug's actual root cause: NOT a missing IPC event but a structural callback gap in `inference/pipeline.py:process_shoot_with_model`. Per-photo `on_photo_complete` only fires during the XMP-write loop (line 368); the two slow phases — parallel preview extraction (272-282) and single-batch GPU `engine.predict` (304) — emit zero callbacks. Documented in memory `project_saha_live_progress_bug.md` for P3.

### P0 implementation
- **`92703dd ui: trim process-page controls and remove inert history tab`** (local, not pushed). 2 files changed, +17 / -176.
- `saha-app/src/components/editor.jsx`: removed `ThresholdSlider`, `Check`, `SKIP_TOGGLE_OPTIONS`, `_isOptionActive`; trimmed `CentreAction` (drops threshold/output sections, keeps Profile section only); trimmed `Editor` state (drops `threshold`/`writeXmpInPlace`/`flagLowConfidence`, keeps `skipFields` for profile-driven defaults); simplified `RightComplete` (removed Flagged tile, kept Failed); `handleProcess` now sends `flag_low_confidence: false` explicitly and omits the three other deprecated fields (Pydantic defaults apply); removed unused `useRef` import.
- `saha-app/src/components/shell.jsx`: removed History entry from `NavRail.items`, removed `kind === 'history'` SVG branch in `RailIcon`, removed obsolete "Phase 7.4+" comment.
- Backend `ProcessRequest` fields (`confidence_threshold`, `write_xmp_in_place`, `flag_low_confidence`, `preserve_wb`) intentionally left in place as deprecated defaults — keeps `tests/api/test_callback_bridge.py` and `tests/api/test_process_route.py` green. API cleanup queued for P7.
- 705/705 tests passing before and after the commit.

### Memory writes
- `~/.claude/projects/-Users-darshil-sonnaeditor/memory/project_saha_ui_rebuild.md` — approved phase order, 7 refinements, process rules.
- `~/.claude/projects/-Users-darshil-sonnaeditor/memory/project_saha_live_progress_bug.md` — P3 root cause + minimal-fix recipe.
- `MEMORY.md` index updated with both entries.

## What was completed in previous sessions

All previous work is already in `HANDOVER.md`. Specifically: Phases 0-7 done, Phase 8 deferred; v1.2.3 production lineage shipping; v2 SLIDER_FIELDS locked at 147; v1/v2 slider_set shape-mismatch refactor done (HANDOVER Part 6 item 18); Mode B Rebuild track COMPLETE 2026-05-14 across Steps 1-3 (HANDOVER Part 6 item 17); predictions-sidecar schema extended with `profile_type` / `profile_id` / `base_checkpoint` / `slider_set_version` (Part 6 item 19 RESOLVED). 705/705 tests passing at session entry.

## Current task status

**Between phases.** P0 is committed locally (`92703dd`) on `main`, **not pushed to origin/main**. Working tree state:
- `origin/main` at `e0fb649` — 1 commit behind local HEAD
- Local HEAD `92703dd` — P0 commit
- Working tree dirty with the same pre-existing carry-overs as session entry (none introduced by this session): `M SESSION_STATE.md` (will be overwritten by this `/save`) + 6 untracked diagnostic items under `scripts/`. P0 touched only `saha-app/src/components/{editor,shell}.jsx`.

The new UI passes the test suite but **has not been smoke-tested in the actual Electron app** this session (audit + commit only). Visual confirmation deferred to whichever session runs `npm run dev` next — likely useful to bundle with P1 before its commit.

## Next task

**P1 — Profile-type backend prep** (~45 min, 🟢, single commit).

Scope:
- Add `profile_type: Optional[str] = None` to `api/models.py:Profile`. Document inline: `None` = Mode A trained (legacy + current v1.2.3); `"mode_b_initial"` = preset-derived; future categories possible.
- In `routes/profiles.py:_build_profile`, read `sidecar.get("profile_type")` and pass into the `Profile(...)` constructor (sidecar is already loaded at line 76 via `_read_sidecar`).
- Add a unit test in `tests/api/test_profiles.py` covering the two cases: sidecar without the field → `profile_type is None`; sidecar carrying `"mode_b_initial"` → flows through.
- No UI change. The UI consumer (badge on the processing-page selector + the profiles list) lands in P5.

Suggested commit message: `api: surface profile_type from ckpt sidecar in Profile model`

Process for P1 (unchanged from P0):
- Show diff + run `uv run pytest tests/` before committing.
- Wait for approval before the commit lands.
- Single-purpose commit. No Co-Authored-By trailer.

## Open decisions / blockers

- **P0 commit `92703dd` not pushed to origin/main.** No conflict with the plan (push when convenient), but worth noting.
- **Visual smoke test of P0 in Electron deferred.** No `npm run dev` performed this session — only test-suite regression. Recommend running through the app once before P1's commit to confirm the trimmed Process page still feels right.
- **Diagnostic scripts still untracked.** Carry-over from earlier sessions (`scripts/audit_all_sliders_v1.2.3.py`, `scripts/compare_saha_xmps.py`, `scripts/compare_three_way.py`, `scripts/output/`, `scripts/tint_deep_dive.py`, `scripts/verify_temperature_clamp.py`). Decision still pending: commit as operational artifacts, gitignore, or delete. User chose to defer at P0 start.
- **WB head skip-connection code is implemented** (HANDOVER item 15, 2026-05-28) and now needs a v2 training run plus all-slider audit. **Phase 5 Mode B re-validation** (item 17 follow-up) remains queued.

## Files changed this session

### Source code (modified — P0 commit `92703dd`)
- `saha-app/src/components/editor.jsx` — removed ThresholdSlider, Check, SKIP_TOGGLE_OPTIONS, _isOptionActive; trimmed CentreAction props + Editor state + handleProcess body + RightComplete; cleaned imports
- `saha-app/src/components/shell.jsx` — removed History tab + history SVG branch + obsolete comment from NavRail

### Memory (new)
- `~/.claude/projects/-Users-darshil-sonnaeditor/memory/project_saha_ui_rebuild.md` — approved phase order + refinements + process rules
- `~/.claude/projects/-Users-darshil-sonnaeditor/memory/project_saha_live_progress_bug.md` — P3 root cause + fix recipe
- `~/.claude/projects/-Users-darshil-sonnaeditor/memory/MEMORY.md` — index extended with two entries

### Session state
- `SESSION_STATE.md` — overwritten by this `/save`
