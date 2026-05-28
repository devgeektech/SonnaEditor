# Issue Report: Inference & Pipeline Failures

## Summary
Multiple problems were found in the inference pipeline causing incorrect Lightroom slider outputs and delayed UI progress updates. This issue document records each problem, the root cause, the fix applied, tests added, reproduction steps, and recommended next actions.

---

## 1) Incorrect embedding IDs at inference time (High)
- Symptom: Model predictions for metadata-sensitive sliders (WB, Exposure, ToneCurve) were poor even when model weights were correct.
- Root cause: `InferenceEngine._build_batch()` was zeroing all categorical embedding ID tensors (camera body/make/model, lens, profile, wb_preset) instead of mapping the extracted metadata strings into the checkpoint `EmbeddingRegistry`. This caused every photo to use the `unknown` embedding, removing camera/lens/profile signal at inference.
- Files involved:
  - `src/sonna_editor/inference/engine.py` (original bug location)
  - `src/sonna_editor/model/architecture.py` (EmbeddingRegistry API)
  - `src/sonna_editor/data/extract.py` (metadata extraction)
- Fix applied: Map metadata strings to registry IDs using the checkpoint's `EmbeddingRegistry` in `_build_batch()`; fall back to `unknown` index 0 for novel/missing values.
  - Patch: updated `_build_batch()` to compute `camera_make_id`, `camera_model_id`, `lens_id`, `camera_profile_id`, and `wb_preset_id` arrays and convert them to `torch.tensor(...)` instead of zeros.
- Tests added:
  - `tests/test_inference_pipeline_integration.py::test_build_batch_maps_metadata_strings_to_registry_ids`
- Verification: Ran the v1 inference integration tests (integration tests pass).

---

## 2) UI progress callbacks fired too late (Medium)
- Symptom: The Electron UI progress indicator did not update until the XMP sidecar write completed for each photo; on slow files this made the UI appear frozen during inference/extraction.
- Root cause: `process_shoot_with_model()` called `on_photo_complete` only after performing `write_xmp()` (file IO) for each photo. This meant that the per-photo callback and websocket broadcast waited for the disk write to finish.
- Files involved:
  - `src/sonna_editor/inference/pipeline.py` (process_shoot_with_model)
  - `src/sonna_editor/api/callbacks.py` (make_photo_callback)
- Fix applied: Reordered logic in `process_shoot_with_model()` to call `on_photo_complete` immediately after predictions are available (before `write_xmp`). The callback still reports `xmp_path` only when available; callers should tolerate a `None` or non-existent `xmp_path` until writing finishes.
- Tests added / updated:
  - `tests/api/test_callback_bridge.py` (existing tests run and passed). Specific callback/cancel test validated behavior.
- Verification: Ran callback bridge tests (all passed).

---

## 3) Related observations and low-risk design choices
- `Temperature` is predicted in log-Kelvin space, and an epistemic clamp is applied before exponentiation; ensure any downstream debugging uses the correct prediction space.
- `extract_metadata()` provides `as_shot_wb` to avoid reopening RAWs in `write_xmp()`; this remains the preferred path.
- Unknown categorical values intentionally map to index `0` (registry entry `"unknown"`) to avoid OOB embedding lookups.

---

## Reproduction steps (local)
1. Prepare a small folder of RAW files with varied `Make/Model` and WB presets (or use test fixtures in `tests/`).
2. Run the pipeline via the API or `process_shoot_with_model()` with `on_photo_complete` callback attached.
3. Observe that predictions now reflect camera/lens differences and that UI updates occur as each prediction is produced (no blocking until XMP write completes).

---

## Files changed (summary)
- `src/sonna_editor/inference/engine.py` — fixed `_build_batch()` mapping from metadata to registry IDs.
- `src/sonna_editor/inference/pipeline.py` — moved `on_photo_complete` invocation to before `write_xmp()`.
- `tests/test_inference_pipeline_integration.py` — added regression test for mapping behavior.
- `project_knowledge.md` — documented the fix and the callback change.

---

## Suggested next steps
- Run full test suite: `python -m pytest -q` to confirm no regressions across unrelated modules.
- UI integration test: run the Electron UI and process a small folder to validate live progress updates and visual feedback.
- Audit Mode B `checkpoint_builder` and other places that construct registry mappings to ensure consistency in casing/normalization for metadata labels.
- Add a small integration benchmark measuring per-photo latency: extract -> predict -> callback -> xmp write; surface these numbers in logs for future optimizations.

---

## Notes / Risks
- Firing callbacks before `write_xmp()` means that `xmp_path` may not yet exist when the client receives the `photo_complete` message. The UI code already tolerates this (it uses the image + predictions for display). If you prefer to count `photos_processed` only after successful XMP write, adjust `make_photo_callback` or move the increment after writes — trade-off between progress timeliness and semantic counting.


---

If you want, I can open GitHub issues based on this `issue.md` content, or create separate issue files per bug. Which would you prefer?