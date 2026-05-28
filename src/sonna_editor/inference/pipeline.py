"""End-to-end pipeline: folder of RAW files → model inference → XMP sidecars."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
from PIL import Image

from sonna_editor import config
from sonna_editor.data.extract import extract_metadata, extract_preview
from sonna_editor.data.xmp import write_xmp
from sonna_editor.inference.engine import InferenceEngine
from sonna_editor.model.postprocess import predictions_to_dict

_logger = logging.getLogger(__name__)

RAW_EXTENSIONS: frozenset[str] = frozenset(
    {".cr3", ".cr2", ".nef", ".arw", ".dng", ".rw2", ".orf", ".raf"}
)

# Mean per-slider std above which a photo is flagged as low-confidence
_UNCERTAINTY_THRESHOLD = 3.0

# Fields the v1 model predicts but should NOT be written to XMP.
# Perspective*: PerspectiveScale ≈ 99.15 (vs default 100) causes a visible
# black border on every photo because Lightroom zooms out to show the canvas edge.
# LensManualDistortionAmount / VignetteAmount: near-zero noise, no training signal.
# Deferred to v2 once the model has enough data to learn geometric corrections reliably.
#
# Skip semantics — two distinct behaviors depending on field type:
# - Generic skip (Perspective*, LensManualDistortionAmount, VignetteAmount,
#   plus any non-WB extra_skip_fields entry): the attribute is OMITTED from
#   the output XMP. Lightroom uses its own default for that slider.
# - WB skip (Temperature, Tint via extra_skip_fields or preserve_wb): the
#   AsShot value from the source RAW's WB metadata is WRITTEN EXPLICITLY in
#   place of the model prediction. Deterministic — Lightroom reads the
#   explicit AsShot value rather than inferring WB mode from a missing
#   attribute (which has undocumented behavior for partial WB writes:
#   missing Tint with present Temperature may default to Tint=0 rather
#   than AsShot Tint). The substitution logic lives in
#   _apply_wb_skip_substitution and runs after filtered_slider_dict is
#   built, just before the write_xmp call.
#
# Mode B (preset/pipeline.py) uses a different "Temperature=0 sentinel"
# mechanism for "preset doesn't specify WB" — see xmp.py write_xmp for
# that path. The two modes are intentionally asymmetric for now; see
# HANDOVER Part 6 "WB skip semantics unification" for the deferred cleanup.
_V1_SKIP_FIELDS: frozenset[str] = frozenset({
    "PerspectiveVertical",
    "PerspectiveHorizontal",
    "PerspectiveRotate",
    "PerspectiveScale",
    "PerspectiveAspect",
    "LensManualDistortionAmount",
    "VignetteAmount",
})


# Fields that get the WB substitution treatment when skipped (see above).
_WB_SKIP_FIELDS: frozenset[str] = frozenset({"Temperature", "Tint"})


def _apply_wb_skip_substitution(
    slider_dict: dict[str, float],
    effective_skip: frozenset[str],
    as_shot_wb: tuple[float, float] | None,
) -> None:
    """In-place: substitute AsShot values for skipped Temperature/Tint.

    Implements the WB-specific skip semantics documented on
    `_V1_SKIP_FIELDS`. When Temperature or Tint is in `effective_skip`
    AND `as_shot_wb` is available, write the AsShot value into the dict
    so the downstream write_xmp produces a deterministic crs: attribute
    with the camera's measurement rather than omitting the attribute.

    When `as_shot_wb` is None (RAW unreadable or extraction failed), the
    fields stay omitted from `slider_dict` — write_xmp will skip them and
    Lightroom will fall back to its own behavior. This is the rare edge
    case the new semantics can't fix.
    """
    if as_shot_wb is None:
        return
    if "Temperature" in effective_skip:
        slider_dict["Temperature"] = float(as_shot_wb[0])
    if "Tint" in effective_skip:
        slider_dict["Tint"] = float(as_shot_wb[1])

# Always-on XMP postprocess rules. These crs: attributes are written to every
# XMP regardless of model prediction or version (v1.2.3 and v2). They are
# NOT in config.SLIDER_FIELDS by design — they're binary toggles, not
# regression targets.
#
# - LensProfileEnable="1": Sonna always wants lens profile correction applied.
#   When True, Lightroom uses the camera+lens metadata to apply the embedded
#   correction profile (distortion, vignetting, chromatic aberration).
# - AutoLateralCA="1": Sonna always wants automatic lateral chromatic
#   aberration removal. Absent from real LR Classic 15.3 XMPs even with lens
#   profile enabled (verified 2026-05-13 via Canon R6 + RF24-70mm export);
#   the historical LensProfileChromaticAberrationScale slider was a separate
#   manual control that was removed from LR's Lens Profile sub-panel.
ALWAYS_ON_POSTPROCESS: dict[str, str] = {
    "LensProfileEnable": "1",
    "AutoLateralCA": "1",
}

# Epistemic clamp: bounds the model's log-space Temperature prediction to the
# range covered by training data. Refusing to extrapolate beyond observed
# signal — prevents catastrophic exp() amplification when the model drifts
# out-of-distribution. v1.2.3 training data: 9,746 photos, Temperature range
# 2037 K to 9400 K (zero NaN). log(2037) ≈ 7.619, log(9400) ≈ 9.149.
#
# This is an EPISTEMIC clamp (allowed under Decision 8): bounded by what the
# model has training evidence for. NOT a stylistic clamp (banned by Decision
# 8): no assumptions about "reasonable" or "typical" Temperatures within
# Lightroom's valid range. Applied to the log-space prediction before exp()
# converts back to Kelvin, equivalent to clipping the Kelvin output to
# [2037, 9400].
TEMPERATURE_LOG_CLAMP: tuple[float, float] = (
    math.log(2037.0),
    math.log(9400.0),
)


def _apply_temperature_clamp(slider_dict: dict[str, float]) -> None:
    """Apply the Temperature epistemic clamp in-place.

    Mutates `slider_dict["Temperature"]` if present and non-zero. Silent —
    no logging or per-call statistics; verification of firing rate lives
    in scripts/verify_temperature_clamp.py.
    """
    temp_k = slider_dict.get("Temperature")
    if temp_k is None or temp_k <= 0:
        return
    log_pred = math.log(temp_k)
    log_min, log_max = TEMPERATURE_LOG_CLAMP
    if log_pred < log_min:
        slider_dict["Temperature"] = math.exp(log_min)
    elif log_pred > log_max:
        slider_dict["Temperature"] = math.exp(log_max)


_PREDICTIONS_FILENAME = "sonna_predictions.json"


def _extract_one(raw_path: Path, target_size: int) -> tuple[Image.Image, dict]:
    preview = extract_preview(raw_path, target_size=target_size)
    meta = extract_metadata(raw_path)
    return preview, meta


def process_shoot_with_model(
    input_dir: Path,
    model_path: Path,
    output_dir: Optional[Path] = None,
    batch_size: int = 32,
    max_workers: int = 4,
    uncertainty: bool = False,
    n_uncertainty_samples: int = 10,
    dry_run: bool = False,
    device: Optional[str] = None,
    save_predictions: bool = True,
    preserve_wb: bool = False,
    extra_skip_fields: Optional[Iterable[str]] = None,
    on_photo_complete: Optional[Callable[[dict], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """
    Run model inference on all RAW files in input_dir and write XMP sidecars.

    Args:
        input_dir:             Folder to scan for RAW files.
        model_path:            Path to trained checkpoint (.ckpt).
        output_dir:            Override: write XMPs here instead of next to RAWs.
        batch_size:            Images per inference batch.
        max_workers:           Threads for parallel preview extraction.
        uncertainty:           If True, run MC dropout and flag low-confidence shots.
        n_uncertainty_samples: Forward passes per image for MC dropout.
        dry_run:               Extract and predict but don't write XMPs.
        device:                "mps", "cpu", or None (auto).
        save_predictions:      Write sonna_predictions.json alongside XMPs for
                               later use by the continuous learning capture module.
        preserve_wb:           Deprecated compat shim. If True, adds
                               ["Temperature", "Tint"] to extra_skip_fields.
                               Kept for one release; new callers should pass
                               extra_skip_fields=["Temperature","Tint"] directly.
        extra_skip_fields:     User-toggled list of slider fields where the
                               model's prediction is suppressed in the XMP
                               write. Unioned with _V1_SKIP_FIELDS.

                               Behavior is field-type-dependent (Mode A):
                               - Generic fields: attribute is OMITTED from
                                 the XMP; Lightroom uses its own default.
                               - WB fields (Temperature, Tint): the AsShot
                                 value from the source RAW's WB metadata is
                                 WRITTEN in place of the model prediction.
                                 Deterministic — see _V1_SKIP_FIELDS doc and
                                 _apply_wb_skip_substitution.

                               Model still predicts these and predictions still
                               land in sonna_predictions.json; only the XMP
                               write is filtered/substituted. Recorded in the
                               sidecar's v1_skip_fields list so the finetune
                               capture pipeline correctly attributes them as
                               "model_filtered" source.
        on_photo_complete:     Optional callback fired after each XMP is written.
                               Receives a dict with keys: name, raw_path,
                               predicted_values, std (Tensor[135] or None),
                               status ("ok"|"flag"|"fail"), elapsed_seconds,
                               xmp_path. Exceptions in the callback are logged
                               and swallowed — they MUST NOT crash the run.
        cancel_event:          Optional threading.Event. If set, the per-photo
                               loop exits cleanly after the in-flight photo is
                               written. Cancellation cannot interrupt the GPU
                               batch (one batch covers the whole shoot in v1).

    Returns:
        dict with keys: processed, failed, failures, output_paths, low_confidence,
                        predictions_path, cancelled.
    """
    raws = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in RAW_EXTENSIONS
    )
    if not raws:
        return {
            "processed": 0, "failed": 0,
            "failures": [], "output_paths": [], "low_confidence": [],
            "predictions_path": None,
        }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load engine first so we know what resolution to extract previews at ---
    # v1.0.x ckpts run at 384; v1.1.0+ at 512. The engine reads its own resolution
    # from arch_config / state-dict shape, so we don't have to branch here.
    run_timestamp = datetime.now(timezone.utc).isoformat()
    engine = InferenceEngine(model_path, device=device)
    engine.warmup()
    target_size = engine._image_resolution  # source of truth for this run

    # --- Read the ckpt's sidecar JSON for profile metadata ---
    # Same lookup pattern as engine.py:123. Propagates profile_type,
    # profile_id, base_checkpoint into sonna_predictions.json so Phase 5
    # capture can distinguish Mode B deltas (baseline = preset+survey) from
    # Mode A deltas (baseline = trained model output). For legacy ckpts
    # without these fields, the values stay None and Phase 5 treats them
    # as Mode A by default.
    _ckpt_sidecar_path = model_path.with_suffix(".json")
    _ckpt_sidecar: dict = {}
    if _ckpt_sidecar_path.exists():
        try:
            _ckpt_sidecar = json.loads(_ckpt_sidecar_path.read_text())
        except (json.JSONDecodeError, OSError):
            _ckpt_sidecar = {}

    # --- Extract previews + metadata in parallel ---
    previews: list[Image.Image] = []
    metadatas: list[dict] = []
    good_raws: list[Path] = []
    failures: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_path = {pool.submit(_extract_one, p, target_size): p for p in raws}
        for future in as_completed(future_to_path):
            raw_path = future_to_path[future]
            try:
                preview, meta = future.result()
                previews.append(preview)
                metadatas.append(meta)
                good_raws.append(raw_path)
            except Exception as exc:
                failures.append({"path": str(raw_path), "error": str(exc)})

    if not good_raws:
        return {
            "processed": 0, "failed": len(failures),
            "failures": failures, "output_paths": [], "low_confidence": [],
            "predictions_path": None,
        }

    # Sort so output order matches filesystem order
    combined = sorted(zip(good_raws, previews, metadatas), key=lambda t: t[0].name)
    good_raws, previews, metadatas = [list(x) for x in zip(*combined)]

    # --- Inference (engine already loaded above so we knew the target_size) ---

    if uncertainty:
        preds, std_preds = engine.predict_with_uncertainty(
            previews, metadatas,
            n_samples=n_uncertainty_samples,
            batch_size=batch_size,
        )
    else:
        preds = engine.predict(previews, metadatas, batch_size=batch_size)
        std_preds = None

    # --- Write XMPs ---
    output_paths: list[str] = []
    low_confidence: list[dict] = []
    # Full (unfiltered) predictions keyed by filename — for sonna_predictions.json
    full_predictions_by_file: dict[str, dict[str, float]] = {}

    xmp_dir = output_dir if output_dir is not None else input_dir

    # Combine the static v1 skip set with the user-toggled extras (and the
    # legacy preserve_wb shim). Single frozenset, applied identically per photo.
    user_skip: set[str] = set(extra_skip_fields or ())
    if preserve_wb:
        user_skip |= {"Temperature", "Tint"}
    effective_skip: frozenset[str] = _V1_SKIP_FIELDS | frozenset(user_skip)

    cancelled = False
    for i, raw_path in enumerate(good_raws):
        photo_start = time.monotonic()
        full_slider_dict = predictions_to_dict(preds, batch_idx=i)  # all 135 fields
        # Epistemic clamp on Temperature — bounded to training data range.
        # See TEMPERATURE_LOG_CLAMP definition above for rationale.
        _apply_temperature_clamp(full_slider_dict)
        full_predictions_by_file[raw_path.name] = full_slider_dict

        filtered_slider_dict = {
            k: v for k, v in full_slider_dict.items()
            if k not in effective_skip
        }
        # WB skip semantics: substitute AsShot values for skipped
        # Temperature/Tint rather than omitting (see _V1_SKIP_FIELDS doc).
        _apply_wb_skip_substitution(
            filtered_slider_dict, effective_skip, metadatas[i].get("as_shot_wb")
        )

        if output_dir is not None:
            xmp_path = output_dir / raw_path.with_suffix(".xmp").name
        else:
            xmp_path = raw_path.with_suffix(".xmp")

        photo_status = "ok"
        if std_preds is not None:
            mean_std = float(std_preds[i].mean())
            if mean_std > _UNCERTAINTY_THRESHOLD:
                low_confidence.append({
                    "path": str(raw_path),
                    "mean_std": round(mean_std, 3),
                })
                photo_status = "flag"

        # Fire the per-photo callback now that predictions are available so the
        # UI can surface live progress during the (potentially slow) XMP write.
        # Note: xmp_path may not exist yet when this callback runs; callers
        # should handle a missing `xmp_path` value. Exceptions are swallowed
        # to avoid crashing the run.
        if on_photo_complete is not None:
            try:
                on_photo_complete({
                    "name": raw_path.name,
                    "raw_path": str(raw_path),
                    "xmp_path": str(xmp_path) if (not dry_run) else None,
                    "predicted_values": full_slider_dict,
                    "std": std_preds[i] if std_preds is not None else None,
                    "status": photo_status,
                    "elapsed_seconds": time.monotonic() - photo_start,
                })
            except Exception as cb_exc:  # noqa: BLE001 — never crash the run
                _logger.warning("on_photo_complete raised %r; continuing", cb_exc)

        if not dry_run:
            # Pass the cached As-Shot WB (computed during the parallel extract
            # phase) so write_xmp doesn't reopen the RAW just to derive WB
            # for the Pre-Saha snapshot — saves ~200 ms per CR3/ARW.
            write_xmp(
                xmp_path, filtered_slider_dict,
                source_raw_path=raw_path,
                as_shot_wb=metadatas[i].get("as_shot_wb"),
                extra_attributes=ALWAYS_ON_POSTPROCESS,
            )
            output_paths.append(str(xmp_path))

        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break

    # --- Save prediction sidecar ---
    predictions_path: Optional[str] = None
    if save_predictions:
        sidecar = {
            "model_path": str(model_path.resolve()),
            "model_version": model_path.stem,  # e.g. "model-v1.0.1" → stem
            "run_timestamp": run_timestamp,
            # Profile identity propagated from the ckpt's sidecar JSON.
            # None for legacy ckpts (e.g. v1.2.3 production) that predate
            # these fields; Mode B initial ckpts populate all four.
            # Phase 5 capture branches on profile_type to pick the right
            # delta baseline (Mode A trained vs Mode B preset-derived).
            "profile_type":       _ckpt_sidecar.get("profile_type"),
            "profile_id":         _ckpt_sidecar.get("profile_id"),
            "base_checkpoint":    _ckpt_sidecar.get("base_checkpoint"),
            "slider_set_version": engine._model._slider_set_version,
            # v1_skip_fields includes both the static skip set and any
            # user-toggled extras for this run. The finetune capture
            # pipeline reads this to mark these fields as "model_filtered"
            # (not user_final / lr_default) when computing deltas.
            "v1_skip_fields": sorted(effective_skip),
            "static_skip_fields": sorted(_V1_SKIP_FIELDS),
            "user_skip_fields": sorted(user_skip),
            "slider_fields": list(config.SLIDER_FIELDS),
            "photos": full_predictions_by_file,
        }
        pred_path = xmp_dir / _PREDICTIONS_FILENAME
        pred_path.write_text(json.dumps(sidecar, indent=2))
        predictions_path = str(pred_path)

    return {
        "processed": len(output_paths) if not dry_run else len(good_raws),
        "failed": len(failures),
        "failures": failures,
        "output_paths": output_paths,
        "low_confidence": low_confidence,
        "predictions_path": predictions_path,
        "cancelled": cancelled,
    }
