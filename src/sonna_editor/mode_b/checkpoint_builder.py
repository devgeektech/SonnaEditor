"""Mode B preset-to-checkpoint converter.

Step 2 of the Mode B rebuild track (HANDOVER Part 6 item 17).

Consumes a Lightroom preset (.xmp) and a style-survey JSON (from Step 1),
produces a SonnaEditor checkpoint with:

- Backbone (ConvNeXt-Tiny) + metadata encoder warm-loaded from a v1.2.3
  base checkpoint — byte-identical to the source weights.
- Output-head final-layer WEIGHTS inherited from the base ckpt unchanged,
  so each prediction is image-aware (deviates per photo based on its
  features just like the base model would).
- Output-head final-layer BIASES are SHIFTED by a calibration delta:
  ``b_new = b_base + delta_preset + delta_survey`` in the model's
  *prediction space* (log-Kelvin for Temperature, raw units elsewhere).
  ``delta_preset = preset_value − LR_DEFAULTS[field]`` (in pred space),
  ``delta_survey`` is the survey shift converted to pred space the same
  way. For a preset matching LR defaults + neutral survey, both deltas
  are zero and the model is byte-equivalent to the base ckpt — predictions
  match v1.2.x exactly. For stylised presets and/or non-neutral survey,
  the delta shifts EVERY photo's prediction by the same amount in pred
  space, anchoring the model at the user's calibration without erasing
  per-photo behaviour.

Earlier revisions of this module had two bugs that interacted:
1. (commit 526f5b7 fixed) Head weights were zeroed, collapsing every
   prediction to the bias-only output.
2. (this commit fixes) Even after weights were inherited, the bias was
   REPLACED with ``preset + survey_offset`` in pred space rather than
   added as a delta — for Temperature that put the bias at log(5500) ≈
   8.6 atop a base bias of ~0.02, shifting every photo's predicted
   Kelvin by exp(8.59) ≈ 5400×, all clipped to the postprocess upper
   bound, producing the same flat output users were seeing.

The resulting checkpoint loads via SonnaEditor.from_checkpoint() identically
to a Mode A checkpoint and is ready for Phase 5 fine-tuning. A sidecar
JSON marks ``profile_type: "mode_b_initial"`` for Step 3 (inference path)
to identify.

Targets v1 slider set (135 outputs) since the production base checkpoint
is v1.2.3. v2 (147) is not addressed here; once a v2 backbone exists, the
converter generalises by extending HEAD_SLICES with the five extension
heads.

Public surface:
- HEAD_SLICES: head attr name -> (start, end) in the 135-vector.
- PROFILE_TYPE: sidecar JSON marker constant.
- INHERITED_SKIP_FIELDS: default_skip_fields inherited from v1.2.3.
- compute_bias_vector(): preset + survey -> {slider_field: pred_space_DELTA}.
- apply_biases_to_model(): ADD deltas to inherited biases (weights untouched).
- build_mode_b_checkpoint(): full orchestration (file in -> ckpt + sidecar out).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Final

import torch

from sonna_editor import config
from sonna_editor.data.xmp import LR_DEFAULTS, read_xmp
from sonna_editor.mode_b.survey import QUESTION_SLIDER_MAP, load_survey
from sonna_editor.model.architecture import SonnaEditor


PROFILE_TYPE: Final[str] = "mode_b_initial"
SLIDER_SET_VERSION: Final[str] = "v1"
V1_OUTPUT_COUNT: Final[int] = 135

# default_skip_fields inherited from v1.2.3 production profile
# (v1_learning/model-v1.2.3-prod256.json). Mode B initial checkpoints adopt
# the same skips because the base checkpoint's underlying failure modes are
# still present after Mode B initialisation — only the biases change.
INHERITED_SKIP_FIELDS: Final[list[str]] = [
    "ColorGradeMidtoneHue",
    "SplitToningShadowHue",
    "Tint",
]

# Maps each v1 output head's attribute name to its (start, end) slice in the
# 135-element prediction vector. Order matches architecture.SonnaEditor.forward.
# Single source of truth — used by both apply_biases_to_model and verification.
HEAD_SLICES: Final[list[tuple[str, int, int]]] = [
    ("tone_head",           0,   8),
    ("presence_head",       8,  11),
    ("wb_head",            11,  13),  # [0]=log_temperature, [1]=tint
    ("hsl_head",           13,  37),
    ("parametric_head",    37,  44),
    ("color_grading_head", 44,  58),
    ("calibration_head",   58,  64),
    ("detail_head",        64,  68),
    ("noise_head",         68,  72),
    ("effects_head",       72,  80),
    ("lens_head",          80,  82),
    ("transform_head",     82,  87),
    ("tone_curve_head",    87, 135),
]

# Sanity at import time. If the v1 head sequence ever changes, the slices
# above must be updated in lockstep — fail loudly rather than silently
# misroute biases.
_total_head_outputs = sum(end - start for _, start, end in HEAD_SLICES)
assert _total_head_outputs == V1_OUTPUT_COUNT, (
    f"HEAD_SLICES total {_total_head_outputs} != v1 output count {V1_OUTPUT_COUNT}"
)
for _i, (_, _s, _e) in enumerate(HEAD_SLICES):
    if _i == 0:
        assert _s == 0, "first head must start at 0"
    else:
        prev_end = HEAD_SLICES[_i - 1][2]
        assert _s == prev_end, (
            f"head slice {_i} ({HEAD_SLICES[_i][0]}) starts at {_s}, "
            f"expected {prev_end} for contiguous coverage"
        )


# ---------------------------------------------------------------------------
# Bias computation
# ---------------------------------------------------------------------------

def _extract_survey_offsets(survey: dict) -> dict[str, float]:
    """Pull {slider_field: offset} from a survey JSON payload.

    The survey JSON's ``questions[key]`` entries are self-describing — each
    contains ``slider_field`` so we don't need to import QUESTION_SLIDER_MAP.
    Survey questions with answer=0 still appear here with offset=0.0; the
    bias computation treats them as no-ops naturally.
    """
    if "questions" not in survey:
        raise ValueError("Survey JSON missing 'questions' key")
    offsets: dict[str, float] = {}
    for key, entry in survey["questions"].items():
        if "slider_field" not in entry or "offset" not in entry:
            raise ValueError(
                f"Survey question {key!r} missing slider_field or offset"
            )
        offsets[entry["slider_field"]] = float(entry["offset"])
    return offsets


def _preset_value(preset: dict, field: str) -> float:
    """Pick the preset value for a field, falling back to LR_DEFAULTS.

    read_xmp() returns float for numeric values, str for non-numeric, None
    when absent. Anything that isn't a finite float falls back to the
    Lightroom default.
    """
    raw = preset.get(field)
    if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
        return float(raw)
    return float(LR_DEFAULTS[field])


def compute_bias_vector(
    preset: dict,
    survey: dict,
) -> dict[str, float]:
    """Build the 135-entry bias-DELTA vector in prediction space.

    Returns the shift to ADD to each head's inherited bias — NOT the
    absolute target value. ``apply_biases_to_model`` is the consumer; it
    calls ``final_linear.bias.add_(...)`` so the output becomes
    ``y = W_inherited · x + (b_base + delta)`` per slider.

    For each slider in the v1 slider set:
    1. Pick the preset value (or LR_DEFAULTS if absent).
    2. Add the survey offset if this slider is survey-addressable.
       Survey offsets live in human units (Kelvin for Temperature, raw
       units elsewhere — see survey.OFFSET_MAGNITUDES).
    3. Clamp the resulting target to the slider's valid Lightroom range.
    4. Compute the delta against LR_DEFAULTS, converting to prediction
       space (only Temperature differs — ``log(target) − log(default)``).

    For a preset that matches LR defaults with a neutral survey every
    delta is exactly 0 and the resulting model is byte-equivalent to the
    base ckpt. Earlier revisions returned absolute targets here and the
    consumer overwrote ``final_linear.bias`` with them; that ignored the
    base ckpt's trained bias and shifted every photo's prediction by a
    huge constant in log space (see module docstring).
    """
    if not isinstance(preset, dict):
        raise TypeError(f"preset must be a dict, got {type(preset).__name__}")

    survey_offsets = _extract_survey_offsets(survey)

    v1_fields = config.SLIDER_FIELDS[:V1_OUTPUT_COUNT]
    deltas: dict[str, float] = {}
    for field in v1_fields:
        target = _preset_value(preset, field)
        if field in survey_offsets:
            target = target + survey_offsets[field]

        lo, hi = config.SLIDER_RANGES[field]
        target_clamped = max(lo, min(hi, target))
        default = float(LR_DEFAULTS[field])
        # LR_DEFAULTS are always inside SLIDER_RANGES, but clamp defensively
        # in case a future field's default lands outside its declared range.
        default_clamped = max(lo, min(hi, default))

        if field == "Temperature":
            # Both endpoints in log-Kelvin. Range-clamping above guarantees
            # target_clamped >= 2000 so log is safe.
            deltas[field] = math.log(target_clamped) - math.log(default_clamped)
        else:
            deltas[field] = target_clamped - default_clamped

    assert len(deltas) == V1_OUTPUT_COUNT, (
        f"Delta vector length {len(deltas)} != {V1_OUTPUT_COUNT}"
    )
    return deltas


# ---------------------------------------------------------------------------
# Applying biases to a SonnaEditor model
# ---------------------------------------------------------------------------

def apply_biases_to_model(
    model: SonnaEditor,
    bias_vector: dict[str, float],
) -> None:
    """Add the per-slider calibration delta to each v1 head's final bias.

    ``bias_vector`` is a delta vector (the return of compute_bias_vector
    after the 2026-05 fix). Mutates ``model`` in place. After this call,
    the model produces ``y = W_inherited · x + (b_base + delta)`` per
    slider — i.e. the base ckpt's per-photo behaviour shifted by the
    user's calibration in prediction space. Weights are NOT touched.

    For a delta vector that's all zero (neutral preset + neutral survey),
    the model is byte-equivalent to the base ckpt and predictions match
    v1.2.x exactly. This is the property that makes Lite calibration
    additive rather than destructive.
    """
    if model._slider_set_version != SLIDER_SET_VERSION:
        raise ValueError(
            f"apply_biases_to_model requires a v1 model "
            f"(slider_set_version={SLIDER_SET_VERSION!r}); got "
            f"{model._slider_set_version!r}"
        )

    v1_fields = config.SLIDER_FIELDS[:V1_OUTPUT_COUNT]
    missing = [f for f in v1_fields if f not in bias_vector]
    if missing:
        raise ValueError(
            f"bias_vector missing {len(missing)} v1 sliders: {missing[:5]}..."
        )

    for head_name, start, end in HEAD_SLICES:
        head = getattr(model, head_name)
        final_linear = head[-1]
        if not isinstance(final_linear, torch.nn.Linear):
            raise TypeError(
                f"Expected {head_name}[-1] to be nn.Linear, got "
                f"{type(final_linear).__name__}"
            )
        head_dim = end - start
        if final_linear.out_features != head_dim:
            raise ValueError(
                f"{head_name} final-linear out_features={final_linear.out_features} "
                f"!= slice width {head_dim}"
            )

        deltas = [bias_vector[v1_fields[i]] for i in range(start, end)]
        with torch.no_grad():
            final_linear.bias.add_(
                torch.tensor(deltas, dtype=final_linear.bias.dtype)
            )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _neutral_metadata_for(model: SonnaEditor) -> dict[str, torch.Tensor]:
    """Build a single-row neutral metadata dict for a forward-pass smoke test.

    Values are chosen so MetadataEncoder.forward() doesn't take any NaN-rescue
    path. Embedding IDs use 0, which always exists for v1.1.0 models because
    embedding tables start at min capacity (≥4).
    """
    common = {
        "iso":               torch.tensor([100.0]),
        "shutter_speed":     torch.tensor([1 / 125.0]),
        "aperture":          torch.tensor([5.6]),
        "focal_length":      torch.tensor([50.0]),
        "lens_id":           torch.tensor([0], dtype=torch.long),
        "camera_profile_id": torch.tensor([0], dtype=torch.long),
        "wb_preset_id":      torch.tensor([0], dtype=torch.long),
        "histogram":         torch.zeros(1, 96),
    }
    if model._arch_version == 0:
        common["camera_body_id"] = torch.tensor([0], dtype=torch.long)
    else:
        common["camera_make_id"]  = torch.tensor([0], dtype=torch.long)
        common["camera_model_id"] = torch.tensor([0], dtype=torch.long)
        common["as_shot_temperature"] = torch.tensor([5500.0])
        common["as_shot_tint"]        = torch.tensor([0.0])
    return common


def verify_checkpoint(
    ckpt_path: Path,
    bias_vector: dict[str, float],
    base_ckpt_path: Path,
    *,
    abs_tol: float = 1e-4,
) -> None:
    """Cross-check a Mode B ckpt against its base ckpt + the calibration deltas.

    After the 2026-05 delta-bias fix the new ckpt's final-linear bias
    must equal ``base_bias + bias_vector[field]`` per slider, with the
    final-linear *weights* byte-identical to the base. This function
    loads both ckpts and asserts both invariants directly — no forward
    pass, so the verification is deterministic and independent of
    SonnaEditor's penultimate activations.

    Raises RuntimeError with a descriptive message on first mismatch.
    """
    new_model = SonnaEditor.from_checkpoint(
        ckpt_path, target_slider_set_version=SLIDER_SET_VERSION
    )
    base_model = SonnaEditor.from_checkpoint(
        base_ckpt_path, target_slider_set_version=SLIDER_SET_VERSION
    )
    new_model.eval()
    base_model.eval()

    v1_fields = config.SLIDER_FIELDS[:V1_OUTPUT_COUNT]
    for head_name, start, end in HEAD_SLICES:
        new_lin = getattr(new_model, head_name)[-1]
        base_lin = getattr(base_model, head_name)[-1]

        if not torch.equal(new_lin.weight, base_lin.weight):
            max_diff = (new_lin.weight - base_lin.weight).abs().max().item()
            raise RuntimeError(
                f"Verification failed for {head_name}: final-linear weights "
                f"diverged from base ckpt (max |diff| = {max_diff:.6f}). "
                f"Mode B must inherit head weights byte-for-byte."
            )

        for i in range(start, end):
            field = v1_fields[i]
            base_b = float(base_lin.bias[i - start].item())
            delta = bias_vector[field]
            expected = base_b + delta
            produced = float(new_lin.bias[i - start].item())
            err = abs(produced - expected)
            if err > abs_tol:
                raise RuntimeError(
                    f"Verification failed for {field} ({head_name} bias[{i - start}]): "
                    f"produced={produced:.6f}, expected={expected:.6f} "
                    f"(base={base_b:.6f} + delta={delta:.6f}), "
                    f"|err|={err:.6f} > tol={abs_tol:.6f}"
                )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_PROFILE_ID_SAFE_CHARS = re.compile(r"[^a-z0-9]+")


def _generate_profile_id(display_name: str, *, now: _dt.datetime | None = None) -> str:
    """Build a slug + UTC timestamp profile ID.

    Example: 'Mode B - Wedding Lite' -> 'mode-b-wedding-lite-20260514-1102'.
    """
    slug = _PROFILE_ID_SAFE_CHARS.sub("-", display_name.lower()).strip("-")
    if not slug:
        slug = "mode-b-profile"
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    stamp = now.strftime("%Y%m%d-%H%M")
    return f"{slug}-{stamp}"


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_mode_b_checkpoint(
    preset_path: Path,
    survey_path: Path,
    base_ckpt_path: Path,
    output_ckpt_path: Path,
    *,
    profile_name: str,
    profile_id: str | None = None,
    skip_verification: bool = False,
) -> Path:
    """Build a Mode B initial checkpoint + sidecar.

    Steps:
    1. Validate inputs exist.
    2. Parse the preset XMP (read_xmp); if all fields come back None, raise.
    3. Load the survey JSON.
    4. Load the base checkpoint as a v1 SonnaEditor.
    5. Compute bias vector and apply to the model in place.
    6. Save the new checkpoint to output_ckpt_path (does NOT touch base ckpt).
    7. Write the sidecar JSON next to the checkpoint.
    8. Verify the resulting checkpoint loads and forward-matches the targets.

    Returns the path to the sidecar JSON.
    """
    preset_path = Path(preset_path)
    survey_path = Path(survey_path)
    base_ckpt_path = Path(base_ckpt_path)
    output_ckpt_path = Path(output_ckpt_path)

    if not preset_path.exists():
        raise FileNotFoundError(f"Preset not found: {preset_path}")
    if not survey_path.exists():
        raise FileNotFoundError(f"Survey JSON not found: {survey_path}")
    if not base_ckpt_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt_path}")
    if not output_ckpt_path.parent.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {output_ckpt_path.parent}"
        )

    preset = read_xmp(preset_path)
    if all(v is None for v in preset.values()):
        raise ValueError(
            f"Preset {preset_path} parsed but contained no recognisable "
            f"slider values. Confirm the file is a Lightroom .xmp preset "
            f"(not an empty template) and uses the expected crs: namespace."
        )

    survey = load_survey(survey_path)

    # Read the base ckpt's image_resolution before instantiating the model.
    # The Mode B sidecar must record THIS value, not config.IMAGE_RESOLUTION
    # (the global default for new training runs). The base ckpt's resolution
    # is what the warm-loaded backbone was trained on; recording the global
    # default instead would mis-route InferenceEngine's preview extraction
    # (engine.py:128 reads `resolution` from the sidecar first). For v1.2.3,
    # that's 256 — the previous behaviour recorded 512 (the current global
    # default) which made the engine extract previews at the wrong size.
    _base_ckpt_blob = torch.load(
        base_ckpt_path, map_location="cpu", weights_only=False
    )
    _base_arch_cfg = _base_ckpt_blob.get("arch_config") or {}
    base_resolution: int = int(
        _base_arch_cfg.get("image_resolution") or config.IMAGE_RESOLUTION
    )

    model = SonnaEditor.from_checkpoint(
        base_ckpt_path, target_slider_set_version=SLIDER_SET_VERSION
    )

    bias_vector = compute_bias_vector(preset, survey)
    apply_biases_to_model(model, bias_vector)

    model.save_checkpoint(output_ckpt_path)

    pid = profile_id or _generate_profile_id(profile_name)
    sidecar_path = output_ckpt_path.with_suffix(".json")

    # Effective default_skip_fields for the new Lite ckpt. INHERITED_SKIP_FIELDS
    # captures the base ckpt's architecturally-broken sliders; the survey
    # captures the user's explicit calibration choice for a subset of those.
    # When a slider appears in BOTH, the survey wins — the user's calibration
    # would otherwise be silently stripped at XMP-write time and the survey
    # question would be functionally meaningless. Inherited skips not covered
    # by the survey stay in place (still architecturally untrustworthy).
    _survey_covered = set(QUESTION_SLIDER_MAP.values())
    effective_skip_fields = [
        f for f in INHERITED_SKIP_FIELDS if f not in _survey_covered
    ]

    sidecar = {
        "profile_type": PROFILE_TYPE,
        "profile_id": pid,
        "display_name": profile_name,
        "resolution": base_resolution,
        "base_checkpoint": str(base_ckpt_path),
        "base_checkpoint_sha256": _compute_sha256(base_ckpt_path),
        "source_preset": str(preset_path),
        "source_survey": str(survey_path),
        "survey_version": survey.get("version"),
        "default_skip_fields": effective_skip_fields,
        "slider_set_version": SLIDER_SET_VERSION,
        "arch_version": model._arch_version,
        "date_iso": _dt.datetime.now(_dt.timezone.utc).replace(
            microsecond=0
        ).isoformat(),
        "notes": (
            "Initialised via Mode B preset-to-checkpoint converter (Step 2). "
            "Backbone + metadata encoder warm-loaded from base checkpoint; "
            "output-head final-layer weights zeroed and biases set to "
            "preset+survey prescribed values in prediction space."
        ),
        "experimental": False,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    if not skip_verification:
        verify_checkpoint(output_ckpt_path, bias_vector, base_ckpt_path)

    return sidecar_path
