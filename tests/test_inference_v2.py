"""v2 inference-pipeline constants tests (commit 57ab8bf)."""
from __future__ import annotations

from pathlib import Path

from datetime import datetime, timezone

import pytest

from sonna_editor import config
from sonna_editor.inference.pipeline import (
    ALWAYS_ON_POSTPROCESS,
    DENOISE_STRENGTH_LEVELS,
    _V1_SKIP_FIELDS,
    _apply_denoise_if_needed,
    _apply_wb_skip_substitution,
    _stabilise_rgb_tone_curve_endpoints,
    smooth_burst_predictions,
)


def test_always_on_postprocess_constant() -> None:
    assert ALWAYS_ON_POSTPROCESS == {
        "LensProfileEnable": "1",
        "AutoLateralCA": "1",
    }


def test_v1_skip_fields_does_not_intersect_v2_extension() -> None:
    """_V1_SKIP_FIELDS must not accidentally drop any of the 12 v2 fields
    when inference writes XMPs from a v2 model. Verified programmatically
    to prevent silent regressions if either set grows."""
    v2_fields = set(config.SLIDER_FIELDS[135:])
    intersection = set(_V1_SKIP_FIELDS) & v2_fields
    assert intersection == set(), f"unexpected intersection: {intersection}"


# ──────────────────────────────────────────────────────────────────────
# WB skip substitution (Mode A): when Temperature/Tint are in skip_fields,
# write AsShot values explicitly rather than omit. Documented on
# _V1_SKIP_FIELDS; logic in _apply_wb_skip_substitution.
# ──────────────────────────────────────────────────────────────────────

def test_wb_substitution_writes_asshot_for_skipped_temperature() -> None:
    d = {"Temperature": 5500.0, "Tint": 10.0, "Exposure2012": 0.5}
    _apply_wb_skip_substitution(d, frozenset({"Temperature"}), as_shot_wb=(3500.0, 5.0))
    assert d["Temperature"] == 3500.0  # substituted from AsShot
    assert d["Tint"] == 10.0           # untouched (not in skip)
    assert d["Exposure2012"] == 0.5    # non-WB field untouched


def test_wb_substitution_writes_asshot_for_skipped_tint() -> None:
    d = {"Temperature": 5500.0, "Tint": 10.0}
    _apply_wb_skip_substitution(d, frozenset({"Tint"}), as_shot_wb=(3500.0, 5.0))
    assert d["Temperature"] == 5500.0  # untouched
    assert d["Tint"] == 5.0            # substituted


def test_wb_substitution_writes_asshot_for_both() -> None:
    """preserve_wb=True (or extra_skip_fields=[Temperature,Tint]) path."""
    d = {"Temperature": 5500.0, "Tint": 10.0}
    _apply_wb_skip_substitution(d, frozenset({"Temperature", "Tint"}), as_shot_wb=(3500.0, 5.0))
    assert d["Temperature"] == 3500.0
    assert d["Tint"] == 5.0


def test_wb_substitution_no_op_when_neither_skipped() -> None:
    d = {"Temperature": 5500.0, "Tint": 10.0}
    _apply_wb_skip_substitution(d, frozenset({"Saturation"}), as_shot_wb=(3500.0, 5.0))
    assert d["Temperature"] == 5500.0
    assert d["Tint"] == 10.0


def test_wb_substitution_no_op_when_as_shot_wb_missing() -> None:
    """When as_shot_wb extraction failed (None), the substitution silently
    leaves the dict alone — the caller's earlier filtering already removed
    skipped keys, so write_xmp will omit them. Edge case: as_shot_wb None."""
    d = {"Temperature": 5500.0}  # Tint already filtered out (skipped)
    _apply_wb_skip_substitution(d, frozenset({"Tint"}), as_shot_wb=None)
    assert "Tint" not in d
    assert d["Temperature"] == 5500.0


def test_wb_substitution_overwrites_model_prediction_when_already_in_dict() -> None:
    """If skipped Temperature somehow remained in the dict (e.g. caller
    didn't pre-filter), the substitution still replaces it with AsShot.
    Defensive — the actual pipeline.py path pre-filters, so this is just
    invariant-preservation."""
    d = {"Temperature": 5500.0}  # model's prediction
    _apply_wb_skip_substitution(d, frozenset({"Temperature"}), as_shot_wb=(3500.0, 5.0))
    assert d["Temperature"] == 3500.0  # AsShot wins


def test_rgb_tone_curve_endpoint_stabilisation_preserves_neutral_black_and_white() -> None:
    sliders = {
        "ToneCurveRed_Pt1_X": 4.0,
        "ToneCurveRed_Pt1_Y": 3.0,
        "ToneCurveRed_Pt3_Y": 71.0,
        "ToneCurveRed_Pt6_X": 218.0,
        "ToneCurveRed_Pt6_Y": 255.0,
        "ToneCurveGreen_Pt1_X": 1.0,
        "ToneCurveGreen_Pt1_Y": 0.0,
        "ToneCurveGreen_Pt4_Y": 129.0,
        "ToneCurveGreen_Pt6_X": 240.0,
        "ToneCurveGreen_Pt6_Y": 221.0,
        "ToneCurveBlue_Pt1_X": 0.0,
        "ToneCurveBlue_Pt1_Y": 2.0,
        "ToneCurveBlue_Pt5_Y": 195.0,
        "ToneCurveBlue_Pt6_X": 213.0,
        "ToneCurveBlue_Pt6_Y": 234.0,
        "ToneCurve_Pt6_Y": 245.0,
    }

    _stabilise_rgb_tone_curve_endpoints(sliders)

    for prefix in ("ToneCurveRed", "ToneCurveGreen", "ToneCurveBlue"):
        assert sliders[f"{prefix}_Pt1_X"] == 0.0
        assert sliders[f"{prefix}_Pt1_Y"] == 0.0
        assert sliders[f"{prefix}_Pt6_X"] == 255.0
        assert sliders[f"{prefix}_Pt6_Y"] == 255.0
    assert sliders["ToneCurveRed_Pt3_Y"] == 71.0
    assert sliders["ToneCurveGreen_Pt4_Y"] == 129.0
    assert sliders["ToneCurveBlue_Pt5_Y"] == 195.0
    assert sliders["ToneCurve_Pt6_Y"] == 245.0


def test_wb_substitution_falls_back_to_omission_when_as_shot_wb_missing_end_to_end(
    tmp_path: Path,
) -> None:
    """Full chain: filter + substitution(None) + write_xmp → crs:Tint absent.

    Documents the rare edge case where AsShot WB extraction failed (corrupt
    RAW / unsupported format). When the helper can't substitute, the
    previously-filtered Tint stays absent from the dict, and write_xmp omits
    the crs: attribute. This is the only path in Mode A where the LR
    partial-WB ambiguity still exists — fallback to the old "omit and
    hope LR does the right thing" behavior.
    """
    import re

    from sonna_editor.data.xmp import write_xmp

    # Simulate the per-photo state
    full_dict: dict[str, float] = {
        "Exposure2012": 0.5,
        "Tint": 8.0,           # model's collapsed prediction
        "Temperature": 6000.0,
    }
    effective_skip = frozenset({"Tint"})

    # Filter first (mirrors pipeline.py per-photo loop)
    filtered = {k: v for k, v in full_dict.items() if k not in effective_skip}
    assert "Tint" not in filtered  # filter removed it

    # Helper called with as_shot_wb=None (extraction failure)
    _apply_wb_skip_substitution(filtered, effective_skip, as_shot_wb=None)
    assert "Tint" not in filtered  # still absent — substitution couldn't recover

    # Write XMP and confirm crs:Tint is absent (fallback path)
    out = tmp_path / "fallback.xmp"
    write_xmp(out, filtered)
    text = out.read_text()
    assert re.search(r'crs:Tint="[^"]*"', text) is None, (
        "crs:Tint should be ABSENT in output XMP when as_shot_wb is None "
        "(fallback to omission)"
    )
    # Sanity: other model fields still written
    assert 'crs:Exposure2012="+0.5"' in text
    assert 'crs:Temperature="+6000"' in text


def test_smooth_burst_predictions_stabilizes_burst() -> None:
    dt1 = datetime(2026, 6, 1, 14, 0, 0, tzinfo=timezone.utc)
    dt2 = datetime(2026, 6, 1, 14, 0, 1, tzinfo=timezone.utc)
    dt3 = datetime(2026, 6, 1, 14, 0, 2, tzinfo=timezone.utc)
    # 4th shot after a 10s pause
    dt4 = datetime(2026, 6, 1, 14, 0, 12, tzinfo=timezone.utc)

    preds = [
        {"Exposure2012": 0.20, "Temperature": 5200.0},
        {"Exposure2012": 0.40, "Temperature": 5600.0},
        {"Exposure2012": 0.30, "Temperature": 5400.0},
        {"Exposure2012": 0.80, "Temperature": 6200.0},
    ]
    metas = [
        {"capture_datetime": dt1, "iso": 400},
        {"capture_datetime": dt2, "iso": 400},
        {"capture_datetime": dt3, "iso": 400},
        {"capture_datetime": dt4, "iso": 400},
    ]

    smooth_burst_predictions(preds, metas, max_burst_gap_seconds=4.0)

    # Median exposure for first 3 shots is 0.30.
    # shot 0: 0.7 * 0.20 + 0.3 * 0.30 = 0.23
    assert preds[0]["Exposure2012"] == pytest.approx(0.23, abs=0.01)
    # shot 1: 0.7 * 0.40 + 0.3 * 0.30 = 0.37
    assert preds[1]["Exposure2012"] == pytest.approx(0.37, abs=0.01)
    # shot 2: 0.7 * 0.30 + 0.3 * 0.30 = 0.30
    assert preds[2]["Exposure2012"] == pytest.approx(0.30, abs=0.01)
    # shot 3 was outside burst gap, remains unblended
    assert preds[3]["Exposure2012"] == 0.80
    assert preds[3]["Temperature"] == 6200.0


def test_apply_denoise_strength_levels() -> None:
    meta = {"iso": 3200}
    for strength, expected_lum in DENOISE_STRENGTH_LEVELS.items():
        sliders = {}
        res = _apply_denoise_if_needed(
            sliders,
            meta,
            denoise_enabled=True,
            denoise_iso_threshold=1200,
            denoise_strength=strength,
        )
        assert res is not None
        assert res["applied"] is True
        assert res["strength"] == strength
        assert sliders["LuminanceSmoothing"] == expected_lum

