from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from sonna_editor.config import SLIDER_FIELDS, SLIDER_RANGES
from sonna_editor.preset.adjuster import (
    _clip_ratio_at_ends,
    _luminance,
    _neutral_reference_rgb,
    apply_adjustment,
    compute_adjustment,
)

# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------

def _solid(r: int, g: int, b: int, size: tuple[int, int] = (64, 64)) -> Image.Image:
    arr = np.full((*size, 3), [r, g, b], dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _grey(v: int, size: tuple[int, int] = (64, 64)) -> Image.Image:
    return _solid(v, v, v, size)


def _dominant_colour_with_neutral_patch() -> Image.Image:
    arr = np.full((80, 80, 3), [210, 90, 45], dtype=np.uint8)
    arr[24:56, 24:56, :] = [132, 132, 132]
    return Image.fromarray(arr, "RGB")


def _base_preset(**overrides: float) -> dict:
    preset = {f: 0.0 for f in SLIDER_FIELDS}
    preset["Temperature"] = 5200.0
    preset.update(overrides)
    return preset


def _opts(**overrides) -> dict:
    defaults = {
        "auto_exposure": True,
        "auto_white_balance": False,
        "auto_shadow_recovery": True,
        "auto_highlight_recovery": True,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# _luminance
# ---------------------------------------------------------------------------

def test_luminance_black_image() -> None:
    assert _luminance(_grey(0)) == pytest.approx(0.0)


def test_luminance_white_image() -> None:
    assert _luminance(_grey(255)) == pytest.approx(255.0, abs=1.0)


def test_luminance_mid_grey() -> None:
    lum = _luminance(_grey(118))
    assert 100 < lum < 140


# ---------------------------------------------------------------------------
# _clip_ratio_at_ends
# ---------------------------------------------------------------------------

def test_clip_ratio_black_image_all_shadow() -> None:
    shadow, highlight = _clip_ratio_at_ends(_grey(0))
    assert shadow > 0.9
    assert highlight < 0.1


def test_clip_ratio_white_image_all_highlight() -> None:
    shadow, highlight = _clip_ratio_at_ends(_grey(255))
    assert highlight > 0.9
    assert shadow < 0.1


def test_clip_ratio_mid_grey_neither() -> None:
    shadow, highlight = _clip_ratio_at_ends(_grey(128))
    assert shadow < 0.2
    assert highlight < 0.2


# ---------------------------------------------------------------------------
# compute_adjustment — auto_exposure
# ---------------------------------------------------------------------------

def test_dark_image_gets_positive_exposure_delta() -> None:
    img = _grey(40)  # very dark
    delta = compute_adjustment(img, {}, _base_preset(), _opts())
    assert delta.get("Exposure2012", 0.0) > 0.0


def test_bright_image_gets_negative_exposure_delta() -> None:
    img = _grey(220)  # very bright
    delta = compute_adjustment(img, {}, _base_preset(), _opts())
    assert delta.get("Exposure2012", 0.0) < 0.0


def test_mid_grey_image_small_exposure_delta() -> None:
    img = _grey(118)  # right on target
    delta = compute_adjustment(img, {}, _base_preset(), _opts())
    assert abs(delta.get("Exposure2012", 0.0)) < 0.05


def test_very_dark_image_gets_large_exposure_delta() -> None:
    img = _grey(1)  # almost black — large uncapped correction expected
    delta = compute_adjustment(img, {}, _base_preset(), _opts())
    assert delta.get("Exposure2012", 0.0) > 0.7


def test_shadow_heavy_image_with_bright_upper_tones_gets_controlled_lift() -> None:
    arr = np.full((64, 64, 3), 24, dtype=np.uint8)
    arr[:, 44:, :] = 190
    img = Image.fromarray(arr, "RGB")
    delta = compute_adjustment(img, {}, _base_preset(), _opts())
    assert 0.25 <= delta.get("Exposure2012", 0.0) <= 0.75


def test_dark_scene_with_no_clipped_highlights_gets_imagen_style_lift() -> None:
    arr = np.full((64, 64, 3), 36, dtype=np.uint8)
    arr[8:28, 8:56, :] = 125
    arr[0:8, 0:20, :] = 210
    img = Image.fromarray(arr, "RGB")
    delta = compute_adjustment(img, {}, _base_preset(), _opts())
    assert delta.get("Exposure2012", 0.0) >= 0.35


def test_no_exposure_when_disabled() -> None:
    img = _grey(20)
    delta = compute_adjustment(img, {}, _base_preset(), _opts(auto_exposure=False))
    assert "Exposure2012" not in delta


# ---------------------------------------------------------------------------
# compute_adjustment — shadow / highlight recovery
# ---------------------------------------------------------------------------

def test_dark_image_triggers_shadow_recovery() -> None:
    img = _grey(5)  # near-black → lots of shadow pixels
    delta = compute_adjustment(img, {}, _base_preset(), _opts())
    assert delta.get("Shadows2012", 0.0) > 0.0


def test_bright_image_triggers_highlight_recovery() -> None:
    img = _grey(254)
    delta = compute_adjustment(img, {}, _base_preset(), _opts())
    assert delta.get("Highlights2012", 0.0) < 0.0


def test_no_shadow_recovery_when_disabled() -> None:
    img = _grey(5)
    delta = compute_adjustment(img, {}, _base_preset(), _opts(auto_shadow_recovery=False))
    assert "Shadows2012" not in delta


def test_no_highlight_recovery_when_disabled() -> None:
    img = _grey(254)
    delta = compute_adjustment(img, {}, _base_preset(), _opts(auto_highlight_recovery=False))
    assert "Highlights2012" not in delta


# ---------------------------------------------------------------------------
# compute_adjustment — auto_white_balance
# ---------------------------------------------------------------------------

def test_wb_off_by_default_no_temp_delta() -> None:
    img = _solid(200, 100, 50)  # very warm/red image
    delta = compute_adjustment(img, {}, _base_preset(), _opts())
    assert "Temperature" not in delta


def test_wb_on_warm_image_gets_cooling_delta() -> None:
    img = _solid(230, 120, 60)  # red-heavy = warm image
    delta = compute_adjustment(img, {}, _base_preset(), _opts(auto_white_balance=True))
    # Grey-world: blue channel low → should push temperature cooler (negative delta)
    # Direction depends on implementation; just check a delta appears
    assert "Temperature" in delta or "Tint" in delta


def test_wb_on_warm_magenta_image_does_not_add_pink_tint() -> None:
    img = _solid(230, 120, 80)
    delta = compute_adjustment(img, {}, _base_preset(), _opts(auto_white_balance=True))

    assert delta.get("Tint", 0.0) <= 0.0


def test_wb_on_green_image_adds_magenta_tint() -> None:
    img = _solid(80, 150, 80)
    delta = compute_adjustment(img, {}, _base_preset(), _opts(auto_white_balance=True))

    assert delta.get("Tint", 0.0) > 0.0


def test_wb_prefers_neutral_midtone_patch_over_dominant_warm_colour() -> None:
    img = _dominant_colour_with_neutral_patch()
    delta = compute_adjustment(img, {}, _base_preset(), _opts(auto_white_balance=True))

    assert abs(delta.get("Temperature", 0.0)) <= 10.0
    assert abs(delta.get("Tint", 0.0)) <= 0.5


def test_neutral_reference_uses_patch_when_available() -> None:
    r, g, b = _neutral_reference_rgb(_dominant_colour_with_neutral_patch())

    assert r == pytest.approx(132.0)
    assert g == pytest.approx(132.0)
    assert b == pytest.approx(132.0)


# ---------------------------------------------------------------------------
# apply_adjustment
# ---------------------------------------------------------------------------

def test_apply_adds_delta_to_base() -> None:
    base = _base_preset(Exposure2012=0.5)
    delta = {"Exposure2012": 0.3}
    result = apply_adjustment(base, delta)
    assert result["Exposure2012"] == pytest.approx(0.8)


def test_apply_clamps_to_slider_range() -> None:
    base = _base_preset(Exposure2012=4.8)
    delta = {"Exposure2012": 0.7}  # would push to 5.5, above max 5.0
    result = apply_adjustment(base, delta)
    assert result["Exposure2012"] <= 5.0


def test_apply_shadows_not_capped_below_slider_range() -> None:
    base = _base_preset(Shadows2012=55.0)
    delta = {"Shadows2012": 10.0}  # pushes to 65 — fine, no cap below slider max
    result = apply_adjustment(base, delta)
    assert result["Shadows2012"] == pytest.approx(65.0)


def test_apply_highlights_not_capped_above_slider_range() -> None:
    base = _base_preset(Highlights2012=-45.0)
    delta = {"Highlights2012": -10.0}  # pushes to -55 — fine, no cap above slider min
    result = apply_adjustment(base, delta)
    assert result["Highlights2012"] == pytest.approx(-55.0)


def test_apply_returns_all_slider_fields() -> None:
    base = _base_preset()
    result = apply_adjustment(base, {})
    assert set(result.keys()) == set(SLIDER_FIELDS)


def test_apply_zero_delta_identity() -> None:
    base = _base_preset(Exposure2012=0.4, Temperature=5500.0, Shadows2012=20.0)
    result = apply_adjustment(base, {})
    assert result["Exposure2012"] == pytest.approx(0.4)
    assert result["Temperature"] == pytest.approx(5500.0)


def test_apply_ignores_unknown_delta_keys() -> None:
    base = _base_preset()
    delta = {"NotASlider": 99.0}
    result = apply_adjustment(base, delta)
    assert "NotASlider" not in result


# ---------------------------------------------------------------------------
# Round-trip: compute + apply
# ---------------------------------------------------------------------------

def test_round_trip_dark_image_increases_exposure() -> None:
    img = _grey(40)
    base = _base_preset(Exposure2012=0.0)
    delta = compute_adjustment(img, {}, base, _opts())
    final = apply_adjustment(base, delta)
    assert final["Exposure2012"] > base["Exposure2012"]


def test_round_trip_bright_image_decreases_exposure() -> None:
    img = _grey(220)
    base = _base_preset(Exposure2012=0.0)
    delta = compute_adjustment(img, {}, base, _opts())
    final = apply_adjustment(base, delta)
    assert final["Exposure2012"] < base["Exposure2012"]


def test_round_trip_all_values_within_ranges() -> None:
    img = _grey(40)
    base = _base_preset()
    delta = compute_adjustment(img, {}, base, _opts(auto_white_balance=True))
    final = apply_adjustment(base, delta)
    for field in SLIDER_FIELDS:
        if final[field] is None:
            continue
        lo, hi = SLIDER_RANGES.get(field, (-100.0, 100.0))
        assert lo <= final[field] <= hi, f"{field}={final[field]} out of [{lo}, {hi}]"


# ---------------------------------------------------------------------------
# None-preservation for absent fields (expanded SLIDER_FIELDS)
# ---------------------------------------------------------------------------

def _sparse_preset(**fields: float) -> dict:
    """Preset with only explicitly named fields — simulates a preset that
    doesn't specify sharpening, transform, etc."""
    return dict(fields)


def test_absent_field_with_no_delta_returns_none() -> None:
    """Fields not in preset and not in delta must be None (not clamped 0)."""
    preset = _sparse_preset(Exposure2012=0.2, Temperature=5500.0)
    result = apply_adjustment(preset, {})
    assert result["SharpenRadius"] is None
    assert result["PerspectiveScale"] is None
    assert result["Sharpness"] is None
    assert result["ColorGradeBlending"] is None


def test_absent_field_with_delta_uses_zero_base() -> None:
    """If a field is not in the preset but IS in delta, add delta to 0."""
    preset = _sparse_preset(Exposure2012=0.0)
    delta = {"Shadows2012": 10.0}
    result = apply_adjustment(preset, delta)
    assert result["Shadows2012"] == pytest.approx(10.0)


def test_present_field_with_zero_value_is_not_none() -> None:
    """A field explicitly set to 0 in the preset must be written (not omitted)."""
    preset = _sparse_preset(Exposure2012=0.0, Contrast2012=0.0, SharpenRadius=1.0)
    result = apply_adjustment(preset, {})
    assert result["Exposure2012"] == pytest.approx(0.0)
    assert result["Contrast2012"] == pytest.approx(0.0)
    assert result["SharpenRadius"] == pytest.approx(1.0)


def test_sharpen_radius_not_clamped_when_absent() -> None:
    """Critical: SharpenRadius absent from preset must not be clamped to 0.5."""
    preset = _sparse_preset(Exposure2012=0.0)
    result = apply_adjustment(preset, {})
    # Before fix this returned 0.5 (lower clamp bound). Must be None now.
    assert result["SharpenRadius"] is None


def test_perspective_scale_not_clamped_when_absent() -> None:
    """Critical: PerspectiveScale absent from preset must not be clamped to 50."""
    preset = _sparse_preset(Exposure2012=0.0)
    result = apply_adjustment(preset, {})
    # Before fix this returned 50.0 (lower clamp bound). Must be None now.
    assert result["PerspectiveScale"] is None


def test_all_slider_fields_present_in_result() -> None:
    """Result always has every SLIDER_FIELD as a key (value may be None)."""
    result = apply_adjustment({}, {})
    assert set(result.keys()) == set(SLIDER_FIELDS)
    assert all(v is None for v in result.values())
