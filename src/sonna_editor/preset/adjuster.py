from __future__ import annotations

import numpy as np
from PIL import Image

from sonna_editor.config import SLIDER_FIELDS, SLIDER_RANGES

# Target middle-grey luminance (0-255)
_TARGET_LUMINANCE = 118.0

# Auto-WB clamp
_MAX_TEMP_DELTA = 300.0
_MAX_TINT_DELTA = 5.0

# Shadow/highlight clipping thresholds
_SHADOW_BIN_RATIO = 0.10    # bottom 10 % of histogram range
_SHADOW_PIXEL_RATIO = 0.25  # >25 % pixels in shadow bins → recover
_HIGHLIGHT_BIN_RATIO = 0.10
_HIGHLIGHT_PIXEL_RATIO = 0.05

_SHADOW_BUMP = 10.0
_HIGHLIGHT_BUMP = -10.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _luminance(image: Image.Image) -> float:
    """Mean perceptual luminance of image (0-255)."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    # Rec. 709 coefficients
    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    return float(lum.mean())


def _stops_to_hit_target(current_luma: float) -> float:
    """Exposure delta in stops needed to reach TARGET_LUMINANCE."""
    if current_luma <= 0:
        return 5.0
    ratio = _TARGET_LUMINANCE / current_luma
    return float(np.log2(ratio))


def _grey_world_wb(image: Image.Image) -> tuple[float, float]:
    """Estimate temp/tint correction using the grey-world assumption.

    Returns (temp_delta, tint_delta) — additive corrections to apply on top
    of the base preset's Temperature and Tint values.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    mean_r = arr[:, :, 0].mean()
    mean_g = arr[:, :, 1].mean()
    mean_b = arr[:, :, 2].mean()

    if mean_g == 0:
        return 0.0, 0.0

    rg = mean_r / mean_g  # >1 means red-heavy → image is warm → cool down
    bg = mean_b / mean_g  # >1 means blue-heavy → image is cool → warm up

    # Heuristic: each 0.1 deviation in rg/bg ≈ ~100K temp shift
    temp_delta = (bg - rg) * 1000.0  # positive = warmer correction needed
    tint_delta = (rg - bg) * 10.0    # crude tint estimation

    return float(temp_delta), float(tint_delta)


def _clip_ratio_at_ends(image: Image.Image) -> tuple[float, float]:
    """Return (shadow_ratio, highlight_ratio) — fraction of pixels in the
    darkest/brightest 10 % of the luminance range."""
    arr = np.asarray(image.convert("L"), dtype=np.uint8).flatten()
    total = len(arr)
    if total == 0:
        return 0.0, 0.0
    shadow_thresh = int(255 * _SHADOW_BIN_RATIO)
    highlight_thresh = int(255 * (1.0 - _HIGHLIGHT_BIN_RATIO))
    shadow_ratio = float((arr <= shadow_thresh).sum()) / total
    highlight_ratio = float((arr >= highlight_thresh).sum()) / total
    return shadow_ratio, highlight_ratio


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_adjustment(
    image: Image.Image,
    metadata: dict,
    base_preset: dict,
    options: dict,
) -> dict:
    """Compute per-photo deltas to apply on top of a base preset.

    Options (all bool, default shown):
        auto_exposure (True)
        auto_white_balance (False)
        auto_shadow_recovery (True)
        auto_highlight_recovery (True)

    Returns a delta dict with only the fields that need overriding.
    """
    delta: dict[str, float] = {}

    auto_exposure = options.get("auto_exposure", True)
    auto_wb = options.get("auto_white_balance", False)
    auto_shadow = options.get("auto_shadow_recovery", True)
    auto_highlight = options.get("auto_highlight_recovery", True)

    if auto_exposure:
        luma = _luminance(image)
        raw_delta = _stops_to_hit_target(luma)
        if abs(raw_delta) > 0.01:
            delta["Exposure2012"] = raw_delta

    if auto_wb:
        temp_d, tint_d = _grey_world_wb(image)
        temp_d = float(np.clip(temp_d, -_MAX_TEMP_DELTA, _MAX_TEMP_DELTA))
        tint_d = float(np.clip(tint_d, -_MAX_TINT_DELTA, _MAX_TINT_DELTA))
        if abs(temp_d) > 10:
            delta["Temperature"] = temp_d
        if abs(tint_d) > 0.5:
            delta["Tint"] = tint_d

    shadow_ratio, highlight_ratio = _clip_ratio_at_ends(image)

    if auto_shadow and shadow_ratio > _SHADOW_PIXEL_RATIO:
        delta["Shadows2012"] = _SHADOW_BUMP

    if auto_highlight and highlight_ratio > _HIGHLIGHT_PIXEL_RATIO:
        delta["Highlights2012"] = _HIGHLIGHT_BUMP

    return delta


def apply_adjustment(base_preset: dict, delta: dict) -> dict:
    """Combine base preset + delta, clamped to valid Lightroom slider ranges.

    Fields absent from both the preset and the delta return None so that
    write_xmp omits them, letting Lightroom keep its own defaults.

    For additive deltas (Exposure, Shadows, etc.) the delta is added to the
    base. For absolute deltas (Temperature, Tint when using WB override) the
    delta is added to the base value.
    """
    result: dict = {}
    for f in SLIDER_FIELDS:
        base = base_preset.get(f)   # None if not in preset
        d = delta.get(f)            # None if no per-photo adjustment
        if base is None and d is None:
            result[f] = None        # omit from XMP; Lightroom uses its own default
        else:
            result[f] = (base if base is not None else 0.0) + (d if d is not None else 0.0)

    # Clamp non-None values to valid ranges.
    # Temperature=0 is the "not specified" sentinel (preset uses as-shot WB).
    # Preserve it as 0 so write_xmp can omit the field; don't clamp to 2000K.
    for field in SLIDER_FIELDS:
        if result[field] is None:
            continue
        if field == "Temperature" and result[field] == 0.0 and "Temperature" not in delta:
            continue
        lo, hi = SLIDER_RANGES.get(field, (-100.0, 100.0))
        result[field] = float(np.clip(result[field], lo, hi))

    return result
