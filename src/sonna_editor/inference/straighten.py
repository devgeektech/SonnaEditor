"""Preview-based auto-straightening for Lightroom XMP output.

This module deliberately stays outside the trained model. Straightening is a
geometry postprocess: estimate a small horizon/architectural tilt from the RAW
preview, then write Lightroom crop metadata only when confidence is reasonable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, radians, sin

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class StraightenResult:
    """Detected straighten angle and confidence for one image preview."""

    angle_degrees: float
    confidence: float
    applied: bool
    reason: str
    edge_count: int


_MAX_WORKING_EDGE = 640
_MIN_APPLY_ANGLE = 0.08
_MAX_APPLY_ANGLE = 5.0
_MIN_CONFIDENCE = 0.18
_MIN_EDGE_COUNT = 500
_MIN_AXIS_SUPPORT = 0.18
_PROJECTION_STEP_DEGREES = 0.1
_MAX_EDGE_POINTS = 4000


def _resize_for_analysis(image: Image.Image) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= _MAX_WORKING_EDGE:
        return image.convert("L")
    scale = _MAX_WORKING_EDGE / float(longest)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.convert("L").resize(size, Image.Resampling.BILINEAR)


def _axis_residual_degrees(tangent_angle_degrees: np.ndarray) -> np.ndarray:
    """Map line angles to residual tilt from the nearest 0/90-degree axis."""
    return ((tangent_angle_degrees + 45.0) % 90.0) - 45.0


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = float(sorted_weights.sum()) * 0.5
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff)])


def _axis_concentration(residual_degrees: np.ndarray, weights: np.ndarray) -> float:
    """Return 0..1 concentration of residuals around one dominant axis tilt."""
    doubled = np.deg2rad(residual_degrees * 2.0)
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    x = float(np.sum(np.cos(doubled) * weights)) / total
    y = float(np.sum(np.sin(doubled) * weights)) / total
    return min(1.0, max(0.0, hypot(x, y)))


def _axis_distance_degrees(values: np.ndarray, centre: float) -> np.ndarray:
    """Return angular distance to centre, with 90-degree axis periodicity."""
    return np.abs(((values - centre + 45.0) % 90.0) - 45.0)


def _axis_support(
    residual_degrees: np.ndarray,
    weights: np.ndarray,
    centre: float,
    tolerance_degrees: float = 2.0,
) -> float:
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    near_axis = _axis_distance_degrees(residual_degrees, centre) <= tolerance_degrees
    return float(weights[near_axis].sum()) / total


def _projection_search_angle(mask: np.ndarray, weights_image: np.ndarray) -> tuple[float, float]:
    """Find the small rotation that makes strong edges most axis-aligned."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0, 0.0

    weights = weights_image[ys, xs].astype(np.float64)
    if xs.size > _MAX_EDGE_POINTS:
        order = np.argsort(weights)[-_MAX_EDGE_POINTS:]
        xs = xs[order]
        ys = ys[order]
        weights = weights[order]

    x = xs.astype(np.float64) - (mask.shape[1] - 1) / 2.0
    y = ys.astype(np.float64) - (mask.shape[0] - 1) / 2.0

    candidates = np.arange(
        -_MAX_APPLY_ANGLE,
        _MAX_APPLY_ANGLE + _PROJECTION_STEP_DEGREES / 2.0,
        _PROJECTION_STEP_DEGREES,
    )
    scores: list[float] = []
    for angle in candidates:
        theta = radians(float(angle))
        c = cos(theta)
        s = sin(theta)
        xr = x * c - y * s
        yr = x * s + y * c
        xbins = np.rint(xr - xr.min()).astype(np.int32)
        ybins = np.rint(yr - yr.min()).astype(np.int32)
        xhist = np.bincount(xbins, weights=weights)
        yhist = np.bincount(ybins, weights=weights)
        scores.append(float(np.sum(xhist * xhist) + np.sum(yhist * yhist)))

    score_arr = np.asarray(scores, dtype=np.float64)
    best_index = int(np.argmax(score_arr))
    best_score = float(score_arr[best_index])
    if best_score <= 0:
        return 0.0, 0.0

    baseline = float(np.median(score_arr))
    confidence = max(0.0, min(1.0, (best_score - baseline) / best_score * 2.0))
    return float(candidates[best_index]), confidence


def estimate_straighten_angle(image: Image.Image) -> StraightenResult:
    """Estimate a Lightroom CropAngle from a preview image.

    The estimator uses image gradients rather than a learned model. It finds
    strong edges, converts edge normals into line/tangent angles, reduces those
    angles to residual tilt from horizontal/vertical axes, then uses a weighted
    median for robustness. The returned angle is the correction to apply.
    """
    gray = np.asarray(_resize_for_analysis(image), dtype=np.float32) / 255.0
    if gray.ndim != 2 or min(gray.shape) < 32:
        return StraightenResult(0.0, 0.0, False, "image_too_small", 0)

    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    if not np.isfinite(mag).all():
        return StraightenResult(0.0, 0.0, False, "invalid_gradient", 0)

    nonzero = mag[mag > 0]
    if nonzero.size == 0:
        return StraightenResult(0.0, 0.0, False, "no_edges", 0)
    threshold = float(np.percentile(nonzero, 75.0))
    if threshold <= 0:
        return StraightenResult(0.0, 0.0, False, "no_edges", 0)

    mask = mag >= threshold
    edge_count = int(mask.sum())
    if edge_count < _MIN_EDGE_COUNT:
        return StraightenResult(0.0, 0.0, False, "too_few_edges", edge_count)

    tangent = np.degrees(np.arctan2(gy[mask], gx[mask])) + 90.0
    residual = _axis_residual_degrees(tangent)
    weights = mag[mask].astype(np.float64)
    residual = residual.astype(np.float64)

    median_residual = _weighted_median(residual, weights)
    centred = residual - median_residual
    inlier = np.abs(centred) <= 3.0
    if int(inlier.sum()) >= max(50, edge_count // 20):
        median_residual = _weighted_median(residual[inlier], weights[inlier])

    projection_angle, projection_confidence = _projection_search_angle(mask, mag)
    median_angle = -float(median_residual)
    median_support = _axis_support(residual, weights, median_residual)

    if (
        abs(projection_angle) >= _MIN_APPLY_ANGLE
        and projection_confidence >= _MIN_CONFIDENCE
    ):
        angle = projection_angle
        support = 1.0
    else:
        angle = median_angle
        support = median_support
    angle = max(-_MAX_APPLY_ANGLE, min(_MAX_APPLY_ANGLE, angle))
    concentration = _axis_concentration(residual, weights)
    confidence = max(concentration, projection_confidence) * min(1.0, support / _MIN_AXIS_SUPPORT)

    abs_angle = abs(angle)
    if abs_angle < _MIN_APPLY_ANGLE:
        return StraightenResult(0.0, confidence, False, "angle_too_small", edge_count)
    if support < _MIN_AXIS_SUPPORT:
        return StraightenResult(angle, confidence, False, "weak_axis_support", edge_count)
    if confidence < _MIN_CONFIDENCE:
        return StraightenResult(angle, confidence, False, "low_confidence", edge_count)

    return StraightenResult(round(angle, 4), confidence, True, "applied", edge_count)


def crop_angle_attributes(result: StraightenResult) -> dict[str, str]:
    """Return Lightroom CRS attributes for an applied straighten result."""
    if not result.applied:
        return {}
    return {
        "HasCrop": "True",
        "CropAngle": _format_lightroom_angle(result.angle_degrees),
    }


def _format_lightroom_angle(angle: float) -> str:
    if angle == int(angle):
        return f"+{int(angle)}" if angle > 0 else str(int(angle))
    text = f"{angle:+.4f}".rstrip("0").rstrip(".")
    return text if text != "-0" else "0"


def rotated_content_scale(angle_degrees: float, width: int, height: int) -> float:
    """Return the minimal scale needed to hide empty corners after rotation.

    This is diagnostic only for now. Lightroom owns the actual crop rendering,
    but the value is useful in tests and future UI reporting.
    """
    if width <= 0 or height <= 0:
        return 1.0
    theta = abs(radians(angle_degrees))
    c = abs(cos(theta))
    s = abs(sin(theta))
    new_w = width * c + height * s
    new_h = width * s + height * c
    return max(new_w / width, new_h / height)
