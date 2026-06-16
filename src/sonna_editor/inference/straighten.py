"""Preview-based auto-straightening for Lightroom XMP output.

This module deliberately stays outside the trained model. Straightening is a
geometry postprocess: estimate a small horizon/architectural tilt from the RAW
preview, then write Lightroom crop metadata only when confidence is reasonable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians, sin

import cv2
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
_MIN_CONFIDENCE = 0.35
_MIN_EDGE_COUNT = 120
_MIN_LINE_COUNT = 2
_MIN_LINE_LENGTH_PX = 45.0
_HOUGH_THRESHOLD = 28
_MAX_LINE_GAP = 12


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


def _opencv_edges(gray: np.ndarray) -> np.ndarray:
    image_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(image_u8, (5, 5), 0)
    median = float(np.median(blurred))
    lower = int(max(20, 0.66 * median))
    upper = int(min(220, max(lower + 30, 1.33 * median)))
    return cv2.Canny(blurred, lower, upper, apertureSize=3, L2gradient=True)


def _hough_line_residuals(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    min_line_length = max(_MIN_LINE_LENGTH_PX, min(edges.shape) * 0.18)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=_HOUGH_THRESHOLD,
        minLineLength=float(min_line_length),
        maxLineGap=_MAX_LINE_GAP,
    )
    if lines is None:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    residuals: list[float] = []
    weights: list[float] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = (float(v) for v in line)
        dx = x2 - x1
        dy = y2 - y1
        length = hypot(dx, dy)
        if length < min_line_length:
            continue
        angle = degrees(atan2(dy, dx))
        residual = float(_axis_residual_degrees(np.asarray([angle], dtype=np.float64))[0])
        residuals.append(residual)
        weights.append(length)

    return np.asarray(residuals, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def estimate_straighten_angle(image: Image.Image) -> StraightenResult:
    """Estimate a Lightroom CropAngle from a preview image.

    The estimator uses OpenCV, not a learned model. It detects strong preview
    edges with Canny, extracts straight horizontal/vertical candidates through
    a probabilistic Hough transform, reduces line angles to residual tilt from
    Lightroom's nearest 0/90-degree axes, then writes the correction angle.
    """
    gray = np.asarray(_resize_for_analysis(image), dtype=np.float32) / 255.0
    if gray.ndim != 2 or min(gray.shape) < 32:
        return StraightenResult(0.0, 0.0, False, "image_too_small", 0)

    if not np.isfinite(gray).all():
        return StraightenResult(0.0, 0.0, False, "invalid_image", 0)

    edges = _opencv_edges(gray)
    edge_count = int(np.count_nonzero(edges))
    if edge_count == 0:
        return StraightenResult(0.0, 0.0, False, "no_edges", 0)
    if edge_count < _MIN_EDGE_COUNT:
        return StraightenResult(0.0, 0.0, False, "too_few_edges", edge_count)

    residual, weights = _hough_line_residuals(edges)
    if residual.size < _MIN_LINE_COUNT or weights.size < _MIN_LINE_COUNT:
        return StraightenResult(0.0, 0.0, False, "too_few_lines", edge_count)

    median_residual = _weighted_median(residual, weights)
    support = _axis_support(residual, weights, median_residual, tolerance_degrees=2.5)
    concentration = _axis_concentration(residual, weights)
    line_count_score = min(1.0, residual.size / 8.0)
    confidence = concentration * max(support, 0.0) * line_count_score
    angle = -float(median_residual)
    angle = max(-_MAX_APPLY_ANGLE, min(_MAX_APPLY_ANGLE, angle))

    abs_angle = abs(angle)
    if abs_angle < _MIN_APPLY_ANGLE:
        return StraightenResult(0.0, confidence, False, "angle_too_small", edge_count)
    if support < 0.25:
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
