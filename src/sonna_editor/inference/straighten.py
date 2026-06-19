"""Preview-based auto-straightening for Lightroom XMP output.

This module deliberately stays outside the trained model. Straightening is a
geometry postprocess: estimate a small horizon/architectural tilt from the RAW
preview, then write Lightroom straighten metadata only when confidence is
reasonable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians, sin

import cv2
import numpy as np
from PIL import Image


STRAIGHTEN_ENGINE_VERSION = "opencv-scene-horizon-lines-v3"


@dataclass(frozen=True)
class StraightenResult:
    """Detected straighten angle and confidence for one image preview."""

    angle_degrees: float
    confidence: float
    applied: bool
    reason: str
    edge_count: int
    line_count: int = 0
    total_line_length: float = 0.0
    scene_type: str = "unknown"
    horizon_score: float = 0.0
    axis_score: float = 0.0
    horizontal_line_count: int = 0
    vertical_line_count: int = 0


@dataclass(frozen=True)
class _LineObservation:
    residual: float
    orientation: str
    weight: float
    length: float
    mid_x: float
    mid_y: float


@dataclass(frozen=True)
class _AngleCandidate:
    angle: float
    confidence: float
    support: float
    concentration: float
    line_count: int
    total_weight: float


_MAX_WORKING_EDGE = 640
_MIN_APPLY_ANGLE = 0.08
_MAX_APPLY_ANGLE = 5.0
_MIN_CONFIDENCE = 0.28
_MIN_HORIZON_CONFIDENCE = 0.22
_MIN_MIXED_CONFIDENCE = 0.34
_MIN_EDGE_COUNT = 120
_MAX_EDGE_DENSITY = 0.32
_MIN_LINE_COUNT = 1
_MIN_LINE_LENGTH_PX = 24.0
_HOUGH_THRESHOLD = 18
_MAX_LINE_GAP = 16
_AXIS_CANDIDATE_TOLERANCE_DEGREES = 7.0
_AXIS_SUPPORT_TOLERANCE_DEGREES = 2.8
_HORIZON_BAND_TOP = 0.16
_HORIZON_BAND_BOTTOM = 0.82
_CENTRE_HORIZON_BAND_TOP = 0.34
_CENTRE_HORIZON_BAND_BOTTOM = 0.66


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


def _analysis_image(gray: np.ndarray) -> np.ndarray:
    image_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    equalised = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_u8)
    return cv2.GaussianBlur(equalised, (3, 3), 0)


def _opencv_edges(image_u8: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(image_u8, 40, 120, apertureSize=3, L2gradient=True)
    if int(np.count_nonzero(edges)) >= _MIN_EDGE_COUNT:
        return edges
    return cv2.Canny(image_u8, 16, 64, apertureSize=3, L2gradient=True)


def _orientation_from_tangent(tangent_angle_degrees: float) -> str:
    folded = ((tangent_angle_degrees + 90.0) % 180.0) - 90.0
    return "horizontal" if abs(folded) <= 45.0 else "vertical"


def _segment_observations(
    segments: np.ndarray,
    min_line_length: float,
    *,
    weight_scale: float = 1.0,
) -> list[_LineObservation]:
    observations: list[_LineObservation] = []
    for segment in segments.reshape(-1, 4):
        x1, y1, x2, y2 = (float(v) for v in segment)
        dx = x2 - x1
        dy = y2 - y1
        length = hypot(dx, dy)
        if length < min_line_length:
            continue
        angle = degrees(atan2(dy, dx))
        residual = float(_axis_residual_degrees(np.asarray([angle], dtype=np.float64))[0])
        if abs(residual) > _AXIS_CANDIDATE_TOLERANCE_DEGREES:
            continue
        observations.append(
            _LineObservation(
                residual=residual,
                orientation=_orientation_from_tangent(angle),
                weight=length * weight_scale,
                length=length,
                mid_x=(x1 + x2) * 0.5,
                mid_y=(y1 + y2) * 0.5,
            )
        )
    return observations


def _hough_line_observations(edges: np.ndarray) -> list[_LineObservation]:
    min_line_length = max(_MIN_LINE_LENGTH_PX, min(edges.shape) * 0.10)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=_HOUGH_THRESHOLD,
        minLineLength=float(min_line_length),
        maxLineGap=_MAX_LINE_GAP,
    )
    if lines is None:
        return []
    return _segment_observations(lines[:, 0, :], min_line_length, weight_scale=1.15)


def _lsd_line_observations(image_u8: np.ndarray) -> list[_LineObservation]:
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = detector.detect(image_u8)[0]
    if lines is None:
        return []
    min_line_length = max(_MIN_LINE_LENGTH_PX, min(image_u8.shape) * 0.06)
    return _segment_observations(lines[:, 0, :], min_line_length, weight_scale=1.0)


def _standard_hough_line_observations(edges: np.ndarray) -> list[_LineObservation]:
    """Detect broad axis-aligned line evidence when segments are fragmented."""
    min_edge = min(edges.shape)
    threshold = max(36, min(120, int(min_edge * 0.14)))
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180.0, threshold=threshold)
    if lines is None:
        return []

    observations: list[_LineObservation] = []
    base_weight = max(_MIN_LINE_LENGTH_PX, min_edge * 0.22)
    for rho_theta in lines[:80, 0, :]:
        rho, theta = (float(v) for v in rho_theta)
        tangent_angle = degrees(theta) - 90.0
        residual = float(
            _axis_residual_degrees(np.asarray([tangent_angle], dtype=np.float64))[0]
        )
        if abs(residual) > _AXIS_CANDIDATE_TOLERANCE_DEGREES:
            continue
        # HoughLines does not expose vote counts in this binding, so keep the
        # fallback lighter than explicit segment detections.
        observations.append(
            _LineObservation(
                residual=residual,
                orientation=_orientation_from_tangent(tangent_angle),
                weight=base_weight * 0.55,
                length=base_weight,
                mid_x=edges.shape[1] * 0.5,
                # The normal-form rho is only a rough position estimate here,
                # but it is still useful enough for horizon band scoring.
                mid_y=max(0.0, min(float(edges.shape[0] - 1), abs(rho))),
            )
        )
    return observations


def _opencv_line_observations(image_u8: np.ndarray, edges: np.ndarray) -> list[_LineObservation]:
    observations = _hough_line_observations(edges) + _lsd_line_observations(image_u8)
    if len(observations) < 2:
        observations.extend(_standard_hough_line_observations(edges))
    return observations


def _candidate_from_observations(
    observations: list[_LineObservation],
    image_shape: tuple[int, int],
) -> _AngleCandidate | None:
    if not observations:
        return None
    residual = np.asarray([item.residual for item in observations], dtype=np.float64)
    weights = np.asarray([item.weight for item in observations], dtype=np.float64)
    if residual.size < _MIN_LINE_COUNT or weights.size < _MIN_LINE_COUNT:
        return None

    median_residual = _weighted_median(residual, weights)
    support = _axis_support(
        residual,
        weights,
        median_residual,
        tolerance_degrees=_AXIS_SUPPORT_TOLERANCE_DEGREES,
    )
    concentration = _axis_concentration(residual, weights)
    min_side = float(min(image_shape))
    line_count_score = min(1.0, residual.size / 4.0)
    line_length_score = min(1.0, float(weights.sum()) / (min_side * 1.2))
    confidence = concentration * max(support, 0.0) * max(line_count_score, line_length_score)
    angle = max(-_MAX_APPLY_ANGLE, min(_MAX_APPLY_ANGLE, float(median_residual)))
    return _AngleCandidate(
        angle=angle,
        confidence=confidence,
        support=support,
        concentration=concentration,
        line_count=int(residual.size),
        total_weight=float(weights.sum()),
    )


def _position_band_score(observations: list[_LineObservation], height: int) -> float:
    if not observations or height <= 0:
        return 0.0
    total = sum(item.weight for item in observations)
    if total <= 0:
        return 0.0
    score = 0.0
    for item in observations:
        y_ratio = item.mid_y / float(height)
        if _HORIZON_BAND_TOP <= y_ratio <= _HORIZON_BAND_BOTTOM:
            score += item.weight
    return score / total


def _centre_band_observations(
    observations: list[_LineObservation],
    height: int,
) -> list[_LineObservation]:
    if height <= 0:
        return []
    return [
        item
        for item in observations
        if _CENTRE_HORIZON_BAND_TOP
        <= item.mid_y / float(height)
        <= _CENTRE_HORIZON_BAND_BOTTOM
    ]


def _horizontal_coverage_score(observations: list[_LineObservation], width: int) -> float:
    if width <= 0:
        return 0.0
    horizontal_length = sum(item.length for item in observations)
    return min(1.0, horizontal_length / float(width))


def _angle_consistency(first: _AngleCandidate | None, second: _AngleCandidate | None) -> float:
    if first is None or second is None:
        return 0.0
    distance = abs(float(_axis_distance_degrees(np.asarray([first.angle]), second.angle)[0]))
    return max(0.0, 1.0 - distance / 4.0)


def _scene_candidates(
    observations: list[_LineObservation],
    image_shape: tuple[int, int],
) -> tuple[str, _AngleCandidate | None, float, float, int, int]:
    height, width = image_shape
    horizontal = [item for item in observations if item.orientation == "horizontal"]
    centre_horizontal = _centre_band_observations(horizontal, height)
    horizon_source = centre_horizontal or horizontal
    vertical = [item for item in observations if item.orientation == "vertical"]

    axis_candidate = _candidate_from_observations(observations, image_shape)
    horizon_candidate = _candidate_from_observations(horizon_source, image_shape)
    vertical_candidate = _candidate_from_observations(vertical, image_shape)

    horizon_score = 0.0
    if horizon_candidate is not None:
        band_score = _position_band_score(horizon_source, height)
        coverage_score = _horizontal_coverage_score(horizon_source, width)
        centre_bonus = 0.12 if centre_horizontal else 0.0
        horizon_score = horizon_candidate.confidence * (
            0.55 + 0.30 * band_score + 0.15 * coverage_score + centre_bonus
        )

    axis_score = axis_candidate.confidence if axis_candidate is not None else 0.0
    if axis_candidate is not None and horizon_candidate is not None and vertical_candidate is not None:
        consistency = _angle_consistency(horizon_candidate, vertical_candidate)
        balance = min(1.0, min(horizon_candidate.total_weight, vertical_candidate.total_weight) / max(1.0, min(image_shape) * 0.35))
        axis_score = max(axis_score, axis_candidate.confidence * (0.75 + 0.25 * consistency) * (0.70 + 0.30 * balance))

    if horizon_candidate is not None and horizon_score >= max(axis_score * 1.05, _MIN_HORIZON_CONFIDENCE):
        return "horizon", horizon_candidate, horizon_score, axis_score, len(horizontal), len(vertical)
    if (
        axis_candidate is not None
        and horizon_candidate is not None
        and vertical_candidate is not None
        and axis_score >= max(horizon_score * 0.92, _MIN_CONFIDENCE)
        and _angle_consistency(horizon_candidate, vertical_candidate) >= 0.35
    ):
        return "architecture", axis_candidate, horizon_score, axis_score, len(horizontal), len(vertical)
    if axis_candidate is not None:
        return "mixed_axis", axis_candidate, horizon_score, axis_score, len(horizontal), len(vertical)
    return "unclassified", None, horizon_score, axis_score, len(horizontal), len(vertical)


def estimate_straighten_angle(image: Image.Image) -> StraightenResult:
    """Estimate a Lightroom straighten angle from a preview image.

    The estimator uses OpenCV, not a learned model. It detects strong preview
    edges with Canny, extracts straight horizontal/vertical candidates through
    a probabilistic Hough transform, reduces line angles to residual tilt from
    Lightroom's nearest 0/90-degree axes, then writes the correction angle.
    """
    gray = np.asarray(_resize_for_analysis(image), dtype=np.float32) / 255.0
    if gray.ndim != 2 or min(gray.shape) < 32:
        return StraightenResult(0.0, 0.0, False, "image_too_small", 0, scene_type="invalid")

    if not np.isfinite(gray).all():
        return StraightenResult(0.0, 0.0, False, "invalid_image", 0, scene_type="invalid")

    image_u8 = _analysis_image(gray)
    edges = _opencv_edges(image_u8)
    edge_count = int(np.count_nonzero(edges))
    if edge_count == 0:
        return StraightenResult(0.0, 0.0, False, "no_edges", 0, scene_type="blank")
    if edge_count < _MIN_EDGE_COUNT:
        return StraightenResult(
            0.0, 0.0, False, "too_few_edges", edge_count, scene_type="detail"
        )
    if edge_count / float(edges.size) > _MAX_EDGE_DENSITY:
        return StraightenResult(
            0.0, 0.0, False, "edge_texture", edge_count, scene_type="texture"
        )

    observations = _opencv_line_observations(image_u8, edges)
    line_count = len(observations)
    total_line_length = round(float(sum(item.weight for item in observations)), 2)
    scene_type, candidate, horizon_score, axis_score, horizontal_count, vertical_count = (
        _scene_candidates(observations, gray.shape)
    )
    if candidate is None:
        return StraightenResult(
            0.0,
            0.0,
            False,
            "too_few_lines",
            edge_count,
            line_count,
            total_line_length,
            scene_type=scene_type,
            horizon_score=round(horizon_score, 4),
            axis_score=round(axis_score, 4),
            horizontal_line_count=horizontal_count,
            vertical_line_count=vertical_count,
        )

    confidence = candidate.confidence
    # Lightroom's visible correction direction matches the OpenCV line
    # residuals for the straighten routes we write, so keep the sign instead
    # of negating it.
    angle = float(candidate.angle)
    angle = max(-_MAX_APPLY_ANGLE, min(_MAX_APPLY_ANGLE, angle))

    abs_angle = abs(angle)
    if abs_angle < _MIN_APPLY_ANGLE:
        return StraightenResult(
            0.0,
            confidence,
            False,
            "angle_too_small",
            edge_count,
            line_count,
            total_line_length,
            scene_type=scene_type,
            horizon_score=round(horizon_score, 4),
            axis_score=round(axis_score, 4),
            horizontal_line_count=horizontal_count,
            vertical_line_count=vertical_count,
        )
    if candidate.support < 0.25:
        return StraightenResult(
            angle,
            confidence,
            False,
            "weak_axis_support",
            edge_count,
            line_count,
            total_line_length,
            scene_type=scene_type,
            horizon_score=round(horizon_score, 4),
            axis_score=round(axis_score, 4),
            horizontal_line_count=horizontal_count,
            vertical_line_count=vertical_count,
        )
    min_confidence = (
        _MIN_HORIZON_CONFIDENCE
        if scene_type == "horizon"
        else _MIN_CONFIDENCE
        if scene_type == "architecture"
        else _MIN_MIXED_CONFIDENCE
    )
    if confidence < min_confidence:
        return StraightenResult(
            angle,
            confidence,
            False,
            "low_confidence",
            edge_count,
            line_count,
            total_line_length,
            scene_type=scene_type,
            horizon_score=round(horizon_score, 4),
            axis_score=round(axis_score, 4),
            horizontal_line_count=horizontal_count,
            vertical_line_count=vertical_count,
        )

    return StraightenResult(
        round(angle, 4),
        confidence,
        True,
        "applied",
        edge_count,
        line_count,
        total_line_length,
        scene_type=scene_type,
        horizon_score=round(horizon_score, 4),
        axis_score=round(axis_score, 4),
        horizontal_line_count=horizontal_count,
        vertical_line_count=vertical_count,
    )


def perspective_rotate_attributes(
    result: StraightenResult,
    image_size: tuple[int, int] | None = None,
) -> dict[str, str]:
    """Return Lightroom CRS Transform attributes for an applied straighten result.

    This branch intentionally avoids CropAngle/CropTop/CropLeft/CropBottom/
    CropRight and uses Lightroom's Transform rotation path instead. The
    PerspectiveScale value zooms just enough to hide empty corners after the
    rotation when a preview size is available.
    """
    if not result.applied:
        return {}
    scale = 100.0
    if image_size is not None:
        width, height = image_size
        scale = max(
            100.0,
            min(150.0, rotated_content_scale(result.angle_degrees, width, height) * 100.0),
        )
    return {
        "PerspectiveRotate": _format_lightroom_angle(result.angle_degrees),
        "PerspectiveScale": _format_scale(scale),
        "AlreadyApplied": "False",
    }


def _format_lightroom_angle(angle: float) -> str:
    if angle == int(angle):
        return f"+{int(angle)}" if angle > 0 else str(int(angle))
    text = f"{angle:+.4f}".rstrip("0").rstrip(".")
    return text if text != "-0" else "0"


def _format_scale(scale: float) -> str:
    text = f"{scale:.4f}".rstrip("0").rstrip(".")
    return text if text else "100"


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
