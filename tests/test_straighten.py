from __future__ import annotations

import math

import pytest
from PIL import Image, ImageDraw

from sonna_editor.inference.straighten import (
    STRAIGHTEN_ENGINE_VERSION,
    estimate_straighten_angle,
    perspective_rotate_attributes,
    rotated_content_scale,
)


def _tilted_line_image(angle_degrees: float) -> Image.Image:
    image = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(image)
    cx, cy = 200, 150
    length = 520
    theta = math.radians(angle_degrees)
    dx = math.cos(theta) * length / 2
    dy = math.sin(theta) * length / 2
    draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill="black", width=5)
    return image


def _rotate_points(
    points: list[tuple[float, float]],
    angle_degrees: float,
    centre: tuple[float, float],
) -> list[tuple[float, float]]:
    theta = math.radians(angle_degrees)
    c = math.cos(theta)
    s = math.sin(theta)
    cx, cy = centre
    rotated: list[tuple[float, float]] = []
    for x, y in points:
        x -= cx
        y -= cy
        rotated.append((cx + x * c - y * s, cy + x * s + y * c))
    return rotated


def _tilted_room_image(angle_degrees: float) -> Image.Image:
    image = Image.new("RGB", (800, 533), (235, 235, 230))
    draw = ImageDraw.Draw(image)
    centre = (400.0, 266.0)
    lines = [[(60.0, y), (740.0, y)] for y in (80.0, 160.0, 260.0, 380.0, 460.0)]
    lines.extend([[(x, 60.0), (x, 480.0)] for x in (120.0, 300.0, 520.0, 680.0)])
    for line in lines:
        draw.line(_rotate_points(line, angle_degrees, centre), fill=(80, 80, 80), width=3)
    return image


def _faint_tilted_room_image(angle_degrees: float) -> Image.Image:
    image = Image.new("RGB", (800, 533), (218, 218, 214))
    draw = ImageDraw.Draw(image)
    centre = (400.0, 266.0)
    lines = [[(90.0, y), (710.0, y)] for y in (115.0, 250.0, 420.0)]
    lines.extend([[(x, 95.0), (x, 455.0)] for x in (180.0, 620.0)])
    for line in lines:
        draw.line(_rotate_points(line, angle_degrees, centre), fill=(184, 184, 180), width=2)
    return image


def _short_segment_image(angle_degrees: float) -> Image.Image:
    image = Image.new("RGB", (640, 420), (232, 230, 224))
    draw = ImageDraw.Draw(image)
    centre = (320.0, 210.0)
    segments = [
        [(80.0, 105.0), (190.0, 105.0)],
        [(240.0, 108.0), (350.0, 108.0)],
        [(420.0, 112.0), (535.0, 112.0)],
        [(120.0, 285.0), (250.0, 285.0)],
        [(330.0, 292.0), (470.0, 292.0)],
        [(105.0, 110.0), (105.0, 230.0)],
        [(522.0, 120.0), (522.0, 265.0)],
    ]
    for segment in segments:
        draw.line(_rotate_points(segment, angle_degrees, centre), fill=(92, 92, 88), width=2)
    return image


def _fragmented_horizon_image(angle_degrees: float) -> Image.Image:
    image = Image.new("RGB", (640, 420), (228, 228, 224))
    draw = ImageDraw.Draw(image)
    centre = (320.0, 210.0)
    theta = math.radians(angle_degrees)
    for x in range(80, 560, 44):
        p1 = (float(x), 205.0)
        p2 = (float(x + 22), 205.0)
        draw.line(_rotate_points([p1, p2], angle_degrees, centre), fill=(70, 70, 68), width=2)
    for y in (125.0, 285.0):
        x1 = 320.0 - math.cos(theta) * 220.0
        y1 = y - math.sin(theta) * 220.0
        x2 = 320.0 + math.cos(theta) * 220.0
        y2 = y + math.sin(theta) * 220.0
        draw.line((x1, y1, x2, y2), fill=(150, 150, 146), width=1)
    return image


def _horizon_with_opposing_verticals_image(
    horizon_angle_degrees: float,
    vertical_angle_degrees: float,
) -> Image.Image:
    image = Image.new("RGB", (720, 480), (166, 194, 226))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 235, 720, 480), fill=(92, 130, 86))
    centre = (360.0, 240.0)
    horizon = [(40.0, 230.0), (680.0, 230.0)]
    draw.line(
        _rotate_points(horizon, horizon_angle_degrees, centre),
        fill=(40, 68, 48),
        width=4,
    )

    vertical_centre = (360.0, 240.0)
    for x in (120.0, 170.0, 220.0, 500.0, 555.0, 610.0):
        line = [(x, 70.0), (x, 430.0)]
        draw.line(
            _rotate_points(line, vertical_angle_degrees, vertical_centre),
            fill=(76, 76, 76),
            width=3,
        )
    return image


def _centre_horizon_with_offcentre_distractors_image(
    centre_angle_degrees: float,
    distractor_angle_degrees: float,
) -> Image.Image:
    image = Image.new("RGB", (720, 480), (226, 224, 218))
    draw = ImageDraw.Draw(image)
    centre = (360.0, 240.0)
    central_horizon = [(60.0, 240.0), (660.0, 240.0)]
    draw.line(
        _rotate_points(central_horizon, centre_angle_degrees, centre),
        fill=(45, 45, 44),
        width=5,
    )
    for y in (86.0, 112.0, 372.0, 398.0):
        line = [(80.0, y), (640.0, y)]
        draw.line(
            _rotate_points(line, distractor_angle_degrees, centre),
            fill=(92, 92, 88),
            width=3,
        )
    return image


def _soft_portrait_like_image() -> Image.Image:
    image = Image.new("RGB", (420, 560), (212, 198, 184))
    draw = ImageDraw.Draw(image)
    draw.ellipse((132, 72, 288, 242), fill=(178, 132, 112))
    draw.rounded_rectangle((105, 240, 315, 515), radius=80, fill=(58, 66, 78))
    draw.line((150, 300, 270, 360), fill=(70, 75, 86), width=2)
    draw.line((260, 285, 165, 390), fill=(72, 78, 88), width=2)
    return image


def test_estimate_straighten_angle_detects_small_tilt() -> None:
    result = estimate_straighten_angle(_tilted_line_image(3.0))

    assert result.applied is True
    assert result.reason == "applied"
    assert result.confidence >= 0.18
    assert result.angle_degrees == pytest.approx(3.0, abs=0.5)
    assert result.line_count > 0
    assert result.total_line_length > 0


def test_estimate_straighten_angle_leaves_blank_image_untouched() -> None:
    result = estimate_straighten_angle(Image.new("RGB", (400, 300), "white"))

    assert result.applied is False
    assert result.angle_degrees == 0.0
    assert result.reason == "no_edges"


def test_estimate_straighten_angle_detects_room_geometry() -> None:
    result = estimate_straighten_angle(_tilted_room_image(-2.0))

    assert result.applied is True
    assert result.angle_degrees == pytest.approx(-2.0, abs=0.5)


def test_estimate_straighten_angle_detects_faint_geometry() -> None:
    result = estimate_straighten_angle(_faint_tilted_room_image(2.5))

    assert result.applied is True
    assert result.angle_degrees == pytest.approx(2.5, abs=0.7)


def test_estimate_straighten_angle_detects_short_segments() -> None:
    result = estimate_straighten_angle(_short_segment_image(-2.0))

    assert result.applied is True
    assert result.angle_degrees == pytest.approx(-2.0, abs=0.7)


def test_estimate_straighten_angle_detects_fragmented_horizon() -> None:
    result = estimate_straighten_angle(_fragmented_horizon_image(2.0))

    assert result.applied is True
    assert result.angle_degrees == pytest.approx(2.0, abs=0.8)
    assert result.scene_type in {"horizon", "mixed_axis"}


def test_estimate_straighten_angle_prefers_horizon_over_vertical_distractors() -> None:
    result = estimate_straighten_angle(
        _horizon_with_opposing_verticals_image(
            horizon_angle_degrees=3.0,
            vertical_angle_degrees=-2.0,
        )
    )

    assert result.applied is True
    assert result.scene_type == "horizon"
    assert result.angle_degrees == pytest.approx(3.0, abs=0.7)
    assert result.horizon_score > result.axis_score * 0.8
    assert result.horizontal_line_count > 0
    assert result.vertical_line_count > 0


def test_estimate_straighten_angle_prefers_centre_horizon_over_offcentre_lines() -> None:
    result = estimate_straighten_angle(
        _centre_horizon_with_offcentre_distractors_image(
            centre_angle_degrees=3.0,
            distractor_angle_degrees=-3.0,
        )
    )

    assert result.applied is True
    assert result.scene_type == "horizon"
    assert result.angle_degrees == pytest.approx(3.0, abs=0.7)


def test_estimate_straighten_angle_skips_soft_portrait_without_structure() -> None:
    result = estimate_straighten_angle(_soft_portrait_like_image())

    assert result.applied is False
    assert result.reason in {"angle_too_small", "too_few_edges", "too_few_lines"}


def test_estimate_straighten_angle_skips_random_texture() -> None:
    noise = Image.effect_noise((640, 420), 60).convert("RGB")

    result = estimate_straighten_angle(noise)

    assert result.applied is False


def test_perspective_rotate_attributes_only_for_applied_result() -> None:
    image = _tilted_line_image(2.0)
    applied = estimate_straighten_angle(image)
    skipped = estimate_straighten_angle(Image.new("RGB", (400, 300), "white"))

    attrs = perspective_rotate_attributes(applied, image.size)
    assert attrs["PerspectiveRotate"].startswith("+")
    assert float(attrs["PerspectiveScale"]) > 100.0
    assert attrs["AlreadyApplied"] == "False"
    assert "HasCrop" not in attrs
    assert "CropAngle" not in attrs
    assert "CropTop" not in attrs
    assert perspective_rotate_attributes(skipped, image.size) == {}


def test_perspective_rotate_attributes_avoid_crop_bounds() -> None:
    image = _tilted_room_image(3.0)
    result = estimate_straighten_angle(image)

    attrs = perspective_rotate_attributes(result, image.size)

    assert set(attrs) == {
        "PerspectiveRotate",
        "PerspectiveScale",
        "AlreadyApplied",
    }
    assert float(attrs["PerspectiveRotate"]) == pytest.approx(3.0, abs=0.7)
    assert 100.0 < float(attrs["PerspectiveScale"]) < 115.0


def test_straighten_engine_version_is_recorded_for_diagnostics() -> None:
    assert STRAIGHTEN_ENGINE_VERSION.startswith("opencv-")


def test_rotated_content_scale_is_minimal_above_one() -> None:
    assert rotated_content_scale(0.0, 400, 300) == pytest.approx(1.0)
    assert rotated_content_scale(3.0, 400, 300) > 1.0
