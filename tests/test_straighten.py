from __future__ import annotations

import math

import pytest
from PIL import Image, ImageDraw

from sonna_editor.inference.straighten import (
    crop_angle_attributes,
    estimate_straighten_angle,
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


def test_estimate_straighten_angle_detects_small_tilt() -> None:
    result = estimate_straighten_angle(_tilted_line_image(3.0))

    assert result.applied is True
    assert result.reason == "applied"
    assert result.confidence >= 0.18
    assert result.angle_degrees == pytest.approx(-3.0, abs=0.5)


def test_estimate_straighten_angle_leaves_blank_image_untouched() -> None:
    result = estimate_straighten_angle(Image.new("RGB", (400, 300), "white"))

    assert result.applied is False
    assert result.angle_degrees == 0.0
    assert result.reason == "no_edges"


def test_estimate_straighten_angle_detects_room_geometry() -> None:
    result = estimate_straighten_angle(_tilted_room_image(-2.0))

    assert result.applied is True
    assert result.angle_degrees == pytest.approx(2.0, abs=0.5)


def test_estimate_straighten_angle_detects_faint_geometry() -> None:
    result = estimate_straighten_angle(_faint_tilted_room_image(2.5))

    assert result.applied is True
    assert result.angle_degrees == pytest.approx(-2.5, abs=0.7)


def test_estimate_straighten_angle_detects_short_segments() -> None:
    result = estimate_straighten_angle(_short_segment_image(-2.0))

    assert result.applied is True
    assert result.angle_degrees == pytest.approx(2.0, abs=0.7)


def test_estimate_straighten_angle_skips_random_texture() -> None:
    noise = Image.effect_noise((640, 420), 60).convert("RGB")

    result = estimate_straighten_angle(noise)

    assert result.applied is False


def test_crop_angle_attributes_only_for_applied_result() -> None:
    applied = estimate_straighten_angle(_tilted_line_image(2.0))
    skipped = estimate_straighten_angle(Image.new("RGB", (400, 300), "white"))

    assert crop_angle_attributes(applied)["HasCrop"] == "True"
    assert crop_angle_attributes(applied)["CropAngle"].startswith("-")
    assert crop_angle_attributes(skipped) == {}


def test_rotated_content_scale_is_minimal_above_one() -> None:
    assert rotated_content_scale(0.0, 400, 300) == pytest.approx(1.0)
    assert rotated_content_scale(3.0, 400, 300) > 1.0
