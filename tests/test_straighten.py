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


def test_crop_angle_attributes_only_for_applied_result() -> None:
    applied = estimate_straighten_angle(_tilted_line_image(2.0))
    skipped = estimate_straighten_angle(Image.new("RGB", (400, 300), "white"))

    assert crop_angle_attributes(applied)["HasCrop"] == "True"
    assert crop_angle_attributes(applied)["CropAngle"].startswith("-")
    assert crop_angle_attributes(skipped) == {}


def test_rotated_content_scale_is_minimal_above_one() -> None:
    assert rotated_content_scale(0.0, 400, 300) == pytest.approx(1.0)
    assert rotated_content_scale(3.0, 400, 300) > 1.0
