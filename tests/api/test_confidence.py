"""Tests for api.confidence.scalar_confidence."""

from __future__ import annotations

import pytest
import torch

from sonna_editor import config
from sonna_editor.api.confidence import scalar_confidence


def test_all_zero_std_returns_one() -> None:
    std = torch.zeros(len(config.SLIDER_FIELDS))
    assert scalar_confidence(std) == 1.0


def test_large_std_clamps_to_zero() -> None:
    # Saturate every key slider's normaliser by a factor of 10.
    std = torch.zeros(len(config.SLIDER_FIELDS))
    for slider in config.KEY_CONFIDENCE_SLIDERS:
        idx = config.SLIDER_FIELDS.index(slider)
        std[idx] = config.CONFIDENCE_NORM_STDS[slider] * 10
    result = scalar_confidence(std)
    assert result == 0.0


def test_intermediate_std_is_in_unit_interval() -> None:
    std = torch.zeros(len(config.SLIDER_FIELDS))
    for slider in config.KEY_CONFIDENCE_SLIDERS:
        idx = config.SLIDER_FIELDS.index(slider)
        std[idx] = config.CONFIDENCE_NORM_STDS[slider] * 0.5
    result = scalar_confidence(std)
    assert 0.0 < result < 1.0
    assert result == pytest.approx(0.5, abs=1e-6)


def test_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError):
        scalar_confidence(torch.zeros(2, len(config.SLIDER_FIELDS)))
