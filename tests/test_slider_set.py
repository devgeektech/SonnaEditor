"""Unit tests for sonna_editor.slider_set."""
from __future__ import annotations

import pytest
import torch

from sonna_editor import config
from sonna_editor.slider_set import (
    V1_OUTPUT_COUNT,
    V2_OUTPUT_COUNT,
    fields_for_version,
    fields_matching_tensor,
    v1_fields,
)


# ---------------------------------------------------------------------------
# v1_fields
# ---------------------------------------------------------------------------

def test_v1_fields_returns_135_entries() -> None:
    assert len(v1_fields()) == V1_OUTPUT_COUNT == 135


def test_v1_fields_starts_with_exposure() -> None:
    # Exposure2012 is locked at index 0 (HANDOVER Decision 6).
    assert v1_fields()[0] == "Exposure2012"


def test_v1_fields_ends_at_last_v1_entry() -> None:
    # Locked-append-only: idx 134 is the last v1 field, never re-indexed.
    # Tone curve fields fill indices 87-134; the final v1 entry is
    # ToneCurveBlue_Pt6_Y.
    fields = v1_fields()
    assert fields[-1] == "ToneCurveBlue_Pt6_Y"


def test_v1_fields_matches_slider_fields_prefix() -> None:
    assert v1_fields() == list(config.SLIDER_FIELDS[:V1_OUTPUT_COUNT])


def test_v1_fields_returns_a_new_list_each_call() -> None:
    # Returning a copy prevents accidental mutation of the canonical list.
    a, b = v1_fields(), v1_fields()
    assert a == b
    a.append("MUTATED")
    assert b[-1] != "MUTATED"


# ---------------------------------------------------------------------------
# fields_for_version
# ---------------------------------------------------------------------------

def test_fields_for_version_v1_returns_135() -> None:
    fields = fields_for_version("v1")
    assert len(fields) == V1_OUTPUT_COUNT
    assert fields == v1_fields()


def test_fields_for_version_v2_returns_147() -> None:
    fields = fields_for_version("v2")
    assert len(fields) == V2_OUTPUT_COUNT == 147
    assert fields == list(config.SLIDER_FIELDS[:V2_OUTPUT_COUNT])


def test_fields_for_version_v2_extends_v1() -> None:
    # Locked-append-only: v2 == v1 + 12 new fields at the end.
    v1 = fields_for_version("v1")
    v2 = fields_for_version("v2")
    assert v2[: len(v1)] == v1
    assert len(v2) - len(v1) == 12


def test_fields_for_version_rejects_unknown_version() -> None:
    with pytest.raises(ValueError, match="unknown slider_set_version"):
        fields_for_version("v3")


def test_fields_for_version_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="unknown slider_set_version"):
        fields_for_version("")


def test_fields_for_version_is_case_sensitive() -> None:
    # SonnaEditor uses lowercase "v1"/"v2"; mixed case is a bug.
    with pytest.raises(ValueError, match="unknown slider_set_version"):
        fields_for_version("V1")


# ---------------------------------------------------------------------------
# fields_matching_tensor
# ---------------------------------------------------------------------------

def test_fields_matching_tensor_v1_shape_2d() -> None:
    t = torch.zeros(4, 135)
    fields = fields_matching_tensor(t)
    assert len(fields) == V1_OUTPUT_COUNT
    assert fields == v1_fields()


def test_fields_matching_tensor_v2_shape_2d() -> None:
    t = torch.zeros(4, 147)
    fields = fields_matching_tensor(t)
    assert len(fields) == V2_OUTPUT_COUNT


def test_fields_matching_tensor_v1_shape_1d() -> None:
    # Single-photo slice (after batch_idx) is 1D.
    t = torch.zeros(135)
    fields = fields_matching_tensor(t)
    assert len(fields) == V1_OUTPUT_COUNT


def test_fields_matching_tensor_v1_shape_3d() -> None:
    # MC-dropout outputs are [n_samples, N, 135].
    t = torch.zeros(10, 4, 135)
    fields = fields_matching_tensor(t)
    assert len(fields) == V1_OUTPUT_COUNT


def test_fields_matching_tensor_rejects_zero_dim() -> None:
    t = torch.tensor(3.14)
    with pytest.raises(ValueError, match="no dimensions"):
        fields_matching_tensor(t)


def test_fields_matching_tensor_rejects_unknown_count() -> None:
    t = torch.zeros(4, 130)  # neither v1 nor v2
    with pytest.raises(ValueError, match="not a supported"):
        fields_matching_tensor(t)


def test_fields_matching_tensor_rejects_oversize_count() -> None:
    t = torch.zeros(4, 200)
    with pytest.raises(ValueError, match="not a supported"):
        fields_matching_tensor(t)


def test_fields_matching_tensor_rejects_zero_length_last_dim() -> None:
    t = torch.zeros(4, 0)
    with pytest.raises(ValueError, match="not a supported"):
        fields_matching_tensor(t)
