"""Version-aware accessors for the slider-field list.

Every site that pairs ``config.SLIDER_FIELDS`` with a model output tensor must
go through one of these helpers — direct iteration over ``SLIDER_FIELDS``
in code that handles model output is the bug pattern that caused two
production breakages on 2026-05-14 after the v2 expansion (commit
``3d0d90c``) grew ``SLIDER_FIELDS`` from 135 → 147 while v1 models still
output length-135 tensors.

Three accessors, picked by what the caller already knows:

- ``v1_fields()`` — the locked 135-field v1 slice. Use when the caller is
  explicitly targeting v1 (e.g. Mode B preset-to-checkpoint converter,
  v1-specific audits).
- ``fields_for_version(slider_set_version)`` — the right slice for the
  loaded model's version. Use when a model or training context exposes
  ``slider_set_version`` directly (loss objects, fine-tune pipelines).
- ``fields_matching_tensor(tensor)`` — the slice matching the tensor's
  last dim. Use when a function receives a prediction tensor without
  separate version info (postprocess, predictions_to_dict, generic
  diagnostics).

Locked-append-only (HANDOVER Decision 6) guarantees the first 135
entries of ``SLIDER_FIELDS`` are the v1 fields, in v1 order, forever.
Slicing is therefore semantically correct.
"""
from __future__ import annotations

from typing import Final

import torch

from sonna_editor import config


V1_OUTPUT_COUNT: Final[int] = 135
V2_OUTPUT_COUNT: Final[int] = 147

_SUPPORTED_LENGTHS: Final[dict[int, str]] = {
    V1_OUTPUT_COUNT: "v1",
    V2_OUTPUT_COUNT: "v2",
}
_SUPPORTED_VERSIONS: Final[dict[str, int]] = {v: k for k, v in _SUPPORTED_LENGTHS.items()}


# Sanity at import: the v1/v2 counts must still match the first 135 / full 147
# of SLIDER_FIELDS. If a future SLIDER_FIELDS edit ever changes the indexing,
# fail at import rather than letting a wrong slice ship.
assert len(config.SLIDER_FIELDS) >= V2_OUTPUT_COUNT, (
    f"SLIDER_FIELDS is length {len(config.SLIDER_FIELDS)}, expected at least "
    f"{V2_OUTPUT_COUNT}. Did the slider list shrink?"
)


def v1_fields() -> list[str]:
    """Return the 135-field v1 slice (idx 0-134)."""
    return list(config.SLIDER_FIELDS[:V1_OUTPUT_COUNT])


def fields_for_version(slider_set_version: str) -> list[str]:
    """Return the field list for a specific slider_set_version.

    Raises:
        ValueError: if ``slider_set_version`` is not one of "v1", "v2".
    """
    if slider_set_version not in _SUPPORTED_VERSIONS:
        raise ValueError(
            f"unknown slider_set_version={slider_set_version!r}; "
            f"expected one of {sorted(_SUPPORTED_VERSIONS)}"
        )
    n = _SUPPORTED_VERSIONS[slider_set_version]
    return list(config.SLIDER_FIELDS[:n])


def fields_matching_tensor(tensor: torch.Tensor) -> list[str]:
    """Return the field list that matches the tensor's last dim.

    Accepts any tensor whose last dim is a supported model output count
    (135 for v1, 147 for v2). Use when a function receives a prediction
    tensor and needs the corresponding field names without separate
    version info.

    Raises:
        ValueError: if the tensor is empty or has zero dimensions, or
            if its last dim is not a supported output count.
    """
    if tensor.ndim == 0:
        raise ValueError(
            "fields_matching_tensor: tensor has no dimensions; cannot infer "
            "slider set"
        )
    n = tensor.shape[-1]
    if n not in _SUPPORTED_LENGTHS:
        raise ValueError(
            f"fields_matching_tensor: tensor last dim {n} is not a supported "
            f"slider count. Supported: {sorted(_SUPPORTED_LENGTHS)}. If "
            f"SLIDER_FIELDS grew (e.g. v3), update slider_set.py to "
            f"recognise the new length."
        )
    return list(config.SLIDER_FIELDS[:n])
