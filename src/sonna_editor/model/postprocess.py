from __future__ import annotations

import torch

from sonna_editor import config
from sonna_editor.slider_set import fields_matching_tensor

_TEMPERATURE_IDX: int = config.SLIDER_FIELDS.index("Temperature")

# Range tensors built once at import time; moved to device on first use via .to()
_RANGE_LO = torch.tensor(
    [config.SLIDER_RANGES[f][0] for f in config.SLIDER_FIELDS], dtype=torch.float32
)
_RANGE_HI = torch.tensor(
    [config.SLIDER_RANGES[f][1] for f in config.SLIDER_FIELDS], dtype=torch.float32
)


def postprocess_predictions(predictions: torch.Tensor) -> torch.Tensor:
    """
    Convert model output (prediction space) to Lightroom-valid slider values.

    Transformations applied:
    - Temperature (index 11): model outputs log(Kelvin) → exp() to get Kelvin.
    - All values: clamped to valid Lightroom slider ranges.

    Slider-set agnostic: the range tensors are sliced to match the prediction
    tensor's last dim, so v1 models ([B, 135]) and v2 models ([B, 147]) both
    work. Locked-append-only (HANDOVER Decision 6) guarantees the first 135
    range entries are the v1 ranges, so slicing the 147-length range tensor
    to [..., :n_out] is correct for any n_out in {135, 147}.

    Args:
        predictions: Tensor[B, N] from SonnaEditor.forward() where N is 135
            for v1 models or 147 for v2 models.

    Returns:
        Tensor[B, N] in Lightroom units, clamped to valid ranges.
    """
    out = predictions.clone()

    out[:, _TEMPERATURE_IDX] = torch.exp(out[:, _TEMPERATURE_IDX])

    n_out = out.shape[-1]
    if n_out > _RANGE_LO.shape[0]:
        raise ValueError(
            f"postprocess_predictions: predictions has {n_out} fields but "
            f"config.SLIDER_FIELDS only defines {_RANGE_LO.shape[0]} ranges. "
            f"Either the slider list shrank or the model is from a future "
            f"slider_set_version not yet in config."
        )
    lo = _RANGE_LO[:n_out].to(out.device).unsqueeze(0)  # [1, n_out]
    hi = _RANGE_HI[:n_out].to(out.device).unsqueeze(0)  # [1, n_out]
    out = torch.max(torch.min(out, hi), lo)

    return out


def predictions_to_dict(predictions: torch.Tensor, batch_idx: int = 0) -> dict[str, float]:
    """Extract one prediction row as a slider dict ready for xmp.write_xmp().

    Field names are inferred from the prediction tensor's last dim via
    ``fields_matching_tensor`` — works for v1 (135) and v2 (147) outputs
    without the caller passing version info.
    """
    row = predictions[batch_idx]
    fields = fields_matching_tensor(predictions)
    return {field: float(row[i]) for i, field in enumerate(fields)}
