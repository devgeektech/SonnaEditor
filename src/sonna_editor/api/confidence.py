"""Per-slider MC-dropout std → single scalar confidence in [0, 1].

The /api/process endpoint maps `flag_low_confidence: true` to
`uncertainty=True` on the inference pipeline, which produces a per-slider
std for every photo. The UI wants ONE number it can compare against a
threshold slider — this module computes it.
"""

from __future__ import annotations

import torch

from sonna_editor import config


def scalar_confidence(per_slider_std: torch.Tensor) -> float:
    """Reduce per-slider uncertainty to a single confidence score.

    Formula:
        normed_i = min(1.0, std_i / CONFIDENCE_NORM_STDS[slider_i])
        confidence = clamp(1.0 - mean(normed_i over KEY_CONFIDENCE_SLIDERS), 0, 1)

    Each slider's std is normalised by the matching value in
    CONFIDENCE_NORM_STDS — uncertainty equal to the training-set spread
    contributes 1.0 (saturating to "no confidence"). Reducing across only
    the 8 Basic-panel sliders keeps the score interpretable: noise on a
    rarely-edited slider shouldn't dominate the photographer-visible signal.

    Args:
        per_slider_std: Tensor of shape [N_SLIDERS] (135 in v1) — the std
                        slice for one photo from
                        ``InferenceEngine.predict_with_uncertainty``.

    Returns:
        A confidence in [0, 1]. 1.0 = maximally confident (all-zero std);
        0.0 = uncertainty across the key sliders saturates the normalisation.

    Raises:
        IndexError: if the tensor is shorter than the largest required
                    slider index — guards against shape mismatch.
    """
    if per_slider_std.ndim != 1:
        raise ValueError(
            f"per_slider_std must be 1-D; got shape {tuple(per_slider_std.shape)}"
        )

    normed: list[float] = []
    for slider in config.KEY_CONFIDENCE_SLIDERS:
        idx = config.SLIDER_FIELDS.index(slider)
        norm = config.CONFIDENCE_NORM_STDS[slider]
        std_val = float(per_slider_std[idx].item())
        normed.append(min(1.0, max(0.0, std_val) / norm))

    raw = 1.0 - (sum(normed) / len(normed))
    return max(0.0, min(1.0, raw))
