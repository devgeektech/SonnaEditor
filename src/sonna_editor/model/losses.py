from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
from typing import Literal, Mapping, Optional, overload

import torch
import torch.nn as nn

from sonna_editor.config import (
    EXPOSURE_SCENE_LOSS_WEIGHT,
    SIGN_WRONG_PENALTY_WEIGHT,
    SLIDER_FIELDS,
    SLIDER_LOSS_WEIGHTS,
    SLIDER_RANGES,
    SPREAD_LOSS_WEIGHT,
    TEMPERATURE_BUCKET_LOSS_WEIGHT,
    TINT_BUCKET_LOSS_WEIGHT,
)
from sonna_editor.slider_set import fields_for_version

# Temperature and Tint are at the same indices (11, 12) in v1 and v2 — locked
# by HANDOVER Decision 6 (append-only). Module-level constants are therefore
# safe across both slider_set_versions.
_TEMPERATURE_IDX: int = SLIDER_FIELDS.index("Temperature")
_TINT_IDX: int = SLIDER_FIELDS.index("Tint")
_EXPOSURE_IDX: int = SLIDER_FIELDS.index("Exposure2012")

# ---------------------------------------------------------------------------
# Per-row skipped-row logging
# ---------------------------------------------------------------------------
# WeightedSliderLoss.forward() masks individual rows out of loss math when
# their inputs are unsafe (Inf in any slider truth, AsShot non-positive/Inf,
# AsShot Tint Inf). The bad rows' raw_paths are appended to this file so
# we can spot patterns across epochs.
_SKIPPED_LOG_PATH = Path(tempfile.gettempdir()) / "saha_skipped_rows.log"
_skipped_log_inited = False


MetadataValue = torch.Tensor | list[str] | str


def _log_skipped(reasons: dict[str, int], invalid_rows: list[int],
                 raw_paths: Optional[list[str]]) -> None:
    """Append a line to the OS temp skipped-row log when rows are masked out.

    Format:
      HH:MM:SS  n_invalid=N/B  reasons={...}  rows=[(idx, path), ...]
    """
    global _skipped_log_inited
    if not invalid_rows:
        return
    line = (f"{time.strftime('%H:%M:%S')}  "
            f"n_invalid={len(invalid_rows)}  reasons={reasons}")
    if raw_paths is not None:
        pairs = [(i, raw_paths[i] if i < len(raw_paths) else "?")
                 for i in invalid_rows[:10]]
        line += f"  rows={pairs}"
    if len(invalid_rows) > 10:
        line += f"  + {len(invalid_rows) - 10} more"
    try:
        with _SKIPPED_LOG_PATH.open("a") as f:
            if not _skipped_log_inited:
                f.write(f"# saha skipped-rows log — started "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                _skipped_log_inited = True
            f.write(line + "\n")
    except OSError:
        # Logging is best-effort; never crash training because of disk issues.
        pass

# Sign-wrong deadbands — match audit thresholds. Below these, direction is
# ambiguous (truth_corr near zero), so no sign penalty applies.
_SIGN_WRONG_EPS_LOG_TEMP: float = 0.04   # ≈ ±4% Kelvin shift (~220 K at 5500 K)
_SIGN_WRONG_EPS_TINT: float = 5.0        # raw Tint units
_TINT_RANGE_NORM: float = 300.0          # Tint full range hi - lo


def _build_range_tensors(slider_set_version: str) -> tuple[torch.Tensor, torch.Tensor]:
    fields = fields_for_version(slider_set_version)
    lo = torch.tensor([SLIDER_RANGES[f][0] for f in fields], dtype=torch.float32)
    hi = torch.tensor([SLIDER_RANGES[f][1] for f in fields], dtype=torch.float32)
    # Temperature: model predicts log(Kelvin); override bounds to log-space
    lo[_TEMPERATURE_IDX] = math.log(2000.0)
    hi[_TEMPERATURE_IDX] = math.log(50000.0)
    return lo, hi


def _build_weight_tensor(slider_set_version: str) -> torch.Tensor:
    fields = fields_for_version(slider_set_version)
    return torch.tensor(
        [SLIDER_LOSS_WEIGHTS[f] for f in fields], dtype=torch.float32
    )


class WeightedSliderLoss(nn.Module):
    """Range-normalised weighted MSE across all Lightroom slider predictions.

    Slider count is set by ``slider_set_version`` at construction time:
        - "v1" → 135 outputs (matches v1.2.3 shipping checkpoint)
        - "v2" → 147 outputs (current default architecture)
    Predictions and targets passed to ``forward()`` MUST match the configured
    version's output count or broadcast will fail loudly.

    Each slider is normalised to [0, 1] before MSE using its full Lightroom valid
    range, so a 0.5-stop Exposure error contributes the same as a 25-unit Shadows
    error. All weights default to 1.0 (Neutral Learner — no style opinions).

    Temperature handling:
        - Dataset targets are raw Kelvin (e.g. 5500.0).
        - Model predictions are log(Kelvin).
        - The loss log-transforms the temperature target internally before normalising
          with log-space bounds [log(2000), log(50000)].

    Absent targets (None in Parquet → NaN in tensor):
        - Masked out of both numerator and denominator so the model is not penalised
          for predicting any value where ground truth is absent.
        - If an entire batch has NaN for a field, that field contributes 0 to the loss.
    """

    def __init__(self, slider_set_version: str) -> None:
        """slider_set_version is required ("v1" or "v2"). Buffers are sized to
        match the loaded model's output count — pass the model's
        ``_slider_set_version`` from the training module. A default value is
        intentionally NOT provided: defaulting to one version silently
        produces broadcast-mismatch crashes when paired with the other, which
        is the exact bug class this refactor eliminates.
        """
        super().__init__()
        # fields_for_version raises on unknown values; surface the bad input
        # at construction rather than at first forward().
        fields_for_version(slider_set_version)
        self._slider_set_version = slider_set_version
        lo, hi = _build_range_tensors(slider_set_version)
        weights = _build_weight_tensor(slider_set_version)
        # Register as buffers so they move with .to(device) automatically.
        # Length is 135 (v1) or 147 (v2) depending on slider_set_version.
        self.register_buffer("_lo", lo)
        self.register_buffer("_hi", hi)
        self.register_buffer("_w", weights)

    @overload
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[Mapping[str, MetadataValue]] = None,
        *,
        return_components: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[Mapping[str, MetadataValue]] = None,
        *,
        return_components: Literal[True],
    ) -> dict[str, torch.Tensor]: ...

    def forward(
        self,
        predictions: torch.Tensor,                  # [B, 135] — Temperature in log(K) space
        targets: torch.Tensor,                      # [B, 135] — Temperature in raw Kelvin; NaN = absent
        metadata: Optional[Mapping[str, MetadataValue]] = None,
        *,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Returns scalar total loss, OR (if return_components=True) a dict with
        keys: total, mse, exposure_scene, spread, temp_bucket, tint_bucket, sign_wrong. All
        scalars on the same device as predictions. Components with no valid
        data return 0.

        NaN-safety: invalid rows are masked out of all loss components rather
        than dropping the entire batch. A row is invalid (excluded from loss
        math) if any of:
          - Inf in any of its slider truths (overflow risk in MSE)
          - as_shot_temperature is Inf or ≤0 (used by temp_bucket / sign_wrong)
          - as_shot_tint is Inf (used by tint_bucket / sign_wrong)
        Rows with NaN slider truth are NOT invalid — per-field NaN masking
        already handles those cleanly. Same for NaN AsShot (component-internal
        masks handle that).

        If the model's `predictions` tensor contains non-finite values
        (suggesting earlier weight corruption), all rows are excluded and a
        zero loss with intact gradient is returned (no further NaN
        propagation).

        Bad-row raw_paths are appended to the OS temp skipped-row log when
        present in `metadata["raw_path"]`.

        metadata is an optional dict produced by SonnaDataset.__getitem__. Keys
        the loss reads:
          as_shot_temperature: Tensor[B] raw Kelvin, NaN if unresolvable.
          as_shot_tint: Tensor[B] raw Tint units, NaN if unresolvable.
          raw_path: optional list[str] for skipped-row logging.
        Tint bucketing uses ground-truth Tint from `targets`, so no metadata
        entry is needed for that term.
        """
        B = predictions.size(0)
        device = predictions.device

        # ── Per-row validity mask ──
        # Default everyone valid; subtract rows that would cause NaN propagation.
        row_valid = torch.ones(B, dtype=torch.bool, device=device)
        reasons: dict[str, int] = {}

        # Predictions: if the model produced non-finite outputs for some rows,
        # exclude those rows entirely (a weight corruption may have happened
        # earlier; mask the bad rows so we don't propagate further NaN).
        pred_bad = (~torch.isfinite(predictions)).any(dim=-1)
        if pred_bad.any():
            row_valid &= ~pred_bad
            reasons["non_finite_predictions"] = int(pred_bad.sum().item())

        # Truth Inf — would overflow MSE. NaN truth is fine (per-field masked).
        # Note: torch.isinf(NaN) = False, so this only flags Inf, not NaN.
        inf_truth = torch.isinf(targets).any(dim=-1)
        if inf_truth.any():
            row_valid &= ~inf_truth
            reasons["inf_in_slider_truth"] = int(inf_truth.sum().item())

        if metadata is not None:
            ast = metadata.get("as_shot_temperature")
            if isinstance(ast, torch.Tensor):
                # Treat ≤0 or Inf as bad. NaN AsShot is still valid (the
                # temp_bucket component already gates on as_shot_valid).
                ast_bad = (torch.isinf(ast) | (ast <= 0)) & ~torch.isnan(ast)
                if ast_bad.any():
                    row_valid &= ~ast_bad
                    reasons["as_shot_temperature_inf_or_nonpositive"] = int(ast_bad.sum().item())
            atn = metadata.get("as_shot_tint")
            if isinstance(atn, torch.Tensor):
                atn_bad = torch.isinf(atn)
                if atn_bad.any():
                    row_valid &= ~atn_bad
                    reasons["as_shot_tint_inf"] = int(atn_bad.sum().item())

        # If ANY row was invalidated, log it.
        if reasons:
            bad_idx = (~row_valid).nonzero(as_tuple=True)[0].tolist()
            raw_paths = metadata.get("raw_path") if metadata else None
            if isinstance(raw_paths, str):
                # Single-row batch (rare) — wrap to list.
                raw_paths = [raw_paths]
            elif not isinstance(raw_paths, list) or not all(isinstance(path, str) for path in raw_paths):
                raw_paths = None
            _log_skipped(reasons, bad_idx, raw_paths)

        # Replace non-finite predictions with zero for downstream arithmetic.
        # The row_valid mask already excludes those rows from loss contribution,
        # but the intermediate `(pred - tgt)²` would still go NaN→NaN×0=NaN
        # without this substitution. Same defence as the Inf→NaN strip on tgt.
        if (~torch.isfinite(predictions)).any():
            predictions = predictions.masked_fill(
                ~torch.isfinite(predictions), 0.0
            )

        # If literally every row is invalid (extreme case), return a zero loss
        # connected to predictions for a zero gradient.
        if not row_valid.any():
            zero = (predictions.nan_to_num(0.0) * 0).sum()
            if return_components:
                return {
                    "total":       zero,
                    "mse":         zero.detach(),
                    "exposure_scene": zero.detach(),
                    "spread":      zero.detach(),
                    "temp_bucket": zero.detach(),
                    "tint_bucket": zero.detach(),
                    "sign_wrong":  zero.detach(),
                    "_all_rows_skipped": torch.tensor(1.0, device=device),
                }
            return zero

        # ── Term 1: existing range-normalised weighted MSE ──
        # Log-transform raw Kelvin targets to match model's prediction space.
        # Clone temp_col to break view aliasing before modifying tgt in-place.
        tgt = targets.clone()
        # Strip Inf BEFORE downstream nan_to_num — the default nan_to_num
        # replaces +Inf with fp32's max (~3.4e38), which then overflows to
        # NaN when squared in MSE. By converting Inf→NaN here, the existing
        # NaN handling (mask-out per field) cleanly excludes those values.
        # Combined with row_valid above, Inf-truth rows contribute 0 to MSE.
        inf_mask = torch.isinf(tgt)
        if inf_mask.any():
            tgt = tgt.masked_fill(inf_mask, float("nan"))
        temp_col = tgt[:, _TEMPERATURE_IDX].clone()
        valid_temp = ~torch.isnan(temp_col) & (temp_col > 0)
        invalid_temp = ~torch.isnan(temp_col) & ~valid_temp
        tgt[valid_temp, _TEMPERATURE_IDX] = torch.log(temp_col[valid_temp])
        # Non-NaN but invalid (≤0 K) temperatures are treated as absent
        tgt[invalid_temp, _TEMPERATURE_IDX] = float("nan")

        # Mask: True where target is present (not NaN) AND the row passed the
        # validity check above. Invalid rows contribute 0 to every component.
        mask = ~torch.isnan(tgt) & row_valid.unsqueeze(-1)   # [B, 135]

        # Normalise both pred and target to [0, 1] using slider ranges.
        # Fill NaN in tgt with 0 before computing tgt_norm so the computation
        # graph is NaN-free; masked positions are zeroed out by mask below.
        lo = self._lo.unsqueeze(0)   # [1, 135]
        hi = self._hi.unsqueeze(0)   # [1, 135]
        span = hi - lo               # [1, 135]

        pred_norm = (predictions - lo) / span                 # [B, 135]
        tgt_norm  = (tgt.nan_to_num(nan=0.0) - lo) / span    # [B, 135]

        # Weighted squared error, zeroed out where target is absent.
        w = self._w.unsqueeze(0)     # [1, 135]
        sq_err = w * (pred_norm - tgt_norm) ** 2
        sq_err = sq_err * mask
        n_valid = mask.sum().clamp(min=1)
        mse = sq_err.sum() / n_valid

        # ── Term 2: spread penalty (one-sided hinge, averaged over fields) ──
        spread = self._spread_term(pred_norm, tgt_norm, mask)

        # ── Term 2b: focused Exposure2012 penalty for low-light collapse ──
        # The all-slider MSE averages over 135/147 fields, so even a bumped
        # Exposure2012 field weight can be diluted. This term keeps exposure
        # directly visible to the optimiser and gives low-luminance/high-lift
        # rows more influence without changing their target slider values.
        scene_stats = None
        if metadata is not None:
            value = metadata.get("scene_stats")
            scene_stats = value if isinstance(value, torch.Tensor) else None
        exposure_scene = self._exposure_scene_term(
            pred_exposure=predictions[:, _EXPOSURE_IDX],
            truth_exposure=targets[:, _EXPOSURE_IDX],
            tgt_mask=mask[:, _EXPOSURE_IDX],
            scene_stats=scene_stats,
        )

        # ── Term 3: per-bucket Temperature penalty (log-space) ──
        # Uses predictions+targets in raw model-output Kelvin units (predictions
        # at idx_temp is log-K; targets in the SAME `tgt` tensor we already
        # log-transformed above, so we can read it directly).
        as_shot_temp = None
        if metadata is not None:
            value = metadata.get("as_shot_temperature")
            as_shot_temp = value if isinstance(value, torch.Tensor) else None
        temp_bucket = self._temperature_bucket_term(
            pred_log_temp=predictions[:, _TEMPERATURE_IDX],
            tgt_log_temp=tgt[:, _TEMPERATURE_IDX],
            tgt_mask=mask[:, _TEMPERATURE_IDX],
            as_shot_temp=as_shot_temp,
        )

        # ── Term 4: per-bucket Tint penalty (raw units, ground-truth bucketed) ──
        # Tint is not log-transformed — use targets directly (not tgt, which has
        # the Temperature column rewritten but Tint untouched). For clarity we
        # read from targets here, masked by mask[:, Tint].
        tint_bucket = self._tint_bucket_term(
            pred_tint=predictions[:, _TINT_IDX],
            truth_tint=targets[:, _TINT_IDX],
            tgt_mask=mask[:, _TINT_IDX],
        )

        # ── Term 5: symmetric sign-wrong penalty for Temperature + Tint ──
        # AsShot-referenced direction-mismatch penalty. Hinge² on the
        # range-normalised pred correction; non-zero only when pred has
        # opposite sign from truth (relative to AsShot). Mathematically
        # symmetric across cold↔warm and green↔magenta so no directional
        # bias is introduced.
        as_shot_tint = None
        if metadata is not None:
            value = metadata.get("as_shot_tint")
            as_shot_tint = value if isinstance(value, torch.Tensor) else None
        sign_wrong = self._sign_wrong_term(
            pred_log_temp=predictions[:, _TEMPERATURE_IDX],
            tgt_log_temp=tgt[:, _TEMPERATURE_IDX],
            temp_mask=mask[:, _TEMPERATURE_IDX],
            as_shot_temp=as_shot_temp,
            pred_tint=predictions[:, _TINT_IDX],
            truth_tint=targets[:, _TINT_IDX],
            tint_mask=mask[:, _TINT_IDX],
            as_shot_tint=as_shot_tint,
        )

        total = (
            mse
            + EXPOSURE_SCENE_LOSS_WEIGHT    * exposure_scene
            + SPREAD_LOSS_WEIGHT             * spread
            + TEMPERATURE_BUCKET_LOSS_WEIGHT * temp_bucket
            + TINT_BUCKET_LOSS_WEIGHT        * tint_bucket
            + SIGN_WRONG_PENALTY_WEIGHT      * sign_wrong
        )

        if return_components:
            return {
                "total":       total,
                "mse":         mse.detach(),
                "exposure_scene": exposure_scene.detach(),
                "spread":      spread.detach(),
                "temp_bucket": temp_bucket.detach(),
                "tint_bucket": tint_bucket.detach(),
                "sign_wrong":  sign_wrong.detach(),
            }
        return total

    # ------------------------------------------------------------------
    # Term implementations
    # ------------------------------------------------------------------

    def _spread_term(
        self,
        pred_norm: torch.Tensor,   # [B, 135] in [0, 1] normalised space
        tgt_norm:  torch.Tensor,   # [B, 135] normalised (NaN already nan_to_num'd to 0)
        mask:      torch.Tensor,   # [B, 135] bool
    ) -> torch.Tensor:
        """Penalise per-field std_pred < std_truth (already-normalised inputs).

        Computes batch-level std per field with a masked-mean approach. One-sided
        hinge so wandering fields aren't dragged further. Averages over the set
        of fields with ≥2 valid samples in this batch — empty fields contribute
        nothing.

        Returns a scalar. Range normalisation already happened (inputs are in
        [0,1] per-field), so no extra range division here. Weights `self._w` are
        applied per-field.
        """
        m_f = mask.float()                                      # [B, 135]
        valid_per_field = m_f.sum(dim=0)                       # [135]
        has_data = (valid_per_field >= 2).float()              # [135]
        n_eff = valid_per_field.clamp(min=1.0)

        # Masked per-field mean (target uses pre-nan-to-num'd tgt_norm — masked
        # entries are 0, so we use mask-weighted mean to get true mean over valid).
        pred_mean = (pred_norm * m_f).sum(dim=0) / n_eff       # [135]
        tgt_mean  = (tgt_norm  * m_f).sum(dim=0) / n_eff       # [135]

        pred_var = ((pred_norm - pred_mean.unsqueeze(0)) ** 2 * m_f).sum(dim=0) / n_eff
        tgt_var  = ((tgt_norm  - tgt_mean.unsqueeze(0))  ** 2 * m_f).sum(dim=0) / n_eff

        # Add small epsilon before sqrt to avoid d/dx sqrt(0) = inf, which
        # would produce NaN gradients through 0 * inf even on no-data fields
        # that the has_data mask later zeroes out.
        _eps = 1e-12
        pred_std = (pred_var.clamp(min=0) + _eps).sqrt()       # [135]
        tgt_std  = (tgt_var.clamp(min=0)  + _eps).sqrt()

        gap = torch.clamp(tgt_std - pred_std, min=0.0)         # one-sided hinge
        per_field = self._w * gap ** 2                          # [135]
        per_field = per_field * has_data

        denom = has_data.sum().clamp(min=1.0)
        return per_field.sum() / denom

    def _exposure_scene_term(
        self,
        pred_exposure: torch.Tensor,   # [B] raw Exposure2012 stops
        truth_exposure: torch.Tensor,  # [B] raw Exposure2012 stops
        tgt_mask: torch.Tensor,        # [B] bool — True where truth is valid
        scene_stats: Optional[torch.Tensor],  # [B, 6], mean luminance at col 0
    ) -> torch.Tensor:
        """Direct Exposure2012 loss with dark-scene emphasis.

        The regular all-slider MSE is intentionally broad, but exposure mistakes
        dominate visual acceptance. This term uses the same range-normalised
        units as the main loss, with a multiplier for dark previews and large
        positive exposure targets. The multiplier is normalised back to mean 1
        over the valid batch, so it redistributes emphasis rather than simply
        inflating loss based on batch composition.
        """
        zero = pred_exposure.new_zeros(())
        if scene_stats is None:
            return zero
        valid = tgt_mask & ~torch.isnan(truth_exposure)
        if valid.sum() == 0:
            return zero

        exposure_range = (self._hi[_EXPOSURE_IDX] - self._lo[_EXPOSURE_IDX]).clamp(min=1e-6)
        err = ((pred_exposure - truth_exposure.nan_to_num(0.0)) / exposure_range) ** 2

        weights = torch.ones_like(err)
        if scene_stats.ndim != 2 or scene_stats.shape[1] == 0:
            return zero
        mean_luminance = scene_stats[:, 0].float()
        finite_lum = torch.isfinite(mean_luminance)
        dark_strength = ((0.35 - mean_luminance.nan_to_num(0.35)) / 0.35).clamp(0.0, 1.0)
        weights = weights + torch.where(finite_lum, 2.0 * dark_strength, torch.zeros_like(weights))

        lift_strength = ((truth_exposure.nan_to_num(0.0) - 0.50) / 1.0).clamp(0.0, 1.0)
        weights = weights + 1.5 * lift_strength
        weights = weights / weights[valid].mean().clamp(min=1e-6)

        per_photo = (err * weights * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
        bucket_loss = zero
        n_buckets = 0
        finite_lum = torch.isfinite(mean_luminance)
        buckets = (
            valid & finite_lum & (mean_luminance < 0.25),
            valid & finite_lum & (mean_luminance >= 0.25) & (mean_luminance < 0.45),
            valid & finite_lum & (mean_luminance >= 0.45),
        )
        for bucket in buckets:
            if bucket.sum() < 2:
                continue
            mean_gap = ((pred_exposure - truth_exposure.nan_to_num(0.0)) / exposure_range)[bucket].mean()
            bucket_loss = bucket_loss + mean_gap.square()
            n_buckets += 1
        if n_buckets == 0:
            return per_photo
        return per_photo + bucket_loss / float(n_buckets)

    def _temperature_bucket_term(
        self,
        pred_log_temp: torch.Tensor,   # [B] log(K)
        tgt_log_temp:  torch.Tensor,   # [B] log(K) for valid, NaN for absent
        tgt_mask:      torch.Tensor,   # [B] bool, True where target is valid
        as_shot_temp:  Optional[torch.Tensor],  # [B] raw K or NaN
    ) -> torch.Tensor:
        """Bucket photos by AsShot Kelvin and penalise mean(pred_corr - truth_corr).

        Returns 0 if as_shot_temp is None, or no bucket has ≥2 valid samples.
        """
        zero = pred_log_temp.new_zeros(())
        if as_shot_temp is None:
            return zero
        as_shot_valid = ~torch.isnan(as_shot_temp) & (as_shot_temp > 0)
        base_mask = tgt_mask & as_shot_valid                    # [B]
        if base_mask.sum() < 2:
            return zero

        # AsShot in raw K, predictions/targets in log-K. Compute log AsShot.
        # nan_to_num so the log doesn't NaN-propagate; masked entries are dropped.
        log_as_shot = torch.log(as_shot_temp.nan_to_num(1.0).clamp(min=1.0))
        pred_corr  = pred_log_temp - log_as_shot                # [B]
        # nan_to_num the truth before the subtraction — for rows with NaN truth
        # (Temperature absent), this substitutes 0 which is then masked out by
        # m_f below. Without it, NaN × 0 = NaN poisons the per-bucket sum and
        # the whole loss becomes NaN — caused the 2026-05-12 batch-389 crash on
        # the Bvlgari cold-shoot batches where some rows have Temperature truth
        # of NaN. (`torch.sign(NaN) = 0` happens to save `_sign_wrong_term` from
        # the same problem, but the bucket term does no sign() call.)
        truth_corr = tgt_log_temp.nan_to_num(0.0) - log_as_shot

        cold_m    = base_mask & (as_shot_temp < 4500.0)
        neutral_m = base_mask & (as_shot_temp >= 4500.0) & (as_shot_temp <= 6500.0)
        warm_m    = base_mask & (as_shot_temp > 6500.0)

        loss = zero
        n_buckets = 0
        for m in (cold_m, neutral_m, warm_m):
            n = m.sum()
            if n < 2:
                continue
            m_f = m.float()
            denom = m_f.sum()
            pm = (pred_corr * m_f).sum() / denom
            tm = (truth_corr * m_f).sum() / denom
            loss = loss + (pm - tm) ** 2
            n_buckets += 1
        if n_buckets == 0:
            return zero
        return loss / float(n_buckets)

    def _tint_bucket_term(
        self,
        pred_tint:  torch.Tensor,   # [B] raw Tint units
        truth_tint: torch.Tensor,   # [B] raw Tint units (NaN allowed)
        tgt_mask:   torch.Tensor,   # [B] bool — True where truth_tint is valid
    ) -> torch.Tensor:
        """Bucket by |truth_tint| (low/mid/high), penalise mean gap.

        Range-normalised by Tint span (300). Returns 0 if no bucket has ≥2
        valid samples.
        """
        zero = pred_tint.new_zeros(())
        if tgt_mask.sum() < 2:
            return zero
        # Replace NaN with 0 to keep the computation graph NaN-free; masked
        # entries are dropped below.
        truth = truth_tint.nan_to_num(0.0)
        abs_truth = truth.abs()
        low_m  = tgt_mask & (abs_truth < 5.0)
        mid_m  = tgt_mask & (abs_truth >= 5.0) & (abs_truth < 15.0)
        high_m = tgt_mask & (abs_truth >= 15.0)
        loss = zero
        n_buckets = 0
        range_ = 300.0
        for m in (low_m, mid_m, high_m):
            n = m.sum()
            if n < 2:
                continue
            m_f = m.float()
            denom = m_f.sum()
            pm = (pred_tint * m_f).sum() / denom
            tm = (truth     * m_f).sum() / denom
            loss = loss + ((pm - tm) / range_) ** 2
            n_buckets += 1
        if n_buckets == 0:
            return zero
        return loss / float(n_buckets)

    def _sign_wrong_term(
        self,
        pred_log_temp: torch.Tensor,   # [B] log(K)
        tgt_log_temp:  torch.Tensor,   # [B] log(K) (NaN already set where target absent)
        temp_mask:     torch.Tensor,   # [B] bool
        as_shot_temp:  Optional[torch.Tensor],  # [B] raw K or NaN
        pred_tint:     torch.Tensor,   # [B] raw Tint units
        truth_tint:    torch.Tensor,   # [B] raw Tint units (NaN allowed)
        tint_mask:     torch.Tensor,   # [B] bool — True where truth_tint valid
        as_shot_tint:  Optional[torch.Tensor],  # [B] raw Tint or NaN
    ) -> torch.Tensor:
        """Symmetric hinge² on direction mismatch, averaged over active fields.

        For each field f ∈ {Temperature, Tint}:
            δt = truth - reference,  δp = pred - reference
            s  = sign(δt) outside deadband
            h  = relu(-s * δp / range_norm_f)
            field_loss(f) = mean(h² * outside_deadband_mask) over valid photos
        Returns the unweighted average across active fields (≥2 supporting photos).
        Zero when neither field has support — caller's coefficient still applied
        but contributes nothing.
        """
        zero = pred_log_temp.new_zeros(())

        # ---- Temperature field ----
        temp_loss = zero
        n_active = 0
        if as_shot_temp is not None:
            as_temp_valid = ~torch.isnan(as_shot_temp) & (as_shot_temp > 0)
            base_temp = temp_mask & as_temp_valid
            if base_temp.sum() >= 2:
                log_as_temp = torch.log(as_shot_temp.nan_to_num(1.0).clamp(min=1.0))
                # Range = log(50000/2000) ≈ 3.21888, read from _hi/_lo to stay in sync.
                temp_range = (self._hi[_TEMPERATURE_IDX] - self._lo[_TEMPERATURE_IDX]).clamp(min=1e-6)
                truth_corr = tgt_log_temp  - log_as_temp
                pred_corr  = pred_log_temp - log_as_temp
                outside = truth_corr.abs() > _SIGN_WRONG_EPS_LOG_TEMP
                eff_mask = base_temp & outside
                if eff_mask.sum() >= 2:
                    # sign(truth_corr) is zero at exact zero, but outside-deadband
                    # guarantees |truth_corr| > 0, so sign ∈ {-1, +1}.
                    s = torch.sign(truth_corr)
                    q = pred_corr / temp_range
                    hinge = torch.relu(-s * q)
                    per_photo = hinge * hinge
                    mf = eff_mask.float()
                    denom = mf.sum().clamp(min=1.0)
                    temp_loss = (per_photo * mf).sum() / denom
                    n_active += 1

        # ---- Tint field ----
        tint_loss = zero
        if as_shot_tint is not None:
            as_tint_valid = ~torch.isnan(as_shot_tint)
            base_tint = tint_mask & as_tint_valid
            if base_tint.sum() >= 2:
                ref_tint = as_shot_tint.nan_to_num(0.0)
                truth_corr = truth_tint.nan_to_num(0.0) - ref_tint
                pred_corr  = pred_tint                 - ref_tint
                outside = truth_corr.abs() > _SIGN_WRONG_EPS_TINT
                eff_mask = base_tint & outside
                if eff_mask.sum() >= 2:
                    s = torch.sign(truth_corr)
                    q = pred_corr / _TINT_RANGE_NORM
                    hinge = torch.relu(-s * q)
                    per_photo = hinge * hinge
                    mf = eff_mask.float()
                    denom = mf.sum().clamp(min=1.0)
                    tint_loss = (per_photo * mf).sum() / denom
                    n_active += 1

        if n_active == 0:
            return zero
        return (temp_loss + tint_loss) / float(n_active)

    @torch.no_grad()
    def direction_stats(
        self,
        predictions: torch.Tensor,                  # [B, 135] (Temperature is log-K)
        targets:     torch.Tensor,                  # [B, 135] (Temperature in raw K)
        metadata:    Optional[Mapping[str, MetadataValue]] = None,
    ) -> dict[str, tuple[int, int]]:
        """Per-field (n_wrong, n_total) direction-mismatch counts on this batch.

        Reference per field:
            Temperature: log(AsShot Kelvin)
            Tint:        AsShot Tint
            else:        identity (0.0) — model's "no change" prediction

        Deadband: ±2% of the slider's range from the reference, except
        Temperature (0.04 log-K) and Tint (5 units) which match the
        AsShot-bucketing thresholds.

        A photo contributes to a field only when:
            - truth is valid (not NaN)
            - reference is valid (AsShot present, where applicable)
            - |truth - reference| > deadband (otherwise direction is ambiguous)

        n_wrong counts photos where sign(pred - reference) ≠ sign(truth - reference)
        AND |pred - reference| is also outside the (slightly tighter) pred deadband.
        Predictions inside the pred deadband (under-correction) are NOT counted —
        only firm wrong-side predictions (active overcorrection) trip this metric.
        """
        B = predictions.shape[0]
        device = predictions.device

        # AsShot tensors (may be missing if metadata is None)
        as_shot_temp = None
        as_shot_tint = None
        if metadata is not None:
            temp_value = metadata.get("as_shot_temperature")
            tint_value = metadata.get("as_shot_tint")
            as_shot_temp = temp_value if isinstance(temp_value, torch.Tensor) else None
            as_shot_tint = tint_value if isinstance(tint_value, torch.Tensor) else None

        # Range tensor (already on self._lo / self._hi)
        lo = self._lo
        hi = self._hi

        # Truth deadband: 2% of range. Truth inside this band is direction-ambiguous.
        deadband = 0.02 * (hi - lo)
        deadband_list = deadband.tolist()
        deadband_list[_TEMPERATURE_IDX] = _SIGN_WRONG_EPS_LOG_TEMP
        deadband_list[_TINT_IDX] = _SIGN_WRONG_EPS_TINT

        # Pred deadband: 1.875% of range (= 3.75 for HSL). Smallest tested
        # threshold that suppresses under-correction false positives for
        # LuminanceAdjustmentPurple across e4/e6/e7 v1.2.0-smoketest ckpts
        # (4.0% / 21.0% / 10.5% wrong-direction at db_pred=3.75 — all under
        # the 25% callback threshold; db_pred=3.5 rebounds e6 to 36.3%).
        # Temperature and Tint reuse the same epsilon for both deadbands.
        pred_deadband = 0.01875 * (hi - lo)
        pred_deadband_list = pred_deadband.tolist()
        pred_deadband_list[_TEMPERATURE_IDX] = _SIGN_WRONG_EPS_LOG_TEMP
        pred_deadband_list[_TINT_IDX] = _SIGN_WRONG_EPS_TINT

        # Convert truth Temperature column from raw K → log-K to match predictions.
        # NaN propagates; we mask later.
        tgt_temp_raw = targets[:, _TEMPERATURE_IDX]
        valid_temp = ~torch.isnan(tgt_temp_raw) & (tgt_temp_raw > 0)
        tgt = targets.clone()
        tgt[valid_temp, _TEMPERATURE_IDX] = torch.log(tgt_temp_raw[valid_temp])
        tgt[~valid_temp, _TEMPERATURE_IDX] = float("nan")

        # Per-field reference column, sized to this loss's slider_set_version
        # (135 for v1, 147 for v2). Matches the predictions/targets last dim.
        fields = fields_for_version(self._slider_set_version)
        ref = torch.zeros(B, len(fields), device=device)
        if as_shot_temp is not None:
            log_as_temp = torch.where(
                ~torch.isnan(as_shot_temp) & (as_shot_temp > 0),
                torch.log(as_shot_temp.nan_to_num(1.0).clamp(min=1.0)),
                torch.full_like(as_shot_temp, float("nan")),
            )
            ref[:, _TEMPERATURE_IDX] = log_as_temp
        else:
            ref[:, _TEMPERATURE_IDX] = float("nan")
        if as_shot_tint is not None:
            ref[:, _TINT_IDX] = torch.where(
                ~torch.isnan(as_shot_tint),
                as_shot_tint,
                torch.full_like(as_shot_tint, float("nan")),
            )
        else:
            ref[:, _TINT_IDX] = float("nan")

        truth_corr = tgt - ref          # [B, 135]
        pred_corr  = predictions - ref  # [B, 135]

        result: dict[str, tuple[int, int]] = {}
        for i, field in enumerate(fields):
            db = deadband_list[i]
            db_pred = pred_deadband_list[i]
            ref_col = ref[:, i]
            tgt_col = tgt[:, i]
            # AsShot-referenced fields need a non-NaN reference; identity-referenced
            # fields have ref=0 which is always valid.
            ref_valid = ~torch.isnan(ref_col)
            tgt_valid = ~torch.isnan(tgt_col)
            tc = truth_corr[:, i]
            pc = pred_corr[:, i]
            outside_truth = tc.abs() > db
            # Only photos with truth firmly outside the deadband are judgable.
            base = ref_valid & tgt_valid & outside_truth
            n_total = int(base.sum().item())
            if n_total == 0:
                result[field] = (0, 0)
                continue
            # Wrong direction = pred FIRMLY on the OPPOSITE side of truth.
            # Pred inside its deadband (under-correction near identity) is NOT
            # counted; only predictions firmly outside the band on the opposite
            # side count as active overcorrection.
            opposite = (tc * pc) < 0.0
            pred_firm = pc.abs() > db_pred
            wrong = base & opposite & pred_firm
            n_wrong = int(wrong.sum().item())
            result[field] = (n_wrong, n_total)
        return result

    @torch.no_grad()
    def per_field_mae(
        self,
        predictions: torch.Tensor,   # [B, 135] v1 / [B, 147] v2
        targets: torch.Tensor,        # [B, 135] v1 / [B, 147] v2
    ) -> dict[str, float]:
        """Compute mean absolute error per field in original slider units.

        Temperature is reported in Kelvin (pred is exp-ed back before comparison).
        Fields with all-NaN targets in the batch are reported as NaN.
        Returns one entry per field for this loss's slider_set_version.
        """
        tgt = targets.clone()

        # Convert Temperature prediction from log-space back to Kelvin for reporting
        pred = predictions.clone()
        pred[:, _TEMPERATURE_IDX] = torch.exp(pred[:, _TEMPERATURE_IDX])

        mask = ~torch.isnan(tgt)
        result: dict[str, float] = {}
        fields = fields_for_version(self._slider_set_version)
        for i, field in enumerate(fields):
            m = mask[:, i]
            if not m.any():
                result[field] = float("nan")
                continue
            abs_err = (pred[m, i] - tgt[m, i]).abs()
            result[field] = float(abs_err.mean())

        return result
    _lo: torch.Tensor
    _hi: torch.Tensor
    _w: torch.Tensor
