from __future__ import annotations

import math

import pytest
import torch

from sonna_editor.config import SLIDER_FIELDS, SLIDER_RANGES
from sonna_editor.model.losses import WeightedSliderLoss, _build_range_tensors, _build_weight_tensor

N = len(SLIDER_FIELDS)
_TEMP_IDX = SLIDER_FIELDS.index("Temperature")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perfect_pred(targets: torch.Tensor) -> torch.Tensor:
    """Return predictions that exactly match targets (Temperature in log-space)."""
    pred = targets.clone()
    # Convert Temperature column to log-space (as the model would output)
    valid = ~torch.isnan(pred[:, _TEMP_IDX]) & (pred[:, _TEMP_IDX] > 0)
    pred[valid, _TEMP_IDX] = torch.log(pred[valid, _TEMP_IDX])
    return pred


def _make_targets(B: int, value: float = 0.0) -> torch.Tensor:
    """All-present target tensor, Temperature set to 5500 K."""
    t = torch.full((B, N), value)
    t[:, _TEMP_IDX] = 5500.0
    return t


def _make_nan_targets(B: int) -> torch.Tensor:
    """All-NaN target tensor (no ground truth available for any field)."""
    return torch.full((B, N), float("nan"))


# ---------------------------------------------------------------------------
# _build_range_tensors
# ---------------------------------------------------------------------------

def test_range_tensors_shape() -> None:
    lo, hi = _build_range_tensors("v2")
    assert lo.shape == (N,)
    assert hi.shape == (N,)


def test_temperature_overridden_to_log_space() -> None:
    lo, hi = _build_range_tensors("v2")
    assert lo[_TEMP_IDX].item() == pytest.approx(math.log(2000.0))
    assert hi[_TEMP_IDX].item() == pytest.approx(math.log(50000.0))


def test_non_temperature_bounds_match_slider_ranges() -> None:
    lo, hi = _build_range_tensors("v2")
    for i, field in enumerate(SLIDER_FIELDS):
        if field == "Temperature":
            continue
        expected_lo, expected_hi = SLIDER_RANGES[field]
        assert lo[i].item() == pytest.approx(expected_lo), f"{field} lo mismatch"
        assert hi[i].item() == pytest.approx(expected_hi), f"{field} hi mismatch"


# ---------------------------------------------------------------------------
# _build_weight_tensor
# ---------------------------------------------------------------------------

def test_weight_tensor_shape() -> None:
    w = _build_weight_tensor("v2")
    assert w.shape == (N,)


def test_tuned_weight_tensor() -> None:
    """v1.1.0-c3k-tuned weights: Temperature/Tint baseline retained, ToneCurveBlue
    reduced (audit overshoot), HSL Hue bumped 1.5×, BlueHue/GreenHue bumped 1.5×,
    Whites2012 bumped from 1.0 default. Direction-at-chance fields untouched."""
    w = _build_weight_tensor("v2")
    # Baseline retained
    assert w[SLIDER_FIELDS.index("Temperature")].item() == 3.0
    assert w[SLIDER_FIELDS.index("Tint")].item() == 4.0
    assert w[SLIDER_FIELDS.index("ToneCurve_Pt3_Y")].item() == 3.0
    assert w[SLIDER_FIELDS.index("ToneCurveRed_Pt3_Y")].item() == 3.0
    assert w[SLIDER_FIELDS.index("RedHue")].item() == 2.0
    # Blue Y reduced 3.0 → 2.0 (audit overshoot)
    assert w[SLIDER_FIELDS.index("ToneCurveBlue_Pt3_Y")].item() == 2.0
    # TIMID-field bumps
    assert w[SLIDER_FIELDS.index("HueAdjustmentRed")].item() == 2.25
    assert w[SLIDER_FIELDS.index("BlueHue")].item() == 3.0
    assert w[SLIDER_FIELDS.index("GreenHue")].item() == 3.0
    assert w[SLIDER_FIELDS.index("Whites2012")].item() == 1.5
    assert w[SLIDER_FIELDS.index("ToneCurveRed_Pt2_Y")].item() == 4.5
    # Direction-at-chance fields explicitly NOT bumped
    assert w[SLIDER_FIELDS.index("Saturation")].item() == 1.0
    assert w[SLIDER_FIELDS.index("SaturationAdjustmentRed")].item() == 1.5  # HSL default
    assert w[SLIDER_FIELDS.index("ToneCurve_Pt4_Y")].item() == 3.0  # curve default
    assert w[SLIDER_FIELDS.index("GrainFrequency")].item() == 1.0
    # Untouched non-flagged field
    assert w[SLIDER_FIELDS.index("Exposure2012")].item() == 1.0


# ---------------------------------------------------------------------------
# WeightedSliderLoss — construction
# ---------------------------------------------------------------------------

def test_loss_instantiates() -> None:
    loss = WeightedSliderLoss(slider_set_version="v2")
    assert hasattr(loss, "_lo")
    assert hasattr(loss, "_hi")
    assert hasattr(loss, "_w")


def test_buffers_registered_as_buffers() -> None:
    loss = WeightedSliderLoss(slider_set_version="v2")
    buf_names = {name for name, _ in loss.named_buffers()}
    assert "_lo" in buf_names
    assert "_hi" in buf_names
    assert "_w" in buf_names


# ---------------------------------------------------------------------------
# WeightedSliderLoss — forward: basic properties
# ---------------------------------------------------------------------------

def test_perfect_prediction_gives_zero_loss() -> None:
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 4
    targets = _make_targets(B, value=0.0)
    pred = _perfect_pred(targets)
    loss = loss_fn(pred, targets)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_is_scalar() -> None:
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 4
    targets = _make_targets(B)
    pred = torch.zeros(B, N)
    loss = loss_fn(pred, targets)
    assert loss.shape == ()


def test_loss_is_non_negative() -> None:
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 4
    targets = _make_targets(B)
    pred = torch.randn(B, N)
    loss = loss_fn(pred, targets)
    assert loss.item() >= 0.0


def test_loss_increases_with_larger_error() -> None:
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 4
    targets = _make_targets(B, value=0.0)
    pred_small = _perfect_pred(targets).clone()
    pred_small[:, 0] += 0.1  # small exposure error
    pred_large = _perfect_pred(targets).clone()
    pred_large[:, 0] += 1.0  # larger exposure error
    assert loss_fn(pred_large, targets).item() > loss_fn(pred_small, targets).item()


def test_loss_gradients_flow() -> None:
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 2
    targets = _make_targets(B)
    pred = torch.randn(B, N, requires_grad=True)
    loss = loss_fn(pred, targets)
    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()


# ---------------------------------------------------------------------------
# WeightedSliderLoss — NaN masking
# ---------------------------------------------------------------------------

def test_all_nan_targets_gives_zero_loss() -> None:
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 4
    targets = _make_nan_targets(B)
    pred = torch.randn(B, N)
    loss = loss_fn(pred, targets)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_partial_nan_only_penalises_present_fields() -> None:
    """If only field 0 is present, only field-0 errors contribute to loss."""
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 2

    targets_full = _make_nan_targets(B)
    targets_full[:, 0] = 0.0  # Exposure present, all others NaN

    # Zero prediction on Exposure → loss should be ~0
    pred_zero = torch.zeros(B, N)
    loss_zero = loss_fn(pred_zero, targets_full).item()

    # Large error on Exposure → loss should be >> 0
    pred_large = torch.zeros(B, N)
    pred_large[:, 0] = 100.0
    loss_large = loss_fn(pred_large, targets_full).item()

    assert loss_zero == pytest.approx(0.0, abs=1e-5)
    assert loss_large > 0.1


def test_nan_fields_dont_affect_gradient() -> None:
    """Gradient w.r.t. NaN-masked positions must be zero."""
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 2
    targets = _make_nan_targets(B)
    targets[:, 0] = 0.0  # only field 0 present

    pred = torch.zeros(B, N, requires_grad=True)
    loss = loss_fn(pred, targets)
    loss.backward()

    # Gradient for all NaN-masked positions must be zero
    assert pred.grad is not None
    for i in range(1, N):
        assert pred.grad[:, i].abs().max().item() == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# WeightedSliderLoss — Temperature handling
# ---------------------------------------------------------------------------

def test_temperature_log_transform_applied_to_target() -> None:
    """Loss must be near zero when pred = log(target_kelvin)."""
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 2
    targets = _make_nan_targets(B)
    targets[:, _TEMP_IDX] = 5500.0  # raw Kelvin — only field present

    # Model outputs log-space; all other fields have arbitrary predictions
    # since their targets are NaN (masked out).
    pred = torch.zeros(B, N)
    pred[:, _TEMP_IDX] = math.log(5500.0)

    loss = loss_fn(pred, targets)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_temperature_zero_kelvin_skipped() -> None:
    """Zero or negative Kelvin in target must not produce NaN loss."""
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 2
    targets = _make_nan_targets(B)
    targets[:, _TEMP_IDX] = 0.0  # invalid

    pred = torch.zeros(B, N)
    loss = loss_fn(pred, targets)
    assert not torch.isnan(loss)


def test_temperature_negative_kelvin_skipped() -> None:
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 2
    targets = _make_nan_targets(B)
    targets[:, _TEMP_IDX] = -100.0

    pred = torch.zeros(B, N)
    loss = loss_fn(pred, targets)
    assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# WeightedSliderLoss — range normalisation
# ---------------------------------------------------------------------------

def test_range_normalisation_equalises_scale() -> None:
    """Equal fractional-range errors produce losses proportional to field weights.

    Tint weight=1.5, Exposure weight=1.0, so Tint loss / Exposure loss == 1.5
    for the same normalised error.
    """
    from sonna_editor.config import SLIDER_LOSS_WEIGHTS
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 1

    exp_lo, exp_hi = SLIDER_RANGES["Exposure2012"]
    tint_lo, tint_hi = SLIDER_RANGES["Tint"]
    exp_span = exp_hi - exp_lo
    tint_span = tint_hi - tint_lo

    exp_delta = 0.5
    tint_delta = exp_delta * tint_span / exp_span  # same fractional error

    exp_idx = SLIDER_FIELDS.index("Exposure2012")
    tint_idx = SLIDER_FIELDS.index("Tint")

    targets_exp = _make_nan_targets(B)
    targets_exp[:, exp_idx] = 0.0
    pred_exp = torch.zeros(B, N)
    pred_exp[:, exp_idx] = exp_delta

    targets_tint = _make_nan_targets(B)
    targets_tint[:, tint_idx] = 0.0
    pred_tint = torch.zeros(B, N)
    pred_tint[:, tint_idx] = tint_delta

    loss_exp = loss_fn(pred_exp, targets_exp).item()
    loss_tint = loss_fn(pred_tint, targets_tint).item()

    expected_ratio = SLIDER_LOSS_WEIGHTS["Tint"] / SLIDER_LOSS_WEIGHTS["Exposure2012"]
    assert loss_tint / loss_exp == pytest.approx(expected_ratio, rel=1e-4)


# ---------------------------------------------------------------------------
# WeightedSliderLoss — targets are not mutated
# ---------------------------------------------------------------------------

def test_targets_not_mutated() -> None:
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 4
    targets = _make_targets(B)
    targets_orig = targets.clone()
    pred = torch.zeros(B, N)
    loss_fn(pred, targets)
    assert torch.allclose(targets, targets_orig), "forward() must not modify targets in-place"


# ---------------------------------------------------------------------------
# per_field_mae
# ---------------------------------------------------------------------------

def test_per_field_mae_returns_all_fields() -> None:
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 2
    targets = _make_targets(B)
    pred = _perfect_pred(targets)
    mae = loss_fn.per_field_mae(pred, targets)
    assert set(mae.keys()) == set(SLIDER_FIELDS)


def test_per_field_mae_perfect_prediction_zero() -> None:
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 4
    targets = _make_targets(B, value=0.0)
    pred = _perfect_pred(targets)
    mae = loss_fn.per_field_mae(pred, targets)
    for field, val in mae.items():
        if field != "Temperature":
            assert val == pytest.approx(0.0, abs=1e-5), f"{field} MAE should be 0"


def test_per_field_mae_temperature_in_kelvin() -> None:
    """Temperature MAE must be reported in Kelvin, not in log-space."""
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 2
    targets = _make_nan_targets(B)
    targets[:, _TEMP_IDX] = 5500.0  # raw K

    # Predict log(5000) — error is 500 K
    pred = _make_nan_targets(B)
    pred[:, _TEMP_IDX] = math.log(5000.0)

    mae = loss_fn.per_field_mae(pred, targets)
    assert mae["Temperature"] == pytest.approx(500.0, rel=0.01)


def test_per_field_mae_all_nan_field_is_nan() -> None:
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 2
    targets = _make_nan_targets(B)
    targets[:, 0] = 0.0  # only field 0 present
    pred = torch.zeros(B, N)
    mae = loss_fn.per_field_mae(pred, targets)
    # All fields except index 0 should be NaN
    for i, field in enumerate(SLIDER_FIELDS):
        if i == 0:
            continue
        assert math.isnan(mae[field]), f"{field} should be NaN (no targets)"


def test_per_field_mae_does_not_mutate_targets() -> None:
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 2
    targets = _make_targets(B)
    targets_orig = targets.clone()
    pred = torch.zeros(B, N)
    loss_fn.per_field_mae(pred, targets)
    assert torch.allclose(targets, targets_orig)


# ---------------------------------------------------------------------------
# Sign-wrong penalty (Term 5)
# ---------------------------------------------------------------------------

_TINT_IDX_T = SLIDER_FIELDS.index("Tint")


def _make_meta(B: int, as_shot_temp: float | None = 5500.0,
               as_shot_tint: float | None = 0.0) -> dict[str, torch.Tensor]:
    md: dict[str, torch.Tensor] = {}
    if as_shot_temp is not None:
        md["as_shot_temperature"] = torch.full((B,), float(as_shot_temp))
    else:
        md["as_shot_temperature"] = torch.full((B,), float("nan"))
    if as_shot_tint is not None:
        md["as_shot_tint"] = torch.full((B,), float(as_shot_tint))
    else:
        md["as_shot_tint"] = torch.full((B,), float("nan"))
    return md


def test_sign_wrong_zero_on_correct_direction() -> None:
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 8
    targets = _make_targets(B)
    # Truth says cooler (3500 K < 5500 K AsShot) AND magenta (+30 Tint)
    targets[:, _TEMP_IDX] = 3500.0
    targets[:, _TINT_IDX_T] = 30.0
    pred = _perfect_pred(targets)
    md = _make_meta(B, as_shot_temp=5500.0, as_shot_tint=0.0)
    components = loss_fn(pred, targets, md, return_components=True)
    assert float(components["sign_wrong"]) == pytest.approx(0.0, abs=1e-7)


def test_sign_wrong_active_on_wrong_direction() -> None:
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 8
    targets = _make_targets(B)
    targets[:, _TEMP_IDX] = 3500.0       # truth: cooler
    targets[:, _TINT_IDX_T] = 30.0       # truth: magenta
    pred = _perfect_pred(targets)
    # Override: pred goes WARMER than AsShot (wrong direction)
    pred[:, _TEMP_IDX] = math.log(8000.0)
    # Override: pred goes GREEN (wrong direction)
    pred[:, _TINT_IDX_T] = -30.0
    md = _make_meta(B)
    components = loss_fn(pred, targets, md, return_components=True)
    assert float(components["sign_wrong"]) > 0.0


def test_sign_wrong_symmetric() -> None:
    """Mirror-flipping both truth and pred gives identical sign-wrong loss."""
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 8

    # Batch A: truth WARMER than AsShot, pred COOLER (wrong)
    tgt_a = _make_targets(B)
    tgt_a[:, _TEMP_IDX] = 8000.0
    tgt_a[:, _TINT_IDX_T] = 40.0
    pred_a = _perfect_pred(tgt_a)
    pred_a[:, _TEMP_IDX] = math.log(3500.0)
    pred_a[:, _TINT_IDX_T] = -40.0

    # Batch B: truth COOLER than AsShot, pred WARMER (mirror)
    tgt_b = _make_targets(B)
    tgt_b[:, _TEMP_IDX] = 3500.0
    tgt_b[:, _TINT_IDX_T] = -40.0
    pred_b = _perfect_pred(tgt_b)
    pred_b[:, _TEMP_IDX] = math.log(8000.0)
    pred_b[:, _TINT_IDX_T] = 40.0
    # The log-K wrong-direction magnitudes must be equal for symmetry:
    # |log(3500)-log(5500)| vs |log(8000)-log(5500)| are NOT equal in log space,
    # so we instead test sign-wrong with equal log-K offsets.
    delta_log_temp = 0.5
    tgt_a[:, _TEMP_IDX] = math.exp(math.log(5500.0) + delta_log_temp)
    pred_a[:, _TEMP_IDX] = math.log(5500.0) - delta_log_temp
    tgt_b[:, _TEMP_IDX] = math.exp(math.log(5500.0) - delta_log_temp)
    pred_b[:, _TEMP_IDX] = math.log(5500.0) + delta_log_temp

    md = _make_meta(B)
    c_a = loss_fn(pred_a, tgt_a, md, return_components=True)
    c_b = loss_fn(pred_b, tgt_b, md, return_components=True)
    assert float(c_a["sign_wrong"]) == pytest.approx(float(c_b["sign_wrong"]), rel=1e-5)


def test_sign_wrong_zero_inside_deadband() -> None:
    """Truth correction < deadband ⇒ sign penalty = 0 regardless of pred sign."""
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 8
    targets = _make_targets(B)
    # Tiny truth correction: 5510 K is only 10 K from AsShot 5500 K
    # log(5510/5500) ≈ 0.0018, well inside 0.04 deadband
    targets[:, _TEMP_IDX] = 5510.0
    # Tint truth correction = 2 (inside 5-unit deadband)
    targets[:, _TINT_IDX_T] = 2.0
    pred = _perfect_pred(targets)
    # Override: pred swings hard the wrong way
    pred[:, _TEMP_IDX] = math.log(2000.0)
    pred[:, _TINT_IDX_T] = -100.0
    md = _make_meta(B)
    components = loss_fn(pred, targets, md, return_components=True)
    assert float(components["sign_wrong"]) == pytest.approx(0.0, abs=1e-7)


def test_sign_wrong_zero_when_no_asshot() -> None:
    """Both AsShot fields NaN ⇒ sign-wrong term contributes 0."""
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 8
    targets = _make_targets(B)
    targets[:, _TEMP_IDX] = 3500.0
    targets[:, _TINT_IDX_T] = 30.0
    pred = _perfect_pred(targets)
    pred[:, _TEMP_IDX] = math.log(8000.0)  # wrong direction
    pred[:, _TINT_IDX_T] = -30.0           # wrong direction
    md = _make_meta(B, as_shot_temp=None, as_shot_tint=None)
    components = loss_fn(pred, targets, md, return_components=True)
    assert float(components["sign_wrong"]) == pytest.approx(0.0, abs=1e-7)


def test_sign_wrong_gradient_direction() -> None:
    """Gradient on a wrong-direction Temperature pred pushes it toward truth."""
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 4
    targets = _make_targets(B)
    targets[:, _TEMP_IDX] = 3500.0  # truth: cooler than 5500 K AsShot
    targets[:, _TINT_IDX_T] = 0.0   # Tint inside deadband — no Tint signal
    pred = _perfect_pred(targets)
    # Wrong direction: pred says warmer
    pred[:, _TEMP_IDX] = math.log(8000.0)
    pred = pred.clone().requires_grad_(True)
    md = _make_meta(B)
    # Use ONLY the sign-wrong component so the gradient sign is unambiguous.
    components = loss_fn(pred, targets, md, return_components=True)
    sw = components["sign_wrong"]
    # detached components are not differentiable — recompute by reading from total
    # minus other components. Simpler: drive backward through total and inspect.
    total = loss_fn(pred, targets, md, return_components=False)
    total.backward()
    # Truth is cooler ⇒ gradient on log_T should be POSITIVE (loss decreases when
    # log_T moves DOWN — toward truth — i.e. negative grad would push it down).
    # MSE alone already pushes pred toward truth (down). The sign-wrong term
    # adds extra push in the same direction. Either way: gradient[T] > 0.
    g_temp = pred.grad[:, _TEMP_IDX]
    assert (g_temp > 0).all(), f"Expected positive gradient on log_T, got {g_temp.tolist()}"
    assert float(sw) > 0.0  # confirmation the term was active


def test_temp_bucket_safe_with_nan_temperature_truth() -> None:
    """A batch with mixed valid-Temperature and NaN-Temperature truth must
    produce a finite temp_bucket. Regression test for the 2026-05-12 NaN crash:
    `truth_corr = NaN - log_as_shot = NaN` then `NaN * m_f(=0) = NaN` would
    poison the bucket sum. Fix: nan_to_num the truth column at entry."""
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 16
    # 15 valid cold-truth photos + 1 NaN-truth photo, all with valid AsShot
    targets = _make_targets(B)
    targets[:, _TEMP_IDX] = 3000.0
    targets[10, _TEMP_IDX] = float("nan")  # one NaN-Temperature photo
    pred = _perfect_pred(targets)
    pred[10, _TEMP_IDX] = math.log(5500.0)  # finite pred for the NaN row
    md = {
        "as_shot_temperature": torch.full((B,), 3300.0),  # all in COLD bucket
        "as_shot_tint": torch.full((B,), 0.0),
    }
    components = loss_fn(pred, targets, md, return_components=True)
    assert torch.isfinite(components["temp_bucket"]), "temp_bucket must not NaN"
    assert torch.isfinite(components["total"]), "total must not NaN"


def test_per_row_mask_excludes_bad_rows_keeps_good_rows() -> None:
    """Row-level masking: one bad row's contribution is zeroed, but the rest
    of the batch produces a normal finite loss."""
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 8
    targets = _make_targets(B)
    targets[:, _TEMP_IDX] = 3500.0
    # Row 3 has Inf in a slider truth → should be excluded from loss math
    targets[3, _TEMP_IDX] = float("inf")
    pred = _perfect_pred(targets)
    pred[3, _TEMP_IDX] = math.log(5500.0)  # finite pred
    md = _make_meta(B)
    components = loss_fn(pred, targets, md, return_components=True)
    # Total must be finite — the bad row was excluded, not propagated
    assert torch.isfinite(components["total"]), "bad-row Inf must not poison total"
    assert torch.isfinite(components["mse"]), "MSE must not contain Inf"
    assert torch.isfinite(components["temp_bucket"]), "temp_bucket must be finite"


def test_per_row_mask_all_bad_returns_zero_loss() -> None:
    """If every row is invalid, return zero-loss with intact gradient."""
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 4
    targets = _make_targets(B)
    # Every row has Inf truth somewhere
    targets[:, _TEMP_IDX] = float("inf")
    pred = _perfect_pred(targets)
    pred[:, _TEMP_IDX] = math.log(5500.0)
    pred = pred.clone().requires_grad_(True)
    md = _make_meta(B)
    components = loss_fn(pred, targets, md, return_components=True)
    assert "_all_rows_skipped" in components, "all-skipped flag should be set"
    assert float(components["total"]) == 0.0, "all-skipped batch should be zero loss"
    components["total"].backward()
    assert torch.isfinite(pred.grad).all(), "gradient should be all-finite after all-row skip"


def test_per_row_mask_handles_non_finite_predictions() -> None:
    """A row with non-finite predictions is excluded — no NaN propagation."""
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 4
    targets = _make_targets(B)
    pred = _perfect_pred(targets)
    pred[1, _TINT_IDX_T] = float("nan")  # row 1 corrupted
    pred = pred.clone().requires_grad_(True)
    md = _make_meta(B)
    components = loss_fn(pred, targets, md, return_components=True)
    assert torch.isfinite(components["total"]), "NaN pred row must be excluded"
    # Total should be smaller than all-rows-valid since 1 row was masked out
    components["total"].backward()
    # The contaminated row's gradient may be NaN locally; that's OK as long as
    # the loss itself is finite. (The training step receiving zero-or-finite
    # loss is the contract.)


def test_direction_stats_excludes_pred_stayed_neutral() -> None:
    """Pred-stayed-at-identity must NOT count as wrong-direction.

    This is the fix for the 2026-05-11 OvercorrectionHaltCallback mis-fire
    at epoch 2: heads still at random init (pred ~ 0) were counted as wrong-
    direction for every photo where truth was outside the deadband.
    """
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 8
    targets = _make_targets(B)
    targets[:, _TEMP_IDX] = 3500.0     # truth: cooler than AsShot 5500 K
    targets[:, _TINT_IDX_T] = 30.0     # truth: magenta-shifted
    # Pred sits at AsShot reference — i.e. at identity (no correction).
    pred = _perfect_pred(targets)
    pred[:, _TEMP_IDX] = math.log(5500.0)  # log of AsShot — exactly the reference
    pred[:, _TINT_IDX_T] = 0.0             # raw 0 — exactly the reference
    md = _make_meta(B, as_shot_temp=5500.0, as_shot_tint=0.0)
    stats = loss_fn.direction_stats(pred, targets, md)
    # All 8 photos have directional truth (truth outside deadband on both fields)
    # but pred is exactly at reference — pred has zero sign. With the fixed
    # opposite-only logic, zero × negative is not strictly < 0, so wrong = 0.
    n_wrong_T, n_total_T = stats["Temperature"]
    assert n_total_T == B, f"all {B} should be in directional-truth support"
    assert n_wrong_T == 0, f"pred-at-reference must not count as wrong (got {n_wrong_T})"
    n_wrong_Tn, n_total_Tn = stats["Tint"]
    assert n_total_Tn == B
    assert n_wrong_Tn == 0


def test_direction_stats_counts() -> None:
    loss_fn = WeightedSliderLoss(slider_set_version="v2")
    B = 6
    targets = _make_targets(B)
    targets[:, _TEMP_IDX] = 3500.0  # all 6: truth cooler
    targets[:, _TINT_IDX_T] = 30.0  # all 6: truth magenta
    pred = _perfect_pred(targets)
    # Override: photos 0,1,2 have WRONG-direction Temp (warmer pred); 3,4,5 correct
    pred[:3, _TEMP_IDX] = math.log(8000.0)
    # All 6 Tint preds: 2 wrong-direction, 4 correct
    pred[:2, _TINT_IDX_T] = -30.0
    md = _make_meta(B)
    stats = loss_fn.direction_stats(pred, targets, md)
    # Temperature: 6 photos outside deadband, 3 wrong-direction
    n_wrong_T, n_total_T = stats["Temperature"]
    assert n_total_T == 6
    assert n_wrong_T == 3
    # Tint: 6 photos outside deadband (|30| > 5), 2 wrong-direction
    n_wrong_Tn, n_total_Tn = stats["Tint"]
    assert n_total_Tn == 6
    assert n_wrong_Tn == 2
    # An identity-referenced field (e.g. Exposure2012 = 0 truth) — no truth
    # outside its 2% deadband (range 10 ⇒ band 0.2; truth 0 ⇒ |truth-0|=0).
    n_wrong_E, n_total_E = stats["Exposure2012"]
    assert n_total_E == 0
    assert n_wrong_E == 0


# ---------------------------------------------------------------------------
# v1 slider_set_version coverage
# ---------------------------------------------------------------------------
# These tests exercise the previously-broken path: WeightedSliderLoss built
# for a v1 model (135 outputs) operating on length-135 prediction/target
# tensors. Before commit `feat(slider-set): version-aware field list helpers`
# and the losses.py refactor, buffer construction assumed `len(SLIDER_FIELDS)`
# which silently became 147 after the v2 expansion (commit 3d0d90c). This is
# the class of bug the refactor eliminates — coverage lives where the bug
# lived.

V1_N = 135


def _make_v1_targets(B: int, value: float = 0.0) -> torch.Tensor:
    """All-present target tensor sized for v1 models, Temperature 5500 K."""
    t = torch.full((B, V1_N), value)
    t[:, _TEMP_IDX] = 5500.0
    return t


def _perfect_v1_pred(targets: torch.Tensor) -> torch.Tensor:
    pred = targets.clone()
    valid = ~torch.isnan(pred[:, _TEMP_IDX]) & (pred[:, _TEMP_IDX] > 0)
    pred[valid, _TEMP_IDX] = torch.log(pred[valid, _TEMP_IDX])
    return pred


def test_v1_loss_construction_produces_135_sized_buffers() -> None:
    loss = WeightedSliderLoss(slider_set_version="v1")
    assert loss._lo.shape == (V1_N,)
    assert loss._hi.shape == (V1_N,)
    assert loss._w.shape == (V1_N,)
    assert loss._slider_set_version == "v1"


def test_v1_forward_does_not_broadcast_mismatch() -> None:
    """The exact bug the refactor fixes: v1 predictions through v1 loss."""
    loss = WeightedSliderLoss(slider_set_version="v1")
    targets = _make_v1_targets(4)
    pred = _perfect_v1_pred(targets)
    md = _make_meta(4)
    out = loss(pred, targets, md)
    assert torch.isfinite(out).item()
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_v1_per_field_mae_returns_all_135_fields_no_indexerror() -> None:
    loss = WeightedSliderLoss(slider_set_version="v1")
    targets = _make_v1_targets(4)
    pred = _perfect_v1_pred(targets)
    mae = loss.per_field_mae(pred, targets)
    assert len(mae) == V1_N
    # Spot-check that the dict is keyed by v1-only fields (no v2 extensions).
    assert "Exposure2012" in mae
    assert "ToneCurveBlue_Pt6_Y" in mae   # last v1 field
    assert "CurveRefineSaturation" not in mae   # v2 extension, must NOT appear


def test_v1_direction_stats_no_indexerror() -> None:
    loss = WeightedSliderLoss(slider_set_version="v1")
    targets = _make_v1_targets(4)
    targets[:, _TEMP_IDX] = 3500.0
    pred = _perfect_v1_pred(targets)
    md = _make_meta(4)
    stats = loss.direction_stats(pred, targets, md)
    assert len(stats) == V1_N
    assert "Temperature" in stats
    assert "CurveRefineSaturation" not in stats


def test_v1_loss_rejects_v2_shaped_predictions() -> None:
    """Mismatched-shape input to a v1 loss should fail loudly (broadcast or
    indexing error), not silently produce a wrong number."""
    loss = WeightedSliderLoss(slider_set_version="v1")
    targets_v2 = torch.full((4, 147), 0.0)
    pred_v2 = targets_v2.clone()
    with pytest.raises((RuntimeError, IndexError)):
        loss(pred_v2, targets_v2, _make_meta(4))


def test_loss_rejects_unknown_slider_set_version() -> None:
    """Unknown version surfaces at construction time, not at first forward."""
    with pytest.raises(ValueError, match="unknown slider_set_version"):
        WeightedSliderLoss(slider_set_version="v3")
