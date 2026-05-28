"""Custom PyTorch Lightning callbacks for training monitoring and alerting.

All callbacks print a clearly marked 🚨 ALERT message when triggered.
Non-fatal alerts (overfitting, ETA) print and continue; fatal alerts
(NaN loss, low disk) set trainer.should_stop = True.
"""
from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

import pytorch_lightning as pl

_ALERT = "\n🚨 ALERT"


class NaNLossCallback(pl.Callback):
    """Stop training if NaN or Inf training loss occurs for N consecutive batches."""

    def __init__(self, consecutive_threshold: int = 2) -> None:
        self._nan_count = 0
        self._threshold = consecutive_threshold

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        loss_val = trainer.callback_metrics.get("train_loss_step")
        if loss_val is None:
            return
        loss_float = float(loss_val)
        if math.isnan(loss_float) or math.isinf(loss_float):
            self._nan_count += 1
            if self._nan_count >= self._threshold:
                print(
                    f"{_ALERT}: NaN/Inf training loss for {self._nan_count} "
                    f"consecutive batches (batch {batch_idx}). Training halted.\n"
                    "  Check: learning rate, data normalisation, gradient clipping."
                )
                trainer.should_stop = True
        else:
            self._nan_count = 0


class OverfittingCallback(pl.Callback):
    """Alert (non-stopping) if val loss increases for N consecutive epochs.

    Early stopping will eventually stop training. This callback fires sooner
    so the user can inspect while training is still running.
    """

    def __init__(self, patience: int = 5) -> None:
        self._val_losses: list[float] = []
        self._patience = patience
        self._alerted = False

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return
        self._val_losses.append(float(val_loss))

        if len(self._val_losses) < self._patience:
            return

        recent = self._val_losses[-self._patience:]
        monotone_up = all(recent[i] >= recent[i - 1] for i in range(1, len(recent)))

        if monotone_up and not self._alerted:
            self._alerted = True
            print(
                f"{_ALERT}: Val loss increased for {self._patience} consecutive "
                f"epochs — possible overfitting.\n"
                f"  Recent losses: {[f'{v:.4f}' for v in recent]}\n"
                "  Early stopping will trigger if this continues."
            )
        elif not monotone_up:
            self._alerted = False  # reset once loss improves


class DiskSpaceCallback(pl.Callback):
    """Stop training if free disk space on watch_path drops below min_free_gb."""

    def __init__(self, watch_path: Path, min_free_gb: float = 5.0) -> None:
        self._path = Path(watch_path)
        self._min_bytes = min_free_gb * (1024 ** 3)
        self._min_free_gb = min_free_gb

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        try:
            usage = shutil.disk_usage(self._path)
        except OSError:
            print(
                f"{_ALERT}: Cannot read disk usage at {self._path}.\n"
                "  SSD may have been unmounted mid-training. Training halted."
            )
            trainer.should_stop = True
            return

        if usage.free < self._min_bytes:
            free_gb = usage.free / (1024 ** 3)
            print(
                f"{_ALERT}: Only {free_gb:.1f} GB free at {self._path} "
                f"(minimum {self._min_free_gb:.0f} GB). Training halted."
            )
            trainer.should_stop = True


class ETACallback(pl.Callback):
    """Alert if estimated remaining training time exceeds max_hours after a warm-up period."""

    def __init__(self, max_hours: float = 4.0, check_after_epoch: int = 3) -> None:
        self._epoch_times: list[float] = []
        self._max_hours = max_hours
        self._check_after = check_after_epoch
        self._epoch_start: float = 0.0
        self._alerted = False

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self._epoch_start = time.monotonic()

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        elapsed = time.monotonic() - self._epoch_start
        self._epoch_times.append(elapsed)

        epoch = trainer.current_epoch
        if epoch < self._check_after:
            return

        avg_epoch_sec = sum(self._epoch_times) / len(self._epoch_times)
        remaining_epochs = (trainer.max_epochs or 0) - epoch - 1
        if remaining_epochs <= 0:
            return

        eta_hours = (avg_epoch_sec * remaining_epochs) / 3600

        if eta_hours > self._max_hours and not self._alerted:
            self._alerted = True
            print(
                f"{_ALERT}: Estimated {eta_hours:.1f} h remaining "
                f"(>{self._max_hours:.0f} h threshold, after epoch {epoch + 1}).\n"
                f"  Avg epoch: {avg_epoch_sec / 60:.1f} min  |  "
                f"{remaining_epochs} epochs left.\n"
                "  Consider reducing --max-epochs or resuming from checkpoint later."
            )


class LossComponentBalanceCallback(pl.Callback):
    """Halt training if a non-MSE component dominates MSE by > ratio_threshold.

    The four-term Stage 2 loss (MSE + spread + temp_bucket + tint_bucket) is
    sensitive to coefficient mistuning. If a single non-MSE component is many
    times larger than MSE, gradient signal is effectively dominated by that
    one term. This callback fires (with trainer.should_stop = True) when any
    coefficient-weighted non-MSE component exceeds ratio_threshold × MSE.

    Anchored on MSE (rather than max/min) because near-zero components are
    informative — they just mean that term has nothing to penalise. They
    shouldn't trigger a "dominance" alert.

    Defaults to checking from epoch 1 onwards because the Temperature head
    starts at ~0 (random init → log-K ~0, way off the ~8.5 mean), which makes
    epoch-0 temp_bucket transiently huge. By epoch 1 the head has had ~10⁴
    gradient steps and the term should have settled.
    """

    _COMPONENT_KEYS = (
        ("train_loss_spread",      "spread",      "SPREAD_LOSS_WEIGHT"),
        ("train_loss_temp_bucket", "temp_bucket", "TEMPERATURE_BUCKET_LOSS_WEIGHT"),
        ("train_loss_tint_bucket", "tint_bucket", "TINT_BUCKET_LOSS_WEIGHT"),
        ("train_loss_sign_wrong",  "sign_wrong",  "SIGN_WRONG_PENALTY_WEIGHT"),
    )

    def __init__(
        self,
        ratio_threshold: float = 5.0,
        check_after_epoch: int = 1,
    ) -> None:
        self._threshold = ratio_threshold
        self._check_after = check_after_epoch
        self._fired = False

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self._fired or trainer.current_epoch < self._check_after:
            return

        mse_t = trainer.callback_metrics.get("train_loss_mse")
        if mse_t is None:
            return
        mse = float(mse_t)
        if mse <= 1e-9:
            return  # MSE essentially zero (model fully fit, edge case); skip

        from sonna_editor.config import (
            SIGN_WRONG_PENALTY_WEIGHT,
            SPREAD_LOSS_WEIGHT,
            TEMPERATURE_BUCKET_LOSS_WEIGHT,
            TINT_BUCKET_LOSS_WEIGHT,
        )
        coeffs = {
            "SPREAD_LOSS_WEIGHT":             SPREAD_LOSS_WEIGHT,
            "TEMPERATURE_BUCKET_LOSS_WEIGHT": TEMPERATURE_BUCKET_LOSS_WEIGHT,
            "TINT_BUCKET_LOSS_WEIGHT":        TINT_BUCKET_LOSS_WEIGHT,
            "SIGN_WRONG_PENALTY_WEIGHT":      SIGN_WRONG_PENALTY_WEIGHT,
        }

        worst_label: str | None = None
        worst_ratio = 0.0
        snapshot: dict[str, float] = {"mse": mse}
        for key, label, coef_name in self._COMPONENT_KEYS:
            v = trainer.callback_metrics.get(key)
            if v is None:
                return  # not all components reported yet; wait
            raw = float(v)
            weighted = raw * coeffs[coef_name]
            snapshot[label] = weighted
            ratio = weighted / mse
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_label = label

        if worst_ratio > self._threshold and worst_label is not None:
            self._fired = True
            comp_str = "  ".join(f"{k}={v:.2e}" for k, v in snapshot.items())
            print(
                f"{_ALERT}: Loss component imbalance after epoch {trainer.current_epoch + 1}.\n"
                f"  Components (× coefficient): {comp_str}\n"
                f"  {worst_label} is {worst_ratio:.1f}× MSE (threshold {self._threshold:.1f}×).\n"
                "  Rebalance SPREAD_LOSS_WEIGHT / TEMPERATURE_BUCKET_LOSS_WEIGHT / "
                "TINT_BUCKET_LOSS_WEIGHT in config.py before resuming."
            )
            trainer.should_stop = True


class CriticalMAECallback(pl.Callback):
    """Alert if per-field MAE for critical sliders is implausibly high after warm-up.

    Thresholds are generous — meant to catch broken training (wrong normalisation,
    target encoding errors), not normal training variance.
    """

    # metric_key → (field name for display, implausible-above threshold)
    _THRESHOLDS: dict[str, tuple[str, float]] = {
        "val_mae_exposure":    ("Exposure2012",    2.0),    # 2+ stops
        "val_mae_temperature": ("Temperature",     5000.0), # 5000+ K
        "val_mae_shadows":     ("Shadows2012",     80.0),
        "val_mae_highlights":  ("Highlights2012",  80.0),
    }

    def __init__(self, check_after_epoch: int = 5) -> None:
        self._check_after = check_after_epoch

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if trainer.current_epoch < self._check_after:
            return

        for metric_key, (field_name, threshold) in self._THRESHOLDS.items():
            val = trainer.callback_metrics.get(metric_key)
            if val is None:
                continue
            val_float = float(val)
            if val_float > threshold:
                print(
                    f"{_ALERT}: {field_name} MAE = {val_float:.2f} "
                    f"(threshold {threshold}) after epoch {trainer.current_epoch + 1}.\n"
                    "  Check data pipeline — likely a normalisation or target-encoding error."
                )


class OvercorrectionWarningCallback(pl.Callback):
    """Warn (non-stopping) if any field exceeds a per-field wrong-direction percentage.

    Reads from `pl_module._val_direction_outputs` (populated by
    `SonnaLightningModule.validation_step`). Each entry is a per-field
    `dict[field, (n_wrong, n_total)]`. Aggregates across all val batches
    and computes the per-field % wrong-direction. If any judgable field
    (n_total ≥ MIN_SAMPLES) crosses `threshold_pct`, prints an ALERT block
    listing the offending fields. Training is NOT halted — the warning is
    informational, mirroring OverfittingCallback's pattern.

    Originally `OvercorrectionHaltCallback` (with `trainer.should_stop = True`).
    Demoted to warning-only after three halts on the v1.1.0 / v1.2.0 runs:
    two were under-correction false positives caused by the direction-stats
    bug (now fixed via the pred-deadband in losses.py:direction_stats), the
    third was a true positive on a small extreme-truth subset that the model
    couldn't fit on 3K rows. Final test MAE per field is the better channel
    for surfacing those signals than a mid-training halt.

    Warns at most once per "bad streak" — once an epoch comes back clean (no
    offenders), the alert is allowed to fire again on a fresh streak.
    """

    MIN_SAMPLES: int = 5   # ignore fields with < 5 directional truths in val

    def __init__(self, threshold_pct: float = 25.0, check_after_epoch: int = 5) -> None:
        self._threshold = threshold_pct / 100.0
        self._check_after = check_after_epoch
        self._alerted = False

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if trainer.current_epoch < self._check_after:
            return
        per_batch = getattr(pl_module, "_val_direction_outputs", None)
        if not per_batch:
            return

        # Aggregate per-field counts across batches.
        totals: dict[str, list[int]] = {}
        for batch_dict in per_batch:
            for field, (n_wrong, n_total) in batch_dict.items():
                entry = totals.setdefault(field, [0, 0])
                entry[0] += n_wrong
                entry[1] += n_total

        offenders: list[tuple[str, float, int, int]] = []
        for field, (n_wrong, n_total) in totals.items():
            if n_total < self.MIN_SAMPLES:
                continue
            pct = n_wrong / n_total
            if pct > self._threshold:
                offenders.append((field, pct, n_wrong, n_total))

        if not offenders:
            self._alerted = False  # clean epoch resets the streak
            return

        if self._alerted:
            return  # already warned about this streak; stay quiet
        self._alerted = True

        offenders.sort(key=lambda r: -r[1])
        lines = [f"{_ALERT}: per-field overcorrection warning "
                 f"after epoch {trainer.current_epoch + 1}.",
                 f"  Threshold: {self._threshold*100:.1f}% wrong-direction. Offending fields:"]
        for field, pct, n_wrong, n_total in offenders:
            lines.append(f"    {field:<32}  {pct*100:5.1f}%  ({n_wrong}/{n_total})")
        lines.append("  Continuing training (warning-only). "
                     "Inspect final test MAE for these fields.")
        print("\n".join(lines))
