from __future__ import annotations

import math
from typing import Any

import pytorch_lightning as pl
import torch
import torch.optim as optim

from sonna_editor.config import SLIDER_FIELDS
from sonna_editor.model.architecture import SonnaEditor
from sonna_editor.model.losses import WeightedSliderLoss
from sonna_editor.slider_set import fields_for_version

# Fields logged individually at validation time
_KEY_FIELDS = ["Exposure2012", "Temperature", "Shadows2012", "Highlights2012",
               "Whites2012", "Blacks2012", "Clarity2012", "Vibrance", "Saturation"]
_HSL_FIELDS = [f for f in SLIDER_FIELDS if "Adjustment" in f]  # 24 fields
_DISTRIBUTION_FIELDS = [
    "Exposure2012",
    "Contrast2012",
    "Highlights2012",
    "Shadows2012",
    "Temperature",
    "Tint",
]


class SonnaLightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper around SonnaEditor.

    Differential learning rates: backbone at lr/10, heads + metadata encoder at lr.
    Backbone is kept frozen for the first `freeze_backbone_epochs` epochs, then
    unfrozen (still at the reduced LR to avoid destroying pretrained features).

    Scheduler: CosineAnnealingWarmRestarts(T_0=20) — resets every 20 epochs.
    """

    def __init__(
        self,
        model: SonnaEditor,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        freeze_backbone_epochs: int = 10,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.freeze_backbone_epochs = freeze_backbone_epochs

        self.loss_fn = WeightedSliderLoss(
            slider_set_version=model._slider_set_version,
        )

        # Accumulation lists populated per-step, flushed per-epoch
        self._val_mae_outputs: list[dict[str, float]] = []
        self._test_mae_outputs: list[dict[str, float]] = []
        # Per-batch direction-stat counts. Each entry: dict[field, (n_wrong, n_total)].
        # Consumed by OvercorrectionWarningCallback at validation_epoch_end.
        self._val_direction_outputs: list[dict[str, tuple[int, int]]] = []
        self._val_distribution_outputs: list[dict[str, torch.Tensor]] = []

        self.save_hyperparameters(ignore=["model"])

    # ------------------------------------------------------------------
    # Backbone freeze scheduling
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.freeze_backbone_epochs:
            self.model.unfreeze_backbone()
            self._log_metric("backbone_unfrozen_epoch", float(self.current_epoch))

    def _log_metric(self, name: str, value: Any, **kwargs: Any) -> None:
        """Log only when attached to a Trainer.

        Unit tests call step methods directly to inspect loss behavior. Lightning
        warns on `self.log()` in that standalone mode, while real training always
        attaches a Trainer before steps run.
        """
        if getattr(self, "_trainer", None) is None:
            return
        self.log(name, value, **kwargs)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        images, metadata, targets = batch
        predictions = self.model(images, metadata)
        components = self.loss_fn(predictions, targets, metadata, return_components=True)
        loss = components["total"]
        # Per-row mask in the loss layer excludes bad rows from loss math.
        # When EVERY row is invalid (rare extreme), the loss returns a zero
        # total. Log so we can spot if this happens repeatedly.
        if "_all_rows_skipped" in components:
            print(f"⚠ training_step: ALL rows in batch {batch_idx} were masked "
                  f"out by the loss layer's validity check.", flush=True)
        batch_size = images.size(0)
        self._log_metric("train_loss",             loss,                    on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self._log_metric("train_loss_mse",         components["mse"],        on_step=False, on_epoch=True, batch_size=batch_size)
        self._log_metric("train_loss_exposure_scene", components["exposure_scene"], on_step=False, on_epoch=True, batch_size=batch_size)
        self._log_metric("train_loss_spread",      components["spread"],     on_step=False, on_epoch=True, batch_size=batch_size)
        self._log_metric("train_loss_temp_bucket", components["temp_bucket"], on_step=False, on_epoch=True, batch_size=batch_size)
        self._log_metric("train_loss_tint_bucket", components["tint_bucket"], on_step=False, on_epoch=True, batch_size=batch_size)
        self._log_metric("train_loss_sign_wrong",  components["sign_wrong"],  on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ) -> None:
        images, metadata, targets = batch
        batch_size = images.size(0)
        predictions = self.model(images, metadata)
        components = self.loss_fn(predictions, targets, metadata, return_components=True)
        loss = components["total"]
        self._log_metric("val_loss",             loss,                    on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self._log_metric("val_loss_mse",         components["mse"],        on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self._log_metric("val_loss_exposure_scene", components["exposure_scene"], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self._log_metric("val_loss_spread",      components["spread"],     on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self._log_metric("val_loss_temp_bucket", components["temp_bucket"], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self._log_metric("val_loss_tint_bucket", components["tint_bucket"], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self._log_metric("val_loss_sign_wrong",  components["sign_wrong"],  on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        mae = self.loss_fn.per_field_mae(predictions, targets)
        self._val_mae_outputs.append(mae)

        # Per-field direction stats — fed to OvercorrectionWarningCallback.
        dir_stats = self.loss_fn.direction_stats(predictions, targets, metadata)
        self._val_direction_outputs.append(dir_stats)
        self._val_distribution_outputs.append({
            "predictions": predictions.detach().cpu(),
            "targets": targets.detach().cpu(),
        })

    def test_step(
        self,
        batch: tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ) -> None:
        images, metadata, targets = batch
        batch_size = images.size(0)
        predictions = self.model(images, metadata)
        loss = self.loss_fn(predictions, targets, metadata)
        self._log_metric("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        mae = self.loss_fn.per_field_mae(predictions, targets)
        self._test_mae_outputs.append(mae)

    # ------------------------------------------------------------------
    # Epoch-end aggregation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._val_mae_outputs.clear()
        self._val_direction_outputs.clear()
        self._val_distribution_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        self._log_aggregated_mae(self._val_mae_outputs, prefix="val")
        self._log_distribution_stats()

    def on_test_epoch_start(self) -> None:
        self._test_mae_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self._log_aggregated_mae(self._test_mae_outputs, prefix="test")

    def _log_aggregated_mae(self, outputs: list[dict[str, float]], prefix: str) -> None:
        if not outputs:
            return

        # Nanmean across batches for every field
        field_means: dict[str, float] = {}
        for field in SLIDER_FIELDS:
            vals = [d[field] for d in outputs if not math.isnan(d.get(field, math.nan))]
            field_means[field] = (sum(vals) / len(vals)) if vals else math.nan

        # Log key individual fields
        for field in _KEY_FIELDS:
            val = field_means.get(field, math.nan)
            if not math.isnan(val):
                safe_name = field.replace("2012", "").lower()
                self._log_metric(f"{prefix}_mae_{safe_name}", val, prog_bar=(prefix == "val"))

        # Log HSL average
        hsl_vals = [field_means[f] for f in _HSL_FIELDS if not math.isnan(field_means.get(f, math.nan))]
        if hsl_vals:
            self._log_metric(f"{prefix}_mae_hsl_avg", sum(hsl_vals) / len(hsl_vals), prog_bar=(prefix == "val"))

    def _log_distribution_stats(self) -> None:
        if not self._val_distribution_outputs:
            return
        predictions = torch.cat(
            [d["predictions"] for d in self._val_distribution_outputs],
            dim=0,
        )
        targets = torch.cat(
            [d["targets"] for d in self._val_distribution_outputs],
            dim=0,
        )
        fields = fields_for_version(self.model._slider_set_version)
        for field in _DISTRIBUTION_FIELDS:
            if field not in fields:
                continue
            idx = fields.index(field)
            pred = predictions[:, idx].float()
            target = targets[:, idx].float()
            mask = ~torch.isnan(target)
            if field == "Temperature":
                pred = torch.exp(pred)
                mask &= target > 0
            if int(mask.sum().item()) < 2:
                continue
            pred_valid = pred[mask]
            target_valid = target[mask]
            pred_std = torch.std(pred_valid, unbiased=False)
            target_std = torch.std(target_valid, unbiased=False)
            ratio = pred_std / target_std.clamp(min=1e-8)
            safe_name = field.replace("2012", "").lower()
            self._log_metric(f"val_dist_{safe_name}_pred_std", float(pred_std), prog_bar=False)
            self._log_metric(f"val_dist_{safe_name}_target_std", float(target_std), prog_bar=False)
            self._log_metric(f"val_dist_{safe_name}_std_ratio", float(ratio), prog_bar=False)

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        backbone_ids = {
            id(p)
            for module in (
                self.model.backbone_features,
                self.model.backbone_pool,
                self.model.backbone_norm,
            )
            for p in module.parameters()
        }

        backbone_params = [p for p in self.model.parameters() if id(p) in backbone_ids]
        other_params    = [p for p in self.model.parameters() if id(p) not in backbone_ids]

        optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": self.lr / 10},
                {"params": other_params,    "lr": self.lr},
            ],
            weight_decay=self.weight_decay,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=1, eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }
