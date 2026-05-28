"""ResetEarlyStoppingOnUnfreeze callback.

Why this exists
---------------
When `SonnaLightningModule.freeze_backbone_epochs > 0`, the backbone is
frozen for the first N epochs and only the heads + metadata encoder train.
At epoch N, `on_train_epoch_start` in module.py unfreezes the backbone.

Before unfreeze and after unfreeze, val_loss is measuring two functionally
different models — pre-unfreeze val_loss reflects a tiny trainable head on
top of frozen ImageNet features; post-unfreeze val_loss reflects the full
network adapting to the task. Comparing them is apples to oranges.

PyTorch Lightning's stock `EarlyStopping` does NOT know about this
distinction. It accumulates `wait_count` across the unfreeze boundary, so
if val_loss happens to peak on a frozen-head epoch (which is common — the
head overfits fast on a small effective parameter set), EarlyStopping will
fire BEFORE the backbone ever unfreezes. That's exactly what killed the
v1.1.0 full training run (best=epoch 1, EarlyStopping fired at epoch 6,
backbone scheduled to unfreeze at epoch 10 — never happened).

The fix
-------
At the start of the unfreeze epoch (epoch == freeze_backbone_epochs), reset
the EarlyStopping callback's `wait_count` to 0 and `best_score` to the
sentinel (+inf for "min" mode, -inf for "max" mode). This gives the
unfrozen model a fresh window to improve, judged against its own baseline
rather than the frozen-head baseline.

Place this callback AFTER EarlyStopping in the callbacks list — Lightning
runs `on_train_epoch_start` callbacks in registration order. Order doesn't
strictly matter for correctness (the reset happens before EarlyStopping's
on_validation_end check), but registration-order-after is the principle of
least surprise.
"""
from __future__ import annotations

import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping

log = logging.getLogger(__name__)


class ResetEarlyStoppingOnUnfreeze(pl.Callback):
    """Reset EarlyStopping's counters when the backbone unfreezes.

    Parameters
    ----------
    unfreeze_epoch : int
        The epoch number at which the backbone unfreezes. Must match the
        LightningModule's `freeze_backbone_epochs`. Set to 0 to disable
        (no freeze phase → nothing to reset).
    """

    def __init__(self, unfreeze_epoch: int) -> None:
        super().__init__()
        self.unfreeze_epoch = int(unfreeze_epoch)
        self._fired = False

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self._fired:
            return
        if self.unfreeze_epoch <= 0:
            return
        if trainer.current_epoch != self.unfreeze_epoch:
            return

        # Find the EarlyStopping callback (there should be exactly one;
        # if there are several monitoring different metrics, reset them all).
        es_callbacks = [
            cb for cb in trainer.callbacks if isinstance(cb, EarlyStopping)
        ]
        if not es_callbacks:
            log.warning(
                "ResetEarlyStoppingOnUnfreeze: no EarlyStopping callback found; "
                "nothing to reset."
            )
            self._fired = True
            return

        for es in es_callbacks:
            old_wait = es.wait_count
            old_best = float(es.best_score) if es.best_score is not None else None
            es.wait_count = 0
            # Sentinel matching EarlyStopping's mode. `min_delta` is mode-aware
            # internally via `monitor_op`, but `best_score` we reset by hand.
            sentinel = (
                torch.tensor(float("inf")) if es.mode == "min"
                else torch.tensor(float("-inf"))
            )
            es.best_score = sentinel
            log.info(
                "ResetEarlyStoppingOnUnfreeze fired at epoch %d. "
                "Reset EarlyStopping(monitor=%s, mode=%s): "
                "wait_count %d → 0, best_score %s → %s. "
                "Post-unfreeze model gets a fresh patience window.",
                trainer.current_epoch, es.monitor, es.mode,
                old_wait, old_best, sentinel.item(),
            )

        self._fired = True
