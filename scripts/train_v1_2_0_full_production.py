#!/usr/bin/env python
# ruff: noqa: E402
"""v1.2.0 FULL PRODUCTION run — 12.9K stratified dataset, 256px.

The production training run. Forked from train_v1_2_0_3k_smoketest.py with
data-scale changes only — same architecture, same loss config, same image
resolution, same callbacks. The v1.2.0 256 smoke profile passed real-photo
validation; this is the same model trained on the full dataset to recover
the quality the 3K subset was leaving on the table.

Differences from the 3K smoke:
  - Train parquet: train_3k_random_256test.parquet  (3K random subset)
                 → splits_v2_stratified/train.parquet  (full ~9,746 rows)
  - Val/test:     splits/val.parquet, splits/test.parquet  (old)
                → splits_v2_stratified/{val,test}.parquet  (stratified)
  - max_epochs:   15 → 30
  - ETACallback budget: 2h → 12h (overnight run; allow headroom)
  - Output dir, final ckpt, summary all renamed for v1.2.0 full production

Everything else is BYTE-IDENTICAL to v1.2.0 256 smoke:
  - Image resolution: 256
  - freeze_backbone_epochs: 2
  - EarlyStopping(patience=8)
  - ResetEarlyStoppingOnUnfreeze(unfreeze_epoch=2)
  - OvercorrectionWarningCallback(threshold_pct=25.0, check_after_epoch=10)
  - CriticalMAECallback(check_after_epoch=10)
  - LossComponentBalanceCallback(ratio_threshold=5.0, check_after_epoch=8)
  - OverfittingCallback(patience=4)
  - NaNLossCallback, DiskSpaceCallback
  - Loss weights at v1.2.0 baseline (TEMPERATURE_BUCKET=0.10, TINT_BUCKET=1.50,
    SIGN_WRONG=0.15, SPREAD=0.50; SLIDER_LOSS_WEIGHTS in config.py:305-351)
  - lr=3e-4, weight_decay=1e-4, batch_size=16, sample_weight_col="sample_weight"
  - prediction_mode is NOT a thing — absolute prediction is the only path
    (v1.3 delta-mode infra was reverted)

Estimated wall time: ~4-6 h. v1.2.0 256 smoke took ~70 min on 3K with 23
epochs. Scaling: 3K → 9.7K is 3.2×; 23 → 30 epochs is 1.3×. Combined ~4.2×
gives ~4h 50m baseline. The 12h ETACallback budget gives substantial
headroom for variance.

Fallback: if the run produces catastrophic regression, revert via
  `git reset --hard checkpoint-pre-v1.3-delta-prediction`  (== current HEAD)
or rerun this same script — it's deterministic with the seeded shuffle.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── 256px override — MUST come before any other sonna_editor import. ──
# augmentation.py does `from sonna_editor.config import IMAGE_RESOLUTION`,
# which binds the value at import time. Patching config after that import
# would silently fail (training would run at 512). Set this here, then
# import the rest.
import sonna_editor.config as _cfg
from sonna_editor.runtime import preferred_lightning_accelerator
_cfg.IMAGE_RESOLUTION = 256

warnings.filterwarnings(
    "ignore",
    message=r".*isinstance\(treespec, LeafSpec\).*is deprecated.*",
)
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from sonna_editor.model.architecture import SonnaEditor
from sonna_editor.training.callbacks import (
    CriticalMAECallback,
    DiskSpaceCallback,
    ETACallback,
    LossComponentBalanceCallback,
    NaNLossCallback,
    OverfittingCallback,
    OvercorrectionWarningCallback,
)
from sonna_editor.training.datamodule import SonnaDataModule
from sonna_editor.training.module import SonnaLightningModule
from sonna_editor.training.unfreeze_callback import ResetEarlyStoppingOnUnfreeze

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SPLITS_DIR    = PROJECT_ROOT / "v1_learning" / "dataset" / "splits_v2_stratified"
TRAIN_PARQUET = SPLITS_DIR / "train.parquet"
VAL_PARQUET   = SPLITS_DIR / "val.parquet"
TEST_PARQUET  = SPLITS_DIR / "test.parquet"
OUTPUT_DIR    = PROJECT_ROOT / "v1_learning" / "v1_2_0_full_production"
FINAL_CKPT    = PROJECT_ROOT / "v1_learning" / "model-v1.2.0-full-production.ckpt"
SUMMARY_JSON  = PROJECT_ROOT / "v1_learning" / "training_summary_v1.2.0_full_production.json"

FREEZE_BACKBONE_EPOCHS = 2
EARLY_STOPPING_PATIENCE = 8
MAX_EPOCHS = 30
IMAGE_RESOLUTION_OVERRIDE = 256

OVERCORRECTION_CHECK_AFTER     = 10
CRITICAL_MAE_CHECK_AFTER       = 10
LOSS_BALANCE_CHECK_AFTER       = 8
ETA_BUDGET_HOURS               = 12.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--force", action="store_true",
        help="If model-v1.2.0-full-production.ckpt already exists, overwrite it.",
    )
    return p.parse_args()


def _handle_existing_checkpoint(force: bool) -> None:
    if not FINAL_CKPT.exists():
        return
    if not force:
        sys.exit(
            f"refusing to overwrite existing {FINAL_CKPT}.\n"
            f"  re-run with --force to overwrite."
        )
    FINAL_CKPT.unlink()
    log.info("--force: removed existing %s", FINAL_CKPT.name)


def main() -> None:
    args = _parse_args()

    if not TRAIN_PARQUET.exists():
        sys.exit(f"missing {TRAIN_PARQUET}.")
    _handle_existing_checkpoint(args.force)

    # Verify the monkey-patch took effect — augmentation.py read IMAGE_RESOLUTION
    # at import time. If something imported augmentation before our patch line
    # (e.g. via __init__.py side effects), the next assertion fires loudly.
    from sonna_editor.model import augmentation as _aug
    actual_res = _aug.IMAGE_RESOLUTION
    if actual_res != IMAGE_RESOLUTION_OVERRIDE:
        sys.exit(
            f"IMAGE_RESOLUTION override FAILED: augmentation module sees "
            f"{actual_res}, expected {IMAGE_RESOLUTION_OVERRIDE}. "
            f"Some other import path bound the constant before our patch."
        )
    log.info("IMAGE_RESOLUTION override verified: augmentation module sees %d", actual_res)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dm = SonnaDataModule(
        train_parquet=TRAIN_PARQUET,
        val_parquet=VAL_PARQUET,
        test_parquet=TEST_PARQUET,
        batch_size=16,
        num_workers=4,
        sample_weight_col="sample_weight",
        slider_set_version="v1",
    )
    dm.prepare_data()
    dm.setup("fit")
    log.info(
        "Dataset: train=%d val=%d test=%d  registry: %d bodies / %d lenses / %d profiles / %d wb_presets",
        len(dm._train_ds), len(dm._val_ds), len(dm._test_ds),
        len(dm.registry.camera_bodies), len(dm.registry.lenses),
        len(dm.registry.camera_profiles), len(dm.registry.wb_presets),
    )

    model = SonnaEditor(
        registry=dm.registry,
        freeze_backbone=True,
        slider_set_version="v1",
        use_wb_metadata_skip=False,
    )
    lm = SonnaLightningModule(
        model=model,
        lr=3e-4,
        weight_decay=1e-4,
        freeze_backbone_epochs=FREEZE_BACKBONE_EPOCHS,
    )

    ckpt_dir = OUTPUT_DIR / "checkpoints"
    tb_logger = TensorBoardLogger(save_dir=str(OUTPUT_DIR), name="tensorboard")
    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="v1.2.0-prod-{epoch:03d}-val{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            mode="min",
            verbose=True,
        ),
        ResetEarlyStoppingOnUnfreeze(unfreeze_epoch=FREEZE_BACKBONE_EPOCHS),
        LearningRateMonitor(logging_interval="epoch"),
        NaNLossCallback(),
        OverfittingCallback(patience=4),
        LossComponentBalanceCallback(
            ratio_threshold=5.0,
            check_after_epoch=LOSS_BALANCE_CHECK_AFTER,
        ),
        OvercorrectionWarningCallback(
            threshold_pct=25.0,
            check_after_epoch=OVERCORRECTION_CHECK_AFTER,
        ),
        CriticalMAECallback(check_after_epoch=CRITICAL_MAE_CHECK_AFTER),
        ETACallback(max_hours=ETA_BUDGET_HOURS, check_after_epoch=2),
        DiskSpaceCallback(watch_path=PROJECT_ROOT, min_free_gb=5.0),
    ]

    trainer = pl.Trainer(
        accelerator=preferred_lightning_accelerator(),
        devices=1,
        precision="32-true",
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=20,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    log.info(
        "v1.2.0 FULL PRODUCTION training begins. "
        "arch_version=%d  image_resolution=%d  train_rows=%d  max_epochs=%d  "
        "freeze_backbone_epochs=%d  early_stopping_patience=%d. "
        "Recalibrated callbacks: overcorrection.check_after=%d  "
        "critical_mae.check_after=%d  loss_balance.check_after=%d  "
        "eta_budget=%.1fh. "
        "Loss: sign_wrong=%.2f tint_bucket=%.2f temp_bucket=%.2f spread=%.2f",
        model._arch_version, _cfg.IMAGE_RESOLUTION, len(dm._train_ds), MAX_EPOCHS,
        FREEZE_BACKBONE_EPOCHS, EARLY_STOPPING_PATIENCE,
        OVERCORRECTION_CHECK_AFTER, CRITICAL_MAE_CHECK_AFTER,
        LOSS_BALANCE_CHECK_AFTER, ETA_BUDGET_HOURS,
        _cfg.SIGN_WRONG_PENALTY_WEIGHT, _cfg.TINT_BUCKET_LOSS_WEIGHT,
        _cfg.TEMPERATURE_BUCKET_LOSS_WEIGHT, _cfg.SPREAD_LOSS_WEIGHT,
    )
    trainer.fit(lm, datamodule=dm)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    log.info("Best checkpoint: %s", best_ckpt)
    test_results = trainer.test(lm, datamodule=dm, ckpt_path=best_ckpt) if best_ckpt else []

    lm.model.save_checkpoint(FINAL_CKPT)
    log.info("Saved final model → %s", FINAL_CKPT)

    summary = {
        "best_val_loss": float(trainer.checkpoint_callback.best_model_score or 0.0),
        "best_checkpoint": best_ckpt,
        "final_model": str(FINAL_CKPT),
        "epochs_trained": trainer.current_epoch,
        "test_results": test_results[0] if test_results else {},
        "halted_early": trainer.should_stop and trainer.current_epoch < (MAX_EPOCHS - 1),
        "config": {
            "name": "v1.2.0 full production",
            "image_resolution": IMAGE_RESOLUTION_OVERRIDE,
            "train_parquet": str(TRAIN_PARQUET),
            "val_parquet": str(VAL_PARQUET),
            "test_parquet": str(TEST_PARQUET),
            "train_rows": len(dm._train_ds),
            "val_rows": len(dm._val_ds),
            "test_rows": len(dm._test_ds),
            "max_epochs": MAX_EPOCHS,
            "freeze_backbone_epochs": FREEZE_BACKBONE_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "overcorrection_check_after_epoch": OVERCORRECTION_CHECK_AFTER,
            "critical_mae_check_after_epoch": CRITICAL_MAE_CHECK_AFTER,
            "loss_balance_check_after_epoch": LOSS_BALANCE_CHECK_AFTER,
            "eta_budget_hours": ETA_BUDGET_HOURS,
        },
        "hparams": {
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "batch_size": 16,
            "sample_weight_col": "sample_weight",
        },
        "loss_settings": {
            "sign_wrong_penalty_weight": _cfg.SIGN_WRONG_PENALTY_WEIGHT,
            "spread_loss_weight": _cfg.SPREAD_LOSS_WEIGHT,
            "temperature_bucket_loss_weight": _cfg.TEMPERATURE_BUCKET_LOSS_WEIGHT,
            "tint_bucket_loss_weight": _cfg.TINT_BUCKET_LOSS_WEIGHT,
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    log.info("Summary → %s", SUMMARY_JSON)


if __name__ == "__main__":
    main()
