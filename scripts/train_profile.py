#!/usr/bin/env python
"""Train a Sonna Editor profile on a Lightroom-edited photo dataset.

Example:
    uv run scripts/train_profile.py \\
        --train-parquet data/splits/train.parquet \\
        --val-parquet   data/splits/val.parquet   \\
        --test-parquet  data/splits/test.parquet  \\
        --output-dir    checkpoints/v1
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

import sonna_editor.config as config
from sonna_editor.runtime import preferred_lightning_accelerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Sonna Editor profile")
    p.add_argument("--train-parquet", required=True, type=Path, metavar="PATH")
    p.add_argument("--val-parquet",   required=True, type=Path, metavar="PATH")
    p.add_argument("--test-parquet",  required=True, type=Path, metavar="PATH")
    p.add_argument("--output-dir",    required=True, type=Path, metavar="DIR",
                   help="Directory for checkpoints and TensorBoard logs")
    p.add_argument("--max-epochs",    type=int,   default=100)
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--freeze-backbone-epochs", type=int, default=10,
                   help="Number of epochs to keep backbone frozen (default: 10)")
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--resume-from-checkpoint", type=Path, default=None, metavar="CKPT")
    p.add_argument("--slider-set-version", choices=("v1", "v2"), default="v2",
                   help="Target slider set for training. v2 is the current default.")
    p.add_argument("--no-wb-metadata-skip", action="store_true",
                   help="Disable the direct AsShot Temperature/Tint residual in new models.")
    p.add_argument("--image-resolution", type=int, default=None,
                   help="Override model input resolution for training (default: config IMAGE_RESOLUTION)")
    p.add_argument("--temperature-weight", type=float, default=None,
                   help="Override the per-slider loss weight for Temperature")
    p.add_argument("--tint-weight", type=float, default=None,
                   help="Override the per-slider loss weight for Tint")
    p.add_argument("--exposure-weight", type=float, default=None,
                   help="Override the per-slider loss weight for Exposure2012")
    p.add_argument("--temperature-bucket-loss-weight", type=float, default=None,
                   help="Override TEMPERATURE_BUCKET_LOSS_WEIGHT")
    p.add_argument("--tint-bucket-loss-weight", type=float, default=None,
                   help="Override TINT_BUCKET_LOSS_WEIGHT")
    p.add_argument("--spread-loss-weight", type=float, default=None,
                   help="Override SPREAD_LOSS_WEIGHT")
    p.add_argument("--sign-wrong-penalty-weight", type=float, default=None,
                   help="Override SIGN_WRONG_PENALTY_WEIGHT")
    return p.parse_args()


def _apply_training_overrides(args: argparse.Namespace) -> None:
    if args.image_resolution is not None:
        config.IMAGE_RESOLUTION = args.image_resolution
        log.info("Override IMAGE_RESOLUTION=%d", config.IMAGE_RESOLUTION)
    if args.temperature_weight is not None:
        config.SLIDER_LOSS_WEIGHTS["Temperature"] = args.temperature_weight
        log.info("Override Temperature weight=%0.2f", args.temperature_weight)
    if args.tint_weight is not None:
        config.SLIDER_LOSS_WEIGHTS["Tint"] = args.tint_weight
        log.info("Override Tint weight=%0.2f", args.tint_weight)
    if args.exposure_weight is not None:
        config.SLIDER_LOSS_WEIGHTS["Exposure2012"] = args.exposure_weight
        log.info("Override Exposure2012 weight=%0.2f", args.exposure_weight)
    if args.temperature_bucket_loss_weight is not None:
        config.TEMPERATURE_BUCKET_LOSS_WEIGHT = args.temperature_bucket_loss_weight
        log.info("Override TEMPERATURE_BUCKET_LOSS_WEIGHT=%0.2f", args.temperature_bucket_loss_weight)
    if args.tint_bucket_loss_weight is not None:
        config.TINT_BUCKET_LOSS_WEIGHT = args.tint_bucket_loss_weight
        log.info("Override TINT_BUCKET_LOSS_WEIGHT=%0.2f", args.tint_bucket_loss_weight)
    if args.spread_loss_weight is not None:
        config.SPREAD_LOSS_WEIGHT = args.spread_loss_weight
        log.info("Override SPREAD_LOSS_WEIGHT=%0.2f", args.spread_loss_weight)
    if args.sign_wrong_penalty_weight is not None:
        config.SIGN_WRONG_PENALTY_WEIGHT = args.sign_wrong_penalty_weight
        log.info("Override SIGN_WRONG_PENALTY_WEIGHT=%0.2f", args.sign_wrong_penalty_weight)


def _load_best_weights_into_model(model, best_ckpt: str) -> None:
    """Load Lightning's best model weights into the native SonnaEditor instance.

    Lightning checkpoint callbacks save the full training module. The app wants
    a native SonnaEditor checkpoint, so copy the best validation weights back
    before calling SonnaEditor.save_checkpoint().
    """
    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    state = {
        k[len("model.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state, strict=True)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _apply_training_overrides(args)

    # Late imports so config overrides are applied before model/loss modules load.
    from sonna_editor.model.architecture import SonnaEditor
    from sonna_editor.training.datamodule import SonnaDataModule
    from sonna_editor.training.module import SonnaLightningModule

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    dm = SonnaDataModule(
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        test_parquet=args.test_parquet,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        slider_set_version=args.slider_set_version,
    )
    dm.prepare_data()
    dm.setup("fit")

    log.info(
        "Dataset: train=%d  val=%d  test=%d",
        len(dm._train_ds), len(dm._val_ds), len(dm._test_ds),
    )
    reg = dm.registry
    log.info(
        "Registry: %d bodies  %d lenses  %d profiles  %d WB presets",
        len(reg.camera_bodies), len(reg.lenses),
        len(reg.camera_profiles), len(reg.wb_presets),
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    if args.resume_from_checkpoint and args.resume_from_checkpoint.exists():
        log.info("Resuming from %s", args.resume_from_checkpoint)
        model = SonnaEditor.from_checkpoint(args.resume_from_checkpoint)
        if model._slider_set_version != args.slider_set_version:
            raise ValueError(
                f"--slider-set-version={args.slider_set_version!r} does not match "
                f"checkpoint slider_set_version={model._slider_set_version!r}"
            )
    else:
        model = SonnaEditor(
            registry=reg,
            freeze_backbone=True,
            slider_set_version=args.slider_set_version,
            use_wb_metadata_skip=not args.no_wb_metadata_skip,
        )
        log.info(
            "Created fresh %s model (backbone frozen for %d epochs, wb_metadata_skip=%s)",
            args.slider_set_version,
            args.freeze_backbone_epochs,
            model._use_wb_metadata_skip,
        )

    lightning_module = SonnaLightningModule(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
    )

    # -----------------------------------------------------------------------
    # Callbacks + logger
    # -----------------------------------------------------------------------
    ckpt_dir = args.output_dir / "checkpoints"
    tb_logger = TensorBoardLogger(save_dir=str(args.output_dir), name="tensorboard")

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch={epoch:03d}-val_loss={val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # -----------------------------------------------------------------------
    # Trainer. fp32 is the safe cross-platform default for CUDA, MPS, and CPU.
    # -----------------------------------------------------------------------
    trainer = pl.Trainer(
        accelerator=preferred_lightning_accelerator(),
        devices=1,
        precision="32-true",
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    trainer.fit(
        lightning_module,
        datamodule=dm,
        ckpt_path=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None,
    )

    # -----------------------------------------------------------------------
    # Test on best checkpoint
    # -----------------------------------------------------------------------
    best_ckpt = trainer.checkpoint_callback.best_model_path
    log.info("Best checkpoint: %s", best_ckpt)
    if best_ckpt:
        test_results = trainer.test(lightning_module, datamodule=dm, ckpt_path=best_ckpt)
    else:
        test_results = trainer.test(lightning_module, datamodule=dm)

    # -----------------------------------------------------------------------
    # Save final model + summary
    # -----------------------------------------------------------------------
    final_model_path = args.output_dir / "model.ckpt"
    if best_ckpt:
        _load_best_weights_into_model(lightning_module.model, best_ckpt)
    lightning_module.model.save_checkpoint(final_model_path)
    log.info("Saved final model to %s", final_model_path)
    sidecar_path = final_model_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps({
        "display_name": f"Sonna trained profile ({args.slider_set_version})",
        "checkpoint_path": str(final_model_path),
        "image_resolution": config.IMAGE_RESOLUTION,
        "slider_set_version": lightning_module.model._slider_set_version,
        "use_wb_metadata_skip": lightning_module.model._use_wb_metadata_skip,
        "default_skip_fields": [],
    }, indent=2))
    log.info("Saved model sidecar to %s", sidecar_path)

    summary = {
        "best_val_loss": float(trainer.checkpoint_callback.best_model_score or 0.0),
        "best_checkpoint": best_ckpt,
        "final_model": str(final_model_path),
        "test_results": test_results[0] if test_results else {},
        "epochs_trained": trainer.current_epoch,
        "hparams": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "image_resolution": config.IMAGE_RESOLUTION,
            "slider_set_version": args.slider_set_version,
            "use_wb_metadata_skip": not args.no_wb_metadata_skip,
            "temperature_weight": args.temperature_weight,
            "tint_weight": args.tint_weight,
            "exposure_weight": args.exposure_weight,
            "temperature_bucket_loss_weight": args.temperature_bucket_loss_weight,
            "tint_bucket_loss_weight": args.tint_bucket_loss_weight,
            "spread_loss_weight": args.spread_loss_weight,
            "sign_wrong_penalty_weight": args.sign_wrong_penalty_weight,
        },
    }
    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Training summary written to %s", summary_path)


if __name__ == "__main__":
    main()
