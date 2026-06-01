#!/usr/bin/env python
"""Train a Sonna Editor profile on a Lightroom-edited photo dataset.

Example:
    uv run scripts/train_profile.py \\
        --train-parquet data/splits/train.parquet \\
        --val-parquet   data/splits/val.parquet   \\
        --test-parquet  data/splits/test.parquet  \\
        --output-dir    checkpoints/v1
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

warnings.filterwarnings(
    "ignore",
    message=r".*isinstance\(treespec, LeafSpec\).*is deprecated.*",
)
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

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

_VERSIONED_MODEL_RE = re.compile(r"^model-v(\d+)\.(\d+)\.(\d+)(?:-\w+)?\.ckpt$")
_PUBLISH_VERSION_RE = re.compile(r"^(?:model-)?v\d+\.\d+\.\d+(?:-\w+)?$")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Sonna Editor profile")
    p.add_argument("--train-parquet", required=True, type=Path, metavar="PATH")
    p.add_argument("--val-parquet",   required=True, type=Path, metavar="PATH")
    p.add_argument("--test-parquet",  required=True, type=Path, metavar="PATH")
    p.add_argument("--output-dir",    required=True, type=Path, metavar="DIR",
                   help="Directory for checkpoints and TensorBoard logs")
    p.add_argument("--max-epochs",    type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--freeze-backbone-epochs", type=int, default=3,
                   help="Number of epochs to keep backbone frozen (default: 3)")
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--resume-from-checkpoint", type=Path, default=None, metavar="CKPT")
    p.add_argument("--slider-set-version", choices=("v1", "v2"), default="v2",
                   help="Target slider set for training. v2 is the current default.")
    p.add_argument("--no-wb-metadata-skip", action="store_true",
                   help="Disable the direct AsShot Temperature/Tint residual in new models.")
    p.add_argument("--no-target-prior-init", action="store_true",
                   help="Disable fresh-model output bias initialisation from train target medians.")
    p.add_argument("--image-resolution", type=int, default=512,
                   help="Model input resolution for training (default: 512)")
    p.add_argument("--temperature-weight", type=float, default=4.0,
                   help="Override the per-slider loss weight for Temperature")
    p.add_argument("--tint-weight", type=float, default=4.0,
                   help="Override the per-slider loss weight for Tint")
    p.add_argument("--exposure-weight", type=float, default=5.0,
                   help="Override the per-slider loss weight for Exposure2012")
    p.add_argument("--temperature-bucket-loss-weight", type=float, default=0.15,
                   help="Override TEMPERATURE_BUCKET_LOSS_WEIGHT")
    p.add_argument("--tint-bucket-loss-weight", type=float, default=2.0,
                   help="Override TINT_BUCKET_LOSS_WEIGHT")
    p.add_argument("--spread-loss-weight", type=float, default=None,
                   help="Override SPREAD_LOSS_WEIGHT")
    p.add_argument("--exposure-scene-loss-weight", type=float, default=4.0,
                   help="Override EXPOSURE_SCENE_LOSS_WEIGHT")
    p.add_argument("--sign-wrong-penalty-weight", type=float, default=0.2,
                   help="Override SIGN_WRONG_PENALTY_WEIGHT")
    p.add_argument("--profile-name", default=None,
                   help="Display name written to the profile sidecar for the UI")
    p.add_argument("--publish-dir", type=Path, default=config.CHECKPOINTS_DIR,
                   help="Directory scanned by the frontend profile API (default: v1_learning)")
    p.add_argument("--publish-version", default=None,
                   help="Optional version stem, e.g. v2.0.0 or model-v2.0.0")
    p.add_argument("--no-publish", action="store_true",
                   help="Only save output-dir/model.ckpt; do not copy a versioned profile for the UI")
    return p.parse_args()


def _apply_training_overrides(args: argparse.Namespace) -> None:
    provided_flags = {
        arg
        for raw in sys.argv[1:]
        if raw.startswith("--")
        for arg in (raw.split("=", 1)[0],)
    }

    def apply_value(
        *,
        flag: str,
        label: str,
        current: float | int,
        value: float | int | None,
        setter: Callable[[float | int], None],
        fmt: str,
    ) -> None:
        if value is None:
            return
        setter(value)
        prefix = "Override" if flag in provided_flags else "Training recipe"
        if flag not in provided_flags and value == current:
            return
        log.info("%s %s=%s", prefix, label, fmt % value)

    apply_value(
        flag="--image-resolution",
        label="IMAGE_RESOLUTION",
        current=config.IMAGE_RESOLUTION,
        value=args.image_resolution,
        setter=lambda v: setattr(config, "IMAGE_RESOLUTION", int(v)),
        fmt="%d",
    )
    apply_value(
        flag="--temperature-weight",
        label="Temperature weight",
        current=config.SLIDER_LOSS_WEIGHTS["Temperature"],
        value=args.temperature_weight,
        setter=lambda v: config.SLIDER_LOSS_WEIGHTS.__setitem__("Temperature", float(v)),
        fmt="%0.2f",
    )
    apply_value(
        flag="--tint-weight",
        label="Tint weight",
        current=config.SLIDER_LOSS_WEIGHTS["Tint"],
        value=args.tint_weight,
        setter=lambda v: config.SLIDER_LOSS_WEIGHTS.__setitem__("Tint", float(v)),
        fmt="%0.2f",
    )
    apply_value(
        flag="--exposure-weight",
        label="Exposure2012 weight",
        current=config.SLIDER_LOSS_WEIGHTS["Exposure2012"],
        value=args.exposure_weight,
        setter=lambda v: config.SLIDER_LOSS_WEIGHTS.__setitem__("Exposure2012", float(v)),
        fmt="%0.2f",
    )
    apply_value(
        flag="--temperature-bucket-loss-weight",
        label="TEMPERATURE_BUCKET_LOSS_WEIGHT",
        current=config.TEMPERATURE_BUCKET_LOSS_WEIGHT,
        value=args.temperature_bucket_loss_weight,
        setter=lambda v: setattr(config, "TEMPERATURE_BUCKET_LOSS_WEIGHT", float(v)),
        fmt="%0.2f",
    )
    apply_value(
        flag="--tint-bucket-loss-weight",
        label="TINT_BUCKET_LOSS_WEIGHT",
        current=config.TINT_BUCKET_LOSS_WEIGHT,
        value=args.tint_bucket_loss_weight,
        setter=lambda v: setattr(config, "TINT_BUCKET_LOSS_WEIGHT", float(v)),
        fmt="%0.2f",
    )
    apply_value(
        flag="--spread-loss-weight",
        label="SPREAD_LOSS_WEIGHT",
        current=config.SPREAD_LOSS_WEIGHT,
        value=args.spread_loss_weight,
        setter=lambda v: setattr(config, "SPREAD_LOSS_WEIGHT", float(v)),
        fmt="%0.2f",
    )
    apply_value(
        flag="--exposure-scene-loss-weight",
        label="EXPOSURE_SCENE_LOSS_WEIGHT",
        current=config.EXPOSURE_SCENE_LOSS_WEIGHT,
        value=args.exposure_scene_loss_weight,
        setter=lambda v: setattr(config, "EXPOSURE_SCENE_LOSS_WEIGHT", float(v)),
        fmt="%0.2f",
    )
    apply_value(
        flag="--sign-wrong-penalty-weight",
        label="SIGN_WRONG_PENALTY_WEIGHT",
        current=config.SIGN_WRONG_PENALTY_WEIGHT,
        value=args.sign_wrong_penalty_weight,
        setter=lambda v: setattr(config, "SIGN_WRONG_PENALTY_WEIGHT", float(v)),
        fmt="%0.2f",
    )


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


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "sonna-trained-profile"


def _next_publish_stem(publish_dir: Path, slider_set_version: str) -> str:
    """Return the next model-vX.Y.Z stem for the frontend-scanned directory."""
    major = 2 if slider_set_version == "v2" else 1
    candidates: list[tuple[int, int, int]] = []
    if publish_dir.exists():
        for path in publish_dir.glob(f"model-v{major}.*.ckpt"):
            match = _VERSIONED_MODEL_RE.match(path.name)
            if not match:
                continue
            major_v, minor_v, patch_v = (int(part) for part in match.groups())
            candidates.append((major_v, minor_v, patch_v))

    if not candidates:
        return f"model-v{major}.0.0"

    cur_major, cur_minor, cur_patch = max(candidates)
    return f"model-v{cur_major}.{cur_minor}.{cur_patch + 1}"


def _normalise_publish_stem(value: str) -> str:
    raw = value.removesuffix(".ckpt")
    if not _PUBLISH_VERSION_RE.match(raw):
        raise ValueError(
            "--publish-version must look like v2.0.0 or model-v2.0.0 "
            "(optional single suffix like -prod is allowed)"
        )
    return raw if raw.startswith("model-") else f"model-{raw}"


def _profile_sidecar_payload(
    *,
    checkpoint_path: Path,
    display_name: str,
    profile_id: str,
    slider_set_version: str,
    arch_version: int,
    use_wb_metadata_skip: bool,
    train_rows: int,
    val_loss: float,
) -> dict:
    return {
        "display_name": display_name,
        "profile_type": "mode_a_trained",
        "profile_id": profile_id,
        "checkpoint_path": str(checkpoint_path),
        "date_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "resolution": config.IMAGE_RESOLUTION,
        "image_resolution": config.IMAGE_RESOLUTION,
        "slider_set_version": slider_set_version,
        "arch_version": arch_version,
        "use_wb_metadata_skip": use_wb_metadata_skip,
        "default_skip_fields": [],
        "train_rows": train_rows,
        "val_loss": val_loss,
    }


def _write_sidecar(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _training_target_priors(train_parquet: Path, slider_set_version: str) -> dict[str, float]:
    """Return median training targets for each slider field.

    These priors are used only for fresh models. They put output-head biases in
    a sensible starting state before gradient descent, which matters on small
    datasets where random head outputs can dominate early validation behavior.
    """
    import pandas as pd

    from sonna_editor.slider_set import fields_for_version

    df = pd.read_parquet(train_parquet)
    priors: dict[str, float] = {}
    for field in fields_for_version(slider_set_version):
        if field not in df.columns:
            if field in config.SLIDER_DEFAULTS:
                priors[field] = float(config.SLIDER_DEFAULTS[field])
            continue
        values = pd.to_numeric(df[field], errors="coerce")
        if field == "Temperature":
            values = values[values > 0]
        if values.notna().any():
            priors[field] = float(values.median())
        elif field in config.SLIDER_DEFAULTS:
            priors[field] = float(config.SLIDER_DEFAULTS[field])
    return priors


def _trainer_log_every_n_steps(num_train_batches: int, preferred: int = 10) -> int:
    """Pick a Lightning log interval that stays valid on tiny datasets."""
    if num_train_batches <= 0:
        return 1
    return max(1, min(preferred, num_train_batches))


def _publish_profile_checkpoint(
    *,
    source_ckpt: Path,
    publish_dir: Path,
    publish_version: str | None,
    display_name: str,
    slider_set_version: str,
    arch_version: int,
    use_wb_metadata_skip: bool,
    train_rows: int,
    val_loss: float,
) -> Path:
    """Copy the trained native checkpoint into the directory scanned by the UI."""
    publish_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        _normalise_publish_stem(publish_version)
        if publish_version
        else _next_publish_stem(publish_dir, slider_set_version)
    )
    dest = publish_dir / f"{stem}.ckpt"
    if dest.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing published checkpoint: {dest}"
        )

    shutil.copy2(source_ckpt, dest)
    profile_id = f"{_slugify(display_name)}-{stem.removeprefix('model-')}"
    _write_sidecar(
        dest.with_suffix(".json"),
        _profile_sidecar_payload(
            checkpoint_path=dest.resolve(),
            display_name=display_name,
            profile_id=profile_id,
            slider_set_version=slider_set_version,
            arch_version=arch_version,
            use_wb_metadata_skip=use_wb_metadata_skip,
            train_rows=train_rows,
            val_loss=val_loss,
        ),
    )
    return dest


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
    num_train_batches = len(dm.train_dataloader())
    log_every_n_steps = _trainer_log_every_n_steps(num_train_batches)
    if log_every_n_steps < 10:
        log.info(
            "Small training split: %d train batches; using log_every_n_steps=%d",
            num_train_batches,
            log_every_n_steps,
        )
    reg = dm.registry
    log.info(
        "Registry: %d bodies  %d makes  %d models  %d lenses  %d profiles  %d WB presets",
        len(reg.camera_bodies), len(reg.camera_makes), len(reg.camera_models),
        len(reg.lenses), len(reg.camera_profiles), len(reg.wb_presets),
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
        if not args.no_target_prior_init:
            priors = _training_target_priors(args.train_parquet, args.slider_set_version)
            model.initialise_output_priors(priors)
            log.info(
                "Initialised fresh output heads from training target medians "
                "(Exposure2012=%0.3f, Temperature=%0.0f, Tint=%0.2f)",
                priors.get("Exposure2012", 0.0),
                priors.get("Temperature", 0.0),
                priors.get("Tint", 0.0),
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
        log_every_n_steps=log_every_n_steps,
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
    best_val_loss = float(trainer.checkpoint_callback.best_model_score or 0.0)
    display_name = args.profile_name or f"Sonna trained profile ({args.slider_set_version})"
    final_model_path = args.output_dir / "model.ckpt"
    if best_ckpt:
        _load_best_weights_into_model(lightning_module.model, best_ckpt)
    lightning_module.model.save_checkpoint(final_model_path)
    log.info("Saved final model to %s", final_model_path)
    sidecar_path = final_model_path.with_suffix(".json")
    _write_sidecar(
        sidecar_path,
        _profile_sidecar_payload(
            checkpoint_path=final_model_path.resolve(),
            display_name=display_name,
            profile_id=_slugify(display_name),
            slider_set_version=lightning_module.model._slider_set_version,
            arch_version=lightning_module.model._arch_version,
            use_wb_metadata_skip=lightning_module.model._use_wb_metadata_skip,
            train_rows=len(dm._train_ds),
            val_loss=best_val_loss,
        ),
    )
    log.info("Saved model sidecar to %s", sidecar_path)

    published_model_path: str | None = None
    if not args.no_publish:
        published = _publish_profile_checkpoint(
            source_ckpt=final_model_path,
            publish_dir=args.publish_dir,
            publish_version=args.publish_version,
            display_name=display_name,
            slider_set_version=lightning_module.model._slider_set_version,
            arch_version=lightning_module.model._arch_version,
            use_wb_metadata_skip=lightning_module.model._use_wb_metadata_skip,
            train_rows=len(dm._train_ds),
            val_loss=best_val_loss,
        )
        published_model_path = str(published)
        log.info("Published frontend-visible profile to %s", published)

    summary = {
        "best_val_loss": best_val_loss,
        "best_checkpoint": best_ckpt,
        "final_model": str(final_model_path),
        "published_model": published_model_path,
        "test_results": test_results[0] if test_results else {},
        "epochs_trained": trainer.current_epoch,
        "hparams": {
            "arch_version": lightning_module.model._arch_version,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "image_resolution": config.IMAGE_RESOLUTION,
            "slider_set_version": args.slider_set_version,
            "use_wb_metadata_skip": not args.no_wb_metadata_skip,
            "target_prior_init": not args.no_target_prior_init,
            "temperature_weight": args.temperature_weight,
            "tint_weight": args.tint_weight,
            "exposure_weight": args.exposure_weight,
            "temperature_bucket_loss_weight": args.temperature_bucket_loss_weight,
            "tint_bucket_loss_weight": args.tint_bucket_loss_weight,
            "spread_loss_weight": args.spread_loss_weight,
            "exposure_scene_loss_weight": args.exposure_scene_loss_weight,
            "sign_wrong_penalty_weight": args.sign_wrong_penalty_weight,
        },
    }
    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Training summary written to %s", summary_path)


if __name__ == "__main__":
    main()
