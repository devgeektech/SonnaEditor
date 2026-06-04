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
from typing import Any, Callable

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


class _ProfileEpochBridgeCallback(pl.Callback):
    """Forward validation epoch metrics to an optional UI/job callback."""

    def __init__(
        self,
        *,
        on_epoch_complete: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: Any | None = None,
    ) -> None:
        self._on_epoch_complete = on_epoch_complete
        self._cancel_event = cancel_event

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self._on_epoch_complete is not None and not trainer.sanity_checking:
            metrics = trainer.callback_metrics
            self._on_epoch_complete({
                "epoch": trainer.current_epoch,
                "train_loss": metrics.get("train_loss"),
                "val_loss": metrics.get("val_loss"),
            })
        if self._cancel_event is not None and self._cancel_event.is_set():
            trainer.should_stop = True


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
    p.add_argument(
        "--backbone-unfreeze-strategy",
        choices=("partial", "full", "progressive"),
        default="partial",
        help=(
            "Backbone freeze schedule: partial preserves legacy stage-0/1 freeze; "
            "full freezes all stages until --freeze-backbone-epochs; progressive "
            "freezes all stages then unfreezes upper/mid/all stages at epochs 5/10/15."
        ),
    )
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--resume-from-checkpoint", type=Path, default=None, metavar="CKPT")
    p.add_argument(
        "--base-model-checkpoint",
        type=Path,
        default=None,
        metavar="CKPT",
        help=(
            "Initialise model weights from a native SonnaEditor checkpoint "
            "without resuming optimizer/epoch state."
        ),
    )
    p.add_argument("--slider-set-version", choices=("v1", "v2"),
                   default=config.CURRENT_SLIDER_SET_VERSION,
                   help=argparse.SUPPRESS)
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
    major = 2 if slider_set_version == config.CURRENT_SLIDER_SET_VERSION else 1
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
    foundation_provenance: dict[str, Any] | None = None,
) -> dict:
    payload = {
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
    if foundation_provenance:
        payload.update(foundation_provenance)
    return payload


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


def _is_image_foundation_warm_start(checkpoint_path: Path | None) -> bool:
    if checkpoint_path is None:
        return False
    from sonna_editor.foundation import foundation_requires_slider_prior_initialisation

    return foundation_requires_slider_prior_initialisation(checkpoint_path)


def _trainer_log_every_n_steps(num_train_batches: int, preferred: int = 10) -> int:
    """Pick a Lightning log interval that stays valid on tiny datasets."""
    if num_train_batches <= 0:
        return 1
    return max(1, min(preferred, num_train_batches))


def _warm_start_model_from_checkpoint(
    *,
    model_cls: type,
    checkpoint_path: Path,
    registry: Any,
    slider_set_version: str,
) -> Any:
    """Create a training-registry model and copy compatible base weights.

    Foundation checkpoints may have a different categorical registry from the
    Personal AI dataset. Reusing the checkpoint's registry would make camera and
    lens IDs mean the wrong thing, so warm-starts keep the new training registry
    and skip categorical embedding tables while copying shared visual/metadata
    layers and heads.
    """
    from sonna_editor.foundation import is_image_foundation_checkpoint

    if is_image_foundation_checkpoint(checkpoint_path):
        log.info("Warm-starting from image-to-image foundation backbone")
        model = model_cls(
            registry=registry,
            freeze_backbone=True,
            _pretrained_backbone=False,
            arch_version=3,
            slider_set_version=slider_set_version,
            use_wb_metadata_skip=True,
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state: dict[str, torch.Tensor] = ckpt["model_state"]
        current_state = model.state_dict()
        filtered_state = {
            key: value
            for key, value in state.items()
            if key.startswith("backbone_features.")
            and key in current_state
            and current_state[key].shape == value.shape
        }
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        log.info(
            "Warm-start copied %d image-foundation backbone tensors from %s; "
            "skipped %d missing and %d unexpected tensors",
            len(filtered_state),
            checkpoint_path,
            len(missing),
            len(unexpected),
        )
        return model

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state: dict[str, torch.Tensor] = ckpt["model_state"]
    arch_config = ckpt.get("arch_config", {}) or {}
    source_slider_set_version = arch_config.get("slider_set_version")
    if source_slider_set_version is None:
        num_sliders = arch_config.get("num_sliders", model_cls._V1_OUTPUT_COUNT)
        source_slider_set_version = (
            "v2" if num_sliders >= model_cls._V2_OUTPUT_COUNT else "v1"
        )
    if source_slider_set_version != slider_set_version:
        raise ValueError(
            f"--slider-set-version={slider_set_version!r} does not match "
            f"base checkpoint slider_set_version={source_slider_set_version!r}"
        )

    arch_version = arch_config.get("arch_version")
    if arch_version is None:
        if "metadata_encoder.scene_stats_mlp.0.weight" in state:
            arch_version = 2
        elif "metadata_encoder.make_emb.weight" in state:
            arch_version = 1
        else:
            arch_version = 0

    model = model_cls(
        registry=registry,
        freeze_backbone=True,
        _pretrained_backbone=False,
        arch_version=int(arch_version),
        slider_set_version=slider_set_version,
        use_wb_metadata_skip=bool(arch_config.get("use_wb_metadata_skip", False)),
    )
    current_state = model.state_dict()
    filtered_state = {
        key: value
        for key, value in state.items()
        if key in current_state
        and current_state[key].shape == value.shape
        and not key.endswith("_emb.weight")
    }
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    log.info(
        "Warm-start copied %d tensors from %s; skipped %d missing and %d unexpected tensors",
        len(filtered_state),
        checkpoint_path,
        len(missing),
        len(unexpected),
    )
    return model


def _foundation_provenance(checkpoint_path: Path | None) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None
    from sonna_editor.foundation import describe_foundation_checkpoint

    return describe_foundation_checkpoint(checkpoint_path)


def _initialise_image_foundation_output_priors(
    *,
    model: Any,
    base_model_checkpoint: Path | None,
    train_parquet: Path,
    slider_set_version: str,
    disabled: bool,
) -> dict[str, float] | None:
    """Initialise random slider heads after an image-foundation warm start."""
    if disabled or not _is_image_foundation_warm_start(base_model_checkpoint):
        return None
    priors = _training_target_priors(train_parquet, slider_set_version)
    model.initialise_output_priors(priors)
    return priors


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
    foundation_provenance: dict[str, Any] | None = None,
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
            foundation_provenance=foundation_provenance,
        ),
    )
    return dest


def train_profile(args: argparse.Namespace) -> dict:
    """Train and optionally publish a Sonna profile from parsed arguments."""
    config.ensure_runtime_directories()
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
    resume_checkpoint = getattr(args, "resume_from_checkpoint", None)
    base_model_checkpoint = getattr(args, "base_model_checkpoint", None)
    if resume_checkpoint and base_model_checkpoint:
        raise ValueError(
            "Use either --resume-from-checkpoint for an interrupted run or "
            "--base-model-checkpoint for a warm start, not both."
        )

    checkpoint_to_load = resume_checkpoint or base_model_checkpoint
    if checkpoint_to_load and not checkpoint_to_load.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_to_load}")
    foundation_provenance = (
        _foundation_provenance(base_model_checkpoint)
        if base_model_checkpoint is not None and resume_checkpoint is None
        else None
    )

    if checkpoint_to_load:
        if resume_checkpoint:
            log.info("Resuming trainer state from %s", resume_checkpoint)
            model = SonnaEditor.from_checkpoint(resume_checkpoint)
        else:
            log.info("Warm-starting model weights from %s", base_model_checkpoint)
            model = _warm_start_model_from_checkpoint(
                model_cls=SonnaEditor,
                checkpoint_path=base_model_checkpoint,
                registry=reg,
                slider_set_version=args.slider_set_version,
            )
        if model._slider_set_version != args.slider_set_version:
            raise ValueError(
                f"--slider-set-version={args.slider_set_version!r} does not match "
                f"checkpoint slider_set_version={model._slider_set_version!r}"
            )
        priors = _initialise_image_foundation_output_priors(
            model=model,
            base_model_checkpoint=base_model_checkpoint if not resume_checkpoint else None,
            train_parquet=args.train_parquet,
            slider_set_version=args.slider_set_version,
            disabled=args.no_target_prior_init,
        )
        if priors is not None:
            log.info(
                "Initialised image-foundation warm-start heads from training target "
                "medians (Exposure2012=%0.3f, Temperature=%0.0f, Tint=%0.2f)",
                priors.get("Exposure2012", 0.0),
                priors.get("Temperature", 0.0),
                priors.get("Tint", 0.0),
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
        backbone_unfreeze_strategy=getattr(args, "backbone_unfreeze_strategy", "partial"),
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
    if getattr(args, "on_epoch_complete", None) is not None or getattr(args, "cancel_event", None) is not None:
        callbacks.append(
            _ProfileEpochBridgeCallback(
                on_epoch_complete=getattr(args, "on_epoch_complete", None),
                cancel_event=getattr(args, "cancel_event", None),
            )
        )

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
        enable_progress_bar=getattr(args, "enable_progress_bar", True),
    )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    trainer.fit(
        lightning_module,
        datamodule=dm,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint else None,
    )

    if getattr(args, "cancel_event", None) is not None and args.cancel_event.is_set():
        summary = {
            "cancelled": True,
            "best_val_loss": float(trainer.checkpoint_callback.best_model_score or 0.0),
            "best_checkpoint": trainer.checkpoint_callback.best_model_path,
            "final_model": None,
            "published_model": None,
            "test_results": {},
            "epochs_trained": trainer.current_epoch,
            "hparams": {
                "arch_version": lightning_module.model._arch_version,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "freeze_backbone_epochs": args.freeze_backbone_epochs,
                "backbone_unfreeze_strategy": getattr(
                    args, "backbone_unfreeze_strategy", "partial"
                ),
                "image_resolution": config.IMAGE_RESOLUTION,
                "slider_set_version": args.slider_set_version,
                "use_wb_metadata_skip": not args.no_wb_metadata_skip,
                "target_prior_init": not args.no_target_prior_init,
                "base_model_checkpoint": str(base_model_checkpoint) if base_model_checkpoint else None,
                "foundation_provenance": foundation_provenance,
            },
        }
        summary_path = args.output_dir / "training_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        log.info("Training cancelled; summary written to %s", summary_path)
        return summary

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
    display_name = args.profile_name or "Sonna trained profile"
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
            foundation_provenance=foundation_provenance,
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
            foundation_provenance=foundation_provenance,
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
            "backbone_unfreeze_strategy": getattr(
                args, "backbone_unfreeze_strategy", "partial"
            ),
            "image_resolution": config.IMAGE_RESOLUTION,
            "slider_set_version": args.slider_set_version,
            "use_wb_metadata_skip": not args.no_wb_metadata_skip,
            "target_prior_init": not args.no_target_prior_init,
            "base_model_checkpoint": str(base_model_checkpoint) if base_model_checkpoint else None,
            "foundation_provenance": foundation_provenance,
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
    return summary


def main() -> None:
    train_profile(_parse_args())


if __name__ == "__main__":
    main()
