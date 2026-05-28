"""Fine-tuning pipeline — produces a new versioned checkpoint from captured user edits.

Original checkpoints are NEVER overwritten. All writes go to new versioned files.
Fine-tuning is always invoked explicitly; this module never auto-triggers training.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from sonna_editor import config
from sonna_editor.inference.engine import _load_from_checkpoint
from sonna_editor.model.architecture import SonnaEditor
from sonna_editor.runtime import preferred_lightning_accelerator
from sonna_editor.training.datamodule import SonnaDataModule
from sonna_editor.training.module import SonnaLightningModule

_logger = logging.getLogger(__name__)


class _BridgeCallback(pl.Callback):
    """Forwards Lightning epoch-end signals into a sync user callback.

    Also honours a cooperative cancellation Event by setting
    ``trainer.should_stop = True`` (Lightning checks this between epochs).
    """

    def __init__(
        self,
        on_epoch_complete: Callable[[dict[str, Any]], None],
        cancel_event: Optional[threading.Event],
    ) -> None:
        super().__init__()
        self._cb = on_epoch_complete
        self._cancel_event = cancel_event

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        try:
            metrics = trainer.callback_metrics
            tl = metrics.get("train_loss")
            vl = metrics.get("val_loss")
            self._cb({
                "epoch": int(trainer.current_epoch),
                "train_loss": float(tl) if tl is not None else None,
                "val_loss": float(vl) if vl is not None else None,
            })
        except Exception as e:  # noqa: BLE001 — never crash training
            _logger.warning("on_epoch_complete callback raised %r", e)

        if self._cancel_event is not None and self._cancel_event.is_set():
            trainer.should_stop = True

_VERSION_RE = re.compile(r"model-v(\d+)\.(\d+)\.(\d+)")


def _bump_version(output_dir: Path, base_checkpoint: Path) -> str:
    """
    Return the next version label by scanning output_dir and base_checkpoint for
    model-v{major}.{minor}.{patch} filenames and incrementing the highest patch.
    """
    candidates = list(output_dir.glob("model-v*.ckpt")) + [base_checkpoint]
    best: tuple[int, int, int] = (1, 0, 0)
    for p in candidates:
        m = _VERSION_RE.search(p.name)
        if m:
            triple = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if triple > best:
                best = triple
    major, minor, patch = best
    return f"model-v{major}.{minor}.{patch + 1}"


def _aggregate_mae(val_mae_outputs: list[dict[str, float]]) -> dict[str, float]:
    """Nanmean across validation batches for every slider field present.

    Derives the field set from the input dicts (union of keys) rather than
    iterating config.SLIDER_FIELDS — after Commit 2a, per_field_mae returns
    a dict sized to the loss's slider_set_version (135 for v1, 147 for v2),
    so the input keys ARE the canonical field set. Iterating SLIDER_FIELDS
    unconditionally produced 12 always-NaN entries for v1 inputs.
    """
    if not val_mae_outputs:
        return {}
    present = set().union(*(d.keys() for d in val_mae_outputs))
    # Order by config.SLIDER_FIELDS for deterministic output.
    fields = [f for f in config.SLIDER_FIELDS if f in present]
    result: dict[str, float] = {}
    for field in fields:
        vals = [
            d[field]
            for d in val_mae_outputs
            if not math.isnan(d.get(field, math.nan))
        ]
        result[field] = sum(vals) / len(vals) if vals else math.nan
    return result


def _evaluate(
    model: SonnaEditor,
    val_parquet: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[float, dict[str, float]]:
    """
    Evaluate model on val_parquet. Returns (val_loss, per_field_mae).

    Uses CPU accelerator — evaluation-only, no GPU needed for a small val set.
    Model is moved to CPU for the duration of this call.
    """
    device_before = next(model.parameters()).device
    model.cpu()

    lm = SonnaLightningModule(model=model, lr=1e-4, freeze_backbone_epochs=0)
    dm = SonnaDataModule(
        train_parquet=val_parquet,
        val_parquet=val_parquet,
        test_parquet=val_parquet,
        batch_size=batch_size,
        num_workers=num_workers,
        registry=model.registry,
        slider_set_version=model._slider_set_version,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.validate(lm, datamodule=dm)

    val_loss = float(trainer.callback_metrics.get("val_loss", float("nan")))
    per_field_mae = _aggregate_mae(lm._val_mae_outputs)

    model.to(device_before)
    return val_loss, per_field_mae


def _atomic_save(model: SonnaEditor, path: Path) -> None:
    """
    Save checkpoint atomically: write to .ckpt.tmp then os.replace() to final path.
    The original file at `path` is never partially overwritten.
    """
    tmp_path = path.with_suffix(".ckpt.tmp")
    model.save_checkpoint(tmp_path)
    os.replace(tmp_path, path)


def _write_version_sidecar(path: Path, metadata: dict) -> None:
    """Write a .json sidecar alongside the checkpoint with training metadata."""
    sidecar_path = path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(metadata, indent=2))


def _grow_registry_for_df(model: SonnaEditor, df: pd.DataFrame) -> None:
    """Register any camera bodies / lenses / profiles / WB presets absent from model.registry."""
    col_method = [
        ("camera_body",          model.add_camera_body),
        ("lens_model",           model.add_lens),
        ("camera_profile",       model.add_camera_profile),
        ("white_balance_preset", model.add_wb_preset),
    ]
    for col, add_fn in col_method:
        if col not in df.columns:
            continue
        for val in df[col].dropna().unique():
            s = str(val)
            if s:
                add_fn(s)


def finetune_model(
    base_checkpoint: Path,
    finetune_parquet: Path,
    val_parquet: Path,
    output_dir: Path,
    *,
    n_capture_rows: int = 0,
    n_original_rows: int = 0,
    lr: float = 1e-4,
    max_epochs: int = 30,
    patience: int = 5,
    freeze_backbone_epochs: int = 5,
    batch_size: int = 16,
    num_workers: int = 4,
    on_epoch_complete: Optional[Callable[[dict[str, Any]], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """
    Fine-tune the model on a combined dataset (original training + user captures).

    The base checkpoint is NEVER modified. The fine-tuned model is saved to a new
    versioned file in output_dir. If validation loss regresses, the checkpoint is
    saved with a '-candidate' suffix and the caller must decide whether to promote it.

    Args:
        base_checkpoint:  Path to the checkpoint to fine-tune from.
        finetune_parquet: Combined Parquet from prepare_finetune_dataset().
        val_parquet:      The SAME validation split used during original training.
        output_dir:       Directory for the new versioned checkpoint.
        n_capture_rows:   Number of captured rows in finetune_parquet (for reporting).
        n_original_rows:  Number of original rows in finetune_parquet (for reporting).
        lr:               Fine-tune learning rate (default 1e-4, 3× lower than training).
        max_epochs:       Maximum training epochs (default 30).
        patience:         Early-stopping patience on val_loss (default 5).
        freeze_backbone_epochs: Epochs to keep backbone frozen (default 5).
        batch_size:       Batch size for fine-tuning (default 16).
        num_workers:      DataLoader workers (default 4).

    Returns:
        dict with keys: base_version, ft_version, checkpoint_path, checkpoint_status,
        base_val_loss, ft_val_loss, improvement_pct, improved, epochs_trained,
        n_capture_rows, n_original_rows, base_per_field_mae, ft_per_field_mae.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_checkpoint = Path(base_checkpoint).resolve()
    base_version = _VERSION_RE.search(base_checkpoint.name)
    base_version_label = base_version.group(0) if base_version else base_checkpoint.stem

    # --- Load base checkpoint ---
    base_model = _load_from_checkpoint(base_checkpoint, device="cpu")

    # --- Grow registry for any new cameras/lenses in the fine-tune set ---
    df_finetune = pd.read_parquet(finetune_parquet)
    _grow_registry_for_df(base_model, df_finetune)

    # --- Evaluate base model before training ---
    print(f"Evaluating base model ({base_version_label}) on validation set...")
    base_val_loss, base_per_field_mae = _evaluate(
        base_model, val_parquet, batch_size=batch_size, num_workers=num_workers
    )
    print(f"  base val_loss = {base_val_loss:.6f}")

    # --- Set up DataModule for fine-tuning ---
    dm = SonnaDataModule(
        train_parquet=finetune_parquet,
        val_parquet=val_parquet,
        test_parquet=val_parquet,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_weight_col="sample_weight",
        registry=base_model.registry,
        slider_set_version=base_model._slider_set_version,
    )

    # --- Lightning module ---
    lm = SonnaLightningModule(
        model=base_model,
        lr=lr,
        freeze_backbone_epochs=freeze_backbone_epochs,
    )

    # --- Trainer callbacks ---
    with tempfile.TemporaryDirectory(prefix="sonna_ft_ckpts_") as tmp_dir:
        ckpt_callback = ModelCheckpoint(
            dirpath=tmp_dir,
            filename="ft-{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        cb_list: list[pl.Callback] = [ckpt_callback, early_stop, lr_monitor]
        if on_epoch_complete is not None or cancel_event is not None:
            cb_list.append(_BridgeCallback(
                on_epoch_complete=on_epoch_complete or (lambda _: None),
                cancel_event=cancel_event,
            ))

        trainer = pl.Trainer(
            accelerator=preferred_lightning_accelerator(),
            devices=1,
            precision="32-true",
            max_epochs=max_epochs,
            callbacks=cb_list,
            enable_progress_bar=True,
            logger=False,
            enable_checkpointing=True,
        )

        print(f"\nFine-tuning for up to {max_epochs} epochs (patience={patience})...")
        trainer.fit(lm, datamodule=dm)
        epochs_trained = trainer.current_epoch + 1

        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        if not best_ckpt_path:
            raise RuntimeError("No best checkpoint was saved — training may have failed immediately.")

        # --- Load best checkpoint and copy registry ---
        best_model = _load_from_checkpoint(Path(best_ckpt_path), device="cpu")
        best_model.registry = base_model.registry

    # tmp_dir and its Lightning checkpoints are now cleaned up

    # --- Evaluate fine-tuned model ---
    print("\nEvaluating fine-tuned model on validation set...")
    ft_val_loss, ft_per_field_mae = _evaluate(
        best_model, val_parquet, batch_size=batch_size, num_workers=num_workers
    )
    print(f"  ft val_loss   = {ft_val_loss:.6f}")

    # --- Determine outcome ---
    improved = ft_val_loss < base_val_loss
    improvement_pct = (base_val_loss - ft_val_loss) / base_val_loss * 100.0

    # --- Version and save ---
    ft_label = _bump_version(output_dir, base_checkpoint)
    filename = f"{ft_label}.ckpt" if improved else f"{ft_label}-candidate.ckpt"
    out_path = output_dir / filename

    _atomic_save(best_model, out_path)
    _write_version_sidecar(out_path, {
        "version": ft_label,
        "date_iso": datetime.now(timezone.utc).isoformat(),
        "base_version": base_version_label,
        "base_val_loss": base_val_loss,
        "ft_val_loss": ft_val_loss,
        "improvement_pct": round(improvement_pct, 3),
        "n_capture_rows": n_capture_rows,
        "n_original_rows": n_original_rows,
        "epochs_trained": epochs_trained,
        "checkpoint_status": "promoted" if improved else "candidate",
    })

    return {
        "base_version": base_version_label,
        "ft_version": ft_label,
        "checkpoint_path": str(out_path.resolve()),
        "checkpoint_status": "promoted" if improved else "candidate",
        "base_val_loss": base_val_loss,
        "ft_val_loss": ft_val_loss,
        "improvement_pct": improvement_pct,
        "improved": improved,
        "epochs_trained": epochs_trained,
        "n_capture_rows": n_capture_rows,
        "n_original_rows": n_original_rows,
        "base_per_field_mae": base_per_field_mae,
        "ft_per_field_mae": ft_per_field_mae,
    }
