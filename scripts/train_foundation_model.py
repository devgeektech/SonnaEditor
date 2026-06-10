#!/usr/bin/env python
"""Build/train/promote the Sonna foundation checkpoint.

This is the canonical path for creating the base checkpoint used by Lite
profile creation. It is intentionally separate from Personal AI profile
training: the output is promoted to the configured hidden foundation folder,
not to the frontend profile directory.

Supported training modes:
- parameter-supervised: RAW+XMP, catalog-derived splits, or other prepared
  parquet splits -> Lightroom sliders
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from sonna_editor import config
from sonna_editor.foundation import (
    ensure_foundation_repo_layout,
    promote_foundation_checkpoint,
    resolve_foundation_checkpoint,
    validate_foundation_checkpoint,
)
from sonna_editor.training.profile_runner import train_profile

log = logging.getLogger(__name__)

_DEFAULT_MIN_FOUNDATION_TRAIN_ROWS = 75
_SMALL_FOUNDATION_TRAIN_ROWS = 500
_DEFAULT_FOUNDATION_BACKBONE_STRATEGY = "progressive"
_DEFAULT_FOUNDATION_BACKBONE_LAYERS = "stage:7"
_SMALL_FOUNDATION_BACKBONE_STRATEGY = "custom"
_SMALL_FOUNDATION_BACKBONE_LAYERS = "none"
_TONE_PRESENCE_RETRY_WEIGHTS: dict[str, float] = {
    "Exposure2012": 10.0,
    "Whites2012": 10.0,
    "Blacks2012": 10.0,
    "Highlights2012": 8.0,
    "Shadows2012": 6.0,
    "Vibrance": 6.0,
    "Saturation": 6.0,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the foundation checkpoint from RAW+XMP data or prepared "
            "Lightroom-parameter parquet splits."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--raw-xmp-dir",
        type=Path,
        help="Folder containing edited RAW files with matching .xmp sidecars.",
    )
    source.add_argument(
        "--splits-dir",
        type=Path,
        help="Existing directory containing train.parquet, val.parquet, and test.parquet.",
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=config.TRAINING_WORKSPACE_DIR,
        help=(
            "External workspace for generated datasets and training runs "
            f"(default: {config.TRAINING_WORKSPACE_DIR})."
        ),
    )
    parser.add_argument(
        "--foundation-repo",
        type=Path,
        default=config.FOUNDATION_REPO_DIR,
        help=(
            "Foundation model folder "
            f"(default: {config.FOUNDATION_REPO_DIR})."
        ),
    )
    parser.add_argument("--profile-name", default="Sonna Foundation")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--version-stem", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-resolution", type=int, default=config.IMAGE_RESOLUTION)
    parser.add_argument(
        "--field-loss-weight",
        action="append",
        default=[],
        metavar="FIELD=WEIGHT",
        help=(
            "Override a named Lightroom slider loss weight during foundation "
            "training. Repeatable, e.g. --field-loss-weight Whites2012=6."
        ),
    )
    parser.add_argument(
        "--tone-presence-retry",
        action="store_true",
        help=(
            "Apply the reviewed foundation retry recipe for runs that pass WB/HSL "
            "but fail tone/presence gates. This raises loss pressure on Exposure, "
            "Whites, Blacks, Highlights, Shadows, Vibrance, and Saturation. "
            "Explicit --field-loss-weight values override this preset per field."
        ),
    )
    parser.add_argument(
        "--min-foundation-train-rows",
        type=int,
        default=_DEFAULT_MIN_FOUNDATION_TRAIN_ROWS,
        help=(
            "Minimum train split size required for normal foundation promotion "
            f"(default: {_DEFAULT_MIN_FOUNDATION_TRAIN_ROWS}). Use "
            "--allow-small-foundation-dataset only for "
            "deliberate smoke/ablation runs."
        ),
    )
    parser.add_argument(
        "--allow-small-foundation-dataset",
        action="store_true",
        help=(
            "Allow foundation training/promotion with fewer rows than "
            "--min-foundation-train-rows. This is intended for smoke tests or "
            "explicit ablations, not production foundation updates."
        ),
    )
    parser.add_argument(
        "--allow-quality-gate-failure",
        action="store_true",
        help=(
            "Promote even when held-out test metrics are outside the foundation "
            "quality gate. Use only after visual review."
        ),
    )
    parser.add_argument(
        "--backbone-unfreeze-strategy",
        choices=("progressive", "custom", "full", "partial"),
        default=_DEFAULT_FOUNDATION_BACKBONE_STRATEGY,
        help=(
            "Foundation backbone schedule. Default progressive starts from "
            "--backbone-trainable-layers and expands at later epochs; custom "
            "keeps that layer spec fixed."
        ),
    )
    parser.add_argument(
        "--backbone-trainable-layers",
        default=_DEFAULT_FOUNDATION_BACKBONE_LAYERS,
        help=(
            "Initial ConvNeXt trainable layer spec for foundation training "
            "(default: stage:7). Examples: block:7:2,stage:6; "
            "block:7:1-2,stage:6; stage:7; none."
        ),
    )
    parser.add_argument("--val-ratio", type=float, default=0.107)
    parser.add_argument("--test-ratio", type=float, default=0.139)
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help=(
            "Start from pretrained/default weights instead of the active foundation "
            "checkpoint. By default each foundation run warm-starts from the active "
            "checkpoint and writes a new versioned checkpoint."
        ),
    )
    return parser.parse_args()


def _batch_size_attempts(initial_batch_size: int) -> list[int]:
    if initial_batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    attempts: list[int] = []
    batch_size = initial_batch_size
    while batch_size >= 1:
        attempts.append(batch_size)
        if batch_size == 1:
            break
        batch_size = max(1, batch_size // 2)
    return attempts


def _is_cuda_memory_failure(exc: Exception) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if (
            "cuda" in message
            and (
                "out of memory" in message
                or "cudnn_status_execution_failed_cudart" in message
                or "cudaerrormemoryallocation" in message
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _clear_cuda_after_failure() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()  # type: ignore[no-untyped-call]
        except RuntimeError:
            pass


def _parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as exc:
        raise RuntimeError(f"Could not read parquet row count: {path}") from exc


def _split_row_counts(splits_dir: Path) -> dict[str, int]:
    return {
        split: _parquet_row_count(splits_dir / f"{split}.parquet")
        for split in ("train", "val", "test")
    }


def _validate_foundation_split_size(
    *,
    splits_dir: Path,
    min_train_rows: int,
    allow_small_dataset: bool,
) -> dict[str, int]:
    counts = _split_row_counts(splits_dir)
    if counts["train"] < min_train_rows and not allow_small_dataset:
        raise RuntimeError(
            "Refusing to train/promote a production foundation checkpoint from "
            f"only {counts['train']} train rows at {splits_dir}. Foundation "
            "updates need enough scene/edit diversity to avoid overfitting the "
            "base model. Add more data, lower --min-foundation-train-rows for "
            "a reviewed run, or pass --allow-small-foundation-dataset for a "
            "deliberate smoke/ablation."
        )
    return counts


_FOUNDATION_METRIC_LIMITS: dict[str, tuple[float, float, str]] = {
    "test_mae_temperature": (350.0, 500.0, "Temperature"),
    "test_mae_tint": (8.0, 12.0, "Tint"),
    "test_mae_exposure": (0.25, 0.50, "Exposure2012"),
    "test_mae_shadows": (8.0, 15.0, "Shadows2012"),
    "test_mae_highlights": (8.0, 18.0, "Highlights2012"),
    "test_mae_whites": (5.0, 25.0, "Whites2012"),
    "test_mae_blacks": (5.0, 25.0, "Blacks2012"),
    "test_mae_clarity": (5.0, 10.0, "Clarity2012"),
    "test_mae_vibrance": (5.0, 10.0, "Vibrance"),
    "test_mae_saturation": (5.0, 10.0, "Saturation"),
    "test_mae_hsl_avg": (10.0, 14.0, "HSL average"),
}


_FOUNDATION_METRIC_FIELD_FALLBACKS: dict[str, str] = {
    "test_mae_temperature": "Temperature",
    "test_mae_tint": "Tint",
    "test_mae_exposure": "Exposure2012",
    "test_mae_shadows": "Shadows2012",
    "test_mae_highlights": "Highlights2012",
    "test_mae_whites": "Whites2012",
    "test_mae_blacks": "Blacks2012",
    "test_mae_clarity": "Clarity2012",
    "test_mae_vibrance": "Vibrance",
    "test_mae_saturation": "Saturation",
}


def _foundation_metric_value(summary: dict[str, Any], key: str) -> float | None:
    test_results = summary.get("test_results") or {}
    value = test_results.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    field = _FOUNDATION_METRIC_FIELD_FALLBACKS.get(key)
    per_field = summary.get("test_per_field_mae") or {}
    value = per_field.get(field) if field else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _foundation_quality_report(summary: dict[str, Any]) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    best_val = summary.get("best_val_loss")
    test_results = summary.get("test_results") or {}
    test_loss = test_results.get("test_loss")
    if (
        isinstance(best_val, (int, float))
        and not isinstance(best_val, bool)
        and best_val > 0
        and isinstance(test_loss, (int, float))
        and not isinstance(test_loss, bool)
    ):
        if test_loss > best_val * 1.35:
            failures.append(
                f"test_loss {test_loss:.6f} is more than 1.35x best_val_loss {best_val:.6f}"
            )
        elif test_loss > best_val * 1.25:
            warnings.append(
                f"test_loss {test_loss:.6f} is more than 1.25x best_val_loss {best_val:.6f}"
            )

    for key, (warn_limit, hard_limit, label) in _FOUNDATION_METRIC_LIMITS.items():
        value = _foundation_metric_value(summary, key)
        if value is None:
            continue
        if value > hard_limit:
            failures.append(f"{label} MAE {value:.4g} exceeds hard limit {hard_limit:g}")
        elif value > warn_limit:
            warnings.append(f"{label} MAE {value:.4g} exceeds target {warn_limit:g}")
    return failures, warnings


def _foundation_quality_failures(summary: dict[str, Any]) -> list[str]:
    failures, _ = _foundation_quality_report(summary)
    return failures


def _field_loss_weights_for_recipe(args: argparse.Namespace) -> list[str]:
    weights = list(getattr(args, "field_loss_weight", []) or [])
    if not getattr(args, "tone_presence_retry", False):
        return weights

    explicit_fields = {
        raw.split("=", 1)[0].strip()
        for raw in weights
        if "=" in raw
    }
    for field, weight in _TONE_PRESENCE_RETRY_WEIGHTS.items():
        if field not in explicit_fields:
            weights.append(f"{field}={weight:g}")
    return weights


def _write_quality_gate_result(
    *,
    training_dir: Path,
    summary: dict[str, Any],
    failures: list[str],
    warnings: list[str] | None = None,
) -> None:
    warnings = warnings or []
    summary["quality_gate_passed"] = not failures
    summary["foundation_quality_failures"] = failures
    summary["foundation_quality_warnings"] = warnings
    summary_path = training_dir / "training_summary.json"
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
            existing["quality_gate_passed"] = summary["quality_gate_passed"]
            existing["foundation_quality_failures"] = failures
            existing["foundation_quality_warnings"] = warnings
            summary_path.write_text(json.dumps(existing, indent=2))
            return
        except Exception:
            log.warning("Could not update quality gate result in %s", summary_path)


def _foundation_capacity_for_split(
    *,
    split_counts: dict[str, int],
    requested_strategy: str,
    requested_layers: str,
) -> tuple[str, str]:
    """Return a safer training capacity for the current foundation split.

    Tiny RAW+XMP continuations have repeatedly overfit when allowed to train the
    final ConvNeXt stage. Keep the broader default for catalog-scale datasets,
    but reduce default small-data runs to metadata/fusion/output heads only.
    Explicit non-default choices are left alone for deliberate ablations.
    """
    train_rows = split_counts["train"]
    using_default_capacity = (
        requested_strategy == _DEFAULT_FOUNDATION_BACKBONE_STRATEGY
        and requested_layers == _DEFAULT_FOUNDATION_BACKBONE_LAYERS
    )
    if train_rows >= _SMALL_FOUNDATION_TRAIN_ROWS or not using_default_capacity:
        return requested_strategy, requested_layers

    log.warning(
        "Small foundation split (%d train rows). Using safer backbone capacity "
        "%s/%s instead of %s/%s to reduce overfitting and collapse. Pass explicit "
        "--backbone-unfreeze-strategy and --backbone-trainable-layers for a "
        "reviewed ablation.",
        train_rows,
        _SMALL_FOUNDATION_BACKBONE_STRATEGY,
        _SMALL_FOUNDATION_BACKBONE_LAYERS,
        requested_strategy,
        requested_layers,
    )
    return _SMALL_FOUNDATION_BACKBONE_STRATEGY, _SMALL_FOUNDATION_BACKBONE_LAYERS


def _train_profile_with_cuda_oom_retry(train_args: argparse.Namespace) -> dict[str, Any]:
    requested_batch_size = int(train_args.batch_size)
    last_error: Exception | None = None
    for attempt_batch_size in _batch_size_attempts(requested_batch_size):
        train_args.batch_size = attempt_batch_size
        if attempt_batch_size != requested_batch_size:
            log.warning(
                "Retrying foundation training with --batch-size %d after CUDA memory failure",
                attempt_batch_size,
            )
        try:
            summary = train_profile(train_args)
            if attempt_batch_size != requested_batch_size:
                summary["auto_reduced_batch_size_from"] = requested_batch_size
                summary["auto_reduced_batch_size_to"] = attempt_batch_size
            return summary
        except Exception as exc:
            if not _is_cuda_memory_failure(exc) or attempt_batch_size == 1:
                raise
            last_error = exc
            log.warning(
                "CUDA memory failure at --batch-size %d. Clearing CUDA cache and retrying.",
                attempt_batch_size,
            )
            _clear_cuda_after_failure()
    raise RuntimeError("Foundation training failed after reducing batch size to 1") from last_error


def _active_foundation_or_none() -> Path | None:
    try:
        return resolve_foundation_checkpoint()
    except (FileNotFoundError, ValueError):
        return None


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _build_dataset_from_raw_xmp(args: argparse.Namespace, run_dir: Path) -> Path:
    from sonna_editor.data.dataset import build_dataset, save_split, split_dataset

    if args.raw_xmp_dir is None:
        raise ValueError("--raw-xmp-dir is required when building a dataset")
    input_dir = Path(args.raw_xmp_dir).expanduser()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"RAW+XMP folder not found: {input_dir}")

    dataset_dir = run_dir / "dataset"
    parquet_path = dataset_dir / "dataset.parquet"
    thumbnails_dir = dataset_dir / "thumbnails"
    splits_dir = dataset_dir / "splits_v2_stratified"

    df = build_dataset(
        input_dir=input_dir,
        output_path=parquet_path,
        profile_name=args.profile_name,
        thumbnail_dir=thumbnails_dir,
        max_workers=args.workers,
    )
    train, val, test = split_dataset(
        df,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    save_split(train, val, test, splits_dir)
    return splits_dir


def _resolve_splits(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.splits_dir is not None:
        splits_dir = Path(args.splits_dir).expanduser()
        missing = [
            name for name in ("train.parquet", "val.parquet", "test.parquet")
            if not (splits_dir / name).exists()
        ]
        if missing:
            raise FileNotFoundError(f"Splits directory missing: {', '.join(missing)}")
        return splits_dir
    return _build_dataset_from_raw_xmp(args, run_dir)


def main() -> None:
    args = _parse_args()
    foundation_repo = args.foundation_repo.expanduser()
    config.FOUNDATION_REPO_DIR = foundation_repo
    os.environ[config.FOUNDATION_REPO_ENV_VAR] = str(foundation_repo)
    ensure_foundation_repo_layout()

    workspace_dir = args.workspace_dir.expanduser()
    run_name = args.run_name or f"foundation-{_timestamp()}"
    run_dir = workspace_dir / "foundation_runs" / run_name
    training_dir = run_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    base_foundation = None if args.no_warm_start else _active_foundation_or_none()
    if base_foundation is not None:
        try:
            validate_foundation_checkpoint(base_foundation)
        except ValueError as exc:
            raise RuntimeError(
                "Active foundation checkpoint is invalid. "
                "If you want a brand-new foundation build, rerun with --no-warm-start. "
                f"Otherwise fix or replace {base_foundation}."
            ) from exc

    splits_dir = _resolve_splits(args, run_dir)
    split_counts = _validate_foundation_split_size(
        splits_dir=splits_dir,
        min_train_rows=args.min_foundation_train_rows,
        allow_small_dataset=args.allow_small_foundation_dataset,
    )
    log.info(
        "Foundation split rows: train=%d val=%d test=%d",
        split_counts["train"],
        split_counts["val"],
        split_counts["test"],
    )
    backbone_strategy, backbone_layers = _foundation_capacity_for_split(
        split_counts=split_counts,
        requested_strategy=args.backbone_unfreeze_strategy,
        requested_layers=args.backbone_trainable_layers,
    )

    train_args = argparse.Namespace(
        train_parquet=splits_dir / "train.parquet",
        val_parquet=splits_dir / "val.parquet",
        test_parquet=splits_dir / "test.parquet",
        output_dir=training_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=1e-4,
        weight_decay=1e-4,
        freeze_backbone_epochs=3,
        backbone_unfreeze_strategy=backbone_strategy,
        backbone_trainable_layers=backbone_layers,
        num_workers=args.workers,
        resume_from_checkpoint=None,
        base_model_checkpoint=base_foundation,
        slider_set_version=config.CURRENT_SLIDER_SET_VERSION,
        no_wb_metadata_skip=False,
        no_target_prior_init=False,
        image_resolution=args.image_resolution,
        checkpoint_monitor="val_visual_score",
        temperature_weight=4.0,
        tint_weight=4.0,
        exposure_weight=5.0,
        field_loss_weight=_field_loss_weights_for_recipe(args),
        temperature_bucket_loss_weight=0.15,
        tint_bucket_loss_weight=2.0,
        spread_loss_weight=None,
        exposure_scene_loss_weight=4.0,
        sign_wrong_penalty_weight=0.2,
        profile_name=args.profile_name,
        publish_dir=config.CHECKPOINTS_DIR,
        publish_version=None,
        no_publish=True,
        enable_progress_bar=True,
        on_epoch_complete=None,
        cancel_event=None,
    )
    summary = _train_profile_with_cuda_oom_retry(train_args)
    final_model = summary.get("final_model")
    if not final_model:
        raise RuntimeError("Training did not produce a final model checkpoint")
    quality_failures, quality_warnings = _foundation_quality_report(summary)
    _write_quality_gate_result(
        training_dir=training_dir,
        summary=summary,
        failures=quality_failures,
        warnings=quality_warnings,
    )
    if quality_failures and not args.allow_quality_gate_failure:
        formatted = "\n  - ".join(quality_failures)
        raise RuntimeError(
            "Refusing to promote foundation checkpoint because held-out metrics "
            "failed the quality gate:\n  - "
            f"{formatted}\n"
            "Inspect the run, add more data or tune the recipe, then rerun. "
            "Pass --allow-quality-gate-failure only after deliberate visual review."
        )
    if quality_warnings:
        formatted = "\n  - ".join(quality_warnings)
        print(
            "Warning: foundation checkpoint has quality warnings and needs "
            f"visual review:\n  - {formatted}",
            file=sys.stderr,
        )

    promoted = promote_foundation_checkpoint(
        source_ckpt=Path(final_model),
        display_name=args.profile_name,
        version_stem=args.version_stem,
        source_run_dir=run_dir,
    )

    print(f"Foundation checkpoint promoted: {promoted}")
    print(f"Foundation manifest:            {foundation_repo / 'foundation_manifest.json'}")
    print(f"Training run:                   {run_dir}")


def run_cli() -> int:
    try:
        main()
        return 0
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        log.exception("Unexpected foundation training failure")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_cli())
