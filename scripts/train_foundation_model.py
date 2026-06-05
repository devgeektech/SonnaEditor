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
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from sonna_editor import config
from sonna_editor.foundation import (
    ensure_foundation_repo_layout,
    promote_foundation_checkpoint,
    resolve_foundation_checkpoint,
)
from sonna_editor.training.profile_runner import train_profile

log = logging.getLogger(__name__)


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
    except FileNotFoundError:
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

    splits_dir = _resolve_splits(args, run_dir)

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
        backbone_unfreeze_strategy="progressive",
        num_workers=args.workers,
        resume_from_checkpoint=None,
        base_model_checkpoint=base_foundation,
        slider_set_version=config.CURRENT_SLIDER_SET_VERSION,
        no_wb_metadata_skip=False,
        no_target_prior_init=False,
        image_resolution=args.image_resolution,
        temperature_weight=4.0,
        tint_weight=4.0,
        exposure_weight=5.0,
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

    promoted = promote_foundation_checkpoint(
        source_ckpt=Path(final_model),
        display_name=args.profile_name,
        version_stem=args.version_stem,
        source_run_dir=run_dir,
    )

    print(f"Foundation checkpoint promoted: {promoted}")
    print(f"Foundation manifest:            {foundation_repo / 'foundation_manifest.json'}")
    print(f"Training run:                   {run_dir}")


if __name__ == "__main__":
    main()
