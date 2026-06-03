#!/usr/bin/env python
"""Build/train/promote the Sonna foundation checkpoint.

This is the canonical path for creating the base checkpoint used by Lite
profile creation. It is intentionally separate from Personal AI profile
training: the output is promoted to the configured foundation repo, not to the
frontend profile directory.

Supported training modes:
- parameter-supervised: RAW+XMP or prepared parquet splits -> Lightroom sliders
- image-supervised: RAW/DNG/image inputs -> edited TIFF targets
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from sonna_editor import config
from sonna_editor.foundation import (
    ensure_foundation_repo_layout,
    promote_foundation_checkpoint,
    resolve_foundation_checkpoint,
)
from sonna_editor.training.image_foundation import train_image_foundation
from sonna_editor.training.profile_runner import train_profile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the foundation checkpoint from RAW+XMP data, prepared splits, "
            "or paired RAW/DNG -> edited TIFF images."
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
    source.add_argument(
        "--raw-image-dir",
        type=Path,
        help="Folder containing RAW/DNG/image inputs for image-to-image foundation training.",
    )
    parser.add_argument(
        "--target-tiff-dir",
        type=Path,
        help="Folder containing edited TIFF targets matched to --raw-image-dir by file stem.",
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
            "Standalone foundation model repo "
            f"(default: {config.FOUNDATION_REPO_DIR})."
        ),
    )
    parser.add_argument("--profile-name", default="Sonna Foundation")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--version-stem", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-resolution", type=int, default=config.IMAGE_RESOLUTION)
    parser.add_argument("--val-ratio", type=float, default=0.107)
    parser.add_argument("--test-ratio", type=float, default=0.139)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help=(
            "Start from pretrained/default weights instead of the active foundation "
            "checkpoint. By default each foundation run warm-starts from the active "
            "checkpoint and writes a new versioned checkpoint."
        ),
    )
    parser.add_argument("--init-git", action="store_true")
    return parser.parse_args()


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
    input_dir = args.raw_xmp_dir.expanduser()
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
        splits_dir = args.splits_dir.expanduser()
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
    if args.raw_image_dir is not None and args.target_tiff_dir is None:
        raise SystemExit("--target-tiff-dir is required with --raw-image-dir")
    if args.raw_image_dir is None and args.target_tiff_dir is not None:
        raise SystemExit("--target-tiff-dir can only be used with --raw-image-dir")

    foundation_repo = args.foundation_repo.expanduser()
    config.FOUNDATION_REPO_DIR = foundation_repo
    os.environ[config.FOUNDATION_REPO_ENV_VAR] = str(foundation_repo)
    ensure_foundation_repo_layout(initialise_git=args.init_git)

    workspace_dir = args.workspace_dir.expanduser()
    run_name = args.run_name or f"foundation-{_timestamp()}"
    run_dir = workspace_dir / "foundation_runs" / run_name
    training_dir = run_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    base_foundation = None if args.no_warm_start else _active_foundation_or_none()

    if args.raw_image_dir is not None:
        summary = train_image_foundation(
            source_dir=args.raw_image_dir,
            target_dir=args.target_tiff_dir,
            output_dir=training_dir,
            profile_name=args.profile_name,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            image_resolution=args.image_resolution,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            l1_weight=args.l1_weight,
            ssim_weight=args.ssim_weight,
            base_model_checkpoint=base_foundation,
        )
    else:
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
        summary = train_profile(train_args)
    final_model = summary.get("final_model")
    if not final_model:
        raise RuntimeError("Training did not produce a final model checkpoint")

    version_stem = args.version_stem or run_name
    promoted = promote_foundation_checkpoint(
        source_ckpt=Path(final_model),
        display_name=args.profile_name,
        version_stem=version_stem,
        source_run_dir=run_dir,
    )

    print(f"Foundation checkpoint promoted: {promoted}")
    print(f"Foundation manifest:            {foundation_repo / 'foundation_manifest.json'}")
    print(f"Training run:                   {run_dir}")


if __name__ == "__main__":
    main()
