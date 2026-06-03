#!/usr/bin/env python3
"""CLI for the continuous learning loop — analyse captured edits and optionally fine-tune.

Usage examples:
  # Analyse only (no training):
  uv run scripts/finetune_profile.py \\
    --base-model v1_learning/model-v1.0.1.ckpt \\
    --captures-dir /tmp/sonna_capture_test \\
    --original-train-parquet v1_learning/dataset/splits/train.parquet \\
    --val-parquet v1_learning/dataset/splits/val.parquet \\
    --dry-run

  # List available checkpoints:
  uv run scripts/finetune_profile.py --list-versions --output-dir v1_learning/

  # Fine-tune with defaults:
  uv run scripts/finetune_profile.py \\
    --base-model v1_learning/model-v1.0.1.ckpt \\
    --captures-dir /path/to/captures \\
    --original-train-parquet v1_learning/dataset/splits/train.parquet \\
    --val-parquet v1_learning/dataset/splits/val.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd


def _print_rule(char: str = "-", width: int = 60) -> None:
    print(char * width)


def _list_versions(output_dir: Path) -> None:
    pattern = re.compile(r"model-v(\d+)\.(\d+)\.(\d+)")
    ckpts = sorted(
        [p for p in output_dir.glob("model-v*.ckpt") if pattern.search(p.name)],
        key=lambda p: [int(x) for x in pattern.search(p.name).groups()],
    )
    if not ckpts:
        print(f"No versioned checkpoints found in {output_dir}")
        return

    _print_rule()
    print(f"{'Checkpoint':<35}  {'Date':<12}  {'val_loss':>10}  {'Status':<12}")
    _print_rule()
    for ckpt in ckpts:
        sidecar = ckpt.with_suffix(".json")
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text())
                date = meta.get("date_iso", "")[:10]
                val_loss = meta.get("ft_val_loss", float("nan"))
                status = meta.get("checkpoint_status", "unknown")
                val_str = f"{val_loss:.6f}" if not math.isnan(val_loss) else "  n/a   "
                print(f"{ckpt.name:<35}  {date:<12}  {val_str:>10}  {status:<12}")
            except Exception:
                print(f"{ckpt.name:<35}  (no sidecar metadata)")
        else:
            print(f"{ckpt.name:<35}  (no sidecar)")
    _print_rule()
    print()
    print("To use any version:")
    print("  process_shoot_model.py --model-checkpoint <path> --input-dir ... --output-dir ...")


def _print_delta_analysis(result: dict) -> None:
    _print_rule()
    print(f"Delta analysis — {result['n_photos']} captured photos")
    _print_rule()
    print(f"{'Field':<28}  {'n':>5}  {'mean_delta':>12}  {'abs_mean':>10}  {'std':>8}")
    _print_rule("-", 70)
    for field, v_str in result.get("most_adjusted_fields", []):
        stats = result["per_field"].get(field, {})
        n = stats.get("n_with_delta", 0)
        mean_d = stats.get("mean_delta", 0.0)
        abs_m = stats.get("abs_mean_delta", 0.0)
        std_d = stats.get("std_delta", 0.0)
        print(f"{field:<28}  {n:>5}  {mean_d:>+12.3f}  {abs_m:>10.3f}  {std_d:>8.3f}")
    _print_rule("-", 70)
    cov = result.get("metadata_coverage", {})
    if cov:
        cov_str = "  ".join(f"{k}={v*100:.0f}%" for k, v in cov.items())
        print(f"Metadata coverage: {cov_str}")
    n_corr = len(result.get("correlations", []))
    if n_corr:
        print(f"Significant metadata correlations: {n_corr}")
        for c in result["correlations"][:3]:
            print(f"  {c['field']} ~ {c['metadata_col']}: r={c['spearman_r']:+.3f}, p={c['p_value']:.4f}, n={c['n']}")
    print()


def _print_finetune_report(metrics: dict) -> None:
    bv = metrics["base_version"]
    fv = metrics["ft_version"]
    n_cap = metrics["n_capture_rows"]
    n_orig = metrics["n_original_rows"]
    improved = metrics["improved"]

    _print_rule("=")
    print("Fine-tuning complete")
    _print_rule("=")
    print(f"  Captures used:    {n_cap} photos")
    print(f"  Original rows:    {n_orig} photos")
    print(f"  Epochs trained:   {metrics['epochs_trained']} / {metrics.get('max_epochs', '?')}")
    print()

    key_fields = [
        ("val_loss",    metrics["base_val_loss"],  metrics["ft_val_loss"]),
        ("MAE Exposure", metrics["base_per_field_mae"].get("Exposure2012", float("nan")),
                         metrics["ft_per_field_mae"].get("Exposure2012", float("nan"))),
        ("MAE Temperature", metrics["base_per_field_mae"].get("Temperature", float("nan")),
                            metrics["ft_per_field_mae"].get("Temperature", float("nan"))),
        ("MAE Shadows",  metrics["base_per_field_mae"].get("Shadows2012", float("nan")),
                         metrics["ft_per_field_mae"].get("Shadows2012", float("nan"))),
        ("MAE HSL avg",  _hsl_avg(metrics["base_per_field_mae"]),
                         _hsl_avg(metrics["ft_per_field_mae"])),
    ]

    print(f"  Validation results (same hold-out as {bv}):")
    _print_rule("-", 70)
    print(f"  {'Metric':<20}  {bv:>12}  {fv:>12}  {'Change':>10}")
    _print_rule("-", 70)
    for label, base_v, ft_v in key_fields:
        if math.isnan(base_v) or math.isnan(ft_v):
            change_str = "   n/a"
        else:
            pct = (base_v - ft_v) / base_v * 100.0
            arrow = "✓" if pct > 0 else "✗"
            change_str = f"{pct:+.1f}% {arrow}"
        print(f"  {label:<20}  {_fmt(base_v):>12}  {_fmt(ft_v):>12}  {change_str:>10}")
    _print_rule("-", 70)
    print()

    ckpt_path = metrics["checkpoint_path"]
    status = metrics["checkpoint_status"]
    if improved:
        print(f"  NEW CHECKPOINT ({status}): {ckpt_path}")
    else:
        pct = metrics["improvement_pct"]
        print(f"  WARNING: Validation loss REGRESSED by {abs(pct):.1f}%")
        print(f"  CANDIDATE CHECKPOINT: {ckpt_path}")
    print()
    print(f"  To use:      process_shoot_model.py --model-checkpoint {ckpt_path}")
    print("  To roll back: use any .ckpt in the output dir with --model-checkpoint")
    _print_rule("=")


def _hsl_avg(per_field_mae: dict) -> float:
    from sonna_editor.config import SLIDER_FIELDS
    hsl = [per_field_mae.get(f, float("nan")) for f in SLIDER_FIELDS if "Adjustment" in f]
    vals = [v for v in hsl if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _fmt(v: float) -> str:
    if math.isnan(v):
        return "n/a"
    if abs(v) < 1:
        return f"{v:.6f}"
    return f"{v:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune the Sonna model on captured user edits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-model", type=Path, help="Base checkpoint to fine-tune from")
    parser.add_argument("--captures-dir", type=Path, help="Dir containing captures.parquet")
    parser.add_argument("--original-train-parquet", type=Path, help="Original training split")
    parser.add_argument("--val-parquet", type=Path, help="Validation split (same as original training)")
    parser.add_argument("--output-dir", type=Path, default=Path("v1_learning"), help="Where to save versioned checkpoint")
    parser.add_argument("--correction-weight", type=float, default=1.0, help="Sample weight for captured edits (default 1.0)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Analyse deltas only, no training")
    parser.add_argument("--auto-promote", action="store_true", help="Promote without prompting if val_loss improves")
    parser.add_argument("--list-versions", action="store_true", help="List checkpoints in output-dir and exit")
    args = parser.parse_args()

    if args.list_versions:
        _list_versions(args.output_dir)
        return

    # Validate required args for training/analysis
    required = [("--base-model", args.base_model),
                ("--captures-dir", args.captures_dir),
                ("--val-parquet", args.val_parquet)]
    if not args.dry_run:
        required.append(("--original-train-parquet", args.original_train_parquet))
    for flag, val in required:
        if val is None:
            parser.error(f"{flag} is required")

    captures_path = args.captures_dir / "captures.parquet"
    if not captures_path.exists():
        print(f"ERROR: {captures_path} not found. Run capture_user_edits() first.", file=sys.stderr)
        sys.exit(1)

    # --- Delta analysis ---
    from sonna_editor.finetune.delta import analyse_deltas

    captures = pd.read_parquet(captures_path)
    print(f"Loaded {len(captures)} captured photos from {captures_path}")
    print()

    delta_result = analyse_deltas(captures)
    _print_delta_analysis(delta_result)

    if args.dry_run:
        print("--dry-run: stopping before training.")
        return

    # --- Prepare combined dataset ---
    from sonna_editor.finetune.delta import prepare_finetune_dataset

    n_orig = len(pd.read_parquet(args.original_train_parquet))
    n_cap = len(captures)

    print(f"Building combined dataset ({n_orig} original + {n_cap} captured)...")
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_f:
        combined_path = Path(tmp_f.name)

    try:
        prepare_finetune_dataset(
            captures=captures,
            original_train_parquet=args.original_train_parquet,
            output_path=combined_path,
            weight_recent=args.correction_weight,
        )

        # --- Fine-tune ---
        from sonna_editor.finetune.retrain import finetune_model

        metrics = finetune_model(
            base_checkpoint=args.base_model,
            finetune_parquet=combined_path,
            val_parquet=args.val_parquet,
            output_dir=args.output_dir,
            n_capture_rows=n_cap,
            n_original_rows=n_orig,
            lr=args.lr,
            max_epochs=args.max_epochs,
            patience=args.patience,
            freeze_backbone_epochs=args.freeze_backbone_epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        metrics["max_epochs"] = args.max_epochs
    finally:
        combined_path.unlink(missing_ok=True)

    # --- Report ---
    _print_finetune_report(metrics)

    # --- Promotion decision ---
    improved = metrics["improved"]
    ft_label = metrics["ft_version"]

    if improved:
        if args.auto_promote:
            print(f"Auto-promoting {ft_label} (--auto-promote).")
        else:
            ans = input(f"Promote {ft_label}? [y/N]: ").strip().lower()
            if ans != "y":
                print("Not promoted. To use anyway:")
                print(f"  process_shoot_model.py --model-checkpoint {metrics['checkpoint_path']}")
                print(f"To roll back completely, continue using: {args.base_model}")
    else:
        pct = abs(metrics["improvement_pct"])
        print(f"Validation loss regressed by {pct:.1f}%. The candidate checkpoint is saved but NOT promoted.")
        ans = input(f"Accept {ft_label}-candidate anyway? [y/N]: ").strip().lower()
        if ans == "y":
            print(f"Accepted. Use: process_shoot_model.py --model-checkpoint {metrics['checkpoint_path']}")
        else:
            print(f"Rejected. Continuing with {args.base_model}")


if __name__ == "__main__":
    main()
