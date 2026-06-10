#!/usr/bin/env python
"""Quick diagnostic: analyze training summary and config to explain current model performance."""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parent.parent


MetricSpec = tuple[str, str, float | None, float | None, str, str]
_FIELD_USABLE_LIMITS = {
    "Temperature": 350.0,
    "Tint": 8.0,
    "Exposure2012": 0.25,
    "Shadows2012": 8.0,
    "Highlights2012": 8.0,
    "Whites2012": 5.0,
    "Blacks2012": 5.0,
    "Clarity2012": 5.0,
    "Vibrance": 5.0,
    "Saturation": 5.0,
}

if hasattr(sys.stdout, "reconfigure"):
    cast(Any, sys.stdout).reconfigure(encoding="utf-8")


def _find_training_summaries() -> list[Path]:
    return sorted(PROJECT_ROOT.rglob("training_summary*.json"))


def _find_published_checkpoints() -> list[Path]:
    return sorted((PROJECT_ROOT / "v1_learning").glob("model-v*.ckpt"))


def _select_path(paths: list[Path], description: str) -> Path | None:
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]

    print(f"Found {len(paths)} {description}:")
    for index, path in enumerate(paths, start=1):
        print(f"  {index}. {path}")

    while True:
        choice = input(f"Select a {description} by number (1-{len(paths)}), or ENTER to cancel: ").strip()
        if choice == "":
            return None
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(paths):
                return paths[index - 1]
        print("Invalid choice. Try again.")


def _match_summary_for_checkpoint(ckpt_path: Path, summaries: list[Path]) -> list[Path]:
    matches: list[Path] = []
    version = None
    if ckpt_path.name.startswith("model-v") and ckpt_path.name.endswith(".ckpt"):
        version = ckpt_path.name[len("model-"):-len(".ckpt")]

    for summary_path in summaries:
        try:
            data = json.loads(summary_path.read_text())
        except Exception:
            continue

        published = data.get("published_model")
        final_model = data.get("final_model")
        if published:
            try:
                published_path = Path(published).resolve()
                if published_path == ckpt_path.resolve():
                    matches.append(summary_path)
                    continue
            except Exception:
                pass
        if final_model:
            try:
                final_path = Path(final_model).resolve()
                if final_path == ckpt_path.resolve():
                    matches.append(summary_path)
                    continue
            except Exception:
                pass

        if version and version in summary_path.name:
            matches.append(summary_path)
            continue
        if version and version in str(summary_path.parent):
            matches.append(summary_path)
    return matches


def _choose_summary(args: argparse.Namespace) -> Path | None:
    available_summaries = _find_training_summaries()
    selected_summary: Path | None = None

    if args.summary_path:
        selected_summary = Path(args.summary_path)
        if not selected_summary.exists():
            print(f"Summary not found: {selected_summary}")
            return None
        return selected_summary

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            return None
        print(f"Selected checkpoint: {ckpt_path}")
        matches = _match_summary_for_checkpoint(ckpt_path, available_summaries)
        if len(matches) == 1:
            print(f"Using summary matched to checkpoint: {matches[0]}")
            return matches[0]
        if matches:
            print("Multiple summary files matched the selected checkpoint:")
            return _select_path(matches, "summary file")
        if available_summaries:
            print("No summary directly matched the selected checkpoint. Please choose from available summaries:")
            return _select_path(available_summaries, "summary file")

    if not available_summaries:
        print("No training summary JSON files were found under the project tree.")
        print("Look for files named like training_summary.json or training_summary_*.json.")
        return None

    return _select_path(available_summaries, "training summary")


def _prompt_checkpoint_choice() -> Path | None:
    checkpoints = _find_published_checkpoints()
    if not checkpoints:
        print("No published checkpoints found in v1_learning/model-v*.ckpt.")
        return None
    if len(checkpoints) == 1:
        return checkpoints[0]
    return _select_path(checkpoints, "published checkpoint")


def _as_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _nested_int(payload: dict[str, Any], key: str) -> int | None:
    value = _as_int(payload.get(key))
    if value is not None:
        return value

    for section_name in ("dataset", "data", "config", "hparams"):
        section = payload.get(section_name)
        if isinstance(section, dict):
            value = _as_int(section.get(key))
            if value is not None:
                return value
    return None


def _resolve_path(value: Any, *, summary_path: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None

    candidate = Path(value)
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(PROJECT_ROOT / candidate)
        candidates.append(summary_path.parent / candidate)

    for path in candidates:
        if path.exists():
            return path
    return None


def _nested_path(payload: dict[str, Any], keys: tuple[str, ...], *, summary_path: Path) -> Path | None:
    for key in keys:
        path = _resolve_path(payload.get(key), summary_path=summary_path)
        if path is not None:
            return path

    for section_name in ("dataset", "data", "config", "hparams"):
        section = payload.get(section_name)
        if not isinstance(section, dict):
            continue
        for key in keys:
            path = _resolve_path(section.get(key), summary_path=summary_path)
            if path is not None:
                return path
    return None


def _parquet_row_count(path: Path) -> int | None:
    try:
        import pyarrow.parquet as pq

        metadata = pq.ParquetFile(path).metadata
        return int(metadata.num_rows)
    except Exception:
        return None


def _split_row_count(
    summary: dict[str, Any],
    summary_path: Path,
    split: str,
) -> int | None:
    row_count = _nested_int(summary, f"{split}_rows")
    if row_count is not None:
        return row_count

    parquet_path = _nested_path(
        summary,
        (f"{split}_parquet", f"{split}_path"),
        summary_path=summary_path,
    )
    if parquet_path is not None:
        row_count = _parquet_row_count(parquet_path)
        if row_count is not None:
            return row_count

    default_path = (
        PROJECT_ROOT
        / "data"
        / "training_workspace"
        / "sonna_personal_001_dataset"
        / "splits_v2_stratified"
        / f"{split}.parquet"
    )
    if default_path.exists():
        return _parquet_row_count(default_path)
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool)


def _finite_number(value: Any) -> float | None:
    if not _is_number(value):
        return None
    number = float(value)
    if number != number:
        return None
    return number


def _status_for_metric(value: Any, ideal: float | None, usable: float | None) -> str:
    if not _is_number(value):
        return "INFO"
    if usable is not None:
        return "🟢 OK" if value <= usable else "🔴 BAD"
    if ideal is not None:
        return "🟢 OK" if value <= ideal else "🔴 BAD"
    return "INFO"


def _field_norm_mae(field: str, value: float) -> float:
    if field == "Temperature":
        return value / 350.0
    try:
        from sonna_editor.config import SLIDER_RANGES

        lo, hi = SLIDER_RANGES[field]
    except Exception:
        return value
    span = max(abs(float(hi) - float(lo)), 1e-9)
    return value / span


def _print_all_parameter_mae(summary: dict[str, Any]) -> None:
    per_field = summary.get("test_per_field_mae")
    if not isinstance(per_field, dict) or not per_field:
        return

    rows: list[tuple[str, float, float, str]] = []
    failures: list[tuple[str, float, float, str]] = []
    for field, raw_value in per_field.items():
        value = _finite_number(raw_value)
        if value is None:
            continue
        limit = _FIELD_USABLE_LIMITS.get(field)
        status = "INFO"
        if limit is not None:
            status = "OK" if value <= limit else "BAD"
            if value > limit:
                failures.append((field, value, _field_norm_mae(field, value), status))
        rows.append((field, value, _field_norm_mae(field, value), status))

    if not rows:
        return

    print(f"\n{'ALL-PARAMETER MAE CHECK':^80}")
    print(f"{'='*80}")
    if failures:
        print("  Fields outside usable limits:")
        print(f"  {'Field':30} {'MAE':>10}   {'Limit':>10}")
        print(f"  {'-'*30} {'-'*10}   {'-'*10}")
        for field, value, _, _ in sorted(failures, key=lambda row: row[1], reverse=True):
            print(f"  {field:30} {value:10.4f}   {_FIELD_USABLE_LIMITS[field]:10.4f}")
    else:
        print("  No stored key-field MAE exceeds the current usable limits.")

    print("\n  Worst stored per-field MAE by normalized slider range:")
    print(f"  {'Field':30} {'MAE':>10}   {'Norm MAE':>10}   Status")
    print(f"  {'-'*30} {'-'*10}   {'-'*10}   {'-'*6}")
    for field, value, norm_mae, status in sorted(rows, key=lambda row: row[2], reverse=True)[:20]:
        print(f"  {field:30} {value:10.4f}   {norm_mae:10.4f}   {status}")


def _print_median_baseline_comparison(
    *,
    summary: dict[str, Any],
    summary_path: Path,
) -> None:
    per_field = summary.get("test_per_field_mae")
    if not isinstance(per_field, dict) or not per_field:
        return

    train_parquet = _nested_path(
        summary,
        ("train_parquet", "train_path"),
        summary_path=summary_path,
    )
    test_parquet = _nested_path(
        summary,
        ("test_parquet", "test_path"),
        summary_path=summary_path,
    )
    if train_parquet is None or test_parquet is None:
        return

    try:
        import pandas as pd
    except Exception:
        return

    try:
        train_df = pd.read_parquet(train_parquet)
        test_df = pd.read_parquet(test_parquet)
    except Exception as exc:
        print(f"\nMedian baseline comparison skipped: could not read parquet splits ({exc}).")
        return

    failing_fields = [
        field
        for field, limit in _FIELD_USABLE_LIMITS.items()
        if _finite_number(per_field.get(field)) is not None
        and _finite_number(per_field.get(field)) > limit
    ]
    if not failing_fields:
        return

    rows: list[dict[str, Any]] = []
    for field in failing_fields:
        if field not in train_df.columns or field not in test_df.columns:
            continue
        train_values = pd.to_numeric(train_df[field], errors="coerce")
        test_values = pd.to_numeric(test_df[field], errors="coerce")
        if field == "Temperature":
            train_values = train_values[train_values > 0]
            test_values = test_values[test_values > 0]
        train_values = train_values.dropna()
        test_values = test_values.dropna()
        if train_values.empty or test_values.empty:
            continue

        baseline = float(train_values.median())
        model_mae = _finite_number(per_field.get(field))
        if model_mae is None:
            continue
        baseline_mae = float((test_values - baseline).abs().mean())
        target_std = float(test_values.std(ddof=0)) if len(test_values) > 1 else math.nan
        improvement = (
            100.0 * (baseline_mae - model_mae) / baseline_mae
            if baseline_mae > 1e-9
            else math.nan
        )
        rows.append({
            "field": field,
            "n": int(len(test_values)),
            "target_std": target_std,
            "baseline_mae": baseline_mae,
            "model_mae": model_mae,
            "improvement": improvement,
        })

    if not rows:
        return

    print(f"\n{'TRAIN-MEDIAN BASELINE CHECK':^80}")
    print(f"{'='*80}")
    print("  Compares failed model MAE against a simple train-split median predictor.")
    print("  Positive improvement means the model is learning beyond a fixed average.")
    print(f"  {'Field':24} {'N':>6} {'Target std':>11} {'Median MAE':>11} {'Model MAE':>10} {'Improve':>9}")
    print(f"  {'-'*24} {'-'*6} {'-'*11} {'-'*11} {'-'*10} {'-'*9}")
    for row in sorted(rows, key=lambda item: item["model_mae"], reverse=True):
        improvement = row["improvement"]
        improvement_text = "unknown" if math.isnan(improvement) else f"{improvement:+.1f}%"
        print(
            f"  {row['field']:24} {row['n']:6d} "
            f"{row['target_std']:11.4f} {row['baseline_mae']:11.4f} "
            f"{row['model_mae']:10.4f} {improvement_text:>9}"
        )


def _target_text(ideal: float | None, usable: float | None, unit: str) -> str:
    if ideal is None and usable is None:
        return "lower is better"

    suffix = unit if unit else ""
    if ideal is not None and usable is not None and usable > ideal:
        return f"<{ideal:g}{suffix} ideal, <{usable:g}{suffix} usable"
    limit = ideal if ideal is not None else usable
    return f"<{limit:g}{suffix}"


def _is_foundation_summary(summary_path: Path, summary: dict[str, Any]) -> bool:
    path_text = str(summary_path).lower()
    if "foundation_runs" in path_text or "foundation" in summary_path.name.lower():
        return True

    hparams = summary.get("hparams")
    if isinstance(hparams, dict):
        base_checkpoint = str(hparams.get("base_model_checkpoint", "")).lower()
        return "foundation" in base_checkpoint
    return False


def _print_next_steps(
    *,
    is_foundation_run: bool,
    final_model: Any,
    published_model: Any,
    val_parquet: Path | None,
) -> None:
    print(f"\n{'NEXT STEPS':^80}")
    print(f"{'='*80}")

    if is_foundation_run:
        print("  1. Treat this as a hidden foundation-model result, not a frontend profile.")
        print("     No published model path is expected for foundation training runs.")
        print("  2. Confirm the active foundation manifest points to the checkpoint you want:")
        print("     SonnaEditorFoundation\\foundation_manifest.json")
        if final_model:
            print("  3. Run a collapse check before trusting the low MAE numbers:")
            print(
                "     uv run python scripts\\analyse_prediction_collapse.py "
                f"--model-path {final_model} --parquet <val.parquet>"
            )
        else:
            print("  3. Run collapse analysis once you know the model checkpoint path.")
        print("  4. Validate visually on a fresh shoot or use this foundation via Lite/Personal AI.")
        return

    if published_model:
        print("  1. Process a small fresh shoot with the published model.")
        print("  2. Open the XMPs in Lightroom and inspect exposure/WB on dark, mixed-light photos.")
        print("  3. Run collapse analysis if the output looks too averaged.")
        print("  4. Fine-tune only after real edited corrections are available.")
        return

    print("  1. This training run has a final model but no published frontend model path.")
    print("  2. If this was meant to create a Personal AI profile, rerun/publish the profile.")
    print("  3. If it was only an experiment, leave it unpublished and compare it to other runs.")
    if val_parquet:
        print("  4. Before publishing, run collapse analysis on the validation split.")
    else:
        print("  4. Before publishing, locate the validation parquet and run collapse analysis.")


def main():
    parser = argparse.ArgumentParser(description="Quick diagnostic on a training summary JSON.")
    parser.add_argument("--summary-path", help="Path to a training summary JSON file.")
    parser.add_argument("--ckpt", help="Optional published checkpoint to select before choosing a matching summary.")
    parser.add_argument("--list-checkpoints", action="store_true", help="List discovered published checkpoints and exit.")
    args = parser.parse_args()

    if args.list_checkpoints:
        checkpoints = _find_published_checkpoints()
        if not checkpoints:
            print("No published checkpoints were found in v1_learning/model-v*.ckpt.")
            return
        print("Published checkpoints found:")
        for path in checkpoints:
            print(f"  {path}")
        return

    summary_path = _choose_summary(args)
    if summary_path is None:
        return

    with open(summary_path) as f:
        summary = json.load(f)
    
    def _fmt_int(value):
        return f"{value:,}" if isinstance(value, int) else "unknown"

    def _fmt_float(value, decimals=2, unit=""):
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            return f"{value:.{decimals}f}{unit}"
        return "unknown"

    def _fmt_bool(value):
        if isinstance(value, bool):
            return str(value)
        return "unknown"

    cfg = summary.get("config") or summary.get("hparams") or {}
    hparams = summary.get("hparams", {})
    loss_cfg = summary.get("loss_settings", {}) or hparams

    train_rows = _split_row_count(summary, summary_path, "train")
    val_rows = _split_row_count(summary, summary_path, "val")
    test_rows = _split_row_count(summary, summary_path, "test")
    val_parquet = _nested_path(
        summary,
        ("val_parquet", "val_path"),
        summary_path=summary_path,
    )
    is_foundation_run = _is_foundation_summary(summary_path, summary)

    print("=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"Summary file: {summary_path}")
    print("\nDataset:")
    print(f"  Train rows: {_fmt_int(train_rows)}")
    print(f"  Val rows:   {_fmt_int(val_rows)}")
    print(f"  Test rows:  {_fmt_int(test_rows)}")
    print(f"  Image resolution: {_fmt_int(cfg.get('image_resolution'))}px")
    print(f"  Max epochs: {_fmt_int(cfg.get('max_epochs'))}")
    print(f"  Batch size: {_fmt_int(hparams.get('batch_size'))}")
    print(f"  LR: {_fmt_float(hparams.get('lr'), 6)}")
    print(f"  Freeze backbone: {_fmt_int(hparams.get('freeze_backbone_epochs'))} epochs")
    print(f"  Backbone strategy: {hparams.get('backbone_unfreeze_strategy', 'unknown')}")
    print(f"  Backbone layers:   {hparams.get('backbone_trainable_layers', 'unknown')}")
    print(f"  Slider set version: {hparams.get('slider_set_version', 'unknown')}")
    print(f"  WB metadata skip: {_fmt_bool(hparams.get('use_wb_metadata_skip'))}")
    print("\nModel files:")
    print(f"  Published model: {summary.get('published_model', 'unknown')}")
    print(f"  Final model:     {summary.get('final_model', 'unknown')}")
    print(f"  Best checkpoint: {summary.get('best_checkpoint', 'unknown')}")
    print("\nLoss weights:")
    print(f"  Temperature bucket: {loss_cfg.get('temperature_bucket_loss_weight', 'unknown')}")
    print(f"  Tint bucket:        {loss_cfg.get('tint_bucket_loss_weight', 'unknown')}")
    print(f"  Exposure weight:    {loss_cfg.get('exposure_weight', 'unknown')}")
    print(f"  Field overrides:    {loss_cfg.get('field_loss_weights') or 'none'}")
    print(f"  Spread loss:        {loss_cfg.get('spread_loss_weight', 'unknown')}")
    print(f"  Sign wrong penalty: {loss_cfg.get('sign_wrong_penalty_weight', 'unknown')}")
    
    # Test results - key metrics
    test_results = summary.get("test_results", {})
    best_val = summary.get("best_val_loss")
    print(f"\n{'CRITICAL METRICS':^80}")
    print(f"{'='*80}")

    metrics_priority: list[MetricSpec] = [
        (
            "Temperature (K)",
            "test_mae_temperature",
            250,
            350,
            "",
            "White balance accuracy",
        ),
        ("Tint", "test_mae_tint", 5, 8, "", "Tint accuracy"),
        (
            "Exposure (stops)",
            "test_mae_exposure",
            0.20,
            0.25,
            "",
            "Exposure accuracy",
        ),
        ("Shadows", "test_mae_shadows", 6, 8, "", "Tone-shape proxy"),
        ("Highlights", "test_mae_highlights", 6, 8, "", "Tone-shape proxy"),
        ("Whites", "test_mae_whites", 3, 5, "", "Highlight detail"),
        ("Blacks", "test_mae_blacks", 3, 5, "", "Shadow detail"),
        ("Clarity", "test_mae_clarity", 3, 5, "", "Local contrast"),
        ("Vibrance", "test_mae_vibrance", 3, 5, "", "Colour boost"),
        ("Saturation", "test_mae_saturation", 3, 5, "", "Saturation"),
        ("HSL Average", "test_mae_hsl_avg", 6, 10, "", "Colour tuning"),
        ("Test loss", "test_loss", None, None, "", "Compare with validation loss"),
    ]

    print(f"  {'Metric':30} {'Value':>10}   {'Status':7}   {'Recommended score':28} Note")
    print(f"  {'-'*30} {'-'*10}   {'-'*7}   {'-'*28} {'-'*18}")
    printed_keys = set()
    for label, key, ideal, usable, unit, note in metrics_priority:
        val = test_results.get(key)
        if val is None:
            continue
        printed_keys.add(key)
        if key == "test_loss" and _is_number(val) and _is_number(best_val):
            usable_loss = best_val * 1.25
            status = "🟢 OK" if val <= usable_loss else "🔴 BAD"
            target = f"<= {usable_loss:.4g} usable"
        else:
            status = _status_for_metric(val, ideal, usable)
            target = _target_text(ideal, usable, unit)
        print(f"  {label:30} {_fmt_float(val, 4):>10}   {status:7}   {target:28} {note}")

    extra_keys = [k for k in sorted(test_results) if k not in printed_keys]
    if extra_keys:
        print("\nAdditional test metrics:")
        for key in extra_keys:
            print(f"  {key:30} {_fmt_float(test_results[key], 4)}")

    _print_all_parameter_mae(summary)
    _print_median_baseline_comparison(
        summary=summary,
        summary_path=summary_path,
    )

    # Training dynamics
    print(f"\n{'TRAINING DYNAMICS':^80}")
    print(f"{'='*80}")
    test_loss = test_results.get("test_loss")
    epochs = summary.get("epochs_trained")
    halted_early = summary.get("halted_early", False)

    print(f"  Best val loss: {_fmt_float(best_val, 6)}")
    print(f"  Test loss:     {_fmt_float(test_loss, 6)}")
    print(f"  Epochs trained: {_fmt_int(epochs)}")
    print(f"  Early stopped:  {_fmt_bool(halted_early)}")
    if isinstance(best_val, (int, float)) and best_val > 0 and isinstance(test_loss, (int, float)):
        overfitting_ratio = test_loss / best_val
        print(f"  Overfitting ratio (test/best_val): {overfitting_ratio:.3f}x")
    else:
        print("  Overfitting ratio (test/best_val): unknown")

    # Recommendations
    print(f"\n{'RECOMMENDATIONS':^80}")
    print(f"{'='*80}")

    recs = []
    temp_mae = test_results.get("test_mae_temperature")
    exposure_mae = test_results.get("test_mae_exposure")
    hsl_mae = test_results.get("test_mae_hsl_avg")

    if _is_number(temp_mae) and temp_mae > 350:
        recs.append(
            "Temperature is above the usable range. Check WB labels first, then consider "
            "raising the temperature bucket weight and retraining."
        )
    elif _is_number(temp_mae):
        recs.append(
            "Temperature is in the green range. Do not change WB loss weights unless "
            "real-photo validation shows a WB problem."
        )

    if _is_number(exposure_mae) and exposure_mae > 0.25:
        recs.append(
            "Exposure is outside the usable range. Add more varied dark/bright scenes "
            "or retrain before publishing."
        )
    elif _is_number(exposure_mae) and exposure_mae > 0.20:
        recs.append(
            "Exposure is usable but just above the ideal <0.20 stop target. Spot-check "
            "low-light and backlit photos before trusting this run."
        )

    if _is_number(hsl_mae) and hsl_mae > 10:
        recs.append(
            "HSL colour tuning is weak. Audit colour labels and compare against another "
            "run before promotion."
        )
    elif _is_number(hsl_mae):
        recs.append("HSL colour tuning is comfortably within target.")

    image_resolution = cfg.get("image_resolution")
    if isinstance(image_resolution, int) and image_resolution < 512:
        recs.append("Input resolution is below 512px. Prefer 512px or higher for new v2 runs.")

    published_model = summary.get("published_model")
    final_model = summary.get("final_model")
    if published_model is None and is_foundation_run:
        recs.append(
            "No published model is normal here because foundation runs stay hidden from "
            "the frontend. Check foundation_manifest.json instead."
        )
    elif published_model is None:
        recs.append(
            "No published model path was found. If this should be a Personal AI profile, "
            "publish or rerun the profile training flow."
        )

    if isinstance(test_rows, int) and test_rows < 50:
        recs.append(
            f"The test split has only {test_rows} photos. Treat the numbers as a smoke "
            "check, then validate on a fresh real shoot."
        )

    if not recs:
        recs.append("Model performance looks reasonable. Validate visually before promotion.")

    for index, rec in enumerate(recs, start=1):
        print(f"  {index}. {rec}")

    _print_next_steps(
        is_foundation_run=is_foundation_run,
        final_model=final_model,
        published_model=published_model,
        val_parquet=val_parquet,
    )

if __name__ == "__main__":
    main()
