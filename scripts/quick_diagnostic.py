#!/usr/bin/env python
"""Quick diagnostic: analyze training summary and config to explain current model performance."""
import argparse
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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

    train_rows = cfg.get("train_rows") if isinstance(cfg.get("train_rows"), int) else summary.get("train_rows")
    val_rows = cfg.get("val_rows") if isinstance(cfg.get("val_rows"), int) else summary.get("val_rows")
    test_rows = cfg.get("test_rows") if isinstance(cfg.get("test_rows"), int) else summary.get("test_rows")

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
    print(f"  Spread loss:        {loss_cfg.get('spread_loss_weight', 'unknown')}")
    print(f"  Sign wrong penalty: {loss_cfg.get('sign_wrong_penalty_weight', 'unknown')}")
    
    # Test results — KEY METRICS
    test_results = summary.get("test_results", {})
    print(f"\n{'CRITICAL METRICS':^80}")
    print(f"{'='*80}")

    metrics_priority = [
        ("Temperature (K)", "test_mae_temperature", 500, "Should be <350K for good WB"),
        ("Tint", "test_mae_tint", None, "Check training logs for stability"),
        ("Exposure (stops)", "test_mae_exposure", 0.25, "Should be <0.20 for good exp"),
        ("Shadows", "test_mae_shadows", None, "Tone curve proxy"),
        ("Highlights", "test_mae_highlights", None, "Tone curve proxy"),
        ("Whites", "test_mae_whites", None, "Highlight detail quality"),
        ("Blacks", "test_mae_blacks", None, "Shadow detail quality"),
        ("Clarity", "test_mae_clarity", None, "Local contrast fidelity"),
        ("Vibrance", "test_mae_vibrance", None, "Color boost accuracy"),
        ("Saturation", "test_mae_saturation", None, "Saturation accuracy"),
        ("HSL Average", "test_mae_hsl_avg", 10, "Color tuning quality"),
        ("Test loss", "test_loss", None, "Overall model loss on test set"),
    ]

    printed_keys = set()
    for label, key, threshold, note in metrics_priority:
        val = test_results.get(key)
        if val is None:
            continue
        printed_keys.add(key)
        status = "🟡 OK"
        if threshold is not None and isinstance(val, (int, float)):
            status = "🔴 BAD" if val > threshold else "🟡 OK"
        print(f"  {label:30} {_fmt_float(val, 4):>8}   {status}   ({note})")

    extra_keys = [k for k in sorted(test_results) if k not in printed_keys]
    if extra_keys:
        print("\nAdditional test metrics:")
        for key in extra_keys:
            print(f"  {key:30} {_fmt_float(test_results[key], 4)}")

    # Training dynamics
    print(f"\n{'TRAINING DYNAMICS':^80}")
    print(f"{'='*80}")
    best_val = summary.get("best_val_loss")
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
    if test_results.get("test_mae_temperature", float('inf')) > 500:
        recs.append("1. Temperature error too high (>500K) — data quality or insufficient training signal.")
    if test_results.get("test_mae_exposure", float('inf')) > 0.25:
        recs.append("2. Exposure error >0.25 stops — consider higher resolution or more data.")
    if test_results.get("test_mae_hsl_avg", float('inf')) > 10:
        recs.append("3. HSL/color tuning weak — may need stronger loss weights or augmentation.")
    if isinstance(cfg.get("image_resolution"), int) and cfg.get("image_resolution") < 512:
        recs.append("4. Input resolution is below 512px — retrain at 512px or higher.")
    if isinstance(loss_cfg.get("temperature_bucket_loss_weight"), (int, float)) and loss_cfg.get("temperature_bucket_loss_weight") < 0.5:
        recs.append("5. Temperature bucket weight low — bump from config.py TEMPERATURE_BUCKET_LOSS_WEIGHT.")
    if summary.get("published_model") is None:
        recs.append("6. No published model path was found in the summary — confirm training/publish completed.")
    if not recs:
        recs.append("Model performance looks reasonable. Consider fine-tuning on specific shots or retraining with more data.")

    for rec in recs:
        print(f"  • {rec}")

    print(f"\n{'NEXT STEPS':^80}")
    print(f"{'='*80}")
    print("""
  Step 1: Run the training command (see below)
  Step 2: Monitor training with TensorBoard (optional)
  Step 3: Validate on new photos after training completes
  Step 4: Fine-tune if needed on specific styles
""")

if __name__ == "__main__":
    main()
