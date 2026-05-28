#!/usr/bin/env python
"""Quick diagnostic: analyze training summary and config to explain current model performance."""
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    summary_path = PROJECT_ROOT / "v1_learning" / "training_summary_v1.2.0_full_production.json"
    if not summary_path.exists():
        print(f"Summary not found: {summary_path}")
        return
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    print("=" * 80)
    print("v1.2.0 FULL PRODUCTION MODEL DIAGNOSTIC")
    print("=" * 80)
    
    # Config
    cfg = summary.get("config", {})
    print(f"\nDataset:")
    print(f"  Train rows: {cfg.get('train_rows'):,}")
    print(f"  Val rows:   {cfg.get('val_rows'):,}")
    print(f"  Test rows:  {cfg.get('test_rows'):,}")
    print(f"  Image resolution: {cfg.get('image_resolution')}px")
    print(f"  Max epochs: {cfg.get('max_epochs')}")
    print(f"  Freeze backbone: {cfg.get('freeze_backbone_epochs')} epochs")
    
    # Loss config
    loss_cfg = summary.get("loss_settings", {})
    print(f"\nLoss weights:")
    print(f"  Temperature bucket: {loss_cfg.get('temperature_bucket_loss_weight')}")
    print(f"  Tint bucket: {loss_cfg.get('tint_bucket_loss_weight')}")
    print(f"  Sign wrong penalty: {loss_cfg.get('sign_wrong_penalty_weight')}")
    print(f"  Spread loss: {loss_cfg.get('spread_loss_weight')}")
    
    # Test results — KEY METRICS
    test_results = summary.get("test_results", {})
    print(f"\n{'CRITICAL METRICS':^80}")
    print(f"{'='*80}")
    
    metrics_priority = [
        ("Temperature (K)", "test_mae_temperature", "Should be <350K for good WB"),
        ("Tint", "test_mae_tint", "Not in summary; check training logs"),
        ("Exposure (stops)", "test_mae_exposure", "Should be <0.20 for good exp"),
        ("Shadows", "test_mae_shadows", "Tone curve proxy"),
        ("Highlights", "test_mae_highlights", "Tone curve proxy"),
        ("HSL Average", "test_mae_hsl_avg", "Color tuning quality"),
    ]
    
    for label, key, note in metrics_priority:
        val = test_results.get(key)
        if val is not None:
            status = "🔴 BAD" if (
                (key == "test_mae_temperature" and val > 500) or
                (key == "test_mae_exposure" and val > 0.25) or
                (key == "test_mae_hsl_avg" and val > 10)
            ) else "🟡 OK"
            print(f"  {label:30} {val:8.2f}   {status}   ({note})")
    
    # Training dynamics
    print(f"\n{'TRAINING DYNAMICS':^80}")
    print(f"{'='*80}")
    best_val = summary.get("best_val_loss", 0)
    test_loss = test_results.get("test_loss", 0)
    epochs = summary.get("epochs_trained", 0)
    halted_early = summary.get("halted_early", False)
    
    print(f"  Best val loss: {best_val:.6f}")
    print(f"  Test loss:     {test_loss:.6f}")
    print(f"  Epochs trained: {epochs}")
    print(f"  Early stopped:  {halted_early}")
    overfitting_ratio = test_loss / best_val if best_val > 0 else 0
    print(f"  Overfitting ratio (test/best_val): {overfitting_ratio:.3f}x")
    
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
    if cfg.get("image_resolution", 0) < 512:
        recs.append("4. Input resolution is 256px — retrain at 512px for better accuracy.")
    if loss_cfg.get("temperature_bucket_loss_weight", 0) < 0.5:
        recs.append("5. Temperature bucket weight low — bump from config.py TEMPERATURE_BUCKET_LOSS_WEIGHT.")
    
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
