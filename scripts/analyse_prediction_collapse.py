#!/usr/bin/env python
"""Analyse validation-set prediction collapse for a trained checkpoint.

Runs the model on a parquet split and reports per-slider prediction/target
mean, standard deviation, min, max, MAE, and std ratio. Low std ratios flag
sliders where the model has collapsed toward average values instead of adapting
per image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sonna_editor.model.architecture import SonnaEditor
from sonna_editor.model.augmentation import ValidationAugmentation
from sonna_editor.model.postprocess import postprocess_predictions
from sonna_editor.runtime import preferred_torch_device, supports_pinned_memory
from sonna_editor.slider_set import fields_for_version
from sonna_editor.training.datamodule import SonnaDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse prediction collapse on a validation split")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/audits/prediction_collapse.md"))
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=50, help="Minimum/default rows to analyse")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--collapse-ratio", type=float, default=0.10)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _format_float(value: float) -> str:
    if pd.isna(value):
        return "nan"
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    return f"{value:.4f}"


def _run_predictions(
    model: SonnaEditor,
    parquet: Path,
    *,
    limit: int,
    batch_size: int,
    device: str,
) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    df = pd.read_parquet(parquet)
    if limit > 0:
        df = df.head(min(limit, len(df))).copy()
    dataset = SonnaDataset(
        df,
        ValidationAugmentation(),
        model.registry,
        slider_set_version=model._slider_set_version,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=supports_pinned_memory(),
    )

    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, metadata, target in tqdm(loader, desc="Predicting", unit="batch"):
            batch_meta = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in metadata.items()
            }
            raw = model(images.to(device), batch_meta)
            preds.append(postprocess_predictions(raw).cpu())
            targets.append(target.cpu())
    return df, torch.cat(preds, dim=0), torch.cat(targets, dim=0)


def _summarise(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    fields: list[str],
    collapse_ratio: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, field in enumerate(fields):
        pred = predictions[:, idx].float()
        target = targets[:, idx].float()
        mask = ~torch.isnan(target)
        if int(mask.sum().item()) == 0:
            continue
        pred_valid = pred[mask]
        target_valid = target[mask]
        pred_std = float(torch.std(pred_valid, unbiased=False))
        target_std = float(torch.std(target_valid, unbiased=False))
        std_ratio = pred_std / target_std if target_std > 1e-8 else float("nan")
        mae = float(torch.mean(torch.abs(pred_valid - target_valid)))
        rows.append({
            "field": field,
            "n": int(mask.sum().item()),
            "target_mean": float(target_valid.mean()),
            "target_std": target_std,
            "target_min": float(target_valid.min()),
            "target_max": float(target_valid.max()),
            "pred_mean": float(pred_valid.mean()),
            "pred_std": pred_std,
            "pred_min": float(pred_valid.min()),
            "pred_max": float(pred_valid.max()),
            "std_ratio": std_ratio,
            "mae": mae,
            "collapsed": bool(target_std > 1e-8 and std_ratio < collapse_ratio),
        })
    return pd.DataFrame(rows).sort_values(
        ["collapsed", "std_ratio"],
        ascending=[False, True],
        na_position="last",
    )


def _write_markdown(
    output: Path,
    *,
    model_path: Path,
    parquet: Path,
    n_rows: int,
    summary: pd.DataFrame,
    collapse_ratio: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    collapsed = summary[summary["collapsed"]]
    lines = [
        "# Prediction Collapse Analysis",
        "",
        f"- Model: `{model_path}`",
        f"- Parquet: `{parquet}`",
        f"- Photos analysed: `{n_rows}`",
        f"- Collapse threshold: `std_ratio < {collapse_ratio:.2f}`",
        f"- Collapsed sliders: `{len(collapsed)}`",
        "",
        "## Collapsed Sliders",
        "",
        "| Field | Target range | Pred range | Target std | Pred std | Std ratio | MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in collapsed.iterrows():
        lines.append(
            "| {field} | {tmin} to {tmax} | {pmin} to {pmax} | {tstd} | {pstd} | {ratio} | {mae} |".format(
                field=row["field"],
                tmin=_format_float(float(row["target_min"])),
                tmax=_format_float(float(row["target_max"])),
                pmin=_format_float(float(row["pred_min"])),
                pmax=_format_float(float(row["pred_max"])),
                tstd=_format_float(float(row["target_std"])),
                pstd=_format_float(float(row["pred_std"])),
                ratio=_format_float(float(row["std_ratio"])),
                mae=_format_float(float(row["mae"])),
            )
        )
    lines.extend([
        "",
        "## Full Slider Summary",
        "",
        "| Field | N | Target mean | Pred mean | Target std | Pred std | Std ratio | MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in summary.iterrows():
        lines.append(
            "| {field} | {n} | {tm} | {pm} | {ts} | {ps} | {ratio} | {mae} |".format(
                field=row["field"],
                n=int(row["n"]),
                tm=_format_float(float(row["target_mean"])),
                pm=_format_float(float(row["pred_mean"])),
                ts=_format_float(float(row["target_std"])),
                ps=_format_float(float(row["pred_std"])),
                ratio=_format_float(float(row["std_ratio"])),
                mae=_format_float(float(row["mae"])),
            )
        )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()
    device = args.device or preferred_torch_device()
    model = SonnaEditor.from_checkpoint(args.model_path, device="cpu")
    fields = fields_for_version(model._slider_set_version)
    source_df, predictions, targets = _run_predictions(
        model,
        args.parquet,
        limit=args.limit,
        batch_size=args.batch_size,
        device=device,
    )
    summary = _summarise(predictions, targets, fields, args.collapse_ratio)
    _write_markdown(
        args.output,
        model_path=args.model_path,
        parquet=args.parquet,
        n_rows=len(source_df),
        summary=summary,
        collapse_ratio=args.collapse_ratio,
    )
    csv_output = args.csv_output or args.output.with_suffix(".csv")
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_output, index=False)
    print(json.dumps({
        "photos": len(source_df),
        "collapsed_sliders": int(summary["collapsed"].sum()),
        "report": str(args.output),
        "csv": str(csv_output),
    }, indent=2))


if __name__ == "__main__":
    main()
