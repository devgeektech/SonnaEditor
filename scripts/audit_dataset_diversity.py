#!/usr/bin/env python
"""Audit scene/edit diversity for a Sonna training parquet."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sonna_editor.config import SCENE_STAT_FIELDS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit training dataset diversity")
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/audits/dataset_diversity.md"))
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def _safe_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _approx_scene_stats_from_histogram(blob: bytes) -> dict[str, float]:
    try:
        hist = np.load(io.BytesIO(blob)).astype(np.float32).reshape(3, 32)
    except Exception:
        return {field: float("nan") for field in SCENE_STAT_FIELDS}
    brightness_hist = hist.mean(axis=0)
    total = float(brightness_hist.sum())
    if total <= 0:
        return {field: float("nan") for field in SCENE_STAT_FIELDS}
    brightness_hist = brightness_hist / total
    centers = (np.arange(32, dtype=np.float32) + 0.5) / 32.0
    cdf = np.cumsum(brightness_hist)
    mean = float((brightness_hist * centers).sum())
    median = float(centers[min(int(np.searchsorted(cdf, 0.5)), 31)])
    std = float(np.sqrt(np.maximum((brightness_hist * (centers - mean) ** 2).sum(), 0.0)))
    p5 = float(centers[min(int(np.searchsorted(cdf, 0.05)), 31)])
    p95 = float(centers[min(int(np.searchsorted(cdf, 0.95)), 31)])
    return {
        "mean_luminance": mean,
        "median_luminance": median,
        "luminance_std": std,
        "highlight_clip_pct": float(brightness_hist[-1]),
        "shadow_clip_pct": float(brightness_hist[0]),
        "dynamic_range": max(0.0, min(1.0, p95 - p5)),
    }


def _ensure_scene_stats(df: pd.DataFrame) -> pd.DataFrame:
    missing = [field for field in SCENE_STAT_FIELDS if field not in df.columns]
    if not missing:
        return df
    if "histogram" not in df.columns:
        for field in missing:
            df[field] = np.nan
        return df
    approx_rows = [_approx_scene_stats_from_histogram(blob) for blob in df["histogram"]]
    approx = pd.DataFrame(approx_rows)
    for field in missing:
        df[field] = approx[field]
    return df


def _count_categories(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    mean_lum = _safe_numeric(df, "mean_luminance")
    dyn = _safe_numeric(df, "dynamic_range")
    highlight = _safe_numeric(df, "highlight_clip_pct")
    shadow = _safe_numeric(df, "shadow_clip_pct")
    temp = _safe_numeric(df, "as_shot_temperature")

    exposure = _safe_numeric(df, "Exposure2012")
    contrast = _safe_numeric(df, "Contrast2012")
    highlights = _safe_numeric(df, "Highlights2012")
    shadows = _safe_numeric(df, "Shadows2012")

    return {
        "brightness": {
            "dark": int((mean_lum < 0.35).sum()),
            "balanced": int(((mean_lum >= 0.35) & (mean_lum <= 0.65)).sum()),
            "bright": int((mean_lum > 0.65).sum()),
        },
        "contrast": {
            "low_dynamic_range": int((dyn < 0.35).sum()),
            "medium_dynamic_range": int(((dyn >= 0.35) & (dyn <= 0.70)).sum()),
            "high_dynamic_range": int((dyn > 0.70).sum()),
            "highlight_clipped": int((highlight > 0.01).sum()),
            "shadow_clipped": int((shadow > 0.01).sum()),
        },
        "white_balance": {
            "warm_artificial_or_tungsten": int((temp < 4500).sum()),
            "daylight_neutral": int(((temp >= 4500) & (temp <= 6500)).sum()),
            "cool_shade_or_flash": int((temp > 6500).sum()),
            "missing_asshot": int(temp.isna().sum()),
        },
        "edit_targets": {
            "large_positive_exposure": int((exposure > 0.75).sum()),
            "large_negative_exposure": int((exposure < -0.75).sum()),
            "strong_contrast": int((contrast.abs() > 25).sum()),
            "strong_highlight_recovery": int((highlights < -35).sum()),
            "strong_shadow_lift": int((shadows > 35).sum()),
        },
    }


def _numeric_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for column in columns:
        values = _safe_numeric(df, column).dropna()
        if values.empty:
            continue
        summary[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "p05": float(values.quantile(0.05)),
            "median": float(values.median()),
            "p95": float(values.quantile(0.95)),
            "max": float(values.max()),
        }
    return summary


def _write_markdown(
    output: Path,
    *,
    parquet: Path,
    df: pd.DataFrame,
    categories: dict[str, dict[str, int]],
    summary: dict[str, dict[str, float]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    shoot_count = int(df["shoot_id"].nunique()) if "shoot_id" in df.columns else 0
    lines = [
        "# Dataset Diversity Audit",
        "",
        f"- Parquet: `{parquet}`",
        f"- Photos: `{len(df)}`",
        f"- Shoots: `{shoot_count}`",
        "",
        "## Scene Buckets",
        "",
    ]
    for section, counts in categories.items():
        lines.append(f"### {section.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Bucket | Count | Share |")
        lines.append("|---|---:|---:|")
        for bucket, count in counts.items():
            share = count / max(len(df), 1)
            lines.append(f"| {bucket} | {count} | {share:.1%} |")
        lines.append("")
    lines.extend([
        "## Numeric Summary",
        "",
        "| Field | Mean | Std | Min | P05 | Median | P95 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for field, stats in summary.items():
        lines.append(
            "| {field} | {mean:.4f} | {std:.4f} | {min:.4f} | {p05:.4f} | {median:.4f} | {p95:.4f} | {max:.4f} |".format(
                field=field,
                **stats,
            )
        )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()
    df = pd.read_parquet(args.parquet)
    df = _ensure_scene_stats(df)
    categories = _count_categories(df)
    summary = _numeric_summary(
        df,
        [
            *SCENE_STAT_FIELDS,
            "as_shot_temperature",
            "Exposure2012",
            "Contrast2012",
            "Highlights2012",
            "Shadows2012",
            "Whites2012",
            "Blacks2012",
            "Temperature",
            "Tint",
        ],
    )
    _write_markdown(
        args.output,
        parquet=args.parquet,
        df=df,
        categories=categories,
        summary=summary,
    )
    payload = {
        "parquet": str(args.parquet),
        "photos": len(df),
        "shoots": int(df["shoot_id"].nunique()) if "shoot_id" in df.columns else 0,
        "categories": categories,
        "summary": summary,
        "report": str(args.output),
    }
    json_output = args.json_output or args.output.with_suffix(".json")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "photos": payload["photos"],
        "shoots": payload["shoots"],
        "report": str(args.output),
        "json": str(json_output),
    }, indent=2))


if __name__ == "__main__":
    main()
