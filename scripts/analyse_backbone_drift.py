#!/usr/bin/env python
"""Compare ConvNeXt backbone drift between two checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


_STAGE_RE = re.compile(r"^backbone_features\.(\d+)\.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure backbone_features weight drift between two checkpoints."
    )
    parser.add_argument("--foundation-checkpoint", required=True, type=Path)
    parser.add_argument("--personal-checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint payload: {path}")
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state = {
            key.removeprefix("model."): value
            for key, value in ckpt["state_dict"].items()
            if key.startswith("model.")
        }
    else:
        raise ValueError(f"Checkpoint has no model_state/state_dict: {path}")
    return {
        key: value.detach().float().cpu()
        for key, value in state.items()
        if isinstance(value, torch.Tensor) and key.startswith("backbone_features.")
    }


def _stage_for_key(key: str) -> str:
    match = _STAGE_RE.match(key)
    return match.group(1) if match else "unknown"


def _tensor_stats(before: torch.Tensor, after: torch.Tensor) -> dict[str, float]:
    delta = after - before
    before_norm = float(torch.linalg.vector_norm(before).item())
    delta_norm = float(torch.linalg.vector_norm(delta).item())
    relative = delta_norm / max(before_norm, 1e-12)
    flat_before = before.flatten()
    flat_after = after.flatten()
    denom = float(torch.linalg.vector_norm(flat_before).item()) * float(
        torch.linalg.vector_norm(flat_after).item()
    )
    cosine = (
        float(torch.dot(flat_before, flat_after).item()) / denom
        if denom > 0
        else math.nan
    )
    return {
        "before_norm": before_norm,
        "delta_norm": delta_norm,
        "relative_delta": relative,
        "cosine": cosine,
        "numel": float(before.numel()),
    }


def analyse_backbone_drift(
    foundation_checkpoint: Path,
    personal_checkpoint: Path,
) -> dict[str, Any]:
    foundation = _load_state(foundation_checkpoint)
    personal = _load_state(personal_checkpoint)
    common_keys = sorted(
        key
        for key in foundation.keys() & personal.keys()
        if foundation[key].shape == personal[key].shape
    )
    if not common_keys:
        raise ValueError("No compatible backbone_features tensors found")

    per_tensor = {
        key: _tensor_stats(foundation[key], personal[key])
        for key in common_keys
    }
    by_stage: dict[str, list[dict[str, float]]] = defaultdict(list)
    for key, stats in per_tensor.items():
        by_stage[_stage_for_key(key)].append(stats)

    stage_summary: dict[str, dict[str, float]] = {}
    for stage, stats_list in sorted(by_stage.items(), key=lambda item: item[0]):
        total_numel = sum(s["numel"] for s in stats_list)
        stage_summary[stage] = {
            "num_tensors": float(len(stats_list)),
            "numel": total_numel,
            "mean_relative_delta": sum(
                s["relative_delta"] * s["numel"] for s in stats_list
            ) / max(total_numel, 1.0),
            "mean_cosine": sum(
                s["cosine"] * s["numel"]
                for s in stats_list
                if not math.isnan(s["cosine"])
            ) / max(total_numel, 1.0),
        }

    return {
        "foundation_checkpoint": str(foundation_checkpoint),
        "personal_checkpoint": str(personal_checkpoint),
        "common_tensors": len(common_keys),
        "stage_summary": stage_summary,
        "worst_tensors": sorted(
            [
                {"name": key, **stats}
                for key, stats in per_tensor.items()
            ],
            key=lambda row: row["relative_delta"],
            reverse=True,
        )[:20],
    }


def _format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Backbone Drift Report",
        "",
        f"- Foundation checkpoint: `{report['foundation_checkpoint']}`",
        f"- Personal checkpoint: `{report['personal_checkpoint']}`",
        f"- Compatible backbone tensors: {report['common_tensors']}",
        "",
        "## Stage Summary",
        "",
        "| Stage | Tensors | Parameters | Mean relative delta | Mean cosine |",
        "|---:|---:|---:|---:|---:|",
    ]
    for stage, stats in report["stage_summary"].items():
        lines.append(
            "| "
            f"{stage} | {int(stats['num_tensors'])} | {int(stats['numel'])} | "
            f"{stats['mean_relative_delta']:.6f} | {stats['mean_cosine']:.6f} |"
        )
    lines.extend([
        "",
        "## Largest Tensor Drifts",
        "",
        "| Tensor | Relative delta | Cosine |",
        "|---|---:|---:|",
    ])
    for row in report["worst_tensors"]:
        lines.append(f"| `{row['name']}` | {row['relative_delta']:.6f} | {row['cosine']:.6f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    report = analyse_backbone_drift(
        args.foundation_checkpoint,
        args.personal_checkpoint,
    )
    if args.output is None:
        print(_format_markdown(report))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".json":
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    else:
        args.output.write_text(_format_markdown(report), encoding="utf-8")
    print(f"Backbone drift report written: {args.output}")


if __name__ == "__main__":
    main()
