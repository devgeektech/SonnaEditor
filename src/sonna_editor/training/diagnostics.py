from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from torch import nn
from torch.utils.data import DataLoader

from sonna_editor.model.architecture import SonnaEditor


@dataclass(frozen=True)
class ParameterBreakdownRow:
    name: str
    total: int
    trainable: int

    @property
    def frozen(self) -> int:
        return self.total - self.trainable


def parameter_counts(model: nn.Module) -> dict[str, int | float]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = total - trainable
    pct = (trainable / total * 100.0) if total else 0.0
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": pct,
    }


def trainable_parameter_breakdown(model: SonnaEditor) -> list[ParameterBreakdownRow]:
    rows: list[ParameterBreakdownRow] = []
    accounted: set[int] = set()

    def add_row(name: str, modules: Iterable[nn.Module]) -> None:
        params = []
        for module in modules:
            params.extend(module.parameters())
        ids = {id(param) for param in params}
        accounted.update(ids)
        total = sum(param.numel() for param in params)
        trainable = sum(param.numel() for param in params if param.requires_grad)
        rows.append(ParameterBreakdownRow(name=name, total=total, trainable=trainable))

    for idx, stage in enumerate(model.backbone_features):
        add_row(f"backbone_features.{idx}", [stage])
    add_row("backbone_norm", [model.backbone_norm])
    add_row("metadata_encoder.fusion_mlp", [model.metadata_encoder.fusion_mlp])

    metadata_children = [
        child
        for name, child in model.metadata_encoder.named_children()
        if name != "fusion_mlp"
    ]
    add_row("metadata_encoder.other", metadata_children)

    head_modules = [
        module
        for name, module in model.named_children()
        if name.endswith("_head")
    ]
    add_row("output_heads", head_modules)

    if hasattr(model, "wb_metadata_skip"):
        add_row("wb_metadata_skip", [model.wb_metadata_skip])

    remainder = [
        param
        for param in model.parameters()
        if id(param) not in accounted
    ]
    if remainder:
        rows.append(
            ParameterBreakdownRow(
                name="other",
                total=sum(param.numel() for param in remainder),
                trainable=sum(param.numel() for param in remainder if param.requires_grad),
            )
        )
    return rows


def backbone_freeze_summary(model: SonnaEditor) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for idx, stage in enumerate(model.backbone_features):
        children = list(stage.children())
        block_summary = []
        for block_idx, child in enumerate(children):
            params = list(child.parameters())
            if not params:
                continue
            total = sum(param.numel() for param in params)
            trainable = sum(param.numel() for param in params if param.requires_grad)
            block_summary.append({
                "block": block_idx,
                "total": total,
                "trainable": trainable,
                "state": "trainable" if trainable == total else "frozen" if trainable == 0 else "mixed",
            })
        params = list(stage.parameters())
        total = sum(param.numel() for param in params)
        trainable = sum(param.numel() for param in params if param.requires_grad)
        summary.append({
            "stage": idx,
            "total": total,
            "trainable": trainable,
            "state": "trainable" if trainable == total else "frozen" if trainable == 0 else "mixed",
            "blocks": block_summary,
        })
    norm_total = sum(param.numel() for param in model.backbone_norm.parameters())
    norm_trainable = sum(
        param.numel() for param in model.backbone_norm.parameters() if param.requires_grad
    )
    summary.append({
        "stage": "norm",
        "total": norm_total,
        "trainable": norm_trainable,
        "state": (
            "trainable"
            if norm_trainable == norm_total
            else "frozen"
            if norm_trainable == 0
            else "mixed"
        ),
        "blocks": [],
    })
    return summary


def estimate_optimizer_steps(
    *,
    batches_per_epoch: int,
    max_epochs: int,
    max_steps: int | None = None,
    limit_train_batches: int | float | None = None,
    accumulate_grad_batches: int = 1,
) -> int:
    effective_batches = batches_per_epoch
    if isinstance(limit_train_batches, int) and limit_train_batches > 0:
        effective_batches = min(effective_batches, limit_train_batches)
    elif isinstance(limit_train_batches, float) and 0.0 < limit_train_batches < 1.0:
        effective_batches = max(1, int(math.floor(effective_batches * limit_train_batches)))
    steps_per_epoch = math.ceil(effective_batches / max(1, accumulate_grad_batches))
    estimated = steps_per_epoch * max_epochs
    if max_steps is not None and max_steps > 0:
        estimated = min(estimated, max_steps)
    return estimated


def dataloader_diagnostics(loader: DataLoader) -> dict[str, Any]:
    sampler = getattr(loader, "sampler", None)
    batch_sampler = getattr(loader, "batch_sampler", None)
    return {
        "batches_per_epoch": len(loader),
        "sampler": type(sampler).__name__ if sampler is not None else None,
        "batch_sampler": type(batch_sampler).__name__ if batch_sampler is not None else None,
        "drop_last": bool(getattr(loader, "drop_last", False)),
    }


def format_parameter_count(value: int | float) -> str:
    if isinstance(value, float):
        return f"{value:.1f}"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)
