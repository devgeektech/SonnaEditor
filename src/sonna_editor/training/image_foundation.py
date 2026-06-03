"""Image-to-image foundation training from RAW/DNG inputs to edited TIFF targets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from sonna_editor import config
from sonna_editor.data.extract import extract_preview
from sonna_editor.runtime import preferred_lightning_accelerator, supports_pinned_memory

FOUNDATION_IMAGE_TYPE = "image_to_image_v1"
TARGET_IMAGE_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
INPUT_IMAGE_EXTENSIONS = config.SUPPORTED_RAW_EXTENSIONS | TARGET_IMAGE_EXTENSIONS


@dataclass(frozen=True)
class ImagePair:
    """A source image and its edited target image."""

    source_path: Path
    target_path: Path


def find_image_pairs(source_dir: Path, target_dir: Path) -> list[ImagePair]:
    """Return source/target pairs matched by file stem.

    The FiveK-style layout stores DNG inputs separately from expert TIFF targets.
    Matching by stem keeps that format simple and avoids inventing labels.
    """
    source_dir = source_dir.expanduser()
    target_dir = target_dir.expanduser()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source image directory not found: {source_dir}")
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Target TIFF directory not found: {target_dir}")

    targets: dict[str, Path] = {}
    for path in sorted(target_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in TARGET_IMAGE_EXTENSIONS:
            targets.setdefault(path.stem.lower(), path)

    pairs: list[ImagePair] = []
    for source in sorted(source_dir.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in INPUT_IMAGE_EXTENSIONS:
            continue
        target = targets.get(source.stem.lower())
        if target is not None:
            pairs.append(ImagePair(source_path=source, target_path=target))
    return pairs


def split_image_pairs(
    pairs: list[ImagePair],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> tuple[list[ImagePair], list[ImagePair], list[ImagePair]]:
    """Deterministically split paired images into train/val/test sets."""
    if not pairs:
        raise ValueError("No image pairs found")
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to < 1")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(pairs), generator=generator).tolist()
    shuffled = [pairs[i] for i in order]

    n = len(shuffled)
    test_n = max(1, round(n * test_ratio)) if n >= 3 and test_ratio > 0 else 0
    val_n = max(1, round(n * val_ratio)) if n - test_n >= 3 and val_ratio > 0 else 0
    train_n = n - val_n - test_n
    if train_n <= 0:
        raise ValueError(
            f"Not enough image pairs for requested split: n={n}, val={val_n}, test={test_n}"
        )

    train = shuffled[:train_n]
    val = shuffled[train_n : train_n + val_n]
    test = shuffled[train_n + val_n :]
    return train, val, test


def write_image_pairs_manifest(path: Path, pairs: list[ImagePair]) -> None:
    """Write a JSONL manifest for reproducible paired-image training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "source_path": str(pair.source_path),
                "target_path": str(pair.target_path),
            },
            sort_keys=True,
        )
        for pair in pairs
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_image_pairs_manifest(path: Path) -> list[ImagePair]:
    """Read a JSONL paired-image manifest."""
    pairs: list[ImagePair] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        pairs.append(
            ImagePair(
                source_path=Path(payload["source_path"]),
                target_path=Path(payload["target_path"]),
            )
        )
    return pairs


def _load_source_image(path: Path, resolution: int) -> Image.Image:
    """Load RAW/DNG via rawpy preview extraction, or ordinary images via PIL."""
    if path.suffix.lower() in config.SUPPORTED_RAW_EXTENSIONS:
        return extract_preview(path, target_size=resolution).convert("RGB")
    return Image.open(path).convert("RGB")


def _load_target_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _resize_crop_to_tensor(image: Image.Image, resolution: int) -> torch.Tensor:
    image = TF.resize(image, resolution, antialias=True)
    image = TF.center_crop(image, [resolution, resolution])
    return TF.to_tensor(image)


class PairedImageDataset(Dataset):
    """Dataset for image-supervised foundation training."""

    def __init__(
        self,
        pairs: list[ImagePair],
        *,
        resolution: int,
        train: bool = False,
    ) -> None:
        if not pairs:
            raise ValueError("PairedImageDataset requires at least one pair")
        self._pairs = pairs
        self._resolution = resolution
        self._train = train

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self._pairs[index]
        try:
            source = _load_source_image(pair.source_path, self._resolution)
            target = _load_target_image(pair.target_path)
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Failed to load image pair: {pair}") from exc

        source_tensor = _resize_crop_to_tensor(source, self._resolution)
        target_tensor = _resize_crop_to_tensor(target, self._resolution)
        if self._train and torch.rand(()) < 0.5:
            source_tensor = torch.flip(source_tensor, dims=[2])
            target_tensor = torch.flip(target_tensor, dims=[2])

        return {
            "source": source_tensor,
            "target": target_tensor,
            "source_path": str(pair.source_path),
            "target_path": str(pair.target_path),
        }


class FoundationEnhancementModel(nn.Module):
    """ConvNeXt encoder plus lightweight decoder for RAW-preview to edited-image learning."""

    def __init__(self, *, pretrained_backbone: bool = True) -> None:
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained_backbone else None
        backbone = models.convnext_tiny(weights=weights)
        self.backbone_features = backbone.features
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(384, 192, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(192, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(96, 48, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(24, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone_features(image)
        output = self.decoder(features)
        if output.shape[-2:] != image.shape[-2:]:
            output = F.interpolate(
                output,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return output

    def save_checkpoint(
        self,
        path: Path,
        *,
        image_resolution: int,
        train_rows: int,
        val_rows: int,
        test_rows: int,
        metrics: dict[str, float],
    ) -> None:
        """Save an image-foundation checkpoint with Sonna-compatible backbone keys."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "foundation_type": FOUNDATION_IMAGE_TYPE,
                "model_state": self.state_dict(),
                "arch_config": {
                    "image_resolution": image_resolution,
                    "backbone": "convnext_tiny",
                    "foundation_type": FOUNDATION_IMAGE_TYPE,
                    "train_rows": train_rows,
                    "val_rows": val_rows,
                    "test_rows": test_rows,
                },
                "metrics": metrics,
            },
            path,
        )


def _ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return a small differentiable SSIM loss over RGB images."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(pred * pred, kernel_size=3, stride=1, padding=1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, kernel_size=3, stride=1, padding=1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, kernel_size=3, stride=1, padding=1) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    )
    return 1.0 - score.clamp(0.0, 1.0).mean()


class FoundationImageLightningModule(pl.LightningModule):
    """Lightning wrapper for paired-image foundation training."""

    def __init__(
        self,
        model: FoundationEnhancementModel,
        *,
        lr: float,
        weight_decay: float,
        l1_weight: float,
        ssim_weight: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.save_hyperparameters(ignore=["model"])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)

    def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        pred = self.model(batch["source"])
        target = batch["target"]
        l1 = F.l1_loss(pred, target)
        ssim = _ssim_loss(pred, target)
        loss = self.l1_weight * l1 + self.ssim_weight * ssim
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=stage != "train",
            on_step=False,
            on_epoch=True,
        )
        self.log(f"{stage}_l1", l1, prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{stage}_ssim_loss", ssim, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


def _loader(
    pairs: list[ImagePair],
    *,
    resolution: int,
    batch_size: int,
    num_workers: int,
    train: bool,
) -> DataLoader:
    return DataLoader(
        PairedImageDataset(pairs, resolution=resolution, train=train),
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=supports_pinned_memory(),
    )


def _best_model_state(best_ckpt: str) -> dict[str, torch.Tensor]:
    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    return {
        key.removeprefix("model."): value
        for key, value in ckpt["state_dict"].items()
        if key.startswith("model.")
    }


def train_image_foundation(
    *,
    source_dir: Path,
    target_dir: Path,
    output_dir: Path,
    profile_name: str,
    max_epochs: int,
    batch_size: int,
    workers: int,
    image_resolution: int,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.107,
    test_ratio: float = 0.139,
    l1_weight: float = 1.0,
    ssim_weight: float = 0.2,
    base_model_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Train an image-to-image foundation model and save `model.ckpt`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = find_image_pairs(source_dir, target_dir)
    train_pairs, val_pairs, test_pairs = split_image_pairs(
        pairs,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    dataset_dir = output_dir.parent / "image_pairs"
    write_image_pairs_manifest(dataset_dir / "train.jsonl", train_pairs)
    write_image_pairs_manifest(dataset_dir / "val.jsonl", val_pairs)
    write_image_pairs_manifest(dataset_dir / "test.jsonl", test_pairs)

    train_loader = _loader(
        train_pairs,
        resolution=image_resolution,
        batch_size=batch_size,
        num_workers=workers,
        train=True,
    )
    val_loader = _loader(
        val_pairs,
        resolution=image_resolution,
        batch_size=batch_size,
        num_workers=workers,
        train=False,
    ) if val_pairs else None
    test_loader = _loader(
        test_pairs,
        resolution=image_resolution,
        batch_size=batch_size,
        num_workers=workers,
        train=False,
    ) if test_pairs else None

    model = FoundationEnhancementModel(pretrained_backbone=base_model_checkpoint is None)
    if base_model_checkpoint is not None:
        ckpt = torch.load(base_model_checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", {})
        current_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in state.items()
            if key.startswith("backbone_features.")
            and key in current_state
            and current_state[key].shape == value.shape
        }
        model.load_state_dict(compatible_state, strict=False)

    module = FoundationImageLightningModule(
        model,
        lr=lr,
        weight_decay=weight_decay,
        l1_weight=l1_weight,
        ssim_weight=ssim_weight,
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="epoch={epoch:03d}-val_loss={val_loss:.4f}",
        monitor="val_loss" if val_loader is not None else "train_loss",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        accelerator=preferred_lightning_accelerator(),
        devices=1,
        precision="32-true",
        max_epochs=max_epochs,
        callbacks=[checkpoint],
        log_every_n_steps=max(1, min(10, math.ceil(len(train_pairs) / batch_size))),
    )
    trainer.fit(module, train_loader, val_loader)

    test_results: list[dict[str, Any]] = []
    if test_loader is not None:
        test_results = trainer.test(module, test_loader, ckpt_path=checkpoint.best_model_path or None)

    if checkpoint.best_model_path:
        module.model.load_state_dict(_best_model_state(checkpoint.best_model_path), strict=True)

    metrics = {
        "best_val_loss": float(checkpoint.best_model_score or 0.0),
        "test_loss": float(test_results[0].get("test_loss", 0.0)) if test_results else 0.0,
    }
    final_model = output_dir / "model.ckpt"
    module.model.save_checkpoint(
        final_model,
        image_resolution=image_resolution,
        train_rows=len(train_pairs),
        val_rows=len(val_pairs),
        test_rows=len(test_pairs),
        metrics=metrics,
    )

    sidecar = {
        "display_name": profile_name,
        "profile_type": "foundation_image_to_image",
        "foundation_type": FOUNDATION_IMAGE_TYPE,
        "checkpoint_path": str(final_model.resolve()),
        "date_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "resolution": image_resolution,
        "train_rows": len(train_pairs),
        "val_rows": len(val_pairs),
        "test_rows": len(test_pairs),
        "losses": {
            "l1_weight": l1_weight,
            "ssim_weight": ssim_weight,
            "lpips_weight": 0.0,
        },
        "notes": (
            "Image-supervised foundation checkpoint trained from paired source images "
            "and edited TIFF targets. It provides reusable ConvNeXt backbone weights; "
            "it is not a Lightroom slider-regression checkpoint."
        ),
    }
    (output_dir / "model.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    summary = {
        "foundation_type": FOUNDATION_IMAGE_TYPE,
        "final_model": str(final_model),
        "best_checkpoint": checkpoint.best_model_path,
        "metrics": metrics,
        "test_results": test_results[0] if test_results else {},
        "train_rows": len(train_pairs),
        "val_rows": len(val_pairs),
        "test_rows": len(test_pairs),
        "image_resolution": image_resolution,
        "base_model_checkpoint": str(base_model_checkpoint) if base_model_checkpoint else None,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
