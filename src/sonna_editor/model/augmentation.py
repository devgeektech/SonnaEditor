from __future__ import annotations

from typing import Optional

import torch
from torchvision.transforms import v2

from sonna_editor.config import IMAGE_RESOLUTION


class TrainingAugmentation(torch.nn.Module):
    """Geometric + colour augmentation pipeline for training images.

    Applied to uint8 PIL images or tensors before conversion to float32.
    Target slider values are NEVER touched by this class.

    Pipeline order:
        1. RandomResizedCrop  — mild zoom only (scale ≥ 0.9) to avoid
           introducing significant exposure/colour shift from heavy crops
        2. RandomHorizontalFlip — safe; lighting is not horizontally symmetric
        3. ColorJitter — applied while dtype is still uint8, before ToDtype,
           so torchvision's integer-aware clipping is used correctly
        4. ToDtype(float32, scale=True) — normalises [0, 255] → [0.0, 1.0]
    """

    def __init__(self, resolution: Optional[int] = None) -> None:
        super().__init__()
        res = resolution if resolution is not None else IMAGE_RESOLUTION
        self._pipeline = v2.Compose([
            v2.RandomResizedCrop(
                res,
                scale=(0.9, 1.0),
                ratio=(1.0, 1.0),
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(
                brightness=0.4,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self._pipeline(image)


class ValidationAugmentation(torch.nn.Module):
    """Deterministic resize + centre crop for validation/inference images.

    No colour augmentation. No randomness. Produces a consistent view of
    each image so validation metrics are reproducible.

    Pipeline order:
        1. Resize  — short-edge to `resolution` (antialias=True)
        2. CenterCrop — square crop to `resolution` × `resolution`
        3. ToDtype(float32, scale=True) — normalises [0, 255] → [0.0, 1.0]

    Pass `resolution` explicitly for inference at a per-profile resolution
    (engine reads it from the profile's sidecar). Defaults to the global
    IMAGE_RESOLUTION at construction time when no override is given.
    """

    def __init__(self, resolution: Optional[int] = None) -> None:
        super().__init__()
        res = resolution if resolution is not None else IMAGE_RESOLUTION
        self._pipeline = v2.Compose([
            v2.Resize(res, antialias=True),
            v2.CenterCrop(res),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self._pipeline(image)
