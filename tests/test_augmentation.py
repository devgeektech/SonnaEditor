from __future__ import annotations

import torch
import pytest
from PIL import Image
import numpy as np

from sonna_editor.config import IMAGE_RESOLUTION
from sonna_editor.model.augmentation import TrainingAugmentation, ValidationAugmentation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_uint8_tensor(H: int = 480, W: int = 640, C: int = 3) -> torch.Tensor:
    """Random uint8 image tensor in [C, H, W] layout."""
    return torch.randint(0, 256, (C, H, W), dtype=torch.uint8)


def _make_pil(H: int = 480, W: int = 640) -> Image.Image:
    arr = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
# TrainingAugmentation — output shape and dtype
# ---------------------------------------------------------------------------

def test_training_output_shape() -> None:
    aug = TrainingAugmentation()
    img = _make_uint8_tensor()
    out = aug(img)
    assert out.shape == (3, IMAGE_RESOLUTION, IMAGE_RESOLUTION)


def test_training_output_dtype_float32() -> None:
    aug = TrainingAugmentation()
    img = _make_uint8_tensor()
    out = aug(img)
    assert out.dtype == torch.float32


def test_training_output_range_zero_to_one() -> None:
    aug = TrainingAugmentation()
    img = _make_uint8_tensor()
    out = aug(img)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_training_accepts_square_input() -> None:
    aug = TrainingAugmentation()
    img = _make_uint8_tensor(H=512, W=512)
    out = aug(img)
    assert out.shape == (3, IMAGE_RESOLUTION, IMAGE_RESOLUTION)


def test_training_accepts_non_square_input() -> None:
    aug = TrainingAugmentation()
    img = _make_uint8_tensor(H=300, W=450)
    out = aug(img)
    assert out.shape == (3, IMAGE_RESOLUTION, IMAGE_RESOLUTION)


# ---------------------------------------------------------------------------
# TrainingAugmentation — randomness (two calls should not always match)
# ---------------------------------------------------------------------------

def test_training_is_stochastic() -> None:
    """Two forward passes on the same image should sometimes differ."""
    aug = TrainingAugmentation()
    img = _make_uint8_tensor(H=512, W=512)
    outputs = [aug(img) for _ in range(10)]
    # With p=0.5 flip + RandomResizedCrop at least some outputs should differ
    all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
    assert not all_same, "Training augmentation should be stochastic"


# ---------------------------------------------------------------------------
# ValidationAugmentation — output shape and dtype
# ---------------------------------------------------------------------------

def test_validation_output_shape() -> None:
    aug = ValidationAugmentation()
    img = _make_uint8_tensor()
    out = aug(img)
    assert out.shape == (3, IMAGE_RESOLUTION, IMAGE_RESOLUTION)


def test_validation_output_dtype_float32() -> None:
    aug = ValidationAugmentation()
    img = _make_uint8_tensor()
    out = aug(img)
    assert out.dtype == torch.float32


def test_validation_output_range_zero_to_one() -> None:
    aug = ValidationAugmentation()
    img = _make_uint8_tensor()
    out = aug(img)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_validation_accepts_square_input() -> None:
    aug = ValidationAugmentation()
    img = _make_uint8_tensor(H=512, W=512)
    out = aug(img)
    assert out.shape == (3, IMAGE_RESOLUTION, IMAGE_RESOLUTION)


def test_validation_accepts_non_square_input() -> None:
    aug = ValidationAugmentation()
    img = _make_uint8_tensor(H=300, W=450)
    out = aug(img)
    assert out.shape == (3, IMAGE_RESOLUTION, IMAGE_RESOLUTION)


# ---------------------------------------------------------------------------
# ValidationAugmentation — determinism
# ---------------------------------------------------------------------------

def test_validation_is_deterministic() -> None:
    """Same input must always produce identical output."""
    aug = ValidationAugmentation()
    img = _make_uint8_tensor(H=512, W=512)
    out1 = aug(img)
    out2 = aug(img)
    assert torch.allclose(out1, out2), "Validation augmentation must be deterministic"


def test_validation_exact_size_preserved() -> None:
    """A square image already at IMAGE_RESOLUTION passes through unchanged (dtype aside)."""
    aug = ValidationAugmentation()
    img = _make_uint8_tensor(H=IMAGE_RESOLUTION, W=IMAGE_RESOLUTION)
    out = aug(img)
    expected = img.to(torch.float32) / 255.0
    assert torch.allclose(out, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Target tensor is NEVER modified
# ---------------------------------------------------------------------------

def test_training_does_not_mutate_input() -> None:
    aug = TrainingAugmentation()
    img = _make_uint8_tensor()
    img_copy = img.clone()
    aug(img)
    assert torch.equal(img, img_copy), "TrainingAugmentation must not mutate input tensor"


def test_validation_does_not_mutate_input() -> None:
    aug = ValidationAugmentation()
    img = _make_uint8_tensor()
    img_copy = img.clone()
    aug(img)
    assert torch.equal(img, img_copy), "ValidationAugmentation must not mutate input tensor"


# ---------------------------------------------------------------------------
# Module interface — both classes are nn.Module subclasses
# ---------------------------------------------------------------------------

def test_training_is_nn_module() -> None:
    import torch.nn as nn
    assert isinstance(TrainingAugmentation(), nn.Module)


def test_validation_is_nn_module() -> None:
    import torch.nn as nn
    assert isinstance(ValidationAugmentation(), nn.Module)


def test_training_eval_mode_does_not_change_output_shape() -> None:
    aug = TrainingAugmentation()
    aug.eval()
    img = _make_uint8_tensor()
    out = aug(img)
    assert out.shape == (3, IMAGE_RESOLUTION, IMAGE_RESOLUTION)


def test_validation_eval_mode_is_still_deterministic() -> None:
    aug = ValidationAugmentation()
    aug.eval()
    img = _make_uint8_tensor(H=512, W=512)
    out1 = aug(img)
    out2 = aug(img)
    assert torch.allclose(out1, out2)
