"""Runtime helpers for cross-platform device and dependency selection."""

from __future__ import annotations

import torch


def preferred_torch_device() -> str:
    """Return the best available PyTorch device on the current machine.

    Preference order is CUDA, then Apple MPS, then CPU. This keeps the app fast
    on workstation GPUs while still running correctly on ordinary Windows,
    macOS, and Linux development machines.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def preferred_lightning_accelerator() -> str:
    """Return the matching PyTorch Lightning accelerator name.

    Lightning accepts `"cuda"`, `"mps"`, and `"cpu"` accelerators directly, so
    we mirror `preferred_torch_device()` to avoid separate platform branches in
    training scripts.
    """
    return preferred_torch_device()


def supports_pinned_memory() -> bool:
    """Return True only when DataLoader pinned memory is useful and supported."""
    return preferred_torch_device() == "cuda"
