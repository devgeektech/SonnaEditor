"""GET /api/health — readiness probe used by the Electron shell to wait for backend."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter

from sonna_editor import config
from sonna_editor.api.models import HealthResponse
from sonna_editor.runtime import preferred_torch_device

router = APIRouter()


def _detect_device() -> str:
    """Report the runtime's preferred PyTorch device for UI diagnostics."""
    return preferred_torch_device()


@lru_cache(maxsize=1)
def _git_sha() -> str:
    """Best-effort current commit sha. Falls back to 'unknown' if git isn't available."""
    repo_root = Path(__file__).resolve().parents[4]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        sha = result.stdout.strip()
        return sha or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _has_model_checkpoint() -> bool:
    if not config.CHECKPOINTS_DIR.is_dir():
        return False
    return any(config.CHECKPOINTS_DIR.glob("model-v*.ckpt"))


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_has_model_checkpoint(),
        device=_detect_device(),
        version=_git_sha(),
    )
