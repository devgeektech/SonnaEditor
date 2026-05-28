"""GET /api/captures — wraps finetune.delta.analyse_deltas() for the UI."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter

from sonna_editor import config
from sonna_editor.api.models import CapturesResponse
from sonna_editor.finetune.delta import analyse_deltas

router = APIRouter()

_CAPTURES_PARQUET = config.CAPTURES_DIR / "captures.parquet"


def _empty_response() -> CapturesResponse:
    empty = analyse_deltas(pd.DataFrame())
    return CapturesResponse(captures_count=0, since=None, **empty)


def _earliest_iso(captures: pd.DataFrame) -> Optional[str]:
    if "capture_time" not in captures.columns:
        return None
    times = captures["capture_time"].dropna()
    if times.empty:
        return None
    try:
        parsed = pd.to_datetime(times, utc=True, errors="coerce").dropna()
        if parsed.empty:
            return None
        return parsed.min().isoformat().replace("+00:00", "Z")
    except (ValueError, TypeError):
        return None


def _captures_path() -> Path:
    """Resolved at call time so tests can monkeypatch config.CAPTURES_DIR."""
    return config.CAPTURES_DIR / "captures.parquet"


@router.get("/captures", response_model=CapturesResponse)
def captures() -> CapturesResponse:
    path = _captures_path()
    if not path.exists():
        return _empty_response()

    df = pd.read_parquet(path)
    if df.empty:
        return _empty_response()

    analysis = analyse_deltas(df)
    return CapturesResponse(
        captures_count=len(df),
        since=_earliest_iso(df),
        **analysis,
    )
