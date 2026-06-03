"""Tests for GET /api/captures."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient


def test_captures_empty_when_no_parquet(client: TestClient) -> None:
    resp = client.get("/api/captures")
    assert resp.status_code == 200
    data = resp.json()
    assert data["captures_count"] == 0
    assert data["since"] is None
    assert data["n_photos"] == 0
    assert data["per_field"] == {}
    assert data["correlations"] == []


def test_captures_wraps_analyse_deltas(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    parquet = isolated_paths["captures_dir"] / "captures.parquet"
    df = pd.DataFrame({
        "id": ["a", "b"],
        "capture_time": ["2026-05-01T10:00:00+00:00", "2026-05-02T10:00:00+00:00"],
        "deltas": ["{}", "{}"],
    })
    df.to_parquet(parquet, index=False)

    fake_analysis = {
        "n_photos": 2,
        "metadata_coverage": {"iso": 1.0},
        "per_field": {"Exposure2012": {"n_with_delta": 2, "mean_delta": 0.1,
                                        "abs_mean_delta": 0.1, "std_delta": 0.0,
                                        "min_delta": 0.1, "max_delta": 0.1}},
        "most_adjusted_fields": [["Exposure2012", 0.1]],
        "correlations": [],
        "filtered_field_deltas": {},
    }

    with patch("sonna_editor.api.routes.captures.analyse_deltas",
               return_value=fake_analysis) as mock_fn:
        resp = client.get("/api/captures")

    assert resp.status_code == 200
    mock_fn.assert_called_once()

    data = resp.json()
    assert data["captures_count"] == 2
    assert data["since"] == "2026-05-01T10:00:00Z"
    assert data["n_photos"] == 2
    assert data["per_field"]["Exposure2012"]["mean_delta"] == 0.1
    assert data["most_adjusted_fields"] == [["Exposure2012", 0.1]]
