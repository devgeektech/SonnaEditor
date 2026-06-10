from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import scripts.quick_diagnostic as quick_diagnostic


def test_print_median_baseline_comparison_reports_failed_fields(
    tmp_path: Path,
    capsys,
) -> None:
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    pd.DataFrame({
        "Exposure2012": [0.0, 0.0, 0.4],
        "Whites2012": [-10.0, -20.0, -30.0],
    }).to_parquet(train_path)
    pd.DataFrame({
        "Exposure2012": [0.3, 0.6],
        "Whites2012": [-35.0, -45.0],
    }).to_parquet(test_path)
    summary_path = tmp_path / "training_summary.json"
    summary = {
        "dataset": {
            "train_parquet": str(train_path),
            "test_parquet": str(test_path),
        },
        "test_per_field_mae": {
            "Exposure2012": 0.4,
            "Whites2012": 12.0,
        },
    }
    summary_path.write_text(json.dumps(summary))

    quick_diagnostic._print_median_baseline_comparison(
        summary=summary,
        summary_path=summary_path,
    )

    output = capsys.readouterr().out
    assert "TRAIN-MEDIAN BASELINE CHECK" in output
    assert "Exposure2012" in output
    assert "Whites2012" in output
    assert "Model MAE" in output
