from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sonna_editor.config import SLIDER_FIELDS
from sonna_editor.data.xmp import LR_DEFAULTS
from sonna_editor.data.audit import (
    _count_unedited,
    _decide_status,
    _estimate_training_time,
    _find_outliers,
    _slider_stats,
    audit_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 50,
    exposure_values: list[float] | None = None,
    all_zero: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        sliders = {f: 0.0 for f in SLIDER_FIELDS}
        if not all_zero:
            # Initialise at full LR defaults (matches a real "untouched"
            # photo: many v1 fields default to non-zero — Sharpness=25,
            # LuminanceNoiseReductionDetail=50, ColorNoiseReduction=25,
            # ParametricSplits, PerspectiveScale=100, etc. — plus the v2
            # extension defaults). The all_zero=True branch deliberately
            # skips this to model a truly unedited photo for the
            # audit-detects-unedited test.
            sliders.update({k: v for k, v in LR_DEFAULTS.items() if k in SLIDER_FIELDS})
            # Simulate a typical edit: ~10 scalar sliders touched, leaving ~63 at zero (below threshold of 80)
            sliders["Exposure2012"] = (exposure_values[i] if exposure_values else rng.uniform(-2, 2))
            sliders["Contrast2012"] = rng.uniform(-30, 30)
            sliders["Highlights2012"] = rng.uniform(-60, 60)
            sliders["Shadows2012"] = rng.uniform(-60, 60)
            sliders["Whites2012"] = rng.uniform(-30, 30)
            sliders["Blacks2012"] = rng.uniform(-30, 30)
            sliders["Temperature"] = rng.uniform(4000, 7000)
            sliders["Tint"] = rng.uniform(-20, 20)
            sliders["Vibrance"] = rng.uniform(-30, 30)
            sliders["Saturation"] = rng.uniform(-20, 20)
        rows.append({
            "id": f"photo_{i:04d}",
            "shoot_id": f"shoot_{i // 5}",
            "raw_path": f"/fake/photo_{i:04d}.cr3",
            "thumbnail_path": f"/fake/thumb_{i:04d}.jpg",
            "iso": rng.choice([100, 200, 400, 800, 1600]),
            "capture_datetime": f"2024-03-{(i % 28) + 1:02d}T10:00:00",
            "camera_body": "Canon EOS R6",
            "camera_profile": "Adobe Standard",
            "histogram": b"\x00" * 16,
            **sliders,
        })
    return pd.DataFrame(rows)


def _write_parquet(df: pd.DataFrame, tmp_path: Path) -> Path:
    p = tmp_path / "dataset.parquet"
    df.to_parquet(p, index=False)
    return p


# ---------------------------------------------------------------------------
# _count_unedited
# ---------------------------------------------------------------------------

def test_count_unedited_all_zero() -> None:
    df = _make_df(n=5, all_zero=True)
    mask = _count_unedited(df)
    assert mask.all()


def test_count_unedited_none_zero() -> None:
    df = _make_df(n=10)
    mask = _count_unedited(df)
    assert not mask.any()


def test_count_unedited_partial() -> None:
    df = _make_df(n=4)
    df.loc[2, SLIDER_FIELDS] = 0.0  # force row 2 to all-zero
    mask = _count_unedited(df)
    assert mask[2]
    assert not mask[0]


# ---------------------------------------------------------------------------
# _find_outliers
# ---------------------------------------------------------------------------

def test_find_outliers_detects_extreme_value() -> None:
    df = _make_df(n=50)
    df.loc[0, "Exposure2012"] = 99.0  # obvious outlier
    outliers = _find_outliers(df)
    assert "Exposure2012" in outliers
    assert "photo_0000" in outliers["Exposure2012"]


def test_find_outliers_clean_data() -> None:
    df = _make_df(n=100)
    outliers = _find_outliers(df)
    assert "Exposure2012" not in outliers or len(outliers.get("Exposure2012", [])) == 0


def test_find_outliers_returns_ids() -> None:
    df = _make_df(n=50)
    df.loc[10, "Tint"] = 9999.0
    outliers = _find_outliers(df)
    assert "Tint" in outliers
    flagged = outliers["Tint"]
    assert all(isinstance(fid, str) for fid in flagged)


# ---------------------------------------------------------------------------
# _slider_stats
# ---------------------------------------------------------------------------

def test_slider_stats_shape() -> None:
    df = _make_df(n=20)
    stats = _slider_stats(df)
    assert stats.shape == (len(SLIDER_FIELDS), 4)
    assert list(stats.columns) == ["mean", "std", "min", "max"]


def test_slider_stats_index_matches_slider_fields() -> None:
    df = _make_df(n=10)
    stats = _slider_stats(df)
    assert list(stats.index) == SLIDER_FIELDS


# ---------------------------------------------------------------------------
# _decide_status
# ---------------------------------------------------------------------------

def test_decide_status_go() -> None:
    assert _decide_status(1000, 0.05, False) == "GO"


def test_decide_status_warn_too_few() -> None:
    assert _decide_status(200, 0.05, False) == "WARN"


def test_decide_status_warn_unedited() -> None:
    assert _decide_status(1000, 0.25, False) == "WARN"


def test_decide_status_warn_mixed_profiles() -> None:
    assert _decide_status(1000, 0.05, True) == "WARN"


def test_decide_status_stop_too_few() -> None:
    assert _decide_status(50, 0.05, False) == "STOP"


def test_decide_status_stop_unedited() -> None:
    assert _decide_status(1000, 0.85, False) == "STOP"


# ---------------------------------------------------------------------------
# _estimate_training_time
# ---------------------------------------------------------------------------

def test_estimate_training_time_returns_positive() -> None:
    minutes, label = _estimate_training_time(1000)
    assert minutes > 0
    assert isinstance(label, str)


def test_estimate_training_time_scales_with_photos() -> None:
    m1, _ = _estimate_training_time(500)
    m2, _ = _estimate_training_time(1000)
    assert m2 > m1


def test_estimate_training_time_large_dataset_uses_hours() -> None:
    _, label = _estimate_training_time(10000)
    assert "hr" in label


# ---------------------------------------------------------------------------
# audit_dataset (integration-style, no real photos needed)
# ---------------------------------------------------------------------------

def test_audit_returns_dict(tmp_path: Path) -> None:
    df = _make_df(n=600)
    parquet = _write_parquet(df, tmp_path)
    result = audit_dataset(parquet, tmp_path / "audit")
    assert isinstance(result, dict)


def test_audit_has_required_keys(tmp_path: Path) -> None:
    df = _make_df(n=600)
    parquet = _write_parquet(df, tmp_path)
    result = audit_dataset(parquet, tmp_path / "audit")
    required = {"status", "n_photos", "n_shoots", "n_unedited", "unedited_ratio",
                "n_outlier_sliders", "high_variance_sliders", "mixed_profiles",
                "training_minutes_estimate", "report_path"}
    assert required.issubset(result.keys())


def test_audit_writes_report(tmp_path: Path) -> None:
    df = _make_df(n=600)
    parquet = _write_parquet(df, tmp_path)
    result = audit_dataset(parquet, tmp_path / "audit")
    assert Path(result["report_path"]).exists()


def test_audit_report_has_sections(tmp_path: Path) -> None:
    df = _make_df(n=600)
    parquet = _write_parquet(df, tmp_path)
    result = audit_dataset(parquet, tmp_path / "audit")
    report = Path(result["report_path"]).read_text()
    for section in ["Summary", "Hardware Estimate", "Data Composition", "Slider Analysis",
                    "Quality Flags", "Recommendations"]:
        assert section in report, f"Missing section: {section}"


def test_audit_status_go_for_clean_data(tmp_path: Path) -> None:
    df = _make_df(n=600)
    parquet = _write_parquet(df, tmp_path)
    result = audit_dataset(parquet, tmp_path / "audit")
    assert result["status"] == "GO"


def test_audit_status_stop_too_few_photos(tmp_path: Path) -> None:
    df = _make_df(n=20)
    parquet = _write_parquet(df, tmp_path)
    result = audit_dataset(parquet, tmp_path / "audit")
    assert result["status"] == "STOP"


def test_audit_detects_unedited_photos(tmp_path: Path) -> None:
    df = _make_df(n=100, all_zero=True)
    parquet = _write_parquet(df, tmp_path)
    result = audit_dataset(parquet, tmp_path / "audit")
    assert result["n_unedited"] == 100


def test_audit_writes_plots_dir(tmp_path: Path) -> None:
    df = _make_df(n=600)
    parquet = _write_parquet(df, tmp_path)
    audit_dir = tmp_path / "audit"
    audit_dataset(parquet, audit_dir)
    assert (audit_dir / "plots").is_dir()


def test_audit_photo_count_matches(tmp_path: Path) -> None:
    df = _make_df(n=42)
    parquet = _write_parquet(df, tmp_path)
    result = audit_dataset(parquet, tmp_path / "audit")
    assert result["n_photos"] == 42
