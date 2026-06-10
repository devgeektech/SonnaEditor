from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from sonna_editor.config import SCENE_STAT_FIELDS, SLIDER_FIELDS
from sonna_editor.data.dataset import (
    _bytes_to_histogram,
    _derive_shoot_id,
    _file_id,
    _find_pairs,
    _histogram_to_bytes,
    _process_pair,
    build_dataset,
    load_dataset,
    save_split,
    split_dataset,
)

FIXTURE_RAW = Path(__file__).parent / "fixtures" / "sample.cr3"
FIXTURE_XMP = Path(__file__).parent / "fixtures" / "sample.xmp"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_extract_result(raw_path: Path, xmp_path: Path | None = None) -> dict:
    img = Image.new("RGB", (384, 256), color=(100, 150, 200))
    hist = np.zeros((3, 32), dtype=np.float32)
    hist[:, 10] = 1.0
    sliders = {f: 0.0 for f in SLIDER_FIELDS}
    sliders["Exposure2012"] = 1.0
    sliders["Temperature"] = 5200.0
    return {
        "raw_path": str(raw_path),
        "xmp_path": str(xmp_path) if xmp_path else None,
        "preview": img,
        "histogram": hist,
        "scene_stats": {
            "mean_luminance": 0.45,
            "median_luminance": 0.44,
            "luminance_std": 0.18,
            "highlight_clip_pct": 0.01,
            "shadow_clip_pct": 0.02,
            "dynamic_range": 0.65,
        },
        "iso": 200,
        "shutter_speed": 1 / 125,
        "aperture": 2.8,
        "focal_length": 50.0,
        "lens_model": "RF 50mm",
        "camera_body": "Canon EOS R6",
        "capture_datetime": datetime(2024, 3, 15, 10, 30),
        "exposure_compensation": 0.0,
        "white_balance_preset": "Auto",
        "camera_profile": "Adobe Standard",
        "width": 5472,
        "height": 3648,
        "sliders": sliders,
    }


def _make_fake_pair(tmp_path: Path, name: str = "photo") -> tuple[Path, Path]:
    raw = tmp_path / f"{name}.cr3"
    xmp = tmp_path / f"{name}.xmp"
    raw.touch()
    xmp.touch()
    return raw, xmp


# ---------------------------------------------------------------------------
# _file_id
# ---------------------------------------------------------------------------

def test_file_id_is_deterministic(tmp_path: Path) -> None:
    raw = tmp_path / "test.cr3"
    raw.touch()
    assert _file_id(raw) == _file_id(raw)


def test_file_id_differs_for_different_paths(tmp_path: Path) -> None:
    a = tmp_path / "a.cr3"
    b = tmp_path / "b.cr3"
    a.touch()
    b.touch()
    assert _file_id(a) != _file_id(b)


def test_file_id_is_64_hex_chars(tmp_path: Path) -> None:
    raw = tmp_path / "test.cr3"
    raw.touch()
    fid = _file_id(raw)
    assert len(fid) == 64
    assert all(c in "0123456789abcdef" for c in fid)


# ---------------------------------------------------------------------------
# _find_pairs
# ---------------------------------------------------------------------------

def test_find_pairs_finds_raw_with_xmp(tmp_path: Path) -> None:
    raw = tmp_path / "photo.cr3"
    xmp = tmp_path / "photo.xmp"
    raw.touch()
    xmp.touch()
    pairs = _find_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0][0] == raw
    assert pairs[0][1] == xmp


def test_find_pairs_raw_without_xmp(tmp_path: Path) -> None:
    raw = tmp_path / "photo.nef"
    raw.touch()
    pairs = _find_pairs(tmp_path)
    assert pairs == []


def test_find_pairs_ignores_non_raw(tmp_path: Path) -> None:
    (tmp_path / "photo.jpg").touch()
    (tmp_path / "doc.pdf").touch()
    pairs = _find_pairs(tmp_path)
    assert pairs == []


def test_find_pairs_recursive(tmp_path: Path) -> None:
    sub = tmp_path / "2024" / "march"
    sub.mkdir(parents=True)
    raw = sub / "shot.cr3"
    raw.touch()
    (sub / "shot.xmp").touch()
    pairs = _find_pairs(tmp_path)
    assert len(pairs) == 1


def test_find_pairs_multiple_formats(tmp_path: Path) -> None:
    for ext in [".cr3", ".nef", ".arw"]:
        raw = tmp_path / f"photo{ext}"
        raw.touch()
        (tmp_path / "photo.xmp").touch()
    pairs = _find_pairs(tmp_path)
    assert len(pairs) == 3


# ---------------------------------------------------------------------------
# Histogram serialisation
# ---------------------------------------------------------------------------

def test_histogram_round_trip() -> None:
    hist = np.random.rand(3, 32).astype(np.float32)
    data = _histogram_to_bytes(hist)
    restored = _bytes_to_histogram(data)
    np.testing.assert_array_almost_equal(hist, restored)


def test_histogram_bytes_is_bytes() -> None:
    hist = np.zeros((3, 32), dtype=np.float32)
    assert isinstance(_histogram_to_bytes(hist), bytes)


# ---------------------------------------------------------------------------
# _derive_shoot_id
# ---------------------------------------------------------------------------

def test_shoot_id_same_shoot(tmp_path: Path) -> None:
    dt1 = datetime(2024, 3, 15, 9, 0)
    dt2 = datetime(2024, 3, 15, 11, 30)  # 2.5 hrs later, same 12-hr window
    assert _derive_shoot_id(dt1, "R6") == _derive_shoot_id(dt2, "R6")


def test_shoot_id_different_shoot(tmp_path: Path) -> None:
    dt1 = datetime(2024, 3, 15, 1, 0)
    dt2 = datetime(2024, 3, 15, 14, 0)  # 13 hrs later — different window
    assert _derive_shoot_id(dt1, "R6") != _derive_shoot_id(dt2, "R6")


def test_shoot_id_different_camera_same_time(tmp_path: Path) -> None:
    dt = datetime(2024, 3, 15, 10, 0)
    assert _derive_shoot_id(dt, "R6") != _derive_shoot_id(dt, "R5")


def test_shoot_id_none_datetime(tmp_path: Path) -> None:
    sid = _derive_shoot_id(None, "R6")
    assert sid.startswith("unknown_")


def test_shoot_id_offset_aware_datetime(tmp_path: Path) -> None:
    aware_dt = datetime(2024, 3, 15, 10, 0, tzinfo=timezone.utc)
    assert _derive_shoot_id(aware_dt, "R6") == _derive_shoot_id(
        datetime(2024, 3, 15, 11, 0, tzinfo=timezone.utc), "R6"
    )


# ---------------------------------------------------------------------------
# _process_pair (mocked extract)
# ---------------------------------------------------------------------------

def test_process_pair_returns_row(tmp_path: Path) -> None:
    raw, xmp = _make_fake_pair(tmp_path)
    thumb_dir = tmp_path / "thumbs"
    thumb_dir.mkdir()

    with patch(
        "sonna_editor.data.dataset.extract_all",
        return_value=_make_fake_extract_result(raw, xmp),
    ):
        row = _process_pair((raw, xmp, "test_profile", thumb_dir))

    assert row is not None
    assert row["profile"] == "test_profile"
    assert row["Exposure2012"] == pytest.approx(1.0)
    assert row["Temperature"] == pytest.approx(5200.0)
    assert isinstance(row["histogram"], bytes)
    assert row["mean_luminance"] == pytest.approx(0.45)


def test_process_pair_saves_thumbnail(tmp_path: Path) -> None:
    raw, xmp = _make_fake_pair(tmp_path)
    thumb_dir = tmp_path / "thumbs"
    thumb_dir.mkdir()

    with patch(
        "sonna_editor.data.dataset.extract_all",
        return_value=_make_fake_extract_result(raw, xmp),
    ):
        row = _process_pair((raw, xmp, "p", thumb_dir))

    thumb_path = Path(row["thumbnail_path"])
    assert thumb_path.exists()
    img = Image.open(thumb_path)
    assert img.format == "JPEG"


def test_process_pair_returns_none_on_failure(tmp_path: Path) -> None:
    raw, xmp = _make_fake_pair(tmp_path)
    thumb_dir = tmp_path / "thumbs"
    thumb_dir.mkdir()

    with patch(
        "sonna_editor.data.dataset.extract_all",
        side_effect=RuntimeError("extraction failed"),
    ):
        row = _process_pair((raw, xmp, "p", thumb_dir))

    assert row is None


def test_process_pair_all_slider_fields_present(tmp_path: Path) -> None:
    raw, xmp = _make_fake_pair(tmp_path)
    thumb_dir = tmp_path / "thumbs"
    thumb_dir.mkdir()

    with patch(
        "sonna_editor.data.dataset.extract_all",
        return_value=_make_fake_extract_result(raw, xmp),
    ):
        row = _process_pair((raw, xmp, "p", thumb_dir))

    for field in SLIDER_FIELDS:
        assert field in row, f"Missing slider field: {field}"


def test_process_pair_accepts_timezone_aware_capture_datetime_string(tmp_path: Path) -> None:
    raw, xmp = _make_fake_pair(tmp_path)
    thumb_dir = tmp_path / "thumbs"
    thumb_dir.mkdir()
    extracted = _make_fake_extract_result(raw, xmp)
    extracted["capture_datetime"] = "2024-03-15T23:30:00+13:00"

    with patch("sonna_editor.data.dataset.extract_all", return_value=extracted):
        row = _process_pair((raw, xmp, "p", thumb_dir))

    assert row is not None
    assert row["shoot_id"] == _derive_shoot_id(
        datetime(2024, 3, 15, 10, 30, tzinfo=timezone.utc),
        "Canon EOS R6",
    )
    assert row["capture_datetime"] == "2024-03-15T23:30:00+13:00"


# ---------------------------------------------------------------------------
# build_dataset (mocked, synthetic)
# ---------------------------------------------------------------------------

def _make_synthetic_input(tmp_path: Path, n: int = 3) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for i in range(n):
        (input_dir / f"photo_{i:03d}.cr3").touch()
        (input_dir / f"photo_{i:03d}.xmp").touch()
    return input_dir


def _fake_extract(raw_path, xmp_path=None):
    import hashlib
    seed = int(hashlib.md5(str(raw_path).encode()).hexdigest()[:8], 16)
    dt = datetime(2024, 1, int(seed % 28) + 1, int(seed % 12) + 1, 0)
    result = _make_fake_extract_result(Path(raw_path), Path(xmp_path) if xmp_path else None)
    result["capture_datetime"] = dt
    return result


def test_build_dataset_returns_dataframe(tmp_path: Path) -> None:
    input_dir = _make_synthetic_input(tmp_path, n=3)

    with patch("sonna_editor.data.dataset.extract_all", side_effect=_fake_extract):
        df = build_dataset(
            input_dir=input_dir,
            output_path=tmp_path / "out.parquet",
            profile_name="test",
            thumbnail_dir=tmp_path / "thumbs",
            max_workers=1,
        )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_build_dataset_has_expected_columns(tmp_path: Path) -> None:
    input_dir = _make_synthetic_input(tmp_path, n=2)

    with patch("sonna_editor.data.dataset.extract_all", side_effect=_fake_extract):
        df = build_dataset(
            input_dir=input_dir,
            output_path=tmp_path / "out.parquet",
            profile_name="test",
            thumbnail_dir=tmp_path / "thumbs",
            max_workers=1,
        )

    required = {"id", "profile", "raw_path", "thumbnail_path", "shoot_id", "histogram"}
    required |= set(SLIDER_FIELDS)
    required |= {"as_shot_temperature", "as_shot_tint"}
    required |= set(SCENE_STAT_FIELDS)
    assert required.issubset(df.columns)


def test_build_dataset_writes_parquet(tmp_path: Path) -> None:
    input_dir = _make_synthetic_input(tmp_path, n=2)
    out = tmp_path / "out.parquet"

    with patch("sonna_editor.data.dataset.extract_all", side_effect=_fake_extract):
        build_dataset(input_dir, out, "test", tmp_path / "thumbs", max_workers=1)

    assert out.exists()
    assert out.stat().st_size > 0


def test_build_dataset_empty_dir_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No RAW files with matching XMP sidecars"):
        build_dataset(empty, tmp_path / "out.parquet", "test", tmp_path / "thumbs")


def test_build_dataset_skips_rows_without_xmp(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    with open(input_dir / "with_xmp.cr3", "wb") as f:
        f.write(b"raw")
    with open(input_dir / "with_xmp.xmp", "wb") as f:
        f.write(b"xmp")
    with open(input_dir / "missing_xmp.nef", "wb") as f:
        f.write(b"raw")

    with patch("sonna_editor.data.dataset.extract_all", side_effect=_fake_extract):
        df = build_dataset(
            input_dir=input_dir,
            output_path=tmp_path / "out.parquet",
            profile_name="test",
            thumbnail_dir=tmp_path / "thumbs",
            max_workers=1,
        )

    assert len(df) == 1
    assert df.iloc[0]["raw_path"] == str(input_dir / "with_xmp.cr3")


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

def test_load_dataset_round_trip(tmp_path: Path) -> None:
    input_dir = _make_synthetic_input(tmp_path, n=2)
    out = tmp_path / "out.parquet"

    with patch("sonna_editor.data.dataset.extract_all", side_effect=_fake_extract):
        df_written = build_dataset(input_dir, out, "test", tmp_path / "thumbs", max_workers=1)

    df_loaded = load_dataset(out)
    assert len(df_loaded) == len(df_written)
    assert set(df_loaded.columns) == set(df_written.columns)


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------

def _make_split_df(
    n_shoots: int = 10,
    photos_per_shoot: int = 5,
    skewed: bool = False,
) -> pd.DataFrame:
    """Build a fake photo dataset for split tests.

    Each shoot gets a synthetic AsShot Temperature and final Temperature.
    When `skewed=True`, two shoots are forced to be heavily cooling-biased
    (large delta) while the rest are near-neutral — used to verify the
    stratified split keeps these cooling-heavy shoots from concentrating
    in a single split.
    """
    rng = np.random.RandomState(0)
    rows = []
    for shoot in range(n_shoots):
        # AsShot in 4500-6500K range for most shoots; vary per shoot.
        asshot = float(rng.uniform(4500, 6500))
        # Final temperature delta — most shoots near zero, optionally make
        # two shoots strongly cooling.
        if skewed and shoot in (0, 1):
            delta = -1500.0
        else:
            delta = float(rng.uniform(-200, 200))
        final_temp = asshot + delta
        for photo in range(photos_per_shoot):
            rows.append({
                "id": f"shoot{shoot}_photo{photo}",
                "shoot_id": f"shoot_{shoot}",
                "raw_path": f"/fake/shoot{shoot}/photo{photo}.cr3",
                "as_shot_temperature": asshot,
                **{f: 0.0 for f in SLIDER_FIELDS},
                "Temperature": final_temp,
            })
    return pd.DataFrame(rows)


def test_split_dataset_sizes(tmp_path: Path) -> None:
    df = _make_split_df(n_shoots=20, photos_per_shoot=5)
    train, val, test = split_dataset(df, val_ratio=0.1, test_ratio=0.1)
    total = len(train) + len(val) + len(test)
    assert total == len(df)


def test_split_dataset_no_shoot_leakage() -> None:
    df = _make_split_df(n_shoots=20, photos_per_shoot=5)
    train, val, test = split_dataset(df)

    train_shoots = set(train["shoot_id"])
    val_shoots = set(val["shoot_id"])
    test_shoots = set(test["shoot_id"])

    assert train_shoots.isdisjoint(val_shoots), "shoot leakage: train ∩ val"
    assert train_shoots.isdisjoint(test_shoots), "shoot leakage: train ∩ test"
    assert val_shoots.isdisjoint(test_shoots), "shoot leakage: val ∩ test"


def test_split_dataset_train_larger_than_val_and_test() -> None:
    df = _make_split_df(n_shoots=20, photos_per_shoot=5)
    train, val, test = split_dataset(df)
    assert len(train) > len(val)
    assert len(train) > len(test)


def test_split_dataset_missing_group_col_raises() -> None:
    df = pd.DataFrame({"id": [1, 2, 3]})
    with pytest.raises(ValueError, match="group_col"):
        split_dataset(df, group_col="shoot_id")


# ---------------------------------------------------------------------------
# stratified_group_split — covers the post-v1.2.0 split rework
# ---------------------------------------------------------------------------

def test_stratified_split_distributes_cooling_shoots_across_splits() -> None:
    """Cooling-heavy shoots should land in different strata than neutral shoots,
    and stratified allocation should put at least one cooling-heavy shoot in
    each of train/val/test (not concentrate them in one split).

    Pre-stratification GroupShuffleSplit on this skewed input would routinely
    place both cooling-heavy shoots in val OR test (small-sample variance).
    Stratification places them in distinct strata so they get split apart.
    """
    from sonna_editor.data.dataset import stratified_group_split
    # 25 shoots; shoots 0,1 are heavily cooling, rest are near-neutral.
    df = _make_split_df(n_shoots=25, photos_per_shoot=10, skewed=True)
    train, val, test = stratified_group_split(
        df, val_ratio=0.2, test_ratio=0.2, n_strata=5, random_state=42
    )
    # Shoots 0 and 1 each have delta = -1500K; they should be in stratum 0
    # (the most cooling-heavy bucket). Quantile bucketing puts them in
    # different splits since the algorithm shuffles within stratum and
    # allocates ≥1 to each split per stratum.
    cooling_shoots = {"shoot_0", "shoot_1"}
    cooling_in_train = cooling_shoots & set(train["shoot_id"])
    cooling_in_val = cooling_shoots & set(val["shoot_id"])
    cooling_in_test = cooling_shoots & set(test["shoot_id"])
    placed = sum(1 for s in (cooling_in_train, cooling_in_val, cooling_in_test) if s)
    assert placed >= 2, (
        f"both cooling-heavy shoots concentrated in <2 splits "
        f"(train={cooling_in_train}, val={cooling_in_val}, test={cooling_in_test})"
    )


def test_stratified_split_no_photo_leakage() -> None:
    from sonna_editor.data.dataset import stratified_group_split
    df = _make_split_df(n_shoots=25, photos_per_shoot=10, skewed=True)
    train, val, test = stratified_group_split(df, n_strata=5, random_state=42)
    assert set(train["id"]).isdisjoint(set(val["id"]))
    assert set(train["id"]).isdisjoint(set(test["id"]))
    assert set(val["id"]).isdisjoint(set(test["id"]))
    assert len(train) + len(val) + len(test) == len(df)


def test_stratified_split_small_shoot_count_does_not_crash() -> None:
    from sonna_editor.data.dataset import stratified_group_split
    df = _make_split_df(n_shoots=1, photos_per_shoot=3)
    train, val, test = stratified_group_split(
        df, val_ratio=0.1, test_ratio=0.1, n_strata=5, random_state=42
    )
    assert len(train) + len(val) + len(test) == len(df)
    assert set(train["id"]).isdisjoint(set(val["id"]))
    assert set(train["id"]).isdisjoint(set(test["id"]))
    assert set(val["id"]).isdisjoint(set(test["id"]))


def test_stratified_split_deterministic_with_seed() -> None:
    from sonna_editor.data.dataset import stratified_group_split
    df = _make_split_df(n_shoots=25, photos_per_shoot=10, skewed=True)
    t1, v1, te1 = stratified_group_split(df, n_strata=5, random_state=42)
    t2, v2, te2 = stratified_group_split(df, n_strata=5, random_state=42)
    assert sorted(t1["shoot_id"].unique()) == sorted(t2["shoot_id"].unique())
    assert sorted(v1["shoot_id"].unique()) == sorted(v2["shoot_id"].unique())
    assert sorted(te1["shoot_id"].unique()) == sorted(te2["shoot_id"].unique())


def test_split_dataset_balances_exposure_across_splits() -> None:
    """Default split balancing must consider Exposure, not only WB deltas."""
    df = _make_split_df(n_shoots=30, photos_per_shoot=5)
    for shoot in range(30):
        exposure = 1.2 if shoot % 3 == 0 else (-0.4 if shoot % 3 == 1 else 0.2)
        df.loc[df["shoot_id"] == f"shoot_{shoot}", "Exposure2012"] = exposure

    train, val, test = split_dataset(df, val_ratio=0.2, test_ratio=0.2)
    global_mean = float(df["Exposure2012"].mean())
    for split in (train, val, test):
        assert abs(float(split["Exposure2012"].mean()) - global_mean) < 0.35


def test_stratified_split_handles_missing_asshot() -> None:
    """A shoot with no AsShot Temperature should still land in some split,
    not crash the splitter."""
    from sonna_editor.data.dataset import stratified_group_split
    df = _make_split_df(n_shoots=25, photos_per_shoot=10)
    # Wipe AsShot for one whole shoot.
    df.loc[df["shoot_id"] == "shoot_5", "as_shot_temperature"] = np.nan
    train, val, test = stratified_group_split(df, n_strata=5, random_state=42)
    in_any = "shoot_5" in set(train["shoot_id"]) | set(val["shoot_id"]) | set(test["shoot_id"])
    assert in_any, "shoot with missing AsShot was dropped"


def test_stratified_split_balancing_post_pass_prioritises_test() -> None:
    """After balancing, test photo-share should be within tolerance of target_ratio,
    even when initial per-stratum integer allocation under-allocates test."""
    from sonna_editor.data.dataset import stratified_group_split
    # 100 shoots × 10 photos = 1000 photos. Target test=13.9% ≈ 139 photos.
    df = _make_split_df(n_shoots=100, photos_per_shoot=10)
    train, val, test = stratified_group_split(
        df, val_ratio=0.107, test_ratio=0.139, n_strata=5, random_state=42,
        test_priority_tolerance=0.02,
    )
    test_pct = len(test) / len(df)
    assert 0.119 <= test_pct <= 0.159, (
        f"test_pct={test_pct:.3f} outside ±2pp of 0.139 target after balancing"
    )


def test_stratified_split_backward_compat_when_stratify_off() -> None:
    """stratify_on=None falls back to the original GroupShuffleSplit behaviour."""
    from sonna_editor.data.dataset import stratified_group_split
    df = _make_split_df(n_shoots=20, photos_per_shoot=5)
    train, val, test = stratified_group_split(
        df, val_ratio=0.1, test_ratio=0.1, stratify_on=None, random_state=42
    )
    # No leakage; sizes sum.
    assert set(train["shoot_id"]).isdisjoint(set(val["shoot_id"]))
    assert len(train) + len(val) + len(test) == len(df)


# ---------------------------------------------------------------------------
# save_split
# ---------------------------------------------------------------------------

def test_save_split_writes_three_files(tmp_path: Path) -> None:
    df = _make_split_df(n_shoots=10)
    train, val, test = split_dataset(df)
    save_split(train, val, test, tmp_path / "splits")

    assert (tmp_path / "splits" / "train.parquet").exists()
    assert (tmp_path / "splits" / "val.parquet").exists()
    assert (tmp_path / "splits" / "test.parquet").exists()


def test_save_split_creates_output_dir(tmp_path: Path) -> None:
    df = _make_split_df()
    train, val, test = split_dataset(df)
    out = tmp_path / "new" / "nested" / "splits"
    save_split(train, val, test, out)
    assert out.exists()


# ---------------------------------------------------------------------------
# Integration test — real CR3 fixture
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_build_dataset_real_photo(tmp_path: Path) -> None:
    if not FIXTURE_RAW.exists():
        pytest.skip("Real CR3 fixture not present")

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "sample.cr3").symlink_to(FIXTURE_RAW)
    if FIXTURE_XMP.exists():
        (input_dir / "sample.xmp").symlink_to(FIXTURE_XMP)

    df = build_dataset(
        input_dir=input_dir,
        output_path=tmp_path / "out.parquet",
        profile_name="integration_test",
        thumbnail_dir=tmp_path / "thumbs",
        max_workers=1,
    )

    assert len(df) == 1
    assert df["camera_body"].iloc[0] is not None
    assert df["Exposure2012"].iloc[0] is not None
