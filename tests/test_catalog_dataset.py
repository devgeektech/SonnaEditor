from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sonna_editor.data import catalog_dataset


def test_catalog_row_includes_as_shot_wb(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_path = tmp_path / "IMG_0001.CR3"
    raw_path.write_bytes(b"fake raw")
    thumbnail_dir = tmp_path / "thumbs"
    thumbnail_dir.mkdir()

    monkeypatch.setattr(
        catalog_dataset,
        "extract_preview",
        lambda path: Image.new("RGB", (16, 16), color=(128, 128, 128)),
    )
    monkeypatch.setattr(
        catalog_dataset,
        "extract_metadata",
        lambda path: {
            "capture_datetime": None,
            "camera_body": "Canon R5",
            "as_shot_wb": (4850.0, 7.5),
        },
    )
    monkeypatch.setattr(
        catalog_dataset,
        "compute_histogram",
        lambda image: np.zeros((3, 32), dtype=np.float32),
    )

    sliders: dict[str, float | None] = {field: None for field in catalog_dataset.SLIDER_FIELDS}
    row = catalog_dataset._process_catalog_row(
        (str(raw_path), None, "sonna_v2", str(thumbnail_dir), sliders)
    )

    assert row is not None
    assert row["xmp_path"] is None
    assert row["as_shot_temperature"] == 4850.0
    assert row["as_shot_tint"] == 7.5


class _FakeCatalogConnection:
    def close(self) -> None:
        pass


def _edited_slider_dict() -> dict[str, float | None]:
    sliders: dict[str, float | None] = {
        field: None for field in catalog_dataset.SLIDER_FIELDS
    }
    for field in catalog_dataset._SCALAR_SLIDER_FIELDS[:30]:
        sliders[field] = 10.0
    return sliders


def _unedited_slider_dict() -> dict[str, float | None]:
    sliders: dict[str, float | None] = {field: None for field in catalog_dataset.SLIDER_FIELDS}
    return sliders


def test_catalog_builder_skips_unedited_rows_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_unedited = tmp_path / "unedited.dng"
    raw_edited = tmp_path / "edited.dng"
    raw_unedited.write_bytes(b"raw")
    raw_edited.write_bytes(b"raw")
    output_path = tmp_path / "dataset.parquet"
    thumbnail_dir = tmp_path / "thumbs"
    photos = [
        {
            "image_id": 1,
            "file_path": raw_unedited,
            "capture_time": "2026-01-01 10:00:00",
            "has_develop_settings": True,
        },
        {
            "image_id": 2,
            "file_path": raw_edited,
            "capture_time": "2026-01-01 11:00:00",
            "has_develop_settings": True,
        },
    ]
    slider_by_id = {
        1: _unedited_slider_dict(),
        2: _edited_slider_dict(),
    }

    monkeypatch.setattr(
        catalog_dataset,
        "connect_catalog",
        lambda catalog_path: _FakeCatalogConnection(),
    )
    monkeypatch.setattr(
        catalog_dataset,
        "find_edited_photos",
        lambda conn, collection_name=None: photos,
    )
    monkeypatch.setattr(
        catalog_dataset,
        "get_develop_settings",
        lambda conn, image_id: slider_by_id[image_id],
    )
    monkeypatch.setattr(
        catalog_dataset,
        "_process_catalog_row",
        lambda args: {
            "id": Path(args[0]).stem,
            "raw_path": args[0],
            "thumbnail_path": str(thumbnail_dir / f"{Path(args[0]).stem}.jpg"),
            "shoot_id": "shoot-1",
            **args[4],
        },
    )

    df, stats = catalog_dataset.build_dataset_from_catalog(
        catalog_path=tmp_path / "catalog.lrcat",
        output_path=output_path,
        profile_name="test",
        thumbnail_dir=thumbnail_dir,
        max_workers=1,
        skip_unedited=True,
    )

    assert len(df) == 1
    assert df.iloc[0]["raw_path"] == str(raw_edited)
    assert stats["skip_unedited"] == 1
    assert stats["skip_unedited_filter_enabled"] == 1
    saved = pd.read_parquet(output_path)
    assert len(saved) == 1


def test_catalog_builder_can_include_unedited_rows_for_fivek_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_unedited = tmp_path / "unedited.dng"
    raw_edited = tmp_path / "edited.dng"
    raw_unedited.write_bytes(b"raw")
    raw_edited.write_bytes(b"raw")
    output_path = tmp_path / "dataset.parquet"
    thumbnail_dir = tmp_path / "thumbs"
    photos = [
        {
            "image_id": 1,
            "file_path": raw_unedited,
            "capture_time": "2026-01-01 10:00:00",
            "has_develop_settings": True,
        },
        {
            "image_id": 2,
            "file_path": raw_edited,
            "capture_time": "2026-01-01 11:00:00",
            "has_develop_settings": True,
        },
    ]
    slider_by_id = {
        1: _unedited_slider_dict(),
        2: _edited_slider_dict(),
    }

    monkeypatch.setattr(
        catalog_dataset,
        "connect_catalog",
        lambda catalog_path: _FakeCatalogConnection(),
    )
    monkeypatch.setattr(
        catalog_dataset,
        "find_edited_photos",
        lambda conn, collection_name=None: photos,
    )
    monkeypatch.setattr(
        catalog_dataset,
        "get_develop_settings",
        lambda conn, image_id: slider_by_id[image_id],
    )
    monkeypatch.setattr(
        catalog_dataset,
        "_process_catalog_row",
        lambda args: {
            "id": Path(args[0]).stem,
            "raw_path": args[0],
            "thumbnail_path": str(thumbnail_dir / f"{Path(args[0]).stem}.jpg"),
            "shoot_id": "shoot-1",
            **args[4],
        },
    )

    df, stats = catalog_dataset.build_dataset_from_catalog(
        catalog_path=tmp_path / "catalog.lrcat",
        output_path=output_path,
        profile_name="test",
        thumbnail_dir=thumbnail_dir,
        max_workers=1,
        skip_unedited=False,
    )

    assert len(df) == 2
    assert stats["skip_unedited"] == 0
    assert stats["skip_unedited_filter_enabled"] == 0
