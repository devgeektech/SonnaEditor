from __future__ import annotations

from pathlib import Path

import numpy as np
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

    sliders = {field: None for field in catalog_dataset.SLIDER_FIELDS}
    row = catalog_dataset._process_catalog_row(
        (str(raw_path), None, "sonna_v2", str(thumbnail_dir), sliders)
    )

    assert row is not None
    assert row["xmp_path"] is None
    assert row["as_shot_temperature"] == 4850.0
    assert row["as_shot_tint"] == 7.5
