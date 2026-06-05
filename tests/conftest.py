"""Shared pytest fixtures for sonna-editor tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sonna_editor.config import SLIDER_FIELDS


# --------------------------------------------------------------------------- #
# Synthetic Lightroom catalog fixture
# --------------------------------------------------------------------------- #


# Photo IDs used by tests
PHOTO_A_EDITED_XMP = 1001     # XMP-format develop settings, file present
PHOTO_B_EDITED_LUA = 1002     # Lua-format develop settings, file present
PHOTO_C_UNEDITED   = 1003
PHOTO_D_REJECTED   = 1004     # pick = -1
PHOTO_E_PICKED     = 1005     # pick =  1
PHOTO_F_FIVE_STAR  = 1006     # rating = 5
PHOTO_G_RED_LABEL  = 1007     # colorLabels = "Red"
PHOTO_H_MISSING    = 1008     # file does not exist on disk
PHOTO_J_EMPTY      = 1009     # hasDevelopSettings=1 but blob is empty
PHOTO_K_ALL_FIELDS = 1010     # XMP with every SLIDER_FIELDS value


_PHOTO_A_XMP = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
    '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" '
    'rdf:about="" crs:ProcessVersion="15.4" crs:HasSettings="True" '
    'crs:Exposure2012="+0.50" crs:Temperature="5500" crs:Tint="0"/>'
    '</rdf:RDF></x:xmpmeta>'
)

_PHOTO_B_LUA = (
    "s = {\n"
    " Exposure2012 = -1.2,\n"
    " Temperature = 4800,\n"
    " HueAdjustmentRed = -3,\n"
    " ToneCurvePV2012 = { 0, 0, 255, 255 },\n"
    "}"
)

# History snapshot for photo A (must be ignored by current-settings query)
_PHOTO_A_HISTORY_XMP = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
    '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" '
    'rdf:about="" crs:Exposure2012="-9.99" crs:Temperature="9000"/>'
    '</rdf:RDF></x:xmpmeta>'
)


def _make_all_fields_xmp() -> str:
    overrides = {
        "Exposure2012": "+1.25",
        "Temperature": "5200",
        "HueAdjustmentOrange": "+7",
    }
    attrs = []
    for f in SLIDER_FIELDS:
        if f in overrides:
            attrs.append(f'crs:{f}="{overrides[f]}"')
        else:
            attrs.append(f'crs:{f}="0"')
    return (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" '
        'rdf:about="" crs:ProcessVersion="15.4" crs:HasSettings="True" '
        + " ".join(attrs) +
        '/></rdf:RDF></x:xmpmeta>'
    )


_SCHEMA_SQL = """
CREATE TABLE Adobe_variablesTable (
    id_local INTEGER PRIMARY KEY,
    name TEXT,
    value TEXT
);

CREATE TABLE AgLibraryRootFolder (
    id_local INTEGER PRIMARY KEY,
    absolutePath TEXT
);

CREATE TABLE AgLibraryFolder (
    id_local INTEGER PRIMARY KEY,
    rootFolder INTEGER,
    pathFromRoot TEXT
);

CREATE TABLE AgLibraryFile (
    id_local INTEGER PRIMARY KEY,
    folder INTEGER,
    idx_filename TEXT
);

CREATE TABLE Adobe_images (
    id_local INTEGER PRIMARY KEY,
    rootFile INTEGER,
    captureTime TEXT,
    rating REAL,
    colorLabels TEXT,
    pick REAL
);

CREATE TABLE Adobe_imageDevelopSettings (
    id_local INTEGER PRIMARY KEY,
    image INTEGER,
    hasDevelopSettings INTEGER,
    beforeDigest TEXT,
    developSettings TEXT
);

CREATE TABLE AgLibraryCollection (
    id_local INTEGER PRIMARY KEY,
    name TEXT,
    imageCount REAL
);

CREATE TABLE AgLibraryCollectionImage (
    id_local INTEGER PRIMARY KEY,
    collection INTEGER,
    image INTEGER,
    pick REAL,
    positionInCollection REAL
);
"""


def _build_synthetic_catalog(lrcat_path: Path, photos_root: Path) -> None:
    """Populate a fresh .lrcat-like SQLite file with the test scenarios."""
    conn = sqlite3.connect(str(lrcat_path))
    try:
        conn.executescript(_SCHEMA_SQL)

        # Adobe_DBVersion in known-good range
        conn.execute(
            "INSERT INTO Adobe_variablesTable (id_local, name, value) VALUES (?, ?, ?)",
            (1, "Adobe_DBVersion", "0900000"),
        )

        # Root folders
        # Root 1 — real directory on disk
        conn.execute(
            "INSERT INTO AgLibraryRootFolder (id_local, absolutePath) VALUES (?, ?)",
            (1, str(photos_root) + "/"),
        )
        # Root 2 — for the missing-file scenario
        conn.execute(
            "INSERT INTO AgLibraryRootFolder (id_local, absolutePath) VALUES (?, ?)",
            (2, "/tmp/nonexistent_volume_xyz/"),
        )

        # Folders
        conn.execute(
            "INSERT INTO AgLibraryFolder (id_local, rootFolder, pathFromRoot) VALUES (?, ?, ?)",
            (10, 1, "2024/"),
        )
        conn.execute(
            "INSERT INTO AgLibraryFolder (id_local, rootFolder, pathFromRoot) VALUES (?, ?, ?)",
            (11, 2, "2024/"),
        )

        # Build the on-disk folder for present files
        present_dir = photos_root / "2024"
        present_dir.mkdir(parents=True, exist_ok=True)

        # Insert helper
        next_file_id = [100]

        def add_photo(
            image_id: int,
            filename: str,
            *,
            folder_id: int = 10,
            capture_time: str | None = "2024-01-01T10:00:00",
            rating: float = 0.0,
            color_label: str = "",
            pick: float = 0.0,
            has_develop_settings: int = 0,
            develop_settings: str | None = None,
            create_file: bool = True,
        ) -> None:
            file_id = next_file_id[0]
            next_file_id[0] += 1

            conn.execute(
                "INSERT INTO AgLibraryFile (id_local, folder, idx_filename) "
                "VALUES (?, ?, ?)",
                (file_id, folder_id, filename),
            )
            conn.execute(
                "INSERT INTO Adobe_images "
                "(id_local, rootFile, captureTime, rating, colorLabels, pick) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (image_id, file_id, capture_time, rating, color_label, pick),
            )
            if has_develop_settings or develop_settings is not None:
                conn.execute(
                    "INSERT INTO Adobe_imageDevelopSettings "
                    "(image, hasDevelopSettings, beforeDigest, developSettings) "
                    "VALUES (?, ?, NULL, ?)",
                    (image_id, has_develop_settings, develop_settings),
                )
            if create_file and folder_id == 10:
                (present_dir / filename).touch()

        # A. Edited, XMP-format settings, present file
        add_photo(
            PHOTO_A_EDITED_XMP, "photo_a.cr3",
            capture_time="2024-01-01T10:00:00",
            rating=3.0, color_label="", pick=0.0,
            has_develop_settings=1, develop_settings=_PHOTO_A_XMP,
        )

        # History snapshot row for A (must be excluded everywhere)
        conn.execute(
            "INSERT INTO Adobe_imageDevelopSettings "
            "(image, hasDevelopSettings, beforeDigest, developSettings) "
            "VALUES (?, ?, ?, ?)",
            (PHOTO_A_EDITED_XMP, 1, "abc123", _PHOTO_A_HISTORY_XMP),
        )

        # B. Edited, Lua-format settings
        add_photo(
            PHOTO_B_EDITED_LUA, "photo_b.cr3",
            capture_time="2024-01-01T11:00:00",
            rating=2.0, color_label="", pick=0.0,
            has_develop_settings=1, develop_settings=_PHOTO_B_LUA,
        )

        # C. Unedited
        add_photo(
            PHOTO_C_UNEDITED, "photo_c.cr3",
            capture_time="2024-01-01T12:00:00",
            rating=0.0, color_label="", pick=0.0,
            has_develop_settings=0,
        )

        # D. Rejected (pick = -1)
        add_photo(
            PHOTO_D_REJECTED, "photo_d.cr3",
            capture_time="2024-01-01T13:00:00",
            rating=0.0, color_label="", pick=-1.0,
            has_develop_settings=0,
        )

        # E. Picked (pick = 1)
        add_photo(
            PHOTO_E_PICKED, "photo_e.cr3",
            capture_time="2024-01-01T14:00:00",
            rating=0.0, color_label="", pick=1.0,
            has_develop_settings=0,
        )

        # F. 5-star
        add_photo(
            PHOTO_F_FIVE_STAR, "photo_f.cr3",
            capture_time="2024-01-01T15:00:00",
            rating=5.0, color_label="", pick=0.0,
            has_develop_settings=0,
        )

        # G. Red label
        add_photo(
            PHOTO_G_RED_LABEL, "photo_g.cr3",
            capture_time="2024-01-01T16:00:00",
            rating=0.0, color_label="Red", pick=0.0,
            has_develop_settings=0,
        )

        # H. Missing file (separate root)
        add_photo(
            PHOTO_H_MISSING, "photo_h.cr3",
            folder_id=11,
            capture_time="2024-01-01T17:00:00",
            rating=0.0, color_label="", pick=0.0,
            has_develop_settings=0,
            create_file=False,
        )

        # J. Empty develop settings (corruption)
        add_photo(
            PHOTO_J_EMPTY, "photo_j.cr3",
            capture_time="2024-01-01T18:00:00",
            rating=0.0, color_label="", pick=0.0,
            has_develop_settings=1, develop_settings="",
        )

        # K. All 37 sliders set
        add_photo(
            PHOTO_K_ALL_FIELDS, "photo_k.cr3",
            capture_time="2024-01-01T19:00:00",
            rating=4.0, color_label="", pick=0.0,
            has_develop_settings=1, develop_settings=_make_all_fields_xmp(),
        )

        conn.execute(
            "INSERT INTO AgLibraryCollection (id_local, name, imageCount) VALUES (?, ?, ?)",
            (1, "C", 2),
        )
        conn.execute(
            "INSERT INTO AgLibraryCollection (id_local, name, imageCount) VALUES (?, ?, ?)",
            (2, "A", 1),
        )
        conn.execute(
            "INSERT INTO AgLibraryCollectionImage "
            "(id_local, collection, image, pick, positionInCollection) "
            "VALUES (?, ?, ?, ?, ?)",
            (1, 1, PHOTO_A_EDITED_XMP, 0, 1),
        )
        conn.execute(
            "INSERT INTO AgLibraryCollectionImage "
            "(id_local, collection, image, pick, positionInCollection) "
            "VALUES (?, ?, ?, ?, ?)",
            (2, 1, PHOTO_B_EDITED_LUA, 0, 2),
        )
        conn.execute(
            "INSERT INTO AgLibraryCollectionImage "
            "(id_local, collection, image, pick, positionInCollection) "
            "VALUES (?, ?, ?, ?, ?)",
            (3, 2, PHOTO_G_RED_LABEL, 0, 1),
        )

        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def synthetic_catalog(tmp_path: Path) -> Path:
    """Build a minimal Lightroom-shaped SQLite catalog under tmp_path.

    Returns the path to the .lrcat file. The caller is responsible for
    opening it via `connect_catalog`.
    """
    photos_root = tmp_path / "photos"
    photos_root.mkdir()
    lrcat = tmp_path / "test.lrcat"
    _build_synthetic_catalog(lrcat, photos_root)
    return lrcat
