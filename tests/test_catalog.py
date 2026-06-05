"""Tests for the Lightroom catalog reader."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sonna_editor.config import SLIDER_FIELDS
from sonna_editor.data import xmp as xmp_module
from sonna_editor.data.catalog import (
    CatalogError,
    CatalogLockedError,
    CatalogVersionError,
    DevelopSettingsParseError,
    _build_path,
    _extract_lua_table,
    _normalise_to_slider_fields,
    _parse_lua_blob,
    connect_catalog,
    export_xmps_for_photos,
    find_edited_photos,
    get_develop_settings,
)

from tests.conftest import (
    PHOTO_A_EDITED_XMP,
    PHOTO_B_EDITED_LUA,
    PHOTO_C_UNEDITED,
    PHOTO_D_REJECTED,
    PHOTO_E_PICKED,
    PHOTO_F_FIVE_STAR,
    PHOTO_G_RED_LABEL,
    PHOTO_H_MISSING,
    PHOTO_J_EMPTY,
    PHOTO_K_ALL_FIELDS,
)


# --------------------------------------------------------------------------- #
# connect_catalog
# --------------------------------------------------------------------------- #


def test_connect_catalog_opens_readonly(synthetic_catalog: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    try:
        # query_only must be active
        cur = conn.execute("PRAGMA query_only;")
        assert cur.fetchone()[0] == 1
        # Sanity: a SELECT works
        cur = conn.execute("SELECT COUNT(*) FROM Adobe_images")
        assert cur.fetchone()[0] >= 10
    finally:
        conn.close()


def test_locked_catalog_raises(synthetic_catalog: Path) -> None:
    lock = synthetic_catalog.parent / (synthetic_catalog.name + ".lock")
    lock.write_text("")
    with pytest.raises(CatalogLockedError):
        connect_catalog(synthetic_catalog)


def test_journal_file_raises_locked(synthetic_catalog: Path) -> None:
    journal = synthetic_catalog.parent / (synthetic_catalog.name + "-journal")
    journal.write_text("")
    with pytest.raises(CatalogLockedError):
        connect_catalog(synthetic_catalog)


def test_wal_file_does_not_raise(synthetic_catalog: Path) -> None:
    # WAL/SHM files are normal SQLite bookkeeping — not an LR lock indicator
    wal = synthetic_catalog.parent / (synthetic_catalog.name + "-wal")
    wal.write_bytes(b"")
    conn = connect_catalog(synthetic_catalog)
    conn.close()


def test_wrong_suffix_raises(tmp_path: Path) -> None:
    bad = tmp_path / "catalog.sqlite"
    bad.touch()
    with pytest.raises(CatalogError, match=".lrcat"):
        connect_catalog(bad)


def test_nonexistent_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.lrcat"
    with pytest.raises(CatalogError, match="not found"):
        connect_catalog(missing)


def test_missing_table_raises_version_error(synthetic_catalog: Path) -> None:
    # Drop a required table directly (synthetic catalog only — never on a real one)
    conn = sqlite3.connect(str(synthetic_catalog))
    conn.execute("DROP TABLE Adobe_imageDevelopSettings")
    conn.commit()
    conn.close()

    with pytest.raises(CatalogVersionError, match="Adobe_imageDevelopSettings"):
        connect_catalog(synthetic_catalog)


# --------------------------------------------------------------------------- #
# find_edited_photos — filtering and shape
# --------------------------------------------------------------------------- #


def _index_by_id(photos: list[dict]) -> dict[int, dict]:
    return {p["image_id"]: p for p in photos}


def test_find_excludes_rejected(synthetic_catalog: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    photos = find_edited_photos(conn)
    ids = {p["image_id"] for p in photos}
    assert PHOTO_D_REJECTED not in ids


def test_find_returns_expected_keys(synthetic_catalog: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    photos = find_edited_photos(conn)
    expected = {
        "image_id", "file_path", "capture_time", "rating", "color_label",
        "pick_flag", "has_develop_settings", "is_missing",
    }
    for p in photos:
        assert set(p.keys()) == expected


def test_find_marks_has_develop_settings(synthetic_catalog: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    by_id = _index_by_id(find_edited_photos(conn))
    assert by_id[PHOTO_A_EDITED_XMP]["has_develop_settings"] is True
    assert by_id[PHOTO_B_EDITED_LUA]["has_develop_settings"] is True
    assert by_id[PHOTO_C_UNEDITED]["has_develop_settings"] is False
    assert by_id[PHOTO_K_ALL_FIELDS]["has_develop_settings"] is True


def test_find_marks_missing_files(synthetic_catalog: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    by_id = _index_by_id(find_edited_photos(conn))
    assert by_id[PHOTO_H_MISSING]["is_missing"] is True
    assert by_id[PHOTO_A_EDITED_XMP]["is_missing"] is False


def test_find_filter_min_rating(synthetic_catalog: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    photos = find_edited_photos(conn, min_rating=5)
    ids = {p["image_id"] for p in photos}
    assert PHOTO_F_FIVE_STAR in ids
    assert PHOTO_A_EDITED_XMP not in ids
    assert PHOTO_C_UNEDITED not in ids


def test_find_filter_min_flag_picked_only(synthetic_catalog: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    photos = find_edited_photos(conn, min_flag=1)
    ids = {p["image_id"] for p in photos}
    assert PHOTO_E_PICKED in ids
    assert PHOTO_A_EDITED_XMP not in ids
    assert PHOTO_D_REJECTED not in ids


def test_find_filter_color_label(synthetic_catalog: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    photos = find_edited_photos(conn, min_color_label="Red")
    ids = {p["image_id"] for p in photos}
    assert ids == {PHOTO_G_RED_LABEL}


def test_find_filter_collection_name(synthetic_catalog: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    photos = find_edited_photos(conn, collection_name="C")
    ids = {p["image_id"] for p in photos}
    assert ids == {PHOTO_A_EDITED_XMP, PHOTO_B_EDITED_LUA}


def test_find_orders_by_capture_time(synthetic_catalog: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    photos = find_edited_photos(conn)
    times = [p["capture_time"] for p in photos]
    assert times == sorted(times)


def test_find_builds_absolute_path(synthetic_catalog: Path, tmp_path: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    by_id = _index_by_id(find_edited_photos(conn))
    p = by_id[PHOTO_A_EDITED_XMP]
    assert p["file_path"].is_absolute()
    assert p["file_path"].name == "photo_a.cr3"
    assert p["file_path"].exists()


# --------------------------------------------------------------------------- #
# get_develop_settings — XMP, Lua, errors
# --------------------------------------------------------------------------- #


def test_get_develop_settings_xmp(synthetic_catalog: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    settings = get_develop_settings(conn, PHOTO_A_EDITED_XMP)
    assert set(settings.keys()) == set(SLIDER_FIELDS)
    assert settings["Exposure2012"] == pytest.approx(0.5)
    assert settings["Temperature"] == pytest.approx(5500.0)
    assert settings["Tint"] == pytest.approx(0.0)


def test_get_develop_settings_lua(synthetic_catalog: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    settings = get_develop_settings(conn, PHOTO_B_EDITED_LUA)
    assert set(settings.keys()) == set(SLIDER_FIELDS)
    assert settings["Exposure2012"] == pytest.approx(-1.2)
    assert settings["Temperature"] == pytest.approx(4800.0)
    assert settings["HueAdjustmentRed"] == pytest.approx(-3.0)


def test_get_develop_settings_empty_blob_raises(synthetic_catalog: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    with pytest.raises(DevelopSettingsParseError):
        get_develop_settings(conn, PHOTO_J_EMPTY)


def test_get_develop_settings_all_fields(synthetic_catalog: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    settings = get_develop_settings(conn, PHOTO_K_ALL_FIELDS)
    # All 37 fields present and numeric
    for f in SLIDER_FIELDS:
        assert settings[f] is not None, f"missing field {f}"
        assert isinstance(settings[f], float), f"non-float for {f}: {settings[f]!r}"
    assert settings["Exposure2012"] == pytest.approx(1.25)
    assert settings["Temperature"] == pytest.approx(5200.0)
    assert settings["HueAdjustmentOrange"] == pytest.approx(7.0)


def test_history_row_excluded(synthetic_catalog: Path) -> None:
    """Photo A has a beforeDigest='abc123' history row; it must be ignored."""
    conn = connect_catalog(synthetic_catalog)
    settings = get_develop_settings(conn, PHOTO_A_EDITED_XMP)
    # Current settings: Exposure2012=0.5, NOT -9.99 (history)
    assert settings["Exposure2012"] == pytest.approx(0.5)
    assert settings["Temperature"] == pytest.approx(5500.0)


# --------------------------------------------------------------------------- #
# Lua parser — edge cases
# --------------------------------------------------------------------------- #


def test_lua_missing_fields_return_none() -> None:
    raw = _parse_lua_blob("s = {\n Exposure2012 = 0.3,\n}")
    settings = _normalise_to_slider_fields(raw)
    assert settings["Exposure2012"] == pytest.approx(0.3)
    for f in SLIDER_FIELDS:
        if f == "Exposure2012":
            continue
        if f.startswith("ToneCurve"):
            # Tone curve fields always return identity defaults (never None) when absent
            assert isinstance(settings[f], float), f"expected float for {f}, got {settings[f]!r}"
        else:
            assert settings[f] is None, f"expected {f}=None, got {settings[f]!r}"


def test_lua_negative_float() -> None:
    raw = _parse_lua_blob("s = {\n Exposure2012 = -2.75,\n}")
    settings = _normalise_to_slider_fields(raw)
    assert settings["Exposure2012"] == pytest.approx(-2.75)


def test_lua_positive_integer() -> None:
    raw = _parse_lua_blob("s = {\n Temperature = 5500,\n}")
    settings = _normalise_to_slider_fields(raw)
    assert settings["Temperature"] == pytest.approx(5500.0)
    assert isinstance(settings["Temperature"], float)


def test_lua_skips_table_values() -> None:
    """Table-valued fields (e.g. ToneCurvePV2012) must not be captured."""
    blob = (
        "s = {\n"
        " Exposure2012 = 0.0,\n"
        " ToneCurvePV2012 = { 0, 0, 255, 255 },\n"
        "}"
    )
    raw = _parse_lua_blob(blob)
    assert "Exposure2012" in raw
    assert "ToneCurvePV2012" not in raw


def test_lua_signed_positive_float() -> None:
    """LR sometimes writes '+0.60' style signed positive floats."""
    raw = _parse_lua_blob("s = {\n Exposure2012 = +0.60,\n}")
    settings = _normalise_to_slider_fields(raw)
    assert settings["Exposure2012"] == pytest.approx(0.60)


def test_lua_unsigned_positive_float() -> None:
    raw = _parse_lua_blob("s = {\n Exposure2012 = 1.5,\n}")
    settings = _normalise_to_slider_fields(raw)
    assert settings["Exposure2012"] == pytest.approx(1.5)


def test_lua_last_field_with_closing_brace() -> None:
    """Regression: Whites2012 is the last field in the Lua table and ends with }
    (no trailing comma). The parser must not silently drop it.

    Real catalog format: 'Whites2012 = -21 }' or 'Whites2012 = -21 },'
    when the inner settings dict closes inside a larger structure.
    """
    blob = (
        "s = {\n"
        " Exposure2012 = 0.3,\n"
        " Vibrance = 10,\n"
        " WhiteBalance = \"As Shot\",\n"
        " Whites2012 = -21 }"   # no trailing comma, just closing brace
    )
    raw = _parse_lua_blob(blob)
    settings = _normalise_to_slider_fields(raw)
    assert settings["Whites2012"] == pytest.approx(-21.0), (
        "Whites2012 dropped — regex does not allow closing brace on last field"
    )
    assert settings["Exposure2012"] == pytest.approx(0.3)


def test_lua_last_field_with_closing_brace_and_comma() -> None:
    """Regression: variant where the inner table closes with '},' because
    the outer structure continues after it — 'Whites2012 = -70 },'."""
    blob = (
        "s = {\n"
        " Exposure2012 = 0.5,\n"
        " Whites2012 = -70 },\n"   # closing brace + comma (inner table, outer continues)
        " UUID = \"ABC123\" }\n"
    )
    raw = _parse_lua_blob(blob)
    settings = _normalise_to_slider_fields(raw)
    assert settings["Whites2012"] == pytest.approx(-70.0)


# --------------------------------------------------------------------------- #
# export_xmps_for_photos
# --------------------------------------------------------------------------- #


def test_export_manual_prints_instructions(
    synthetic_catalog: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    conn = connect_catalog(synthetic_catalog)
    photos = find_edited_photos(conn)
    export_xmps_for_photos(conn, photos, output_method="manual")
    captured = capsys.readouterr().out
    assert "Save Metadata to Files" in captured


def test_export_auto_writes_sidecars(synthetic_catalog: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    photos = find_edited_photos(conn)
    # Exclude Photo J (empty blob fixture) — it would cause an intentional error
    photos_clean = [p for p in photos if p["image_id"] != PHOTO_J_EMPTY]
    summary = export_xmps_for_photos(conn, photos_clean, output_method="auto")

    assert summary["written"] >= 1
    assert summary["errors"] == 0

    by_id = _index_by_id(photos)
    a = by_id[PHOTO_A_EDITED_XMP]
    sidecar = a["file_path"].with_suffix(".xmp")
    assert sidecar.exists(), "expected sidecar for photo A"
    assert "Exposure2012" in sidecar.read_text()


def test_export_auto_does_not_overwrite_existing(synthetic_catalog: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    photos = find_edited_photos(conn)
    by_id = _index_by_id(photos)
    a = by_id[PHOTO_A_EDITED_XMP]
    sidecar = a["file_path"].with_suffix(".xmp")
    sidecar.write_text("PRE-EXISTING")

    summary = export_xmps_for_photos(conn, photos, output_method="auto")
    assert sidecar.read_text() == "PRE-EXISTING"
    assert summary["skipped_existing"] >= 1


def test_export_auto_skips_missing_source(
    synthetic_catalog: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Photo H is missing on disk. Give it develop settings so the exporter sees it."""
    import logging

    conn = connect_catalog(synthetic_catalog)
    photos = find_edited_photos(conn)
    # Force Photo H to appear as having develop settings so the missing check fires
    by_id = _index_by_id(photos)
    if PHOTO_H_MISSING in by_id:
        by_id[PHOTO_H_MISSING]["has_develop_settings"] = True
    with caplog.at_level(logging.WARNING, logger="sonna_editor.data.catalog"):
        summary = export_xmps_for_photos(conn, list(by_id.values()), output_method="auto")
    assert summary["skipped_missing"] >= 1
    assert any("missing" in r.message.lower() for r in caplog.records)


def test_export_returns_summary_dict(synthetic_catalog: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    photos = find_edited_photos(conn)
    summary = export_xmps_for_photos(conn, photos, output_method="auto")
    assert set(summary.keys()) == {"written", "skipped_existing", "skipped_missing", "errors"}


def test_export_manual_returns_zero_summary(
    synthetic_catalog: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    conn = connect_catalog(synthetic_catalog)
    photos = find_edited_photos(conn)
    summary = export_xmps_for_photos(conn, photos, output_method="manual")
    assert summary == {"written": 0, "skipped_existing": 0, "skipped_missing": 0, "errors": 0}


# --------------------------------------------------------------------------- #
# Integration test — real catalog (skipped if absent)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Reviewer-flagged gaps (added by QA pass)
# --------------------------------------------------------------------------- #


def test_export_auto_missing_source_writes_no_xmp(
    synthetic_catalog: Path, tmp_path: Path
) -> None:
    """Photo H lives under a nonexistent root. Confirm that NO sidecar file
    is created anywhere — neither at the (nonexistent) source path, nor at
    its would-be sidecar path.
    """
    conn = connect_catalog(synthetic_catalog)
    photos = find_edited_photos(conn)
    by_id = _index_by_id(photos)
    # Force-flag has_develop_settings so the missing branch is exercised.
    if PHOTO_H_MISSING in by_id:
        by_id[PHOTO_H_MISSING]["has_develop_settings"] = True

    photo_h = by_id[PHOTO_H_MISSING]
    expected_sidecar = photo_h["file_path"].with_suffix(".xmp")

    summary = export_xmps_for_photos(conn, list(by_id.values()), output_method="auto")

    # Counted as skipped, not written, not errored.
    assert summary["skipped_missing"] >= 1
    # The XMP file MUST NOT exist on disk under the missing root.
    assert not expected_sidecar.exists()
    # And the parent directory of the missing root must not have been created.
    assert not photo_h["file_path"].parent.exists()


def test_get_develop_settings_multiple_active_rows(synthetic_catalog: Path) -> None:
    """If the catalog has multiple rows with beforeDigest IS NULL for the
    same image (which shouldn't normally happen, but is allowed by the schema),
    get_develop_settings must still return a valid dict without crashing.
    """
    # Inject a second beforeDigest=NULL row for photo A using a direct
    # (non-readonly) connection. Use a different blob so we can confirm it
    # parsed *some* valid row.
    second_blob = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" '
        'rdf:about="" crs:Exposure2012="+0.50" crs:Temperature="5500" crs:Tint="0"/>'
        '</rdf:RDF></x:xmpmeta>'
    )
    raw_conn = sqlite3.connect(str(synthetic_catalog))
    raw_conn.execute(
        "INSERT INTO Adobe_imageDevelopSettings "
        "(image, hasDevelopSettings, beforeDigest, developSettings) "
        "VALUES (?, ?, NULL, ?)",
        (PHOTO_A_EDITED_XMP, 1, second_blob),
    )
    raw_conn.commit()
    raw_conn.close()

    conn = connect_catalog(synthetic_catalog)
    # Must not raise. Which row wins is undefined, but the call must succeed
    # and return a normalised dict.
    settings = get_develop_settings(conn, PHOTO_A_EDITED_XMP)
    assert set(settings.keys()) == set(SLIDER_FIELDS)
    # Both candidate rows have Exposure2012=0.5, so this assertion is robust.
    assert settings["Exposure2012"] == pytest.approx(0.5)


def test_build_path_empty_root_raises() -> None:
    """An empty root_path (e.g. unmounted volume) must surface as CatalogError,
    not as a silent Path(\"folder/file\") that points to the cwd.
    """
    with pytest.raises(CatalogError, match="empty absolutePath"):
        _build_path("", "2024/", "photo.cr3")


def test_export_invalid_output_method_raises(synthetic_catalog: Path) -> None:
    conn = connect_catalog(synthetic_catalog)
    photos = find_edited_photos(conn)
    with pytest.raises(ValueError, match="Unknown output_method"):
        export_xmps_for_photos(conn, photos, output_method="invalid")


def test_get_develop_settings_shape_matches_read_xmp(
    synthetic_catalog: Path, tmp_path: Path
) -> None:
    """The dict returned by get_develop_settings must have the same keys and
    matching types (per-key, for non-None values) as xmp.read_xmp on a sidecar
    written by xmp.write_xmp from those same settings.
    """
    conn = connect_catalog(synthetic_catalog)
    catalog_settings = get_develop_settings(conn, PHOTO_K_ALL_FIELDS)

    # Round-trip through the XMP writer/reader.
    sidecar = tmp_path / "shape_check.xmp"
    xmp_module.write_xmp(sidecar, catalog_settings, source_raw_path=None)
    xmp_settings = xmp_module.read_xmp(sidecar)

    # Same key set.
    assert set(catalog_settings.keys()) == set(xmp_settings.keys())
    assert set(catalog_settings.keys()) == set(SLIDER_FIELDS)

    # Per-key type agreement for non-None values on both sides.
    for field in SLIDER_FIELDS:
        cv = catalog_settings[field]
        xv = xmp_settings[field]
        if cv is None or xv is None:
            # Either side may legitimately be None; nothing to compare.
            continue
        assert type(cv) is type(xv), (
            f"type mismatch on {field}: catalog={type(cv).__name__} "
            f"xmp={type(xv).__name__}"
        )


# --------------------------------------------------------------------------- #
# _extract_lua_table — last-match-wins regression test
# --------------------------------------------------------------------------- #


def test_extract_lua_table_picks_last_match_when_nested_block_precedes() -> None:
    """LR blobs include a nested Look/Profile reference block that ALSO carries
    its own stub tone curves. The nested block always serialises before the
    active top-level develop settings, so the parser must pick the LAST match
    to recover the user's actual curve and ignore the profile-default stub.
    Regression for the bug discovered on 0H5A8434.CR3 (Medik8 Launch).
    """
    blob = (
        "s = {\n"
        # Nested Look/Profile reference — stub identity curves
        "Look = { Group = ZSTR \"Profiles:Camera Matching\",\n"
        "Name = \"Camera Neutral\",\n"
        "Parameters = { ToneCurveName2012 = \"\",\n"
        "ToneCurvePV2012 = { 0, 0, 22, 16, 40, 35, 255, 255 },\n"
        "ToneCurvePV2012Red = { 0, 0, 255, 255 },\n"
        "ToneCurvePV2012Green = { 0, 0, 255, 255 },\n"
        "ToneCurvePV2012Blue = { 0, 0, 255, 255 },\n"
        "Version = \"18.2\" },\n"
        "SupportsAmount = false,\n"
        "UUID = \"NESTED_UUID\" },\n"
        # Active develop settings — user's real curves
        "ToneCurveName2012 = \"Custom\",\n"
        "ToneCurvePV2012 = { 0, 19, 60, 64, 153, 158, 255, 247 },\n"
        "ToneCurvePV2012Red = { 0, 0, 53, 30, 120, 122, 255, 255 },\n"
        "ToneCurvePV2012Green = { 0, 0, 52, 30, 118, 120, 255, 255 },\n"
        "ToneCurvePV2012Blue = { 0, 0, 48, 23, 123, 124, 255, 255 },\n"
        "Exposure2012 = 0.5,\n"
        "}"
    )
    composite = _extract_lua_table(blob, "ToneCurvePV2012")
    red       = _extract_lua_table(blob, "ToneCurvePV2012Red")
    green     = _extract_lua_table(blob, "ToneCurvePV2012Green")
    blue      = _extract_lua_table(blob, "ToneCurvePV2012Blue")
    # Must pick the active (last) curves, not the nested profile stubs
    assert composite == [(0, 19), (60, 64), (153, 158), (255, 247)]
    assert red       == [(0, 0), (53, 30), (120, 122), (255, 255)]
    assert green     == [(0, 0), (52, 30), (118, 120), (255, 255)]
    assert blue      == [(0, 0), (48, 23), (123, 124), (255, 255)]


def test_extract_lua_table_single_occurrence_still_works() -> None:
    """When the key appears only once, the last-match logic must still return it."""
    blob = "s = { ToneCurvePV2012 = { 0, 0, 128, 200, 255, 255 } }"
    assert _extract_lua_table(blob, "ToneCurvePV2012") == [(0, 0), (128, 200), (255, 255)]


def test_extract_lua_table_missing_key_returns_empty() -> None:
    blob = "s = { Exposure2012 = 0.5 }"
    assert _extract_lua_table(blob, "ToneCurvePV2012Red") == []


@pytest.mark.integration
def test_real_catalog(tmp_path: Path) -> None:  # noqa: ARG001
    lrcat = Path.home() / "sonnaeditor/test_data/sonnaeditor08_05/sonnaeditor08_05.lrcat"
    if not lrcat.exists():
        pytest.skip("Real catalog not present")
    conn = connect_catalog(lrcat)
    photos = find_edited_photos(conn)
    assert len(photos) > 0
    edited = [p for p in photos if p["has_develop_settings"]]
    if edited:
        settings = get_develop_settings(conn, edited[0]["image_id"])
        assert set(settings.keys()) == set(SLIDER_FIELDS)
