"""Lightroom Classic catalog (.lrcat) reader.

Read-only access to Adobe Lightroom Classic catalogs. Extracts edited photos
and their develop settings without ever modifying the catalog.

Three-layer read-only enforcement:
1. Refuses to open if a sibling .lrcat.lock file is present.
2. Opens via SQLite URI with mode=ro.
3. Issues PRAGMA query_only = 1.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

from sonna_editor.config import SLIDER_FIELDS
from sonna_editor.data import xmp
from sonna_editor.data.xmp import (
    _IDENTITY_CURVE_POINTS,
    _normalize_curve,
    _parse_xmp_bytes,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class CatalogError(RuntimeError):
    """Base exception for catalog-related errors."""


class CatalogLockedError(CatalogError):
    """Raised when the catalog is locked (Lightroom is open)."""


class CatalogVersionError(CatalogError):
    """Raised when the catalog schema is unrecognised or unsupported."""


class DevelopSettingsParseError(CatalogError):
    """Raised when develop settings blob cannot be parsed."""


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #


_REQUIRED_TABLES = (
    "Adobe_images",
    "AgLibraryFile",
    "AgLibraryFolder",
    "AgLibraryRootFolder",
    "Adobe_imageDevelopSettings",
)

# LR 11 - 15+ known DB version range
_KNOWN_DB_VERSION_MIN = "0800000"
_KNOWN_DB_VERSION_MAX = "9999999"

# LR 15+ schema: active settings row is identified by id_local = Adobe_images.developSettingsIDCache
# LR 11-14 schema: active row identified by image = id_local AND beforeDigest IS NULL
# Settings text column: "text" (LR 15+) or "developSettings" (LR 11-14)
# Has-settings flag: check text IS NOT NULL (LR 15+) or hasDevelopSettings (LR 11-14)

_FIND_PHOTOS_SQL_V2 = """
SELECT
    ai.id_local                                              AS image_id,
    ai.captureTime                                           AS capture_time,
    COALESCE(ai.rating, 0)                                   AS rating,
    COALESCE(ai.colorLabels, '')                             AS color_label,
    COALESCE(ai.pick, 0)                                     AS pick_flag,
    rf.absolutePath                                          AS root_path,
    fol.pathFromRoot                                         AS folder_path,
    af.idx_filename                                          AS filename,
    CASE WHEN ids.id_local IS NOT NULL
         AND ids.text IS NOT NULL AND ids.text != ''
         THEN 1 ELSE 0 END                                   AS has_develop_settings
FROM Adobe_images AS ai
JOIN AgLibraryFile        AS af  ON af.id_local  = ai.rootFile
JOIN AgLibraryFolder      AS fol ON fol.id_local = af.folder
JOIN AgLibraryRootFolder  AS rf  ON rf.id_local  = fol.rootFolder
LEFT JOIN Adobe_imageDevelopSettings AS ids
       ON ids.id_local = ai.developSettingsIDCache
WHERE COALESCE(ai.pick, 0) > -1
  AND COALESCE(ai.rating, 0) >= :min_rating
  AND (:min_flag IS NULL OR COALESCE(ai.pick, 0) >= :min_flag)
  AND (:min_color_label IS NULL OR COALESCE(ai.colorLabels, '') = :min_color_label)
ORDER BY ai.captureTime ASC, ai.id_local ASC
"""

_FIND_PHOTOS_SQL_V1 = """
SELECT
    ai.id_local                                              AS image_id,
    ai.captureTime                                           AS capture_time,
    COALESCE(ai.rating, 0)                                   AS rating,
    COALESCE(ai.colorLabels, '')                             AS color_label,
    COALESCE(ai.pick, 0)                                     AS pick_flag,
    rf.absolutePath                                          AS root_path,
    fol.pathFromRoot                                         AS folder_path,
    af.idx_filename                                          AS filename,
    CASE WHEN ids.image IS NOT NULL
         AND COALESCE(ids.hasDevelopSettings, 0) = 1
         THEN 1 ELSE 0 END                                   AS has_develop_settings
FROM Adobe_images AS ai
JOIN AgLibraryFile        AS af  ON af.id_local  = ai.rootFile
JOIN AgLibraryFolder      AS fol ON fol.id_local = af.folder
JOIN AgLibraryRootFolder  AS rf  ON rf.id_local  = fol.rootFolder
LEFT JOIN Adobe_imageDevelopSettings AS ids
       ON ids.image = ai.id_local
      AND ids.beforeDigest IS NULL
WHERE COALESCE(ai.pick, 0) > -1
  AND COALESCE(ai.rating, 0) >= :min_rating
  AND (:min_flag IS NULL OR COALESCE(ai.pick, 0) >= :min_flag)
  AND (:min_color_label IS NULL OR COALESCE(ai.colorLabels, '') = :min_color_label)
ORDER BY ai.captureTime ASC, ai.id_local ASC
"""

_DEVELOP_SETTINGS_SQL_V2 = """
SELECT ids.text FROM Adobe_imageDevelopSettings ids
JOIN Adobe_images ai ON ids.id_local = ai.developSettingsIDCache
WHERE ai.id_local = ?
"""

_DEVELOP_SETTINGS_SQL_V1 = """
SELECT developSettings FROM Adobe_imageDevelopSettings
WHERE image = ? AND beforeDigest IS NULL
"""


_LUA_VALUE_RE = re.compile(
    r'^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*('
    r'"(?:[^"\\]|\\.)*"'
    r'|[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
    r'|true|false'
    r')\s*}?\s*,?\s*$',   # }? allows last field in nested table: "Whites2012 = -70 }"
    re.MULTILINE,
)

# Maps Lua key names (tone curve table keys in catalog blobs) → SLIDER_FIELDS prefix
_LUA_CURVE_KEYS: dict[str, str] = {
    "ToneCurvePV2012":      "ToneCurve",
    "ToneCurvePV2012Red":   "ToneCurveRed",
    "ToneCurvePV2012Green": "ToneCurveGreen",
    "ToneCurvePV2012Blue":  "ToneCurveBlue",
}


# --------------------------------------------------------------------------- #
# Connection / schema verification
# --------------------------------------------------------------------------- #


def _check_lock_file(path: Path) -> None:
    """Raise CatalogLockedError if any sibling lock/WAL file exists.

    Lightroom leaves behind .lrcat.lock, .lrcat-wal, .lrcat-shm, or
    .lrcat-journal when it is running or was force-killed. Any of these
    means we cannot safely read. Note: this check is best-effort — always
    close Lightroom Classic before running Sonna Editor.
    """
    # .lock = Lightroom's own advisory lock (LR is definitely running)
    # -journal = SQLite rollback journal (active write transaction)
    # -wal / -shm = SQLite WAL mode files (normal even when LR is closed; ignore)
    suffixes = [".lock", "-journal"]
    found = [
        str(path.parent / (path.name + s))
        for s in suffixes
        if (path.parent / (path.name + s)).exists()
    ]
    if found:
        raise CatalogLockedError(
            f"Lightroom appears to be open (lock files found: {', '.join(found)}). "
            "Close Lightroom Classic completely and retry."
        )


def _read_db_version(conn: sqlite3.Connection) -> str | None:
    """Return Adobe_DBVersion if present, else None."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='Adobe_variablesTable'"
    )
    if cur.fetchone() is None:
        logger.info(
            "Adobe_variablesTable not found; catalog version is unknown — proceeding optimistically."
        )
        return None
    cur = conn.execute(
        "SELECT value FROM Adobe_variablesTable WHERE name = 'Adobe_DBVersion'"
    )
    row = cur.fetchone()
    if row is None:
        return None
    return str(row[0]) if row[0] is not None else None


def _detect_schema_version(conn: sqlite3.Connection) -> str:
    """Return 'v2' for LR 15+ schema, 'v1' for LR 11-14 schema.

    v2 is identified by the presence of developSettingsIDCache on Adobe_images
    and the text column on Adobe_imageDevelopSettings.
    """
    img_cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(Adobe_images)")
    }
    ids_cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(Adobe_imageDevelopSettings)")
    }
    if "developSettingsIDCache" in img_cols and "text" in ids_cols:
        logger.debug("Detected LR 15+ catalog schema (v2).")
        return "v2"
    logger.debug("Detected LR 11-14 catalog schema (v1).")
    return "v1"


def _verify_schema(conn: sqlite3.Connection) -> None:
    """Verify required tables exist; warn if DB version is outside known range."""
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing = {row[0] for row in cur.fetchall()}

    for table in _REQUIRED_TABLES:
        if table not in existing:
            raise CatalogVersionError(f"Required table missing: {table}")

    db_version = _read_db_version(conn)
    if db_version is not None:
        if not (_KNOWN_DB_VERSION_MIN <= db_version <= _KNOWN_DB_VERSION_MAX):
            logger.warning(
                "Catalog Adobe_DBVersion=%s is outside known-tested range "
                "(%s..%s). Proceeding with schema auto-detection.",
                db_version, _KNOWN_DB_VERSION_MIN, _KNOWN_DB_VERSION_MAX,
            )


def connect_catalog(lrcat_path: Path) -> sqlite3.Connection:
    """Open a Lightroom catalog read-only.

    Raises CatalogError if the path is not a .lrcat file or does not exist.
    Raises CatalogLockedError if Lightroom is open or SQLite cannot acquire
    a read lock. Raises CatalogVersionError if the schema is missing required
    tables.
    """
    lrcat_path = Path(lrcat_path)

    if lrcat_path.suffix.lower() != ".lrcat":
        raise CatalogError(
            f"{lrcat_path} is not a Lightroom catalog (.lrcat file). "
            "Find your catalog in Lightroom → Edit → Catalog Settings → Location."
        )
    if not lrcat_path.exists():
        raise CatalogError(f"Catalog not found: {lrcat_path}")
    if not lrcat_path.is_file():
        raise CatalogError(f"Expected a file, got a directory: {lrcat_path}")

    _check_lock_file(lrcat_path)

    uri = f"file:{lrcat_path.resolve()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=2.0)
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        if "locked" in msg or "unable to open" in msg:
            raise CatalogLockedError(
                f"Could not open catalog read-only ({e}). "
                "Is Lightroom Classic running?"
            ) from e
        raise

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = 1;")
    _verify_schema(conn)
    return conn


# --------------------------------------------------------------------------- #
# Photo discovery
# --------------------------------------------------------------------------- #


def _build_path(root: str, folder: str, filename: str) -> Path:
    if not root:
        raise CatalogError(
            f"Catalog contains a root folder with an empty absolutePath "
            f"(folder={folder!r}, file={filename!r}). The volume may be unmounted."
        )
    return Path(root) / folder / filename


def find_edited_photos(
    conn: sqlite3.Connection,
    min_color_label: str | None = None,
    min_rating: int = 0,
    min_flag: int | None = None,
) -> list[dict[str, Any]]:
    """Return photo rows from the catalog matching the given filters.

    Each dict has keys:
        image_id, file_path, capture_time, rating, color_label, pick_flag,
        has_develop_settings, is_missing
    """
    params = {
        "min_rating": min_rating,
        "min_flag": min_flag,
        "min_color_label": min_color_label,
    }
    sql = _FIND_PHOTOS_SQL_V2 if _detect_schema_version(conn) == "v2" else _FIND_PHOTOS_SQL_V1
    cur = conn.execute(sql, params)
    out: list[dict[str, Any]] = []
    for row in cur:
        file_path = _build_path(row["root_path"], row["folder_path"], row["filename"])
        out.append({
            "image_id": int(row["image_id"]),
            "file_path": file_path,
            "capture_time": row["capture_time"],
            "rating": int(row["rating"]),
            "color_label": row["color_label"],
            "pick_flag": int(row["pick_flag"]),
            "has_develop_settings": bool(row["has_develop_settings"]),
            "is_missing": not file_path.exists(),
        })
    return out


# --------------------------------------------------------------------------- #
# Develop settings parsing
# --------------------------------------------------------------------------- #


def _parse_xmp_blob(text: str) -> dict[str, float | str | None]:
    return _parse_xmp_bytes(text.encode("utf-8"))


def _extract_lua_table(blob: str, key: str) -> list[tuple[int, int]]:
    """Extract a Lua nested-table value as a list of (x, y) int pairs.

    LR catalog stores tone curves as e.g.:
        ToneCurvePV2012 = {
          0, 19,
          64, 77,
          ...
        }
    Uses bracket-counting to robustly extract the table body.
    Returns [] if the key is not found or has no numeric content.

    Picks the LAST occurrence of the key in the blob. LR serialises the active
    develop settings at the top level of the blob, but a blob can ALSO contain
    nested Look/Profile-reference blocks (with `Version = "..."`,
    `SupportsAmount`, `UUID = "..."`) that include their own stub tone curves
    — those blocks always appear earlier in the serialisation, so picking the
    last match correctly recovers the user's actual curves and ignores the
    profile-default stubs. This matches the existing scalar-field behaviour in
    `_parse_lua_blob`, which uses `re.finditer` and lets the later
    occurrence overwrite the earlier one.
    """
    pattern = key + r"\s*=\s*\{"
    last: re.Match | None = None
    for cand in re.finditer(pattern, blob):
        last = cand
    if last is None:
        return []
    m = last
    start = m.end() - 1  # position of the opening '{'
    depth = 0
    end = start
    for i in range(start, len(blob)):
        if blob[i] == "{":
            depth += 1
        elif blob[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    body = blob[start + 1:end]
    nums = re.findall(r"\d+", body)
    if len(nums) < 2:
        return []
    pairs = [(int(nums[i]), int(nums[i + 1])) for i in range(0, len(nums) - 1, 2)]
    return pairs


def _parse_lua_blob(text: str) -> dict[str, str | float]:
    """Extract scalar and tone-curve fields from a Lua-table develop settings blob."""
    raw: dict[str, str | float] = {}
    for m in _LUA_VALUE_RE.finditer(text):
        name, value = m.group(1), m.group(2)
        raw[name] = value

    # Extract tone curve channels from nested Lua tables
    for lua_key, prefix in _LUA_CURVE_KEYS.items():
        points = _extract_lua_table(text, lua_key)
        normalized = _normalize_curve(points) or list(_IDENTITY_CURVE_POINTS)
        for n, (px, py) in enumerate(normalized, start=1):
            raw[f"{prefix}_Pt{n}_X"] = float(px)
            raw[f"{prefix}_Pt{n}_Y"] = float(py)

    return raw


def _normalise_to_slider_fields(
    raw: dict[str, Any],
) -> dict[str, float | str | None]:
    """Project an arbitrary parsed-settings dict onto SLIDER_FIELDS keys."""
    result: dict[str, float | str | None] = {f: None for f in SLIDER_FIELDS}
    for f in SLIDER_FIELDS:
        val = raw.get(f)
        if val is None:
            continue
        # Strip surrounding quotes from Lua string literals
        if isinstance(val, str) and val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        try:
            result[f] = float(val)
        except (TypeError, ValueError):
            result[f] = str(val) if val is not None else None
    return result


def get_develop_settings(
    conn: sqlite3.Connection,
    image_id: int,
) -> dict[str, float | str | None]:
    """Return the active (current) develop settings for an image.

    History snapshots (rows where beforeDigest IS NOT NULL) are ignored.
    """
    sql = _DEVELOP_SETTINGS_SQL_V2 if _detect_schema_version(conn) == "v2" else _DEVELOP_SETTINGS_SQL_V1
    cur = conn.execute(sql, (image_id,))
    row = cur.fetchone()
    if row is None:
        raise DevelopSettingsParseError(
            f"No develop settings row found for image_id={image_id}"
        )

    blob = row[0]
    if blob is None:
        raise DevelopSettingsParseError(
            f"Develop settings blob is NULL for image_id={image_id}"
        )

    if isinstance(blob, bytes):
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError as e:
            raise DevelopSettingsParseError(
                f"Develop settings blob is not valid UTF-8 for image_id={image_id}: {e}"
            ) from e
    else:
        text = str(blob)

    stripped = text.strip()
    if not stripped:
        raise DevelopSettingsParseError(
            f"Develop settings blob is empty for image_id={image_id}"
        )

    if (stripped.startswith("<?xpacket")
            or stripped.startswith("<x:xmpmeta")
            or stripped.startswith("<rdf:")):
        raw = _parse_xmp_blob(text)
    elif (stripped.startswith("s ")
          or stripped.startswith("s=")
          or stripped.startswith("s={")):
        raw = _parse_lua_blob(text)
    else:
        raise DevelopSettingsParseError(
            f"unrecognised develop settings format: {text[:60]!r}"
        )

    return _normalise_to_slider_fields(raw)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def export_xmps_for_photos(
    conn: sqlite3.Connection,
    photos: list[dict[str, Any]],
    output_method: str = "manual",
) -> dict[str, int]:
    """Export develop settings as XMP sidecars next to each RAW.

    output_method:
        "manual" — print Lightroom instructions to stdout (does not write).
                   Returns {"written": 0, ...} with all counts zero.
        "auto"   — write XMP sidecars directly from catalog data.
                   Skips photos with missing files or pre-existing sidecars.
                   Never overwrites an existing sidecar.

    Returns a summary dict with keys: written, skipped_existing,
    skipped_missing, errors. Inspect "errors" before assuming success.
    """
    if output_method == "manual":
        print(
            "Manual XMP export instructions:\n"
            "  1. In Lightroom Classic, select all the relevant photos.\n"
            "  2. Metadata -> Save Metadata to Files (Cmd+S).\n"
            "  3. Wait for Lightroom to finish writing .xmp sidecars next to the RAWs.\n"
            "  4. Re-run the ingestion pipeline; it will read the sidecars directly.\n"
            f"  -> {len(photos)} photos in scope."
        )
        return {"written": 0, "skipped_existing": 0, "skipped_missing": 0, "errors": 0}

    if output_method != "auto":
        raise ValueError(f"Unknown output_method: {output_method!r}")

    written = 0
    skipped_existing = 0
    skipped_missing = 0
    errors = 0
    error_details: list[str] = []

    for photo in photos:
        if not photo.get("has_develop_settings"):
            continue
        if photo.get("is_missing"):
            skipped_missing += 1
            logger.warning(
                "Skipping image_id=%s: source file is missing (%s)",
                photo["image_id"], photo["file_path"],
            )
            continue

        sidecar_path = photo["file_path"].with_suffix(".xmp")
        if sidecar_path.exists():
            skipped_existing += 1
            logger.warning(
                "Skipping image_id=%s: sidecar already exists at %s (not overwriting)",
                photo["image_id"], sidecar_path,
            )
            continue

        try:
            settings = get_develop_settings(conn, photo["image_id"])
            xmp.write_xmp(sidecar_path, settings, source_raw_path=photo["file_path"])
            written += 1
        except Exception as e:  # noqa: BLE001
            errors += 1
            detail = f"image_id={photo['image_id']} ({photo['file_path']}): {e}"
            error_details.append(detail)
            logger.error("Failed to write XMP for %s", detail)

    summary = {
        "written": written,
        "skipped_existing": skipped_existing,
        "skipped_missing": skipped_missing,
        "errors": errors,
    }
    if errors:
        logger.error(
            "XMP export finished with %d error(s). Details: %s",
            errors, "; ".join(error_details),
        )
    logger.info(
        "XMP export summary: %d written, %d skipped (existing), "
        "%d skipped (missing), %d errors.",
        written, skipped_existing, skipped_missing, errors,
    )
    return summary
