"""Migration script unit tests (commit d681429)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from sonna_editor import config


# Load the migration script as a module (it lives in scripts/, not src/).
_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "migrate_labels_to_v2.py"
_spec = importlib.util.spec_from_file_location("migrate_labels_to_v2", _SCRIPT_PATH)
assert _spec is not None
assert _spec.loader is not None
_migrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_migrate)


def _make_v1_parquet(path: Path, rows: int = 3) -> None:
    """Build a synthetic v1 parquet (135 slider columns + raw_path)."""
    data: dict = {"raw_path": [f"/nonexistent/raw_{i}.cr3" for i in range(rows)]}
    for field in config.SLIDER_FIELDS[:135]:
        data[field] = [0.0] * rows
    pd.DataFrame(data).to_parquet(path, index=False)


class TestMigrateLabelsToV2:
    def test_v2_extension_fields_count(self) -> None:
        assert len(_migrate.V2_EXTENSION_FIELDS) == 12
        assert _migrate.V2_EXTENSION_FIELDS == list(config.SLIDER_FIELDS[135:])

    def test_migrate_v1_parquet_adds_12_columns(self, tmp_path: Path) -> None:
        """v1 input gets 12 new columns; non-existent RAW paths fall back
        to SLIDER_DEFAULTS for all 12 v2 fields."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        _make_v1_parquet(in_dir / "val.parquet", rows=3)

        result = _migrate.migrate_split(
            in_dir / "val.parquet", out_dir / "val.parquet",
            workers=2, limit=None, dry_run=False,
        )

        assert result["rows"] == 3
        assert result["error_rows"] == 3  # all RAW paths are non-existent
        for field in _migrate.V2_EXTENSION_FIELDS:
            assert result["fallback_counts"][field] == 3

        out_df = pd.read_parquet(out_dir / "val.parquet")
        for field in _migrate.V2_EXTENSION_FIELDS:
            assert field in out_df.columns
            assert all(out_df[field] == config.SLIDER_DEFAULTS[field])

    def test_migrate_idempotent_skips_v2_parquet(self, tmp_path: Path) -> None:
        """Parquet that already has all 12 v2 columns is skipped (no re-extract)."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        _make_v1_parquet(in_dir / "val.parquet", rows=2)

        # First migration writes a v2 output
        _migrate.migrate_split(
            in_dir / "val.parquet", out_dir / "val.parquet",
            workers=2, limit=None, dry_run=False,
        )

        # Now run migration AGAIN treating the v2 output as input
        result = _migrate.migrate_split(
            out_dir / "val.parquet", tmp_path / "out2" / "val.parquet",
            workers=2, limit=None, dry_run=False,
        )

        assert result.get("skipped") is True

    def test_migrate_preserves_original_parquet(self, tmp_path: Path) -> None:
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        src = in_dir / "val.parquet"
        _make_v1_parquet(src, rows=2)
        original_mtime = src.stat().st_mtime
        original_size = src.stat().st_size

        _migrate.migrate_split(
            src, out_dir / "val.parquet",
            workers=2, limit=None, dry_run=False,
        )

        assert src.stat().st_mtime == original_mtime
        assert src.stat().st_size == original_size

    def test_migrate_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        _make_v1_parquet(in_dir / "val.parquet", rows=2)

        _migrate.migrate_split(
            in_dir / "val.parquet", out_dir / "val.parquet",
            workers=2, limit=None, dry_run=True,
        )

        assert not (out_dir / "val.parquet").exists()
