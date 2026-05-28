from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sonna_editor.data.dng import (
    DNGConversionError,
    DNGConverterNotFoundError,
    batch_convert,
    convert_to_dng,
    get_dng_converter_version,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


# ---------------------------------------------------------------------------
# get_dng_converter_version
# ---------------------------------------------------------------------------

class TestGetVersion:
    def test_returns_version_string(self, tmp_path):
        fake_bin = tmp_path / "Adobe DNG Converter"
        fake_bin.touch()
        with (
            patch("sonna_editor.data.dng.DNG_CONVERTER_PATH", fake_bin),
            patch("subprocess.run", return_value=_make_completed(stdout="Adobe DNG Converter version 16.0")) as mock_run,
        ):
            version = get_dng_converter_version()
        assert "16.0" in version
        mock_run.assert_called_once()

    def test_raises_if_binary_missing(self, tmp_path):
        missing = tmp_path / "nonexistent"
        with patch("sonna_editor.data.dng.DNG_CONVERTER_PATH", missing):
            with pytest.raises(DNGConverterNotFoundError):
                get_dng_converter_version()


# ---------------------------------------------------------------------------
# convert_to_dng
# ---------------------------------------------------------------------------

class TestConvertToDng:
    def _patch_binary(self, tmp_path: Path):
        fake_bin = tmp_path / "Adobe DNG Converter"
        fake_bin.touch()
        return patch("sonna_editor.data.dng.DNG_CONVERTER_PATH", fake_bin)

    def test_basic_command_construction(self, tmp_path):
        input_file = tmp_path / "IMG_001.CR3"
        input_file.touch()
        output_dir = tmp_path / "dng"
        expected_output = output_dir / "IMG_001.dng"

        def fake_run(cmd, **kwargs):
            expected_output.touch()
            return _make_completed()

        with (
            self._patch_binary(tmp_path),
            patch("subprocess.run", side_effect=fake_run) as mock_run,
        ):
            result = convert_to_dng(input_file, output_dir)

        assert result == expected_output
        cmd = mock_run.call_args[0][0]
        assert "-c" in cmd          # compress on by default
        assert "-d" in cmd
        assert str(output_dir) in cmd
        assert str(input_file) in cmd

    def test_no_compress_flag(self, tmp_path):
        input_file = tmp_path / "IMG_001.CR3"
        input_file.touch()
        output_dir = tmp_path / "dng"
        expected_output = output_dir / "IMG_001.dng"

        def fake_run(cmd, **kwargs):
            expected_output.touch()
            return _make_completed()

        with (
            self._patch_binary(tmp_path),
            patch("subprocess.run", side_effect=fake_run) as mock_run,
        ):
            convert_to_dng(input_file, output_dir, compress=False)

        cmd = mock_run.call_args[0][0]
        assert "-c" not in cmd

    def test_embed_original_flag(self, tmp_path):
        input_file = tmp_path / "IMG_001.NEF"
        input_file.touch()
        output_dir = tmp_path / "dng"
        expected_output = output_dir / "IMG_001.dng"

        def fake_run(cmd, **kwargs):
            expected_output.touch()
            return _make_completed()

        with (
            self._patch_binary(tmp_path),
            patch("subprocess.run", side_effect=fake_run) as mock_run,
        ):
            convert_to_dng(input_file, output_dir, embed_original=True)

        cmd = mock_run.call_args[0][0]
        assert "-e" in cmd

    def test_raises_on_nonzero_exit(self, tmp_path):
        input_file = tmp_path / "IMG_001.CR3"
        input_file.touch()
        output_dir = tmp_path / "dng"

        with (
            self._patch_binary(tmp_path),
            patch("subprocess.run", return_value=_make_completed(returncode=1, stderr="error")),
        ):
            with pytest.raises(DNGConversionError, match="exit 1"):
                convert_to_dng(input_file, output_dir)

    def test_raises_if_output_missing_after_success(self, tmp_path):
        input_file = tmp_path / "IMG_001.CR3"
        input_file.touch()
        output_dir = tmp_path / "dng"

        with (
            self._patch_binary(tmp_path),
            patch("subprocess.run", return_value=_make_completed()),
        ):
            with pytest.raises(DNGConversionError, match="output not found"):
                convert_to_dng(input_file, output_dir)

    def test_raises_on_unsupported_format(self, tmp_path):
        input_file = tmp_path / "photo.jpg"
        input_file.touch()
        output_dir = tmp_path / "dng"

        with self._patch_binary(tmp_path):
            with pytest.raises(ValueError, match="Unsupported"):
                convert_to_dng(input_file, output_dir)

    def test_raises_if_binary_missing(self, tmp_path):
        input_file = tmp_path / "IMG_001.CR3"
        input_file.touch()
        missing = tmp_path / "nonexistent"
        with patch("sonna_editor.data.dng.DNG_CONVERTER_PATH", missing):
            with pytest.raises(DNGConverterNotFoundError):
                convert_to_dng(input_file, tmp_path)

    def test_creates_output_dir(self, tmp_path):
        input_file = tmp_path / "IMG_001.CR3"
        input_file.touch()
        output_dir = tmp_path / "nested" / "dng"
        expected_output = output_dir / "IMG_001.dng"

        def fake_run(cmd, **kwargs):
            expected_output.parent.mkdir(parents=True, exist_ok=True)
            expected_output.touch()
            return _make_completed()

        with (
            self._patch_binary(tmp_path),
            patch("subprocess.run", side_effect=fake_run),
        ):
            result = convert_to_dng(input_file, output_dir)

        assert result == expected_output


# ---------------------------------------------------------------------------
# batch_convert
# ---------------------------------------------------------------------------

class TestBatchConvert:
    def _patch_binary(self, tmp_path: Path):
        fake_bin = tmp_path / "Adobe DNG Converter"
        fake_bin.touch()
        return patch("sonna_editor.data.dng.DNG_CONVERTER_PATH", fake_bin)

    def _patch_pool(self, imap_results: list):
        """Patch multiprocessing.Pool so imap returns a fixed list without forking."""
        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.imap.return_value = iter(imap_results)
        return patch("sonna_editor.data.dng.Pool", return_value=mock_pool)

    def test_returns_list_same_length(self, tmp_path):
        inputs = [tmp_path / f"IMG_{i:03d}.CR3" for i in range(3)]
        output_dir = tmp_path / "dng"
        fake_outputs = [output_dir / f"IMG_{i:03d}.dng" for i in range(3)]

        with self._patch_binary(tmp_path), self._patch_pool(fake_outputs):
            results = batch_convert(inputs, output_dir, max_workers=2)

        assert len(results) == 3
        assert results == fake_outputs

    def test_writes_failure_log_on_errors(self, tmp_path):
        inputs = [tmp_path / "IMG_001.CR3", tmp_path / "IMG_002.CR3"]
        output_dir = tmp_path / "dng"
        output_dir.mkdir(parents=True, exist_ok=True)
        # First succeeds, second fails (None)
        fake_outputs = [output_dir / "IMG_001.dng", None]

        with self._patch_binary(tmp_path), self._patch_pool(fake_outputs):
            results = batch_convert(inputs, output_dir, max_workers=1)

        assert results[1] is None
        assert (output_dir / "failures.log").exists()

    def test_no_failure_log_when_all_succeed(self, tmp_path):
        inputs = [tmp_path / "IMG_001.CR3"]
        output_dir = tmp_path / "dng"
        output_dir.mkdir(parents=True, exist_ok=True)
        fake_outputs = [output_dir / "IMG_001.dng"]

        with self._patch_binary(tmp_path), self._patch_pool(fake_outputs):
            batch_convert(inputs, output_dir, max_workers=1)

        assert not (output_dir / "failures.log").exists()


# ---------------------------------------------------------------------------
# Integration test (skipped by default — requires real DNG Converter + RAW)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_real_conversion(tmp_path):
    """Convert a real CR3 to DNG. Requires a RAW file at the path below."""
    input_path = Path("/Users/darshil/Desktop/0S6A6020.CR3")
    if not input_path.exists():
        pytest.skip("Test RAW file not present")

    output = convert_to_dng(input_path, tmp_path / "dng")
    assert output.exists()
    assert output.suffix == ".dng"
    assert output.stat().st_size > 0
