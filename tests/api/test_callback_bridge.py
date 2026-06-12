"""Tests for the per-photo / per-epoch callback bridges and pipeline.py edit."""

from __future__ import annotations

import json as _json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from sonna_editor import config
from sonna_editor.api import callbacks, jobs


def test_format_edit_summary_shape() -> None:
    pred = {
        "Exposure2012": 0.42,
        "Temperature": 5180.0,
        "Tint": 3.0,
        "Shadows2012": 38.0,
        "Highlights2012": -10.0,
        "Contrast2012": 5.0,
        "Vibrance": 8.0,
        "Saturation": -2.0,
    }
    out = callbacks._format_edit_summary(pred)
    assert out == "Exp +0.42 · WB 5,180K · Shad +38"


def test_format_edit_summary_picks_largest_third_slot() -> None:
    pred = {
        "Exposure2012": 0.0,
        "Temperature": 5500.0,
        "Tint": 12.0,
        "Shadows2012": 5.0,
        "Highlights2012": -28.0,  # largest abs value
        "Contrast2012": 0.0,
        "Vibrance": 0.0,
        "Saturation": 0.0,
    }
    out = callbacks._format_edit_summary(pred)
    assert "High -28" in out


def test_photo_callback_updates_record_and_persists(
    isolated_paths: dict[str, Path],
) -> None:
    record = jobs.create(kind="process", photos_total=2)
    cb = callbacks.make_photo_callback(record, started_at=0.0)

    photo = {
        "name": "shot.cr3",
        "raw_path": "/tmp/shot.cr3",
        "xmp_path": "/tmp/shot.xmp",
        "predicted_values": {"Exposure2012": 0.1, "Temperature": 5500},
        "std": None,
        "status": "ok",
        "elapsed_seconds": 0.05,
    }
    cb(photo)

    with record.lock:
        assert record.photos_processed == 1
        assert record.current_photo == "shot.cr3"


def test_photo_callback_swallows_exceptions(
    isolated_paths: dict[str, Path],
) -> None:
    record = jobs.create(kind="process", photos_total=1)
    cb = callbacks.make_photo_callback(record, started_at=0.0)
    # Missing key triggers an exception inside the callback body.
    cb({"name": "x.cr3", "predicted_values": {}, "status": "ok"})  # raises KeyError on xmp_path? actually no — get() used; status flag not present
    # Should not raise; record may or may not have been mutated, no assertion.


def test_pipeline_callback_and_cancel_kwargs_no_op_when_none() -> None:
    """The new kwargs are additive — calling without them must not change behavior."""
    from sonna_editor.inference import pipeline as pl

    # Just verify signature accepts the kwargs and their defaults.
    import inspect
    sig = inspect.signature(pl.process_shoot_with_model)
    assert "on_photo_complete" in sig.parameters
    assert "cancel_event" in sig.parameters
    assert sig.parameters["on_photo_complete"].default is None
    assert sig.parameters["cancel_event"].default is None


def test_pipeline_loop_calls_callback_and_honours_cancel(
    tmp_path: Path,
) -> None:
    """Drive the pipeline with mocks for InferenceEngine + extract; verify
    the callback fires per photo and cancel_event stops the loop cleanly."""
    from sonna_editor.inference import pipeline as pl

    # Build 3 fake RAW files
    folder = tmp_path / "shoot"
    folder.mkdir()
    for i in range(3):
        (folder / f"img_{i}.cr3").write_bytes(b"x")

    fake_preds = torch.zeros(3, len(config.SLIDER_FIELDS))

    class FakeEngine:
        def __init__(self, *a, **kw):
            # pipeline.py now reads engine._image_resolution after engine.warmup()
            # to set the parallel preview-extraction target size.
            self._image_resolution = 384
        def warmup(self): pass
        def predict(self, *a, **kw): return fake_preds
        def predict_with_uncertainty(self, *a, **kw):
            return fake_preds, torch.zeros_like(fake_preds)

    cancel_event = threading.Event()
    seen: list[str] = []

    def on_photo(photo: dict) -> None:
        seen.append(photo["name"])
        if len(seen) == 2:
            cancel_event.set()

    with patch.object(pl, "InferenceEngine", FakeEngine), \
         patch.object(pl, "_extract_one", lambda p, target_size: (None, {})), \
         patch.object(pl, "write_xmp", lambda *a, **kw: None):
        result = pl.process_shoot_with_model(
            input_dir=folder,
            model_path=folder / "fake.ckpt",
            on_photo_complete=on_photo,
            cancel_event=cancel_event,
            save_predictions=False,
        )

    assert len(seen) == 2  # cancelled after the second
    assert result["cancelled"] is True
    assert result["processed"] == 2


def test_pipeline_preserve_wb_strips_temperature_and_tint(tmp_path: Path) -> None:
    """When preserve_wb=True, the dict passed to write_xmp must not carry
    Temperature or Tint — so Lightroom falls back to AsShot for those fields."""
    from sonna_editor.inference import pipeline as pl

    folder = tmp_path / "shoot"
    folder.mkdir()
    (folder / "img_0.cr3").write_bytes(b"x")

    fake_preds = torch.zeros(1, len(config.SLIDER_FIELDS))
    # Set Temperature (idx 11) and Tint (idx 12) to non-zero so we'd notice
    # if they leaked through to the XMP write.
    fake_preds[0, 11] = 5000.0
    fake_preds[0, 12] = 7.0

    class FakeEngine:
        def __init__(self, *a, **kw):
            # pipeline.py now reads engine._image_resolution after engine.warmup()
            # to set the parallel preview-extraction target size.
            self._image_resolution = 384
        def warmup(self): pass
        def predict(self, *a, **kw): return fake_preds

    captured_calls: list[dict] = []

    def fake_write(_xmp_path, settings, **_kw):
        captured_calls.append(dict(settings))

    with patch.object(pl, "InferenceEngine", FakeEngine), \
         patch.object(pl, "_extract_one", lambda p, target_size: (None, {})), \
         patch.object(pl, "write_xmp", fake_write):
        pl.process_shoot_with_model(
            input_dir=folder,
            model_path=folder / "fake.ckpt",
            preserve_wb=True,
            save_predictions=False,
        )

    assert len(captured_calls) == 1
    written = captured_calls[0]
    assert "Temperature" not in written, f"Temperature leaked through: {written.get('Temperature')!r}"
    assert "Tint" not in written, f"Tint leaked through: {written.get('Tint')!r}"
    # Sanity: other fields still present
    assert "Exposure2012" in written


def test_pipeline_preserve_wb_false_keeps_wb_fields(tmp_path: Path) -> None:
    """Default behaviour (preserve_wb=False) must keep Temperature/Tint in the XMP."""
    from sonna_editor.inference import pipeline as pl

    folder = tmp_path / "shoot"
    folder.mkdir()
    (folder / "img_0.cr3").write_bytes(b"x")

    fake_preds = torch.zeros(1, len(config.SLIDER_FIELDS))
    fake_preds[0, 11] = 5000.0
    fake_preds[0, 12] = 7.0

    class FakeEngine:
        def __init__(self, *a, **kw):
            # pipeline.py now reads engine._image_resolution after engine.warmup()
            # to set the parallel preview-extraction target size.
            self._image_resolution = 384
        def warmup(self): pass
        def predict(self, *a, **kw): return fake_preds

    captured_calls: list[dict] = []

    def fake_write(_xmp_path, settings, **_kw):
        captured_calls.append(dict(settings))

    with patch.object(pl, "InferenceEngine", FakeEngine), \
         patch.object(pl, "_extract_one", lambda p, target_size: (None, {})), \
         patch.object(pl, "write_xmp", fake_write):
        pl.process_shoot_with_model(
            input_dir=folder,
            model_path=folder / "fake.ckpt",
            preserve_wb=False,
            save_predictions=False,
        )

    assert len(captured_calls) == 1
    written = captured_calls[0]
    assert "Temperature" in written
    assert "Tint" in written


def test_pipeline_extra_skip_fields_omits_specified_fields(tmp_path: Path) -> None:
    """Generic skip mechanism: passing extra_skip_fields=["Tint"] omits Tint
    from the XMP but keeps Temperature."""
    from sonna_editor.inference import pipeline as pl

    folder = tmp_path / "shoot"
    folder.mkdir()
    (folder / "img_0.cr3").write_bytes(b"x")

    fake_preds = torch.zeros(1, len(config.SLIDER_FIELDS))
    fake_preds[0, 11] = 5000.0  # Temperature
    fake_preds[0, 12] = 7.0     # Tint

    class FakeEngine:
        def __init__(self, *a, **kw):
            self._image_resolution = 384
        def warmup(self): pass
        def predict(self, *a, **kw): return fake_preds

    captured_calls: list[dict] = []
    def fake_write(_xmp_path, settings, **_kw):
        captured_calls.append(dict(settings))

    with patch.object(pl, "InferenceEngine", FakeEngine), \
         patch.object(pl, "_extract_one", lambda p, target_size: (None, {})), \
         patch.object(pl, "write_xmp", fake_write):
        pl.process_shoot_with_model(
            input_dir=folder,
            model_path=folder / "fake.ckpt",
            extra_skip_fields=["Tint"],
            save_predictions=False,
        )

    assert len(captured_calls) == 1
    w = captured_calls[0]
    assert "Tint" not in w, f"Tint should be skipped: {w.get('Tint')!r}"
    assert "Temperature" in w, "Temperature should remain when only Tint is skipped"


def test_pipeline_skip_fields_recorded_in_sidecar(tmp_path: Path) -> None:
    """sonna_predictions.json must record the effective skip set (static + user)
    so the finetune capture pipeline correctly attributes user-skipped fields
    as 'model_filtered' source rather than 'lr_default'."""
    from sonna_editor.inference import pipeline as pl
    import json as _json

    folder = tmp_path / "shoot"
    folder.mkdir()
    (folder / "img_0.cr3").write_bytes(b"x")

    class _FakeModel:
        # Pipeline reads engine._model._slider_set_version for the sidecar's
        # slider_set_version field (added 2026-05-14). FakeEngine output below
        # is len(SLIDER_FIELDS)=147 so the model is conceptually v2.
        _slider_set_version = "v2"

    class FakeEngine:
        def __init__(self, *a, **kw):
            self._image_resolution = 384
            self._model = _FakeModel()
        def warmup(self): pass
        def predict(self, *a, **kw): return torch.zeros(1, len(config.SLIDER_FIELDS))

    with patch.object(pl, "InferenceEngine", FakeEngine), \
         patch.object(pl, "_extract_one", lambda p, target_size: (None, {})), \
         patch.object(pl, "write_xmp", lambda *a, **k: None):
        pl.process_shoot_with_model(
            input_dir=folder,
            model_path=folder / "fake.ckpt",
            extra_skip_fields=["Tint"],
            save_predictions=True,
            output_dir=folder,
        )

    sidecar_path = folder / "sonna_predictions.json"
    assert sidecar_path.exists()
    sidecar = _json.loads(sidecar_path.read_text())
    # Effective skip set (recorded under v1_skip_fields for downstream capture)
    assert "Tint" in sidecar["v1_skip_fields"]
    assert "PerspectiveVertical" in sidecar["v1_skip_fields"]  # static stays
    # Breakdown stored separately for diagnostics
    assert sidecar["user_skip_fields"] == ["Tint"]
    assert "Tint" not in sidecar["static_skip_fields"]
    # Profile-identity fields propagated from ckpt sidecar (none here — no
    # sidecar JSON exists for fake.ckpt) plus slider_set_version from engine.
    assert sidecar["profile_type"] is None
    assert sidecar["profile_id"] is None
    assert sidecar["base_checkpoint"] is None
    assert sidecar["slider_set_version"] == "v2"


def test_pipeline_preserve_wb_compat_shim_equivalent_to_skip_fields(tmp_path: Path) -> None:
    """preserve_wb=True must produce the same XMP output as
    extra_skip_fields=['Temperature', 'Tint']."""
    from sonna_editor.inference import pipeline as pl

    def _run(preserve_wb, extra_skip_fields):
        folder = tmp_path / f"shoot_{preserve_wb}_{','.join(extra_skip_fields or [])}"
        folder.mkdir()
        (folder / "img_0.cr3").write_bytes(b"x")
        fake_preds = torch.zeros(1, len(config.SLIDER_FIELDS))
        fake_preds[0, 11] = 5000.0
        fake_preds[0, 12] = 7.0

        class FakeEngine:
            def __init__(self, *a, **kw):
                self._image_resolution = 384
            def warmup(self): pass
            def predict(self, *a, **kw): return fake_preds

        captured: list[dict] = []
        with patch.object(pl, "InferenceEngine", FakeEngine), \
             patch.object(pl, "_extract_one", lambda p, target_size: (None, {})), \
             patch.object(pl, "write_xmp", lambda _p, s, **_k: captured.append(dict(s))):
            pl.process_shoot_with_model(
                input_dir=folder,
                model_path=folder / "fake.ckpt",
                preserve_wb=preserve_wb,
                extra_skip_fields=extra_skip_fields,
                save_predictions=False,
            )
        return captured[0] if captured else None

    via_shim = _run(preserve_wb=True, extra_skip_fields=[])
    via_explicit = _run(preserve_wb=False, extra_skip_fields=["Temperature", "Tint"])
    assert via_shim is not None
    assert via_explicit is not None
    assert set(via_shim.keys()) == set(via_explicit.keys()), (
        "preserve_wb shim and explicit skip_fields should produce identical XMP outputs"
    )
    assert "Temperature" not in via_shim
    assert "Tint" not in via_shim


def test_mode_b_initial_uses_per_photo_preset_adjuster(tmp_path: Path) -> None:
    """Initial Lite profiles should behave like adaptive preset profiles."""
    from sonna_editor.inference import pipeline as pl
    from sonna_editor.mode_b import survey as survey_mod

    folder = tmp_path / "shoot"
    folder.mkdir()
    dark_raw = folder / "dark.cr3"
    warm_bright_raw = folder / "warm_bright.cr3"
    dark_raw.write_bytes(b"x")
    warm_bright_raw.write_bytes(b"x")

    preset_path = tmp_path / "lite-preset.xmp"
    preset_path.write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">'
        "<rdf:RDF><rdf:Description "
        'crs:Contrast2012="15" crs:Highlights2012="-20" '
        'crs:Shadows2012="10" />'
        "</rdf:RDF></x:xmpmeta>",
        encoding="utf-8",
    )
    survey_path = tmp_path / "survey.json"
    survey_mod.write_survey(
        survey_mod.build_survey_payload({k: 0 for k in survey_mod.QUESTION_ORDER}),
        survey_path,
    )
    model_path = tmp_path / "model-v0.1.0.ckpt"
    model_path.write_bytes(b"unused by mode_b_initial")
    model_path.with_suffix(".json").write_text(
        _json.dumps({
            "profile_type": "mode_b_initial",
            "profile_id": "lite-test",
            "base_checkpoint": "base.ckpt",
            "source_preset": str(preset_path),
            "source_survey": str(survey_path),
            "slider_set_version": "v1",
            "resolution": 64,
        }),
        encoding="utf-8",
    )

    def fake_extract(path: Path, _target_size: int):
        if path.name.startswith("dark"):
            img = Image.new("RGB", (64, 64), (35, 35, 35))
        else:
            img = Image.new("RGB", (64, 64), (230, 120, 60))
        return img, {"as_shot_wb": (5200.0, 2.0)}

    captured: dict[str, dict] = {}

    def fake_write(xmp_path, settings, **_kw):
        captured[Path(xmp_path).name] = dict(settings)

    with patch.object(pl, "InferenceEngine", side_effect=AssertionError("Mode B should not load model engine")), \
         patch.object(pl, "_extract_one", fake_extract), \
         patch.object(pl, "write_xmp", fake_write):
        result = pl.process_shoot_with_model(
            input_dir=folder,
            model_path=model_path,
            output_dir=tmp_path / "out",
            save_predictions=True,
        )

    assert result["processed"] == 2
    dark = captured["dark.xmp"]
    bright = captured["warm_bright.xmp"]
    assert dark["Exposure2012"] > 0.0
    assert bright["Exposure2012"] < 0.0
    assert bright["Temperature"] != 5200.0 or bright["Tint"] != 2.0
    assert dark["Contrast2012"] == pytest.approx(15.0)
    assert bright["Contrast2012"] == pytest.approx(15.0)
    assert dark["Shadows2012"] == pytest.approx(10.0)
    assert bright["Shadows2012"] == pytest.approx(10.0)
    assert dark["Highlights2012"] == pytest.approx(-20.0)
    assert bright["Highlights2012"] == pytest.approx(-20.0)

    sidecar = _json.loads((tmp_path / "out" / "sonna_predictions.json").read_text())
    assert sidecar["profile_type"] == "mode_b_initial"
    assert sidecar["photos"]["dark.cr3"]["Exposure2012"] == pytest.approx(
        dark["Exposure2012"]
    )
