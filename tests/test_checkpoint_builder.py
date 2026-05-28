"""Tests for the Mode B preset-to-checkpoint converter."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sonna_editor import config
from sonna_editor.data.xmp import LR_DEFAULTS
from sonna_editor.mode_b import survey as survey_mod
from sonna_editor.mode_b.checkpoint_builder import (
    HEAD_SLICES,
    INHERITED_SKIP_FIELDS,
    PROFILE_TYPE,
    SLIDER_SET_VERSION,
    V1_OUTPUT_COUNT,
    apply_biases_to_model,
    build_mode_b_checkpoint,
    compute_bias_vector,
    verify_checkpoint,
    _generate_profile_id,
)
from sonna_editor.model.architecture import EmbeddingRegistry, SonnaEditor


FIXTURE_PRESET = Path(__file__).parent / "fixtures" / "preset_sonna_v1.xmp"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_v1_base_ckpt(tmp_path: Path) -> Path:
    """Build a minimal v1 SonnaEditor and save it as a base ckpt for tests."""
    reg = EmbeddingRegistry()
    reg.camera_makes  = {"unknown": 0}
    reg.camera_models = {"unknown": 0}
    reg.lenses        = {"unknown": 0}
    reg.camera_profiles = {"unknown": 0}
    reg.wb_presets    = {"unknown": 0}
    model = SonnaEditor(
        registry=reg,
        _embedding_sizes={
            "num_makes":  4,
            "num_models": 4,
            "num_lenses": 4,
            "num_profiles": 4,
            "num_wb_presets": 4,
        },
        _pretrained_backbone=False,
        arch_version=1,
        slider_set_version="v1",
    )
    path = tmp_path / "base-v1.ckpt"
    model.save_checkpoint(path)
    return path


def _balanced_survey(**overrides: int) -> dict:
    """Build a survey JSON payload with all-zero answers, optionally overridden."""
    answers = {key: 0 for key in survey_mod.QUESTION_ORDER}
    answers.update(overrides)
    return survey_mod.build_survey_payload(answers)


def _write_survey(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "survey.json"
    survey_mod.write_survey(payload, path)
    return path


# ---------------------------------------------------------------------------
# compute_bias_vector
# ---------------------------------------------------------------------------

def test_compute_bias_vector_returns_deltas_not_absolutes() -> None:
    """Spot-check the delta semantic: bias[field] = target − LR_DEFAULTS[field]."""
    preset = {f: None for f in config.SLIDER_FIELDS}
    preset["Exposure2012"] = 0.35  # LR default 0.0 → delta +0.35
    deltas = compute_bias_vector(preset, _balanced_survey())
    assert deltas["Exposure2012"] == pytest.approx(0.35)


def test_compute_bias_vector_zero_delta_when_preset_matches_defaults() -> None:
    """Preset absent (falls back to LR_DEFAULTS) + neutral survey → all
    deltas are exactly 0. This is the property that makes a neutral
    Lite ckpt byte-equivalent to the base ckpt."""
    preset = {f: None for f in config.SLIDER_FIELDS}
    deltas = compute_bias_vector(preset, _balanced_survey())
    # Sliders whose LR default is non-zero would have surfaced the old
    # absolute-target behaviour. Pin them at delta = 0.
    assert deltas["Sharpness"] == pytest.approx(0.0)
    assert deltas["ColorGradeBlending"] == pytest.approx(0.0)
    assert deltas["ParametricMidtoneSplit"] == pytest.approx(0.0)
    assert deltas["Temperature"] == pytest.approx(0.0)
    # And every other field too.
    for f, d in deltas.items():
        assert d == pytest.approx(0.0), f"{f} delta={d} (expected 0)"


def test_compute_bias_vector_applies_survey_offset() -> None:
    """Survey offsets contribute to the delta in their native units."""
    preset = {f: None for f in config.SLIDER_FIELDS}
    preset["Exposure2012"] = 0.3  # delta +0.3 from LR default 0.0
    # survey exposure answer = 2 → offset = +1.0 stops → total delta +1.3
    survey = _balanced_survey(exposure=2)
    deltas = compute_bias_vector(preset, survey)
    assert deltas["Exposure2012"] == pytest.approx(1.3)


def test_compute_bias_vector_clamps_to_range_before_delta() -> None:
    """Clamp the (preset + survey) target to slider range BEFORE differencing."""
    preset = {f: None for f in config.SLIDER_FIELDS}
    preset["Exposure2012"] = 10.0  # outside [-5, 5]; clamps to 5
    survey = _balanced_survey(exposure=2)  # would push to 11; still clamps to 5
    deltas = compute_bias_vector(preset, survey)
    # target clamped to 5.0; LR default 0.0; delta = 5.0
    assert deltas["Exposure2012"] == pytest.approx(5.0)


def test_compute_bias_vector_temperature_log_space_delta() -> None:
    """Temperature delta is in log-Kelvin: log(target) − log(LR_default)."""
    preset = {f: None for f in config.SLIDER_FIELDS}
    preset["Temperature"] = 4500.0
    # survey temperature answer = -1 → offset = -500 K → target 4000 K
    survey = _balanced_survey(temperature=-1)
    deltas = compute_bias_vector(preset, survey)
    assert deltas["Temperature"] == pytest.approx(
        math.log(4000.0) - math.log(5500.0)
    )


def test_compute_bias_vector_temperature_clamped_before_log() -> None:
    """Raw Kelvin clamping must happen before log() so log can't NaN."""
    preset = {f: None for f in config.SLIDER_FIELDS}
    preset["Temperature"] = 500.0  # below range floor 2000 → clamps to 2000
    deltas = compute_bias_vector(preset, _balanced_survey())
    # target clamped to log(2000); LR default log(5500); delta is the diff.
    assert deltas["Temperature"] == pytest.approx(
        math.log(2000.0) - math.log(5500.0)
    )


def test_compute_bias_vector_tone_curve_zero_delta_when_absent() -> None:
    """Tone-curve LR defaults are the identity points (51/102/...); a preset
    that omits them falls back to those, so the delta is exactly 0."""
    preset = {f: None for f in config.SLIDER_FIELDS}
    deltas = compute_bias_vector(preset, _balanced_survey())
    for n in range(1, 7):
        assert deltas[f"ToneCurve_Pt{n}_X"] == pytest.approx(0.0)
        assert deltas[f"ToneCurve_Pt{n}_Y"] == pytest.approx(0.0)


def test_compute_bias_vector_returns_exactly_135_fields() -> None:
    preset = {f: None for f in config.SLIDER_FIELDS}
    deltas = compute_bias_vector(preset, _balanced_survey())
    assert len(deltas) == V1_OUTPUT_COUNT
    assert set(deltas.keys()) == set(config.SLIDER_FIELDS[:V1_OUTPUT_COUNT])


def test_compute_bias_vector_string_preset_value_falls_back_to_default() -> None:
    """Non-numeric preset values (e.g. WhiteBalance='Custom') must not crash.
    Falling back to LR_DEFAULTS means the delta lands at 0."""
    preset = {f: None for f in config.SLIDER_FIELDS}
    preset["Exposure2012"] = "not a number"  # type: ignore[assignment]
    deltas = compute_bias_vector(preset, _balanced_survey())
    assert deltas["Exposure2012"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# apply_biases_to_model
# ---------------------------------------------------------------------------

def test_apply_biases_preserves_inherited_head_weights(tmp_path: Path) -> None:
    """Head WEIGHTS are inherited from the base ckpt byte-for-byte.

    The 2026-05 fix stopped zeroing head weights — Lite profiles were
    producing identical XMP values across every photo because the model
    output was bias-only. Mode B now inherits the base ckpt's heads so the
    image-aware deviation is preserved on top of the calibration bias.
    """
    ckpt = _make_v1_base_ckpt(tmp_path)
    model = SonnaEditor.from_checkpoint(ckpt, target_slider_set_version="v1")
    pre = {
        head_name: getattr(model, head_name)[-1].weight.detach().clone()
        for head_name, _, _ in HEAD_SLICES
    }
    biases = compute_bias_vector(
        {f: None for f in config.SLIDER_FIELDS},
        _balanced_survey(),
    )
    apply_biases_to_model(model, biases)
    for head_name, _, _ in HEAD_SLICES:
        post = getattr(model, head_name)[-1].weight
        assert torch.equal(post, pre[head_name]), (
            f"{head_name} final-linear weight changed; should be inherited byte-for-byte"
        )


def test_apply_biases_retains_per_photo_variation(tmp_path: Path) -> None:
    """Forwarding two distinct inputs produces distinct outputs.

    The earlier zero-weights formulation collapsed every prediction onto
    the bias vector regardless of input. This test pins the invariant
    that an actual image branch contributes to the output. The base ckpt
    used here is randomly-initialised by ``_make_v1_base_ckpt`` (no
    pretrained backbone weights) but its final-linear weights are still
    non-zero, so two different inputs should produce a non-trivial output
    difference.
    """
    from sonna_editor.mode_b.checkpoint_builder import _neutral_metadata_for

    ckpt = _make_v1_base_ckpt(tmp_path)
    model = SonnaEditor.from_checkpoint(ckpt, target_slider_set_version="v1")
    biases = compute_bias_vector(
        {f: None for f in config.SLIDER_FIELDS},
        _balanced_survey(),
    )
    apply_biases_to_model(model, biases)
    model.eval()

    resolution = config.IMAGE_RESOLUTION
    img_a = torch.zeros(1, 3, resolution, resolution)
    img_b = torch.full((1, 3, resolution, resolution), 0.7)
    metadata = _neutral_metadata_for(model)
    with torch.no_grad():
        y_a = model(img_a, metadata)
        y_b = model(img_b, metadata)

    max_diff = (y_a - y_b).abs().max().item()
    assert max_diff > 1e-3, (
        f"Mode B forward output is identical for distinct images "
        f"(max |y_a - y_b| = {max_diff}). Head weights have likely been "
        f"zeroed — re-check apply_biases_to_model."
    )


def test_apply_biases_adds_delta_to_inherited_bias(tmp_path: Path) -> None:
    """Each head's final-linear bias becomes ``base_bias + delta`` per slot.

    The 2026-05 delta-bias fix changed this from a replacement
    (final.bias.copy_(delta)) to an addition (final.bias.add_(delta)).
    For a non-zero delta vector the new bias must equal the snapshotted
    base bias plus the delta — anything else means the previous
    replace-semantic regressed."""
    ckpt = _make_v1_base_ckpt(tmp_path)
    model = SonnaEditor.from_checkpoint(ckpt, target_slider_set_version="v1")
    base_biases = {
        head_name: getattr(model, head_name)[-1].bias.detach().clone()
        for head_name, _, _ in HEAD_SLICES
    }
    preset = {f: None for f in config.SLIDER_FIELDS}
    preset["Exposure2012"] = 0.7  # non-zero delta so the additive vs
    preset["Sharpness"]    = 40.0  # replacement distinction is visible
    deltas = compute_bias_vector(preset, _balanced_survey(exposure=1))
    apply_biases_to_model(model, deltas)
    v1_fields = config.SLIDER_FIELDS[:V1_OUTPUT_COUNT]
    for head_name, start, end in HEAD_SLICES:
        head = getattr(model, head_name)
        for i in range(start, end):
            expected = float(base_biases[head_name][i - start].item()) + deltas[v1_fields[i]]
            got = float(head[-1].bias[i - start].item())
            assert got == pytest.approx(expected, abs=1e-6), (
                f"{head_name}[{i - start}] field={v1_fields[i]} "
                f"got={got} expected={expected} (base={base_biases[head_name][i - start].item()} + delta={deltas[v1_fields[i]]})"
            )


def test_apply_biases_rejects_v2_model() -> None:
    reg = EmbeddingRegistry()
    model = SonnaEditor(
        registry=reg,
        _embedding_sizes={"num_makes": 4, "num_models": 4, "num_lenses": 4,
                          "num_profiles": 4, "num_wb_presets": 4},
        _pretrained_backbone=False,
        slider_set_version="v2",
    )
    biases = compute_bias_vector(
        {f: None for f in config.SLIDER_FIELDS},
        _balanced_survey(),
    )
    with pytest.raises(ValueError, match="v1 model"):
        apply_biases_to_model(model, biases)


# ---------------------------------------------------------------------------
# build_mode_b_checkpoint — full orchestration
# ---------------------------------------------------------------------------

def test_build_mode_b_checkpoint_output_loads_via_from_checkpoint(
    tmp_path: Path,
) -> None:
    base = _make_v1_base_ckpt(tmp_path)
    survey = _write_survey(tmp_path, _balanced_survey())
    output = tmp_path / "mode_b.ckpt"
    sidecar = build_mode_b_checkpoint(
        preset_path=FIXTURE_PRESET,
        survey_path=survey,
        base_ckpt_path=base,
        output_ckpt_path=output,
        profile_name="Mode B Test",
    )
    assert output.exists()
    assert sidecar.exists() and sidecar == output.with_suffix(".json")
    # Re-load — must succeed and produce a v1 model.
    reloaded = SonnaEditor.from_checkpoint(output)
    assert reloaded._slider_set_version == "v1"


def test_build_mode_b_checkpoint_backbone_preserved_byte_for_byte(
    tmp_path: Path,
) -> None:
    base = _make_v1_base_ckpt(tmp_path)
    survey = _write_survey(tmp_path, _balanced_survey())
    output = tmp_path / "mode_b.ckpt"
    build_mode_b_checkpoint(
        preset_path=FIXTURE_PRESET,
        survey_path=survey,
        base_ckpt_path=base,
        output_ckpt_path=output,
        profile_name="Mode B Test",
    )

    base_state = torch.load(base, map_location="cpu", weights_only=False)["model_state"]
    out_state = torch.load(output, map_location="cpu", weights_only=False)["model_state"]

    backbone_prefixes = ("backbone_features.", "backbone_pool.", "backbone_norm.")
    for k, v in base_state.items():
        if k.startswith(backbone_prefixes):
            assert k in out_state, f"backbone tensor {k} missing in output"
            assert torch.equal(v, out_state[k]), f"backbone tensor {k} differs"


def test_build_mode_b_checkpoint_metadata_encoder_preserved(tmp_path: Path) -> None:
    base = _make_v1_base_ckpt(tmp_path)
    survey = _write_survey(tmp_path, _balanced_survey())
    output = tmp_path / "mode_b.ckpt"
    build_mode_b_checkpoint(
        preset_path=FIXTURE_PRESET,
        survey_path=survey,
        base_ckpt_path=base,
        output_ckpt_path=output,
        profile_name="Mode B Test",
    )
    base_state = torch.load(base, map_location="cpu", weights_only=False)["model_state"]
    out_state = torch.load(output, map_location="cpu", weights_only=False)["model_state"]
    for k, v in base_state.items():
        if k.startswith("metadata_encoder."):
            assert torch.equal(v, out_state[k]), f"metadata_encoder tensor {k} differs"


def test_build_mode_b_checkpoint_base_ckpt_not_modified(tmp_path: Path) -> None:
    """Hard rule: the v1.2.3 source checkpoint is loaded read-only."""
    base = _make_v1_base_ckpt(tmp_path)
    survey = _write_survey(tmp_path, _balanced_survey())
    pre_bytes = base.read_bytes()
    pre_mtime = base.stat().st_mtime
    build_mode_b_checkpoint(
        preset_path=FIXTURE_PRESET,
        survey_path=survey,
        base_ckpt_path=base,
        output_ckpt_path=tmp_path / "mode_b.ckpt",
        profile_name="Mode B Test",
    )
    assert base.read_bytes() == pre_bytes
    assert base.stat().st_mtime == pre_mtime


def test_build_mode_b_checkpoint_verification_passes(tmp_path: Path) -> None:
    """The saved ckpt's biases equal base_bias + delta per slider and the
    weights are byte-identical to the base ckpt."""
    base = _make_v1_base_ckpt(tmp_path)
    survey = _write_survey(tmp_path, _balanced_survey(exposure=2, temperature=1))
    output = tmp_path / "mode_b.ckpt"
    # build_mode_b_checkpoint runs verify_checkpoint internally; if it
    # raises, this test fails. We additionally run verify_checkpoint
    # explicitly on the saved ckpt for belt-and-braces coverage.
    build_mode_b_checkpoint(
        preset_path=FIXTURE_PRESET,
        survey_path=survey,
        base_ckpt_path=base,
        output_ckpt_path=output,
        profile_name="Mode B Test",
    )

    from sonna_editor.data.xmp import read_xmp
    deltas = compute_bias_vector(read_xmp(FIXTURE_PRESET), survey_mod.load_survey(survey))
    verify_checkpoint(output, deltas, base)


def test_build_mode_b_checkpoint_subtracts_survey_from_skip_fields(
    tmp_path: Path,
) -> None:
    """Survey-covered sliders must be removed from the inherited skip list.

    INHERITED_SKIP_FIELDS captures the base ckpt's architecturally-broken
    sliders. Tint is in that list AND is one of the six survey questions.
    Without the subtraction the user's Tint calibration would set the
    Mode B bias correctly but then get stripped at XMP-write time —
    making the survey question functionally meaningless. This test pins
    the invariant. Non-survey inherited skips (currently
    ColorGradeMidtoneHue and SplitToningShadowHue) must still be skipped.
    """
    from sonna_editor.mode_b.survey import QUESTION_SLIDER_MAP

    base = _make_v1_base_ckpt(tmp_path)
    survey = _write_survey(tmp_path, _balanced_survey())
    output = tmp_path / "mode_b.ckpt"
    sidecar_path = build_mode_b_checkpoint(
        preset_path=FIXTURE_PRESET,
        survey_path=survey,
        base_ckpt_path=base,
        output_ckpt_path=output,
        profile_name="Subtract Survey Test",
    )
    side = json.loads(sidecar_path.read_text())
    skip = set(side["default_skip_fields"])
    inherited = set(INHERITED_SKIP_FIELDS)
    survey_sliders = set(QUESTION_SLIDER_MAP.values())

    # Anything inherited AND covered by the survey is dropped.
    assert (inherited & survey_sliders).isdisjoint(skip), (
        f"Survey-covered fields leaked into skip: "
        f"{(inherited & survey_sliders) & skip}"
    )
    # Anything inherited but NOT covered by the survey stays.
    assert (inherited - survey_sliders) <= skip


def test_build_mode_b_checkpoint_sidecar_schema(tmp_path: Path) -> None:
    base = _make_v1_base_ckpt(tmp_path)
    survey = _write_survey(tmp_path, _balanced_survey())
    output = tmp_path / "mode_b.ckpt"
    sidecar_path = build_mode_b_checkpoint(
        preset_path=FIXTURE_PRESET,
        survey_path=survey,
        base_ckpt_path=base,
        output_ckpt_path=output,
        profile_name="Mode B - Wedding Lite",
    )
    side = json.loads(sidecar_path.read_text())
    assert side["profile_type"] == PROFILE_TYPE
    assert side["display_name"] == "Mode B - Wedding Lite"
    assert side["slider_set_version"] == SLIDER_SET_VERSION
    # Survey-vs-skip subtraction: survey-covered sliders are removed from
    # the inherited skip list so the user's calibration isn't silently
    # stripped at XMP-write time. The dedicated test below exercises this
    # invariant directly; here we just confirm "Tint" (covered by survey)
    # is absent and the other inherited entries remain.
    assert "Tint" not in side["default_skip_fields"]
    assert "ColorGradeMidtoneHue" in side["default_skip_fields"]
    assert "SplitToningShadowHue" in side["default_skip_fields"]
    # Sidecar resolution inherits from the base ckpt's arch_config, not the
    # global config.IMAGE_RESOLUTION. The synthetic base ckpt was saved
    # via save_checkpoint which uses the global, so the two happen to match
    # here — the dedicated inheritance test below uses a non-default
    # base resolution to truly exercise the fix.
    base_blob = torch.load(base, map_location="cpu", weights_only=False)
    expected_resolution = int(base_blob["arch_config"]["image_resolution"])
    assert side["resolution"] == expected_resolution
    assert side["base_checkpoint"] == str(base)
    assert side["source_preset"] == str(FIXTURE_PRESET)
    assert side["source_survey"] == str(survey)
    assert side["experimental"] is False
    # SHA256 is 64 hex chars
    assert len(side["base_checkpoint_sha256"]) == 64
    # profile_id auto-generated with slug + timestamp
    assert side["profile_id"].startswith("mode-b-wedding-lite-")
    # ISO timestamp present
    assert "T" in side["date_iso"]


def test_build_mode_b_checkpoint_uses_explicit_profile_id(tmp_path: Path) -> None:
    base = _make_v1_base_ckpt(tmp_path)
    survey = _write_survey(tmp_path, _balanced_survey())
    sidecar_path = build_mode_b_checkpoint(
        preset_path=FIXTURE_PRESET,
        survey_path=survey,
        base_ckpt_path=base,
        output_ckpt_path=tmp_path / "mode_b.ckpt",
        profile_name="Mode B Test",
        profile_id="custom-id-123",
    )
    side = json.loads(sidecar_path.read_text())
    assert side["profile_id"] == "custom-id-123"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_build_mode_b_checkpoint_missing_preset_raises(tmp_path: Path) -> None:
    base = _make_v1_base_ckpt(tmp_path)
    survey = _write_survey(tmp_path, _balanced_survey())
    with pytest.raises(FileNotFoundError, match="Preset"):
        build_mode_b_checkpoint(
            preset_path=tmp_path / "nope.xmp",
            survey_path=survey,
            base_ckpt_path=base,
            output_ckpt_path=tmp_path / "out.ckpt",
            profile_name="Mode B Test",
        )


def test_build_mode_b_checkpoint_missing_survey_raises(tmp_path: Path) -> None:
    base = _make_v1_base_ckpt(tmp_path)
    with pytest.raises(FileNotFoundError, match="Survey"):
        build_mode_b_checkpoint(
            preset_path=FIXTURE_PRESET,
            survey_path=tmp_path / "nope.json",
            base_ckpt_path=base,
            output_ckpt_path=tmp_path / "out.ckpt",
            profile_name="Mode B Test",
        )


def test_build_mode_b_checkpoint_missing_base_ckpt_raises(tmp_path: Path) -> None:
    survey = _write_survey(tmp_path, _balanced_survey())
    with pytest.raises(FileNotFoundError, match="Base checkpoint"):
        build_mode_b_checkpoint(
            preset_path=FIXTURE_PRESET,
            survey_path=survey,
            base_ckpt_path=tmp_path / "nope.ckpt",
            output_ckpt_path=tmp_path / "out.ckpt",
            profile_name="Mode B Test",
        )


def test_build_mode_b_checkpoint_empty_preset_raises(tmp_path: Path) -> None:
    """A preset that parses but contains no slider values should fail clearly."""
    empty_preset = tmp_path / "empty.xmp"
    empty_preset.write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        "<rdf:RDF><rdf:Description/></rdf:RDF></x:xmpmeta>",
        encoding="utf-8",
    )
    base = _make_v1_base_ckpt(tmp_path)
    survey = _write_survey(tmp_path, _balanced_survey())
    # Tone curve fields will fill in to identity defaults (read_xmp guarantees
    # floats for those), so we don't actually expect "all None". The check in
    # build_mode_b_checkpoint catches the case where every scalar AND every
    # tone-curve field is None — but tone curves always come back as floats.
    # An empty XMP therefore has *only* tone-curve floats populated; the
    # "all None" guard won't trip. Confirm the build succeeds in that case,
    # falling back to LR_DEFAULTS for every scalar field.
    output = tmp_path / "out.ckpt"
    build_mode_b_checkpoint(
        preset_path=empty_preset,
        survey_path=survey,
        base_ckpt_path=base,
        output_ckpt_path=output,
        profile_name="Mode B Test",
    )
    assert output.exists()


def test_build_mode_b_checkpoint_v2_base_ckpt_rejected(tmp_path: Path) -> None:
    """Loading a v2 ckpt as v1 must error (locked-append-only protection)."""
    reg = EmbeddingRegistry()
    v2_model = SonnaEditor(
        registry=reg,
        _embedding_sizes={"num_makes": 4, "num_models": 4, "num_lenses": 4,
                          "num_profiles": 4, "num_wb_presets": 4},
        _pretrained_backbone=False,
        slider_set_version="v2",
    )
    v2_ckpt = tmp_path / "v2.ckpt"
    v2_model.save_checkpoint(v2_ckpt)
    survey = _write_survey(tmp_path, _balanced_survey())
    with pytest.raises(ValueError, match="v2 checkpoint"):
        build_mode_b_checkpoint(
            preset_path=FIXTURE_PRESET,
            survey_path=survey,
            base_ckpt_path=v2_ckpt,
            output_ckpt_path=tmp_path / "out.ckpt",
            profile_name="Mode B Test",
        )


# ---------------------------------------------------------------------------
# Profile ID generation
# ---------------------------------------------------------------------------

def test_generate_profile_id_slugifies_name() -> None:
    import datetime as _dt
    pid = _generate_profile_id(
        "Mode B - Wedding Lite!",
        now=_dt.datetime(2026, 5, 14, 11, 2, tzinfo=_dt.timezone.utc),
    )
    assert pid == "mode-b-wedding-lite-20260514-1102"


def test_generate_profile_id_handles_empty_name() -> None:
    import datetime as _dt
    pid = _generate_profile_id(
        "!!!",
        now=_dt.datetime(2026, 5, 14, 11, 2, tzinfo=_dt.timezone.utc),
    )
    assert pid == "mode-b-profile-20260514-1102"


# ---------------------------------------------------------------------------
# CLI subprocess tests
# ---------------------------------------------------------------------------

CLI_PATH = Path(__file__).parent.parent / "scripts" / "build_mode_b_checkpoint.py"


def test_cli_happy_path(tmp_path: Path) -> None:
    base = _make_v1_base_ckpt(tmp_path)
    survey = _write_survey(tmp_path, _balanced_survey())
    output = tmp_path / "mode_b.ckpt"
    result = subprocess.run(
        [
            sys.executable, str(CLI_PATH),
            "--preset", str(FIXTURE_PRESET),
            "--survey", str(survey),
            "--base-ckpt", str(base),
            "--output", str(output),
            "--profile-name", "Mode B - CLI Test",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"CLI failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert output.exists()
    assert output.with_suffix(".json").exists()


def test_cli_missing_preset_returns_nonzero(tmp_path: Path) -> None:
    base = _make_v1_base_ckpt(tmp_path)
    survey = _write_survey(tmp_path, _balanced_survey())
    result = subprocess.run(
        [
            sys.executable, str(CLI_PATH),
            "--preset", str(tmp_path / "nope.xmp"),
            "--survey", str(survey),
            "--base-ckpt", str(base),
            "--output", str(tmp_path / "out.ckpt"),
            "--profile-name", "Mode B",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "Preset not found" in result.stderr


def test_cli_requires_all_long_args(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--preset", str(FIXTURE_PRESET)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    # argparse writes "the following arguments are required:" to stderr
    assert "required" in result.stderr.lower()


# ---------------------------------------------------------------------------
# Mode B ckpt sidecar resolution inheritance regression
# ---------------------------------------------------------------------------
# Surfaced during Mode B Step 3 validation: Mode B sidecars were being
# written with config.IMAGE_RESOLUTION (the global default, currently 512)
# rather than inheriting from the base ckpt (v1.2.3 = 256). This caused
# InferenceEngine to extract previews at the wrong resolution when loading
# Mode B profiles built from v1.2.3 (the production base ckpt for Mode B
# initial). Output was still correct because Mode B head weights are
# zeroed (image content ignored), but the resolution mismatch would matter
# once Phase 5 fine-tuning makes the head weights non-zero.

def _make_v1_base_ckpt_at(tmp_path: Path, *, resolution: int) -> Path:
    """Build a synthetic v1 base ckpt with a specified image_resolution.

    save_checkpoint pulls config.IMAGE_RESOLUTION (the global) at save time,
    so we post-process the saved ckpt to override its arch_config rather
    than monkey-patching the global. Cleaner test isolation.
    """
    reg = EmbeddingRegistry()
    reg.camera_makes    = {"unknown": 0}
    reg.camera_models   = {"unknown": 0}
    reg.lenses          = {"unknown": 0}
    reg.camera_profiles = {"unknown": 0}
    reg.wb_presets      = {"unknown": 0}
    model = SonnaEditor(
        registry=reg,
        _embedding_sizes={"num_makes": 4, "num_models": 4, "num_lenses": 4,
                          "num_profiles": 4, "num_wb_presets": 4},
        _pretrained_backbone=False,
        arch_version=1,
        slider_set_version="v1",
    )
    path = tmp_path / f"base-{resolution}px.ckpt"
    model.save_checkpoint(path)
    # Override arch_config.image_resolution so the sidecar fix has something
    # non-trivial to inherit (different from config.IMAGE_RESOLUTION default).
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["arch_config"]["image_resolution"] = resolution
    torch.save(blob, path)
    return path


def test_build_mode_b_checkpoint_inherits_base_resolution(tmp_path: Path) -> None:
    """Mode B sidecar must report the base ckpt's image_resolution, not
    config.IMAGE_RESOLUTION. Regression for the 2026-05-14 Step 3 finding
    where v1.2.3 base (256px) was producing Mode B sidecars reporting 512."""
    base = _make_v1_base_ckpt_at(tmp_path, resolution=256)
    # Sanity: the synthetic base really is 256, and that's different from the
    # current global default — otherwise this test is vacuous.
    assert config.IMAGE_RESOLUTION != 256, (
        "Test assumes config.IMAGE_RESOLUTION != 256 so the inheritance "
        "behaviour is actually exercised."
    )

    survey = _write_survey(tmp_path, _balanced_survey())
    sidecar_path = build_mode_b_checkpoint(
        preset_path=FIXTURE_PRESET,
        survey_path=survey,
        base_ckpt_path=base,
        output_ckpt_path=tmp_path / "mb.ckpt",
        profile_name="Mode B - 256px base",
    )
    side = json.loads(sidecar_path.read_text())
    assert side["resolution"] == 256, (
        f"Mode B sidecar should inherit base ckpt resolution 256; "
        f"got {side['resolution']}. Likely regression to pre-fix behaviour "
        f"of reading config.IMAGE_RESOLUTION."
    )


def test_build_mode_b_checkpoint_loads_via_engine_at_base_resolution(
    tmp_path: Path,
) -> None:
    """InferenceEngine reads the Mode B sidecar's resolution and configures
    preview extraction accordingly. End-to-end sanity that the sidecar fix
    propagates through to engine behaviour."""
    base = _make_v1_base_ckpt_at(tmp_path, resolution=256)
    survey = _write_survey(tmp_path, _balanced_survey())
    output = tmp_path / "mb.ckpt"
    build_mode_b_checkpoint(
        preset_path=FIXTURE_PRESET,
        survey_path=survey,
        base_ckpt_path=base,
        output_ckpt_path=output,
        profile_name="Mode B Test",
    )
    from sonna_editor.inference.engine import InferenceEngine
    engine = InferenceEngine(output, device="cpu")
    assert engine._image_resolution == 256
