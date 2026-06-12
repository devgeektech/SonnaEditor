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
from sonna_editor.mode_b import survey as survey_mod
from sonna_editor.mode_b.checkpoint_builder import (
    HEAD_SLICES,
    HEAD_SLICES_BY_VERSION,
    INHERITED_SKIP_FIELDS,
    PROFILE_TYPE,
    V1_OUTPUT_COUNT,
    V2_OUTPUT_COUNT,
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


def _make_v2_base_ckpt(tmp_path: Path) -> Path:
    """Build a minimal v2 SonnaEditor and save it as a base ckpt for tests."""
    reg = EmbeddingRegistry()
    reg.camera_makes = {"unknown": 0}
    reg.camera_models = {"unknown": 0}
    reg.lenses = {"unknown": 0}
    reg.camera_profiles = {"unknown": 0}
    reg.wb_presets = {"unknown": 0}
    model = SonnaEditor(
        registry=reg,
        _embedding_sizes={
            "num_makes": 4,
            "num_models": 4,
            "num_lenses": 4,
            "num_profiles": 4,
            "num_wb_presets": 4,
        },
        _pretrained_backbone=False,
        arch_version=2,
        slider_set_version="v2",
    )
    path = tmp_path / "base-v2.ckpt"
    model.save_checkpoint(path)
    return path


def _balanced_survey(**overrides: int) -> dict:
    """Build a survey JSON payload with all-zero answers, optionally overridden."""
    answers = {key: 0 for key in survey_mod.QUESTION_ORDER}
    answers.update(overrides)
    return survey_mod.build_survey_payload(answers)


def _empty_preset() -> dict[str, object]:
    return {f: None for f in config.SLIDER_FIELDS}


def _write_survey(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "survey.json"
    survey_mod.write_survey(payload, path)
    return path


# ---------------------------------------------------------------------------
# compute_bias_vector
# ---------------------------------------------------------------------------

def test_compute_bias_vector_returns_absolute_targets() -> None:
    """Spot-check the target semantic: bias[field] = preset + survey target."""
    preset = _empty_preset()
    preset["Exposure2012"] = 0.35
    targets = compute_bias_vector(preset, _balanced_survey())
    assert targets["Exposure2012"] == pytest.approx(0.35)


def test_compute_bias_vector_uses_lr_defaults_when_preset_absent() -> None:
    """Preset absent (falls back to LR_DEFAULTS) + neutral survey → all
    targets land at Lightroom defaults in prediction space."""
    preset = _empty_preset()
    targets = compute_bias_vector(preset, _balanced_survey())
    assert targets["Exposure2012"] == pytest.approx(0.0)
    assert targets["Sharpness"] == pytest.approx(25.0)
    assert targets["ColorGradeBlending"] == pytest.approx(50.0)
    assert targets["ParametricMidtoneSplit"] == pytest.approx(50.0)
    assert targets["Temperature"] == pytest.approx(math.log(5500.0))


def test_compute_bias_vector_applies_survey_offset() -> None:
    """Survey offsets contribute to the target in their native units."""
    preset = _empty_preset()
    preset["Exposure2012"] = 0.3
    # survey exposure answer = 2 -> offset = +1.0 stops -> target +1.3
    survey = _balanced_survey(exposure=2)
    targets = compute_bias_vector(preset, survey)
    assert targets["Exposure2012"] == pytest.approx(1.3)


def test_compute_bias_vector_applies_all_six_survey_offsets() -> None:
    preset = _empty_preset()
    preset["Contrast2012"] = 12.0
    survey = _balanced_survey(contrast=2, saturation=-1, shadows=1)

    targets = compute_bias_vector(preset, survey)

    assert targets["Contrast2012"] == pytest.approx(42.0)
    assert targets["Saturation"] == pytest.approx(-10.0)
    assert targets["Shadows2012"] == pytest.approx(20.0)


def test_compute_bias_vector_clamps_to_range() -> None:
    """Clamp the (preset + survey) target to slider range."""
    preset = _empty_preset()
    preset["Exposure2012"] = 10.0  # outside [-5, 5]; clamps to 5
    survey = _balanced_survey(exposure=2)  # would push to 11; still clamps to 5
    targets = compute_bias_vector(preset, survey)
    assert targets["Exposure2012"] == pytest.approx(5.0)


def test_compute_bias_vector_temperature_log_space_target() -> None:
    """Temperature target is in log-Kelvin."""
    preset = _empty_preset()
    preset["Temperature"] = 4500.0
    # survey temperature answer = -1 → offset = -500 K → target 4000 K
    survey = _balanced_survey(temperature=-1)
    targets = compute_bias_vector(preset, survey)
    assert targets["Temperature"] == pytest.approx(math.log(4000.0))


def test_compute_bias_vector_temperature_clamped_before_log() -> None:
    """Raw Kelvin clamping must happen before log() so log can't NaN."""
    preset = _empty_preset()
    preset["Temperature"] = 500.0  # below range floor 2000 → clamps to 2000
    targets = compute_bias_vector(preset, _balanced_survey())
    assert targets["Temperature"] == pytest.approx(math.log(2000.0))


def test_compute_bias_vector_tone_curve_identity_when_absent() -> None:
    """Tone-curve LR defaults are the identity points (51/102/...); a preset
    that omits them falls back to those."""
    preset = _empty_preset()
    targets = compute_bias_vector(preset, _balanced_survey())
    expected = [0, 51, 102, 153, 204, 255]
    for n in range(1, 7):
        assert targets[f"ToneCurve_Pt{n}_X"] == pytest.approx(expected[n - 1])
        assert targets[f"ToneCurve_Pt{n}_Y"] == pytest.approx(expected[n - 1])


def test_compute_bias_vector_returns_exactly_135_fields() -> None:
    preset = _empty_preset()
    deltas = compute_bias_vector(preset, _balanced_survey())
    assert len(deltas) == V1_OUTPUT_COUNT
    assert set(deltas.keys()) == set(config.SLIDER_FIELDS[:V1_OUTPUT_COUNT])


def test_compute_bias_vector_returns_v2_fields_when_requested() -> None:
    preset = _empty_preset()
    deltas = compute_bias_vector(
        preset,
        _balanced_survey(),
        slider_set_version="v2",
    )
    assert len(deltas) == V2_OUTPUT_COUNT
    assert set(deltas.keys()) == set(config.SLIDER_FIELDS[:V2_OUTPUT_COUNT])


def test_compute_bias_vector_string_preset_value_falls_back_to_default() -> None:
    """Non-numeric preset values (e.g. WhiteBalance='Custom') must not crash.
    Falling back to LR_DEFAULTS means the target lands at the default."""
    preset = _empty_preset()
    preset["Exposure2012"] = "not a number"  # type: ignore[assignment]
    targets = compute_bias_vector(preset, _balanced_survey())
    assert targets["Exposure2012"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# apply_biases_to_model
# ---------------------------------------------------------------------------

def test_apply_biases_zeroes_final_head_weights(tmp_path: Path) -> None:
    """Initial Mode B heads are bias-only so preset output is not stacked."""
    ckpt = _make_v1_base_ckpt(tmp_path)
    model = SonnaEditor.from_checkpoint(ckpt, target_slider_set_version="v1")
    biases = compute_bias_vector(
        _empty_preset(),
        _balanced_survey(),
    )
    apply_biases_to_model(model, biases)
    for head_name, _, _ in HEAD_SLICES:
        post = getattr(model, head_name)[-1].weight
        assert torch.equal(post, torch.zeros_like(post)), (
            f"{head_name} final-linear weight should be zero for initial Mode B"
        )


def test_apply_biases_initial_output_is_preset_faithful(tmp_path: Path) -> None:
    """Forwarding distinct inputs produces the same calibrated initial output."""
    from sonna_editor.mode_b.checkpoint_builder import _neutral_metadata_for

    ckpt = _make_v1_base_ckpt(tmp_path)
    model = SonnaEditor.from_checkpoint(ckpt, target_slider_set_version="v1")
    preset = _empty_preset()
    preset["Exposure2012"] = 0.7
    biases = compute_bias_vector(
        preset,
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

    assert torch.equal(y_a, y_b)
    assert float(y_a[0, 0].item()) == pytest.approx(0.7, abs=1e-6)


def test_apply_biases_sets_absolute_target_bias(tmp_path: Path) -> None:
    """Each head's final-linear bias becomes the preset/survey target."""
    ckpt = _make_v1_base_ckpt(tmp_path)
    model = SonnaEditor.from_checkpoint(ckpt, target_slider_set_version="v1")
    preset = _empty_preset()
    preset["Exposure2012"] = 0.7
    preset["Sharpness"] = 40.0
    targets = compute_bias_vector(preset, _balanced_survey(exposure=1))
    apply_biases_to_model(model, targets)
    v1_fields = config.SLIDER_FIELDS[:V1_OUTPUT_COUNT]
    for head_name, start, end in HEAD_SLICES:
        head = getattr(model, head_name)
        for i in range(start, end):
            expected = targets[v1_fields[i]]
            got = float(head[-1].bias[i - start].item())
            assert got == pytest.approx(expected, abs=1e-6), (
                f"{head_name}[{i - start}] field={v1_fields[i]} "
                f"got={got} expected={expected}"
            )


def test_apply_biases_supports_v2_model_and_zeroes_wb_skip() -> None:
    reg = EmbeddingRegistry()
    model = SonnaEditor(
        registry=reg,
        _embedding_sizes={"num_makes": 4, "num_models": 4, "num_lenses": 4,
                          "num_profiles": 4, "num_wb_presets": 4},
        _pretrained_backbone=False,
        slider_set_version="v2",
    )
    biases = compute_bias_vector(
        _empty_preset(),
        _balanced_survey(),
        slider_set_version="v2",
    )
    apply_biases_to_model(model, biases)
    for head_name, _, _ in HEAD_SLICES_BY_VERSION["v2"]:
        after = getattr(model, head_name)[-1].weight
        assert torch.equal(after, torch.zeros_like(after))
    assert torch.equal(
        model.wb_metadata_skip.weight,
        torch.zeros_like(model.wb_metadata_skip.weight),
    )
    assert torch.equal(
        model.wb_metadata_skip.bias,
        torch.zeros_like(model.wb_metadata_skip.bias),
    )


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
    """The saved ckpt's final layer is zero-weight, target-bias Mode B."""
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
    targets = compute_bias_vector(read_xmp(FIXTURE_PRESET), survey_mod.load_survey(survey))
    verify_checkpoint(output, targets, base)


def test_build_mode_b_checkpoint_subtracts_survey_from_skip_fields(
    tmp_path: Path,
) -> None:
    """Survey-covered sliders must be removed from the inherited skip list.

    INHERITED_SKIP_FIELDS captures the base ckpt's architecturally-broken
    sliders. Tint is in that list AND is one of the Lite survey questions.
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
    assert side["slider_set_version"] == "v1"
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


def test_build_mode_b_checkpoint_empty_preset_uses_defaults(tmp_path: Path) -> None:
    """A minimal XMP with no scalar sliders falls back to Lightroom defaults."""
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


def test_build_mode_b_checkpoint_preserves_v2_slider_set(tmp_path: Path) -> None:
    """Lite creation from a v2 base must not down-convert the checkpoint."""
    v2_ckpt = _make_v2_base_ckpt(tmp_path)
    survey = _write_survey(tmp_path, _balanced_survey())
    output = tmp_path / "out.ckpt"
    sidecar_path = build_mode_b_checkpoint(
        preset_path=FIXTURE_PRESET,
        survey_path=survey,
        base_ckpt_path=v2_ckpt,
        output_ckpt_path=output,
        profile_name="Mode B Test",
    )

    reloaded = SonnaEditor.from_checkpoint(output)
    assert reloaded._slider_set_version == "v2"
    side = json.loads(sidecar_path.read_text())
    assert side["slider_set_version"] == "v2"

    base_model = SonnaEditor.from_checkpoint(v2_ckpt)
    out_model = SonnaEditor.from_checkpoint(output)
    assert out_model._arch_version == base_model._arch_version
    for head_name, _, _ in HEAD_SLICES_BY_VERSION["v2"]:
        weight = getattr(out_model, head_name)[-1].weight
        assert torch.equal(weight, torch.zeros_like(weight))
    assert torch.equal(
        out_model.wb_metadata_skip.weight,
        torch.zeros_like(out_model.wb_metadata_skip.weight),
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
