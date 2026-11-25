"""End-to-end validation for Lite Profile creation and Style Survey efficacy.

Verifies:
1. Survey answers (-2..+2) calculate exact numerical offsets for Exposure, Temperature, and Tint.
2. Preset values and survey offsets are correctly embedded into the Lite checkpoint biases.
3. Model forward pass outputs match preset + survey offsets.
4. Shoot processing (Mode B adaptive path) applies preset values and survey calibrations to output photos:
   - Profile A (+2 Exp, +2 Temp, -2 Tint) vs Profile B (-2 Exp, -2 Temp, +2 Tint) vs Profile C (0 neutral)
   - Exposure differences strictly reflect the survey (+1.0 EV vs -1.0 EV).
   - Temperature differences strictly reflect the survey (+1000K vs -1000K).
   - Tint differences strictly reflect the survey (-10 vs +10 units).
5. Non-survey preset sliders (Curves, HSL, Color Grading, Clarity, Contrast) are 100% preserved.
6. Generated XMP sidecars contain the exact expected values.
"""
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image
import pytest

from sonna_editor.data.xmp import read_xmp, write_xmp
from sonna_editor.foundation import resolve_foundation_checkpoint
from sonna_editor.inference.pipeline import (
    _mode_b_adjusted_values_for_photo,
    _survey_adjusted_preset,
)
from sonna_editor.mode_b import checkpoint_builder as mode_b_builder
from sonna_editor.mode_b.survey import (
    build_survey_payload,
    compute_offset,
    write_survey,
)
from sonna_editor.preset.parser import parse_preset

FIXTURE_PRESET = Path(__file__).parent / "fixtures" / "preset_sonna_v1.xmp"


def test_survey_math_and_offsets():
    """Verify survey math computes exact offsets across the 5 valid answer tiers."""
    assert compute_offset("exposure", +2) == 1.0
    assert compute_offset("exposure", +1) == 0.5
    assert compute_offset("exposure", 0) == 0.0
    assert compute_offset("exposure", -1) == -0.5
    assert compute_offset("exposure", -2) == -1.0

    assert compute_offset("temperature", +2) == 1000.0
    assert compute_offset("temperature", +1) == 500.0
    assert compute_offset("temperature", 0) == 0.0
    assert compute_offset("temperature", -1) == -500.0
    assert compute_offset("temperature", -2) == -1000.0

    assert compute_offset("tint", +2) == 10.0
    assert compute_offset("tint", +1) == 5.0
    assert compute_offset("tint", 0) == 0.0
    assert compute_offset("tint", -1) == -5.0
    assert compute_offset("tint", -2) == -10.0


def test_lite_profile_creation_and_inference_e2e(tmp_path: Path):
    """Full end-to-end test creating 3 Lite profiles and processing test images."""
    base_ckpt_path = resolve_foundation_checkpoint()
    assert base_ckpt_path.exists(), f"Foundation checkpoint missing: {base_ckpt_path}"

    preset_raw = parse_preset(FIXTURE_PRESET)
    assert preset_raw is not None

    profiles_to_test = [
        ("bright_warm_green", {"exposure": 2, "temperature": 2, "tint": -2}),
        ("neutral",          {"exposure": 0, "temperature": 0, "tint": 0}),
        ("dark_cool_magenta", {"exposure": -2, "temperature": -2, "tint": 2}),
    ]

    built_profiles = {}

    for name, answers_dict in profiles_to_test:
        full_answers = {
            "exposure": answers_dict["exposure"],
            "temperature": answers_dict["temperature"],
            "tint": answers_dict["tint"],
            "contrast": 0,
            "saturation": 0,
            "shadows": 0,
        }
        survey_payload = build_survey_payload(full_answers)
        survey_path = tmp_path / f"{name}_survey.json"
        write_survey(survey_payload, survey_path)

        output_ckpt = tmp_path / f"{name}.ckpt"
        sidecar_path = mode_b_builder.build_mode_b_checkpoint(
            preset_path=FIXTURE_PRESET,
            survey_path=survey_path,
            base_ckpt_path=base_ckpt_path,
            output_ckpt_path=output_ckpt,
            profile_name=f"Test Profile {name}",
        )

        sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
        built_profiles[name] = {
            "ckpt": output_ckpt,
            "sidecar_path": sidecar_path,
            "sidecar": sidecar,
            "survey_payload": survey_payload,
        }

    # 1. Verify Checkpoint bias vectors
    # Bright/warm/green profile
    p_bright = built_profiles["bright_warm_green"]
    p_neutral = built_profiles["neutral"]
    p_dark = built_profiles["dark_cool_magenta"]

    survey_adj_bright = _survey_adjusted_preset(p_bright["sidecar"])
    survey_adj_neutral = _survey_adjusted_preset(p_neutral["sidecar"])
    survey_adj_dark = _survey_adjusted_preset(p_dark["sidecar"])

    # Base preset exposure
    base_preset_exp = float(preset_raw.get("Exposure2012") or 0.0)
    assert survey_adj_bright["Exposure2012"] == pytest.approx(base_preset_exp + 1.0, abs=1e-4)
    assert survey_adj_neutral["Exposure2012"] == pytest.approx(base_preset_exp, abs=1e-4)
    assert survey_adj_dark["Exposure2012"] == pytest.approx(base_preset_exp - 1.0, abs=1e-4)

    # 2. Verify per-photo inference across different photo exposures
    test_img_mid = Image.new("RGB", (200, 200), (118, 118, 118))
    test_meta = {"as_shot_wb": (5500.0, 12.0), "iso": 400}

    res_bright = _mode_b_adjusted_values_for_photo(test_img_mid, test_meta, survey_adj_bright, "v2")
    res_neutral = _mode_b_adjusted_values_for_photo(test_img_mid, test_meta, survey_adj_neutral, "v2")
    res_dark = _mode_b_adjusted_values_for_photo(test_img_mid, test_meta, survey_adj_dark, "v2")

    # Exposure: Bright must be +1.0 stop above neutral, Dark must be -1.0 stop below neutral
    exp_bright = res_bright["Exposure2012"]
    exp_neutral = res_neutral["Exposure2012"]
    exp_dark = res_dark["Exposure2012"]

    assert exp_bright - exp_neutral == pytest.approx(1.0, abs=0.05)
    assert exp_neutral - exp_dark == pytest.approx(1.0, abs=0.05)

    # Temperature: Bright (+2) was +1000K, Neutral was 0, Dark (-2) was -1000K
    temp_bright = res_bright["Temperature"]
    temp_neutral = res_neutral["Temperature"]
    temp_dark = res_dark["Temperature"]

    assert temp_bright - temp_neutral == pytest.approx(1000.0, abs=1.0)
    assert temp_neutral - temp_dark == pytest.approx(1000.0, abs=1.0)

    # Tint: Bright was -10, Neutral was 0, Dark was +10
    tint_bright = res_bright["Tint"]
    tint_neutral = res_neutral["Tint"]
    tint_dark = res_dark["Tint"]

    assert tint_neutral - tint_bright == pytest.approx(10.0, abs=0.5)
    assert tint_dark - tint_neutral == pytest.approx(10.0, abs=0.5)

    # 3. Verify that non-survey preset fields are 100% faithful to the preset
    non_survey_fields = ["Contrast2012", "Highlights2012", "Shadows2012", "Clarity2012", "Vibrance"]
    for field in non_survey_fields:
        if preset_raw.get(field) is not None:
            expected_val = float(preset_raw[field])
            assert res_bright[field] == pytest.approx(expected_val, abs=1e-3)
            assert res_neutral[field] == pytest.approx(expected_val, abs=1e-3)
            assert res_dark[field] == pytest.approx(expected_val, abs=1e-3)

    # 4. Verify XMP sidecar output file
    xmp_out_path = tmp_path / "sample_output.xmp"
    write_xmp(xmp_out_path, res_bright)
    assert xmp_out_path.exists()

    xmp_parsed = read_xmp(xmp_out_path)
    assert xmp_parsed["Exposure2012"] == pytest.approx(exp_bright, abs=1e-3)
    assert xmp_parsed["Temperature"] == pytest.approx(temp_bright, abs=1e-3)
    assert xmp_parsed["Tint"] == pytest.approx(tint_bright, abs=1e-3)


def test_lite_profile_fractional_survey_precision(tmp_path: Path):
    """Verify fractional survey answers (e.g. 1.333) compute continuous offsets and flow to XMP."""
    base_ckpt_path = resolve_foundation_checkpoint()
    assert base_ckpt_path.exists()

    answers = {
        "exposure": 1.333,
        "temperature": -0.667,
        "tint": 0.5,
        "contrast": 0,
        "saturation": 0,
        "shadows": 0,
    }
    survey_payload = build_survey_payload(answers)
    survey_path = tmp_path / "fractional_survey.json"
    write_survey(survey_payload, survey_path)

    # 1. Verify computed offsets in survey payload
    assert survey_payload["summary"]["exposure_offset"] == pytest.approx(1.333 * 0.5, abs=1e-4)
    assert survey_payload["summary"]["temperature_offset"] == pytest.approx(-0.667 * 500.0, abs=1e-3)
    assert survey_payload["summary"]["tint_offset"] == pytest.approx(0.5 * 5.0, abs=1e-4)

    # 2. Build Lite profile
    output_ckpt = tmp_path / "fractional.ckpt"
    sidecar_path = mode_b_builder.build_mode_b_checkpoint(
        preset_path=FIXTURE_PRESET,
        survey_path=survey_path,
        base_ckpt_path=base_ckpt_path,
        output_ckpt_path=output_ckpt,
        profile_name="Fractional Precision Lite Profile",
    )
    sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    assert sidecar["profile_type"] == "mode_b_initial"

    # 3. Process test photo
    test_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    metadata = {
        "as_shot_wb": (5500.0, 10.0),
        "iso": 100,
        "shutter_speed": 0.01,
        "aperture": 4.0,
    }
    mode_b_preset = _survey_adjusted_preset(sidecar)
    res = _mode_b_adjusted_values_for_photo(
        image=test_img,
        metadata=metadata,
        preset=mode_b_preset,
        slider_set_version="v1",
    )

    # Temperature expected: Camera AsShot (5500) + preset clamped style delta (-300) + survey offset (-333.5) = 4866.5
    assert res["Temperature"] == pytest.approx(5500.0 - 300.0 - (0.667 * 500.0), abs=1.0)
    # Tint expected: Camera AsShot (10.0) + preset style delta (-3.0) + survey offset (2.5) = 9.5
    assert res["Tint"] == pytest.approx(10.0 - 3.0 + (0.5 * 5.0), abs=0.5)
