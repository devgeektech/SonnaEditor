from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from sonna_editor.config import SLIDER_FIELDS
from sonna_editor.preset.parser import parse_preset, validate_preset

FIXTURE_DIR = Path(__file__).parent / "fixtures"
PRESET_XMP = FIXTURE_DIR / "preset_sonna_v1.xmp"
PRESET_LRTEMPLATE = FIXTURE_DIR / "preset_sonna_legacy.lrtemplate"
PRESET_XMPSETTINGS = FIXTURE_DIR / "preset_sonna_warm.xmpsettings"


# ---------------------------------------------------------------------------
# parse_preset — .xmp format
# ---------------------------------------------------------------------------

def test_parse_xmp_returns_all_slider_fields() -> None:
    result = parse_preset(PRESET_XMP)
    assert set(result.keys()) == set(SLIDER_FIELDS)


def test_parse_xmp_correct_exposure() -> None:
    result = parse_preset(PRESET_XMP)
    assert result["Exposure2012"] == pytest.approx(0.35)


def test_parse_xmp_correct_temperature() -> None:
    result = parse_preset(PRESET_XMP)
    assert result["Temperature"] == pytest.approx(5200.0)


def test_parse_xmp_correct_highlights() -> None:
    result = parse_preset(PRESET_XMP)
    assert result["Highlights2012"] == pytest.approx(-45.0)


def test_parse_xmp_missing_fields_are_none() -> None:
    result = parse_preset(PRESET_XMP)
    # Fields absent from the preset file must be None, not 0.0.
    # This prevents incorrect clamping (e.g. PerspectiveScale 0→50, Temperature 0→2000K).
    assert result["HueAdjustmentRed"] is None
    assert result["LuminanceAdjustmentBlue"] is None


# ---------------------------------------------------------------------------
# parse_preset — .lrtemplate format
# ---------------------------------------------------------------------------

def test_parse_lrtemplate_returns_all_slider_fields() -> None:
    result = parse_preset(PRESET_LRTEMPLATE)
    assert set(result.keys()) == set(SLIDER_FIELDS)


def test_parse_lrtemplate_correct_exposure() -> None:
    result = parse_preset(PRESET_LRTEMPLATE)
    assert result["Exposure2012"] == pytest.approx(0.55)


def test_parse_lrtemplate_correct_temperature() -> None:
    result = parse_preset(PRESET_LRTEMPLATE)
    assert result["Temperature"] == pytest.approx(5400.0)


def test_parse_lrtemplate_correct_hsl_value() -> None:
    result = parse_preset(PRESET_LRTEMPLATE)
    assert result["SaturationAdjustmentRed"] == pytest.approx(-10.0)
    assert result["SaturationAdjustmentBlue"] == pytest.approx(-8.0)


def test_parse_lrtemplate_missing_fields_are_none() -> None:
    result = parse_preset(PRESET_LRTEMPLATE)
    assert result["Dehaze"] is None
    assert result["HueAdjustmentRed"] is None


# ---------------------------------------------------------------------------
# parse_preset — .xmpsettings format
# ---------------------------------------------------------------------------

def test_parse_xmpsettings_returns_all_slider_fields() -> None:
    result = parse_preset(PRESET_XMPSETTINGS)
    assert set(result.keys()) == set(SLIDER_FIELDS)


def test_parse_xmpsettings_correct_temperature() -> None:
    result = parse_preset(PRESET_XMPSETTINGS)
    assert result["Temperature"] == pytest.approx(6200.0)


def test_parse_xmpsettings_correct_hue_adjustment() -> None:
    result = parse_preset(PRESET_XMPSETTINGS)
    assert result["HueAdjustmentOrange"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# parse_preset — edge cases
# ---------------------------------------------------------------------------

def test_parse_xmp_xpacket_wrapped(tmp_path: Path) -> None:
    content = (
        '<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">\n'
        '  <rdf:RDF>\n'
        '    <rdf:Description crs:Exposure2012="1.25" crs:Temperature="4800"/>\n'
        '  </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>'
    )
    p = tmp_path / "wrapped.xmp"
    p.write_text(content, encoding="utf-8")
    result = parse_preset(p)
    assert result["Exposure2012"] == pytest.approx(1.25)
    assert result["Temperature"] == pytest.approx(4800.0)


def test_parse_lrtemplate_ignores_non_slider_keys(tmp_path: Path) -> None:
    content = textwrap.dedent("""\
        s = {
            value = {
                settings = {
                    Exposure2012 = 0.75,
                    SomeRandomKey = 99,
                    AnotherKey = 123,
                },
            },
        }
    """)
    p = tmp_path / "test.lrtemplate"
    p.write_text(content)
    result = parse_preset(p)
    assert result["Exposure2012"] == pytest.approx(0.75)
    assert "SomeRandomKey" not in result


def test_parse_preset_unsupported_extension_raises(tmp_path: Path) -> None:
    p = tmp_path / "preset.txt"
    p.write_text("Exposure2012 = 1.0")
    with pytest.raises(ValueError, match="Unsupported preset format"):
        parse_preset(p)


def test_parse_lrtemplate_local_adjustments_ignored(tmp_path: Path) -> None:
    content = textwrap.dedent("""\
        s = {
            value = {
                settings = {
                    Exposure2012 = 0.50,
                    GradientBasedCorrections = {
                        { What = "GradientLinear", Exposure2012 = -1.5 },
                    },
                },
            },
        }
    """)
    p = tmp_path / "local.lrtemplate"
    p.write_text(content)
    result = parse_preset(p)
    # Should parse the global exposure but not the local one
    assert result["Exposure2012"] == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# validate_preset
# ---------------------------------------------------------------------------

def test_validate_clean_preset_returns_no_warnings() -> None:
    preset = parse_preset(PRESET_XMP)
    warnings = validate_preset(preset)
    assert warnings == []


def test_validate_flags_extreme_exposure() -> None:
    preset = {f: 0.0 for f in SLIDER_FIELDS}
    preset["Exposure2012"] = 4.5
    warnings = validate_preset(preset)
    assert any("Exposure2012" in w for w in warnings)


def test_validate_flags_extreme_saturation() -> None:
    preset = {f: 0.0 for f in SLIDER_FIELDS}
    preset["Saturation"] = -85.0
    warnings = validate_preset(preset)
    assert any("Saturation" in w for w in warnings)


def test_validate_flags_monochrome_hsl() -> None:
    preset = {f: 0.0 for f in SLIDER_FIELDS}
    for f in SLIDER_FIELDS:
        if f.startswith("SaturationAdjustment"):
            preset[f] = -100.0
    warnings = validate_preset(preset)
    assert any("monochrome" in w.lower() or "HSL" in w for w in warnings)


def test_validate_returns_list() -> None:
    preset = {f: 0.0 for f in SLIDER_FIELDS}
    result = validate_preset(preset)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tone curve parsing in .xmp presets
# ---------------------------------------------------------------------------

def test_parse_xmp_reads_tone_curve_rdf_seq(tmp_path: Path) -> None:
    """Regression: parser must extract rdf:Seq tone curve elements, not silently drop them.
    4 raw rdf:li points → normalized to 6 via piecewise-linear interpolation.
    """
    content = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"'
        ' xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
        ' xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">\n'
        '  <rdf:RDF>\n'
        '    <rdf:Description crs:Exposure2012="0.5">\n'
        '      <crs:ToneCurvePV2012>\n'
        '        <rdf:Seq>\n'
        '          <rdf:li>0, 10</rdf:li>\n'
        '          <rdf:li>85, 90</rdf:li>\n'
        '          <rdf:li>170, 175</rdf:li>\n'
        '          <rdf:li>255, 245</rdf:li>\n'
        '        </rdf:Seq>\n'
        '      </crs:ToneCurvePV2012>\n'
        '    </rdf:Description>\n'
        '  </rdf:RDF>\n'
        '</x:xmpmeta>\n'
    )
    p = tmp_path / "curves.xmp"
    p.write_text(content, encoding="utf-8")
    result = parse_preset(p)

    # Composite: 4 raw points → interpolated to 6 at x=[0,51,102,153,204,255]
    assert result["ToneCurve_Pt1_X"] == pytest.approx(0.0)
    assert result["ToneCurve_Pt1_Y"] == pytest.approx(10.0)   # direct from (0,10)
    assert result["ToneCurve_Pt6_X"] == pytest.approx(255.0)
    assert result["ToneCurve_Pt6_Y"] == pytest.approx(245.0)  # direct from (255,245)
    # Channels absent from preset are None (not identity defaults)
    assert result["ToneCurveRed_Pt1_X"] is None
    assert result["ToneCurveBlue_Pt6_Y"] is None


def test_parse_xmp_tone_curves_absent_returns_none(tmp_path: Path) -> None:
    """Absent tone curves in an .xmp preset must be None, not identity floats."""
    content = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"'
        ' xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
        ' xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">\n'
        '  <rdf:RDF>\n'
        '    <rdf:Description crs:Exposure2012="1.0"/>\n'
        '  </rdf:RDF>\n'
        '</x:xmpmeta>\n'
    )
    p = tmp_path / "no_curves.xmp"
    p.write_text(content, encoding="utf-8")
    result = parse_preset(p)

    for f in SLIDER_FIELDS:
        if f.startswith("ToneCurve"):
            assert result[f] is None, f"expected None for absent {f}, got {result[f]!r}"


def test_parse_real_dp_event_preset_tone_curves() -> None:
    """DP Event.xmp has non-identity curves — spot-check key values.
    Composite: 7-point → 6 via downsampling (idxs=[0,1,2,4,5,6]).
    Red: 4-point → 6 via interpolation at x=[0,51,102,153,204,255].
    """
    real_preset = Path("test_data/Preset/DP Event.xmp")
    if not real_preset.exists():
        pytest.skip("test_data/Preset/DP Event.xmp not present")
    result = parse_preset(real_preset)
    # Composite: first point (0,19) and last point (255,247) are preserved by downsampling
    assert result["ToneCurve_Pt1_Y"] == pytest.approx(19.0)
    assert result["ToneCurve_Pt6_X"] == pytest.approx(255.0)
    assert result["ToneCurve_Pt6_Y"] == pytest.approx(247.0)
    # Red: raw endpoints (0,0) and (255,255) are preserved through interpolation
    assert result["ToneCurveRed_Pt1_Y"] == pytest.approx(0.0)
    assert result["ToneCurveRed_Pt6_Y"] == pytest.approx(255.0)
    # Red shadow pull-down: at x=51 (interp between (0,0) and (53,29)), y ≈ 28 — not identity (51)
    assert result["ToneCurveRed_Pt2_X"] == pytest.approx(51.0)
    assert result["ToneCurveRed_Pt2_Y"] == pytest.approx(28.0, abs=2)
