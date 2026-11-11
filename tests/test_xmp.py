from __future__ import annotations

from pathlib import Path

import pytest

from sonna_editor.config import SLIDER_FIELDS
from sonna_editor.data.xmp import read_xmp, write_xmp

FIXTURE = Path(__file__).parent / "fixtures" / "sample_edit.xmp"
requires_sample_edit_xmp = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason=f"XMP fixture not present: {FIXTURE}",
)


@requires_sample_edit_xmp
class TestReadXmp:
    def test_fixture_parses(self):
        result = read_xmp(FIXTURE)
        assert isinstance(result, dict)
        assert set(SLIDER_FIELDS).issubset(result.keys())

    def test_fixture_exposure(self):
        result = read_xmp(FIXTURE)
        # sample_edit.xmp has crs:Exposure2012="+0.60"
        assert result["Exposure2012"] == pytest.approx(0.60)

    def test_fixture_temperature(self):
        result = read_xmp(FIXTURE)
        assert result["Temperature"] == pytest.approx(5500.0)

    def test_fixture_hsl_present(self):
        result = read_xmp(FIXTURE)
        # HueAdjustmentOrange="+7" in the fixture
        assert result["HueAdjustmentOrange"] == pytest.approx(7.0)
        assert result["HueAdjustmentYellow"] == pytest.approx(-22.0)

    def test_fixture_saturation_adjustment(self):
        result = read_xmp(FIXTURE)
        assert result["SaturationAdjustmentOrange"] == pytest.approx(-9.0)

    def test_fixture_luminance_adjustment(self):
        result = read_xmp(FIXTURE)
        assert result["LuminanceAdjustmentYellow"] == pytest.approx(24.0)

    def test_missing_field_returns_none(self, tmp_path):
        # Write an XMP that only has Exposure2012
        xmp_path = tmp_path / "sparse.xmp"
        write_xmp(xmp_path, {"Exposure2012": 0.5})
        result = read_xmp(xmp_path)
        assert result["Exposure2012"] == pytest.approx(0.5)
        assert result["Contrast2012"] is None
        assert result["HueAdjustmentRed"] is None

    def test_nonexistent_dng_returns_nones(self, tmp_path):
        # A file that has no XMP block
        fake_dng = tmp_path / "empty.dng"
        fake_dng.write_bytes(b"\x00" * 64)
        result = read_xmp(fake_dng)
        assert all(v is None for v in result.values())


class TestWriteXmp:
    def _full_settings(self) -> dict:
        settings: dict = {}
        for field in SLIDER_FIELDS:
            settings[field] = 0.0
        settings["Exposure2012"] = 1.25
        settings["Temperature"] = 5200.0
        settings["Tint"] = -5.0
        settings["HueAdjustmentOrange"] = 7.0
        settings["SaturationAdjustmentGreen"] = -10.0
        settings["LuminanceAdjustmentBlue"] = 8.0
        return settings

    def test_writes_file(self, tmp_path):
        xmp_path = tmp_path / "out.xmp"
        write_xmp(xmp_path, self._full_settings())
        assert xmp_path.exists()
        assert xmp_path.stat().st_size > 0

    def test_xmp_packet_markers_present(self, tmp_path):
        xmp_path = tmp_path / "out.xmp"
        write_xmp(xmp_path, self._full_settings())
        raw = xmp_path.read_bytes()
        assert b"<?xpacket begin=" in raw
        assert b'<?xpacket end="w"?>' in raw

    def test_has_settings_true(self, tmp_path):
        xmp_path = tmp_path / "out.xmp"
        write_xmp(xmp_path, self._full_settings())
        raw = xmp_path.read_text()
        assert 'HasSettings="True"' in raw

    def test_process_version(self, tmp_path):
        xmp_path = tmp_path / "out.xmp"
        write_xmp(xmp_path, self._full_settings())
        raw = xmp_path.read_text()
        assert 'ProcessVersion="15.4"' in raw

    def test_crs_namespace_declared(self, tmp_path):
        xmp_path = tmp_path / "out.xmp"
        write_xmp(xmp_path, self._full_settings())
        raw = xmp_path.read_text()
        assert "http://ns.adobe.com/camera-raw-settings/1.0/" in raw


class TestRoundTrip:
    def test_numeric_round_trip(self, tmp_path):
        settings = {field: 0.0 for field in SLIDER_FIELDS}
        settings["Exposure2012"] = 1.25
        settings["Temperature"] = 5200.0
        settings["Tint"] = -5.0
        settings["HueAdjustmentOrange"] = 7.0
        settings["SaturationAdjustmentGreen"] = -10.0
        settings["LuminanceAdjustmentBlue"] = 8.0
        settings["Blacks2012"] = -27.0
        settings["Whites2012"] = -15.0

        xmp_path = tmp_path / "roundtrip.xmp"
        write_xmp(xmp_path, settings)
        result = read_xmp(xmp_path)

        for field in SLIDER_FIELDS:
            expected = settings[field]
            assert result[field] == pytest.approx(expected, abs=1e-4), (
                f"{field}: wrote {expected}, read back {result[field]}"
            )

    def test_zero_values_round_trip(self, tmp_path):
        settings = {field: 0.0 for field in SLIDER_FIELDS}
        xmp_path = tmp_path / "zeros.xmp"
        write_xmp(xmp_path, settings)
        result = read_xmp(xmp_path)
        for field in SLIDER_FIELDS:
            if field == "Temperature":
                # Temperature=0 means "as-shot" and is intentionally omitted from
                # the XMP file so Lightroom uses the RAW's embedded WB. Reads back as None.
                assert result[field] is None, "Temperature=0 should be omitted (as-shot)"
            else:
                assert result[field] == pytest.approx(0.0), f"{field} failed"

    def test_pre_saha_snapshot_present(self, tmp_path):
        """Every written XMP must include a 'Pre-Saha' snapshot for before/after."""
        xmp_path = tmp_path / "snap.xmp"
        write_xmp(xmp_path, {"Exposure2012": 1.5, "Shadows2012": 30.0})
        raw = xmp_path.read_text()

        assert "<crs:Snapshots>" in raw
        assert 'crs:Name="Pre-Saha"' in raw
        assert 'crs:Type="Develop"' in raw
        # Parameters string covers every scalar slider at its LR default —
        # check a few tone fields are there with the identity value.
        assert "Exposure2012 = 0" in raw
        assert "Shadows2012 = 0" in raw
        # Tone curves serialised as flattened pairs
        assert "ToneCurvePV2012 = 0, 0, 51, 51" in raw
        # Sharpening uses LR's 25 default, not 0
        assert "Sharpness = 25" in raw

    def test_pre_saha_snapshot_uses_extracted_wb(self, tmp_path, monkeypatch):
        """When source_raw_path yields As-Shot WB, snapshot Temperature/Tint reflect it."""
        from sonna_editor.data import xmp as xmp_mod
        # Simulate a successful AsShot extraction returning 4200K / -3.5 tint.
        monkeypatch.setattr(xmp_mod, "_extract_as_shot_wb", lambda p: (4200.0, -3.5))

        xmp_path = tmp_path / "wb.xmp"
        fake_raw = tmp_path / "fake.cr3"
        fake_raw.write_bytes(b"")
        write_xmp(xmp_path, {"Exposure2012": 0.5}, source_raw_path=fake_raw)
        raw = xmp_path.read_text()

        assert "Temperature = 4200" in raw
        assert "Tint = -3.5" in raw

    def test_pre_saha_snapshot_falls_back_when_extraction_fails(self, tmp_path, monkeypatch):
        """If As-Shot extraction returns None, snapshot uses 5500/0 placeholder."""
        from sonna_editor.data import xmp as xmp_mod
        monkeypatch.setattr(xmp_mod, "_extract_as_shot_wb", lambda p: None)

        xmp_path = tmp_path / "wb_fallback.xmp"
        fake_raw = tmp_path / "fake.cr3"
        fake_raw.write_bytes(b"")
        write_xmp(xmp_path, {"Exposure2012": 0.5}, source_raw_path=fake_raw)
        raw = xmp_path.read_text()

        assert "Temperature = 5500" in raw
        assert "Tint = 0" in raw

    def test_none_values_excluded(self, tmp_path):
        settings: dict = {"Exposure2012": 0.5}
        xmp_path = tmp_path / "sparse.xmp"
        write_xmp(xmp_path, settings)
        raw = xmp_path.read_text()
        # Pre-Saha snapshot lists every field at its default in crs:Parameters,
        # so naked substring checks pick that up. Verify the crs:Contrast2012
        # ATTRIBUTE on the main Description is absent (the snapshot serialises
        # parameters as 'Contrast2012 = 0' inside an attribute value).
        assert 'crs:Contrast2012="' not in raw

    def test_source_raw_path(self, tmp_path):
        xmp_path = tmp_path / "out.xmp"
        raw_path = Path("/shoots/job123/IMG_0001.CR3")
        write_xmp(xmp_path, {"Exposure2012": 0.0}, source_raw_path=raw_path)
        raw = xmp_path.read_text()
        assert "IMG_0001.CR3" in raw

    @requires_sample_edit_xmp
    def test_fixture_values_survives_write_read(self, tmp_path):
        original = read_xmp(FIXTURE)
        xmp_path = tmp_path / "rewritten.xmp"
        write_xmp(xmp_path, original)
        reread = read_xmp(xmp_path)

        for field in SLIDER_FIELDS:
            orig_val = original[field]
            new_val = reread[field]
            if orig_val is None:
                assert new_val is None, f"{field}: None not preserved"
            elif isinstance(orig_val, float):
                assert new_val == pytest.approx(orig_val, abs=1e-4), (
                    f"{field}: {orig_val} -> {new_val}"
                )


class TestNewFieldsRoundTrip:
    """XMP round-trip tests for all 45 new slider fields added in the 37→82 expansion."""

    def _roundtrip(self, settings: dict, tmp_path) -> dict:
        xmp_path = tmp_path / "test.xmp"
        write_xmp(xmp_path, settings)
        return read_xmp(xmp_path)

    def test_parametric_tone_curve(self, tmp_path) -> None:
        settings = {
            "ParametricHighlights": 20.0,
            "ParametricLights": -15.0,
            "ParametricDarks": 10.0,
            "ParametricShadows": -25.0,
            "ParametricHighlightSplit": 75.0,
            "ParametricMidtoneSplit": 50.0,
            "ParametricShadowSplit": 25.0,
        }
        result = self._roundtrip(settings, tmp_path)
        for field, expected in settings.items():
            assert result[field] == pytest.approx(expected, abs=1e-4), field

    def test_color_grading_shadows(self, tmp_path) -> None:
        settings = {
            "SplitToningShadowHue": 30.0,
            "SplitToningShadowSaturation": 20.0,
            "ColorGradeShadowLum": 10.0,
        }
        result = self._roundtrip(settings, tmp_path)
        for field, expected in settings.items():
            assert result[field] == pytest.approx(expected, abs=1e-4), field

    def test_color_grading_midtones(self, tmp_path) -> None:
        settings = {
            "ColorGradeMidtoneHue": 62.0,
            "ColorGradeMidtoneSat": 0.0,
            "ColorGradeMidtoneLum": 12.0,
        }
        result = self._roundtrip(settings, tmp_path)
        for field, expected in settings.items():
            assert result[field] == pytest.approx(expected, abs=1e-4), field

    def test_color_grading_highlights(self, tmp_path) -> None:
        settings = {
            "SplitToningHighlightHue": 180.0,
            "SplitToningHighlightSaturation": 15.0,
            "ColorGradeHighlightLum": -8.0,
        }
        result = self._roundtrip(settings, tmp_path)
        for field, expected in settings.items():
            assert result[field] == pytest.approx(expected, abs=1e-4), field

    def test_color_grading_global_and_blending(self, tmp_path) -> None:
        settings = {
            "ColorGradeBlending": 50.0,
            "ColorGradeGlobalHue": 0.0,
            "ColorGradeGlobalSat": 0.0,
            "ColorGradeGlobalLum": 0.0,
        }
        result = self._roundtrip(settings, tmp_path)
        for field, expected in settings.items():
            assert result[field] == pytest.approx(expected, abs=1e-4), field

    def test_colorgrade_lum_suffix_not_luminance(self, tmp_path) -> None:
        """Verify abbreviated Lum attribute (not Luminance) is written to XMP."""
        settings = {"ColorGradeShadowLum": 15.0}
        xmp_path = tmp_path / "lum.xmp"
        write_xmp(xmp_path, settings)
        raw = xmp_path.read_text()
        assert "ColorGradeShadowLum=" in raw
        assert "ColorGradeShadowLuminance" not in raw

    def test_camera_calibration(self, tmp_path) -> None:
        settings = {
            "RedHue": 5.0,
            "RedSaturation": -10.0,
            "GreenHue": -3.0,
            "GreenSaturation": 7.0,
            "BlueHue": 2.0,
            "BlueSaturation": -5.0,
        }
        result = self._roundtrip(settings, tmp_path)
        for field, expected in settings.items():
            assert result[field] == pytest.approx(expected, abs=1e-4), field

    def test_sharpening(self, tmp_path) -> None:
        settings = {
            "Sharpness": 56.0,
            "SharpenDetail": 25.0,
            "SharpenEdgeMasking": 90.0,
        }
        result = self._roundtrip(settings, tmp_path)
        for field, expected in settings.items():
            assert result[field] == pytest.approx(expected, abs=1e-4), field

    def test_sharpening_radius_float_range(self, tmp_path) -> None:
        """SharpenRadius has non-integer range (0.5–3.0)."""
        for radius in [0.5, 1.0, 1.5, 2.0, 3.0]:
            result = self._roundtrip({"SharpenRadius": radius}, tmp_path)
            assert result["SharpenRadius"] == pytest.approx(radius, abs=1e-4), (
                f"SharpenRadius={radius} failed round-trip"
            )

    def test_noise_reduction(self, tmp_path) -> None:
        settings = {
            "LuminanceSmoothing": 14.0,
            "LuminanceNoiseReductionDetail": 50.0,
            "LuminanceNoiseReductionContrast": 0.0,
            "ColorNoiseReduction": 25.0,
        }
        result = self._roundtrip(settings, tmp_path)
        for field, expected in settings.items():
            assert result[field] == pytest.approx(expected, abs=1e-4), field

    def test_effects_vignette_and_grain(self, tmp_path) -> None:
        settings = {
            "PostCropVignetteAmount": -30.0,
            "PostCropVignetteMidpoint": 50.0,
            "PostCropVignetteRoundness": 0.0,
            "GrainAmount": 20.0,
        }
        result = self._roundtrip(settings, tmp_path)
        for field, expected in settings.items():
            assert result[field] == pytest.approx(expected, abs=1e-4), field

    def test_lens_corrections(self, tmp_path) -> None:
        settings = {
            "LensManualDistortionAmount": 5.0,
            "VignetteAmount": -10.0,
        }
        result = self._roundtrip(settings, tmp_path)
        for field, expected in settings.items():
            assert result[field] == pytest.approx(expected, abs=1e-4), field

    def test_transform_standard_fields(self, tmp_path) -> None:
        settings = {
            "PerspectiveVertical": 10.0,
            "PerspectiveHorizontal": -5.0,
            "PerspectiveAspect": 0.0,
        }
        result = self._roundtrip(settings, tmp_path)
        for field, expected in settings.items():
            assert result[field] == pytest.approx(expected, abs=1e-4), field

    def test_transform_non_standard_ranges(self, tmp_path) -> None:
        """PerspectiveRotate (-10, 10) and PerspectiveScale (50, 150)."""
        settings = {
            "PerspectiveRotate": 5.5,
            "PerspectiveScale": 110.0,
        }
        result = self._roundtrip(settings, tmp_path)
        assert result["PerspectiveRotate"] == pytest.approx(5.5, abs=1e-4)
        assert result["PerspectiveScale"] == pytest.approx(110.0, abs=1e-4)

    def test_none_values_for_new_fields_omitted(self, tmp_path) -> None:
        """New fields set to None must not appear as attributes on the main Description.

        The Pre-Saha snapshot lists every SLIDER_FIELDS at its LR default in
        crs:Parameters, so substring checks would pick those up. The contract
        we care about is that the main rdf:Description has no crs:Field=
        attribute when settings[field] is None.
        """
        settings = {"Exposure2012": 0.5, "SharpenRadius": None, "PerspectiveScale": None}
        xmp_path = tmp_path / "sparse.xmp"
        write_xmp(xmp_path, settings)
        raw = xmp_path.read_text()
        assert 'crs:SharpenRadius="' not in raw
        assert 'crs:PerspectiveScale="' not in raw

    def test_all_50_new_fields_round_trip(self, tmp_path) -> None:
        """Full round-trip for the 50 scalar new fields added in the 37→87 expansion."""
        from sonna_editor.config import SLIDER_FIELDS
        new_fields = SLIDER_FIELDS[37:87]  # scalar new fields (excludes tone curves at 87+)
        assert len(new_fields) == 50

        settings = {f: 1.0 for f in new_fields}
        # Use values within each field's valid range (hue fields must be 0-360)
        settings.update({
            "SplitToningShadowHue": 45.0,
            "ColorGradeMidtoneHue": 90.0,
            "SplitToningHighlightHue": 180.0,
            "ColorGradeGlobalHue": 270.0,
            "SharpenRadius": 1.5,
            "PerspectiveRotate": 2.5,
            "PerspectiveScale": 105.0,
        })

        result = self._roundtrip(settings, tmp_path)
        for field in new_fields:
            expected = settings[field]
            assert result[field] == pytest.approx(expected, abs=1e-4), (
                f"{field}: wrote {expected}, read back {result[field]}"
            )


class TestFixedFieldNamesRoundTrip:
    """Round-trip tests for the 15 field name corrections (10 renames + 5 new)."""

    def _roundtrip(self, settings: dict, tmp_path) -> dict:
        xmp_path = tmp_path / "test.xmp"
        write_xmp(xmp_path, settings)
        return read_xmp(xmp_path)

    # --- Category C: renamed fields ---

    def test_calibration_short_names_roundtrip(self, tmp_path) -> None:
        """Calibration fields use short XMP names (no CameraCalibration prefix)."""
        settings = {
            "RedHue": 8.0, "RedSaturation": -12.0,
            "GreenHue": -4.0, "GreenSaturation": 6.0,
            "BlueHue": 3.0, "BlueSaturation": -7.0,
        }
        result = self._roundtrip(settings, tmp_path)
        for field, expected in settings.items():
            assert result[field] == pytest.approx(expected, abs=1e-4), field

    def test_calibration_attributes_in_raw_xmp(self, tmp_path) -> None:
        """Verify crs:RedHue is written, not crs:CameraCalibrationRedHue."""
        xmp_path = tmp_path / "cal.xmp"
        write_xmp(xmp_path, {"RedHue": 5.0, "GreenHue": -3.0})
        raw = xmp_path.read_text()
        assert 'crs:RedHue=' in raw
        assert 'CameraCalibration' not in raw

    def test_split_toning_shadow_hue_and_sat(self, tmp_path) -> None:
        """Shadow wheel Hue+Sat use legacy SplitToning XMP attribute names."""
        settings = {"SplitToningShadowHue": 337.0, "SplitToningShadowSaturation": 6.0}
        result = self._roundtrip(settings, tmp_path)
        assert result["SplitToningShadowHue"] == pytest.approx(337.0, abs=1e-4)
        assert result["SplitToningShadowSaturation"] == pytest.approx(6.0, abs=1e-4)

    def test_split_toning_highlight_hue_and_sat(self, tmp_path) -> None:
        """Highlight wheel Hue+Sat use legacy SplitToning XMP attribute names."""
        settings = {"SplitToningHighlightHue": 75.0, "SplitToningHighlightSaturation": 6.0}
        result = self._roundtrip(settings, tmp_path)
        assert result["SplitToningHighlightHue"] == pytest.approx(75.0, abs=1e-4)
        assert result["SplitToningHighlightSaturation"] == pytest.approx(6.0, abs=1e-4)

    def test_split_toning_attributes_in_raw_xmp(self, tmp_path) -> None:
        """Verify SplitToningShadowHue is written, not ColorGradeShadowHue."""
        xmp_path = tmp_path / "st.xmp"
        write_xmp(xmp_path, {"SplitToningShadowHue": 90.0, "SplitToningHighlightHue": 200.0})
        raw = xmp_path.read_text()
        assert 'crs:SplitToningShadowHue=' in raw
        assert 'crs:SplitToningHighlightHue=' in raw
        assert 'ColorGradeShadowHue' not in raw
        assert 'ColorGradeHighlightHue' not in raw

    # --- Category D: new fields ---

    def test_split_toning_balance(self, tmp_path) -> None:
        for val in [-50.0, 0.0, 30.0]:
            result = self._roundtrip({"SplitToningBalance": val}, tmp_path)
            assert result["SplitToningBalance"] == pytest.approx(val, abs=1e-4), (
                f"SplitToningBalance={val} failed round-trip"
            )

    def test_grain_size_and_frequency(self, tmp_path) -> None:
        settings = {"GrainAmount": 20.0, "GrainSize": 15.0, "GrainFrequency": 25.0}
        result = self._roundtrip(settings, tmp_path)
        for field, expected in settings.items():
            assert result[field] == pytest.approx(expected, abs=1e-4), field

    def test_grain_attributes_in_raw_xmp(self, tmp_path) -> None:
        xmp_path = tmp_path / "grain.xmp"
        write_xmp(xmp_path, {"GrainSize": 15.0, "GrainFrequency": 20.0})
        raw = xmp_path.read_text()
        assert 'crs:GrainSize=' in raw
        assert 'crs:GrainFrequency=' in raw
        assert 'GrainRoughness' not in raw

    def test_postcrop_vignette_feather(self, tmp_path) -> None:
        for val in [0.0, 50.0, 100.0]:
            result = self._roundtrip({"PostCropVignetteFeather": val}, tmp_path)
            assert result["PostCropVignetteFeather"] == pytest.approx(val, abs=1e-4), (
                f"PostCropVignetteFeather={val} failed round-trip"
            )

    def test_postcrop_vignette_highlight_contrast(self, tmp_path) -> None:
        for val in [0.0, 50.0, 100.0]:
            result = self._roundtrip({"PostCropVignetteHighlightContrast": val}, tmp_path)
            assert result["PostCropVignetteHighlightContrast"] == pytest.approx(val, abs=1e-4), (
                f"PostCropVignetteHighlightContrast={val} failed round-trip"
            )

    def test_all_15_corrected_fields_roundtrip(self, tmp_path) -> None:
        """Single test hitting all 15 fixed fields: 10 renames + 5 new."""
        settings = {
            # 10 renamed (Category C)
            "RedHue": 5.0, "RedSaturation": -8.0,
            "GreenHue": -2.0, "GreenSaturation": 4.0,
            "BlueHue": 1.0, "BlueSaturation": -3.0,
            "SplitToningShadowHue": 337.0, "SplitToningShadowSaturation": 6.0,
            "SplitToningHighlightHue": 75.0, "SplitToningHighlightSaturation": 6.0,
            # 5 new (Category D)
            "SplitToningBalance": 30.0,
            "GrainSize": 15.0, "GrainFrequency": 20.0,
            "PostCropVignetteFeather": 50.0,
            "PostCropVignetteHighlightContrast": 25.0,
        }
        result = self._roundtrip(settings, tmp_path)
        for field, expected in settings.items():
            assert result[field] == pytest.approx(expected, abs=1e-4), (
                f"{field}: wrote {expected}, read back {result[field]}"
            )


class TestToneCurveRoundTrip:
    """Round-trip and normalization tests for the 48 point tone curve fields (6 pts × 4 channels)."""

    _IDENTITY_6 = [(0, 0), (51, 51), (102, 102), (153, 153), (204, 204), (255, 255)]

    def _roundtrip(self, settings: dict, tmp_path) -> dict:
        xmp_path = tmp_path / "test.xmp"
        write_xmp(xmp_path, settings)
        return read_xmp(xmp_path)

    def _curve_settings(self, prefix: str, points: list[tuple[int, int]]) -> dict:
        """Build a settings dict for a single channel from a list of 6 control points."""
        assert len(points) == 6
        settings: dict = {}
        for n, (px, py) in enumerate(points, start=1):
            settings[f"{prefix}_Pt{n}_X"] = float(px)
            settings[f"{prefix}_Pt{n}_Y"] = float(py)
        return settings

    # --- Identity curve ---

    def test_identity_curve_composite_roundtrip(self, tmp_path) -> None:
        """Identity curve (no-edit) survives a write → read cycle unchanged."""
        settings = self._curve_settings("ToneCurve", self._IDENTITY_6)
        result = self._roundtrip(settings, tmp_path)
        for n, (px, py) in enumerate(self._IDENTITY_6, start=1):
            assert result[f"ToneCurve_Pt{n}_X"] == pytest.approx(float(px), abs=1e-4)
            assert result[f"ToneCurve_Pt{n}_Y"] == pytest.approx(float(py), abs=1e-4)

    # --- S-curve (6 points, non-identity) ---

    def test_s_curve_composite_roundtrip(self, tmp_path) -> None:
        pts = [(0, 0), (51, 55), (102, 110), (153, 162), (204, 215), (255, 255)]
        settings = self._curve_settings("ToneCurve", pts)
        result = self._roundtrip(settings, tmp_path)
        for n, (px, py) in enumerate(pts, start=1):
            assert result[f"ToneCurve_Pt{n}_X"] == pytest.approx(float(px), abs=1e-4)
            assert result[f"ToneCurve_Pt{n}_Y"] == pytest.approx(float(py), abs=1e-4)

    # --- All 4 channels independently ---

    @pytest.mark.parametrize("prefix,crs_suffix", [
        ("ToneCurve", "ToneCurvePV2012"),
        ("ToneCurveRed", "ToneCurvePV2012Red"),
        ("ToneCurveGreen", "ToneCurvePV2012Green"),
        ("ToneCurveBlue", "ToneCurvePV2012Blue"),
    ])
    def test_each_channel_roundtrip(self, prefix: str, crs_suffix: str, tmp_path) -> None:
        pts = [(0, 10), (51, 56), (102, 112), (153, 163), (204, 214), (255, 245)]
        settings = self._curve_settings(prefix, pts)
        result = self._roundtrip(settings, tmp_path)
        for n, (px, py) in enumerate(pts, start=1):
            assert result[f"{prefix}_Pt{n}_X"] == pytest.approx(float(px), abs=1e-4), (
                f"{prefix}_Pt{n}_X"
            )
            assert result[f"{prefix}_Pt{n}_Y"] == pytest.approx(float(py), abs=1e-4), (
                f"{prefix}_Pt{n}_Y"
            )

    # --- XMP element structure ---

    def test_tone_curve_written_as_seq_element(self, tmp_path) -> None:
        """Tone curves must appear as rdf:Seq child elements, not crs: attributes."""
        settings = self._curve_settings("ToneCurve", self._IDENTITY_6)
        xmp_path = tmp_path / "curve.xmp"
        write_xmp(xmp_path, settings)
        raw = xmp_path.read_text()
        assert "<crs:ToneCurvePV2012>" in raw
        assert "<rdf:Seq>" in raw
        assert "<rdf:li>" in raw
        assert 'crs:ToneCurve_Pt1_X=' not in raw, "Pt scalars must NOT appear as attributes"

    def test_all_4_curve_elements_present(self, tmp_path) -> None:
        """All 4 channel elements must be written even when not specified (identity defaults)."""
        xmp_path = tmp_path / "default_curves.xmp"
        write_xmp(xmp_path, {})
        raw = xmp_path.read_text()
        for tag in ["ToneCurvePV2012", "ToneCurvePV2012Red", "ToneCurvePV2012Green", "ToneCurvePV2012Blue"]:
            assert f"<crs:{tag}>" in raw, f"Missing element crs:{tag}"

    def test_identity_default_writes_6_li_elements(self, tmp_path) -> None:
        """Each channel written with empty settings must produce exactly 6 rdf:li entries.

        The Pre-Saha snapshot adds one extra <rdf:li> wrapping its rdf:Bag
        entry, so we count the curve-specific ones (which uniquely contain
        a comma-separated 'x, y' pair) rather than every <rdf:li>.
        """
        import re
        xmp_path = tmp_path / "default_curves.xmp"
        write_xmp(xmp_path, {})
        raw = xmp_path.read_text()
        # Curve points are <rdf:li>X, Y</rdf:li> — match digit, comma, space, digit
        curve_li = re.findall(r"<rdf:li>\d+,\s*\d+</rdf:li>", raw)
        assert len(curve_li) == 24, f"expected 24 curve points, got {len(curve_li)}"

    # --- Missing channel → identity defaults ---

    def test_missing_channel_returns_identity(self, tmp_path) -> None:
        """When a channel is absent from XMP, parser must return identity curve values."""
        # Write only composite channel — R/G/B will be absent, should default to identity
        pts = [(0, 5), (51, 56), (102, 107), (153, 158), (204, 209), (255, 250)]
        settings = self._curve_settings("ToneCurve", pts)
        xmp_path = tmp_path / "partial.xmp"
        write_xmp(xmp_path, settings)
        result = read_xmp(xmp_path)
        # R/G/B channels should have 6-point identity defaults
        for prefix in ["ToneCurveRed", "ToneCurveGreen", "ToneCurveBlue"]:
            for n, (px, py) in enumerate(self._IDENTITY_6, start=1):
                assert result[f"{prefix}_Pt{n}_X"] == pytest.approx(float(px), abs=1e-4)
                assert result[f"{prefix}_Pt{n}_Y"] == pytest.approx(float(py), abs=1e-4)

    # --- Missing element (hand-crafted XMP) → identity defaults (FLAG 3 coverage) ---

    def test_parse_xmp_with_no_curve_elements_returns_identity(self, tmp_path) -> None:
        """_parse_tone_curve_element returning [] must fall through to 6-point identity defaults.

        write_xmp always writes 4 curve elements, so we must hand-craft an XMP that
        has none to hit the 'missing element → identity' code path in _parse_xmp_bytes.
        """
        from sonna_editor.data.xmp import _parse_xmp_bytes

        minimal_xmp = b"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
      crs:Exposure2012="+0.50"
      crs:ProcessVersion="15.4"
    />
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
        result = _parse_xmp_bytes(minimal_xmp)

        assert result["Exposure2012"] == pytest.approx(0.5, abs=1e-4)
        for prefix in ["ToneCurve", "ToneCurveRed", "ToneCurveGreen", "ToneCurveBlue"]:
            for n, (px, py) in enumerate(self._IDENTITY_6, start=1):
                assert result[f"{prefix}_Pt{n}_X"] == pytest.approx(float(px), abs=1e-4), (
                    f"Missing {prefix}_Pt{n}_X should default to {px}"
                )
                assert result[f"{prefix}_Pt{n}_Y"] == pytest.approx(float(py), abs=1e-4), (
                    f"Missing {prefix}_Pt{n}_Y should default to {py}"
                )

    def test_parse_xmp_with_partial_channels_fills_missing_with_identity(self, tmp_path) -> None:
        """XMP with only composite channel (4 raw rdf:li): green and blue must default to identity,
        composite gets interpolated to 6 points."""
        from sonna_editor.data.xmp import _parse_xmp_bytes

        partial_xmp = b"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
      <crs:ToneCurvePV2012>
        <rdf:Seq>
          <rdf:li>0, 10</rdf:li>
          <rdf:li>85, 95</rdf:li>
          <rdf:li>170, 180</rdf:li>
          <rdf:li>255, 245</rdf:li>
        </rdf:Seq>
      </crs:ToneCurvePV2012>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
        result = _parse_xmp_bytes(partial_xmp)

        # Composite: 4 raw points → interpolated to 6 at x=[0,51,102,153,204,255]
        assert result["ToneCurve_Pt1_Y"] == pytest.approx(10.0, abs=1e-4)   # (0,10) direct
        assert result["ToneCurve_Pt6_Y"] == pytest.approx(245.0, abs=1e-4)  # (255,245) direct

        # Missing channels should have 6-point identity defaults
        for prefix in ["ToneCurveRed", "ToneCurveGreen", "ToneCurveBlue"]:
            for n, (px, py) in enumerate(self._IDENTITY_6, start=1):
                assert result[f"{prefix}_Pt{n}_X"] == pytest.approx(float(px), abs=1e-4)
                assert result[f"{prefix}_Pt{n}_Y"] == pytest.approx(float(py), abs=1e-4)

    # --- _normalize_curve unit tests ---

    def test_normalize_curve_6_points_unchanged(self) -> None:
        from sonna_editor.data.xmp import _normalize_curve
        pts = [(0, 5), (51, 56), (102, 107), (153, 158), (204, 209), (255, 250)]
        assert _normalize_curve(pts) == pts

    def test_normalize_curve_4_points_interpolates(self) -> None:
        """4 < 6 → piecewise-linear interpolation at fixed x targets."""
        from sonna_editor.data.xmp import _normalize_curve
        pts = [(0, 0), (85, 95), (170, 180), (255, 255)]
        result = _normalize_curve(pts)
        assert result is not None
        assert len(result) == 6
        assert result[0] == (0, 0)
        assert result[5] == (255, 255)
        # x=51 is between (0,0) and (85,95): y = 0 + (51/85)*95 ≈ 57
        assert result[1][0] == 51
        assert result[1][1] == pytest.approx(57, abs=1)

    def test_normalize_curve_5_points_interpolates(self) -> None:
        """5 < 6 → piecewise-linear interpolation at fixed x targets."""
        from sonna_editor.data.xmp import _normalize_curve
        pts = [(0, 0), (64, 70), (128, 132), (192, 198), (255, 255)]
        result = _normalize_curve(pts)
        assert result is not None
        assert len(result) == 6
        assert result[0] == (0, 0)
        assert result[5] == (255, 255)

    def test_normalize_curve_7_points_downsamples(self) -> None:
        """7 > 6 → even-spaced index downsampling to 6 points."""
        from sonna_editor.data.xmp import _normalize_curve
        pts = [(0, 19), (42, 55), (84, 90), (128, 132), (170, 175), (212, 218), (255, 255)]
        result = _normalize_curve(pts)
        assert result is not None
        assert len(result) == 6
        # idxs = [round(i*6/5) for i in range(6)] = [0, 1, 2, 4, 5, 6]
        assert result[0] == pts[0]
        assert result[1] == pts[1]
        assert result[2] == pts[2]
        assert result[3] == pts[4]
        assert result[4] == pts[5]
        assert result[5] == pts[6]

    def test_normalize_curve_9_points_downsamples(self) -> None:
        """9 > 6 → even-spaced index downsampling."""
        from sonna_editor.data.xmp import _normalize_curve
        pts = [(i * 28, i * 28 + (2 if i > 0 else 0)) for i in range(9)]
        pts[-1] = (255, 255)
        result = _normalize_curve(pts)
        assert result is not None
        assert len(result) == 6
        # idxs = [round(i*8/5) for i in range(6)] = [0, 2, 3, 5, 6, 8]
        assert result[0] == pts[0]
        assert result[5] == pts[8]

    def test_normalize_curve_2_points_interpolates(self) -> None:
        from sonna_editor.data.xmp import _normalize_curve
        # Linear: y = x (identity endpoints)
        pts = [(0, 0), (255, 255)]
        result = _normalize_curve(pts)
        assert result is not None
        assert len(result) == 6
        assert result[0] == (0, 0)
        assert result[5] == (255, 255)
        # At x=51: y = 51 (linear)
        assert result[1][0] == 51
        assert result[1][1] == pytest.approx(51, abs=1)
        # At x=204: y = 204 (linear)
        assert result[4][0] == 204
        assert result[4][1] == pytest.approx(204, abs=1)

    def test_normalize_curve_3_points_interpolates(self) -> None:
        from sonna_editor.data.xmp import _normalize_curve
        # Simplified S-curve: (0,0), (128, 140), (255, 255)
        pts = [(0, 0), (128, 140), (255, 255)]
        result = _normalize_curve(pts)
        assert result is not None
        assert len(result) == 6
        assert result[0] == (0, 0)
        assert result[5] == (255, 255)
        # x=51 is between 0 and 128
        assert result[1][0] == 51

    def test_normalize_curve_empty_returns_none(self) -> None:
        from sonna_editor.data.xmp import _normalize_curve
        assert _normalize_curve([]) is None

    def test_normalize_curve_single_point_returns_none(self) -> None:
        from sonna_editor.data.xmp import _normalize_curve
        assert _normalize_curve([(128, 128)]) is None

    # --- Real XMP fixture has parseable curves ---

    def test_fixture_xmp_has_non_identity_composite_curve(self) -> None:
        """DP Event.xmp has a real S-curve on the composite channel (7 raw points → 6)."""
        from pathlib import Path
        from sonna_editor.data.xmp import read_xmp

        preset_path = Path("test_data/Preset/DP Event.xmp")
        if not preset_path.exists():
            pytest.skip("test_data/Preset/DP Event.xmp not present")
        result = read_xmp(preset_path)
        # Confirmed: Pt1_Y = 19 (lift at shadows, first of 7 raw points → stays as Pt1)
        assert result["ToneCurve_Pt1_Y"] == pytest.approx(19.0, abs=1e-4)
        # 7-point downsampled to 6: idxs=[0,1,2,4,5,6], last is pts[6]=(255,247)
        assert result["ToneCurve_Pt6_X"] == pytest.approx(255.0, abs=1e-4)
        assert result["ToneCurve_Pt6_Y"] == pytest.approx(247.0, abs=1e-4)

    # --- Full 48-field round-trip ---

    def test_all_48_tone_curve_fields_roundtrip(self, tmp_path) -> None:
        """Full round-trip for all 48 tone curve scalar fields."""
        from sonna_editor.config import SLIDER_FIELDS
        curve_fields = [f for f in SLIDER_FIELDS if f.startswith("ToneCurve")]
        assert len(curve_fields) == 48

        # Use distinct values per channel to catch any channel-mixing bugs
        base_pts = {
            "ToneCurve":       [(0, 5),  (51, 56),  (102, 107), (153, 158), (204, 209), (255, 250)],
            "ToneCurveRed":    [(0, 10), (51, 61),  (102, 112), (153, 163), (204, 214), (255, 245)],
            "ToneCurveGreen":  [(0, 0),  (51, 46),  (102, 97),  (153, 148), (204, 199), (255, 255)],
            "ToneCurveBlue":   [(0, 15), (51, 66),  (102, 117), (153, 168), (204, 219), (255, 240)],
        }
        settings: dict = {}
        for prefix, pts in base_pts.items():
            settings.update(self._curve_settings(prefix, pts))

        result = self._roundtrip(settings, tmp_path)
        for prefix, pts in base_pts.items():
            for n, (px, py) in enumerate(pts, start=1):
                assert result[f"{prefix}_Pt{n}_X"] == pytest.approx(float(px), abs=1e-4), (
                    f"{prefix}_Pt{n}_X"
                )
                assert result[f"{prefix}_Pt{n}_Y"] == pytest.approx(float(py), abs=1e-4), (
                    f"{prefix}_Pt{n}_Y"
                )


class TestExtraAttributes:
    """write_xmp(extra_attributes=...) support for postprocess rules (commit 57ab8bf)."""

    def test_write_xmp_with_extra_attributes_includes_them(self, tmp_path: Path) -> None:
        out = tmp_path / "with.xmp"
        write_xmp(out, {"Exposure2012": 0.5}, extra_attributes={
            "LensProfileEnable": "1",
            "AutoLateralCA": "1",
        })
        text = out.read_text()
        assert 'crs:LensProfileEnable="1"' in text
        assert 'crs:AutoLateralCA="1"' in text

    def test_write_xmp_without_extra_attributes_omits_them(self, tmp_path: Path) -> None:
        out = tmp_path / "without.xmp"
        write_xmp(out, {"Exposure2012": 0.5})  # no extra_attributes
        text = out.read_text()
        assert "LensProfileEnable" not in text
        assert "AutoLateralCA" not in text

    def test_write_xmp_extra_attributes_preserves_slider_fields(self, tmp_path: Path) -> None:
        out = tmp_path / "test.xmp"
        write_xmp(out, {"Exposure2012": 0.5}, extra_attributes={"LensProfileEnable": "1"})
        text = out.read_text()
        assert 'crs:Exposure2012="+0.5"' in text  # slider write still happens
        assert 'crs:LensProfileEnable="1"' in text

    def test_write_xmp_with_crop_angle_attributes(self, tmp_path: Path) -> None:
        out = tmp_path / "straightened.xmp"

        write_xmp(
            out,
            {"Exposure2012": 0.5},
            extra_attributes={
                "HasCrop": "True",
                "CropTop": "0",
                "CropLeft": "0",
                "CropBottom": "1",
                "CropRight": "1",
                "CropAngle": "-2.25",
                "CropConstrainToWarp": "0",
                "CropConstrainToUnitSquare": "1",
                "AlreadyApplied": "False",
            },
        )

        text = out.read_text()
        assert 'crs:HasCrop="True"' in text
        assert 'crs:CropTop="0"' in text
        assert 'crs:CropLeft="0"' in text
        assert 'crs:CropBottom="1"' in text
        assert 'crs:CropRight="1"' in text
        assert 'crs:CropAngle="-2.25"' in text
        assert 'crs:CropConstrainToWarp="0"' in text
        assert 'crs:CropConstrainToUnitSquare="1"' in text
        assert 'crs:AlreadyApplied="False"' in text
