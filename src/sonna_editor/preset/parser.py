from __future__ import annotations

import logging
import re
from pathlib import Path

from lxml import etree

from sonna_editor.config import SLIDER_FIELDS
from sonna_editor.data.xmp import (
    _CURVE_CHANNELS,
    _normalize_curve,
    _parse_tone_curve_element,
)

logger = logging.getLogger(__name__)

# Namespace map for XMP parsing
_NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "crs": "http://ns.adobe.com/camera-raw-settings/1.0/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
}

# Fields that indicate local adjustments are present
_LOCAL_ADJUSTMENT_INDICATORS = {
    "RetouchAreas", "GradientBasedCorrections", "PaintBasedCorrections",
    "CircularGradientBasedCorrections",
}

# Sliders whose absolute value beyond this threshold is "extreme"
_EXTREME_THRESHOLDS: dict[str, float] = {
    "Exposure2012": 3.0,
    "Contrast2012": 80.0,
    "Highlights2012": 90.0,
    "Shadows2012": 90.0,
    "Whites2012": 90.0,
    "Blacks2012": 90.0,
    "Saturation": 70.0,
    "Vibrance": 70.0,
    "Clarity2012": 80.0,
    "Dehaze": 70.0,
    "Temperature": None,  # absolute value doesn't make sense for WB
    "Tint": 100.0,
}

# Regex for Lua scalar values: key = value,
_LUA_SCALAR_RE = re.compile(
    r'^\s*(\w+)\s*=\s*([+-]?\d+(?:\.\d+)?)\s*,?\s*$'
)
# Also match string values like WhiteBalance = "Auto",
_LUA_STRING_RE = re.compile(
    r'^\s*(\w+)\s*=\s*"([^"]*)"\s*,?\s*$'
)
# Detect local adjustment table keys
_LUA_LOCAL_ADJ_RE = re.compile(
    r'^\s*(RetouchAreas|GradientBasedCorrections|PaintBasedCorrections'
    r'|CircularGradientBasedCorrections)\s*='
)


# ---------------------------------------------------------------------------
# Format-specific parsers
# ---------------------------------------------------------------------------

def _parse_xmp_preset(path: Path) -> tuple[dict, set[str]]:
    """Parse a .xmp or .xmpsettings preset file.

    Returns (slider_values, extra_keys) where extra_keys is a set of
    non-slider crs: attribute names found (used for local-adjustment detection).
    """
    xml = path.read_bytes()
    # Strip xpacket wrappers if present
    xml = re.sub(rb'<\?xpacket[^?]*\?>', b'', xml).strip()
    root = etree.fromstring(xml)

    values: dict[str, float] = {}
    extra_keys: set[str] = set()

    crs_ns = _NS["crs"]
    rdf_ns = _NS["rdf"]

    # Attributes can live on rdf:Description directly or as child elements
    for desc in root.iter(f"{{{rdf_ns}}}Description"):
        for attr_name, attr_val in desc.attrib.items():
            if attr_name.startswith(f"{{{crs_ns}}}"):
                key = attr_name[len(f"{{{crs_ns}}}"):]
                _classify_crs_key(key, attr_val, values, extra_keys)

        for child in desc:
            if child.tag.startswith(f"{{{crs_ns}}}"):
                key = child.tag[len(f"{{{crs_ns}}}"):]
                text = (child.text or "").strip()
                _classify_crs_key(key, text, values, extra_keys)

    # Tone curves are rdf:Seq child elements, not plain attributes.
    # Parse them using the shared helper; absent curves stay out of `values`
    # so apply_adjustment will treat them as None (Lightroom keeps its default).
    descs = list(root.iter(f"{{{_NS['rdf']}}}Description"))
    for prefix, crs_tag in _CURVE_CHANNELS.items():
        raw_points: list[tuple[int, int]] = []
        for desc in descs:
            pts = _parse_tone_curve_element(desc, crs_tag)
            if pts:
                raw_points = pts
                break
        normalized = _normalize_curve(raw_points)
        if normalized is not None:
            for n, (px, py) in enumerate(normalized, start=1):
                values[f"{prefix}_Pt{n}_X"] = float(px)
                values[f"{prefix}_Pt{n}_Y"] = float(py)

    return values, extra_keys


def _classify_crs_key(
    key: str,
    raw_value: str,
    values: dict[str, float],
    extra_keys: set[str],
) -> None:
    if key in SLIDER_FIELDS:
        try:
            values[key] = float(raw_value)
        except (ValueError, TypeError):
            pass
    else:
        extra_keys.add(key)


def _parse_lrtemplate(path: Path) -> tuple[dict, bool]:
    """Parse a legacy .lrtemplate (Lua table) preset file.

    Returns (slider_values, has_local_adjustments).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    values: dict[str, float] = {}
    has_local = False

    for line in text.splitlines():
        if _LUA_LOCAL_ADJ_RE.match(line):
            has_local = True
            continue
        m = _LUA_SCALAR_RE.match(line)
        if m:
            key, raw = m.group(1), m.group(2)
            if key in SLIDER_FIELDS:
                try:
                    values[key] = float(raw)
                except ValueError:
                    pass

    return values, has_local


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_preset(path: Path) -> dict:
    """Parse a Lightroom preset file and return a slider-values dict.

    Supported formats: .xmp, .xmpsettings (XMP-based), .lrtemplate (Lua).
    Fields absent from the preset file are None (Lightroom will use its own default).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".xmp", ".xmpsettings"}:
        raw_values, _ = _parse_xmp_preset(path)
    elif suffix == ".lrtemplate":
        raw_values, _ = _parse_lrtemplate(path)
    else:
        raise ValueError(
            f"Unsupported preset format: '{suffix}'. "
            f"Expected .xmp, .xmpsettings, or .lrtemplate"
        )

    # Fields not present in the preset file are None (absent), not 0.0.
    # Fields like PerspectiveScale (range 50-150) or SharpenRadius (0.5-3.0)
    # must remain None so apply_adjustment skips clamping them to wrong values.
    result = {field: raw_values.get(field) for field in SLIDER_FIELDS}
    logger.debug("Parsed preset %s: %d explicit sliders", path.name,
                 sum(1 for v in result.values() if v is not None))
    return result


def validate_preset(preset: dict) -> list[str]:
    """Return a list of warnings about a parsed preset dict.

    Checks for extreme slider values. Local-adjustment detection is
    best-effort (the parsed dict loses that information; check the source
    file directly if needed).
    """
    warnings: list[str] = []

    for field, threshold in _EXTREME_THRESHOLDS.items():
        if threshold is None or field not in preset:
            continue
        val = preset.get(field)
        if val is None:
            continue
        if abs(val) > threshold:
            warnings.append(
                f"{field} value {val:.1f} is extreme (threshold ±{threshold:.0f}). "
                f"Double-check this is intentional."
            )

    # Flag HSL saturation kill (all channels at -100)
    hsl_sat = [f for f in SLIDER_FIELDS if f.startswith("SaturationAdjustment")]
    if all((preset.get(f) or 0.0) <= -95.0 for f in hsl_sat):
        warnings.append(
            "All HSL Saturation channels are at or near -100. "
            "This produces a near-monochrome result — intentional?"
        )

    return warnings
