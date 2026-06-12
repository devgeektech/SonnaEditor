from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Mapping

import lxml.etree as etree

from sonna_editor.config import SLIDER_FIELDS

_logger = logging.getLogger(__name__)

# XMP namespace URIs
_NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "crs": "http://ns.adobe.com/camera-raw-settings/1.0/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "tiff": "http://ns.adobe.com/tiff/1.0/",
    "exif": "http://ns.adobe.com/exif/1.0/",
}

_CRS_NS = _NS["crs"]
_RDF_NS = _NS["rdf"]

_PROCESS_VERSION = "15.4"

# Tone curve constants
_IDENTITY_CURVE_POINTS: list[tuple[int, int]] = [
    (0, 0), (51, 51), (102, 102), (153, 153), (204, 204), (255, 255)
]
_N_CURVE_PTS = 6
_CURVE_X_TARGETS = [0, 51, 102, 153, 204, 255]  # 255 / 5 = 51 exactly

# Maps Python field prefix → XMP element name (crs: attribute)
_CURVE_CHANNELS: dict[str, str] = {
    "ToneCurve":      "ToneCurvePV2012",
    "ToneCurveRed":   "ToneCurvePV2012Red",
    "ToneCurveGreen": "ToneCurvePV2012Green",
    "ToneCurveBlue":  "ToneCurvePV2012Blue",
}

_TONE_CURVE_FIELDS: frozenset[str] = frozenset(
    f for f in SLIDER_FIELDS if f.startswith("ToneCurve")
)

# Lightroom RAW-import defaults — what every slider shows on a freshly imported,
# untouched RAW. Used to build the "Pre-Saha" snapshot so clicking it in LR's
# Snapshots panel reverts the photo to its pre-edit visual state.
#
# Most adjustment sliders default to 0. Non-zero defaults are LR's own
# documented baselines for sharpening, noise reduction, parametric tone curve
# split points, vignette, grain, and colour grading blending.
LR_DEFAULTS: dict[str, float] = {
    # Tone (8) — all 0
    "Exposure2012": 0.0, "Contrast2012": 0.0,
    "Highlights2012": 0.0, "Shadows2012": 0.0,
    "Whites2012": 0.0, "Blacks2012": 0.0,
    "Clarity2012": 0.0, "Dehaze": 0.0,
    # Presence (3) — all 0
    "Texture": 0.0, "Vibrance": 0.0, "Saturation": 0.0,
    # WB (2) — Temperature/Tint default to 5500/0 but are normally overridden
    # at write time by the camera's As-Shot values extracted from the RAW.
    "Temperature": 5500.0, "Tint": 0.0,
    # HSL (24) — all 0
    **{f"HueAdjustment{c}": 0.0 for c in
       ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")},
    **{f"SaturationAdjustment{c}": 0.0 for c in
       ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")},
    **{f"LuminanceAdjustment{c}": 0.0 for c in
       ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")},
    # Parametric tone curve (7)
    "ParametricHighlights": 0.0, "ParametricLights": 0.0,
    "ParametricDarks": 0.0, "ParametricShadows": 0.0,
    "ParametricHighlightSplit": 75.0,
    "ParametricMidtoneSplit": 50.0,
    "ParametricShadowSplit": 25.0,
    # Color Grading (14) — Blending defaults to 50, everything else 0
    "SplitToningShadowHue": 0.0, "SplitToningShadowSaturation": 0.0,
    "ColorGradeShadowLum": 0.0,
    "ColorGradeMidtoneHue": 0.0, "ColorGradeMidtoneSat": 0.0, "ColorGradeMidtoneLum": 0.0,
    "SplitToningHighlightHue": 0.0, "SplitToningHighlightSaturation": 0.0,
    "ColorGradeHighlightLum": 0.0,
    "ColorGradeBlending": 50.0,
    "ColorGradeGlobalHue": 0.0, "ColorGradeGlobalSat": 0.0, "ColorGradeGlobalLum": 0.0,
    "SplitToningBalance": 0.0,
    # Camera Calibration (6) — all 0
    "RedHue": 0.0, "RedSaturation": 0.0,
    "GreenHue": 0.0, "GreenSaturation": 0.0,
    "BlueHue": 0.0, "BlueSaturation": 0.0,
    # Sharpening (4)
    "Sharpness": 25.0,         # LR's RAW import default — not 0
    "SharpenRadius": 1.0,
    "SharpenDetail": 25.0,
    "SharpenEdgeMasking": 0.0,
    # Noise reduction (4)
    "LuminanceSmoothing": 0.0,
    "LuminanceNoiseReductionDetail": 50.0,
    "LuminanceNoiseReductionContrast": 0.0,
    "ColorNoiseReduction": 25.0,
    # Effects (8) — vignette + grain
    "PostCropVignetteAmount": 0.0,
    "PostCropVignetteMidpoint": 50.0,
    "PostCropVignetteRoundness": 0.0,
    "PostCropVignetteFeather": 50.0,
    "PostCropVignetteHighlightContrast": 0.0,
    "GrainAmount": 0.0,
    "GrainSize": 25.0,
    "GrainFrequency": 50.0,
    # Lens (2) — 0
    "LensManualDistortionAmount": 0.0,
    "VignetteAmount": 0.0,
    # Transform (5) — 0
    "PerspectiveVertical": 0.0, "PerspectiveHorizontal": 0.0,
    "PerspectiveRotate": 0.0,
    "PerspectiveScale": 100.0,   # 100 = no scale
    "PerspectiveAspect": 0.0,
    # === v2 extensions (idx 135-146) — added 2026-05-13 ===
    # Values match config.SLIDER_DEFAULTS exactly. These two tables serve
    # different purposes (LR_DEFAULTS for snapshots/XMP-read defaults;
    # SLIDER_DEFAULTS for migration backfill) and the duplication is a known
    # smell flagged for future consolidation.
    "ColorNoiseReductionDetail": 50.0,
    "ColorNoiseReductionSmoothness": 50.0,
    "DefringePurpleAmount": 0.0,
    "DefringePurpleHueLo": 30.0,
    "DefringePurpleHueHi": 70.0,
    "DefringeGreenAmount": 0.0,
    "DefringeGreenHueLo": 40.0,
    "DefringeGreenHueHi": 60.0,
    "LensProfileDistortionScale": 100.0,
    "LensProfileVignettingScale": 100.0,
    "ShadowTint": 0.0,
    "CurveRefineSaturation": 100.0,
    # Tone curves (48) — identity at 6 evenly-spaced control points
    **{
        f"ToneCurve{ch}_Pt{n}_{ax}": float(v)
        for ch in ("", "Red", "Green", "Blue")
        for n, v in enumerate([0, 51, 102, 153, 204, 255], start=1)
        for ax in ("X", "Y")
    },
}

# Sanity check at import time — if a slider is added to SLIDER_FIELDS without
# a matching default, the snapshot would silently misrepresent it.
_missing_defaults = set(SLIDER_FIELDS) - set(LR_DEFAULTS)
assert not _missing_defaults, f"LR_DEFAULTS missing: {_missing_defaults}"

# Matches XMP packet in binary files (DNG / embedded)
_XMP_PACKET_RE = re.compile(
    rb"<x:xmpmeta[\s\S]*?</x:xmpmeta>",
    re.DOTALL,
)


# ── As-Shot Temperature/Tint extraction from RAW WB metadata ───────────────

def compute_as_shot_wb(cam_wb, rgb_xyz_matrix) -> tuple[float, float] | None:
    """Pure-math version of As-Shot Temperature/Tint computation.

    Inputs come from a libraw/rawpy decode that the caller already performed:
      cam_wb         — raw.camera_whitebalance (RGGB multipliers)
      rgb_xyz_matrix — raw.rgb_xyz_matrix (XYZ → camera, Adobe ColorMatrix2)

    Splitting this from the I/O wrapper lets the extract phase compute the
    value during its existing rawpy.imread call (no second read), and the
    XMP writer just consume the cached result.

    Pipeline → CIE colour science:
      1. Invert cam_wb → AsShotNeutral (scene white in camera RGB).
      2. Multiply by inv(rgb_xyz_matrix) → CIE XYZ. (libraw's matrix is
         XYZ→camera despite the name; verified empirically — M @ [1,1,1]
         returns values far from D65 white, while inv(M) gives sensible
         chromaticities.)
      3. Normalise to xy chromaticity.
      4. McCamy 1992 cubic → correlated colour temperature (CCT).
      5. Krystek 1985 Planckian locus in CIE 1960 uv → signed perpendicular
         distance to the locus → LR's Tint axis (positive=magenta).

    Refs: McCamy, "Correlated color temperature as an explicit function of
    chromaticity coordinates", Color Research & Application 17(2), 1992.
    Krystek, "An algorithm to calculate correlated color temperature",
    Color Research & Application 10(1), 1985.

    Returns (kelvin, tint) clamped to LR valid ranges (2000–50000K, ±150),
    or None if the inputs are missing/degenerate.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    try:
        cam_wb = list(cam_wb)
    except TypeError:
        return None
    if len(cam_wb) < 3 or any(v <= 0 for v in cam_wb[:3]):
        return None
    if rgb_xyz_matrix is None:
        return None
    rgb_xyz = np.asarray(rgb_xyz_matrix)
    if rgb_xyz.ndim != 2 or rgb_xyz.shape[1] != 3:
        return None

    # AsShotNeutral: scene white in camera RGB. Camera applied (multipliers)
    # to push the captured white to neutral, so the scene white is the inverse,
    # normalised so green = 1.
    g = cam_wb[1]
    neutral = np.array([g / cam_wb[0], 1.0, g / cam_wb[2]])

    M_xyz_to_cam = rgb_xyz[:3, :3]
    try:
        M_cam_to_xyz = np.linalg.inv(M_xyz_to_cam)
    except np.linalg.LinAlgError:
        return None
    XYZ = M_cam_to_xyz @ neutral
    s = float(XYZ.sum())
    if s <= 0:
        return None
    x = float(XYZ[0]) / s
    y = float(XYZ[1]) / s

    # McCamy CCT
    n = (x - 0.3320) / (0.1858 - y)
    cct = 449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33

    # CIE 1960 uv
    denom = -2.0 * x + 12.0 * y + 3.0
    if denom == 0:
        return None
    u = 4.0 * x / denom
    v = 6.0 * y / denom

    # Planckian locus uv (Krystek)
    T = max(1000.0, min(50000.0, cct))
    u_p = (0.860117757 + 1.54118254e-4 * T + 1.28641212e-7 * T * T) / \
          (1.0 + 8.42420235e-4 * T + 7.08145163e-7 * T * T)
    v_p = (0.317398726 + 4.22806245e-5 * T + 4.20481691e-8 * T * T) / \
          (1.0 - 2.89741816e-5 * T + 1.61456053e-7 * T * T)

    duv = ((u - u_p) ** 2 + (v - v_p) ** 2) ** 0.5
    if v < v_p:  # below the locus = greener scene → negative LR tint
        duv = -duv
    # Empirical: LR's Tint range ±150 corresponds to roughly ±0.05 in uv duv.
    tint = duv * 3000.0

    cct = max(2000.0, min(50000.0, cct))
    tint = max(-150.0, min(150.0, tint))
    return (round(cct), round(tint, 1))


def _extract_as_shot_wb(raw_path: Path) -> tuple[float, float] | None:
    """Open the RAW with rawpy and extract its As-Shot Temperature/Tint.

    Fallback path used by write_xmp when the caller didn't pre-compute the
    value via compute_as_shot_wb during the extract phase. Reading the RAW
    header is the expensive part (~200 ms on CR3); the inference pipeline
    avoids this by passing as_shot_wb=meta["as_shot_wb"] through write_xmp.
    """
    try:
        import rawpy
    except ImportError:
        return None
    try:
        with rawpy.imread(str(raw_path)) as raw:
            cam_wb = list(raw.camera_whitebalance)
            rgb_xyz = raw.rgb_xyz_matrix.copy()
    except Exception as e:  # noqa: BLE001 — any libraw failure
        _logger.debug("rawpy failed on %s: %s", raw_path.name, e)
        return None
    return compute_as_shot_wb(cam_wb, rgb_xyz)


# ── "Pre-Saha" snapshot construction ───────────────────────────────────────

def _format_snapshot_value(field: str, value: float) -> str:
    """Format a slider value the way Lightroom writes its snapshot Parameters.

    Integer values get no decimal. Floats keep up to 6 decimals to match LR's
    own precision when it serialises a snapshot with non-integer values.
    """
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def _build_snapshot_parameters(
    overrides: dict[str, float] | None = None,
) -> str:
    """Build the multi-line crs:Parameters string for the Pre-Saha snapshot.

    Includes every scalar field in SLIDER_FIELDS at its LR_DEFAULTS value plus
    flattened tone curves ('ToneCurvePV2012 = 0, 0, 51, 51, ...'). Caller can
    pass overrides — typically {"Temperature": ..., "Tint": ...} from
    _extract_as_shot_wb — so the snapshot reverts to the actual As-Shot WB
    rather than the 5500/0 placeholder.
    """
    overrides = overrides or {}
    lines: list[str] = ["ProcessVersion = 15.4"]

    # Scalar fields (everything except ToneCurve_*).
    for field in SLIDER_FIELDS:
        if field.startswith("ToneCurve"):
            continue
        value = overrides.get(field, LR_DEFAULTS[field])
        lines.append(f"{field} = {_format_snapshot_value(field, value)}")

    # Tone curves: each channel as one line of comma-separated x,y pairs.
    for prefix, crs_tag in _CURVE_CHANNELS.items():
        pairs: list[str] = []
        for n in range(1, 7):
            x = LR_DEFAULTS[f"{prefix}_Pt{n}_X"]
            y = LR_DEFAULTS[f"{prefix}_Pt{n}_Y"]
            pairs.append(f"{int(x)}, {int(y)}")
        lines.append(f"{crs_tag} = {', '.join(pairs)}")

    return "\n".join(lines)


def _append_pre_saha_snapshot(
    description: etree._Element,
    nsmap: dict[str, str],
    parameters_string: str,
) -> None:
    """Append the Pre-Saha crs:Snapshots block as a child of the rdf:Description.

    Format:
      <crs:Snapshots>
        <rdf:Bag>
          <rdf:li>
            <rdf:Description
              crs:Name="Pre-Saha"
              crs:UUID="..."
              crs:Type="Develop"
              crs:Parameters="..." />
          </rdf:li>
        </rdf:Bag>
      </crs:Snapshots>
    """
    snapshots = etree.SubElement(description, f"{{{_CRS_NS}}}Snapshots")
    bag = etree.SubElement(snapshots, f"{{{_RDF_NS}}}Bag")
    li = etree.SubElement(bag, f"{{{_RDF_NS}}}li")
    snap_desc = etree.SubElement(li, f"{{{_RDF_NS}}}Description", nsmap=nsmap)
    snap_desc.set(f"{{{_CRS_NS}}}Name", "Pre-Saha")
    snap_desc.set(f"{{{_CRS_NS}}}UUID", str(uuid.uuid4()).upper().replace("-", ""))
    snap_desc.set(f"{{{_CRS_NS}}}Type", "Develop")
    snap_desc.set(f"{{{_CRS_NS}}}Parameters", parameters_string)


def _extract_xmp_bytes_from_binary(path: Path) -> bytes | None:
    """Pull the XMP block out of a DNG or other binary container."""
    data = path.read_bytes()
    m = _XMP_PACKET_RE.search(data)
    if m:
        return m.group(0)
    # Older files may use just <rdf:RDF …> without x:xmpmeta wrapper
    alt = re.search(rb"<rdf:RDF[\s\S]*?</rdf:RDF>", data, re.DOTALL)
    return alt.group(0) if alt else None


def _interp1d(xs: list[float], ys: list[float], tx: float) -> float:
    """Piecewise-linear interpolation at a single query point."""
    if tx <= xs[0]:
        return ys[0]
    if tx >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= tx <= xs[i + 1]:
            t = (tx - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def _normalize_curve(
    points: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    """Normalize a variable-length tone curve to exactly 6 control points.

    Returns None if points is empty (missing curve → identity defaults used by caller).
    - n == 6: return as-is
    - n > 6: even-spaced index downsampling
    - 1 < n < 6: piecewise-linear interpolation at x = 0, 51, 102, 153, 204, 255
    - n <= 1: return None
    """
    if len(points) <= 1:
        return None
    n = len(points)
    if n == _N_CURVE_PTS:
        return list(points)
    if n > _N_CURVE_PTS:
        idxs = [round(i * (n - 1) / (_N_CURVE_PTS - 1)) for i in range(_N_CURVE_PTS)]
        return [points[i] for i in idxs]
    # n in {2, 3, 4, 5}: interpolate at fixed x targets
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [(tx, int(round(_interp1d(xs, ys, float(tx))))) for tx in _CURVE_X_TARGETS]


def _parse_tone_curve_element(
    desc_elem: etree._Element,
    crs_tag: str,
) -> list[tuple[int, int]]:
    """Extract tone curve control points from an rdf:Seq child element.

    Returns a list of (x, y) int tuples, or [] if the element is absent or corrupt.
    """
    el = desc_elem.find(f"{{{_CRS_NS}}}{crs_tag}")
    if el is None:
        return []
    seq = el.find(f"{{{_RDF_NS}}}Seq")
    if seq is None:
        return []
    points: list[tuple[int, int]] = []
    for li in seq.findall(f"{{{_RDF_NS}}}li"):
        text = (li.text or "").strip()
        try:
            parts = [p.strip() for p in text.split(",")]
            x, y = int(parts[0]), int(parts[1])
            points.append((x, y))
        except (ValueError, IndexError):
            import logging
            logging.getLogger(__name__).warning(
                "Skipping malformed tone curve point %r in crs:%s", text, crs_tag
            )
    return points


def _parse_xmp_bytes(xml_bytes: bytes) -> dict[str, float | str | None]:
    """Parse an XMP XML byte string and return slider field values.

    Returns a dict keyed by slider field name. Values are floats for numeric
    fields, strings for non-numeric, None if the field is absent.

    Note: tone curve fields (ToneCurve_Pt*_X/Y) always return float — either
    the parsed curve or identity defaults. They never return None, unlike scalar
    slider fields which return None when absent.
    """
    # Strip XMP packet wrapper if present (lxml can choke on the PI)
    xml_bytes = re.sub(rb"<\?xpacket[^?]*\?>", b"", xml_bytes).strip()

    root = etree.fromstring(xml_bytes)

    # Find rdf:Description element(s)
    descriptions = root.findall(f".//{{{_RDF_NS}}}Description")
    if not descriptions:
        return {field: None for field in SLIDER_FIELDS}

    # Merge attributes from all Description elements (LR sometimes splits them)
    raw: dict[str, str | None] = {field: None for field in SLIDER_FIELDS}
    for desc in descriptions:
        for field in SLIDER_FIELDS:
            if field in _TONE_CURVE_FIELDS:
                continue  # tone curves are child elements, not attributes
            key = f"{{{_CRS_NS}}}{field}"
            val = desc.get(key)
            if val is not None:
                raw[field] = val

    # Convert numeric strings to float; leave non-numeric as str
    result: dict[str, float | str | None] = {}
    for field, val in raw.items():
        if field in _TONE_CURVE_FIELDS:
            continue  # filled below
        if val is None:
            result[field] = None
            continue
        try:
            result[field] = float(val)
        except ValueError:
            result[field] = val

    # Parse tone curve child elements from the first Description that has them
    for prefix, crs_tag in _CURVE_CHANNELS.items():
        raw_points: list[tuple[int, int]] = []
        for desc in descriptions:
            pts = _parse_tone_curve_element(desc, crs_tag)
            if pts:
                raw_points = pts
                break
        normalized = _normalize_curve(raw_points) or list(_IDENTITY_CURVE_POINTS)
        for n, (px, py) in enumerate(normalized, start=1):
            result[f"{prefix}_Pt{n}_X"] = float(px)
            result[f"{prefix}_Pt{n}_Y"] = float(py)

    return result


def read_xmp(path: Path) -> dict[str, float | str | None]:
    """Read Lightroom develop settings from an XMP sidecar or DNG.

    Returns a dict keyed by slider field name. Values are floats for numeric
    fields, strings for non-numeric, None if the field is absent.
    """
    suffix = path.suffix.lower()

    if suffix == ".xmp":
        xml_bytes = path.read_bytes()
    else:
        # DNG or other RAW container — extract embedded XMP
        xml_bytes = _extract_xmp_bytes_from_binary(path)
        if xml_bytes is None:
            return {field: None for field in SLIDER_FIELDS}

    return _parse_xmp_bytes(xml_bytes)


def write_xmp(
    path: Path,
    settings: Mapping[str, float | str | None],
    source_raw_path: Path | None = None,
    as_shot_wb: tuple[float, float] | None = None,
    extra_attributes: Mapping[str, str] | None = None,
) -> None:
    """Write a Lightroom-compatible XMP sidecar to path.

    Scalar slider fields: written only when their value is not None.
    Tone curve fields: all 4 channels (composite, red, green, blue) are always
    written as rdf:Seq child elements. Missing Pt values default to the identity
    curve so Lightroom always gets a valid curve structure.

    as_shot_wb: pre-computed (Temperature, Tint) tuple from a rawpy decode the
        caller already did (e.g. data.extract.extract_metadata's meta["as_shot_wb"]).
        When provided, the Pre-Saha snapshot uses these values directly. When
        None and source_raw_path is set, write_xmp opens the RAW itself via
        _extract_as_shot_wb — slow (~200 ms/CR3) but preserves backwards
        compatibility for callers that don't pre-compute.

    extra_attributes: optional dict of crs: attributes to write verbatim,
        outside the SLIDER_FIELDS loop. Used for postprocess rules that are
        not model predictions (e.g., always-on LensProfileEnable="1",
        AutoLateralCA="1" from inference/pipeline.py).
        extra_attributes values are written via str() — caller responsible
        for Lightroom-compatible formatting.
    """
    # Build namespace map for the rdf:Description element
    nsmap = {
        "x": _NS["x"],
        "rdf": _RDF_NS,
        "crs": _CRS_NS,
        "xmp": _NS["xmp"],
        "dc": _NS["dc"],
        "tiff": _NS["tiff"],
        "exif": _NS["exif"],
    }

    xmpmeta = etree.Element(f"{{{_NS['x']}}}xmpmeta", nsmap={"x": _NS["x"]})
    xmpmeta.set(f"{{{_NS['x']}}}xmptk", "Adobe XMP Core 7.0-c000 1.000000, 0000/00/00-00:00:00        ")

    rdf_rdf = etree.SubElement(xmpmeta, f"{{{_RDF_NS}}}RDF")

    description = etree.SubElement(rdf_rdf, f"{{{_RDF_NS}}}Description", nsmap=nsmap)
    description.set(f"{{{_RDF_NS}}}about", "")

    # Mandatory crs attributes
    description.set(f"{{{_CRS_NS}}}ProcessVersion", _PROCESS_VERSION)
    description.set(f"{{{_CRS_NS}}}HasSettings", "True")

    if source_raw_path is not None:
        description.set(f"{{{_CRS_NS}}}RawFileName", source_raw_path.name)

    # Write tone curve child elements (rdf:Seq) for all 4 channels
    for prefix, crs_tag in _CURVE_CHANNELS.items():
        points: list[tuple[int, int]] = []
        for n in range(1, 7):
            x_val = settings.get(f"{prefix}_Pt{n}_X")
            y_val = settings.get(f"{prefix}_Pt{n}_Y")
            if x_val is None or y_val is None:
                points = []
                break
            points.append((int(round(float(x_val))), int(round(float(y_val)))))
        if not points:
            points = list(_IDENTITY_CURVE_POINTS)
        curve_el = etree.SubElement(description, f"{{{_CRS_NS}}}{crs_tag}")
        seq_el = etree.SubElement(curve_el, f"{{{_RDF_NS}}}Seq")
        for px, py in points:
            li_el = etree.SubElement(seq_el, f"{{{_RDF_NS}}}li")
            li_el.text = f"{px}, {py}"

    # Write scalar slider values (tone curve fields handled above as child elements)
    for field in SLIDER_FIELDS:
        if field in _TONE_CURVE_FIELDS:
            continue
        val = settings.get(field)
        if val is None:
            continue
        # Mode B (preset pipeline) sentinel: Temperature=0 from a preset
        # means "preset doesn't specify white balance" — a convention
        # established in preset/parser.py + preset/adjuster.py. Omit the
        # attribute so Lightroom falls back to the camera's AsShot WB.
        #
        # Mode A (inference pipeline) does NOT pass through this path. When
        # the user skips Temperature via skip_fields, inference/pipeline.py
        # substitutes the AsShot value explicitly (see
        # _apply_wb_skip_substitution) — no Temperature=0 sentinel involved.
        #
        # Intentional asymmetry: two modes use two different mechanisms for
        # the "use AsShot instead of a written value" semantic. Unification
        # is deferred — see HANDOVER Part 6: "WB skip semantics unification".
        if field == "Temperature" and val == 0:
            continue
        if isinstance(val, float):
            # Match Lightroom's formatting: integer floats as "+N" or "N", decimals as-is
            if val == int(val):
                formatted = f"+{int(val)}" if val > 0 else str(int(val))
            else:
                formatted = f"+{val}" if val > 0 else str(val)
        else:
            formatted = str(val)
        description.set(f"{{{_CRS_NS}}}{field}", formatted)

    # Write extra non-slider crs: attributes (postprocess rules from caller).
    # These are NOT in SLIDER_FIELDS by design — they're hardcoded toggles
    # like LensProfileEnable / AutoLateralCA that the model doesn't predict.
    if extra_attributes is not None:
        for field, val in extra_attributes.items():
            description.set(f"{{{_CRS_NS}}}{field}", str(val))

    # Pre-Saha snapshot — captures the photo's identity state so the user can
    # toggle before/after in Lightroom's Snapshots panel. Temperature/Tint
    # come from the source RAW's As-Shot WB when extractable; otherwise they
    # fall back to the LR_DEFAULTS placeholders (5500/0) for that file only.
    overrides: dict[str, float] = {}
    wb = as_shot_wb
    if wb is None and source_raw_path is not None:
        wb = _extract_as_shot_wb(source_raw_path)
    if wb is not None:
        overrides["Temperature"] = wb[0]
        overrides["Tint"] = wb[1]
    elif source_raw_path is not None:
        _logger.info(
            "As-Shot WB extraction failed for %s; Pre-Saha snapshot will "
            "use 5500/0 placeholder", source_raw_path.name,
        )
    parameters_string = _build_snapshot_parameters(overrides)
    _append_pre_saha_snapshot(description, nsmap, parameters_string)

    xml_bytes = etree.tostring(
        xmpmeta,
        pretty_print=True,
        xml_declaration=False,
        encoding="unicode",
    ).encode("utf-8")

    # Wrap in XMP packet markers — no separate xml declaration inside the packet
    packet = (
        b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        + xml_bytes
        + b'<?xpacket end="w"?>\n'
    )

    path.write_bytes(packet)
