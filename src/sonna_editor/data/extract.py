from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rawpy
from PIL import Image
from PIL.ExifTags import TAGS
import lxml.etree as etree

from sonna_editor.config import IMAGE_RESOLUTION, SCENE_STAT_FIELDS, SLIDER_FIELDS
from sonna_editor.data.xmp import compute_as_shot_wb, read_xmp

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

_WB_PRESETS = {
    0: "Auto",
    1: "Daylight",
    2: "Cloudy",
    3: "Tungsten",
    4: "Fluorescent",
    5: "Flash",
    6: "Custom",
    9: "Shade",
    10: "Kelvin",
}

_EXIF_NS = "http://ns.adobe.com/exif/1.0/"
_TIFF_NS = "http://ns.adobe.com/tiff/1.0/"
_AUX_NS = "http://ns.adobe.com/exif/1.0/aux/"
_EXIF_EX_NS = "http://cipa.jp/exif/1.0/"
_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_RAWPY_THUMB_JPEG: Any = getattr(getattr(rawpy, "ThumbFormat"), "JPEG")
_RAWPY_NO_THUMBNAIL_ERROR: type[BaseException] = getattr(
    rawpy, "LibRawNoThumbnailError", Exception
)


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _resize_to_long_edge(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= target:
        return img
    if w >= h:
        new_w, new_h = target, round(h * target / w)
    else:
        new_w, new_h = round(w * target / h), target
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _thumb_data_as_bytes(data: Any) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    return np.asarray(data, dtype=np.uint8).tobytes()


def _thumb_data_as_array(data: Any) -> np.ndarray:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return np.frombuffer(data, dtype=np.uint8)
    return np.asarray(data, dtype=np.uint8)


def _parse_exif_rational(value: str) -> float | None:
    """Parse '1/125' or '60/1' style rational strings to float."""
    try:
        if "/" in str(value):
            num, den = str(value).split("/")
            return float(num) / float(den)
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


def _read_xmp_metadata(xmp_path: Path) -> dict:
    """Extract camera/EXIF metadata from XMP namespaces (not CRS sliders)."""
    try:
        xml_bytes = xmp_path.read_bytes()
        import re
        xml_bytes = re.sub(rb"<\?xpacket[^?]*\?>", b"", xml_bytes).strip()
        root = etree.fromstring(xml_bytes)
    except Exception:
        return {}

    meta: dict = {}
    descriptions = root.findall(f".//{{{_RDF_NS}}}Description")

    for desc in descriptions:
        def g(ns: str, tag: str) -> str | None:
            return desc.get(f"{{{ns}}}{tag}")

        # Exposure time
        if meta.get("shutter_speed") is None:
            raw_et = g(_EXIF_NS, "ExposureTime")
            if raw_et:
                meta["shutter_speed"] = _parse_exif_rational(raw_et)

        # Aperture
        if meta.get("aperture") is None:
            raw_fn = g(_TIFF_NS, "FNumber") or g(_EXIF_NS, "FNumber")
            if raw_fn:
                meta["aperture"] = _parse_exif_rational(raw_fn)

        # ISO — may be in an rdf:Seq child
        if meta.get("iso") is None:
            iso_seq = desc.find(f"{{{_EXIF_NS}}}ISOSpeedRatings/{{{_RDF_NS}}}Seq/{{{_RDF_NS}}}li")
            if iso_seq is not None and iso_seq.text:
                try:
                    meta["iso"] = int(iso_seq.text)
                except ValueError:
                    pass

        # Focal length
        if meta.get("focal_length") is None:
            raw_fl = g(_EXIF_NS, "FocalLength")
            if raw_fl:
                meta["focal_length"] = _parse_exif_rational(raw_fl)

        # Lens model (prefer exifEX over aux)
        if meta.get("lens_model") is None:
            meta["lens_model"] = g(_EXIF_EX_NS, "LensModel") or g(_AUX_NS, "Lens")

        # Camera body
        if meta.get("camera_body") is None:
            make = g(_TIFF_NS, "Make")
            model = g(_TIFF_NS, "Model")
            if model:
                meta["camera_body"] = f"{make} {model}".strip() if make else model

        # Datetime
        if meta.get("capture_datetime") is None:
            raw_dt = g(_EXIF_NS, "DateTimeOriginal")
            if raw_dt:
                try:
                    meta["capture_datetime"] = datetime.fromisoformat(raw_dt)
                except ValueError:
                    pass

        # Exposure compensation
        if meta.get("exposure_compensation") is None:
            raw_ev = g(_EXIF_NS, "ExposureBiasValue")
            if raw_ev:
                meta["exposure_compensation"] = _parse_exif_rational(raw_ev)

        # White balance preset
        if meta.get("white_balance_preset") is None:
            raw_wb = g(_EXIF_NS, "WhiteBalance")
            if raw_wb is not None:
                try:
                    meta["white_balance_preset"] = _WB_PRESETS.get(int(raw_wb), "Unknown")
                except ValueError:
                    meta["white_balance_preset"] = raw_wb

        # Dimensions
        if meta.get("width") is None:
            w = g(_TIFF_NS, "ImageWidth")
            h = g(_TIFF_NS, "ImageLength")
            if w:
                meta["width"] = int(w)
            if h:
                meta["height"] = int(h)

    return meta


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def extract_preview(path: Path, target_size: int = IMAGE_RESOLUTION) -> Image.Image:
    """Return a resized RGB PIL image from the embedded preview in a RAW/DNG.

    Uses the embedded JPEG thumbnail (fast, no full RAW decode). Falls back to
    a half-size rawpy decode if no embedded JPEG exists.
    """
    with rawpy.imread(str(path)) as raw:
        try:
            thumb = raw.extract_thumb()
            if thumb.format == _RAWPY_THUMB_JPEG:
                img = Image.open(io.BytesIO(_thumb_data_as_bytes(thumb.data))).convert("RGB")
            else:
                # Bitmap thumb — wrap directly
                img = Image.fromarray(_thumb_data_as_array(thumb.data)).convert("RGB")
        except _RAWPY_NO_THUMBNAIL_ERROR:
            # Full RAW decode at half size as fallback
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=True,
                output_bps=8,
                no_auto_bright=True,
            )
            img = Image.fromarray(rgb).convert("RGB")

    return _resize_to_long_edge(img, target_size)


def extract_metadata(path: Path) -> dict:
    """Extract camera/shoot metadata from a RAW or DNG file.

    Reads from the embedded JPEG EXIF first, then supplements with a
    sidecar XMP if one exists next to path.
    """
    meta: dict = {
        "iso": None,
        "shutter_speed": None,
        "aperture": None,
        "focal_length": None,
        "lens_model": None,
        "camera_body": None,
        # v1.1.0 added separate make/model. camera_body remains the legacy
        # "{Make} {Model}" combined string for backward compat with v1.0.x.
        "make": None,
        "model": None,
        "capture_datetime": None,
        "exposure_compensation": None,
        "white_balance_preset": None,
        "camera_profile": None,
        "width": None,
        "height": None,
        # As-Shot Temperature/Tint computed inline from libraw's WB matrices.
        # The XMP writer (write_xmp) consumes this so it doesn't have to
        # reopen the RAW just to derive WB for the Pre-Saha snapshot —
        # avoids a duplicate ~200 ms rawpy.imread per photo on CR3/ARW.
        "as_shot_wb": None,
    }

    # --- Embedded EXIF from the RAW thumbnail ---
    try:
        with rawpy.imread(str(path)) as raw:
            # Dimensions from rawpy sizes (crop dimensions = actual image)
            sizes = raw.sizes
            meta["width"] = sizes.width
            meta["height"] = sizes.height

            # As-Shot WB computed inside the existing rawpy handle (no extra read).
            try:
                meta["as_shot_wb"] = compute_as_shot_wb(
                    raw.camera_whitebalance,
                    raw.rgb_xyz_matrix,
                )
            except Exception:
                meta["as_shot_wb"] = None

            thumb = raw.extract_thumb()
            if thumb.format != _RAWPY_THUMB_JPEG:
                raise ValueError("RAW thumbnail does not contain JPEG EXIF data")
            img = Image.open(io.BytesIO(_thumb_data_as_bytes(thumb.data)))
            exif = img.getexif()
            exif_ifd = exif.get_ifd(0x8769)  # ExifIFD

            # Top-level tags
            tag_map = {TAGS.get(k, k): v for k, v in exif.items()}
            make = tag_map.get("Make", "")
            model = tag_map.get("Model")
            if make:
                meta["make"] = str(make).strip()
            if model:
                meta["model"] = str(model).strip()
                meta["camera_body"] = f"{make} {model}".strip() if make else model

            raw_dt = tag_map.get("DateTime")
            if raw_dt and meta["capture_datetime"] is None:
                try:
                    meta["capture_datetime"] = datetime.strptime(raw_dt, "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    pass

            # ExifIFD tags
            exif_tag_map = {TAGS.get(k, k): v for k, v in exif_ifd.items()}
            if meta["iso"] is None and "ISOSpeedRatings" in exif_tag_map:
                meta["iso"] = int(exif_tag_map["ISOSpeedRatings"])
            if meta["shutter_speed"] is None and "ExposureTime" in exif_tag_map:
                meta["shutter_speed"] = float(exif_tag_map["ExposureTime"])
            if meta["aperture"] is None and "FNumber" in exif_tag_map:
                meta["aperture"] = float(exif_tag_map["FNumber"])
            if meta["focal_length"] is None and "FocalLength" in exif_tag_map:
                meta["focal_length"] = float(exif_tag_map["FocalLength"])
            if meta["exposure_compensation"] is None and "ExposureBiasValue" in exif_tag_map:
                meta["exposure_compensation"] = float(exif_tag_map["ExposureBiasValue"])
            if meta["white_balance_preset"] is None and "WhiteBalance" in exif_tag_map:
                meta["white_balance_preset"] = _WB_PRESETS.get(
                    int(exif_tag_map["WhiteBalance"]), "Unknown"
                )
    except Exception:
        pass

    # --- Supplement from XMP sidecar ---
    xmp_path = path.with_suffix(".xmp")
    if not xmp_path.exists():
        # Try uppercase extension
        xmp_path = path.with_suffix(".XMP")

    if xmp_path.exists():
        xmp_meta = _read_xmp_metadata(xmp_path)
        for key, val in xmp_meta.items():
            if meta.get(key) is None and val is not None:
                meta[key] = val

    return meta


def compute_histogram(image: Image.Image, bins: int = 32) -> np.ndarray:
    """Return a (3, bins) normalised RGB histogram array."""
    rgb = image.convert("RGB")
    hist = np.zeros((3, bins), dtype=np.float32)
    for ch in range(3):
        channel = np.array(rgb.getchannel(ch))
        counts, _ = np.histogram(channel, bins=bins, range=(0, 256))
        total = counts.sum()
        hist[ch] = counts / total if total > 0 else counts
    return hist


def compute_scene_statistics(image: Image.Image) -> dict[str, float]:
    """Return luminance statistics in stable 0-1 units.

    These features are intentionally simple and preview-derived. They give the
    regression heads direct access to exposure-relevant information such as
    clipping and dynamic range without requiring a full RAW decode.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luminance = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )
    p5 = float(np.percentile(luminance, 5))
    p95 = float(np.percentile(luminance, 95))
    stats = {
        "mean_luminance": float(luminance.mean()),
        "median_luminance": float(np.median(luminance)),
        "luminance_std": float(luminance.std()),
        "highlight_clip_pct": float((luminance >= 0.98).mean()),
        "shadow_clip_pct": float((luminance <= 0.02).mean()),
        "dynamic_range": max(0.0, min(1.0, p95 - p5)),
    }
    return {field: stats[field] for field in SCENE_STAT_FIELDS}


def extract_all(raw_path: Path, xmp_path: Path | None = None) -> dict:
    """Combine preview, metadata, histogram, and XMP slider values into one dict.

    Used to build training dataset rows.
    """
    if xmp_path is None:
        candidate = raw_path.with_suffix(".xmp")
        if not candidate.exists():
            candidate = raw_path.with_suffix(".XMP")
        xmp_path = candidate if candidate.exists() else None

    preview = extract_preview(raw_path)
    metadata = extract_metadata(raw_path)
    histogram = compute_histogram(preview)
    scene_stats = compute_scene_statistics(preview)

    sliders: dict = {field: None for field in SLIDER_FIELDS}
    if xmp_path is not None:
        sliders = read_xmp(xmp_path)

    return {
        "raw_path": str(raw_path),
        "xmp_path": str(xmp_path) if xmp_path else None,
        "preview": preview,
        "histogram": histogram,
        "scene_stats": scene_stats,
        **metadata,
        "sliders": sliders,
    }
