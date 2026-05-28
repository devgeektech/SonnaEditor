"""v2 inference-pipeline constants tests (commit 57ab8bf)."""
from __future__ import annotations

from sonna_editor import config
from sonna_editor.inference.pipeline import ALWAYS_ON_POSTPROCESS, _V1_SKIP_FIELDS


def test_always_on_postprocess_constant() -> None:
    assert ALWAYS_ON_POSTPROCESS == {
        "LensProfileEnable": "1",
        "AutoLateralCA": "1",
    }


def test_v1_skip_fields_does_not_intersect_v2_extension() -> None:
    """_V1_SKIP_FIELDS must not accidentally drop any of the 12 v2 fields
    when inference writes XMPs from a v2 model. Verified programmatically
    to prevent silent regressions if either set grows."""
    v2_fields = set(config.SLIDER_FIELDS[135:])
    intersection = set(_V1_SKIP_FIELDS) & v2_fields
    assert intersection == set(), f"unexpected intersection: {intersection}"


# ──────────────────────────────────────────────────────────────────────
# WB skip substitution (Mode A): when Temperature/Tint are in skip_fields,
# write AsShot values explicitly rather than omit. Documented on
# _V1_SKIP_FIELDS; logic in _apply_wb_skip_substitution.
# ──────────────────────────────────────────────────────────────────────

from sonna_editor.inference.pipeline import _apply_wb_skip_substitution


def test_wb_substitution_writes_asshot_for_skipped_temperature() -> None:
    d = {"Temperature": 5500.0, "Tint": 10.0, "Exposure2012": 0.5}
    _apply_wb_skip_substitution(d, frozenset({"Temperature"}), as_shot_wb=(3500.0, 5.0))
    assert d["Temperature"] == 3500.0  # substituted from AsShot
    assert d["Tint"] == 10.0           # untouched (not in skip)
    assert d["Exposure2012"] == 0.5    # non-WB field untouched


def test_wb_substitution_writes_asshot_for_skipped_tint() -> None:
    d = {"Temperature": 5500.0, "Tint": 10.0}
    _apply_wb_skip_substitution(d, frozenset({"Tint"}), as_shot_wb=(3500.0, 5.0))
    assert d["Temperature"] == 5500.0  # untouched
    assert d["Tint"] == 5.0            # substituted


def test_wb_substitution_writes_asshot_for_both() -> None:
    """preserve_wb=True (or extra_skip_fields=[Temperature,Tint]) path."""
    d = {"Temperature": 5500.0, "Tint": 10.0}
    _apply_wb_skip_substitution(d, frozenset({"Temperature", "Tint"}), as_shot_wb=(3500.0, 5.0))
    assert d["Temperature"] == 3500.0
    assert d["Tint"] == 5.0


def test_wb_substitution_no_op_when_neither_skipped() -> None:
    d = {"Temperature": 5500.0, "Tint": 10.0}
    _apply_wb_skip_substitution(d, frozenset({"Saturation"}), as_shot_wb=(3500.0, 5.0))
    assert d["Temperature"] == 5500.0
    assert d["Tint"] == 10.0


def test_wb_substitution_no_op_when_as_shot_wb_missing() -> None:
    """When as_shot_wb extraction failed (None), the substitution silently
    leaves the dict alone — the caller's earlier filtering already removed
    skipped keys, so write_xmp will omit them. Edge case: as_shot_wb None."""
    d = {"Temperature": 5500.0}  # Tint already filtered out (skipped)
    _apply_wb_skip_substitution(d, frozenset({"Tint"}), as_shot_wb=None)
    assert "Tint" not in d
    assert d["Temperature"] == 5500.0


def test_wb_substitution_overwrites_model_prediction_when_already_in_dict() -> None:
    """If skipped Temperature somehow remained in the dict (e.g. caller
    didn't pre-filter), the substitution still replaces it with AsShot.
    Defensive — the actual pipeline.py path pre-filters, so this is just
    invariant-preservation."""
    d = {"Temperature": 5500.0}  # model's prediction
    _apply_wb_skip_substitution(d, frozenset({"Temperature"}), as_shot_wb=(3500.0, 5.0))
    assert d["Temperature"] == 3500.0  # AsShot wins


def test_wb_substitution_falls_back_to_omission_when_as_shot_wb_missing_end_to_end(
    tmp_path: "Path",
) -> None:
    """Full chain: filter + substitution(None) + write_xmp → crs:Tint absent.

    Documents the rare edge case where AsShot WB extraction failed (corrupt
    RAW / unsupported format). When the helper can't substitute, the
    previously-filtered Tint stays absent from the dict, and write_xmp omits
    the crs: attribute. This is the only path in Mode A where the LR
    partial-WB ambiguity still exists — fallback to the old "omit and
    hope LR does the right thing" behavior.
    """
    import re

    from pathlib import Path  # noqa: F401  (used by type annotation above)
    from sonna_editor.data.xmp import write_xmp

    # Simulate the per-photo state
    full_dict: dict[str, float] = {
        "Exposure2012": 0.5,
        "Tint": 8.0,           # model's collapsed prediction
        "Temperature": 6000.0,
    }
    effective_skip = frozenset({"Tint"})

    # Filter first (mirrors pipeline.py per-photo loop)
    filtered = {k: v for k, v in full_dict.items() if k not in effective_skip}
    assert "Tint" not in filtered  # filter removed it

    # Helper called with as_shot_wb=None (extraction failure)
    _apply_wb_skip_substitution(filtered, effective_skip, as_shot_wb=None)
    assert "Tint" not in filtered  # still absent — substitution couldn't recover

    # Write XMP and confirm crs:Tint is absent (fallback path)
    out = tmp_path / "fallback.xmp"
    write_xmp(out, filtered)
    text = out.read_text()
    assert re.search(r'crs:Tint="[^"]*"', text) is None, (
        "crs:Tint should be ABSENT in output XMP when as_shot_wb is None "
        "(fallback to omission)"
    )
    # Sanity: other model fields still written
    assert 'crs:Exposure2012="+0.5"' in text
    assert 'crs:Temperature="+6000"' in text
