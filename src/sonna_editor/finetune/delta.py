"""Delta analysis and fine-tune dataset preparation for the continuous learning loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from sonna_editor import config
from sonna_editor.finetune.capture import _SOURCE_MODEL_FILTERED, _SOURCE_USER_FINAL

# Correlation reporting thresholds
_MIN_SPEARMAN_ABS_R = 0.3
_MAX_P_VALUE = 0.05
_MIN_N_FOR_CORRELATION = 10

# Numeric metadata columns used for correlation analysis
_NUMERIC_META_COLS = ["iso", "shutter_speed", "aperture", "focal_length"]


def _parse_deltas(deltas_json: str) -> dict[str, dict]:
    """Deserialise a deltas JSON string to a field → {delta, source} dict."""
    try:
        return json.loads(deltas_json)
    except (json.JSONDecodeError, TypeError):
        return {}


def analyse_deltas(captures: pd.DataFrame) -> dict:
    """
    Analyse per-field deltas across a set of captured edits.

    Only considers fields where source == "user_final" for user-tweak statistics.
    Fields with source == "model_filtered" are reported separately under
    "filtered_field_deltas" as v2 intelligence data.

    Args:
        captures: DataFrame from capture_user_edits().

    Returns:
        Analysis dict — see module docstring for full schema.
    """
    n_photos = len(captures)
    if n_photos == 0:
        return {
            "n_photos": 0,
            "metadata_coverage": {},
            "per_field": {},
            "most_adjusted_fields": [],
            "correlations": [],
            "filtered_field_deltas": {},
        }

    # --- Parse all deltas rows ---
    all_deltas: list[dict[str, dict]] = [
        _parse_deltas(row) for row in captures["deltas"]
    ]

    # Derive the canonical field list from the captures themselves: union of
    # keys across all per-photo delta dicts, ordered by config.SLIDER_FIELDS
    # for deterministic output. After Commits 2a-2b, each per-photo delta
    # dict's keys reflect the model that produced its predictions (135 for
    # v1, 147 for v2). Iterating config.SLIDER_FIELDS unconditionally was
    # the bug: it produced 12 always-empty v2-extension entries when input
    # captures were v1-shaped, polluting per_field/correlations/filtered_*
    # outputs.
    _present_fields = set().union(*(d.keys() for d in all_deltas)) if all_deltas else set()
    delta_fields: list[str] = [f for f in config.SLIDER_FIELDS if f in _present_fields]

    # --- Metadata coverage ---
    metadata_coverage: dict[str, float] = {}
    for col in _NUMERIC_META_COLS:
        if col in captures.columns:
            n_non_null = captures[col].notna().sum()
            metadata_coverage[col] = round(n_non_null / n_photos, 3)

    # --- Per-field stats (user_final only) ---
    per_field: dict[str, dict] = {}
    # Collect delta values per field for correlation analysis
    field_delta_arrays: dict[str, list[float]] = {f: [] for f in delta_fields}

    for field in delta_fields:
        user_deltas: list[float] = []
        for row_deltas in all_deltas:
            entry = row_deltas.get(field, {})
            if entry.get("source") == _SOURCE_USER_FINAL and entry.get("delta") is not None:
                user_deltas.append(float(entry["delta"]))
                field_delta_arrays[field].append(float(entry["delta"]))

        if not user_deltas:
            continue

        arr = np.array(user_deltas, dtype=float)
        per_field[field] = {
            "n_with_delta": len(arr),
            "mean_delta": float(np.mean(arr)),
            "abs_mean_delta": float(np.mean(np.abs(arr))),
            "std_delta": float(np.std(arr)),
            "min_delta": float(np.min(arr)),
            "max_delta": float(np.max(arr)),
        }

    # --- Most adjusted fields (sorted by abs_mean_delta) ---
    most_adjusted_fields = sorted(
        [(f, stats_dict["abs_mean_delta"]) for f, stats_dict in per_field.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    # --- Spearman correlations (all metadata × all slider deltas) ---
    correlations: list[dict] = []
    for meta_col in _NUMERIC_META_COLS:
        if meta_col not in captures.columns:
            continue
        meta_vals = captures[meta_col].values

        for field in delta_fields:
            delta_list = field_delta_arrays[field]
            if len(delta_list) < _MIN_N_FOR_CORRELATION:
                continue

            # Align: only rows where this field had a user_final delta
            # We need to match delta_list back to the row index.
            # Re-extract with row alignment for the correlation.
            aligned_meta: list[float] = []
            aligned_delta: list[float] = []
            for i, row_deltas in enumerate(all_deltas):
                entry = row_deltas.get(field, {})
                if entry.get("source") == _SOURCE_USER_FINAL and entry.get("delta") is not None:
                    meta_val = meta_vals[i]
                    if meta_val is not None and not (
                        isinstance(meta_val, float) and np.isnan(meta_val)
                    ):
                        aligned_meta.append(float(meta_val))
                        aligned_delta.append(float(entry["delta"]))

            n = len(aligned_meta)
            if n < _MIN_N_FOR_CORRELATION:
                continue

            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", stats.ConstantInputWarning)
                    r, p = stats.spearmanr(aligned_meta, aligned_delta)
                if np.isnan(r):
                    continue
            except Exception:
                continue

            if abs(r) >= _MIN_SPEARMAN_ABS_R and p <= _MAX_P_VALUE:
                correlations.append({
                    "field": field,
                    "metadata_col": meta_col,
                    "spearman_r": round(float(r), 4),
                    "p_value": round(float(p), 6),
                    "n": n,
                })

    # Sort correlations by |r| descending
    correlations.sort(key=lambda x: abs(x["spearman_r"]), reverse=True)

    # --- Filtered field deltas (model_filtered source — v2 data) ---
    filtered_field_deltas: dict[str, dict] = {}
    for field in delta_fields:
        filtered_deltas: list[float] = []
        for row_deltas in all_deltas:
            entry = row_deltas.get(field, {})
            if entry.get("source") == _SOURCE_MODEL_FILTERED and entry.get("delta") is not None:
                filtered_deltas.append(float(entry["delta"]))
        if filtered_deltas:
            arr = np.array(filtered_deltas, dtype=float)
            filtered_field_deltas[field] = {
                "n": len(arr),
                "mean_delta": float(np.mean(arr)),
                "abs_mean_delta": float(np.mean(np.abs(arr))),
                "std_delta": float(np.std(arr)),
            }

    return {
        "n_photos": n_photos,
        "metadata_coverage": metadata_coverage,
        "per_field": per_field,
        "most_adjusted_fields": most_adjusted_fields,
        "correlations": correlations,
        "filtered_field_deltas": filtered_field_deltas,
    }


def prepare_finetune_dataset(
    captures: pd.DataFrame,
    original_train_parquet: Path,
    output_path: Path,
    weight_recent: float = 1.0,
) -> pd.DataFrame:
    """
    Combine original training data with captured user edits into a fine-tune Parquet.

    All captured photos are included regardless of delta magnitude — filtering is a
    fine-tuning-time decision, not a capture-time decision. Sample weights are equal
    by default (weight_recent=1.0); set weight_recent > 1.0 via the fine-tune CLI
    if you want captured edits to be sampled more frequently.

    Args:
        captures:               DataFrame from capture_user_edits().
        original_train_parquet: Path to the original training Parquet split.
        output_path:            Where to write the combined Parquet.
        weight_recent:          Sample weight for captured rows (default 1.0 = equal).

    Returns:
        Combined DataFrame, also written to output_path.
    """
    # --- Load original training data ---
    original_df = pd.read_parquet(original_train_parquet)
    original_df = original_df.copy()
    original_df["sample_weight"] = 1.0

    # --- Build finetune rows from captures ---
    # Take only the training-compatible columns; drop provenance/delta cols.
    provenance_cols = {
        "model_version", "model_path", "prediction_timestamp",
        "xmp_modified_time", "edit_lag_seconds", "predicted_values", "deltas",
    }
    training_cols = [c for c in captures.columns if c not in provenance_cols]
    capture_rows = captures[training_cols].copy()
    capture_rows["sample_weight"] = weight_recent

    # Align columns to original_df (original may have columns captures lacks, and vice versa)
    combined = pd.concat([original_df, capture_rows], ignore_index=True, sort=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    return combined
