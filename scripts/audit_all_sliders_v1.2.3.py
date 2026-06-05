#!/usr/bin/env python3
"""All-slider behavior audit on v1.2.3 (dp-event-v1.2.3, 135-output v1 model).

Read-only diagnostic. Outputs:
- scripts/output/all_slider_audit_v1.2.3.md (markdown report)
- scripts/output/all_slider_audit_v1.2.3_stats.parquet (raw per-slider stats)

Loads the production v1.2.3 ckpt as v1 (slider_set_version="v1"), runs raw
(unpostprocessed) inference across the full stratified test split, and
categorises each of the 135 sliders into HEALTHY / HIGH ERROR / COLLAPSED /
WRONG DIRECTION / SPARSE TARGET.
"""
from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from sonna_editor import config
from sonna_editor.data.xmp import LR_DEFAULTS
from sonna_editor.model.architecture import SonnaEditor
from sonna_editor.runtime import preferred_torch_device
from sonna_editor.model.augmentation import ValidationAugmentation
from sonna_editor.data.extract import compute_histogram

_logger = logging.getLogger("audit_all_sliders_v123")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Paths + config
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = Path("v1_learning/model-v1.2.3-prod256.ckpt")
DEFAULT_TEST_PARQUET = Path(
    "data/training_workspace/sonna_personal_001_dataset/splits_v2_stratified/test.parquet"
)
OUTPUT_DIR = Path("scripts/output")
REPORT_PATH = OUTPUT_DIR / "all_slider_audit_v1.2.3.md"
STATS_PATH = OUTPUT_DIR / "all_slider_audit_v1.2.3_stats.parquet"


def _find_published_checkpoints() -> list[Path]:
    published_dir = PROJECT_ROOT / "v1_learning"
    if not published_dir.exists():
        return []
    return sorted(published_dir.glob("model-v*.ckpt"))


def _select_path(paths: list[Path], description: str) -> Path | None:
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]

    print(f"Found {len(paths)} {description}:")
    for index, path in enumerate(paths, start=1):
        print(f"  {index}. {path}")

    while True:
        choice = input(f"Select a {description} by number (1-{len(paths)}), or ENTER to cancel: ").strip()
        if choice == "":
            return None
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(paths):
                return paths[index - 1]
        print("Invalid choice. Try again.")


def _locate_test_parquet(default: Path) -> Path | None:
    if default.exists():
        return default
    candidates = sorted(PROJECT_ROOT.rglob("test.parquet"))
    return candidates[0] if candidates else None

BATCH_SIZE = 32
IMAGE_RESOLUTION = 256
TEMPERATURE_IDX = 11

# v1.2.3 model has 135 outputs. config.SLIDER_FIELDS is now 147 (post-v2
# expansion in commit 3d0d90c) so we slice to the v1 subset for this audit.
V1_FIELDS: list[str] = list(config.SLIDER_FIELDS[:135])

# Categorisation thresholds (informed by task brief)
COLLAPSED_STD_RATIO = 0.1
HEALTHY_STD_RATIO = 0.5
HEALTHY_DIRECTION = 0.80
WRONG_DIRECTION_THRESHOLD = 0.55  # below this and signed range → flag
SPARSE_TARGET_RATIO = 0.80          # ≥ this fraction at default → sparse
SPARSE_EPSILON_FRAC = 0.01          # within 1% of range counts as "at default"

# Panel layout for the v1 architecture (135 sliders)
PANELS: list[tuple[str, int, int]] = [
    ("Tone", 0, 8),
    ("Presence", 8, 11),
    ("WB", 11, 13),
    ("HSL Hue", 13, 21),
    ("HSL Saturation", 21, 29),
    ("HSL Luminance", 29, 37),
    ("Parametric Tone Curve", 37, 44),
    ("Color Grading", 44, 58),
    ("Calibration", 58, 64),
    ("Detail Sharpening", 64, 68),
    ("Detail Noise Reduction", 68, 72),
    ("Effects", 72, 80),
    ("Lens Corrections", 80, 82),
    ("Transform", 82, 87),
    ("Tone Curves (Composite)", 87, 99),
    ("Tone Curves (Red)", 99, 111),
    ("Tone Curves (Green)", 111, 123),
    ("Tone Curves (Blue)", 123, 135),
]

# Hue channels that wrap at 360 (Color Grading hue wheel)
CIRCULAR_360_FIELDS: frozenset[str] = frozenset({
    "SplitToningShadowHue",
    "ColorGradeMidtoneHue",
    "SplitToningHighlightHue",
    "ColorGradeGlobalHue",
})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decode_histogram(blob: bytes) -> np.ndarray:
    """The parquet stores histograms as numpy .save bytes — load via BytesIO."""
    return np.load(io.BytesIO(blob))


def _safe_lookup(reg_map: dict[str, int], key) -> int:
    """Look up a string in a registry map; fall back to 0 (unknown) on miss/NaN."""
    if key is None or (isinstance(key, float) and np.isnan(key)):
        return 0
    return reg_map.get(str(key), 0)


def _circular_abs_error(pred: np.ndarray, target: np.ndarray, period: float = 360.0) -> np.ndarray:
    """|pred - target| wrapped to [0, period/2]."""
    diff = np.mod(pred - target, period)
    return np.minimum(diff, period - diff)


def _categorise(
    field: str,
    pred: np.ndarray,
    target: np.ndarray,
    range_lo: float,
    range_hi: float,
    default: float,
    panel_median_norm_mae: float,
) -> tuple[str, dict]:
    """Return (category, key_stats) for one slider.

    Handles NaN in targets (fields the source XMP didn't write are NaN in the
    parquet — typically means "user left at LR default"). Stats are computed
    on the non-NaN subset. sparse_frac counts NaN AS at-default.
    """
    range_span = range_hi - range_lo
    is_circular = field in CIRCULAR_360_FIELDS

    # NaN handling: sparse_frac counts NaN-rows + within-tolerance-of-default rows
    sparse_tol = SPARSE_EPSILON_FRAC * range_span
    nan_mask = np.isnan(target)
    near_default_mask = ~nan_mask & (np.abs(target - default) < sparse_tol)
    sparse_frac = float(np.mean(nan_mask | near_default_mask))

    # All other stats use only the non-NaN target rows ("photos where the user
    # actually set this slider"). If everything is NaN, return blank stats.
    valid = ~nan_mask
    n_valid = int(valid.sum())
    if n_valid == 0:
        return "SPARSE TARGET", {
            "mae": float("nan"), "median_ae": float("nan"), "p95_ae": float("nan"),
            "std_pred": float("nan"), "std_target": float("nan"), "std_ratio": float("nan"),
            "mean_pred": float("nan"), "mean_target": float("nan"), "mean_gap": float("nan"),
            "direction_correct": float("nan"), "corr": float("nan"),
            "sparse_frac": sparse_frac, "norm_mae": float("nan"),
            "is_circular": is_circular, "n_valid": 0,
        }

    pred_v = pred[valid]
    target_v = target[valid]

    # Errors (on valid subset)
    if is_circular:
        ae = _circular_abs_error(pred_v, target_v, 360.0)
    else:
        ae = np.abs(pred_v - target_v)
    mae = float(np.mean(ae))
    median_ae = float(np.median(ae))
    p95_ae = float(np.percentile(ae, 95))

    # Std + ratio
    std_pred = float(np.std(pred_v))
    std_target = float(np.std(target_v))
    std_ratio = (std_pred / std_target) if std_target > 1e-9 else float("nan")

    # Mean + gap
    mean_pred = float(np.mean(pred_v))
    mean_target = float(np.mean(target_v))
    mean_gap = mean_pred - mean_target

    # Direction correctness — only meaningful for signed ranges with both signs
    # represented in the target. Use 1% of range as "near-zero" tolerance.
    has_both_signs = (range_lo < 0 < range_hi)
    if has_both_signs:
        tol = SPARSE_EPSILON_FRAC * range_span
        signed_mask = np.abs(target_v) > tol
        if signed_mask.sum() > 0:
            sign_pred = np.sign(pred_v[signed_mask])
            sign_target = np.sign(target_v[signed_mask])
            direction_correct = float(np.mean(sign_pred == sign_target))
        else:
            direction_correct = float("nan")
    else:
        direction_correct = float("nan")

    # Correlation
    if std_target > 1e-9 and std_pred > 1e-9:
        try:
            corr = float(np.corrcoef(pred_v, target_v)[0, 1])
        except Exception:
            corr = float("nan")
    else:
        corr = float("nan")

    norm_mae = mae / range_span if range_span > 0 else float("nan")

    # Categorisation order matters — most specific first
    if sparse_frac >= SPARSE_TARGET_RATIO:
        category = "SPARSE TARGET"
    elif not np.isnan(std_ratio) and std_ratio < COLLAPSED_STD_RATIO:
        category = "COLLAPSED"
    elif has_both_signs and not np.isnan(direction_correct) and direction_correct < WRONG_DIRECTION_THRESHOLD:
        category = "WRONG DIRECTION"
    elif (
        not np.isnan(std_ratio) and std_ratio >= HEALTHY_STD_RATIO
        and (np.isnan(direction_correct) or direction_correct >= HEALTHY_DIRECTION)
        and norm_mae <= 1.5 * panel_median_norm_mae
    ):
        category = "HEALTHY"
    else:
        category = "HIGH ERROR"

    return category, {
        "mae": mae,
        "median_ae": median_ae,
        "p95_ae": p95_ae,
        "std_pred": std_pred,
        "std_target": std_target,
        "std_ratio": std_ratio,
        "mean_pred": mean_pred,
        "mean_target": mean_target,
        "mean_gap": mean_gap,
        "direction_correct": direction_correct,
        "corr": corr,
        "sparse_frac": sparse_frac,
        "norm_mae": norm_mae,
        "is_circular": is_circular,
        "n_valid": n_valid,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(df: pd.DataFrame, model: SonnaEditor, device: str) -> np.ndarray:
    """Return [N, 135] tensor of RAW (unpostprocessed) predictions in prediction space.

    The audit is defined over the first 135 v1 fields. If a v2 model is loaded,
    the extra 12 outputs are ignored so the old audit remains valid.
    Temperature (idx 11) is in log-K, all other fields in native slider units.
    """
    transform = ValidationAugmentation(resolution=IMAGE_RESOLUTION)
    reg = model.registry
    n = len(df)
    out = np.zeros((n, len(V1_FIELDS)), dtype=np.float32)

    # Pull pandas Series once to avoid per-row column lookup overhead
    thumb_paths = df["thumbnail_path"].tolist()
    isos = df["iso"].astype(float).tolist()
    shutters = df["shutter_speed"].astype(float).tolist()
    apertures = df["aperture"].astype(float).tolist()
    focals = df["focal_length"].astype(float).tolist()
    lens_models = df["lens_model"].tolist()
    makes = df["make"].tolist()
    models = df["model"].tolist()
    cam_profiles = df["camera_profile"].tolist()
    wb_presets = df["white_balance_preset"].tolist()
    as_temps = df["as_shot_temperature"].astype(float).tolist()
    as_tints = df["as_shot_tint"].astype(float).tolist()
    histograms = df["histogram"].tolist()

    model.eval()
    for start in tqdm(range(0, n, BATCH_SIZE), desc="inference"):
        end = min(start + BATCH_SIZE, n)
        img_batch_list: list[torch.Tensor] = []
        hist_batch_list: list[torch.Tensor] = []

        for i in range(start, end):
            img = Image.open(thumb_paths[i]).convert("RGB")
            t = TF.pil_to_tensor(img)
            t = transform(t)
            img_batch_list.append(t)

            # Decode histogram from parquet bytes; if absent/malformed, fall back
            # to computing one from the thumbnail (matches the build-time path).
            hist_blob = histograms[i]
            try:
                hist = _decode_histogram(hist_blob).flatten()
                if hist.shape[0] != 96:
                    hist = compute_histogram(img).flatten()
            except Exception:
                hist = compute_histogram(img).flatten()
            hist_batch_list.append(torch.from_numpy(hist.astype(np.float32)))

        img_batch = torch.stack(img_batch_list).to(device)
        hist_batch = torch.stack(hist_batch_list).to(device)

        # Build metadata batch with registry-looked-up IDs
        meta = {
            "iso": torch.tensor(isos[start:end], dtype=torch.float32, device=device),
            "shutter_speed": torch.tensor(shutters[start:end], dtype=torch.float32, device=device),
            "aperture": torch.tensor(apertures[start:end], dtype=torch.float32, device=device),
            "focal_length": torch.tensor(focals[start:end], dtype=torch.float32, device=device),
            "camera_body_id": torch.tensor(
                [_safe_lookup(reg.camera_bodies, f"{makes[i]} {models[i]}") for i in range(start, end)],
                dtype=torch.long, device=device,
            ),
            "camera_make_id": torch.tensor(
                [_safe_lookup(reg.camera_makes, makes[i]) for i in range(start, end)],
                dtype=torch.long, device=device,
            ),
            "camera_model_id": torch.tensor(
                [_safe_lookup(reg.camera_models, models[i]) for i in range(start, end)],
                dtype=torch.long, device=device,
            ),
            "lens_id": torch.tensor(
                [_safe_lookup(reg.lenses, lens_models[i]) for i in range(start, end)],
                dtype=torch.long, device=device,
            ),
            "camera_profile_id": torch.tensor(
                [_safe_lookup(reg.camera_profiles, cam_profiles[i]) for i in range(start, end)],
                dtype=torch.long, device=device,
            ),
            "wb_preset_id": torch.tensor(
                [_safe_lookup(reg.wb_presets, wb_presets[i]) for i in range(start, end)],
                dtype=torch.long, device=device,
            ),
            "histogram": hist_batch,
            "as_shot_temperature": torch.tensor(as_temps[start:end], dtype=torch.float32, device=device),
            "as_shot_tint": torch.tensor(as_tints[start:end], dtype=torch.float32, device=device),
        }

        with torch.no_grad():
            preds = model(img_batch, meta)

        preds_np = preds.cpu().numpy()
        if preds_np.shape[1] < len(V1_FIELDS):
            raise ValueError(
                f"Model output has {preds_np.shape[1]} sliders, but v1 audit requires "
                f"{len(V1_FIELDS)} fields. Use a v1 checkpoint or audit script matching the "
                "model's slider_set_version."
            )
        if preds_np.shape[1] > len(V1_FIELDS):
            _logger.warning(
                "Model output has %d sliders; slicing to first %d v1 fields for this audit.",
                preds_np.shape[1], len(V1_FIELDS)
            )
            preds_np = preds_np[:, : len(V1_FIELDS)]

        out[start:end] = preds_np

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Report builders
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: float, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v:.{decimals}f}"


def _panel_of(idx: int) -> str:
    for name, lo, hi in PANELS:
        if lo <= idx < hi:
            return name
    return "Unknown"


def build_markdown_report(
    ckpt_path: Path,
    test_parquet_path: Path,
    stats_df: pd.DataFrame,
    temp_log_mae: float,
    temp_kelvin_mae: float,
    temp_kelvin_p95: float,
    tone_curve_metrics: dict,
    worst_photos: list[dict],
    surprises: list[str],
    n_rows: int,
) -> str:
    lines: list[str] = []

    lines.append("# All-Slider Behaviour Audit — v1.2.3 (dp-event-v1.2.3)")
    lines.append("")
    lines.append(f"**Model:** `{ckpt_path}`  ")
    lines.append(f"**Test split:** `{test_parquet_path}`  ({n_rows} photos)  ")
    lines.append(f"**Generated:** {pd.Timestamp.now().isoformat(timespec='seconds')}  ")
    lines.append("**Architecture:** v1, 13 heads, 135 outputs  ")
    lines.append("")
    lines.append("Read-only diagnostic against the live production checkpoint. Raw")
    lines.append("model predictions, no postprocess clamping. Temperature (idx 11)")
    lines.append("is in log-K in the model's prediction space; analysed in both")
    lines.append("log-K and Kelvin views.")
    lines.append("")

    # ── Summary ──
    lines.append("## 1. Summary")
    lines.append("")
    by_cat = stats_df.groupby("category").size().to_dict()
    cats_order = ["HEALTHY", "HIGH ERROR", "COLLAPSED", "WRONG DIRECTION", "SPARSE TARGET"]
    lines.append("| Category | Count | % of 135 |")
    lines.append("|---|---:|---:|")
    for cat in cats_order:
        n = by_cat.get(cat, 0)
        pct = 100.0 * n / len(stats_df)
        lines.append(f"| {cat} | {n} | {pct:.1f}% |")
    lines.append("")

    # ── Per-panel breakdown ──
    lines.append("## 2. Per-panel breakdown (all 135 sliders)")
    lines.append("")
    lines.append("Columns: MAE in native units; std_ratio = std(pred)/std(target);")
    lines.append("dir = sign-agreement % on signed-range sliders; corr = Pearson;")
    lines.append("sparse = fraction of test rows at default.")
    lines.append("")

    for panel_name, lo, hi in PANELS:
        panel_slice = stats_df.iloc[lo:hi]
        lines.append(f"### {panel_name} (idx {lo}-{hi-1})")
        lines.append("")
        lines.append("| idx | field | range | category | mae | std_ratio | dir | corr | sparse |")
        lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|")
        for _, row in panel_slice.iterrows():
            rng = f"[{row['range_lo']:.0f}, {row['range_hi']:.0f}]"
            dir_str = _fmt(row["direction_correct"], 2) if not np.isnan(row["direction_correct"]) else "—"
            corr_str = _fmt(row["corr"], 2)
            spar_str = f"{row['sparse_frac']*100:.0f}%"
            lines.append(
                f"| {row['idx']} | {row['field']} | {rng} | {row['category']} "
                f"| {_fmt(row['mae'], 3)} | {_fmt(row['std_ratio'], 2)} "
                f"| {dir_str} | {corr_str} | {spar_str} |"
            )
        lines.append("")

    # ── Detailed view for non-HEALTHY non-SPARSE sliders ──
    lines.append("## 3. Detailed view — non-HEALTHY non-SPARSE sliders")
    lines.append("")
    flagged = stats_df[
        stats_df["category"].isin(["COLLAPSED", "WRONG DIRECTION", "HIGH ERROR"])
    ].sort_values(["category", "norm_mae"], ascending=[True, False])

    if len(flagged) == 0:
        lines.append("_None — every slider is HEALTHY or SPARSE TARGET._")
        lines.append("")
    else:
        for _, row in flagged.iterrows():
            lines.append(f"### `{row['field']}` — {row['category']}  (idx {row['idx']})")
            lines.append("")
            lines.append(
                f"- range: `[{row['range_lo']:.2f}, {row['range_hi']:.2f}]`,"
                f" default: `{row['default']:.2f}`"
            )
            lines.append(
                f"- MAE: `{_fmt(row['mae'], 3)}` (median `{_fmt(row['median_ae'], 3)}`,"
                f" p95 `{_fmt(row['p95_ae'], 3)}`)"
            )
            lines.append(
                f"- norm_mae: `{_fmt(row['norm_mae'], 4)}` (mae / range_span)"
            )
            lines.append(
                f"- std(pred)={_fmt(row['std_pred'], 3)},"
                f" std(target)={_fmt(row['std_target'], 3)},"
                f" ratio={_fmt(row['std_ratio'], 3)}"
            )
            lines.append(
                f"- mean(pred)={_fmt(row['mean_pred'], 3)},"
                f" mean(target)={_fmt(row['mean_target'], 3)},"
                f" gap={_fmt(row['mean_gap'], 3)}"
            )
            if not np.isnan(row["direction_correct"]):
                lines.append(f"- direction correct on signed-range subset: `{row['direction_correct']*100:.1f}%`")
            lines.append(f"- Pearson corr(pred, target): `{_fmt(row['corr'], 3)}`")
            if row["category"] == "COLLAPSED":
                lines.append("- **Diagnosis:** predictions cluster tightly (std ratio < 0.1). Model is not learning the target spread.")
            elif row["category"] == "WRONG DIRECTION":
                lines.append("- **Diagnosis:** sign agreement on signed-range examples is below 55%. Model is systematically getting the direction wrong.")
            elif row["category"] == "HIGH ERROR":
                lines.append("- **Diagnosis:** model is varying but missing — error is > 1.5× panel-median normalised MAE.")
            lines.append("")

    # ── Tone curve section ──
    lines.append("## 4. Tone curve identity-collapse check")
    lines.append("")
    lines.append(
        "Documented issue (HANDOVER Part 6 item 9): the model converged to near-identity"
        " curve predictions across all 4 channels in v1.0.1. Quantifying here against the"
        " current shipping v1.2.3."
    )
    lines.append("")
    lines.append("| Channel | mean L2 (pred ↔ identity) | mean L2 (target ↔ identity) | identity bias |")
    lines.append("|---|---:|---:|---|")
    for ch_name, m in tone_curve_metrics.items():
        bias = "✗ near-identity" if m["pred_id_dist"] < 0.25 * m["target_id_dist"] else (
            "partial" if m["pred_id_dist"] < 0.6 * m["target_id_dist"] else "OK"
        )
        lines.append(
            f"| {ch_name} | {m['pred_id_dist']:.2f} | {m['target_id_dist']:.2f} | {bias} |"
        )
    lines.append("")
    lines.append(
        "_Distance = mean over photos of L2-norm between (Pt2_Y..Pt5_Y) vs identity (Pt_n_Y == Pt_n_X)._"
        " A near-zero pred-vs-identity distance with non-zero target-vs-identity distance"
        " confirms collapse."
    )
    lines.append("")

    # ── Temperature dual-view ──
    lines.append("## 5. Temperature dual-view (log-K + Kelvin)")
    lines.append("")
    lines.append(f"- log-K MAE (prediction space): `{temp_log_mae:.4f}` (typical training-loss scale)")
    lines.append(f"- Kelvin MAE (user-facing): `{temp_kelvin_mae:.0f} K` (target was <250 K per HANDOVER)")
    lines.append(f"- Kelvin p95 abs error: `{temp_kelvin_p95:.0f} K`")
    lines.append("")

    # ── Worst-offender photos ──
    lines.append("## 6. Worst-offender photos (top 20 by weighted error)")
    lines.append("")
    lines.append("Weighting: sum over fields of `|pred - target| / range_span` (Temperature uses Kelvin/range_K).")
    lines.append("")
    for i, p in enumerate(worst_photos, 1):
        lines.append(f"**{i}. `{p['file_id'][:16]}...`** — shoot `{p['shoot_id']}`, total weighted error `{p['total_err']:.3f}`")
        lines.append(
            f"   - ISO {p['iso']}, {p['camera_body']}, focal {p['focal_length']} mm"
        )
        contrib_str = ", ".join(
            f"{f} (Δ={d:+.3f})" for f, d in p["top_contribs"]
        )
        lines.append(f"   - top contributing sliders: {contrib_str}")
        lines.append("")

    # ── Surprises ──
    lines.append("## 7. Surprises / contradictions")
    lines.append("")
    if not surprises:
        lines.append("_No surprises against existing HANDOVER assumptions._")
    else:
        for s in surprises:
            lines.append(f"- {s}")
    lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Audit all sliders against a published v1 checkpoint.")
    parser.add_argument("--ckpt", type=Path, help="Published checkpoint path to audit.")
    parser.add_argument("--test-parquet", type=Path, help="Path to the test parquet file.")
    parser.add_argument("--list-checkpoints", action="store_true", help="List discovered published checkpoints and exit.")
    args = parser.parse_args()

    if args.list_checkpoints:
        checkpoints = _find_published_checkpoints()
        if not checkpoints:
            print("No published checkpoints were found in v1_learning/model-v*.ckpt.")
            return 0
        print("Published checkpoints found:")
        for path in checkpoints:
            print(f"  {path}")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        available_ckpts = _find_published_checkpoints()
        if not available_ckpts:
            print("No published checkpoints were found in v1_learning/model-v*.ckpt.")
            return 1
        ckpt_path = _select_path(available_ckpts, "published checkpoint")
        if ckpt_path is None:
            print("No checkpoint selected; aborting.")
            return 1

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    test_parquet = args.test_parquet or _locate_test_parquet(DEFAULT_TEST_PARQUET)
    if test_parquet is None or not test_parquet.exists():
        print("Test parquet file not found.")
        print(
            "Expected at data/training_workspace/sonna_personal_001_dataset/"
            "splits_v2_stratified/test.parquet or discoverable under the project tree."
        )
        return 1

    _logger.info("Loading model from %s", ckpt_path)
    device = preferred_torch_device()
    model = SonnaEditor.from_checkpoint(ckpt_path, device="cpu")
    model.to(device)
    _logger.info("Model loaded: slider_set_version=%s, device=%s", model._slider_set_version, device)
    if model._slider_set_version != "v1":
        _logger.warning(
            "Loaded a %s model. This audit targets v1 fields only; extra v2 outputs will be ignored.",
            model._slider_set_version,
        )

    _logger.info("Loading test parquet")
    df = pd.read_parquet(test_parquet)
    _logger.info("Test rows: %d", len(df))

    # Targets [N, 135] in native slider units (Temperature in Kelvin)
    targets = df[list(V1_FIELDS)].astype(np.float32).to_numpy()

    # Raw predictions [N, 135] in prediction space (Temperature in log-K)
    preds_pred_space = run_inference(df, model, device)

    # Build aligned arrays for stats: convert Temperature pred → Kelvin so we
    # can compare to targets in user-facing units. Per-slider stats live in
    # user-facing units; we keep a log-space Temperature stat as a side-note.
    preds_user_space = preds_pred_space.copy()
    preds_user_space[:, TEMPERATURE_IDX] = np.exp(preds_pred_space[:, TEMPERATURE_IDX])

    # Temperature dual-view (NaN-aware — Temperature target NaN means
    # user didn't override As-Shot WB, so exclude those rows)
    temp_targets = targets[:, TEMPERATURE_IDX]
    temp_valid = ~np.isnan(temp_targets)
    if temp_valid.sum() > 0:
        temp_log_mae = float(np.mean(np.abs(
            preds_pred_space[temp_valid, TEMPERATURE_IDX]
            - np.log(np.clip(temp_targets[temp_valid], 1.0, None))
        )))
        temp_kelvin_abs = np.abs(preds_user_space[temp_valid, TEMPERATURE_IDX] - temp_targets[temp_valid])
        temp_kelvin_mae = float(np.mean(temp_kelvin_abs))
        temp_kelvin_p95 = float(np.percentile(temp_kelvin_abs, 95))
    else:
        temp_log_mae = temp_kelvin_mae = temp_kelvin_p95 = float("nan")

    # ── Per-slider stats ──
    # First pass: compute per-slider raw stats so we can compute panel-median norm_mae.
    raw_stats: list[dict] = []
    for idx, field in enumerate(V1_FIELDS):
        lo, hi = config.SLIDER_RANGES[field]
        default = LR_DEFAULTS[field]
        # Need a panel_median placeholder for this first pass; second pass refines.
        cat, st = _categorise(
            field, preds_user_space[:, idx], targets[:, idx],
            lo, hi, default,
            panel_median_norm_mae=0.0,  # placeholder; overridden after panel computed
        )
        raw_stats.append({
            "idx": idx, "field": field, "panel": _panel_of(idx),
            "range_lo": lo, "range_hi": hi, "default": float(default),
            "category": cat, **st,
        })

    stats_df = pd.DataFrame(raw_stats)
    # Panel medians of norm_mae
    panel_medians = stats_df.groupby("panel")["norm_mae"].median().to_dict()

    # Second pass: re-categorise with panel-median context
    final_rows: list[dict] = []
    for idx, field in enumerate(V1_FIELDS):
        lo, hi = config.SLIDER_RANGES[field]
        default = LR_DEFAULTS[field]
        panel = _panel_of(idx)
        pm = panel_medians.get(panel, 0.0)
        cat, st = _categorise(
            field, preds_user_space[:, idx], targets[:, idx],
            lo, hi, default, panel_median_norm_mae=pm,
        )
        final_rows.append({
            "idx": idx, "field": field, "panel": panel,
            "range_lo": lo, "range_hi": hi, "default": float(default),
            "category": cat, **st,
        })
    stats_df = pd.DataFrame(final_rows)

    # ── Tone-curve identity check ──
    # Per HANDOVER, identity means Pt_n_Y == Pt_n_X. We check Y values against
    # their matching X values (skipping endpoints which are pinned at identity).
    # Each channel has 6 Y values at indices: composite=88,90,92,94,96,98 etc.
    # We'll use Pt2_Y..Pt5_Y (the four interior points) per channel.
    tone_curve_metrics: dict[str, dict] = {}
    channel_specs = [
        ("Composite", "ToneCurve"),
        ("Red", "ToneCurveRed"),
        ("Green", "ToneCurveGreen"),
        ("Blue", "ToneCurveBlue"),
    ]
    for ch_name, ch_prefix in channel_specs:
        interior_y_fields = [f"{ch_prefix}_Pt{n}_Y" for n in (2, 3, 4, 5)]
        interior_x_fields = [f"{ch_prefix}_Pt{n}_X" for n in (2, 3, 4, 5)]
        y_idx = [V1_FIELDS.index(f) for f in interior_y_fields]
        x_idx = [V1_FIELDS.index(f) for f in interior_x_fields]

        # "Distance from identity" = L2 over interior Y deviations from matching X
        # Rows where target tone-curve values are NaN are excluded so the metric
        # reflects only photos where the user actually set the curve.
        pred_dev = preds_user_space[:, y_idx] - preds_user_space[:, x_idx]
        target_dev = targets[:, y_idx] - targets[:, x_idx]
        valid_rows = ~np.any(np.isnan(target_dev), axis=1)
        if valid_rows.sum() > 0:
            pred_l2 = np.mean(np.linalg.norm(pred_dev[valid_rows], axis=1))
            target_l2 = np.mean(np.linalg.norm(target_dev[valid_rows], axis=1))
        else:
            pred_l2 = target_l2 = float("nan")
        tone_curve_metrics[ch_name] = {
            "pred_id_dist": float(pred_l2),
            "target_id_dist": float(target_l2),
        }

    # ── Worst-offender photos ──
    # Per-row weighted error: sum over fields of |pred-target| / range_span.
    # NaN targets (user left at default) → contribute 0 to the row total
    # (no ground truth means no error signal).
    ranges = np.array([config.SLIDER_RANGES[f][1] - config.SLIDER_RANGES[f][0]
                       for f in V1_FIELDS], dtype=np.float32)
    ranges_safe = np.where(ranges > 0, ranges, 1.0)
    abs_err_user = np.abs(preds_user_space - targets)
    # For circular fields, use circular distance
    for cf in CIRCULAR_360_FIELDS:
        cidx = V1_FIELDS.index(cf)
        abs_err_user[:, cidx] = _circular_abs_error(preds_user_space[:, cidx], targets[:, cidx], 360.0)
    norm_abs_err = abs_err_user / ranges_safe[None, :]
    norm_abs_err = np.where(np.isnan(targets), 0.0, norm_abs_err)
    per_row_total = norm_abs_err.sum(axis=1)
    worst_idx = np.argsort(per_row_total)[::-1][:20]

    worst_photos: list[dict] = []
    for ridx in worst_idx:
        row = df.iloc[int(ridx)]
        # Top 5 contributing fields for this row
        contribs = norm_abs_err[ridx]
        top5 = np.argsort(contribs)[::-1][:5]
        top_contribs = [
            (V1_FIELDS[int(j)], float(preds_user_space[ridx, j] - targets[ridx, j]))
            for j in top5
        ]
        worst_photos.append({
            "file_id": str(row["id"]),
            "shoot_id": str(row.get("shoot_id", "?")),
            "iso": int(row.get("iso", 0) or 0),
            "camera_body": str(row.get("camera_body", "?")),
            "focal_length": float(row.get("focal_length", 0.0) or 0.0),
            "total_err": float(per_row_total[ridx]),
            "top_contribs": top_contribs,
        })

    # ── Surprises detection ──
    surprises: list[str] = []
    # Surprise 1: any slider categorised differently from what HANDOVER implies.
    # Specifically Temperature should be HIGH ERROR per HANDOVER (~730K MAE).
    temp_row = stats_df[stats_df["field"] == "Temperature"].iloc[0]
    if temp_row["category"] == "HEALTHY":
        surprises.append(
            f"Temperature is classified HEALTHY despite the documented "
            f"730K-MAE failure mode in HANDOVER. Current Kelvin MAE: {temp_kelvin_mae:.0f} K."
        )
    # Surprise 2: any tone-curve channel NOT near-identity collapse
    for ch_name, m in tone_curve_metrics.items():
        if m["pred_id_dist"] >= 0.6 * m["target_id_dist"]:
            surprises.append(
                f"Tone curve {ch_name} is NOT showing the documented identity-collapse "
                f"(pred-vs-identity {m['pred_id_dist']:.2f} vs target-vs-identity {m['target_id_dist']:.2f}). "
                f"HANDOVER says all 4 channels collapse — verify."
            )
    # Surprise 3: COLLAPSED count higher or lower than the documented tone-curve-only collapse
    n_collapsed = (stats_df["category"] == "COLLAPSED").sum()
    n_tc_collapsed = (
        (stats_df["category"] == "COLLAPSED") & stats_df["panel"].str.startswith("Tone Curves")
    ).sum()
    if n_collapsed > n_tc_collapsed:
        non_tc = stats_df[
            (stats_df["category"] == "COLLAPSED") & ~stats_df["panel"].str.startswith("Tone Curves")
        ]["field"].tolist()
        surprises.append(
            f"COLLAPSED sliders found OUTSIDE the tone-curve panels: {non_tc}. "
            f"HANDOVER documents tone-curve collapse only — this is new."
        )

    # ── Save outputs ──
    stats_df.to_parquet(STATS_PATH, index=False)
    _logger.info("Wrote stats parquet to %s", STATS_PATH)

    report = build_markdown_report(
        ckpt_path=ckpt_path,
        test_parquet_path=test_parquet,
        stats_df=stats_df,
        temp_log_mae=temp_log_mae,
        temp_kelvin_mae=temp_kelvin_mae,
        temp_kelvin_p95=temp_kelvin_p95,
        tone_curve_metrics=tone_curve_metrics,
        worst_photos=worst_photos,
        surprises=surprises,
        n_rows=len(df),
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    _logger.info("Wrote report to %s", REPORT_PATH)

    print()
    print("=" * 60)
    print("Audit complete.")
    print(f"  Report: {REPORT_PATH}")
    print(f"  Stats:  {STATS_PATH}")
    print(f"  Test rows analysed: {len(df)}")
    print()
    by_cat = stats_df.groupby("category").size().to_dict()
    for cat in ("HEALTHY", "HIGH ERROR", "COLLAPSED", "WRONG DIRECTION", "SPARSE TARGET"):
        print(f"  {cat:18}: {by_cat.get(cat, 0):3d}")
    print()
    print(f"  Temperature log-K MAE: {temp_log_mae:.4f}")
    print(f"  Temperature Kelvin MAE: {temp_kelvin_mae:.0f} K (p95: {temp_kelvin_p95:.0f} K)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
