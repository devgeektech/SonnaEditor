from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from sonna_editor.config import SLIDER_FIELDS
from sonna_editor.data.dataset import load_dataset

logger = logging.getLogger(__name__)

# Tone curve fields start at index 87 — identity curves have X/Y values that
# are non-zero (e.g. Pt2_X=85, Pt3_X=170, Pt4_X=255), so including them in the
# zero-count would falsely inflate "unedited" scores. Check scalar sliders only.
_SCALAR_SLIDER_FIELDS: list[str] = [f for f in SLIDER_FIELDS if not f.startswith("ToneCurve")]

# Thresholds (calibrated against _SCALAR_SLIDER_FIELDS, not all 119)
_UNEDITED_ZERO_THRESHOLD = 80  # scalar sliders that must be 0.0 to flag as "likely unedited"
_OUTLIER_STD = 3.0             # std devs beyond mean = outlier
_HIGH_VARIANCE_STD = 15.0      # slider std dev above which we flag as high-variance
_STOP_MIN_PHOTOS = 100
_WARN_MIN_PHOTOS = 500
_WARN_UNEDITED_RATIO = 0.20    # 20 % unedited → WARN
_STOP_UNEDITED_RATIO = 0.80    # 80 % unedited → STOP

# Cross-platform training estimate: seconds per epoch per photo at batch 16.
# Treat this as a rough planning number; CUDA, MPS, and CPU hosts vary a lot.
_SECONDS_PER_EPOCH_PER_PHOTO = 0.045  # ~45s / 1000 photos


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _count_unedited(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask of rows where ≥80 scalar slider values are exactly 0.0."""
    zero_counts = (df[_SCALAR_SLIDER_FIELDS] == 0.0).sum(axis=1)
    return zero_counts >= _UNEDITED_ZERO_THRESHOLD


def _find_outliers(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return {slider: [photo_ids]} for values > 3 std devs from mean."""
    outliers: dict[str, list[str]] = {}
    for field in SLIDER_FIELDS:
        col = df[field].dropna()
        if len(col) < 2:
            continue
        mean, std = col.mean(), col.std()
        if std == 0:
            continue
        mask = ((df[field] - mean).abs() > _OUTLIER_STD * std)
        flagged = df.loc[mask, "id"].tolist()
        if flagged:
            outliers[field] = flagged
    return outliers


def _slider_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of mean/std/min/max for each slider."""
    return df[SLIDER_FIELDS].agg(["mean", "std", "min", "max"]).T


def _estimate_training_time(n_photos: int, n_epochs: int = 100) -> tuple[float, str]:
    """Estimate training minutes for n_photos at batch 16 on a typical laptop GPU."""
    seconds = _SECONDS_PER_EPOCH_PER_PHOTO * n_photos * n_epochs
    minutes = seconds / 60
    if minutes < 60:
        label = f"~{minutes:.0f} min"
    else:
        label = f"~{minutes / 60:.1f} hr"
    return minutes, label


def _decide_status(
    n_photos: int,
    unedited_ratio: float,
    mixed_profiles: bool,
) -> str:
    if n_photos < _STOP_MIN_PHOTOS:
        return "STOP"
    if unedited_ratio >= _STOP_UNEDITED_RATIO:
        return "STOP"
    if n_photos < _WARN_MIN_PHOTOS:
        return "WARN"
    if unedited_ratio >= _WARN_UNEDITED_RATIO:
        return "WARN"
    if mixed_profiles:
        return "WARN"
    return "GO"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_iso_distribution(df: pd.DataFrame, plots_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        iso_col = df["iso"].dropna()
        if iso_col.empty:
            return None

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(iso_col.astype(float), bins=30, color="#4A90D9", edgecolor="white", linewidth=0.5)
        ax.set_title("ISO Distribution")
        ax.set_xlabel("ISO")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        fig.tight_layout()
        out = plots_dir / "iso_distribution.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out
    except Exception as e:
        logger.warning("Could not generate ISO plot: %s", e)
        return None


def _plot_slider_distributions(df: pd.DataFrame, plots_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(SLIDER_FIELDS)
        ncols = 6
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 2.5))
        axes_flat = axes.flatten()

        for i, field in enumerate(SLIDER_FIELDS):
            ax = axes_flat[i]
            col = df[field].dropna().astype(float)
            ax.hist(col, bins=25, color="#E87D4D", edgecolor="white", linewidth=0.3)
            label = field.replace("Adjustment", "Adj").replace("2012", "")
            ax.set_title(label, fontsize=7)
            ax.tick_params(labelsize=6)

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("Slider Value Distributions", fontsize=12, y=1.01)
        fig.tight_layout()
        out = plots_dir / "slider_distributions.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return out
    except Exception as e:
        logger.warning("Could not generate slider distributions plot: %s", e)
        return None


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _build_report(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    status: str,
    n_unedited: int,
    unedited_ids: list[str],
    outliers: dict[str, list[str]],
    high_variance: list[str],
    mixed_profiles: bool,
    training_minutes: float,
    training_label: str,
    iso_plot: Path | None,
    slider_plot: Path | None,
) -> str:
    n = len(df)
    n_shoots = df["shoot_id"].nunique() if "shoot_id" in df.columns else "N/A"
    unedited_pct = 100 * n_unedited / n if n > 0 else 0

    cap_dates = pd.to_datetime(df["capture_datetime"], errors="coerce").dropna()
    date_range = (
        f"{cap_dates.min().date()} – {cap_dates.max().date()}"
        if not cap_dates.empty else "unknown"
    )

    camera_counts = df["camera_body"].value_counts() if "camera_body" in df.columns else pd.Series(dtype=int)
    profile_counts = df["camera_profile"].value_counts() if "camera_profile" in df.columns else pd.Series(dtype=int)
    iso_counts = df["iso"].value_counts().sort_index() if "iso" in df.columns else pd.Series(dtype=int)

    status_emoji = {"GO": "✅", "WARN": "⚠️", "STOP": "🛑"}.get(status, "?")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: list[str] = []

    # --- Summary ---
    lines += [
        "# Sonna Editor — Dataset Audit Report",
        "",
        f"Generated: {now}",
        "",
        "## Summary",
        "",
        "| | |",
        "|---|---|",
        f"| **Status** | {status_emoji} **{status}** |",
        f"| Photos | {n:,} |",
        f"| Shoots | {n_shoots} |",
        f"| Date range | {date_range} |",
        f"| Likely unedited | {n_unedited} ({unedited_pct:.1f}%) |",
        f"| Outlier photos | {sum(len(v) for v in outliers.values())} across {len(outliers)} sliders |",
        "",
    ]

    # Recommendation text
    if status == "GO":
        lines.append("Dataset looks good. Proceed to Phase 3 training.\n")
    elif status == "WARN":
        lines.append("Dataset has issues that may affect model quality. Review flags below before training.\n")
    else:
        lines.append("Dataset is not ready for training. Fix the flagged issues first.\n")

    # --- Hardware estimate ---
    lines += [
        "## Hardware Estimate (batch 16, 100 epochs)",
        "",
        "| | |",
        "|---|---|",
        f"| Photos | {n:,} |",
        f"| Estimated training time | {training_label} |",
        f"| Peak memory estimate | ~{max(4, int(n / 250))} GB |",
        "",
    ]

    # --- Data composition ---
    lines += [
        "## Data Composition",
        "",
        "### Camera Bodies",
        "",
    ]
    if camera_counts.empty:
        lines.append("_No camera body data._\n")
    else:
        lines.append("| Camera | Count | % |")
        lines.append("|---|---|---|")
        for cam, cnt in camera_counts.items():
            lines.append(f"| {cam} | {cnt} | {100*cnt/n:.1f}% |")
        lines.append("")

    lines += ["### Camera Profiles", ""]
    if profile_counts.empty:
        lines.append("_No camera profile data._\n")
    else:
        if mixed_profiles:
            lines.append(
                "> **Warning:** Multiple camera profiles detected. "
                "Consider building separate profiles per camera body for better accuracy.\n"
            )
        lines.append("| Profile | Count | % |")
        lines.append("|---|---|---|")
        for prof, cnt in profile_counts.items():
            lines.append(f"| {prof} | {cnt} | {100*cnt/n:.1f}% |")
        lines.append("")

    lines += ["### ISO Distribution", ""]
    if iso_plot:
        lines.append("![ISO Distribution](plots/iso_distribution.png)\n")
    elif not iso_counts.empty:
        lines.append("| ISO | Count |")
        lines.append("|---|---|")
        for iso, cnt in iso_counts.head(15).items():
            lines.append(f"| {iso} | {cnt} |")
        lines.append("")
    else:
        lines.append("_No ISO data._\n")

    # --- Slider analysis ---
    lines += [
        "## Slider Analysis",
        "",
    ]
    if slider_plot:
        lines.append("![Slider Distributions](plots/slider_distributions.png)\n")

    lines += [
        "### Slider Statistics",
        "",
        "| Slider | Mean | Std Dev | Min | Max |",
        "|---|---|---|---|---|",
    ]
    for field in SLIDER_FIELDS:
        row = stats.loc[field]
        lines.append(
            f"| {field} | {row['mean']:.2f} | {row['std']:.2f} | {row['min']:.2f} | {row['max']:.2f} |"
        )
    lines.append("")

    if high_variance:
        lines += [
            "### High-Variance Sliders",
            "",
            f"These sliders have std dev > {_HIGH_VARIANCE_STD:.0f}, indicating inconsistent editing "
            f"that may be hard for the model to learn:",
            "",
        ]
        for field in high_variance:
            std = stats.loc[field, "std"]
            lines.append(f"- **{field}** (std={std:.1f})")
        lines.append("")

    # --- Quality flags ---
    lines += ["## Quality Flags", ""]

    if n_unedited > 0:
        lines += [
            f"### Likely Unedited Photos ({n_unedited})",
            "",
            "Photos where 30+ of 37 sliders are exactly 0.0 — these may not have "
            "been edited in Lightroom and will dilute the training signal.",
            "",
        ]
        if len(unedited_ids) <= 20:
            for uid in unedited_ids:
                raw = df.loc[df["id"] == uid, "raw_path"].values
                path_str = raw[0] if len(raw) > 0 else uid
                lines.append(f"- `{path_str}`")
        else:
            for uid in unedited_ids[:20]:
                raw = df.loc[df["id"] == uid, "raw_path"].values
                path_str = raw[0] if len(raw) > 0 else uid
                lines.append(f"- `{path_str}`")
            lines.append(f"- _...and {n_unedited - 20} more_")
        lines.append("")
    else:
        lines.append("No likely-unedited photos detected. Good.\n")

    if outliers:
        lines += [
            f"### Outlier Photos (>{_OUTLIER_STD:.0f} std devs from mean)",
            "",
            "Review these before training — they may represent intentional creative choices "
            "or extraction errors:",
            "",
        ]
        for field, ids in sorted(outliers.items()):
            lines.append(f"**{field}** — {len(ids)} photo(s):")
            for uid in ids[:5]:
                raw = df.loc[df["id"] == uid, "raw_path"].values
                path_str = raw[0] if len(raw) > 0 else uid
                val = df.loc[df["id"] == uid, field].values
                val_str = f"{val[0]:.2f}" if len(val) > 0 else "?"
                lines.append(f"  - `{path_str}` ({field}={val_str})")
            if len(ids) > 5:
                lines.append(f"  - _...and {len(ids)-5} more_")
            lines.append("")
    else:
        lines.append("No statistical outliers detected.\n")

    # --- Recommendations ---
    lines += ["## Recommendations", ""]

    recs: list[str] = []
    if n_photos := len(df):
        if n_photos < _STOP_MIN_PHOTOS:
            recs.append(f"**STOP**: Only {n_photos} photos — need at least {_STOP_MIN_PHOTOS} to train meaningfully. "
                        f"Add more edited photos.")
        elif n_photos < _WARN_MIN_PHOTOS:
            recs.append(f"**Consider adding more photos**: {n_photos} is workable but {_WARN_MIN_PHOTOS}+ is recommended "
                        f"for stable training.")

    if n_unedited > 0 and unedited_pct >= 10:
        recs.append(f"**Remove or re-edit unedited photos**: {n_unedited} photos ({unedited_pct:.0f}%) appear unedited. "
                    f"Export from Lightroom only photos that have been edited.")

    if mixed_profiles:
        recs.append("**Separate by camera profile**: Multiple camera profiles detected. Build one model profile "
                    "per camera body for more accurate results.")

    if high_variance:
        recs.append(f"**Review high-variance sliders**: {', '.join(high_variance[:5])} have very high std dev. "
                    f"This is expected for creative sliders (HSL) but unexpected for Exposure/Temperature.")

    if not recs:
        recs.append("No action needed — dataset is ready for training.")

    for rec in recs:
        lines.append(f"- {rec}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def audit_dataset(parquet_path: Path, output_dir: Path) -> dict:
    """Analyse a built dataset and produce a quality report.

    Writes audit_report.md and PNG plots to output_dir.
    Returns a summary dict with key metrics and the GO/WARN/STOP status.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    df = load_dataset(parquet_path)
    n = len(df)
    logger.info("Auditing %d photos from %s", n, parquet_path)

    # Core stats
    unedited_mask = _count_unedited(df)
    n_unedited = int(unedited_mask.sum())
    unedited_ids = df.loc[unedited_mask, "id"].tolist()
    unedited_ratio = n_unedited / n if n > 0 else 0.0

    outliers = _find_outliers(df)
    stats = _slider_stats(df)

    high_variance: list[str] = []
    for field in SLIDER_FIELDS:
        std_value = pd.to_numeric(pd.Series([stats.loc[field, "std"]]), errors="coerce").iloc[0]
        if pd.notna(std_value) and float(std_value) > _HIGH_VARIANCE_STD:
            high_variance.append(field)

    profile_counts = df["camera_profile"].value_counts() if "camera_profile" in df.columns else pd.Series(dtype=int)
    mixed_profiles = len(profile_counts) > 3

    status = _decide_status(n, unedited_ratio, mixed_profiles)
    training_minutes, training_label = _estimate_training_time(n)

    # Plots
    iso_plot = _plot_iso_distribution(df, plots_dir)
    slider_plot = _plot_slider_distributions(df, plots_dir)

    # Markdown report
    report_md = _build_report(
        df=df,
        stats=stats,
        status=status,
        n_unedited=n_unedited,
        unedited_ids=unedited_ids,
        outliers=outliers,
        high_variance=high_variance,
        mixed_profiles=mixed_profiles,
        training_minutes=training_minutes,
        training_label=training_label,
        iso_plot=iso_plot,
        slider_plot=slider_plot,
    )

    report_path = output_dir / "audit_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("Audit report written to %s", report_path)

    n_shoots = int(df["shoot_id"].nunique()) if "shoot_id" in df.columns else 0

    summary = {
        "status": status,
        "n_photos": n,
        "n_shoots": n_shoots,
        "n_unedited": n_unedited,
        "unedited_ratio": unedited_ratio,
        "n_outlier_sliders": len(outliers),
        "high_variance_sliders": high_variance,
        "mixed_profiles": mixed_profiles,
        "training_minutes_estimate": training_minutes,
        "report_path": str(report_path),
    }
    return summary
