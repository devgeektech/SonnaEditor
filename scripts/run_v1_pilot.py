#!/usr/bin/env python
"""Stage 1 pilot training — up to 30,000 most recent edited photos.

Runs prerequisite checks, builds dataset from Lightroom catalog, pauses for
your approval, then kicks off training with alerting callbacks active.

Usage:
    uv run scripts/run_v1_pilot.py
    uv run scripts/run_v1_pilot.py --max-epochs 30 --batch-size 16 --workers 4

Prerequisites:
  - Source photo storage must be mounted and readable.
  - Lightroom Classic must be closed (catalog must not be locked).
  - At least 20 GB free disk space.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from sonna_editor.config import SLIDER_FIELDS
from sonna_editor.data.catalog import (
    CatalogError,
    CatalogLockedError,
    connect_catalog,
    find_edited_photos,
)
from sonna_editor.data.catalog_dataset import build_dataset_from_catalog
from sonna_editor.data.dataset import save_split, split_dataset
from sonna_editor.model.architecture import SonnaEditor
from sonna_editor.model.postprocess import postprocess_predictions
from sonna_editor.runtime import preferred_lightning_accelerator, preferred_torch_device
from sonna_editor.training.callbacks import (
    CriticalMAECallback,
    DiskSpaceCallback,
    ETACallback,
    NaNLossCallback,
    OverfittingCallback,
)
from sonna_editor.training.datamodule import SonnaDataModule
from sonna_editor.training.module import SonnaLightningModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — all configurable here
# ---------------------------------------------------------------------------

CATALOG_PATH = Path.home() / "sonnaeditor/test_data/sonnaeditor08_05/sonnaeditor08_05.lrcat"
SSD_PATH = Path("/Volumes/25-08/")
V1_DIR = Path.home() / "sonnaeditor/v1_learning"
DATASET_DIR = V1_DIR / "dataset"
THUMBNAIL_DIR = V1_DIR / "thumbnails"
CHECKPOINT_DIR = V1_DIR / "checkpoints"
LOG_DIR = V1_DIR / "logs"
PREDICTION_DIR = V1_DIR / "predictions"

# All directories the script will write to — checked against local mount at startup
_WRITE_DIRS = [V1_DIR, DATASET_DIR, THUMBNAIL_DIR, CHECKPOINT_DIR, LOG_DIR, PREDICTION_DIR]

MIN_FREE_GB = 20.0
MIN_PHOTOS = 500      # hard-fail only on "nothing at all accessible" — pilot proceeds with whatever is mounted
PILOT_LIMIT = 30_000


# ---------------------------------------------------------------------------
# Belt-and-braces: external volume write protection
# ---------------------------------------------------------------------------

def _mount_point(path: Path) -> str:
    """Walk up to the mount point of path (the nearest ancestor that is_mount())."""
    p = path.resolve() if path.exists() else path
    # Walk to an existing ancestor for is_mount() to work
    while not p.exists():
        p = p.parent
    while not p.is_mount():
        p = p.parent
    return str(p)


def _assert_no_external_writes() -> None:
    """Hard-fail if any write target resolves to an external/removable volume.

    Uses the mount point of Path.home() as the definition of 'local disk'.
    Any write directory that resolves to a different mount is refused.
    This makes it physically impossible to write to an SSD even via a future bug.
    """
    local_mount = _mount_point(Path.home())
    violations: list[str] = []

    for write_dir in _WRITE_DIRS:
        mount = _mount_point(write_dir)
        if mount != local_mount:
            violations.append(
                f"  {write_dir}\n"
                f"    → mount point: {mount!r} (not local disk {local_mount!r})"
            )

    if violations:
        print("\n🚨 SAFETY ABORT: write target(s) resolve to external volume(s):")
        for v in violations:
            print(v)
        print("\nAll training outputs must stay on the local drive.")
        print("Edit V1_DIR in this script to point to a local path.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Sleep prevention (caffeinate)
# ---------------------------------------------------------------------------

_caffeinate_proc: subprocess.Popen | None = None
_caffeinate_active: bool = False


def setup_caffeinate() -> bool:
    """Detect or spawn caffeinate. Returns True if sleep prevention is active.

    If the user already has caffeinate running we use that.  If we spawn our
    own process we do NOT register an atexit handler — on crash the process
    stays alive intentionally so the laptop doesn't sleep while the user
    investigates.  teardown_caffeinate() kills it on clean completion only.
    """
    global _caffeinate_proc, _caffeinate_active

    try:
        r = subprocess.run(
            ["pmset", "-g", "assertions"], capture_output=True, text=True, timeout=5
        )
        if "caffeinate" in r.stdout:
            log.info("User caffeinate detected — using that (not spawning a new one)")
            _caffeinate_active = True
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        _caffeinate_proc = subprocess.Popen(
            ["caffeinate", "-i", "-m", "-d", "-s"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("caffeinate spawned (PID %d)", _caffeinate_proc.pid)
        _caffeinate_active = True
        return True
    except FileNotFoundError:
        log.warning("caffeinate not found — training may sleep on idle")
        _caffeinate_active = False
        return False


def teardown_caffeinate() -> None:
    """Terminate our spawned caffeinate on normal completion only.

    Called explicitly at the end of main() — NOT via atexit — so a crash
    leaves caffeinate running while the user investigates.
    """
    global _caffeinate_proc, _caffeinate_active
    if _caffeinate_proc is not None:
        _caffeinate_proc.terminate()
        log.info("caffeinate terminated (normal completion)")
        _caffeinate_proc = None
    _caffeinate_active = False


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "─", width: int = 68) -> None:
    print(char * width)


def _header(title: str) -> None:
    _sep("═")
    print(f"  {title}")
    _sep("═")


# ---------------------------------------------------------------------------
# Step 1: Prerequisite checks
# ---------------------------------------------------------------------------

def check_prerequisites() -> bool:
    """Run all prerequisite checks. Returns True only if all hard checks pass."""
    _header("PREREQUISITE CHECKS")

    results: list[tuple[str, bool, bool, str]] = []
    # (label, passed, is_hard_fail, message)

    # 1. SSD mounted and readable
    try:
        if SSD_PATH.exists() and SSD_PATH.is_dir():
            next(SSD_PATH.iterdir(), None)
            results.append(("SSD mounted & readable", True, True, str(SSD_PATH)))
        else:
            results.append(("SSD mounted & readable", False, True,
                            f"{SSD_PATH} not found — mount the drive and retry"))
    except (PermissionError, OSError) as e:
        results.append(("SSD mounted & readable", False, True, f"Error: {e}"))

    # 2. Catalog openable and ≥30K accessible edited photos
    catalog_ok = False
    try:
        conn = connect_catalog(CATALOG_PATH)
        photos = find_edited_photos(conn)
        conn.close()

        # Quick file-existence check (uses path from catalog)
        accessible_edited = [
            p for p in photos
            if p["has_develop_settings"] and p["file_path"].exists()
        ]
        n = len(accessible_edited)
        catalog_ok = n >= MIN_PHOTOS
        results.append((
            "Accessible edited photos",
            catalog_ok,
            True,
            f"{n:,} found on mounted volumes" + (
                "" if catalog_ok
                else f" — fewer than {MIN_PHOTOS:,} (catalog or mount issue?)"
            ),
        ))
    except CatalogLockedError as e:
        results.append(("Catalog accessible", False, True,
                        f"Catalog is locked — close Lightroom Classic: {e}"))
    except CatalogError as e:
        results.append(("Catalog accessible", False, True, f"Catalog error: {e}"))

    # 3. Free disk space
    try:
        usage = shutil.disk_usage(Path.home())
        free_gb = usage.free / (1024 ** 3)
        ok = free_gb >= MIN_FREE_GB
        results.append((
            f"≥{MIN_FREE_GB:.0f} GB free disk",
            ok,
            True,
            f"{free_gb:.1f} GB free" + (
                "" if ok else f" — need {MIN_FREE_GB:.0f} GB, free up space first"
            ),
        ))
    except OSError as e:
        results.append(("Disk space check", False, True, str(e)))

    # 4. Sleep prevention (caffeinate managed by main() before this call)
    results.append((
        "Sleep prevention",
        _caffeinate_active,
        False,
        "caffeinate ACTIVE" if _caffeinate_active
        else "caffeinate not available — risk of sleep mid-training (non-fatal)",
    ))

    # 5. Time Machine
    try:
        r = subprocess.run(
            ["tmutil", "status"], capture_output=True, text=True, timeout=5
        )
        running = "Running = 1" in r.stdout
        results.append((
            "Time Machine",
            not running,
            False,
            "idle" if not running
            else "backup running — pause it (tmutil stopbackup) for cleaner performance",
        ))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        results.append(("Time Machine", True, False, "tmutil unavailable — skipped"))

    # 6. Power source
    try:
        r = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5
        )
        on_battery = "Battery Power" in r.stdout
        results.append((
            "Power source",
            not on_battery,
            False,
            "AC power" if not on_battery
            else "running on battery — plug in to avoid GPU throttling and crash on power loss",
        ))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        results.append(("Power source", True, False, "check unavailable — skipped"))

    # 7. Heavy apps (Lightroom, Photos, Final Cut)
    heavy_running: list[str] = []
    for display_name, pattern in [
        ("Lightroom Classic", "Adobe Lightroom"),
        ("Photos.app", "Photos"),
        ("Final Cut Pro", "Final Cut Pro"),
    ]:
        try:
            r = subprocess.run(
                ["pgrep", "-i", "-l", pattern],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                heavy_running.append(display_name)
        except FileNotFoundError:
            break
    if heavy_running:
        results.append((
            "Heavy apps",
            False,
            False,
            f"{', '.join(heavy_running)} running — close them for best GPU performance",
        ))
    else:
        results.append(("Heavy apps", True, False, "none detected"))

    # 8. Screen saver idle time
    try:
        r = subprocess.run(
            ["defaults", "read", "com.apple.screensaver", "idleTime"],
            capture_output=True, text=True, timeout=5,
        )
        raw = r.stdout.strip()
        idle_time = int(raw) if raw.lstrip("-").isdigit() else 0
        if idle_time > 0:
            idle_min = idle_time // 60
            results.append((
                "Screen saver",
                False,
                False,
                f"activates after {idle_min} min — caffeinate should block it, but verify",
            ))
        else:
            results.append(("Screen saver", True, False, "disabled or set to never"))
    except Exception:
        results.append(("Screen saver", True, False, "check skipped"))

    # Print results
    all_hard_pass = True
    for label, ok, is_hard, msg in results:
        icon = "✅" if ok else ("❌" if is_hard else "⚠️ ")
        print(f"  {icon} {label}: {msg}")
        if not ok and is_hard:
            all_hard_pass = False

    print()
    if not all_hard_pass:
        print("One or more prerequisites failed. Fix the issues above and retry.")
    else:
        print("All prerequisites passed.")
        if _caffeinate_active:
            print("  Sleep prevention: ACTIVE")
    _sep()
    return all_hard_pass


# ---------------------------------------------------------------------------
# Step 2: Dataset build
# ---------------------------------------------------------------------------

def build_and_split_dataset(workers: int) -> tuple[Path, Path, Path]:
    """Build Parquet dataset from catalog, split it, return (train, val, test) paths."""
    _header("BUILDING DATASET FROM CATALOG")
    print(f"  Source catalog: {CATALOG_PATH}")
    print(f"  Dataset dir:    {DATASET_DIR}")
    print(f"  Thumbnails:     {THUMBNAIL_DIR}")
    print(f"  Limit:          {PILOT_LIMIT:,} most recent edited photos")
    print(f"  Workers:        {workers}")
    print()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = DATASET_DIR / "dataset.parquet"

    df, stats = build_dataset_from_catalog(
        catalog_path=CATALOG_PATH,
        output_path=parquet_path,
        profile_name="sonna_v1",
        thumbnail_dir=THUMBNAIL_DIR,
        limit=PILOT_LIMIT,
        max_workers=workers,
    )

    _sep()
    print("DATASET BUILD RESULTS")
    _sep()
    print(f"  Total photos in catalog:           {stats['total_in_catalog']:>8,}")
    print(f"  With develop settings:             {stats['total_with_develop_settings']:>8,}")
    print(f"  Skipped — virtual copy dupes:      {stats['skip_virtual_copy']:>8,}")
    print(f"  Skipped — RAW not on SSD:          {stats['skip_missing']:>8,}")
    print(f"  Skipped — appears unedited:        {stats['skip_unedited']:>8,}")
    print(f"  Skipped — develop parse error:     {stats['skip_parse_error']:>8,}")
    print(f"  Skipped — thumbnail extract error: {stats['skip_extraction_error']:>8,}")
    print(f"  INCLUDED IN DATASET:               {stats['included']:>8,}")
    print()

    # Date range and per-year breakdown
    cap_dates = pd.to_datetime(df["capture_datetime"], errors="coerce").dropna()
    if not cap_dates.empty:
        print(f"  Date range: {cap_dates.min().date()} → {cap_dates.max().date()}")
        year_counts = cap_dates.dt.year.value_counts().sort_index()
        print("  Photos per year:")
        for year, count in year_counts.items():
            bar = "█" * min(40, count // max(1, year_counts.max() // 40))
            print(f"    {year}  {count:>6,}  {bar}")
    print()

    # Slider sanity check — key fields only
    key_fields = [
        "Exposure2012", "Temperature", "Tint",
        "Shadows2012", "Highlights2012", "Whites2012", "Blacks2012",
        "Contrast2012", "Clarity2012", "Vibrance", "Saturation",
    ]
    print("  Slider value distributions (sanity check):")
    print(f"    {'Slider':<22}  {'Mean':>7}  {'Std':>7}  {'Min':>7}  {'Max':>7}  {'%None':>6}")
    print(f"    {'─'*22}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}")
    for field in key_fields:
        if field not in df.columns:
            continue
        col = df[field]
        none_pct = 100.0 * col.isna().sum() / max(1, len(col))
        col_f = col.dropna().astype(float)
        if col_f.empty:
            print(f"    {field:<22}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}  {none_pct:>5.1f}%")
        else:
            print(
                f"    {field:<22}  {col_f.mean():>7.2f}  {col_f.std():>7.2f}  "
                f"{col_f.min():>7.2f}  {col_f.max():>7.2f}  {none_pct:>5.1f}%"
            )
    print()

    # Shoot-level train/val/test split
    splits_dir = DATASET_DIR / "splits"
    train_df, val_df, test_df = split_dataset(df)
    save_split(train_df, val_df, test_df, splits_dir)

    n = stats["included"]
    print("  Shoot-level splits (val+test shoots withheld whole):")
    print(f"    train: {len(train_df):,} ({100*len(train_df)/n:.0f}%)")
    print(f"    val:   {len(val_df):,} ({100*len(val_df)/n:.0f}%)")
    print(f"    test:  {len(test_df):,} ({100*len(test_df)/n:.0f}%)")
    print()
    _sep()

    return (
        splits_dir / "train.parquet",
        splits_dir / "val.parquet",
        splits_dir / "test.parquet",
    )


# ---------------------------------------------------------------------------
# Step 3: Pause
# ---------------------------------------------------------------------------

def wait_for_approval() -> None:
    _sep("═")
    print("PAUSE — dataset built, awaiting approval to start training")
    _sep("═")
    print("Review the dataset statistics above.")
    print("If everything looks correct, press Enter to begin training.")
    print("Press Ctrl+C to exit — the dataset will be saved and can be used later.")
    print()
    try:
        input("  → Press Enter to start training: ")
    except KeyboardInterrupt:
        print("\n\nExiting. Dataset is saved at:")
        print(f"  {DATASET_DIR}")
        print("Resume training later with scripts/train_profile.py using those splits.")
        sys.exit(0)
    except EOFError:
        # Non-interactive context (stdout redirected, background job, etc.)
        # Exit cleanly so the user can review the dataset and start training manually.
        print("\n\nPAUSE — non-interactive mode detected. Exiting at dataset review point.")
        print("Dataset is saved. Start training manually when ready:")
        print("  uv run scripts/run_v1_pilot.py --skip-dataset \\")
        print(f"    --train-parquet {DATASET_DIR / 'splits/train.parquet'} \\")
        print(f"    --val-parquet {DATASET_DIR / 'splits/val.parquet'} \\")
        print(f"    --test-parquet {DATASET_DIR / 'splits/test.parquet'}")
        sys.exit(0)


# ---------------------------------------------------------------------------
# Heartbeat callback (30-min interval)
# ---------------------------------------------------------------------------

class HeartbeatCallback(pl.Callback):
    """Log a status line every N minutes: caffeinate status, epoch, loss trend, ETA."""

    def __init__(self, interval_minutes: float = 30.0) -> None:
        self._interval_sec = interval_minutes * 60
        self._last_heartbeat: float = 0.0
        self._train_start: float = 0.0
        self._val_losses: list[float] = []

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._train_start = time.monotonic()
        self._last_heartbeat = self._train_start

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self._val_losses.append(float(val_loss))

        now = time.monotonic()
        if now - self._last_heartbeat < self._interval_sec:
            return
        self._last_heartbeat = now

        elapsed_sec = now - self._train_start
        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs or 0
        avg_epoch_sec = elapsed_sec / max(1, epoch)
        remaining = max(0, max_epochs - epoch)
        eta_h = (avg_epoch_sec * remaining) / 3600

        if len(self._val_losses) >= 2:
            trend_str = f"{self._val_losses[-1] - self._val_losses[-2]:+.4f}"
        else:
            trend_str = "—"

        caff_status = "unknown"
        try:
            r = subprocess.run(
                ["pmset", "-g", "assertions"], capture_output=True, text=True, timeout=5
            )
            caff_status = "ACTIVE" if "caffeinate" in r.stdout else "NOT ACTIVE ⚠️"
        except Exception:
            caff_status = "check failed"

        print(
            f"\n[HEARTBEAT {time.strftime('%H:%M:%S')}]  "
            f"epoch {epoch}/{max_epochs}  |  "
            f"elapsed {elapsed_sec / 3600:.1f}h  |  ETA {eta_h:.1f}h  |  "
            f"val_loss trend {trend_str}  |  "
            f"sleep prevention: {caff_status}"
        )


# ---------------------------------------------------------------------------
# Step 4: Training
# ---------------------------------------------------------------------------

def run_training(
    train_parquet: Path,
    val_parquet: Path,
    test_parquet: Path,
    max_epochs: int,
    batch_size: int,
    workers: int,
    run_label: str = "v1.0.1",
) -> tuple[dict, str | None]:
    """Set up and run training. Returns (summary dict, best_checkpoint_path)."""
    _header(f"TRAINING  (max {max_epochs} epochs, batch {batch_size})")
    print(f"  Checkpoints → {CHECKPOINT_DIR}")
    print(f"  Logs        → {LOG_DIR}")
    print()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    # DataModule
    dm = SonnaDataModule(
        train_parquet=train_parquet,
        val_parquet=val_parquet,
        test_parquet=test_parquet,
        batch_size=batch_size,
        num_workers=workers,
        slider_set_version="v1",
    )
    dm.prepare_data()
    dm.setup("fit")

    log.info(
        "Dataset: train=%d  val=%d  test=%d",
        len(dm._train_ds), len(dm._val_ds), len(dm._test_ds),
    )
    reg = dm.registry
    log.info(
        "Registry: %d camera bodies  %d lenses  %d profiles  %d WB presets",
        len(reg.camera_bodies), len(reg.lenses),
        len(reg.camera_profiles), len(reg.wb_presets),
    )

    # Model
    model = SonnaEditor(
        registry=reg,
        freeze_backbone=True,
        slider_set_version="v1",
        use_wb_metadata_skip=False,
    )
    module = SonnaLightningModule(
        model=model,
        lr=3e-4,
        weight_decay=1e-4,
        freeze_backbone_epochs=5,
    )
    log.info("Backbone frozen for first 5 epochs, then unfrozen at 1/10 LR.")

    # Callbacks
    ckpt_callback = ModelCheckpoint(
        dirpath=str(CHECKPOINT_DIR),
        filename=f"sonna-{run_label}-epoch{{epoch:03d}}-val{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        auto_insert_metric_name=False,
    )

    callbacks = [
        ckpt_callback,
        EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
        # Alerting
        NaNLossCallback(consecutive_threshold=2),
        OverfittingCallback(patience=5),
        DiskSpaceCallback(watch_path=V1_DIR, min_free_gb=5.0),
        ETACallback(max_hours=4.0, check_after_epoch=3),
        CriticalMAECallback(check_after_epoch=5),
        HeartbeatCallback(interval_minutes=30.0),
    ]

    # Loggers
    tb_logger = TensorBoardLogger(save_dir=str(LOG_DIR), name="tensorboard")
    csv_logger = CSVLogger(save_dir=str(LOG_DIR), name="csv")

    # Trainer: choose CUDA, MPS, or CPU at runtime so this pilot can run on any OS.
    trainer = pl.Trainer(
        accelerator=preferred_lightning_accelerator(),
        devices=1,
        precision="32-true",
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=[tb_logger, csv_logger],
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=1.0,   # bounds step size; paired with Temperature weight increase
    )

    # Train
    start_time = time.monotonic()
    try:
        trainer.fit(module, datamodule=dm)
    except Exception as e:
        print(f"\n🚨 ALERT: Training crashed with exception: {type(e).__name__}: {e}")
        raise

    total_sec = time.monotonic() - start_time
    best_ckpt = trainer.checkpoint_callback.best_model_path or None

    # Test on best checkpoint
    log.info("Running test set evaluation on best checkpoint...")
    if best_ckpt:
        test_results = trainer.test(module, datamodule=dm, ckpt_path=best_ckpt)
    else:
        test_results = trainer.test(module, datamodule=dm)

    # Save best model using run label
    final_model_path = V1_DIR / f"model-{run_label}.ckpt"
    if best_ckpt:
        shutil.copy(best_ckpt, final_model_path)
        log.info("Saved best checkpoint as %s", final_model_path)

    # Sample predictions on 5 val photos
    sample_preds = _sample_predictions(module, dm, n=5)

    summary = {
        "best_val_loss": float(trainer.checkpoint_callback.best_model_score or 0.0),
        "best_checkpoint": best_ckpt,
        "final_model": str(final_model_path),
        "epochs_trained": int(trainer.current_epoch) + 1,
        "total_training_minutes": round(total_sec / 60, 1),
        "test_results": test_results[0] if test_results else {},
        "sample_predictions": sample_preds,
        "hparams": {
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "freeze_backbone_epochs": 5,
        },
    }

    summary_path = V1_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Training summary written to %s", summary_path)

    return summary, best_ckpt


def _sample_predictions(
    module: SonnaLightningModule,
    dm: SonnaDataModule,
    n: int = 5,
) -> list[dict]:
    """Run inference on n validation photos and return prediction dicts."""
    key_fields = [
        "Exposure2012", "Temperature", "Shadows2012", "Highlights2012",
        "Contrast2012", "Whites2012", "Blacks2012", "Saturation", "Vibrance",
    ]
    results = []

    device = torch.device(preferred_torch_device())
    module.model.eval()
    module.model.to(device)

    val_ds = dm._val_ds
    indices = list(range(min(n, len(val_ds))))

    for idx in indices:
        try:
            img, metadata, target = val_ds[idx]
            img_batch = img.unsqueeze(0).to(device)
            meta_batch = {k: v.unsqueeze(0).to(device) for k, v in metadata.items()}

            with torch.no_grad():
                raw_pred = module.model(img_batch, meta_batch)
                pred = postprocess_predictions(raw_pred)

            pred_np = pred.cpu().squeeze(0).numpy()
            target_np = target.numpy()
            raw_path = val_ds._df.iloc[idx]["raw_path"]

            per_field: dict[str, dict] = {}
            for field in key_fields:
                fidx = SLIDER_FIELDS.index(field)
                act = float(target_np[fidx])
                prd = float(pred_np[fidx])
                import math
                per_field[field] = {
                    "actual": None if math.isnan(act) else round(act, 3),
                    "predicted": round(prd, 3),
                    "error": None if math.isnan(act) else round(prd - act, 3),
                }

            results.append({
                "photo": Path(raw_path).name,
                "val_index": idx,
                "fields": per_field,
            })
        except Exception as e:
            log.warning("Could not generate prediction for val[%d]: %s", idx, e)

    return results


# ---------------------------------------------------------------------------
# Step 5: Summary
# ---------------------------------------------------------------------------

def print_training_summary(summary: dict, best_ckpt: str | None) -> None:
    _header("TRAINING COMPLETE — SUMMARY")

    total_min = summary["total_training_minutes"]
    print(f"  Epochs run:       {summary['epochs_trained']}")
    print(f"  Total time:       {total_min:.1f} min  ({total_min / 60:.1f} h)")
    print(f"  Best val loss:    {summary['best_val_loss']:.4f}")
    print(f"  Best checkpoint:  {summary.get('best_checkpoint', 'N/A')}")
    print(f"  Final model:      {summary.get('final_model', 'N/A')}")
    print(f"  Summary JSON:     {V1_DIR / 'training_summary.json'}")
    print()

    test = summary.get("test_results", {})
    if test:
        print("  Test set results:")
        for key, val in sorted(test.items()):
            if isinstance(val, (int, float)):
                print(f"    {key}: {val:.4f}")
        print()

    # Sample predictions
    samples = summary.get("sample_predictions", [])
    if samples:
        _sep()
        print("SAMPLE PREDICTIONS (5 validation photos)")
        _sep()
        for s in samples:
            print(f"\n  {s['photo']}")
            print(f"    {'Slider':<22}  {'Actual':>9}  {'Predicted':>9}  {'Error':>9}")
            print(f"    {'─'*22}  {'─'*9}  {'─'*9}  {'─'*9}")
            for field, vals in s["fields"].items():
                act_str = f"{vals['actual']:>9.2f}" if vals["actual"] is not None else f"{'N/A':>9}"
                err_str = f"{vals['error']:>+9.2f}" if vals["error"] is not None else f"{'N/A':>9}"
                print(f"    {field:<22}  {act_str}  {vals['predicted']:>9.2f}  {err_str}")

    print()
    _sep("═")
    print("Training complete. Awaiting next instruction.")
    _sep("═")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1 pilot training — 30,000 most recent edited photos"
    )
    p.add_argument("--max-epochs", type=int, default=30,
                   help="Maximum training epochs (default: 30)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Training batch size (default: 16)")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader / thumbnail worker count (default: 4)")
    p.add_argument("--skip-dataset", action="store_true",
                   help="Skip dataset build and use existing splits in DATASET_DIR/splits/")
    p.add_argument("--run-label", type=str, default="v1.0.1",
                   help="Version label for checkpoint filenames, e.g. v1.0.1 (default: v1.0.1)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Safety check — abort if any write target is on an external volume
    _assert_no_external_writes()

    # Start sleep prevention before anything else (before prereq checks print it)
    setup_caffeinate()

    # Step 1 — Prerequisites (hard fail if any required check fails)
    if not check_prerequisites():
        sys.exit(1)

    # Step 2 — Build dataset (or reuse existing splits)
    if args.skip_dataset:
        splits_dir = DATASET_DIR / "splits"
        train_p = splits_dir / "train.parquet"
        val_p = splits_dir / "val.parquet"
        test_p = splits_dir / "test.parquet"
        missing = [p for p in (train_p, val_p, test_p) if not p.exists()]
        if missing:
            print("\n🚨 --skip-dataset: missing split files:")
            for m in missing:
                print(f"  {m}")
            print("Run without --skip-dataset to build the dataset first.")
            sys.exit(1)
        _header("USING EXISTING DATASET SPLITS")
        for label, p in [("train", train_p), ("val", val_p), ("test", test_p)]:
            rows = len(pd.read_parquet(p, columns=["id"]))
            print(f"  {label:<5} {p}  ({rows:,} rows)")
        print()
        _sep()
    else:
        train_p, val_p, test_p = build_and_split_dataset(args.workers)

    # Step 3 — Pause for approval
    wait_for_approval()

    # Step 4 — Training (caffeinate stays alive through crash; torn down only on success)
    summary, best_ckpt = run_training(
        train_parquet=train_p,
        val_parquet=val_p,
        test_parquet=test_p,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        run_label=args.run_label,
    )

    # Step 5 — Summary
    print_training_summary(summary, best_ckpt)

    # Normal completion — safe to release sleep prevention now
    teardown_caffeinate()


if __name__ == "__main__":
    main()
