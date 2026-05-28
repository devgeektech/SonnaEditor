from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.io import read_image

from sonna_editor.config import SLIDER_FIELDS
from sonna_editor.model.architecture import EmbeddingRegistry
from sonna_editor.model.augmentation import TrainingAugmentation, ValidationAugmentation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> float:
    """Return float value, or 0.0 for None / NaN."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if np.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def _cat_id(mapping: dict[str, int], val) -> int:
    """Look up a categorical string in the ID mapping; fall back to 0 (unknown)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return mapping.get("unknown", 0)
    return mapping.get(str(val), mapping.get("unknown", 0))


def _decode_histogram(data: bytes) -> torch.Tensor:
    """Decode a numpy-serialised (3, 32) histogram to a 96-d float tensor."""
    arr = np.load(io.BytesIO(data)).astype(np.float32)  # (3, 32)
    return torch.from_numpy(arr.flatten())               # (96,)


def build_registry(df: pd.DataFrame) -> EmbeddingRegistry:
    """Build an EmbeddingRegistry from the unique categorical values in df.

    Index 0 is always reserved for 'unknown' so that novel values at
    inference time get a stable fallback embedding rather than an OOB error.

    v1.1.0 added separate camera_makes / camera_models mappings. Older
    `camera_bodies` remains populated for backward compat with v1.0.x ckpts.
    """
    reg = EmbeddingRegistry()
    reg.camera_bodies = {"unknown": 0}
    reg.camera_makes = {"unknown": 0}
    reg.camera_models = {"unknown": 0}
    reg.lenses = {"unknown": 0}
    reg.camera_profiles = {"unknown": 0}
    reg.wb_presets = {"unknown": 0}

    col_map = [
        ("camera_body", reg.camera_bodies),
        ("make", reg.camera_makes),
        ("model", reg.camera_models),
        ("lens_model", reg.lenses),
        ("camera_profile", reg.camera_profiles),
        ("white_balance_preset", reg.wb_presets),
    ]
    for col, mapping in col_map:
        if col not in df.columns:
            continue
        for val in df[col].dropna().unique():
            s = str(val)
            if s not in mapping:
                mapping[s] = len(mapping)

    return reg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SonnaDataset(Dataset):
    """Maps a Parquet DataFrame row to (image, metadata, targets)."""

    def __init__(
        self,
        df: pd.DataFrame,
        transform: torch.nn.Module,
        registry: EmbeddingRegistry,
    ) -> None:
        self._df = df.reset_index(drop=True)
        self._transform = transform
        self._registry = registry
        self._log_problematic_rows()

    def _log_problematic_rows(self) -> None:
        """One-time startup scan: log rows with patterns that will likely be
        masked out by the loss layer. The dataset still returns these rows —
        loss-side masking handles the actual exclusion.

        Logs go to /tmp/saha_skipped_rows.log alongside the per-batch logs.
        """
        import time
        try:
            log_path = Path("/tmp/saha_skipped_rows.log")
            issues: dict[str, int] = {}
            # AsShot null/missing
            if "as_shot_temperature" in self._df.columns:
                ast = pd.to_numeric(self._df["as_shot_temperature"], errors="coerce")
                n_null = int(ast.isna().sum())
                if n_null > 0:
                    issues["as_shot_temperature_null"] = n_null
            # Truth Inf
            from sonna_editor.config import SLIDER_FIELDS as _F
            inf_truth = 0
            for f in _F:
                if f in self._df.columns:
                    col = pd.to_numeric(self._df[f], errors="coerce")
                    inf_truth += int(np.isinf(col.fillna(0)).sum())
            if inf_truth > 0:
                issues["inf_in_truth"] = inf_truth
            # All-zero truth (would suggest a row with no edits applied)
            tone_cols = [c for c in ("Exposure2012","Contrast2012","Highlights2012",
                                      "Shadows2012","Whites2012","Blacks2012")
                         if c in self._df.columns]
            if tone_cols:
                all_zero = (self._df[tone_cols].fillna(0).abs().sum(axis=1) == 0).sum()
                if all_zero > 0:
                    issues["all_zero_tone_truth"] = int(all_zero)

            if issues:
                with log_path.open("a") as f:
                    f.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} "
                            f"SonnaDataset startup scan ({len(self._df)} rows): {issues}\n")
        except Exception:
            # Never fail dataset init because of best-effort logging
            pass

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        row = self._df.iloc[idx]

        # --- Image ---
        thumb_path = str(row["thumbnail_path"])
        img = read_image(thumb_path)  # uint8 [C, H, W]
        if img.shape[0] > 3:
            img = img[:3]            # drop alpha if present
        image = self._transform(img) # float32 [3, H, W]

        # --- Metadata ---
        reg = self._registry

        # AsShot Temperature/Tint — present only on Stage-2-refreshed parquets.
        # Default to NaN when missing; loss masks NaN out of the bucket term.
        def _opt_float(col: str) -> float:
            v = row.get(col)
            if v is None:
                return float("nan")
            try:
                f = float(v)
                return f
            except (TypeError, ValueError):
                return float("nan")

        metadata: dict[str, torch.Tensor] = {
            "iso":            torch.tensor(_safe_float(row.get("iso")),            dtype=torch.float32),
            "shutter_speed":  torch.tensor(_safe_float(row.get("shutter_speed")), dtype=torch.float32),
            "aperture":       torch.tensor(_safe_float(row.get("aperture")),       dtype=torch.float32),
            "focal_length":   torch.tensor(_safe_float(row.get("focal_length")),   dtype=torch.float32),
            # camera_body kept for v1.0.x backward compat; v1.1.0 uses make/model.
            "camera_body_id": torch.tensor(_cat_id(reg.camera_bodies,   row.get("camera_body")),           dtype=torch.long),
            "camera_make_id":  torch.tensor(_cat_id(reg.camera_makes,   row.get("make")),                  dtype=torch.long),
            "camera_model_id": torch.tensor(_cat_id(reg.camera_models,  row.get("model")),                 dtype=torch.long),
            "lens_id":        torch.tensor(_cat_id(reg.lenses,          row.get("lens_model")),            dtype=torch.long),
            "camera_profile_id": torch.tensor(_cat_id(reg.camera_profiles, row.get("camera_profile")),    dtype=torch.long),
            "wb_preset_id":   torch.tensor(_cat_id(reg.wb_presets,      row.get("white_balance_preset")), dtype=torch.long),
            "histogram":      _decode_histogram(row["histogram"]),
            "as_shot_temperature": torch.tensor(_opt_float("as_shot_temperature"), dtype=torch.float32),
            "as_shot_tint":        torch.tensor(_opt_float("as_shot_tint"),        dtype=torch.float32),
            # Pass through the parquet's raw_path so the loss layer can log which
            # photos got masked out of loss math. Default collate turns these into
            # a list[str] of length B inside the batched metadata dict.
            "raw_path":            str(row.get("raw_path", "")),
        }

        # --- Targets ---
        target = torch.tensor(
            [
                float(row[f]) if (row.get(f) is not None and not pd.isna(row[f])) else float("nan")
                for f in SLIDER_FIELDS
            ],
            dtype=torch.float32,
        )

        return image, metadata, target


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class SonnaDataModule(pl.LightningDataModule):
    """Loads train/val/test Parquet splits and exposes DataLoaders.

    Usage:
        dm = SonnaDataModule(train_parquet=..., val_parquet=..., test_parquet=...)
        dm.setup("fit")
        model = SonnaEditor(registry=dm.registry)
    """

    def __init__(
        self,
        train_parquet: Path,
        val_parquet: Path,
        test_parquet: Path,
        batch_size: int = 16,
        num_workers: int = 4,
        sample_weight_col: Optional[str] = None,
        registry: Optional[EmbeddingRegistry] = None,
    ) -> None:
        super().__init__()
        self.train_parquet = Path(train_parquet)
        self.val_parquet   = Path(val_parquet)
        self.test_parquet  = Path(test_parquet)
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.sample_weight_col = sample_weight_col

        self.registry: Optional[EmbeddingRegistry] = registry
        self._train_ds: Optional[SonnaDataset] = None
        self._val_ds:   Optional[SonnaDataset] = None
        self._test_ds:  Optional[SonnaDataset] = None
        self._train_weights: Optional[list[float]] = None

    def prepare_data(self) -> None:
        for p in (self.train_parquet, self.val_parquet, self.test_parquet):
            if not p.exists():
                raise FileNotFoundError(f"Parquet file not found: {p}")

    def setup(self, stage: Optional[str] = None) -> None:
        df_train = pd.read_parquet(self.train_parquet)
        df_val   = pd.read_parquet(self.val_parquet)
        df_test  = pd.read_parquet(self.test_parquet)

        if self.registry is None:
            self.registry = build_registry(df_train)

        if self.sample_weight_col and self.sample_weight_col in df_train.columns:
            self._train_weights = df_train[self.sample_weight_col].fillna(1.0).tolist()

        train_aug = TrainingAugmentation()
        val_aug   = ValidationAugmentation()

        self._train_ds = SonnaDataset(df_train, train_aug, self.registry)
        self._val_ds   = SonnaDataset(df_val,   val_aug,   self.registry)
        self._test_ds  = SonnaDataset(df_test,  val_aug,   self.registry)

    def train_dataloader(self) -> DataLoader:
        weights = self._train_weights
        if weights is not None:
            first = weights[0]
            use_sampler = any(w != first for w in weights)
        else:
            use_sampler = False

        if use_sampler:
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            return DataLoader(
                self._train_ds,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                pin_memory=False,
            )
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=False,  # MPS doesn't support pinned memory
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=False,
        )
