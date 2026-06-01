"""Inference engine — loads a trained SonnaEditor checkpoint and predicts slider values."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

from sonna_editor import config
from sonna_editor.data.extract import compute_histogram, compute_scene_statistics
from sonna_editor.model.architecture import EmbeddingRegistry, SonnaEditor
from sonna_editor.model.augmentation import ValidationAugmentation
from sonna_editor.model.postprocess import postprocess_predictions, predictions_to_dict
from sonna_editor.runtime import preferred_torch_device


def _load_from_checkpoint(path: Path, device: str) -> SonnaEditor:
    """Load SonnaEditor from a Lightning checkpoint or a native checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        # Lightning checkpoint — strip "model." prefix from state dict
        raw_state = ckpt["state_dict"]
        model_state = {
            k[len("model."):]: v
            for k, v in raw_state.items()
            if k.startswith("model.")
        }

        # Lightning ckpts don't carry our arch_config field; detect arch from
        # the state_dict keys. v1.1.0 has make_emb (the canonical v1.1.0
        # marker — distinct from v1.0.x's body_emb).
        if "metadata_encoder.scene_stats_mlp.0.weight" in model_state:
            arch_version = 2
        elif "metadata_encoder.make_emb.weight" in model_state:
            arch_version = 1
        else:
            arch_version = 0

        embedding_sizes: dict[str, int] = {
            "num_lenses":     model_state["metadata_encoder.lens_emb.weight"].shape[0],
            "num_profiles":   model_state["metadata_encoder.profile_emb.weight"].shape[0],
            "num_wb_presets": model_state["metadata_encoder.wb_emb.weight"].shape[0],
        }
        if arch_version == 0:
            embedding_sizes["num_bodies"] = model_state["metadata_encoder.body_emb.weight"].shape[0]
        else:
            embedding_sizes["num_makes"]  = model_state["metadata_encoder.make_emb.weight"].shape[0]
            embedding_sizes["num_models"] = model_state["metadata_encoder.model_emb.weight"].shape[0]

        # Lightning ckpts predate the v2 slider_set_version flag — they're all
        # v1 (135-output, 13 heads). Pin to "v1" so SonnaEditor's v2 default
        # (commit 78511ce) doesn't introduce the 5 extension heads whose
        # weights aren't in the Lightning state_dict.
        model = SonnaEditor(
            registry=EmbeddingRegistry(),
            _embedding_sizes=embedding_sizes,
            _pretrained_backbone=False,
            arch_version=arch_version,
            slider_set_version="v1",
            use_wb_metadata_skip=False,
        )
        model.load_state_dict(model_state)
    else:
        # Native checkpoint (saved via SonnaEditor.save_checkpoint())
        model = SonnaEditor.from_checkpoint(path, device="cpu")

    model.eval()
    model.to(device)
    return model


def _enable_dropout(model: nn.Module) -> None:
    """Set only Dropout layers to training mode for MC dropout sampling."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def _safe_float(val) -> float:
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


class InferenceEngine:
    """
    Loads a trained SonnaEditor checkpoint and runs batched inference.

    Accepts either Lightning checkpoints (e.g. from ModelCheckpoint callback or the
    copied model-v1.0.x.ckpt files) or native SonnaEditor checkpoints.

    Usage:
        engine = InferenceEngine("v1_learning/model-v1.0.1.ckpt")
        engine.warmup()
        preds = engine.predict(images, metadata_list)  # [N, 135] slider values
    """

    def __init__(
        self,
        model_path: Path | str,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = preferred_torch_device()
        self._device = device
        # Resolution resolution order:
        #   1. Sidecar JSON {ckpt_basename}.json `resolution` field (preferred —
        #      explicit per-profile record, written when the ckpt is registered)
        #   2. Native ckpt's arch_config.image_resolution (set by training script)
        #   3. Lightning ckpt heuristic by arch_version (v1.1.x → 512, else 384)
        #   4. Global config.IMAGE_RESOLUTION (fallback for legacy ckpts)
        # The augmentation MUST be constructed with this same resolution so the
        # input to the model matches what it was trained on. Without the explicit
        # arg, ValidationAugmentation would bind to the global at module-load
        # time and silently upscale/downscale every preview to a wrong size.
        model_path = Path(model_path)
        sidecar_path = model_path.with_suffix(".json")
        sidecar_resolution: Optional[int] = None
        if sidecar_path.exists():
            try:
                sidecar = json.loads(sidecar_path.read_text())
                if "resolution" in sidecar:
                    sidecar_resolution = int(sidecar["resolution"])
            except (json.JSONDecodeError, OSError, ValueError, TypeError):
                sidecar_resolution = None

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        arch_cfg = ckpt.get("arch_config") or {}
        is_lightning = "state_dict" in ckpt
        if sidecar_resolution is not None:
            self._image_resolution = sidecar_resolution
        elif is_lightning:
            has_v1_1 = any(
                k.endswith("metadata_encoder.make_emb.weight")
                for k in ckpt["state_dict"].keys()
            )
            self._image_resolution = 512 if has_v1_1 else 384
        else:
            self._image_resolution = int(arch_cfg.get("image_resolution") or config.IMAGE_RESOLUTION)
        self._model = _load_from_checkpoint(model_path, device)
        self._transform = ValidationAugmentation(resolution=self._image_resolution)

    def warmup(self) -> None:
        """Run a dummy forward pass so the selected backend is ready for inference."""
        res = self._image_resolution
        dummy_img = torch.zeros(1, 3, res, res, device=self._device)
        dummy_meta: dict[str, torch.Tensor] = {
            "iso":               torch.zeros(1, device=self._device),
            "shutter_speed":     torch.zeros(1, device=self._device),
            "aperture":          torch.zeros(1, device=self._device),
            "focal_length":      torch.zeros(1, device=self._device),
            "camera_body_id":    torch.zeros(1, dtype=torch.long, device=self._device),
            "camera_make_id":    torch.zeros(1, dtype=torch.long, device=self._device),
            "camera_model_id":   torch.zeros(1, dtype=torch.long, device=self._device),
            "lens_id":           torch.zeros(1, dtype=torch.long, device=self._device),
            "camera_profile_id": torch.zeros(1, dtype=torch.long, device=self._device),
            "wb_preset_id":      torch.zeros(1, dtype=torch.long, device=self._device),
            "histogram":         torch.zeros(1, 96, device=self._device),
            "scene_stats":       torch.zeros(1, 6, device=self._device),
            # v1.1.0+ inputs; v1.0.x models ignore them.
            "as_shot_temperature": torch.full((1,), 5500.0, device=self._device),
            "as_shot_tint":        torch.zeros(1, device=self._device),
        }
        with torch.no_grad():
            self._model(dummy_img, dummy_meta)

    def _build_batch(
        self,
        images: list[Image.Image],
        metadata_list: list[dict],
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_tensors = []
        hist_tensors = []
        scene_stat_tensors = []
        for pil_img in images[start:end]:
            rgb = pil_img.convert("RGB")
            t = TF.pil_to_tensor(rgb)       # uint8 [C, H, W]
            t = self._transform(t)           # float32 [3, H, W]
            img_tensors.append(t)

            hist = compute_histogram(rgb)    # (3, 32) float32
            hist_tensors.append(torch.from_numpy(hist.flatten()))  # (96,)
            scene_stats = [
                compute_scene_statistics(rgb)[field]
                for field in config.SCENE_STAT_FIELDS
            ]
            scene_stat_tensors.append(torch.tensor(scene_stats, dtype=torch.float32))

        img_batch = torch.stack(img_tensors).to(self._device)   # [B, 3, H, W]
        hist_batch = torch.stack(hist_tensors).to(self._device)  # [B, 96]
        scene_stats_batch = torch.stack(scene_stat_tensors).to(self._device)  # [B, 6]

        chunk = metadata_list[start:end]

        # AsShot WB: extract_metadata stores a (kelvin, tint) tuple per photo
        # in m["as_shot_wb"]. Sentinel-fall back to LR defaults when missing
        # so v1.1.0+ models always get sensible values. Old-arch models ignore
        # these keys entirely.
        as_shot_temps: list[float] = []
        as_shot_tints: list[float] = []
        for m in chunk:
            wb = m.get("as_shot_wb")
            if wb is None:
                as_shot_temps.append(5500.0)
                as_shot_tints.append(0.0)
            else:
                as_shot_temps.append(float(wb[0]))
                as_shot_tints.append(float(wb[1]))

        def _cat_id(mapping: dict[str, int], value) -> int:
            if value is None:
                return mapping.get("unknown", 0)
            if isinstance(value, (int, np.integer)):
                idx = int(value)
                return idx if 0 <= idx < len(mapping) else mapping.get("unknown", 0)
            if isinstance(value, float) and math.isnan(value):
                return mapping.get("unknown", 0)
            return mapping.get(str(value), mapping.get("unknown", 0))

        camera_body_ids = [
            _cat_id(self._model.registry.camera_bodies, m.get("camera_body"))
            for m in chunk
        ]
        camera_make_ids = [
            _cat_id(self._model.registry.camera_makes, m.get("make"))
            for m in chunk
        ]
        camera_model_ids = [
            _cat_id(self._model.registry.camera_models, m.get("model"))
            for m in chunk
        ]
        lens_ids = [
            _cat_id(self._model.registry.lenses, m.get("lens_model"))
            for m in chunk
        ]
        profile_ids = [
            _cat_id(self._model.registry.camera_profiles, m.get("camera_profile"))
            for m in chunk
        ]
        wb_preset_ids = [
            _cat_id(self._model.registry.wb_presets, m.get("white_balance_preset"))
            for m in chunk
        ]

        meta_batch: dict[str, torch.Tensor] = {
            "iso":               torch.tensor([_safe_float(m.get("iso"))            for m in chunk], dtype=torch.float32, device=self._device),
            "shutter_speed":     torch.tensor([_safe_float(m.get("shutter_speed")) for m in chunk], dtype=torch.float32, device=self._device),
            "aperture":          torch.tensor([_safe_float(m.get("aperture"))       for m in chunk], dtype=torch.float32, device=self._device),
            "focal_length":      torch.tensor([_safe_float(m.get("focal_length"))   for m in chunk], dtype=torch.float32, device=self._device),
            # Map extracted camera/lens/profile/WB strings into the ckpt's
            # embedding registry IDs. Unknown or novel values fall back to
            # index 0, the stable `unknown` embedding row.
            "camera_body_id":    torch.tensor(camera_body_ids, dtype=torch.long, device=self._device),
            "camera_make_id":    torch.tensor(camera_make_ids, dtype=torch.long, device=self._device),
            "camera_model_id":   torch.tensor(camera_model_ids, dtype=torch.long, device=self._device),
            "lens_id":           torch.tensor(lens_ids, dtype=torch.long, device=self._device),
            "camera_profile_id": torch.tensor(profile_ids, dtype=torch.long, device=self._device),
            "wb_preset_id":      torch.tensor(wb_preset_ids, dtype=torch.long, device=self._device),
            "histogram":         hist_batch,
            "scene_stats":       scene_stats_batch,
            "as_shot_temperature": torch.tensor(as_shot_temps, dtype=torch.float32, device=self._device),
            "as_shot_tint":        torch.tensor(as_shot_tints, dtype=torch.float32, device=self._device),
        }
        return img_batch, meta_batch

    def predict(
        self,
        images: list[Image.Image],
        metadata_list: list[dict],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Run inference on a list of images.

        Args:
            images:        PIL images (from extract_preview or any source).
            metadata_list: Raw metadata dicts (from extract_metadata). Missing
                           fields fall back to zero / unknown-embedding.
            batch_size:    Images per GPU batch.

        Returns:
            Tensor[N, 135] of postprocessed Lightroom slider values.
        """
        n = len(images)
        results: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                img_batch, meta_batch = self._build_batch(images, metadata_list, start, end)
                raw = self._model(img_batch, meta_batch)       # [B, 135] prediction space
                results.append(postprocess_predictions(raw).cpu())
        return torch.cat(results, dim=0)  # [N, 135]

    def predict_with_uncertainty(
        self,
        images: list[Image.Image],
        metadata_list: list[dict],
        n_samples: int = 10,
        batch_size: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        MC dropout inference — returns (mean, std) over n_samples forward passes.

        Args:
            images, metadata_list: Same as predict().
            n_samples: Number of stochastic forward passes (higher = better estimate).
            batch_size: Images per GPU batch per sample.

        Returns:
            (mean [N, 135], std [N, 135]) — std is per-slider uncertainty in LR units.
        """
        samples: list[torch.Tensor] = []
        _enable_dropout(self._model)
        try:
            for _ in range(n_samples):
                samples.append(self.predict(images, metadata_list, batch_size))
        finally:
            self._model.eval()  # restore eval (disables dropout)

        stacked = torch.stack(samples, dim=0)  # [n_samples, N, 135]
        return stacked.mean(dim=0), stacked.std(dim=0)

    def predict_one(
        self,
        image: Image.Image,
        metadata: dict,
    ) -> dict[str, float]:
        """Convenience: predict a single image, return slider dict for write_xmp()."""
        preds = self.predict([image], [metadata], batch_size=1)
        return predictions_to_dict(preds, batch_idx=0)
