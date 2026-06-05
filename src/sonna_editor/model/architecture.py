from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models

from sonna_editor import config


class EmbeddingRegistry:
    """Maps string labels to integer embedding IDs for each metadata category.

    v1.1.0 added `camera_makes` and `camera_models` for the make/model split.
    Older `camera_bodies` is retained for v1.0.x checkpoints' backward compat.
    """

    def __init__(self) -> None:
        self.camera_bodies: dict[str, int] = {}
        self.camera_makes:  dict[str, int] = {}
        self.camera_models: dict[str, int] = {}
        self.lenses: dict[str, int] = {}
        self.camera_profiles: dict[str, int] = {}
        self.wb_presets: dict[str, int] = {}

    def to_dict(self) -> dict:
        return {
            "camera_bodies": self.camera_bodies,
            "camera_makes":  self.camera_makes,
            "camera_models": self.camera_models,
            "lenses": self.lenses,
            "camera_profiles": self.camera_profiles,
            "wb_presets": self.wb_presets,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmbeddingRegistry":
        reg = cls()
        reg.camera_bodies = d.get("camera_bodies", {})
        reg.camera_makes  = d.get("camera_makes",  {})
        reg.camera_models = d.get("camera_models", {})
        reg.lenses = d.get("lenses", {})
        reg.camera_profiles = d.get("camera_profiles", {})
        reg.wb_presets = d.get("wb_presets", {})
        return reg


def _grow_embedding(emb: nn.Embedding) -> nn.Embedding:
    """Return a new Embedding with one extra row initialised to the mean of existing rows."""
    old_w = emb.weight.data  # [N, D]
    new_row = old_w.mean(dim=0, keepdim=True)  # [1, D]
    new_w = torch.cat([old_w, new_row], dim=0)  # [N+1, D]
    new_emb = nn.Embedding(new_w.shape[0], new_w.shape[1])
    new_emb.weight = nn.Parameter(new_w)
    new_emb.to(old_w.device)
    return new_emb


class MetadataEncoder(nn.Module):
    """
    Encodes camera metadata and RGB histogram into a 64-d feature vector.

    Architecture is selected by ``arch_version``:

    arch_version = 0  (v1.0.x — legacy)
        iso(16) + shutter(8) + aperture-raw(8) + focal-onehot(8)
        + body(16) + lens(8) + profile(8) + wb(8) + hist(32)  =  112

    arch_version = 1  (v1.1.0 — current)
        iso(16) + shutter(8) + aperture-LOG(8) + focal-LOG-cont(8)
        + make(4) + model(12) + lens(8) + profile(8) + wb(8) + hist(32)
        + as_shot_temp(8) + as_shot_tint(8)                   =  128

    arch_version = 2  (v2.1 — scene-stat quality pass)
        v1.1.0 inputs + preview luminance scene_stats(16)       =  144

    arch_version = 3  (v3.0 — staged-head quality pass)
        Same metadata encoder layout as arch_version 2. The staged prediction
        change lives in SonnaEditor's output heads, not this encoder.

    The fusion MLP first layer matches the concat dim. v1.0.x camera_body is
    replaced by separate make + model embeddings in v1.1.0 (total dim
    unchanged: 16 → 4 + 12).
    """

    _CONCAT_DIM_V1_0: int = 112
    _CONCAT_DIM_V1_1: int = 128
    _CONCAT_DIM_V2_1: int = 144

    def __init__(
        self,
        num_bodies: int,
        num_lenses: int,
        num_profiles: int,
        num_wb_presets: int,
        arch_version: int = 1,
        num_makes: int = 8,
        num_models: int = 16,
    ) -> None:
        super().__init__()

        if arch_version not in (0, 1, 2, 3):
            raise ValueError(f"unknown arch_version={arch_version}")
        self._arch_version = arch_version

        self.iso_fc = nn.Linear(1, 16)
        self.shutter_fc = nn.Linear(1, 8)
        self.aperture_fc = nn.Linear(1, 8)

        # Focal length encoder differs by arch:
        #   v1.0.x: 8-bin one-hot → Linear(8, 8)
        #   v1.1.0: log(focal_mm) → Linear(1, 8)
        if arch_version == 0:
            self.focal_length_fc = nn.Linear(8, 8)
        else:
            self.focal_length_fc = nn.Linear(1, 8)

        # Camera identity encoder differs by arch:
        #   v1.0.x: single body_emb (Make + Model combined string)
        #   v1.1.0: separate make_emb + model_emb (total dim unchanged)
        if arch_version == 0:
            self.body_emb = nn.Embedding(num_bodies, 16)
        else:
            self.make_emb  = nn.Embedding(num_makes,  4)
            self.model_emb = nn.Embedding(num_models, 12)

        self.lens_emb = nn.Embedding(num_lenses, 8)
        self.profile_emb = nn.Embedding(num_profiles, 8)
        self.wb_emb = nn.Embedding(num_wb_presets, 8)

        self.histogram_mlp = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        if arch_version >= 1:
            # AsShot Temperature in log-K (matches Temperature head's prediction space).
            # AsShot Tint in raw units.
            self.as_shot_temp_fc = nn.Linear(1, 8)
            self.as_shot_tint_fc = nn.Linear(1, 8)

        if arch_version >= 2:
            self.scene_stats_mlp = nn.Sequential(
                nn.Linear(6, 16),
                nn.GELU(),
            )

        if arch_version >= 2:
            concat_dim = self._CONCAT_DIM_V2_1
        elif arch_version >= 1:
            concat_dim = self._CONCAT_DIM_V1_1
        else:
            concat_dim = self._CONCAT_DIM_V1_0
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

        # Focal length bin boundaries (mm) for v1.0.x one-hot path. Unused by v1.1.0
        # but registered unconditionally so the buffer is present on state_dict for
        # arch_version=0 checkpoints' load path.
        self.register_buffer(
            "focal_bins",
            torch.tensor([24.0, 35.0, 50.0, 70.0, 100.0, 135.0, 200.0]),
        )

    def _focal_onehot(self, focal_mm: torch.Tensor) -> torch.Tensor:
        idx = torch.bucketize(focal_mm, self.focal_bins)  # [B], values 0-7
        return nn.functional.one_hot(idx, num_classes=8).float()

    def forward(self, metadata: dict[str, torch.Tensor]) -> torch.Tensor:
        iso_f = self.iso_fc(torch.log1p(metadata["iso"].float().unsqueeze(-1)))
        shutter_f = self.shutter_fc(
            torch.log1p(metadata["shutter_speed"].float().unsqueeze(-1))
        )

        if self._arch_version == 0:
            # v1.0.x: raw aperture + 8-bin focal one-hot + combined body embedding.
            aperture_f = self.aperture_fc(metadata["aperture"].float().unsqueeze(-1))
            focal_f = self.focal_length_fc(
                self._focal_onehot(metadata["focal_length"].float())
            )
            body_f = self.body_emb(metadata["camera_body_id"])
            cam_parts = [body_f]
        else:
            # v1.1.0: log-aperture, continuous log-focal, separate make + model.
            # NaN-safe sentinels match LR_DEFAULTS / typical-shot conventions so a
            # missing field can't corrupt gradients on the rest of the batch.
            aperture_raw = torch.nan_to_num(
                metadata["aperture"].float(), nan=5.6
            ).clamp(min=0.5)
            aperture_f = self.aperture_fc(torch.log2(aperture_raw).unsqueeze(-1))

            focal_raw = torch.nan_to_num(
                metadata["focal_length"].float(), nan=50.0
            ).clamp(min=1.0)
            focal_f = self.focal_length_fc(torch.log(focal_raw).unsqueeze(-1))

            make_f  = self.make_emb(metadata["camera_make_id"])
            model_f = self.model_emb(metadata["camera_model_id"])
            cam_parts = [make_f, model_f]

        lens_f    = self.lens_emb(metadata["lens_id"])
        profile_f = self.profile_emb(metadata["camera_profile_id"])
        wb_f      = self.wb_emb(metadata["wb_preset_id"])
        hist_f    = self.histogram_mlp(metadata["histogram"].float())

        parts = [iso_f, shutter_f, aperture_f, focal_f,
                 *cam_parts, lens_f, profile_f, wb_f, hist_f]

        if self._arch_version >= 1:
            # AsShot Temperature → log-Kelvin → Linear(1, 8). NaN → 5500K sentinel.
            ast = torch.nan_to_num(
                metadata["as_shot_temperature"].float(), nan=5500.0
            ).clamp(min=1.0)
            log_temp = torch.log(ast).unsqueeze(-1)
            parts.append(self.as_shot_temp_fc(log_temp))

            atn = torch.nan_to_num(
                metadata["as_shot_tint"].float(), nan=0.0
            ).clamp(min=-100.0, max=100.0).unsqueeze(-1)
            parts.append(self.as_shot_tint_fc(atn))

        if self._arch_version >= 2:
            raw_scene_stats = metadata.get("scene_stats")
            if raw_scene_stats is None:
                batch = iso_f.shape[0]
                raw_scene_stats = torch.zeros(
                    batch,
                    6,
                    dtype=iso_f.dtype,
                    device=iso_f.device,
                )
            scene_stats = torch.nan_to_num(
                raw_scene_stats.float(), nan=0.0, posinf=1.0, neginf=0.0
            ).clamp(min=0.0, max=1.0)
            parts.append(self.scene_stats_mlp(scene_stats))

        concat = torch.cat(parts, dim=-1)
        # NaN-proofing: assert no non-finite values flowed into the fusion MLP.
        # Cheap (one reduction op per batch) and gives a clear failure point if
        # any future input path skips a clamp.
        if torch.jit.is_scripting() is False and not torch.isfinite(concat).all():
            bad_rows = (~torch.isfinite(concat).all(dim=-1)).nonzero(as_tuple=True)[0]
            raise RuntimeError(
                f"MetadataEncoder produced non-finite concat tensor on "
                f"{bad_rows.numel()} of {concat.size(0)} batch rows "
                f"(rows {bad_rows.tolist()[:5]}...). Check metadata input clamps."
            )
        return self.fusion_mlp(concat)     # [B, 64]


def _make_head(in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
    """MLP with GELU activations and MC-dropout (p=0.1) before the final linear."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.GELU()]
        prev = h
    layers += [nn.Dropout(p=0.1), nn.Linear(prev, out_dim)]
    return nn.Sequential(*layers)


def _last_linear(head: nn.Sequential) -> nn.Linear:
    """Return the final Linear layer from an output head."""
    for module in reversed(head):
        if isinstance(module, nn.Linear):
            return module
    raise TypeError("output head does not contain a Linear layer")


class SonnaEditor(nn.Module):
    """
    Predicts Lightroom slider values from image + camera metadata.

    Output shape depends on ``slider_set_version``:
    - "v1" → Tensor[B, 135] — matches v1.2.3 shipping checkpoint.
    - "v2" → Tensor[B, 147] — adds 12 v2-extension fields (idx 135-146);
      new instantiations default to this.

    Values are in **prediction space**:
    - Temperature (index 11) is log(Kelvin). Call postprocess.postprocess_predictions()
      to convert to Lightroom units before writing XMP.
    - Tone curve fields (indices 87-134): raw control point coordinates in [0, 255].
    - All other values are in native slider units (unclamped).

    The slider_set_version gate respects the locked-append-only rule (HANDOVER
    Decision 6): indices 0-134 are frozen forever; v2 adds 12 outputs through
    five extension heads (noise_ext, defringe, lens_profile, calibration_ext,
    curve_ext). v1 ckpts load into v2 architecture via from_checkpoint with
    target_slider_set_version="v2" + strict=False (the 5 extension heads start
    random-init and learn from scratch on the next retrain). v2 → v1 loads
    raise to avoid silent information loss.

    Loss note (Task 3.2): Temperature targets must be log-transformed before computing
    loss. Use torch.log(target[:, TEMPERATURE_IDX]) for the temperature term.

    Backbone freeze note: freeze_backbone() freezes ConvNeXt stages 0-1.
    unfreeze_backbone() exposes the method but does NOT schedule itself.
    Task 3.3 training module MUST call unfreeze_backbone() at the configured epoch.
    """

    _IMAGE_FEAT_DIM: int = 768
    _META_FEAT_DIM: int = 64
    _FUSION_DIM: int = _IMAGE_FEAT_DIM + _META_FEAT_DIM  # 832

    # Output counts per slider_set_version. v1 is locked at 135 forever
    # (HANDOVER Decision 6 locked-append-only rule). v2 adds 12 outputs
    # through 5 extension heads.
    _V1_OUTPUT_COUNT: int = 135
    _V2_OUTPUT_COUNT: int = 147

    # Min embedding table sizes when no registry entries exist yet
    _MIN_BODIES: int = 8
    _MIN_MAKES: int = 4
    _MIN_MODELS: int = 16
    _MIN_LENSES: int = 16
    _MIN_PROFILES: int = 8
    _MIN_WB: int = 8

    def __init__(
        self,
        registry: Optional[EmbeddingRegistry] = None,
        freeze_backbone: bool = False,
        _embedding_sizes: Optional[dict[str, int]] = None,
        _pretrained_backbone: bool = True,
        arch_version: int = 3,
        slider_set_version: str = "v2",
        use_wb_metadata_skip: bool = True,
    ) -> None:
        super().__init__()

        # arch_version selects the metadata-encoder layout:
        #   0 = v1.0.x legacy (combined camera_body, one-hot focal, raw aperture)
        #   1 = v1.1.0       (separate make+model, log focal, log aperture, AsShot)
        #   2 = v2.1         (v1.1.0 + luminance scene statistics)
        #   3 = v3.0         (v2.1 + staged head conditioning)
        if arch_version not in (0, 1, 2, 3):
            raise ValueError(f"unknown arch_version={arch_version}")
        self._arch_version = arch_version
        self._use_staged_heads = arch_version >= 3

        # slider_set_version selects which output heads exist:
        #   "v1" = 13 heads, 135 outputs (idx 0-134) — matches v1.2.3 shipping ckpt
        #   "v2" = 18 heads, 147 outputs (idx 0-146) — adds 5 extension heads
        # Locked-append-only: indices 0-134 are frozen forever. See HANDOVER
        # Decision 6 for the design rule.
        if slider_set_version not in ("v1", "v2"):
            raise ValueError(
                f"unknown slider_set_version={slider_set_version!r}; "
                f"expected one of 'v1', 'v2'"
            )
        self._slider_set_version = slider_set_version
        self._use_wb_metadata_skip = bool(use_wb_metadata_skip and arch_version >= 1)

        self.registry = registry or EmbeddingRegistry()

        # Determine embedding table sizes. _embedding_sizes is used by from_checkpoint
        # to restore exact checkpoint dimensions; otherwise we use registry + min capacity.
        _sz = _embedding_sizes or {}
        num_bodies = _sz.get("num_bodies", max(len(self.registry.camera_bodies), self._MIN_BODIES))
        num_makes  = _sz.get("num_makes",  max(len(self.registry.camera_makes),  self._MIN_MAKES))
        num_models = _sz.get("num_models", max(len(self.registry.camera_models), self._MIN_MODELS))
        num_lenses = _sz.get("num_lenses", max(len(self.registry.lenses), self._MIN_LENSES))
        num_profiles = _sz.get("num_profiles", max(len(self.registry.camera_profiles), self._MIN_PROFILES))
        num_wb_presets = _sz.get("num_wb_presets", max(len(self.registry.wb_presets), self._MIN_WB))

        # Backbone: ConvNeXt-Tiny pretrained. Classifier head replaced with norm + flatten.
        # _pretrained_backbone=False is used by from_checkpoint (weights come from state_dict).
        _weights = models.ConvNeXt_Tiny_Weights.DEFAULT if _pretrained_backbone else None
        _backbone = models.convnext_tiny(weights=_weights)
        self.backbone_features = _backbone.features    # 8 ConvNeXt stages
        self.backbone_pool = _backbone.avgpool         # AdaptiveAvgPool2d(1) → [B, 768, 1, 1]
        self.backbone_norm = _backbone.classifier[0]   # LayerNorm2d(768)
        # classifier[1] is Flatten — applied manually in forward() for clarity

        self.metadata_encoder = MetadataEncoder(
            num_bodies=num_bodies,
            num_makes=num_makes,
            num_models=num_models,
            num_lenses=num_lenses,
            num_profiles=num_profiles,
            num_wb_presets=num_wb_presets,
            arch_version=arch_version,
        )

        # Output heads — concatenated in SLIDER_FIELDS order.
        # arch_version >= 3 uses staged conditioning: later edit blocks see the
        # earlier block predictions, mirroring a practical editing sequence.
        base_dim = self._FUSION_DIM
        tone_context_dim = 8
        early_context_dim = 8 + 3 + 2
        wb_presence_in_dim = base_dim + tone_context_dim if self._use_staged_heads else base_dim
        later_in_dim = base_dim + early_context_dim if self._use_staged_heads else base_dim

        # WB head: output[0] = log(Temperature), output[1] = Tint (raw units)
        self.tone_head          = _make_head(base_dim, [256, 128], 8)
        self.presence_head      = _make_head(wb_presence_in_dim, [128, 64], 3)
        self.wb_head            = _make_head(wb_presence_in_dim, [128, 64], 2)
        if self._use_wb_metadata_skip:
            # The v1.2.3 audit found that AsShot Temperature/Tint were present
            # in metadata but got diluted by the shared fusion MLP. This
            # identity-initialised residual gives the WB head a direct route:
            # output = learned residual + [log(as_shot_temperature), as_shot_tint].
            self.wb_metadata_skip = nn.Linear(2, 2)
            with torch.no_grad():
                self.wb_metadata_skip.weight.zero_()
                self.wb_metadata_skip.weight[0, 0] = 1.0
                self.wb_metadata_skip.weight[1, 1] = 1.0
                self.wb_metadata_skip.bias.zero_()
        self.hsl_head           = _make_head(later_in_dim, [256, 128], 24)
        self.parametric_head    = _make_head(later_in_dim, [128, 64], 7)
        self.color_grading_head = _make_head(later_in_dim, [128, 64], 14)
        self.calibration_head   = _make_head(later_in_dim, [128, 64], 6)
        self.detail_head        = _make_head(later_in_dim, [64], 4)
        self.noise_head         = _make_head(later_in_dim, [64], 4)
        self.effects_head       = _make_head(later_in_dim, [64], 8)
        self.lens_head          = _make_head(later_in_dim, [64], 2)
        self.transform_head     = _make_head(later_in_dim, [64], 5)
        # Larger hidden dims for tone curves — 6 correlated control points per channel
        self.tone_curve_head    = _make_head(later_in_dim, [256, 128], 48)

        # v2 extension heads — conditionally instantiated. When slider_set_version="v1"
        # these are absent so v1.2.3 ckpts load via load_state_dict(strict=True). When
        # "v2", these 5 heads add 12 outputs (idx 135-146) appended to the forward
        # concat. Hidden dim [64] is sufficient for the sparse-signal extension fields.
        if slider_set_version == "v2":
            self.noise_ext_head       = _make_head(later_in_dim, [64], 2)   # idx 135-136
            self.defringe_head        = _make_head(later_in_dim, [64], 6)   # idx 137-142
            self.lens_profile_head    = _make_head(later_in_dim, [64], 2)   # idx 143-144
            self.calibration_ext_head = _make_head(later_in_dim, [64], 1)   # idx 145
            self.curve_ext_head       = _make_head(later_in_dim, [64], 1)   # idx 146

        if freeze_backbone:
            self.freeze_backbone()

    # ------------------------------------------------------------------
    # Output prior initialisation
    # ------------------------------------------------------------------

    def initialise_output_priors(
        self,
        priors: dict[str, float],
        *,
        zero_final_weights: bool = True,
    ) -> None:
        """Initialise output-head final biases from training target priors.

        Fresh random heads are a poor starting point for small datasets: the
        model can begin with arbitrary Exposure/WB values and spend early epochs
        fighting that noise. This method sets each head's final bias to the
        median training target in the model's prediction space. Temperature is
        converted to log-Kelvin. When the direct WB metadata skip is enabled,
        the WB head predicts a residual on top of AsShot WB, so its initial
        residual is set to exactly zero.
        """
        head_specs: list[tuple[nn.Sequential, list[str]]] = [
            (self.tone_head, config.SLIDER_FIELDS[0:8]),
            (self.presence_head, config.SLIDER_FIELDS[8:11]),
            (self.wb_head, config.SLIDER_FIELDS[11:13]),
            (self.hsl_head, config.SLIDER_FIELDS[13:37]),
            (self.parametric_head, config.SLIDER_FIELDS[37:44]),
            (self.color_grading_head, config.SLIDER_FIELDS[44:58]),
            (self.calibration_head, config.SLIDER_FIELDS[58:64]),
            (self.detail_head, config.SLIDER_FIELDS[64:68]),
            (self.noise_head, config.SLIDER_FIELDS[68:72]),
            (self.effects_head, config.SLIDER_FIELDS[72:80]),
            (self.lens_head, config.SLIDER_FIELDS[80:82]),
            (self.transform_head, config.SLIDER_FIELDS[82:87]),
            (self.tone_curve_head, config.SLIDER_FIELDS[87:135]),
        ]
        if self._slider_set_version == "v2":
            head_specs.extend([
                (self.noise_ext_head, config.SLIDER_FIELDS[135:137]),
                (self.defringe_head, config.SLIDER_FIELDS[137:143]),
                (self.lens_profile_head, config.SLIDER_FIELDS[143:145]),
                (self.calibration_ext_head, config.SLIDER_FIELDS[145:146]),
                (self.curve_ext_head, config.SLIDER_FIELDS[146:147]),
            ])

        with torch.no_grad():
            for head, fields in head_specs:
                layer = _last_linear(head)
                if zero_final_weights:
                    layer.weight.zero_()
                values: list[float] = []
                for field in fields:
                    if self._use_wb_metadata_skip and field in {"Temperature", "Tint"}:
                        values.append(0.0)
                        continue
                    value = priors.get(field)
                    if value is None:
                        value = config.SLIDER_DEFAULTS.get(field, 0.0)
                    if field == "Temperature":
                        value = float(torch.log(torch.tensor(max(float(value), 1.0))).item())
                    values.append(float(value))
                layer.bias.copy_(torch.tensor(values, dtype=layer.bias.dtype, device=layer.bias.device))

    # ------------------------------------------------------------------
    # Backbone freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze ConvNeXt stages 0 and 1 (stem + first downsampler)."""
        for p in self.backbone_features[0].parameters():
            p.requires_grad = False
        for p in self.backbone_features[1].parameters():
            p.requires_grad = False

    def freeze_entire_backbone(self) -> None:
        """Freeze all ConvNeXt feature stages and the classifier norm."""
        for p in self.backbone_features.parameters():
            p.requires_grad = False
        for p in self.backbone_norm.parameters():
            p.requires_grad = False

    def freeze_backbone_pool(self) -> None:
        """Keep the ConvNeXt adaptive pool frozen for accounting consistency."""
        for p in self.backbone_pool.parameters():
            p.requires_grad = False

    def unfreeze_backbone_from_stage(self, first_trainable_stage: int) -> None:
        """Unfreeze ConvNeXt stages from `first_trainable_stage` onward.

        Stages before the threshold stay frozen. The classifier norm is
        trainable whenever any feature stage is trainable.
        """
        n_stages = len(self.backbone_features)
        first = max(0, min(first_trainable_stage, n_stages))
        for idx, stage in enumerate(self.backbone_features):
            trainable = idx >= first
            for p in stage.parameters():
                p.requires_grad = trainable
        norm_trainable = first < n_stages
        for p in self.backbone_norm.parameters():
            p.requires_grad = norm_trainable

    def set_trainable_backbone_layers(self, spec: str) -> None:
        """Configure trainable ConvNeXt stages/blocks from a compact spec.

        ``spec`` accepts comma-separated tokens:
        - ``none``: freeze all ConvNeXt feature stages and norm.
        - ``all``: unfreeze the full ConvNeXt feature extractor and norm.
        - ``from:N``: unfreeze stages N and later.
        - ``stage:N``: unfreeze one full stage.
        - ``block:S:I`` or ``block:S:I-J``: unfreeze block(s) inside a stage.

        Examples:
        - ``block:7:2,stage:6`` -> final ConvNeXt block plus stage-6 downsample.
        - ``block:7:1-2,stage:6`` -> about 12.6M trainable params on v2.
        - ``stage:7`` -> final ConvNeXt stage plus trainable fusion/heads.
        """
        cleaned = (spec or "").strip().lower()
        if not cleaned:
            raise ValueError("backbone trainable layer spec must not be empty")
        tokens = [token.strip() for token in cleaned.split(",") if token.strip()]
        if not tokens:
            raise ValueError("backbone trainable layer spec must not be empty")

        self.freeze_entire_backbone()
        self.freeze_backbone_pool()

        if tokens == ["none"]:
            return
        if tokens == ["all"]:
            self.unfreeze_backbone()
            return

        n_stages = len(self.backbone_features)
        for token in tokens:
            if token == "norm":
                for p in self.backbone_norm.parameters():
                    p.requires_grad = True
                continue
            if token.startswith("from:"):
                stage = int(token.split(":", 1)[1])
                first = max(0, min(stage, n_stages))
                for idx, module in enumerate(self.backbone_features):
                    if idx >= first:
                        for p in module.parameters():
                            p.requires_grad = True
                continue
            if token.startswith("stage:"):
                stage = int(token.split(":", 1)[1])
                if stage < 0 or stage >= n_stages:
                    raise ValueError(f"Backbone stage out of range: {stage}")
                for p in self.backbone_features[stage].parameters():
                    p.requires_grad = True
                continue
            if token.startswith("block:"):
                parts = token.split(":")
                if len(parts) != 3:
                    raise ValueError(f"Invalid backbone block token: {token!r}")
                stage = int(parts[1])
                if stage < 0 or stage >= n_stages:
                    raise ValueError(f"Backbone stage out of range: {stage}")
                stage_module = self.backbone_features[stage]
                block_ids = self._parse_block_ids(parts[2])
                children = list(stage_module.children())
                for block_id in block_ids:
                    if block_id < 0 or block_id >= len(children):
                        raise ValueError(
                            f"Backbone block {block_id} out of range for stage {stage}"
                        )
                    for p in children[block_id].parameters():
                        p.requires_grad = True
                continue
            raise ValueError(f"Unknown backbone trainable layer token: {token!r}")

        if any(p.requires_grad for p in self.backbone_features.parameters()):
            for p in self.backbone_norm.parameters():
                p.requires_grad = True

    @staticmethod
    def _parse_block_ids(raw: str) -> list[int]:
        if "-" in raw:
            start_s, end_s = raw.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid descending block range: {raw!r}")
            return list(range(start, end + 1))
        return [int(raw)]

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all backbone parameters.

        This method exists for the training module to call — it does NOT schedule
        itself. Task 3.3 training module MUST call this at the configured epoch
        (e.g., after warmup). Failing to call it means the backbone stays frozen
        for the entire training run.
        """
        for p in self.backbone_features.parameters():
            p.requires_grad = True
        for p in self.backbone_norm.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Embedding registry growth
    # ------------------------------------------------------------------

    def _add_to_registry(
        self,
        label: str,
        label_map: dict[str, int],
        emb_attr: str,
    ) -> int:
        if label in label_map:
            return label_map[label]
        new_id = len(label_map)
        label_map[label] = new_id
        # Only grow the embedding table when we've exhausted the current capacity.
        # The first _MIN_* IDs map to pre-existing rows (randomly/mean init at startup).
        emb = getattr(self.metadata_encoder, emb_attr)
        if new_id >= emb.num_embeddings:
            setattr(self.metadata_encoder, emb_attr, _grow_embedding(emb))
        return new_id

    def add_camera_body(self, name: str) -> int:
        # v1.0.x path; v1.1.0 doesn't carry body_emb but we still update the
        # registry so the saved dict carries the legacy mapping for any future
        # backward-compat tooling. No-op on the embedding table for arch_version >= 1.
        if self._arch_version == 0:
            return self._add_to_registry(name, self.registry.camera_bodies, "body_emb")
        if name in self.registry.camera_bodies:
            return self.registry.camera_bodies[name]
        new_id = len(self.registry.camera_bodies)
        self.registry.camera_bodies[name] = new_id
        return new_id

    def add_camera_make(self, name: str) -> int:
        return self._add_to_registry(name, self.registry.camera_makes, "make_emb")

    def add_camera_model(self, name: str) -> int:
        return self._add_to_registry(name, self.registry.camera_models, "model_emb")

    def add_lens(self, name: str) -> int:
        return self._add_to_registry(name, self.registry.lenses, "lens_emb")

    def add_camera_profile(self, name: str) -> int:
        return self._add_to_registry(name, self.registry.camera_profiles, "profile_emb")

    def add_wb_preset(self, name: str) -> int:
        return self._add_to_registry(name, self.registry.wb_presets, "wb_emb")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,                 # [B, 3, H, W] — H == W == config.IMAGE_RESOLUTION
        metadata: dict[str, torch.Tensor],
    ) -> torch.Tensor:                        # [B, 135] v1 / [B, 147] v2, prediction space
        # Image branch
        x = self.backbone_features(image)    # [B, 768, H', W']
        x = self.backbone_pool(x)            # [B, 768, 1, 1]
        x = self.backbone_norm(x)            # [B, 768, 1, 1] layer-normed
        img_feat = x.flatten(1)             # [B, 768]

        # Metadata branch
        meta_feat = self.metadata_encoder(metadata)  # [B, 64]

        # Fuse and run heads — concat order matches SLIDER_FIELDS exactly
        fused = torch.cat([img_feat, meta_feat], dim=-1)  # [B, 832]

        tone_out = self.tone_head(fused)
        wb_presence_in = (
            torch.cat([fused, tone_out], dim=-1)
            if self._use_staged_heads
            else fused
        )

        wb_out = self.wb_head(wb_presence_in)
        if self._use_wb_metadata_skip:
            as_shot_temp = torch.nan_to_num(
                metadata["as_shot_temperature"].float(), nan=5500.0
            ).clamp(min=1.0)
            as_shot_tint = torch.nan_to_num(
                metadata["as_shot_tint"].float(), nan=0.0
            ).clamp(min=-100.0, max=100.0)
            wb_skip_in = torch.stack([torch.log(as_shot_temp), as_shot_tint], dim=-1)
            wb_out = wb_out + self.wb_metadata_skip(wb_skip_in)
        presence_out = self.presence_head(wb_presence_in)

        later_in = (
            torch.cat([fused, tone_out, presence_out, wb_out], dim=-1)
            if self._use_staged_heads
            else fused
        )

        outputs = [
            tone_out,                        # [B, 8]   idx 0-7
            presence_out,                    # [B, 3]   idx 8-10
            wb_out,                          # [B, 2]   idx 11-12: [log_temperature, tint]
            self.hsl_head(later_in),         # [B, 24]  idx 13-36
            self.parametric_head(later_in),  # [B, 7]   idx 37-43
            self.color_grading_head(later_in), # [B, 14] idx 44-57
            self.calibration_head(later_in), # [B, 6]   idx 58-63
            self.detail_head(later_in),      # [B, 4]   idx 64-67
            self.noise_head(later_in),       # [B, 4]   idx 68-71
            self.effects_head(later_in),     # [B, 8]   idx 72-79
            self.lens_head(later_in),        # [B, 2]   idx 80-81
            self.transform_head(later_in),   # [B, 5]   idx 82-86
            self.tone_curve_head(later_in),  # [B, 48]  idx 87-134
        ]

        # v2 extension heads — appended only when slider_set_version="v2".
        # Locked-append-only: these never reorder, never replace v1 outputs.
        if self._slider_set_version == "v2":
            outputs.extend([
                self.noise_ext_head(later_in),       # [B, 2]  idx 135-136
                self.defringe_head(later_in),        # [B, 6]  idx 137-142
                self.lens_profile_head(later_in),    # [B, 2]  idx 143-144
                self.calibration_ext_head(later_in), # [B, 1]  idx 145
                self.curve_ext_head(later_in),       # [B, 1]  idx 146
            ])

        return torch.cat(outputs, dim=-1)  # [B, 135] v1 / [B, 147] v2

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path) -> None:
        # num_sliders reflects THIS model's output count (not global config),
        # so old ckpts saved by v1 instances always report 135 even after config
        # grows. slider_set_version is the canonical version indicator added in
        # v2 prep; older v1.2.3 ckpts predate this field and are detected at
        # load time via num_sliders fallback (see from_checkpoint).
        output_count = (
            self._V2_OUTPUT_COUNT if self._slider_set_version == "v2"
            else self._V1_OUTPUT_COUNT
        )
        torch.save(
            {
                "model_state": self.state_dict(),
                "registry": self.registry.to_dict(),
                "arch_config": {
                    "image_resolution": config.IMAGE_RESOLUTION,
                    "num_sliders": output_count,
                    "arch_version": self._arch_version,
                    "slider_set_version": self._slider_set_version,
                    "use_wb_metadata_skip": self._use_wb_metadata_skip,
                },
            },
            path,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        device: str = "cpu",
        target_slider_set_version: Optional[str] = None,
    ) -> "SonnaEditor":
        """
        Load a SonnaEditor from a checkpoint file.

        target_slider_set_version controls how the loaded checkpoint is mapped
        to a model instance:
        - None (default): instantiate at the checkpoint's own slider_set_version.
          load_state_dict uses strict=True. Used for normal inference and continued
          training.
        - "v2": instantiate at v2 regardless of what the checkpoint was saved as.
          When the ckpt is v1, load uses strict=False so the v2 extension heads
          (with no entries in the v1 state_dict) start random-init. Used for
          v1 ckpt → v2 architecture warm-start; the 5 extension heads learn
          from scratch during the next retrain.
        - "v1": only valid when the ckpt is already v1 (effectively a no-op).
          Loading a v2 ckpt as v1 raises ValueError to avoid silent information
          loss from dropping the 5 extension-head weight tensors.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        registry = EmbeddingRegistry.from_dict(ckpt["registry"])
        state = ckpt["model_state"]
        arch_config = ckpt.get("arch_config", {}) or {}

        # arch_version is the canonical source for native ckpts. Legacy ckpts
        # that pre-date v1.1.0 didn't have the field; fall back to state-dict
        # shape detection so v1.0.x ckpts continue to load unchanged.
        arch_version = arch_config.get("arch_version")
        if arch_version is None:
            if "metadata_encoder.scene_stats_mlp.0.weight" in state:
                arch_version = 2
            elif "metadata_encoder.make_emb.weight" in state:
                arch_version = 1
            else:
                arch_version = 0

        # slider_set_version: canonical field added in v2 prep. Pre-v2 ckpts
        # (including v1.2.3) don't have it — infer from num_sliders, falling
        # back to "v1" for any value < 147 (locked-append-only guarantees v1
        # output count is exactly 135).
        source_slider_set_version = arch_config.get("slider_set_version")
        if source_slider_set_version is None:
            num_sliders = arch_config.get("num_sliders", cls._V1_OUTPUT_COUNT)
            source_slider_set_version = (
                "v2" if num_sliders >= cls._V2_OUTPUT_COUNT else "v1"
            )

        effective_slider_set_version = (
            target_slider_set_version
            if target_slider_set_version is not None
            else source_slider_set_version
        )
        use_wb_metadata_skip = bool(arch_config.get("use_wb_metadata_skip", False))

        # Reject v2 → v1 loads explicitly. Loading a v2 ckpt as a v1 model would
        # silently drop the 5 extension-head weight tensors, masking real
        # information loss. v1 → v2 (warm-start) is the only supported
        # cross-version direction.
        if source_slider_set_version == "v2" and effective_slider_set_version == "v1":
            raise ValueError(
                f"Cannot load v2 checkpoint as v1 model: would silently drop "
                f"the {cls._V2_OUTPUT_COUNT - cls._V1_OUTPUT_COUNT} extension-head "
                f"weight tensors (idx 135-146). If you explicitly want to discard "
                f"v2 extension heads, do it in the calling code, not via "
                f"from_checkpoint(target_slider_set_version='v1')."
            )

        # Cross-version loads (v1 ckpt → v2 architecture) need strict=False
        # because the v2 extension heads have no entries in a v1 state_dict.
        strict_load = (effective_slider_set_version == source_slider_set_version)

        # Derive exact embedding table sizes from the state dict so load_state_dict
        # never hits a shape mismatch. v1.0.x has body_emb; v1.1.0 has make+model.
        embedding_sizes: dict[str, int] = {
            "num_lenses":     state["metadata_encoder.lens_emb.weight"].shape[0],
            "num_profiles":   state["metadata_encoder.profile_emb.weight"].shape[0],
            "num_wb_presets": state["metadata_encoder.wb_emb.weight"].shape[0],
        }
        if arch_version == 0:
            embedding_sizes["num_bodies"] = state["metadata_encoder.body_emb.weight"].shape[0]
        else:
            embedding_sizes["num_makes"]  = state["metadata_encoder.make_emb.weight"].shape[0]
            embedding_sizes["num_models"] = state["metadata_encoder.model_emb.weight"].shape[0]

        model = cls(
            registry=registry,
            _embedding_sizes=embedding_sizes,
            _pretrained_backbone=False,
            arch_version=int(arch_version),
            slider_set_version=effective_slider_set_version,
            use_wb_metadata_skip=use_wb_metadata_skip,
        )
        model.load_state_dict(state, strict=strict_load)
        return model
