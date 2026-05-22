"""Diffusion model architectures for BRep generation.

This module implements the denoising diffusion models used to generate B-Rep
face latent sequences conditioned on images, point clouds, or text.

Architecture Overview
---------------------
Two main conditioning strategies are provided:

1. **DiffusionCrossAttn** (formerly ``Diffusion_condition``):
   Uses a 4-layer TransformerDecoder to cross-attend noisy latent tokens to
   condition features, followed by an N-layer TransformerEncoder denoising
   backbone with optional per-layer image adapters.  Includes a CLIP-style
   alignment loss between CAD and image embeddings.

2. **DiffusionConcat** (formerly ``Diffusion_condition_mm`` / multimodal):
   Concatenates a learned condition embedding to the noisy latent along the
   feature dimension, then processes with a single TransformerEncoder.  Uses
   learned modality embeddings (svr, mvr, sketch, pc, txt, uncond) fused
   through an 8-layer TransformerEncoder attention block.

Both classes share:
- A frozen pretrained AutoEncoder for latent extraction (``ae_model``)
- A DDPM noise scheduler from ``diffusers``
- Sinusoidal timestep embeddings via ``sincos_embedding``
- Support for zero-pad or random-pad face packing strategies
"""

from __future__ import annotations

import importlib
import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast, nn, Tensor
from diffusers import DDPMScheduler
from tqdm import tqdm

from .blocks import sincos_embedding


# ---------------------------------------------------------------------------
# Helper: CLIP-style symmetric InfoNCE loss
# ---------------------------------------------------------------------------


def clip_style_symmetric_infonce(cad_emb: Tensor, img_emb: Tensor, tau: float = 0.07) -> Tensor:
    """Compute symmetric CLIP-style contrastive loss between two embedding sets.

    Computes a bidirectional cross-entropy loss over the cosine-similarity
    matrix scaled by a temperature parameter, encouraging matched pairs
    (same index) to have higher similarity than non-matched pairs.

    Args:
        cad_emb: Normalized CAD embeddings of shape ``(B, D)``.
        img_emb: Normalized image/condition embeddings of shape ``(B, D)``.
        tau: Temperature scaling factor for the logits.

    Returns:
        Scalar tensor with the average of CAD->Image and Image->CAD losses.
    """
    # Cosine similarity matrix / temperature
    logits = cad_emb @ img_emb.T / tau  # (B, B)

    labels = torch.arange(cad_emb.size(0), device=cad_emb.device)

    # CAD -> Image
    loss_c2i = F.cross_entropy(logits, labels)

    # Image -> CAD
    loss_i2c = F.cross_entropy(logits.T, labels)

    loss = 0.5 * (loss_c2i + loss_i2c)
    return loss


# ---------------------------------------------------------------------------
# Per-layer adapter with zero-initialized output projection
# ---------------------------------------------------------------------------


class ZeroInitImageAdapter(nn.Module):
    """Per-layer adapter that injects condition information via cross-attention.

    Designed to be inserted between TransformerEncoder layers.  The output
    projection is zero-initialized so that at initialization the adapter is
    an identity function, preserving pretrained backbone behavior.

    Architecture:
        1. LayerNorm on the hidden state ``x``
        2. LayerNorm + Linear projection on ``condition_tokens``
        3. Multi-head cross-attention (query=x, key/value=projected condition)
        4. Zero-initialized linear output projection
        5. Residual addition: ``x + out_proj(attn_output)``

    Args:
        dim_model: Dimension of the main hidden state.
        dim_condition: Dimension of the incoming condition tokens.
        num_heads: Number of attention heads.
    """

    def __init__(self, dim_model: int, dim_condition: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim_model)
        self.cond_norm = nn.LayerNorm(dim_condition)
        self.cond_proj = nn.Linear(dim_condition, dim_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.out_proj = nn.Linear(dim_model, dim_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor, condition_tokens: Tensor) -> Tensor:
        """Apply adapter: cross-attend to condition, add residual.

        Args:
            x: Hidden state from the denoising backbone, shape ``(B, S, D)``.
            condition_tokens: Condition features, shape ``(B, T, D_cond)``.

        Returns:
            Updated hidden state with the same shape as ``x``.
        """
        condition_tokens = self.cond_proj(self.cond_norm(condition_tokens))
        delta, _ = self.attn(
            self.norm(x),
            condition_tokens,
            condition_tokens,
            need_weights=False,
        )
        return x + self.out_proj(delta)


# ---------------------------------------------------------------------------
# DiffusionCrossAttn: cross-attention conditioned diffusion
# ---------------------------------------------------------------------------


class DiffusionCrossAttn(nn.Module):
    """Cross-attention conditioned diffusion model for BRep face generation.

    This model denoises a sequence of face latent tokens conditioned on
    external features (image/point-cloud/text) via cross-attention.  It
    corresponds to the original ``Diffusion_condition`` class.

    Architecture:
        - **p_embed**: Linear(dim_input -> dim_latent) + LayerNorm + SiLU + Linear
        - **net1**: N-layer TransformerEncoder (denoising backbone)
        - **img_adapters** (optional): ZeroInitImageAdapter per N/k layers
        - **cross_attn_pre_proj** + **cross_attn_add_cond** (4-layer TransformerDecoder)
          + **cross_attn_post_proj**: Cross-attention conditioning pathway
        - **Alignment branch**: cad_align_proj, img_align_proj, cad_align_head,
          img_align_head for CLIP-style contrastive alignment
        - **classifier**: Face mask predictor for zero-pad method
        - **noise_scheduler**: DDPMScheduler
        - **time_embed**: sincos -> MLP timestep embedding
        - **ae_model**: Frozen pretrained AutoEncoder (loaded from checkpoint)
        - **fc_out**: Linear -> output projection

    Key Design:
        The ``forward()`` method accepts ``condition_features`` as an argument
        instead of internally calling an encoder. This decouples condition
        extraction from diffusion, enabling strategies to intercept/modify
        conditions.  For backward compatibility, if ``condition_features`` is
        None, it falls back to internal ``extract_condition()`` (legacy path).

    Args:
        cfg: Configuration dictionary with keys:
            - diffusion_latent (int): Latent dimension
            - num_diffusion_layers (int): Number of transformer layers (default 24)
            - layerwise_img_adapter (bool): Enable per-layer adapters
            - layerwise_img_adapter_every (int): Insert adapter every N layers
            - beta_schedule, diffusion_type, beta_start, beta_end, variance_type
            - num_max_faces (int): Maximum face count per sample
            - loss (str): "l1" or "mse"
            - pad_method (str): "zero" or "random"
            - autoencoder (str): AutoEncoder class name
            - autoencoder_weights (str|None): Path to AE checkpoint
            - stored_z (bool): Whether latents are pre-computed
            - use_mean (bool): Use mean (deterministic) or sample from posterior
            - train_decoder (bool): Whether to fine-tune the decoder
            - condition (list[str]): Active condition modalities
            - is_aug (bool): Enable augmentation
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.dim_input = 8 * 2 * 2  # Face latent dimensionality (32)
        self.dim_latent = cfg["diffusion_latent"]
        self.dim_condition = 1024
        self.dim_total = self.dim_latent
        self.time_statics = [0 for _ in range(10)]
        self.lambda_align = 1.0
        self.topo_scale = float(cfg.get("topo_scale", 2.0))
        self.num_diffusion_layers = int(cfg.get("num_diffusion_layers", 24))
        self.layerwise_img_adapter = bool(cfg.get("layerwise_img_adapter", False))
        self.layerwise_img_adapter_every = max(1, int(cfg.get("layerwise_img_adapter_every", 2)))

        # ----- Latent projection -----
        self.p_embed = nn.Sequential(
            nn.Linear(self.dim_input, self.dim_latent),
            nn.LayerNorm(self.dim_latent),
            nn.SiLU(),
            nn.Linear(self.dim_latent, self.dim_latent),
        )

        # ----- Denoising backbone: N-layer TransformerEncoder -----
        layer1 = nn.TransformerEncoderLayer(
            d_model=self.dim_total,
            nhead=self.dim_total // 64,
            norm_first=True,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.net1 = nn.TransformerEncoder(layer1, self.num_diffusion_layers, nn.LayerNorm(self.dim_total))

        # ----- Optional: per-layer image adapters -----
        if self.layerwise_img_adapter:
            self.img_adapter_layer_ids = [
                i for i in range(self.num_diffusion_layers)
                if (i + 1) % self.layerwise_img_adapter_every == 0
            ]
            self.img_adapters = nn.ModuleList([
                ZeroInitImageAdapter(self.dim_total, self.dim_condition, self.dim_total // 64)
                for _ in self.img_adapter_layer_ids
            ])

        # ----- Output projection -----
        self.fc_out = nn.Sequential(
            nn.Linear(self.dim_total, self.dim_total),
            nn.LayerNorm(self.dim_total),
            nn.SiLU(),
            nn.Linear(self.dim_total, self.dim_input),
        )

        # ----- Condition modality flags (for legacy extract_condition) -----
        self.with_img = False
        self.with_pc = False
        self.with_txt = False
        self.is_aug = cfg["is_aug"]
        if "single_img" in cfg["condition"] or "multi_img" in cfg["condition"] or "sketch" in cfg["condition"]:
            self.with_img = True
        if "pc" in cfg["condition"]:
            self.with_pc = True
        if "txt" in cfg["condition"]:
            self.with_txt = True

        # ----- Cross-attention condition pathway -----
        self.cross_attn_pre_proj = nn.Linear(self.dim_latent, 1024)
        cross_attn_layer = nn.TransformerDecoderLayer(
            d_model=1024,
            nhead=1024 // 64,
            norm_first=True,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.cross_attn_add_cond = nn.TransformerDecoder(cross_attn_layer, 4, nn.LayerNorm(1024))
        self.cross_attn_post_proj = nn.Linear(1024, self.dim_latent)

        # ----- Alignment branch (CLIP-style contrastive) -----
        self.cad_align_proj = nn.Linear(1024, 256)
        self.img_align_proj = nn.Linear(1024, 256)
        self.cad_align_head = nn.Linear(256, 256)
        self.img_align_head = nn.Linear(256, 256)

        # ----- Face mask classifier (for zero-pad method) -----
        self.classifier = nn.Sequential(
            nn.Linear(self.dim_input, self.dim_input),
            nn.LayerNorm(self.dim_input),
            nn.SiLU(),
            nn.Linear(self.dim_input, 1),
        )

        # ----- DDPM noise scheduler -----
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule=cfg["beta_schedule"],
            prediction_type=cfg["diffusion_type"],
            beta_start=cfg["beta_start"],
            beta_end=cfg["beta_end"],
            variance_type=cfg["variance_type"],
            clip_sample=False,
        )

        # ----- Timestep embedding: sincos -> MLP -----
        self.time_embed = nn.Sequential(
            nn.Linear(self.dim_total, self.dim_total),
            nn.LayerNorm(self.dim_total),
            nn.SiLU(),
            nn.Linear(self.dim_total, self.dim_total),
        )

        # ----- Hyperparameters -----
        self.num_max_faces = cfg["num_max_faces"]
        self.loss = nn.functional.l1_loss if cfg["loss"] == "l1" else nn.functional.mse_loss
        self.diffusion_type = cfg["diffusion_type"]
        self.pad_method = cfg["pad_method"]

        # ----- Frozen pretrained AutoEncoder -----
        model_mod = importlib.import_module("src.brepnet.model")
        model_cls = getattr(model_mod, cfg["autoencoder"])
        self.ae_model = model_cls(cfg)

        self.is_pretrained = cfg["autoencoder_weights"] is not None
        self.is_stored_z = cfg["stored_z"]
        self.use_mean = cfg["use_mean"]
        self.is_train_decoder = cfg["train_decoder"]
        if self.is_pretrained:
            checkpoint = torch.load(cfg["autoencoder_weights"], weights_only=False, map_location="cpu")["state_dict"]
            weights = {k.replace("model.", ""): v for k, v in checkpoint.items()}
            self.ae_model.load_state_dict(weights)
        if not self.is_train_decoder:
            for param in self.ae_model.parameters():
                param.requires_grad = False
            self.ae_model.eval()

    def train(self, mode: bool = True):
        """Override to keep frozen sub-modules in eval mode.

        Ensures frozen condition encoders (img_model, txt_model) and the
        frozen autoencoder remain in eval mode even when the parent module
        is switched to train mode.  This prevents stochastic-depth/dropout
        inside DINOv2 from perturbing the fixed condition features.
        """
        super().train(mode)
        if getattr(self, "img_model", None) is not None:
            self.img_model.eval()
        if getattr(self, "txt_model", None) is not None:
            self.txt_model.eval()
        if not getattr(self, "is_train_decoder", False) and getattr(self, "ae_model", None) is not None:
            self.ae_model.eval()
        return self

    def get_z(self, v_data: Dict[str, Any], v_test: bool) -> Dict[str, Tensor]:
        """Extract or load the padded face latent tensor.

        Supports two modes:
        - **Stored-z**: Latents are pre-computed and stored in the data dict.
          Samples from N(mean, std) or uses mean directly based on config.
        - **On-the-fly**: Encodes raw data through the frozen AE at inference.

        Args:
            v_data: Batch data dictionary containing either pre-computed
                ``face_features`` or raw geometry for AE encoding.
            v_test: Whether we are in test/inference mode.

        Returns:
            Dictionary with ``"padded_face_z"`` of shape ``(B, num_max_faces, dim_input)``
            and optionally ``"mask"`` for zero-pad mode.
        """
        data = {}
        if self.is_stored_z:
            face_features = v_data["face_features"]
            bs = face_features.shape[0]
            num_face = face_features.shape[1]
            mean = face_features[..., :32]
            std = face_features[..., 32:]
            if self.use_mean:
                face_features = mean
            else:
                face_features = mean + std * torch.randn_like(mean)
            data["padded_face_z"] = face_features.reshape(bs, num_face, -1)
        else:
            with torch.no_grad() and autocast(device_type='cuda', dtype=torch.float32):
                encoding_result = self.ae_model.encode(v_data, True)
                face_features, _, _ = self.ae_model.sample(
                    encoding_result["face_features"], v_is_test=self.use_mean
                )
            dim_latent = face_features.shape[-1]
            num_faces = v_data["num_face_record"]
            bs = num_faces.shape[0]
            # Fill face_z into padded tensor without for-loop
            if self.pad_method == "zero":
                padded_face_z = torch.zeros(
                    (bs, self.num_max_faces, dim_latent),
                    device=face_features.device,
                    dtype=face_features.dtype,
                )
                mask = num_faces[:, None] > torch.arange(self.num_max_faces, device=num_faces.device)
                padded_face_z[mask] = face_features
                data["padded_face_z"] = padded_face_z
                data["mask"] = mask
            else:
                positions = torch.arange(self.num_max_faces, device=face_features.device).unsqueeze(0).repeat(bs, 1)
                mandatory_mask = positions < num_faces[:, None]
                random_indices = (
                    torch.rand((bs, self.num_max_faces), device=face_features.device) * num_faces[:, None]
                ).long()
                indices = torch.where(mandatory_mask, positions, random_indices)
                num_faces_cum = num_faces.cumsum(dim=0).roll(1)
                num_faces_cum[0] = 0
                indices += num_faces_cum[:, None]
                # Permute the indices
                r_indices = torch.argsort(torch.rand((bs, self.num_max_faces), device=face_features.device), dim=1)
                indices = indices.gather(1, r_indices)
                data["padded_face_z"] = face_features[indices]
        return data

    def apply_denoising_backbone(self, noise_features: Tensor, img_memory: Tensor,
                                   topo_bias: Optional[Tensor] = None) -> Tensor:
        """Run the TransformerEncoder backbone with optional per-layer adapters.

        When ``layerwise_img_adapter`` is enabled, image adapters are inserted
        after every ``layerwise_img_adapter_every`` transformer layers.  Each
        adapter cross-attends the hidden state to the condition memory tokens.

        Args:
            noise_features: Input features of shape ``(B, S, dim_total)``.
            img_memory: Condition memory tokens of shape ``(B, T, dim_condition)``
                used by image adapters.  Ignored when adapters are disabled.
            topo_bias: Optional topology attention bias of shape ``(B, S, S)``.
                Positive values encourage attention between face pairs (adjacent
                faces), zero means no bias.  Applied as additive mask to self-
                attention logits.  When provided, we manually iterate over layers
                instead of calling ``self.net1(...)`` directly.

        Returns:
            Denoised features of shape ``(B, S, dim_total)``.
        """
        # Fast path: no adapters and no topology bias
        if not self.layerwise_img_adapter and topo_bias is None:
            return self.net1(noise_features)

        # Prepare topology mask for PyTorch's MultiheadAttention format
        # nn.TransformerEncoderLayer expects src_mask of shape [S, S] or [B*nhead, S, S]
        attn_mask = None
        if topo_bias is not None:
            B, S, _ = topo_bias.shape
            nhead = self.dim_total // 64  # same as nhead in TransformerEncoderLayer
            # Expand: [B, S, S] → [B*nhead, S, S]
            attn_mask = topo_bias.unsqueeze(1).expand(-1, nhead, -1, -1)
            attn_mask = attn_mask.reshape(B * nhead, S, S)

        output = noise_features
        adapter_idx = 0
        for layer_idx, layer in enumerate(self.net1.layers):
            output = layer(output, src_mask=attn_mask)
            if self.layerwise_img_adapter and (layer_idx + 1) % self.layerwise_img_adapter_every == 0:
                output = self.img_adapters[adapter_idx](output, img_memory)
                adapter_idx += 1
        if self.net1.norm is not None:
            output = self.net1.norm(output)
        return output
                adapter_idx += 1
        if self.net1.norm is not None:
            output = self.net1.norm(output)
        return output

    def diffuse(
        self,
        v_feature: Tensor,
        v_timesteps: Tensor,
        v_condition: Optional[Tensor] = None,
        v_align_feature: Optional[Tensor] = None,
        topo_bias: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Perform a single denoising step with cross-attention conditioning.

        This is the core diffusion forward pass:
        1. Compute timestep embedding
        2. Project noisy features through p_embed
        3. Cross-attend to condition via TransformerDecoder
        4. Compute alignment loss between clean CAD and condition embeddings
        5. Add timestep embedding and run denoising backbone
        6. Project to output space

        Args:
            v_feature: Noisy face latents, shape ``(B, num_faces, dim_input)``.
            v_timesteps: Diffusion timesteps, shape ``(B,)``.
            v_condition: Condition features, shape ``(B, 1, T, dim_condition)``
                or ``(B, T, dim_condition)``.
            v_align_feature: Clean face latents for alignment branch,
                shape ``(B, num_faces, dim_input)``.
            topo_bias: Optional topology attention bias, shape ``(B, num_faces, num_faces)``.
                Values in [0, 1] representing adjacency probability.  Applied with
                timestep-dependent weighting (stronger at high noise levels).

        Returns:
            Tuple of (predicted_x0, alignment_loss).
        """
        bs = v_feature.size(0)

        time_embeds = self.time_embed(
            sincos_embedding(v_timesteps, self.dim_total)
        ).unsqueeze(1)

        # -------------------------
        # Diffusion main: use noisy latent
        # -------------------------
        noise_features = self.p_embed(v_feature)

        assert v_condition is not None
        assert v_align_feature is not None

        cad_tgt = self.cross_attn_pre_proj(noise_features)  # (B, 30, 1024)

        # v_condition: [B, 1, 257, 1024] -> [B, 257, 1024]
        img_memory = v_condition[:, 0] if v_condition.dim() == 4 else v_condition

        # -------------------------
        # Alignment branch: use clean latent
        # -------------------------
        align_features = self.p_embed(v_align_feature)
        cad_align_src = self.cross_attn_pre_proj(align_features)  # (B, 30, 1024)

        cad_align_tokens = self.cad_align_proj(cad_align_src)  # [B, 30, 256]
        img_align_tokens = self.img_align_proj(img_memory)  # [B, 257, 256]

        cad_align_global = cad_align_tokens.mean(dim=1)  # [B, 256]
        img_align_global = img_align_tokens.mean(dim=1)  # [B, 256]

        cad_align_emb = F.normalize(self.cad_align_head(cad_align_global), dim=-1)
        img_align_emb = F.normalize(self.img_align_head(img_align_global), dim=-1)

        align_loss = clip_style_symmetric_infonce(cad_align_emb, img_align_emb)

        # -------------------------
        # Diffusion main branch: cross-attention
        # -------------------------
        noise_features_add_cond = self.cross_attn_add_cond(
            tgt=cad_tgt,
            memory=img_memory,
        )

        noise_features_add_cond = self.cross_attn_post_proj(noise_features_add_cond)

        noise_features = noise_features_add_cond + time_embeds

        # --- Apply topology bias with timestep-dependent weighting ---
        weighted_topo_bias = None
        if topo_bias is not None:
            # Higher noise (larger t) → stronger topology guidance
            # Lower noise (smaller t) → weaker guidance (let geometry refine freely)
            t_normalized = v_timesteps.float() / 1000.0  # [B], range 0~1
            topo_weight = t_normalized[:, None, None]  # [B, 1, 1]
            # Scale: adjacency (0~1) → attention bias magnitude
            topo_scale = getattr(self, 'topo_scale', 2.0)
            weighted_topo_bias = topo_bias * topo_weight * topo_scale  # [B, S, S]

        pred_x0 = self.apply_denoising_backbone(noise_features, img_memory, weighted_topo_bias)
        pred_x0 = self.fc_out(pred_x0)

        return pred_x0, align_loss

    def extract_condition(self, v_data: Dict[str, Any]) -> Optional[Tensor]:
        """Legacy condition extraction for backward compatibility.

        This method provides a fallback when ``condition_features`` is not
        supplied to ``forward()`` or ``inference()``.  It requires that the
        appropriate encoder sub-modules (img_model, point_model, txt_model)
        have been attached externally or via a subclass.

        Note:
            In the refactored architecture, condition extraction is handled by
            a separate ``ConditionExtractor`` class.  This method exists only
            for legacy/backward-compatible usage.

        Args:
            v_data: Batch data dictionary with condition inputs.

        Returns:
            Condition tensor of shape ``(B, 1, T, dim_condition)`` or
            ``(B, T, dim_condition)``, or None if no conditions are active.
        """
        condition = None
        if self.with_img:
            if "img_features" in v_data["conditions"]:
                img_feature = v_data["conditions"]["img_features"]
                num_imgs = img_feature.shape[1]
            else:
                imgs = v_data["conditions"]["imgs"]
                num_imgs = imgs.shape[1]
                imgs = imgs.reshape(-1, 3, 224, 224)
                img_feature = self.img_model(imgs).last_hidden_state
            img_idx = v_data["conditions"]["img_id"]
            img_feature = self.img_fc(img_feature)
            if img_idx.shape[-1] > 1:
                camera_embedding = self.camera_embedding(img_idx)
                img_feature = (
                    img_feature.reshape(-1, num_imgs, self.dim_condition) + camera_embedding
                ).mean(dim=1)
            else:
                img_feature = img_feature
            condition = img_feature[:, None]
        elif self.with_pc:
            pc = v_data["conditions"]["points"]
            feat = self.point_model(pc[:, 0, :, :], v_data["id_aug"])
            condition = feat[:, None]
        elif self.with_txt:
            if "txt_features" in v_data["conditions"]:
                txt_feat = v_data["conditions"]["txt_features"]
            else:
                txt = v_data["conditions"]["txt"]
                txt_feat = self.txt_model.encode(
                    txt, show_progress_bar=False, convert_to_numpy=False,
                    device=self.txt_model.device,
                )
                txt_feat = torch.stack(txt_feat, dim=0)
            condition = self.txt_fc(txt_feat)[:, None]
        return condition

    def forward(
        self,
        v_data: Dict[str, Any],
        v_test: bool = False,
        condition_features: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Training forward pass: add noise, denoise, compute losses.

        Args:
            v_data: Batch data dictionary with face geometry and conditions.
            v_test: Whether this is a test-time call (affects latent sampling).
            condition_features: Pre-extracted condition features from an external
                ConditionExtractor.  Shape ``(B, 1, T, D)`` or ``(B, T, D)``.
                If None, falls back to internal ``extract_condition()`` for
                backward compatibility.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Loss dictionary with keys:
            - ``"diffusion_loss"``: Main denoising loss
            - ``"align_loss"``: CLIP-style alignment loss
            - ``"classification"`` (zero-pad only): Face mask BCE loss
            - ``"total_loss"``: Sum of all losses
            - ``"t"``: Tensor of (timestep, per-sample loss) for diagnostics
        """
        encoding_result = self.get_z(v_data, v_test)
        face_z = encoding_result["padded_face_z"]
        device = face_z.device
        bs = face_z.size(0)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

        # Condition: use externally-provided features or fall back to legacy extraction
        if condition_features is None:
            condition = self.extract_condition(v_data)
        else:
            condition = condition_features

        noise = torch.randn(face_z.shape, device=device)
        noise_input = self.noise_scheduler.add_noise(face_z, noise, timesteps)

        # Model forward: denoise with alignment
        pred, align_loss = self.diffuse(
            noise_input,
            timesteps,
            condition,
            v_align_feature=face_z,  # Alignment branch uses clean latent
            topo_bias=v_data.get("face_adj", None),  # Topology bias if available
        )

        loss = {}
        loss_item = self.loss(pred, face_z if self.diffusion_type == "sample" else noise, reduction="none")
        loss["diffusion_loss"] = loss_item.mean()
        loss["align_loss"] = self.lambda_align * align_loss

        if self.pad_method == "zero":
            mask = torch.logical_not((face_z.abs() < 1e-4).all(dim=-1))
            label = self.classifier(pred)
            classification_loss = nn.functional.binary_cross_entropy_with_logits(label[..., 0], mask.float())
            if self.loss == nn.functional.l1_loss:
                classification_loss = classification_loss * 1e-1
            else:
                classification_loss = classification_loss * 1e-4
            loss["classification"] = classification_loss

        loss["total_loss"] = sum(loss.values())
        loss["t"] = torch.stack((timesteps, loss_item.mean(dim=1).mean(dim=1)), dim=1)

        if self.is_train_decoder:
            raise NotImplementedError("Decoder fine-tuning path not fully supported in refactored code.")

        return loss

    def inference(
        self,
        bs: int,
        device: torch.device,
        v_data: Optional[Dict[str, Any]] = None,
        condition_features: Optional[Tensor] = None,
        v_log: bool = True,
        **kwargs,
    ) -> List[Any]:
        """Run DDPM sampling loop to generate BRep face latents.

        Iteratively denoises random Gaussian noise into face latent sequences
        using the learned reverse diffusion process.

        Args:
            bs: Batch size (number of shapes to generate).
            device: Torch device for tensor allocation.
            v_data: Optional batch data for condition extraction (legacy path).
            condition_features: Pre-extracted condition features.  If None and
                conditions are active, falls back to ``extract_condition(v_data)``.
            v_log: Whether to show tqdm progress bar.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            List of reconstructed BRep data items (one per batch element),
            each produced by ``ae_model.inference()``.
        """
        face_features = torch.randn((bs, self.num_max_faces, self.dim_input)).to(device)

        # Resolve condition
        condition = condition_features
        if condition is None and (self.with_img or self.with_pc or self.with_txt):
            condition = self.extract_condition(v_data)
        if condition is not None:
            condition = condition[:bs]

        for t in tqdm(self.noise_scheduler.timesteps, disable=not v_log):
            timesteps = t.reshape(-1).to(device)
            pred_x0, _ = self.diffuse(face_features, timesteps, v_condition=condition, v_align_feature=face_features)
            face_features = self.noise_scheduler.step(pred_x0, t, face_features).prev_sample

        face_z = face_features
        if self.pad_method == "zero":
            label = torch.sigmoid(self.classifier(face_features))[..., 0]
            mask = label > 0.5
        else:
            mask = torch.ones_like(face_z[:, :, 0]).to(bool)

        recon_data = []
        for i in range(bs):
            face_z_item = face_z[i:i + 1][mask[i:i + 1]]
            if self.pad_method == "random":
                # Deduplicate faces by pairwise distance
                threshold = 1e-2
                max_faces = face_z_item.shape[0]
                index = torch.stack(
                    torch.meshgrid(torch.arange(max_faces), torch.arange(max_faces), indexing="ij"), dim=2
                )
                features = face_z_item[index]
                distance = (features[:, :, 0] - features[:, :, 1]).abs().mean(dim=-1)
                final_face_z = []
                for j in range(max_faces):
                    valid = True
                    for k in final_face_z:
                        if distance[j, k] < threshold:
                            valid = False
                            break
                    if valid:
                        final_face_z.append(j)
                face_z_item = face_z_item[final_face_z]
            data_item = self.ae_model.inference(face_z_item)
            recon_data.append(data_item)
        return recon_data


# ---------------------------------------------------------------------------
# DiffusionConcat: concatenation-based multimodal diffusion
# ---------------------------------------------------------------------------


class DiffusionConcat(nn.Module):
    """Concatenation-based multimodal conditioned diffusion model.

    This model concatenates a learned condition embedding to the noisy face
    latent along the feature dimension, then processes the combined
    representation with a single TransformerEncoder.  Supports simultaneous
    conditioning on multiple modalities (SVR, MVR, sketch, point cloud, text)
    through learned modality embeddings fused by an attention block.

    Corresponds to the original ``Diffusion_condition_mm`` class.

    Architecture:
        - **p_embed**: Linear(dim_input -> dim_latent) + LayerNorm + SiLU + Linear
        - **net1**: 24-layer TransformerEncoder operating on dim_total = dim_latent + dim_condition
        - **cond_attn**: 8-layer TransformerEncoder for fusing modality tokens
        - **Learned embeddings**: uncond, svr, mvr, sketch, pc, txt
        - **time_embed**: sincos -> MLP on dim_total
        - **fc_out**: Linear -> output projection (dim_total -> dim_input)

    Key Design:
        The ``forward()`` method accepts ``condition_features`` as an argument
        to decouple condition extraction from diffusion.  Falls back to
        internal ``extract_condition()`` when not provided.

    Args:
        cfg: Configuration dictionary (same keys as DiffusionCrossAttn plus
            ``cond_prob`` for modality dropout probabilities).
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.dim_input = 8 * 2 * 2  # Face latent dimensionality (32)
        self.dim_latent = cfg["diffusion_latent"]
        self.dim_condition = 256
        self.dim_total = self.dim_latent + self.dim_condition
        self.time_statics = [0 for _ in range(10)]

        # ----- Latent projection -----
        self.p_embed = nn.Sequential(
            nn.Linear(self.dim_input, self.dim_latent),
            nn.LayerNorm(self.dim_latent),
            nn.SiLU(),
            nn.Linear(self.dim_latent, self.dim_latent),
        )

        # ----- Denoising backbone: 24-layer TransformerEncoder on concatenated features -----
        layer1 = nn.TransformerEncoderLayer(
            d_model=self.dim_total,
            nhead=self.dim_total // 64,
            norm_first=True,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.net1 = nn.TransformerEncoder(layer1, 24, nn.LayerNorm(self.dim_total))

        # ----- Output projection -----
        self.fc_out = nn.Sequential(
            nn.Linear(self.dim_total, self.dim_total),
            nn.LayerNorm(self.dim_total),
            nn.SiLU(),
            nn.Linear(self.dim_total, self.dim_input),
        )

        # ----- Condition modality flags -----
        self.with_img = False
        self.with_pc = False
        self.with_txt = False
        self.is_aug = cfg["is_aug"]
        if "single_img" in cfg["condition"] or "multi_img" in cfg["condition"] or "sketch" in cfg["condition"]:
            self.with_img = True
        if "pc" in cfg["condition"]:
            self.with_pc = True
        if "txt" in cfg["condition"]:
            self.with_txt = True

        # ----- DDPM noise scheduler -----
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule=cfg["beta_schedule"],
            prediction_type=cfg["diffusion_type"],
            beta_start=cfg["beta_start"],
            beta_end=cfg["beta_end"],
            variance_type=cfg["variance_type"],
            clip_sample=False,
        )

        # ----- Timestep embedding: sincos -> MLP on dim_total -----
        self.time_embed = nn.Sequential(
            nn.Linear(self.dim_total, self.dim_total),
            nn.LayerNorm(self.dim_total),
            nn.SiLU(),
            nn.Linear(self.dim_total, self.dim_total),
        )

        # ----- Hyperparameters -----
        self.num_max_faces = cfg["num_max_faces"]
        self.loss = nn.functional.l1_loss if cfg["loss"] == "l1" else nn.functional.mse_loss
        self.diffusion_type = cfg["diffusion_type"]
        self.pad_method = cfg.get("pad_method", "random")

        # ----- Frozen pretrained AutoEncoder -----
        model_mod = importlib.import_module("src.brepnet.model")
        model_cls = getattr(model_mod, cfg["autoencoder"])
        self.ae_model = model_cls(cfg)

        self.is_pretrained = cfg["autoencoder_weights"] is not None
        self.is_stored_z = cfg["stored_z"]
        self.use_mean = cfg["use_mean"]
        self.is_train_decoder = cfg["train_decoder"]
        if self.is_pretrained:
            checkpoint = torch.load(cfg["autoencoder_weights"], weights_only=False)["state_dict"]
            weights = {k.replace("model.", ""): v for k, v in checkpoint.items()}
            self.ae_model.load_state_dict(weights)
        if not self.is_train_decoder:
            for param in self.ae_model.parameters():
                param.requires_grad = False
            self.ae_model.eval()

        # ----- Modality fusion: 8-layer TransformerEncoder + learned embeddings -----
        layer = nn.TransformerEncoderLayer(
            d_model=self.dim_condition,
            nhead=8,
            norm_first=True,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.cond_attn = nn.TransformerEncoder(layer, 8, nn.LayerNorm(self.dim_condition))

        # Learned modality embeddings
        self.learned_uncond_emb = nn.Parameter(torch.rand(self.dim_condition))
        self.learned_svr_emb = nn.Parameter(torch.rand(self.dim_condition))
        self.learned_mvr_emb = nn.Parameter(torch.rand(self.dim_condition))
        self.learned_sketch_emb = nn.Parameter(torch.rand(self.dim_condition))
        self.learned_pc_emb = nn.Parameter(torch.rand(self.dim_condition))
        self.learned_txt_emb = nn.Parameter(torch.rand(self.dim_condition))

        self.condition = cfg["condition"]
        self.cond_prob = cfg["cond_prob"]
        self.cond_prob_acc = np.cumsum(self.cond_prob)

    def get_z(self, v_data: Dict[str, Any], v_test: bool) -> Dict[str, Tensor]:
        """Extract or load the padded face latent tensor.

        Identical logic to DiffusionCrossAttn.get_z -- supports both stored-z
        and on-the-fly encoding modes.

        Args:
            v_data: Batch data dictionary.
            v_test: Whether in test mode.

        Returns:
            Dictionary with ``"padded_face_z"`` and optionally ``"mask"``.
        """
        data = {}
        if self.is_stored_z:
            face_features = v_data["face_features"]
            bs = face_features.shape[0]
            num_face = face_features.shape[1]
            mean = face_features[..., :32]
            std = face_features[..., 32:]
            if self.use_mean:
                face_features = mean
            else:
                face_features = mean + std * torch.randn_like(mean)
            data["padded_face_z"] = face_features.reshape(bs, num_face, -1)
        else:
            with torch.no_grad() and autocast(device_type='cuda', dtype=torch.float32):
                encoding_result = self.ae_model.encode(v_data, True)
                face_features, _, _ = self.ae_model.sample(
                    encoding_result["face_features"], v_is_test=self.use_mean
                )
            dim_latent = face_features.shape[-1]
            num_faces = v_data["num_face_record"]
            bs = num_faces.shape[0]
            if self.pad_method == "zero":
                padded_face_z = torch.zeros(
                    (bs, self.num_max_faces, dim_latent),
                    device=face_features.device,
                    dtype=face_features.dtype,
                )
                mask = num_faces[:, None] > torch.arange(self.num_max_faces, device=num_faces.device)
                padded_face_z[mask] = face_features
                data["padded_face_z"] = padded_face_z
                data["mask"] = mask
            else:
                positions = torch.arange(self.num_max_faces, device=face_features.device).unsqueeze(0).repeat(bs, 1)
                mandatory_mask = positions < num_faces[:, None]
                random_indices = (
                    torch.rand((bs, self.num_max_faces), device=face_features.device) * num_faces[:, None]
                ).long()
                indices = torch.where(mandatory_mask, positions, random_indices)
                num_faces_cum = num_faces.cumsum(dim=0).roll(1)
                num_faces_cum[0] = 0
                indices += num_faces_cum[:, None]
                # Permute the indices
                r_indices = torch.argsort(
                    torch.rand((bs, self.num_max_faces), device=face_features.device), dim=1
                )
                indices = indices.gather(1, r_indices)
                data["padded_face_z"] = face_features[indices]
        return data

    def diffuse(
        self,
        v_feature: Tensor,
        v_timesteps: Tensor,
        v_condition: Optional[Tensor] = None,
    ) -> Tensor:
        """Perform a single denoising step with concatenated conditioning.

        The condition embedding is tiled to match the face sequence length and
        concatenated along the feature dimension before being processed by
        the TransformerEncoder backbone.

        Args:
            v_feature: Noisy face latents, shape ``(B, num_faces, dim_input)``.
            v_timesteps: Diffusion timesteps, shape ``(B,)``.
            v_condition: Condition embedding, shape ``(B, 1, dim_condition)``.
                If None, uses zero conditioning (unconditional generation).

        Returns:
            Predicted x0 of shape ``(B, num_faces, dim_input)``.
        """
        bs = v_feature.size(0)
        de = v_feature.device
        dt = v_feature.dtype

        time_embeds = self.time_embed(sincos_embedding(v_timesteps, self.dim_total)).unsqueeze(1)
        noise_features = self.p_embed(v_feature)

        # Default to zero condition if none provided
        v_condition = (
            torch.zeros((bs, 1, self.dim_condition), device=de, dtype=dt)
            if v_condition is None
            else v_condition
        )
        # Tile condition to match sequence length and concatenate
        v_condition = v_condition.repeat(1, v_feature.shape[1], 1)
        noise_features = torch.cat([noise_features, v_condition], dim=-1)
        noise_features = noise_features + time_embeds

        pred_x0 = self.net1(noise_features)
        pred_x0 = self.fc_out(pred_x0)
        return pred_x0

    def extract_condition(self, v_data: Dict[str, Any], v_test: bool = False) -> Optional[Tensor]:
        """Legacy condition extraction for backward compatibility.

        Assembles modality tokens from learned embeddings + encoder outputs,
        fuses them via the cond_attn TransformerEncoder, and returns a
        pooled condition vector.

        Note:
            Requires that encoder sub-modules (img_model, point_model, txt_model)
            be attached externally or via a subclass.

        Args:
            v_data: Batch data dictionary with condition inputs.
            v_test: Whether in test/inference mode.

        Returns:
            Condition tensor of shape ``(B, 1, dim_condition)``.
        """
        bs = len(v_data["v_prefix"])
        device = self.learned_uncond_emb.device
        dtype = self.learned_uncond_emb.dtype

        condition = torch.stack([
            self.learned_svr_emb, self.learned_mvr_emb, self.learned_sketch_emb,
            self.learned_pc_emb, self.learned_txt_emb,
        ], dim=0)[None, :].repeat(bs, 1, 1)

        all_condition_names = v_data["conditions"]["names"]
        condition_batch_id = v_data["conditions"]["id_batch"]

        # TXT feature
        if "txt" in all_condition_names:
            if v_data["conditions"]["txt_features"] != []:
                txt_feat = v_data["conditions"]["txt_features"]
            else:
                txt = v_data["conditions"]["txt"]
                txt_feat = self.txt_model.encode(
                    txt, show_progress_bar=False, convert_to_numpy=False,
                    device=self.txt_model.device,
                )
                txt_feat = torch.stack(txt_feat, dim=0)
            txt_features = self.txt_fc(txt_feat)
            condition[condition_batch_id["txt"], 4] = txt_features.to(dtype)

        # PC feature
        if "pc" in all_condition_names:
            pc = v_data["conditions"]["points"]
            pc_feat = self.point_model(pc, v_data["id_aug"] if self.is_aug else None)
            condition[condition_batch_id["pc"], 3] = pc_feat.to(dtype)

        if v_data["conditions"]["img_id"] != []:
            svr_feature, sketch_feature, mvr_feature = self.img_model(v_data)
            condition[condition_batch_id["single_img"], 0] = svr_feature.to(dtype)
            condition[condition_batch_id["multi_img"], 1] = mvr_feature.to(dtype)
            condition[condition_batch_id["sketch"], 2] = sketch_feature.to(dtype)

        condition = self.cond_attn(condition)
        condition = condition.mean(dim=1, keepdim=True)
        return condition

    def forward(
        self,
        v_data: Dict[str, Any],
        v_test: bool = False,
        condition_features: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Training forward pass: add noise, denoise, compute losses.

        Args:
            v_data: Batch data dictionary.
            v_test: Whether in test mode.
            condition_features: Pre-extracted condition features of shape
                ``(B, 1, dim_condition)``.  If None, falls back to internal
                ``extract_condition()`` for backward compatibility.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Loss dictionary with keys:
            - ``"diffusion_loss"``: Main denoising loss
            - ``"total_loss"``: Sum of all losses
            - ``"t"``: Diagnostic (timestep, per-sample loss) tensor
            - Per-modality loss breakdowns (svr, mvr, sketch, pc, txt, uncond, mm)
        """
        encoding_result = self.get_z(v_data, v_test)
        face_z = encoding_result["padded_face_z"]
        device = face_z.device
        bs = face_z.size(0)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

        # Condition: use externally-provided features or fall back to legacy extraction
        if condition_features is None:
            condition = self.extract_condition(v_data)
        else:
            condition = condition_features

        noise = torch.randn(face_z.shape, device=device)
        noise_input = self.noise_scheduler.add_noise(face_z, noise, timesteps)

        # Model forward
        pred = self.diffuse(noise_input, timesteps, condition)

        loss = {}
        loss_item = self.loss(pred, face_z if self.diffusion_type == "sample" else noise, reduction="none")
        loss["diffusion_loss"] = loss_item.mean()
        loss["total_loss"] = sum(loss.values())

        loss["t"] = torch.stack((timesteps, loss_item.mean(dim=1).mean(dim=1)), dim=1)
        loss_item_per_sample = loss_item.mean(dim=1).mean(dim=1)

        # Per-modality loss breakdown
        uncond_mask = [item == "uncond" for item in v_data["conditions"]["names"]]
        loss["uncond_count"] = sum(uncond_mask)
        if loss["uncond_count"] > 0:
            loss["uncond_diffusion_loss"] = loss_item_per_sample[uncond_mask].mean()

        mm_mask = [item == "mm" for item in v_data["conditions"]["names"]]
        loss["mm_count"] = sum(mm_mask)
        if loss["mm_count"] > 0:
            loss["mm_diffusion_loss"] = loss_item_per_sample[mm_mask].mean()

        svr_mask = [item == "single_img" for item in v_data["conditions"]["names"]]
        loss["svr_count"] = sum(svr_mask)
        if loss["svr_count"] > 0:
            loss["svr_diffusion_loss"] = loss_item_per_sample[svr_mask].mean()

        mvr_mask = [item == "multi_img" for item in v_data["conditions"]["names"]]
        loss["mvr_count"] = sum(mvr_mask)
        if loss["mvr_count"] > 0:
            loss["mvr_diffusion_loss"] = loss_item_per_sample[mvr_mask].mean()

        sketch_mask = [item == "sketch" for item in v_data["conditions"]["names"]]
        loss["sketch_count"] = sum(sketch_mask)
        if loss["sketch_count"] > 0:
            loss["sketch_diffusion_loss"] = loss_item_per_sample[sketch_mask].mean()

        pc_mask = [item == "pc" for item in v_data["conditions"]["names"]]
        loss["pc_count"] = sum(pc_mask)
        if loss["pc_count"] > 0:
            loss["pc_diffusion_loss"] = loss_item_per_sample[pc_mask].mean()

        txt_mask = [item == "txt" for item in v_data["conditions"]["names"]]
        loss["txt_count"] = sum(txt_mask)
        if loss["txt_count"] > 0:
            loss["txt_diffusion_loss"] = loss_item_per_sample[txt_mask].mean()

        if self.is_train_decoder:
            raise NotImplementedError("Decoder fine-tuning path not fully supported in refactored code.")

        return loss

    def inference(
        self,
        bs: int,
        device: torch.device,
        v_data: Optional[Dict[str, Any]] = None,
        condition_features: Optional[Tensor] = None,
        v_log: bool = True,
        **kwargs,
    ) -> List[Any]:
        """Run DDPM sampling loop to generate BRep face latents.

        Args:
            bs: Batch size (number of shapes to generate).
            device: Torch device for tensor allocation.
            v_data: Optional batch data for condition extraction (legacy path).
            condition_features: Pre-extracted condition features.  If None and
                conditions are active, falls back to ``extract_condition(v_data)``.
            v_log: Whether to show tqdm progress bar.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            List of reconstructed BRep data items (one per batch element).
        """
        face_features = torch.randn((bs, self.num_max_faces, self.dim_input)).to(device)

        # Resolve condition
        condition = condition_features
        if condition is None and (self.with_img or self.with_pc or self.with_txt):
            condition = self.extract_condition(v_data)
        if condition is not None:
            condition = condition[:bs]

        for t in tqdm(self.noise_scheduler.timesteps, disable=not v_log):
            timesteps = t.reshape(-1).to(device)
            pred_x0 = self.diffuse(face_features, timesteps, v_condition=condition)
            face_features = self.noise_scheduler.step(pred_x0, t, face_features).prev_sample

        face_z = face_features
        mask = torch.ones_like(face_z[:, :, 0]).to(bool)

        recon_data = []
        for i in range(bs):
            face_z_item = face_z[i:i + 1][mask[i:i + 1]]
            # Deduplicate faces by pairwise distance
            threshold = 1e-2
            max_faces = face_z_item.shape[0]
            index = torch.stack(
                torch.meshgrid(torch.arange(max_faces), torch.arange(max_faces), indexing="ij"), dim=2
            )
            features = face_z_item[index]
            distance = (features[:, :, 0] - features[:, :, 1]).abs().mean(dim=-1)
            final_face_z = []
            for j in range(max_faces):
                valid = True
                for k in final_face_z:
                    if distance[j, k] < threshold:
                        valid = False
                        break
                if valid:
                    final_face_z.append(j)
            face_z_item = face_z_item[final_face_z]
            data_item = self.ae_model.inference(face_z_item)
            recon_data.append(data_item)
        return recon_data


# ---------------------------------------------------------------------------
# Backward-compatible aliases (legacy class names)
# ---------------------------------------------------------------------------

Diffusion_condition = DiffusionCrossAttn
Diffusion_condition_mm = DiffusionConcat
Diffusion_condition_multimodal = DiffusionConcat
