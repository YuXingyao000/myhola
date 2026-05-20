"""
Training strategies for bridging the domain gap between real photos and CAD
generation in the HoLa-BRep pipeline.

Problem Statement:
    The 2-stage pipeline (VAE encoder -> diffusion generator) works well when
    conditioned on DINOv2 features from synthetic white-model renders, but
    degrades significantly on real photographs. The core issue is that DINOv2
    features from natural images lack the precise 3D geometry cues present in
    clean white-model renders.

    We have PAIRED DATA: the same CAD object rendered as both a white-model
    image and a realistic photo (via FLUX.1-Kontext), enabling supervised
    domain adaptation.

Three strategies are provided:
    A) KnowledgeDistillation  - Teacher-student distillation
    B) FeatureDomainMapper    - Lightweight feature-space translator
    C) TwoStagePipeline       - Inference-time orchestrator (Image -> PC -> BRep)
"""

from __future__ import annotations

import copy
import math
import types
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from diffusers import DDPMScheduler
from tqdm import tqdm

try:
    from src.brepnet.models.blocks import sincos_embedding
except (ImportError, ModuleNotFoundError):
    # Fallback: sincos_embedding is defined in the main diffusion_model module
    from src.brepnet.diffusion_model import sincos_embedding

if TYPE_CHECKING:
    from src.brepnet.models.diffusion import DiffusionCrossAttn


# =============================================================================
# Strategy A: Knowledge Distillation
# =============================================================================


class KnowledgeDistillation(nn.Module):
    """Teacher-student knowledge distillation for real-photo conditioned diffusion.

    Motivation:
        A diffusion model conditioned on white-model DINOv2 features (the
        "teacher") already produces high-quality B-Rep latents. We want a
        "student" model that achieves similar quality when conditioned on
        real-photo DINOv2 features. Direct training on real photos with only
        diffusion loss is insufficient because the noisy gradient signal cannot
        compensate for the large domain gap in conditioning features.

        By distilling the teacher's per-sample denoising predictions into the
        student, we provide a dense, geometry-aware supervision signal that
        teaches the student to interpret real-photo features as if they were
        white-model features.

    Architecture:
        - Teacher: frozen ``Diffusion_condition`` loaded from a white-model
          checkpoint. Receives white-model DINOv2 features as condition.
        - Student: trainable ``Diffusion_condition`` initialized as a deepcopy
          of the teacher. Only cross-attention layers, alignment heads, and the
          image projection MLP are reset to fresh parameters; the denoising
          backbone retains teacher weights for stable initialization.
          Receives real-photo DINOv2 features as condition.

    Training Protocol:
        Each training step uses the SAME noisy latent and timestep for both
        teacher and student, ensuring the distillation target is consistent.

        Loss = kd_weight * MSE(student_pred, teacher_pred)
             + diffusion_weight * MSE(student_pred, noise_or_x0)
             + align_weight * align_loss

    Usage:
        >>> strategy = KnowledgeDistillation.from_checkpoint(
        ...     teacher_ckpt="path/to/white_model_diffusion.ckpt",
        ...     model_conf=conf_dict,
        ...     kd_weight=1.0, diffusion_weight=1.0, align_weight=0.1,
        ... )
        >>> loss_dict = strategy.training_step(batch)
        >>> recon = strategy.inference(batch, device)
    """

    def __init__(
        self,
        teacher: "DiffusionCrossAttn",
        student: "DiffusionCrossAttn",
        kd_weight: float = 1.0,
        diffusion_weight: float = 1.0,
        align_weight: float = 0.1,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.kd_weight = kd_weight
        self.diffusion_weight = diffusion_weight
        self.align_weight = align_weight

        # Freeze teacher entirely
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    @classmethod
    def from_checkpoint(
        cls,
        teacher_ckpt: str,
        model_conf: Dict[str, Any],
        kd_weight: float = 1.0,
        diffusion_weight: float = 1.0,
        align_weight: float = 0.1,
    ) -> "KnowledgeDistillation":
        """Construct from a teacher checkpoint path.

        Args:
            teacher_ckpt: Path to a Lightning checkpoint or state_dict file
                containing the trained white-model-conditioned diffusion model.
            model_conf: Configuration dict used to instantiate
                ``Diffusion_condition``.
            kd_weight: Weight for knowledge distillation loss term.
            diffusion_weight: Weight for standard diffusion loss term.
            align_weight: Weight for cross-modal alignment loss term.

        Returns:
            Initialized KnowledgeDistillation strategy.
        """
        from src.brepnet.diffusion_model import Diffusion_condition

        # Build teacher and load weights
        # Supports both the original Diffusion_condition and refactored DiffusionCrossAttn
        teacher = Diffusion_condition(model_conf)
        ckpt = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        # Handle Lightning prefix "model."
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        teacher.load_state_dict(state_dict, strict=False)

        # Build student as deepcopy then reset adaptation layers
        student = copy.deepcopy(teacher)
        cls._reset_adaptation_layers(student)

        return cls(
            teacher=teacher,
            student=student,
            kd_weight=kd_weight,
            diffusion_weight=diffusion_weight,
            align_weight=align_weight,
        )

    @staticmethod
    def _reset_adaptation_layers(model: "DiffusionCrossAttn") -> None:
        """Reset cross-attention and alignment layers to fresh random weights.

        The denoising backbone (self-attention transformer encoder) keeps the
        teacher's pretrained weights for a warm start. Only the layers that
        must adapt to the new conditioning domain are re-initialized:
            - cross_attn_add_cond (cross-attention decoder)
            - cross_attn_pre_proj, cross_attn_post_proj
            - img_fc (image feature projection)
            - cad_align_proj, img_align_proj
            - cad_align_head, img_align_head
            - img_adapters (if layerwise adapters are used)
        """
        layers_to_reset = [
            "cross_attn_add_cond",
            "cross_attn_pre_proj",
            "cross_attn_post_proj",
            "img_fc",
            "cad_align_proj",
            "img_align_proj",
            "cad_align_head",
            "img_align_head",
        ]
        for name in layers_to_reset:
            module = getattr(model, name, None)
            if module is not None:
                _reinitialize_module(module)

        # Reset layerwise image adapters if present
        if hasattr(model, "img_adapters"):
            _reinitialize_module(model.img_adapters)

    def train(self, mode: bool = True):
        """Override to keep teacher frozen and its sub-encoders in eval mode."""
        super().train(mode)
        # Teacher is always eval
        self.teacher.eval()
        # Student frozen sub-models stay eval
        if hasattr(self.student, "img_model") and self.student.img_model is not None:
            self.student.img_model.eval()
        if hasattr(self.student, "ae_model") and not self.student.is_train_decoder:
            self.student.ae_model.eval()
        return self

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """Execute one training step with paired white-model and real-photo data.

        Expected batch keys:
            - "face_features": [B, num_faces, dim] pre-computed VAE latents
            - "conditions": dict containing:
                - "imgs" or "img_features": real-photo images/features for student
            - "conditions_white": dict containing:
                - "imgs" or "img_features": white-model images/features for teacher

        Returns:
            Dict of loss tensors including:
                - "kd_loss": distillation loss
                - "diffusion_loss": standard denoising loss
                - "align_loss": cross-modal alignment loss
                - "total_loss": weighted sum
        """
        # Get clean latents
        encoding_result = self.student.get_z(batch, v_test=False)
        face_z = encoding_result["padded_face_z"]
        device = face_z.device
        bs = face_z.size(0)

        # Sample shared timesteps and noise
        timesteps = torch.randint(
            0, self.student.noise_scheduler.config.num_train_timesteps,
            (bs,), device=device,
        ).long()
        noise = torch.randn_like(face_z)
        noisy_latent = self.student.noise_scheduler.add_noise(face_z, noise, timesteps)

        # --- Teacher forward (frozen, white-model condition) ---
        with torch.no_grad():
            # Temporarily swap conditions to white-model for teacher
            batch_teacher = {**batch, "conditions": batch["conditions_white"]}
            teacher_condition = self.teacher.extract_condition(batch_teacher)
            teacher_pred, _ = self.teacher.diffuse(
                noisy_latent, timesteps, teacher_condition, v_align_feature=face_z,
            )

        # --- Student forward (trainable, real-photo condition) ---
        student_condition = self.student.extract_condition(batch)
        student_pred, align_loss = self.student.diffuse(
            noisy_latent, timesteps, student_condition, v_align_feature=face_z,
        )

        # --- Compute losses ---
        # Diffusion target depends on prediction type
        if self.student.diffusion_type == "sample":
            diffusion_target = face_z
        else:
            diffusion_target = noise

        kd_loss = F.mse_loss(student_pred, teacher_pred)
        diffusion_loss = F.mse_loss(student_pred, diffusion_target)

        total_loss = (
            self.kd_weight * kd_loss
            + self.diffusion_weight * diffusion_loss
            + self.align_weight * align_loss
        )

        return {
            "kd_loss": kd_loss,
            "diffusion_loss": diffusion_loss,
            "align_loss": align_loss,
            "total_loss": total_loss,
            "t": torch.stack((
                timesteps,
                F.mse_loss(student_pred, diffusion_target, reduction="none")
                    .mean(dim=-1).mean(dim=-1),
            ), dim=1),
        }

    @torch.no_grad()
    def inference(
        self,
        batch: Dict[str, Any],
        device: torch.device,
        num_steps: Optional[int] = None,
    ) -> List[Any]:
        """Run inference using the student model conditioned on real photos.

        Args:
            batch: Input batch with real-photo conditions.
            device: Target device for generation.
            num_steps: If provided, override the default number of diffusion
                timesteps (useful for accelerated sampling).

        Returns:
            List of reconstructed B-Rep data (one per batch element).
        """
        self.student.eval()
        bs = len(batch["v_prefix"]) if "v_prefix" in batch else 1
        return self.student.inference(bs, device, v_data=batch, v_log=False)


# =============================================================================
# Strategy B: Feature Domain Mapper
# =============================================================================


class FeatureDomainMapper(nn.Module):
    """Lightweight transformer that maps real-photo DINOv2 features into the
    white-model feature space.

    Motivation:
        Instead of retraining the entire diffusion model, we can insert a
        small "adapter" network BEFORE the diffusion model's conditioning
        path. This mapper learns to translate DINOv2 tokens from real photos
        so that they resemble those from white-model renders. The diffusion
        model then operates in its comfort zone without any fine-tuning.

        This approach is:
        - Cheap to train (no diffusion model in the loop)
        - Easy to deploy (just prepend to the feature extraction pipeline)
        - Reversible (remove the mapper to go back to white-model conditioning)

    Architecture:
        A 6-layer transformer encoder with gated residual connections:
            mapped = gate * real_tokens + (1 - gate) * transformer(real_tokens)

        The gate is a learnable scalar initialized to 0.9, meaning the network
        starts as a near-identity mapping and gradually learns the correction.

    Input:  [B, 257, 1024] raw DINOv2 patch tokens from a real photo
    Output: [B, 257, 1024] tokens that approximate white-model DINOv2 features

    Training Loss:
        L = MSE(mapped, white_feat) + cosine_weight * (1 - cosine_sim).mean()

    Deployment:
        After training, call ``plug_into_pipeline(diffusion_model)`` to inject
        the mapper into an existing Diffusion_condition model transparently.

    Usage:
        >>> mapper = FeatureDomainMapper(dim=1024, num_tokens=257, num_layers=6)
        >>> loss_dict = mapper.training_step(batch)  # standalone training
        >>> mapper.plug_into_pipeline(diffusion_model)  # deploy
    """

    def __init__(
        self,
        dim: int = 1024,
        num_tokens: int = 257,
        num_layers: int = 6,
        num_heads: int = 16,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        gate_init: float = 0.9,
        cosine_weight: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.cosine_weight = cosine_weight

        # Learnable gated residual: starts near identity
        # gate ~ 0.9 means output is mostly the original input at init
        self._gate_logit = nn.Parameter(
            torch.tensor(_inverse_sigmoid(gate_init))
        )

        # Input/output layer norms for stability
        self.input_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)

        # 6-layer transformer encoder for feature transformation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(dim),
        )

        # Lightweight input projection (helps with domain shift)
        self.input_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Output projection back to feature space
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Learnable positional embedding for token positions
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_tokens, dim) * 0.02
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize output projection to near-zero for identity-like start."""
        # Make output_proj start near zero so initial output ~= input
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    @property
    def gate(self) -> Tensor:
        """Gating coefficient in [0, 1], controlling residual vs transformed."""
        return torch.sigmoid(self._gate_logit)

    def forward(self, real_features: Tensor) -> Tensor:
        """Map real-photo DINOv2 tokens toward white-model feature space.

        Args:
            real_features: [B, 257, 1024] DINOv2 tokens from real photo.

        Returns:
            [B, 257, 1024] mapped tokens approximating white-model features.
        """
        # Input projection with normalization
        x = self.input_norm(real_features)
        x = self.input_proj(x)

        # Add positional embedding
        x = x + self.pos_embed[:, : x.size(1), :]

        # Transformer feature transformation
        x = self.transformer(x)

        # Output projection
        x = self.output_proj(x)
        x = self.output_norm(x)

        # Gated residual: smoothly interpolate between original and mapped
        gate = self.gate
        mapped = gate * real_features + (1.0 - gate) * x

        return mapped

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """Standalone training step using paired real/white-model features.

        Expected batch keys:
            - "real_features": [B, 257, 1024] DINOv2 tokens from real photos
            - "white_features": [B, 257, 1024] DINOv2 tokens from white-model

        Returns:
            Dict containing:
                - "mse_loss": per-token MSE reconstruction
                - "cosine_loss": 1 - cosine similarity (token-level)
                - "total_loss": weighted combination
                - "gate_value": current gate value for monitoring
        """
        real_feat = batch["real_features"]
        white_feat = batch["white_features"]

        mapped_feat = self.forward(real_feat)

        # MSE loss: element-wise reconstruction
        mse_loss = F.mse_loss(mapped_feat, white_feat)

        # Cosine similarity loss: directional alignment per token
        # Flatten to [B*257, 1024] for cosine computation
        mapped_flat = mapped_feat.reshape(-1, self.dim)
        white_flat = white_feat.reshape(-1, self.dim)
        cosine_sim = F.cosine_similarity(mapped_flat, white_flat, dim=-1)
        cosine_loss = (1.0 - cosine_sim).mean()

        total_loss = mse_loss + self.cosine_weight * cosine_loss

        return {
            "mse_loss": mse_loss,
            "cosine_loss": cosine_loss,
            "total_loss": total_loss,
            "gate_value": self.gate.detach(),
        }

    @torch.no_grad()
    def inference(
        self,
        batch: Dict[str, Any],
        device: torch.device,
    ) -> Tensor:
        """Map real features to white-model space (inference mode).

        Args:
            batch: Must contain "real_features" key.
            device: Target device.

        Returns:
            Mapped feature tensor [B, 257, 1024].
        """
        self.eval()
        real_feat = batch["real_features"].to(device)
        return self.forward(real_feat)

    def plug_into_pipeline(self, diffusion_model: "DiffusionCrossAttn") -> None:
        """Inject this mapper into an existing diffusion model's conditioning path.

        After calling this method, the diffusion model will automatically
        apply the domain mapping to raw DINOv2 features before using them
        for cross-attention conditioning. This is transparent to the rest
        of the pipeline.

        Implementation:
            Monkey-patches ``diffusion_model.extract_condition`` to insert
            the mapper after DINOv2 feature extraction but before the
            img_fc projection.

        Args:
            diffusion_model: A trained ``Diffusion_condition`` instance
                (typically the white-model-conditioned one).

        Example:
            >>> mapper = FeatureDomainMapper.load_trained("mapper.pt")
            >>> mapper.plug_into_pipeline(my_diffusion_model)
            >>> # Now my_diffusion_model accepts real photos transparently
            >>> output = my_diffusion_model.inference(bs=4, device="cuda", v_data=batch)
        """
        mapper = self
        mapper.eval()
        for p in mapper.parameters():
            p.requires_grad = False

        # Store reference on the diffusion model for serialization awareness
        diffusion_model._feature_domain_mapper = mapper

        # Save original extract_condition
        original_extract_condition = diffusion_model.extract_condition

        def patched_extract_condition(v_data, **kwargs):
            """Wrapped extract_condition that applies domain mapping."""
            # If pre-computed features are provided, map them
            if "img_features" in v_data.get("conditions", {}):
                raw_feat = v_data["conditions"]["img_features"]
                # raw_feat shape: [B*num_imgs, 257, 1024] or [B, num_imgs, 257, 1024]
                original_shape = raw_feat.shape
                if raw_feat.dim() == 4:
                    b, n, t, d = raw_feat.shape
                    raw_feat = raw_feat.reshape(b * n, t, d)
                    mapped = mapper(raw_feat)
                    v_data["conditions"]["img_features"] = mapped.reshape(
                        original_shape
                    )
                elif raw_feat.dim() == 3:
                    v_data["conditions"]["img_features"] = mapper(raw_feat)

            # If raw images are provided, we need to intercept after DINOv2
            # extraction. We do this by extracting features first, mapping,
            # then storing them back as pre-computed features.
            elif "imgs" in v_data.get("conditions", {}):
                imgs = v_data["conditions"]["imgs"]
                num_imgs = imgs.shape[1] if imgs.dim() == 5 else 1
                imgs_flat = imgs.reshape(-1, 3, 224, 224)

                with torch.no_grad():
                    if hasattr(diffusion_model.img_model, "forward_features"):
                        # Depth-Anything-V2 style backbone
                        backbone_out = diffusion_model.img_model.forward_features(
                            imgs_flat
                        )
                        raw_feat = torch.cat([
                            backbone_out["x_norm_clstoken"].unsqueeze(1),
                            backbone_out["x_norm_patchtokens"],
                        ], dim=1)
                    else:
                        # HuggingFace DINOv2
                        raw_feat = diffusion_model.img_model(
                            imgs_flat
                        ).last_hidden_state

                # Apply domain mapping
                mapped_feat = mapper(raw_feat)

                # Store as pre-computed features and remove raw images
                # so the original extract_condition uses the mapped features
                v_data["conditions"]["img_features"] = mapped_feat.reshape(
                    -1, num_imgs, mapped_feat.shape[-2], mapped_feat.shape[-1]
                ) if num_imgs > 1 else mapped_feat

            return original_extract_condition(v_data, **kwargs)

        # Apply the patch
        diffusion_model.extract_condition = types.MethodType(
            lambda self, v_data, **kw: patched_extract_condition(v_data, **kw),
            diffusion_model,
        )

    @classmethod
    def load_trained(cls, path: str, **kwargs) -> "FeatureDomainMapper":
        """Load a trained mapper from disk.

        Args:
            path: Path to saved state dict.

        Returns:
            FeatureDomainMapper with loaded weights in eval mode.
        """
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Extract hyperparameters if stored, otherwise use defaults
        hparams = state.pop("_hparams", {})
        model = cls(**{**kwargs, **hparams})
        model.load_state_dict(state.get("model_state_dict", state))
        model.eval()
        return model

    def save(self, path: str) -> None:
        """Save mapper with hyperparameters for easy reloading."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "_hparams": {
                "dim": self.dim,
                "num_tokens": self.num_tokens,
                "cosine_weight": self.cosine_weight,
            },
        }, path)


# =============================================================================
# Strategy C: Two-Stage Pipeline (Image -> Point Cloud -> B-Rep)
# =============================================================================


class TwoStagePipeline(nn.Module):
    """Inference-time orchestrator: Real Photo -> Point Cloud -> B-Rep.

    Motivation:
        When neither distillation data nor paired features are available at
        inference time, we can decompose the problem into two well-solved
        sub-problems:
            1. Image-to-3D: Extract a point cloud from the real photo using
               an off-the-shelf model (TRELLIS, InstantMesh, etc.)
            2. PC-to-BRep: Feed the point cloud into our existing
               PC-conditioned diffusion model (already trained and working).

        This avoids the domain gap entirely by converting the image into a
        modality (point cloud) that the diffusion model already handles well.

    Architecture:
        - Stage 1: External image-to-3D model producing point clouds
        - Stage 2: Pre-trained PC-conditioned Diffusion_condition model

    Limitations:
        - Requires a separate image-to-3D model (not trained here)
        - Two-stage errors compound (noisy PC -> noisy BRep)
        - Slower than single-stage inference

    Usage:
        >>> pipeline = TwoStagePipeline(
        ...     pc_diffusion_model=my_pc_model,
        ...     image_to_3d_fn=my_trellis_wrapper,
        ... )
        >>> results = pipeline.inference(batch, device)
    """

    def __init__(
        self,
        pc_diffusion_model: "DiffusionCrossAttn",
        image_to_3d_fn: Optional[Any] = None,
        image_to_3d_model: Optional[nn.Module] = None,
        num_points: int = 4096,
        normalize_pc: bool = True,
    ):
        """Initialize the two-stage pipeline.

        Args:
            pc_diffusion_model: A trained ``Diffusion_condition`` with
                ``with_pc=True`` (point-cloud conditioned).
            image_to_3d_fn: A callable that takes an image tensor [B, 3, H, W]
                and returns a point cloud [B, N, 6] (xyz + normals).
                Mutually exclusive with ``image_to_3d_model``.
            image_to_3d_model: An nn.Module that performs image-to-3D.
                Its forward method should accept images and return point clouds.
            num_points: Number of points to sample from the generated point cloud.
            normalize_pc: Whether to normalize point clouds to unit sphere.
        """
        super().__init__()
        self.pc_diffusion = pc_diffusion_model
        self.image_to_3d_fn = image_to_3d_fn
        self.image_to_3d_model = image_to_3d_model
        self.num_points = num_points
        self.normalize_pc = normalize_pc

        # Freeze the PC diffusion model (inference only)
        for param in self.pc_diffusion.parameters():
            param.requires_grad = False
        self.pc_diffusion.eval()

        if self.image_to_3d_model is not None:
            for param in self.image_to_3d_model.parameters():
                param.requires_grad = False
            self.image_to_3d_model.eval()

    def train(self, mode: bool = True):
        """This module is inference-only; always stays in eval."""
        super().train(False)  # Always eval
        return self

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """Not applicable - this strategy is inference-only.

        Raises:
            RuntimeError: Always, since this strategy has no trainable components.
        """
        raise RuntimeError(
            "TwoStagePipeline is an inference-only strategy. "
            "It has no trainable parameters and does not support training_step(). "
            "Use KnowledgeDistillation or FeatureDomainMapper for training."
        )

    def _image_to_pointcloud(
        self,
        images: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Convert images to point clouds using the configured backend.

        Args:
            images: [B, 3, H, W] or [B, num_views, 3, H, W] input images.
            device: Target device.

        Returns:
            [B, num_points, 6] point cloud (xyz + normals).
        """
        if images.dim() == 5:
            # Multi-view: use first view or concatenate
            images = images[:, 0]  # Use primary view

        images = images.to(device)

        if self.image_to_3d_model is not None:
            with torch.no_grad():
                pc = self.image_to_3d_model(images)
        elif self.image_to_3d_fn is not None:
            pc = self.image_to_3d_fn(images)
        else:
            raise RuntimeError(
                "No image-to-3D backend configured. Provide either "
                "'image_to_3d_fn' or 'image_to_3d_model' at initialization."
            )

        # Ensure correct shape [B, N, 6]
        if pc.dim() == 2:
            pc = pc.unsqueeze(0)
        if pc.shape[-1] == 3:
            # No normals provided; pad with zeros
            pc = torch.cat([pc, torch.zeros_like(pc)], dim=-1)

        # Subsample to target number of points
        if pc.shape[1] > self.num_points:
            indices = torch.randperm(pc.shape[1], device=device)[: self.num_points]
            pc = pc[:, indices]
        elif pc.shape[1] < self.num_points:
            # Upsample by repeating random points
            deficit = self.num_points - pc.shape[1]
            extra_idx = torch.randint(0, pc.shape[1], (deficit,), device=device)
            pc = torch.cat([pc, pc[:, extra_idx]], dim=1)

        # Normalize to unit sphere
        if self.normalize_pc:
            pc = self._normalize_pointcloud(pc)

        return pc

    @staticmethod
    def _normalize_pointcloud(pc: Tensor) -> Tensor:
        """Normalize point cloud to be centered at origin with unit scale.

        Only normalizes the xyz coordinates (first 3 channels); normals
        (channels 3-6) are re-normalized to unit length.

        Args:
            pc: [B, N, 6] point cloud (xyz + normals).

        Returns:
            Normalized point cloud [B, N, 6].
        """
        xyz = pc[..., :3]
        normals = pc[..., 3:6]

        # Center
        centroid = xyz.mean(dim=1, keepdim=True)
        xyz = xyz - centroid

        # Scale to unit sphere
        max_dist = xyz.norm(dim=-1, keepdim=True).max(dim=1, keepdim=True).values
        max_dist = max_dist.clamp(min=1e-8)
        xyz = xyz / max_dist

        # Re-normalize normals (they might not be unit-length)
        normal_norms = normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normals = normals / normal_norms

        return torch.cat([xyz, normals], dim=-1)

    @torch.no_grad()
    def inference(
        self,
        batch: Dict[str, Any],
        device: torch.device,
    ) -> List[Any]:
        """Run the full two-stage pipeline: Image -> PC -> BRep.

        Args:
            batch: Input batch. Expected keys:
                - "conditions": {"imgs": [B, num_views, 3, H, W]}
                  OR
                - "conditions": {"points": [B, 1, N, 6]} (pre-computed PC)
                - "v_prefix": list of identifiers (length determines batch size)
            device: Target device for computation.

        Returns:
            List of reconstructed B-Rep data (one per batch element).
        """
        self.eval()
        bs = len(batch.get("v_prefix", [""] * 1))

        # Check if point cloud is already provided
        conditions = batch.get("conditions", {})
        if "points" in conditions and conditions["points"] is not None:
            # Directly use provided point cloud
            pc_batch = batch
        else:
            # Stage 1: Image -> Point Cloud
            imgs = conditions.get("imgs")
            if imgs is None:
                raise ValueError(
                    "TwoStagePipeline requires either 'conditions.imgs' "
                    "or 'conditions.points' in the batch."
                )
            point_cloud = self._image_to_pointcloud(imgs, device)

            # Package point cloud into the format expected by PC diffusion
            pc_batch = {
                **batch,
                "conditions": {
                    "points": point_cloud.unsqueeze(1),  # [B, 1, N, 6]
                },
            }
            if "id_aug" not in pc_batch:
                pc_batch["id_aug"] = torch.zeros(bs, dtype=torch.long, device=device)

        # Stage 2: Point Cloud -> BRep (using pre-trained PC-conditioned model)
        return self.pc_diffusion.inference(bs, device, v_data=pc_batch, v_log=False)


# =============================================================================
# Utility Functions
# =============================================================================


def _inverse_sigmoid(x: float) -> float:
    """Compute inverse sigmoid (logit) for initializing gate parameters."""
    x = max(min(x, 1.0 - 1e-6), 1e-6)
    return math.log(x / (1.0 - x))


def _reinitialize_module(module: nn.Module) -> None:
    """Re-initialize all parameters in a module using default initialization.

    Linear layers use Kaiming uniform, LayerNorm uses ones/zeros,
    Embedding uses normal, and other parameters use uniform.
    """
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            nn.init.kaiming_uniform_(child.weight, a=math.sqrt(5))
            if child.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(child.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(child.bias, -bound, bound)
        elif isinstance(child, nn.LayerNorm):
            if child.weight is not None:
                nn.init.ones_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.Embedding):
            nn.init.normal_(child.weight, mean=0.0, std=0.02)
        elif isinstance(child, nn.MultiheadAttention):
            # Reset in-projection and out-projection
            if child.in_proj_weight is not None:
                nn.init.xavier_uniform_(child.in_proj_weight)
            if child.in_proj_bias is not None:
                nn.init.zeros_(child.in_proj_bias)
            nn.init.xavier_uniform_(child.out_proj.weight)
            if child.out_proj.bias is not None:
                nn.init.zeros_(child.out_proj.bias)
