"""
Topology-Guided Diffusion for HoLa-BRep.

This module implements topology-aware modifications to the diffusion pipeline,
inspired by DTGBrepGen's topology-first approach but adapted to work within
HoLa-BRep's holistic latent space.

Key insight: HoLa-BRep currently generates face latents → then predicts topology.
This is backwards. If we can predict/provide topology FIRST, the diffusion model
has structural guidance for generating geometrically consistent faces.

Three approaches (increasing complexity):

1. TopologyPredictor: Predict adjacency from image, inject as attention bias
2. DisentangledLatentVAE: Split latent into [topo | geom], generate topo first
3. TwoPhaseTopologyDiffusion: Phase 1 generates coarse topo-aware latents,
   Phase 2 refines geometry
"""

from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from src.brepnet.models.blocks import sincos_embedding


# ═══════════════════════════════════════════════════════════════════
# Approach 1: Topology Predictor + Attention Bias Injection
# ═══════════════════════════════════════════════════════════════════
# Minimal change. Predict adjacency from condition, use it to bias
# self-attention in the denoising backbone.
# ═══════════════════════════════════════════════════════════════════


class TopologyPredictor(nn.Module):
    """
    Predicts face adjacency matrix from image condition features.

    Given image features, predicts which faces should be adjacent.
    This predicted topology is then used to bias the diffusion model's
    self-attention: faces that should be adjacent attend more to each other.

    Architecture:
        image_features [B, 257, 1024]
            → pool to [B, 1024]
            → MLP → [B, num_faces * num_faces]
            → reshape → [B, num_faces, num_faces]
            → sigmoid → adjacency probabilities

    Usage in diffusion:
        adj_pred = topology_predictor(image_features)  # [B, 30, 30]
        # Convert to attention bias: log(adj + eps) as additive bias
        attn_bias = torch.log(adj_pred + 1e-6)
        # Add to self-attention logits in denoising backbone
    """

    def __init__(self, feature_dim: int = 1024, num_faces: int = 30, hidden_dim: int = 512):
        super().__init__()
        self.num_faces = num_faces

        self.pool = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_faces * num_faces),
        )

    def forward(self, image_features: Tensor) -> Tensor:
        """
        Args:
            image_features: [B, seq_len, feature_dim] (e.g. [B, 257, 1024])

        Returns:
            adj_prob: [B, num_faces, num_faces] symmetric adjacency probabilities
        """
        # Global pool
        pooled = image_features.mean(dim=1)  # [B, 1024]
        hidden = self.pool(pooled)  # [B, 512]

        # Predict adjacency
        adj_flat = self.predictor(hidden)  # [B, 900]
        adj = adj_flat.reshape(-1, self.num_faces, self.num_faces)

        # Make symmetric
        adj = (adj + adj.transpose(-1, -2)) / 2

        return torch.sigmoid(adj)


class TopologyAttentionBias(nn.Module):
    """
    Converts predicted topology into attention bias for the denoising backbone.

    The bias is added to self-attention logits: positions that should be
    topologically connected get positive bias (attend more), unconnected
    positions get negative bias (attend less).

    This is inspired by DTGBrepGen's approach where "shared-edge counts
    influence face-to-face attention."
    """

    def __init__(self, num_heads: int = 12, num_faces: int = 30):
        super().__init__()
        self.num_heads = num_heads
        # Learnable scale per head (how much to trust topology prediction)
        self.scale = nn.Parameter(torch.zeros(num_heads))

    def forward(self, adj_prob: Tensor) -> Tensor:
        """
        Args:
            adj_prob: [B, num_faces, num_faces] from TopologyPredictor

        Returns:
            attn_bias: [B, num_heads, num_faces, num_faces]
                       to be added to attention logits
        """
        # Convert probability to log-space bias
        # Connected faces → positive bias, disconnected → negative
        bias = torch.log(adj_prob + 1e-6) - torch.log(1 - adj_prob + 1e-6)  # logit

        # Per-head scaling (starts at 0 = no effect, learns to use topology)
        scale = torch.sigmoid(self.scale)  # [num_heads]
        attn_bias = bias.unsqueeze(1) * scale[None, :, None, None]  # [B, H, N, N]

        return attn_bias


# ═══════════════════════════════════════════════════════════════════
# Approach 2: Disentangled Latent Space
# ═══════════════════════════════════════════════════════════════════
# Split face latent [32] into [16 topo | 16 geom].
# Encourage disentanglement via auxiliary losses.
# Diffusion generates topo first, then geom conditioned on topo.
# ═══════════════════════════════════════════════════════════════════


class DisentangledLatentProjection(nn.Module):
    """
    Projects fused face features into disentangled topo/geom latents.

    Replaces the simple gaussian_proj in the original VAE.
    Instead of: face_features → [mean, logvar] of shape [N, 32]
    Does:       face_features → [topo_mean, topo_logvar, geom_mean, geom_logvar]
                                 [N, 16]    [N, 16]      [N, 16]    [N, 16]

    Auxiliary losses enforce disentanglement:
    - topo_z alone should predict adjacency (topo_classifier_loss)
    - geom_z alone should reconstruct face points (geom_recon_loss)
    """

    def __init__(self, input_dim: int = 32, topo_dim: int = 16, geom_dim: int = 16):
        super().__init__()
        self.topo_dim = topo_dim
        self.geom_dim = geom_dim
        total_dim = topo_dim + geom_dim

        # Separate projection heads
        self.topo_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, topo_dim * 2),  # mean + logvar
        )
        self.geom_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, geom_dim * 2),  # mean + logvar
        )

        # Auxiliary: topo_z should predict adjacency
        self.topo_classifier = nn.Sequential(
            nn.Linear(topo_dim * 2, topo_dim),
            nn.ReLU(),
            nn.Linear(topo_dim, 1),
        )

    def forward(self, face_features: Tensor, v_is_test: bool = False):
        """
        Args:
            face_features: [N, input_dim] fused face features from encoder

        Returns:
            z: [N, topo_dim + geom_dim] combined latent
            kl_loss: KL divergence
            aux_data: dict with topo_z, geom_z for auxiliary losses
        """
        # Topology latent
        topo_params = self.topo_proj(face_features)  # [N, topo_dim*2]
        topo_mean = topo_params[:, :self.topo_dim]
        topo_logvar = topo_params[:, self.topo_dim:]

        # Geometry latent
        geom_params = self.geom_proj(face_features)  # [N, geom_dim*2]
        geom_mean = geom_params[:, :self.geom_dim]
        geom_logvar = geom_params[:, self.geom_dim:]

        # Reparameterize
        if v_is_test:
            topo_z = topo_mean
            geom_z = geom_mean
        else:
            topo_z = topo_mean + torch.exp(0.5 * topo_logvar) * torch.randn_like(topo_mean)
            geom_z = geom_mean + torch.exp(0.5 * geom_logvar) * torch.randn_like(geom_mean)

        # Combined
        z = torch.cat([topo_z, geom_z], dim=-1)  # [N, 32]

        # KL loss (both parts)
        kl = -0.5 * torch.sum(
            1 + topo_logvar - topo_mean.pow(2) - topo_logvar.exp()
        ) + -0.5 * torch.sum(
            1 + geom_logvar - geom_mean.pow(2) - geom_logvar.exp()
        )

        return z, kl, {
            "topo_z": topo_z,
            "geom_z": geom_z,
            "topo_mean": topo_mean,
            "geom_mean": geom_mean,
        }

    def topology_auxiliary_loss(self, topo_z: Tensor, edge_face_connectivity: Tensor,
                                num_faces: int) -> Tensor:
        """
        Auxiliary loss: topo_z alone should predict face adjacency.

        Args:
            topo_z: [N, topo_dim] topology latents for all faces
            edge_face_connectivity: [E, 3] (edge_id, face_id_1, face_id_2)
            num_faces: total number of faces
        """
        # Create positive pairs (connected faces)
        pos_pairs = topo_z[edge_face_connectivity[:, 1:]]  # [E, 2, topo_dim]
        pos_features = pos_pairs.reshape(-1, self.topo_dim * 2)
        pos_pred = self.topo_classifier(pos_features)

        # Create negative pairs (random non-connected faces)
        num_neg = pos_pairs.shape[0]
        neg_idx = torch.randint(0, num_faces, (num_neg, 2), device=topo_z.device)
        neg_pairs = topo_z[neg_idx]  # [num_neg, 2, topo_dim]
        neg_features = neg_pairs.reshape(-1, self.topo_dim * 2)
        neg_pred = self.topo_classifier(neg_features)

        # BCE loss
        labels = torch.cat([
            torch.ones_like(pos_pred),
            torch.zeros_like(neg_pred),
        ])
        preds = torch.cat([pos_pred, neg_pred])
        return F.binary_cross_entropy_with_logits(preds, labels)


# ═══════════════════════════════════════════════════════════════════
# Approach 3: Two-Phase Diffusion (Topology → Geometry)
# ═══════════════════════════════════════════════════════════════════
# Phase 1: Diffusion generates topo-part of latent (first 16 dims)
# Phase 2: Diffusion generates geom-part (last 16 dims) conditioned on topo
# Both phases use same backbone, just different masks/conditioning.
# ═══════════════════════════════════════════════════════════════════


class TwoPhaseDiffusion(nn.Module):
    """
    Two-phase diffusion for topology-then-geometry generation.

    Instead of generating all 32 dims of face latent simultaneously,
    this generates in two phases:

    Phase 1 (Topology): Generate z[:, :, :16] from noise
        - Conditioned on image features
        - These dims encode structural/topological information

    Phase 2 (Geometry): Generate z[:, :, 16:] from noise
        - Conditioned on image features + Phase 1 result
        - These dims encode precise geometry

    Training: Both phases are trained jointly (shared backbone).
    Inference: Phase 1 runs full denoising → freeze → Phase 2 runs full denoising.

    This is analogous to DTGBrepGen's "topology first, geometry conditioned on
    topology" but within a single latent space rather than separate models.
    """

    def __init__(self, cfg):
        super().__init__()
        self.topo_dim = cfg.get("topo_dim", 16)
        self.geom_dim = cfg.get("geom_dim", 16)
        self.total_dim = self.topo_dim + self.geom_dim
        self.dim_latent = cfg.get("dim_latent", 768)
        self.num_max_faces = cfg.get("num_max_faces", 30)

        # Shared denoising backbone (both phases use same network)
        # Phase indicator embedding tells the network which phase it's in
        self.phase_embed = nn.Embedding(2, self.dim_latent)

        # Input projections (different for each phase)
        self.topo_embed = nn.Sequential(
            nn.Linear(self.topo_dim, self.dim_latent),
            nn.LayerNorm(self.dim_latent),
            nn.SiLU(),
            nn.Linear(self.dim_latent, self.dim_latent),
        )
        self.geom_embed = nn.Sequential(
            nn.Linear(self.geom_dim, self.dim_latent),
            nn.LayerNorm(self.dim_latent),
            nn.SiLU(),
            nn.Linear(self.dim_latent, self.dim_latent),
        )

        # Frozen topo encoder (for Phase 2 conditioning)
        self.topo_condition_proj = nn.Sequential(
            nn.Linear(self.topo_dim, self.dim_latent),
            nn.LayerNorm(self.dim_latent),
            nn.SiLU(),
        )

        # Output projections
        self.topo_out = nn.Sequential(
            nn.Linear(self.dim_latent, self.dim_latent),
            nn.LayerNorm(self.dim_latent),
            nn.SiLU(),
            nn.Linear(self.dim_latent, self.topo_dim),
        )
        self.geom_out = nn.Sequential(
            nn.Linear(self.dim_latent, self.dim_latent),
            nn.LayerNorm(self.dim_latent),
            nn.SiLU(),
            nn.Linear(self.dim_latent, self.geom_dim),
        )

        # Shared denoising backbone (can be the same 24-layer TransformerEncoder)
        layer = nn.TransformerEncoderLayer(
            d_model=self.dim_latent,
            nhead=self.dim_latent // 64,
            norm_first=True,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, cfg.get("num_layers", 24))

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.dim_latent, self.dim_latent),
            nn.LayerNorm(self.dim_latent),
            nn.SiLU(),
            nn.Linear(self.dim_latent, self.dim_latent),
        )

    def forward_phase1(self, noisy_topo: Tensor, timesteps: Tensor,
                       condition: Optional[Tensor] = None) -> Tensor:
        """
        Phase 1: Denoise topology latent.

        Args:
            noisy_topo: [B, N, topo_dim] noisy topology latent
            timesteps: [B] diffusion timesteps
            condition: [B, S, D] image condition features

        Returns:
            pred_topo: [B, N, topo_dim] predicted clean topology
        """
        x = self.topo_embed(noisy_topo)  # [B, N, dim_latent]
        time_emb = self.time_embed(sincos_embedding(timesteps, self.dim_latent)).unsqueeze(1)
        phase_emb = self.phase_embed(torch.zeros(1, device=x.device, dtype=torch.long))

        x = x + time_emb + phase_emb
        x = self.backbone(x)
        return self.topo_out(x)

    def forward_phase2(self, noisy_geom: Tensor, timesteps: Tensor,
                       topo_z: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        """
        Phase 2: Denoise geometry latent, conditioned on topology.

        Args:
            noisy_geom: [B, N, geom_dim] noisy geometry latent
            timesteps: [B] diffusion timesteps
            topo_z: [B, N, topo_dim] clean topology latent (from Phase 1)
            condition: [B, S, D] image condition features

        Returns:
            pred_geom: [B, N, geom_dim] predicted clean geometry
        """
        x = self.geom_embed(noisy_geom)  # [B, N, dim_latent]
        topo_cond = self.topo_condition_proj(topo_z)  # [B, N, dim_latent]
        time_emb = self.time_embed(sincos_embedding(timesteps, self.dim_latent)).unsqueeze(1)
        phase_emb = self.phase_embed(torch.ones(1, device=x.device, dtype=torch.long))

        # Add topology as per-token condition
        x = x + topo_cond + time_emb + phase_emb
        x = self.backbone(x)
        return self.geom_out(x)

    def training_step(self, face_z: Tensor, condition: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Joint training of both phases.

        Randomly picks Phase 1 or Phase 2 per batch (or trains both each step).
        """
        device = face_z.device
        bs = face_z.size(0)

        topo_z = face_z[:, :, :self.topo_dim]  # [B, N, 16]
        geom_z = face_z[:, :, self.topo_dim:]  # [B, N, 16]

        loss = {}

        # Phase 1: topology denoising
        t1 = torch.randint(0, 1000, (bs,), device=device).long()
        noise1 = torch.randn_like(topo_z)
        # (would use noise_scheduler.add_noise in full implementation)
        noisy_topo = topo_z + 0.1 * noise1  # simplified
        pred_topo = self.forward_phase1(noisy_topo, t1, condition)
        loss["topo_loss"] = F.mse_loss(pred_topo, topo_z)

        # Phase 2: geometry denoising (conditioned on clean topo)
        t2 = torch.randint(0, 1000, (bs,), device=device).long()
        noise2 = torch.randn_like(geom_z)
        noisy_geom = geom_z + 0.1 * noise2  # simplified
        pred_geom = self.forward_phase2(noisy_geom, t2, topo_z.detach(), condition)
        loss["geom_loss"] = F.mse_loss(pred_geom, geom_z)

        loss["total_loss"] = loss["topo_loss"] + loss["geom_loss"]
        return loss


# ═══════════════════════════════════════════════════════════════════
# Approach 4: TokenVAE with Dynamic Face Count (no padding/dedup)
# ═══════════════════════════════════════════════════════════════════


class FaceCountPredictor(nn.Module):
    """
    Predicts the number of faces from image condition features.

    Used at inference time to determine how many face tokens to generate,
    eliminating the need for fixed padding and post-hoc deduplication.

    Architecture: Simple MLP classifier (num_classes = max_faces)
    """

    def __init__(self, feature_dim: int = 1024, max_faces: int = 30):
        super().__init__()
        self.max_faces = max_faces
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, max_faces),  # predict class 1..30
        )

    def forward(self, image_features: Tensor) -> Tensor:
        """
        Args:
            image_features: [B, seq_len, dim] or [B, dim]

        Returns:
            logits: [B, max_faces] — class probabilities for face count
        """
        if image_features.dim() == 3:
            x = image_features.mean(dim=1)  # pool
        else:
            x = image_features
        return self.net(x)

    def predict_count(self, image_features: Tensor) -> Tensor:
        """Returns predicted face count [B] (integer)."""
        logits = self.forward(image_features)
        return logits.argmax(dim=-1) + 1  # 1-indexed

    def loss(self, image_features: Tensor, gt_num_faces: Tensor) -> Tensor:
        """
        Args:
            gt_num_faces: [B] ground truth face counts (1-indexed)
        """
        logits = self.forward(image_features)
        targets = gt_num_faces - 1  # to 0-indexed for CE
        return F.cross_entropy(logits, targets.long())
