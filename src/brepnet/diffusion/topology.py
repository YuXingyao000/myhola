"""可选的 oracle / learned 拓扑注意力偏置；单图 baseline 默认关闭。"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def build_soft_topology_attention_bias(
    adjacency_weight: Tensor,
    timesteps: Tensor,
    *,
    num_train_timesteps: int,
    timestep_weight: str,
    parameterization: str,
    scale: float,
    max_multiplier: float,
) -> Tensor:
    if timestep_weight != "linear_high_noise":
        raise ValueError(f"Unknown topology timestep weight '{timestep_weight}'.")

    if parameterization == "additive":
        # Preserve the historical schedule exactly for old configs/checkpoints.
        timestep_scale = timesteps.float() / float(num_train_timesteps)
        pair_bias = adjacency_weight * scale
    elif parameterization == "log_multiplier":
        if max_multiplier <= 0:
            raise ValueError("max_multiplier must be positive.")
        max_timestep = max(num_train_timesteps - 1, 1)
        timestep_scale = timesteps.float() / float(max_timestep)
        pair_bias = torch.log1p(adjacency_weight * (max_multiplier - 1.0))
    else:
        raise ValueError(f"Unknown topology bias parameterization '{parameterization}'.")

    return pair_bias * timestep_scale[:, None, None]


class OracleTopologySelfAttentionMask(nn.Module):
    def __init__(self, cfg: dict, num_train_timesteps: int):
        super().__init__()
        self.enabled = cfg["enabled"]
        self.mode = cfg["mode"]
        self.timestep_weight = cfg["timestep_weight"]
        self.scale = cfg["scale"]
        self.parameterization = cfg.get("parameterization", "additive")
        self.max_multiplier = cfg.get("max_multiplier", 2.0)
        self.num_train_timesteps = num_train_timesteps

    def forward(self, face_adjacency: Tensor | None, timesteps: Tensor) -> Tensor | None:
        if not self.enabled:
            return None
        if face_adjacency is None:
            return None

        adjacency_weight = face_adjacency.to(timesteps.device, dtype=torch.float32).clamp(0.0, 1.0)
        adjacency_bool = adjacency_weight > 0.5
        seq_len = adjacency_weight.shape[1]

        if self.mode == "hard_mask":
            self_loop = torch.eye(seq_len, device=timesteps.device, dtype=torch.bool).unsqueeze(0)
            allowed_attention = adjacency_bool | self_loop
            return ~allowed_attention

        if self.mode == "soft_bias":
            return build_soft_topology_attention_bias(
                adjacency_weight,
                timesteps,
                num_train_timesteps=self.num_train_timesteps,
                timestep_weight=self.timestep_weight,
                parameterization=self.parameterization,
                scale=self.scale,
                max_multiplier=self.max_multiplier,
            )

        raise ValueError(f"Unknown oracle topology self-attention mode '{self.mode}'.")


class LearnedTopologyPredictor(nn.Module):
    """Single-image topology predictor matching the toy model in
    experiments/2026-06-01/topology_from_svr_single.py.

    Input:  images [B, 3, H, W] (single view, identity rotation)
    Output: face_adj_prob [B, max_faces, max_faces] in [0, 1]

    DINOv2 is frozen. The enclosing mask module may load the heads from an
    explicit checkpoint; without one they remain randomly initialized.
    Unfreezing permits gradients but does not supply a training objective.
    """

    def __init__(self, max_faces: int, hidden_dim: int, face_dim: int,
                 dino_model: str = "facebook/dinov2-large"):
        super().__init__()
        from transformers import Dinov2Model

        self.max_faces = max_faces
        self.face_dim = face_dim

        self.dino = Dinov2Model.from_pretrained(dino_model)
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino_dim = self.dino.config.hidden_size

        self.face_head = nn.Sequential(
            nn.Linear(self.dino_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_faces * face_dim),
        )
        self.valid_head = nn.Linear(face_dim, 1)
        self.edge_head = nn.Sequential(
            nn.Linear(face_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Return (face_adj_prob [B, max_faces, max_faces], valid_logits [B, max_faces])."""
        self.dino.eval()
        batch_size = images.shape[0]
        with torch.no_grad():
            image_feature = self.dino(images).last_hidden_state[:, 0]

        face_tokens = self.face_head(image_feature).reshape(batch_size, self.max_faces, self.face_dim)
        valid_logits = self.valid_head(face_tokens).squeeze(-1)

        a = face_tokens.unsqueeze(2).expand(-1, -1, self.max_faces, -1)
        b = face_tokens.unsqueeze(1).expand(-1, self.max_faces, -1, -1)
        pair = torch.cat([a, b, torch.abs(a - b), a * b], dim=-1)
        edge_logits = self.edge_head(pair).squeeze(-1)
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))

        diag = torch.eye(self.max_faces, device=edge_logits.device, dtype=torch.bool)
        edge_logits = edge_logits.masked_fill(diag[None], -20.0)

        face_adj_prob = torch.sigmoid(edge_logits)
        return face_adj_prob, valid_logits


class LearnedTopologySelfAttentionMask(nn.Module):
    """Drop-in replacement for OracleTopologySelfAttentionMask.

    Instead of reading GT face_adj from batch, predicts it from input image
    using a frozen/loaded TopologyPredictor. The downstream behavior (soft_bias
    formulation, timestep weighting) matches OracleTopologySelfAttentionMask.

    Config keys (under model.topology_bias):
        enabled: bool
        mode: "soft_bias" (recommended) or "hard_mask"
        scale: float
        num_train_timesteps: int (auto-injected)
        predictor:
            checkpoint: path to toy model best.pt
            max_faces: int (must match denoiser max_faces)
            hidden_dim: int (toy model hidden_dim, default 512)
            face_dim: int (toy model face_dim, default 128)
            freeze: bool (default true; false permits downstream gradients)
            edge_threshold: float (only for hard_mask mode; default 0.5)
    """

    def __init__(self, cfg: dict, num_train_timesteps: int):
        super().__init__()
        self.enabled = cfg["enabled"]
        self.mode = cfg["mode"]
        self.scale = cfg["scale"]
        self.timestep_weight = cfg["timestep_weight"]
        self.parameterization = cfg.get("parameterization", "additive")
        self.max_multiplier = cfg.get("max_multiplier", 2.0)
        self.num_train_timesteps = num_train_timesteps

        predictor_cfg = cfg["predictor"]
        self.predictor = LearnedTopologyPredictor(
            max_faces=predictor_cfg["max_faces"],
            hidden_dim=predictor_cfg.get("hidden_dim", 512),
            face_dim=predictor_cfg.get("face_dim", 128),
            dino_model=predictor_cfg.get("dino_model", "facebook/dinov2-large"),
        )
        ckpt_path = predictor_cfg.get("checkpoint", None)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            self.predictor.load_state_dict(state, strict=False)

        self.freeze = predictor_cfg.get("freeze", True)
        if self.freeze:
            for p in self.predictor.parameters():
                p.requires_grad = False
            self.predictor.eval()

        self.edge_threshold = predictor_cfg.get("edge_threshold", 0.5)
        self.last_predicted_adj: Tensor | None = None  # cache for aux loss
        self.last_valid_logits: Tensor | None = None

        # 解除冻结不自动增加辅助 loss；hard_mask 阈值也不能向 predictor 回传梯度。

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.predictor.eval()
        return self

    def forward(self, images: Tensor | None, timesteps: Tensor) -> Tensor | None:
        if not self.enabled:
            return None
        if images is None:
            return None

        if self.freeze:
            with torch.no_grad():
                face_adj_prob, valid_logits = self.predictor(images)
        else:
            face_adj_prob, valid_logits = self.predictor(images)

        # Cache for optional auxiliary loss (face_adj BCE against GT)
        self.last_predicted_adj = face_adj_prob
        self.last_valid_logits = valid_logits

        seq_len = face_adj_prob.shape[1]

        if self.mode == "hard_mask":
            adjacency = face_adj_prob > self.edge_threshold
            self_loop = torch.eye(seq_len, device=timesteps.device, dtype=torch.bool).unsqueeze(0)
            allowed_attention = adjacency | self_loop
            return ~allowed_attention

        if self.mode == "soft_bias":
            return build_soft_topology_attention_bias(
                face_adj_prob,
                timesteps,
                num_train_timesteps=self.num_train_timesteps,
                timestep_weight=self.timestep_weight,
                parameterization=self.parameterization,
                scale=self.scale,
                max_multiplier=self.max_multiplier,
            )

        raise ValueError(f"Unknown learned topology self-attention mode '{self.mode}'.")


def build_topology_self_attention(cfg: dict, num_train_timesteps: int) -> nn.Module:
    """Factory: dispatch between oracle and learned topology mask."""
    source = cfg.get("source", "gt_adjacency")
    if source == "gt_adjacency":
        return OracleTopologySelfAttentionMask(cfg, num_train_timesteps)
    if source == "learned":
        return LearnedTopologySelfAttentionMask(cfg, num_train_timesteps)
    raise ValueError(f"Unknown topology_bias source '{source}'.")
