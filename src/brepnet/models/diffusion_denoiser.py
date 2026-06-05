from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.brepnet.models.blocks import sincos_embedding


def clip_style_symmetric_infonce(cad_embedding: Tensor, condition_embedding: Tensor, temperature: float) -> Tensor:
    logits = cad_embedding @ condition_embedding.T / temperature
    labels = torch.arange(cad_embedding.shape[0], device=cad_embedding.device)
    cad_to_condition = F.cross_entropy(logits, labels)
    condition_to_cad = F.cross_entropy(logits.T, labels)
    return 0.5 * (cad_to_condition + condition_to_cad)


class NoConditionFuser(nn.Module):
    def forward(
        self,
        latent_hidden: Tensor,
        condition: Tensor | None,
        alignment_hidden: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        return latent_hidden, None


class CrossAttentionConditionFuser(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        condition_dim: int,
        cross_attention_dim: int,
        num_layers: int,
        alignment_cfg: dict,
    ):
        super().__init__()
        self.query_projection = nn.Linear(hidden_dim, cross_attention_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cross_attention_dim,
            nhead=cross_attention_dim // 64,
            norm_first=True,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers,
            nn.LayerNorm(cross_attention_dim),
        )
        self.condition_projection = nn.Identity() if condition_dim == cross_attention_dim else nn.Linear(condition_dim, cross_attention_dim)
        self.output_projection = nn.Linear(cross_attention_dim, hidden_dim)
        self.alignment_enabled = alignment_cfg["enabled"]
        self.alignment_weight = alignment_cfg["weight"]
        self.alignment_temperature = alignment_cfg["temperature"]
        self.alignment_projection_dim = alignment_cfg["projection_dim"]
        self.cad_alignment_projection = nn.Linear(cross_attention_dim, self.alignment_projection_dim)
        self.condition_alignment_projection = nn.Linear(cross_attention_dim, self.alignment_projection_dim)
        self.cad_alignment_head = nn.Linear(self.alignment_projection_dim, self.alignment_projection_dim)
        self.condition_alignment_head = nn.Linear(self.alignment_projection_dim, self.alignment_projection_dim)

    def prepare_condition(self, condition: Tensor) -> Tensor:
        if condition.dim() == 4:
            condition = condition[:, 0]
        elif condition.dim() == 2:
            condition = condition[:, None]
        return self.condition_projection(condition)

    def forward(
        self,
        latent_hidden: Tensor,
        condition: Tensor,
        alignment_hidden: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        condition_memory = self.prepare_condition(condition)
        cad_tgt = self.query_projection(latent_hidden)

        align_loss = None
        if self.alignment_enabled:
            assert alignment_hidden is not None
            cad_align_src = self.query_projection(alignment_hidden)
            cad_align_tokens = self.cad_alignment_projection(cad_align_src)
            condition_align_tokens = self.condition_alignment_projection(condition_memory)

            cad_global = cad_align_tokens.mean(dim=1)
            condition_global = condition_align_tokens.mean(dim=1)
            cad_embedding = F.normalize(self.cad_alignment_head(cad_global), dim=-1)
            condition_embedding = F.normalize(self.condition_alignment_head(condition_global), dim=-1)
            align_loss = clip_style_symmetric_infonce(
                cad_embedding,
                condition_embedding,
                self.alignment_temperature,
            ) * self.alignment_weight

        latent_hidden = self.decoder(tgt=cad_tgt, memory=condition_memory)
        latent_hidden = self.output_projection(latent_hidden)
        return latent_hidden, align_loss


def build_condition_fuser(fuser_cfg: dict, condition_cfg: dict, hidden_dim: int) -> nn.Module:
    if condition_cfg["type"] == "none":
        return NoConditionFuser()
    if fuser_cfg["type"] == "cross_attention":
        return CrossAttentionConditionFuser(
            hidden_dim=hidden_dim,
            condition_dim=fuser_cfg["condition_dim"],
            cross_attention_dim=fuser_cfg["hidden_dim"],
            num_layers=fuser_cfg["num_layers"],
            alignment_cfg=fuser_cfg["alignment"],
        )
    raise ValueError(f"Unknown condition fuser type '{fuser_cfg['type']}'.")


class OracleTopologySelfAttentionMask(nn.Module):
    def __init__(self, cfg: dict, num_train_timesteps: int):
        super().__init__()
        self.enabled = cfg["enabled"]
        self.mode = cfg["mode"]
        self.timestep_weight = cfg["timestep_weight"]
        self.scale = cfg["scale"]
        self.num_train_timesteps = num_train_timesteps

    def forward(self, face_adjacency: Tensor | None, timesteps: Tensor) -> Tensor | None:
        if not self.enabled:
            return None
        if face_adjacency is None:
            return None

        adjacency = face_adjacency.to(timesteps.device).bool()
        seq_len = adjacency.shape[1]

        if self.mode == "hard_mask":
            self_loop = torch.eye(seq_len, device=timesteps.device, dtype=torch.bool).unsqueeze(0)
            allowed_attention = adjacency | self_loop
            return ~allowed_attention

        if self.mode == "soft_bias":
            adjacency_bias = adjacency.to(dtype=torch.float32)
            timestep_weight = timesteps.float() / float(self.num_train_timesteps)
            return adjacency_bias * timestep_weight[:, None, None] * self.scale

        raise ValueError(f"Unknown oracle topology self-attention mode '{self.mode}'.")


class LearnedTopologyPredictor(nn.Module):
    """Single-image topology predictor matching the toy model in
    experiments/2026-06-01/topology_from_svr_single.py.

    Input:  images [B, 3, H, W] (single view, identity rotation)
    Output: face_adj_prob [B, max_faces, max_faces] in [0, 1]

    DINOv2 is frozen. face_head / valid_head / edge_head are loaded from a
    pretrained checkpoint. Whether they are trainable is controlled by
    `freeze` in config.
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
            freeze: bool (default true; if false, predictor params are trained)
            edge_threshold: float (only for hard_mask mode; default 0.5)
    """

    def __init__(self, cfg: dict, num_train_timesteps: int):
        super().__init__()
        self.enabled = cfg["enabled"]
        self.mode = cfg["mode"]
        self.scale = cfg["scale"]
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
            timestep_weight = timesteps.float() / float(self.num_train_timesteps)
            return face_adj_prob * timestep_weight[:, None, None] * self.scale

        raise ValueError(f"Unknown learned topology self-attention mode '{self.mode}'.")


def build_topology_self_attention(cfg: dict, num_train_timesteps: int) -> nn.Module:
    """Factory: dispatch between oracle and learned topology mask."""
    source = cfg.get("source", "gt_adjacency")
    if source == "gt_adjacency":
        return OracleTopologySelfAttentionMask(cfg, num_train_timesteps)
    if source == "learned":
        return LearnedTopologySelfAttentionMask(cfg, num_train_timesteps)
    raise ValueError(f"Unknown topology_bias source '{source}'.")


class BRepDenoiser(nn.Module):
    def __init__(self, latent_dim: int, cfg: dict, condition_fuser: nn.Module):
        super().__init__()
        self.hidden_dim = cfg["hidden_dim"]
        self.condition_fuser = condition_fuser
        self.input_projection = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.hidden_dim // cfg["nhead_divisor"],
            norm_first=True,
            dim_feedforward=cfg["feedforward_dim"],
            dropout=cfg["dropout"],
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, cfg["num_layers"], nn.LayerNorm(self.hidden_dim))
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, latent_dim),
        )

    def forward(
        self,
        noisy_latent_sequence: Tensor,
        timesteps: Tensor,
        condition: Tensor | None,
        clean_latent_sequence: Tensor | None = None,
        self_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        time_embedding = self.time_embedding(sincos_embedding(timesteps, self.hidden_dim)).unsqueeze(1)
        hidden = self.input_projection(noisy_latent_sequence)
        alignment_hidden = self.input_projection(clean_latent_sequence) if clean_latent_sequence is not None else None
        hidden, align_loss = self.condition_fuser(
            hidden,
            condition,
            alignment_hidden,
        )
        hidden = hidden + time_embedding
        hidden = self.run_backbone(hidden, self_attention_mask)
        return self.output_projection(hidden), align_loss

    def run_backbone(self, hidden: Tensor, self_attention_mask: Tensor | None) -> Tensor:
        if self_attention_mask is None:
            return self.backbone(hidden)

        batch_size, seq_len, _ = self_attention_mask.shape
        nhead = self.backbone.layers[0].self_attn.num_heads
        src_mask = self_attention_mask.unsqueeze(1).expand(-1, nhead, -1, -1)
        src_mask = src_mask.reshape(batch_size * nhead, seq_len, seq_len)

        output = hidden
        for layer in self.backbone.layers:
            output = layer(output, src_mask=src_mask)
        if self.backbone.norm is not None:
            output = self.backbone.norm(output)
        return output
