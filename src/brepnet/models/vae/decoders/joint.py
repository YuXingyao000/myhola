"""Shared-boundary decoder and alternating face-boundary interaction."""

from __future__ import annotations

import math

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence


class BoundaryInteraction(nn.Module):
    """One face -> boundary -> face update, followed by updated ownership."""

    def __init__(self, dim: int, heads: int, dropout: float, edge_to_face: bool):
        super().__init__()
        self.edge_to_face = edge_to_face
        self.face_norm = nn.LayerNorm(dim)
        self.edge_norm = nn.LayerNorm(dim)
        self.face_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.edge_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.side_query = nn.Linear(dim, 2 * dim)
        self.face_key = nn.Linear(dim, dim)
        self.edge_update = nn.Sequential(
            nn.Linear(3 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )
        # Remove these parameters entirely for the unidirectional ablation.
        if edge_to_face:
            self.side_message = nn.Linear(dim, 2 * dim)
            self.face_update = nn.Sequential(
                nn.Linear(2 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
            )

    def ownership(self, faces: Tensor, edges: Tensor, valid: Tensor) -> Tensor:
        batch, queries, dim = edges.shape
        query = self.side_query(edges).reshape(batch, queries, 2, dim)
        logits = torch.einsum("bmsd,bnd->bmsn", query, self.face_key(faces)) / math.sqrt(dim)
        return logits.masked_fill(~valid[:, None, None, :], -torch.inf)

    def forward(
        self, faces: Tensor, edges: Tensor, valid: Tensor, existence_head: nn.Module
    ) -> tuple[Tensor, Tensor]:
        normalized = self.face_norm(faces)
        faces = faces + self.face_attention(
            normalized, normalized, normalized, key_padding_mask=~valid, need_weights=False
        )[0]
        faces = faces.masked_fill(~valid[..., None], 0)
        normalized = self.edge_norm(edges)
        edges = edges + self.edge_attention(normalized, normalized, normalized, need_weights=False)[0]

        assignment = self.ownership(faces, edges, valid).softmax(dim=-1)
        read = torch.einsum("bmsn,bnd->bmsd", assignment, faces).flatten(-2)
        edges = edges + self.edge_update(torch.cat((edges, read), dim=-1))

        if self.edge_to_face:
            # Recompute after the edge update. The SAME ownership weights both
            # express incidence and route edge information to each face.
            assignment = self.ownership(faces, edges, valid).softmax(dim=-1)
            mass = assignment * existence_head(edges).sigmoid()[..., None]
            messages = self.side_message(edges).reshape(*edges.shape[:2], 2, edges.shape[-1])
            write = torch.einsum("bmsn,bmsd->bnd", mass, messages)
            write = write / mass.sum(dim=(1, 2))[..., None].clamp_min(1.0)
            faces = faces + self.face_update(torch.cat((faces, write), dim=-1))
            faces = faces.masked_fill(~valid[..., None], 0)
        return faces, edges


class JointBoundaryDecoder(nn.Module):
    """Decode a variable face set and at most ``num_queries`` boundaries.

    Only batching is padded. There are no duplicate face slots or face positional
    embeddings. Global XYZ heads deliberately differ from the legacy local-grid
    plus bounding-box heads; this must be accounted for in research comparisons.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        num_queries: int = 128,
        num_rounds: int = 3,
        num_heads: int = 8,
        dropout: float = 0.0,
        edge_to_face: bool = True,
        face_resolution: int = 16,
        curve_resolution: int = 16,
    ):
        super().__init__()
        if min(latent_dim, hidden_dim, num_queries, num_rounds, num_heads) < 1:
            raise ValueError("Decoder dimensions, query capacity, and round count must be positive")
        if hidden_dim % num_heads:
            raise ValueError("joint_hidden_dim must be divisible by joint_num_heads")
        self.num_queries = num_queries
        self.face_resolution = face_resolution
        self.curve_resolution = curve_resolution
        self.face_input = nn.Linear(latent_dim, hidden_dim)
        self.edge_queries = nn.Parameter(torch.randn(num_queries, hidden_dim) / math.sqrt(hidden_dim))
        self.rounds = nn.ModuleList([
            BoundaryInteraction(hidden_dim, num_heads, dropout, edge_to_face)
            for _ in range(num_rounds)
        ])
        self.existence_head = nn.Linear(hidden_dim, 1)
        self.face_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, face_resolution * face_resolution * 3),
        )
        self.curve_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, curve_resolution * 3),
        )

    def forward(self, face_latents: Tensor, face_counts: Tensor) -> dict[str, Tensor]:
        if face_latents.ndim != 2:
            raise ValueError("face_latents must be a flat [sum_faces, latent_dim] tensor")
        if face_counts.ndim != 1 or face_counts.dtype not in (torch.int32, torch.int64):
            raise ValueError("face_counts must be a one-dimensional integer tensor")
        counts = face_counts.detach().cpu().tolist()
        if not counts or min(counts) < 1 or sum(counts) != len(face_latents):
            raise ValueError("face_counts must partition the input into nonempty objects")
        if not torch.isfinite(face_latents).all():
            raise ValueError("face_latents contains nonfinite values")
        padded = pad_sequence(face_latents.split(counts), batch_first=True)
        counts_tensor = face_counts.to(face_latents.device)
        valid = torch.arange(padded.shape[1], device=padded.device)[None, :] < counts_tensor[:, None]
        faces = self.face_input(padded).masked_fill(~valid[..., None], 0)
        edges = self.edge_queries[None].expand(len(counts), -1, -1)
        for interaction in self.rounds:
            faces, edges = interaction(faces, edges, valid, self.existence_head)
        ownership_logits = self.rounds[-1].ownership(faces, edges, valid)
        surfaces = self.face_head(faces).reshape(
            len(counts), padded.shape[1], self.face_resolution, self.face_resolution, 3
        ).masked_fill(~valid[..., None, None, None], 0)
        curves = self.curve_head(edges).reshape(len(counts), self.num_queries, self.curve_resolution, 3)
        return {
            "face_points": surfaces,
            "curves": curves,
            "existence_logits": self.existence_head(edges).squeeze(-1),
            "ownership_logits": ownership_logits,
            "face_valid": valid,
            "face_counts": counts_tensor,
            "face_states": faces,
            "edge_states": edges,
        }
