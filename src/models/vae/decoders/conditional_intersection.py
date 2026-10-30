"""Directed, position-conditioned reading of a face's CNN spatial features.

One ordered pair (i, j) keeps one query and reads only face j. The reverse
pair is a separate query; shared XYZ construction remains downstream.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class UVPositionEncoding(nn.Module):
    """Normalized (u, v) -> sine/cosine channels; row=v, column=u."""

    def __init__(self, width: int):
        super().__init__()
        frequency = 10000 ** (2 * torch.arange(width // 4).float() / (width // 2))
        self.register_buffer("frequency", frequency, persistent=False)

    def forward(self, uv: Tensor) -> Tensor:
        phase = 2 * math.pi * uv.float().unsqueeze(-1) / self.frequency
        # [..., 2, width/4, sin/cos] -> [..., width], u channels then v.
        return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-3)

    def grid(self, height: int, width: int) -> Tensor:
        v, u = torch.meshgrid(
            torch.linspace(0, 1, height, device=self.frequency.device),
            torch.linspace(0, 1, width, device=self.frequency.device), indexing="ij",
        )
        return torch.stack((u, v), dim=-1).reshape(-1, 2)


class FaceSpatialMemory(nn.Module):
    """Broadcast the old face token and add a zero-initialized local branch."""

    def __init__(self, local_channels: int, width: int):
        super().__init__()
        self.local_projection = nn.Linear(local_channels, width)
        nn.init.zeros_(self.local_projection.weight)
        nn.init.zeros_(self.local_projection.bias)

    def forward(self, global_tokens: Tensor, local_features: Tensor) -> Tensor:
        local = local_features.flatten(2).transpose(1, 2)  # [N, 256, 96]
        return global_tokens[:, None] + self.local_projection(local)  # [N, 256, 512]


class ReferenceUVHead(nn.Module):
    """The ordered pair predicts one reference in the second face's chart."""

    def __init__(self, face_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2 * face_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2),
        )

    def forward(self, pair_features: Tensor) -> Tensor:
        return self.layers(pair_features.flatten(1)).sigmoid().unsqueeze(1)  # [P, 1, 2]


class ConditionalCrossAttention(nn.Module):
    """Reuse old content Q/K/V, adding separate positional Q/K per head.

    Legacy in_proj_weight, in_proj_bias and out_proj names are retained.
    Memory projection is per face; pair gathering is inside checkpoint chunks.
    """

    def __init__(self, legacy: nn.MultiheadAttention, first_layer: bool):
        super().__init__()
        self.embed_dim = legacy.embed_dim
        self.num_heads = legacy.num_heads
        self.dropout = legacy.dropout
        self.in_proj_weight = legacy.in_proj_weight
        self.in_proj_bias = legacy.in_proj_bias
        self.out_proj = legacy.out_proj
        self.query_position = nn.Linear(self.embed_dim, self.embed_dim)
        self.key_position = nn.Linear(self.embed_dim, self.embed_dim)
        self.query_scale = None if first_layer else nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def _heads(self, tokens: Tensor) -> Tensor:
        batch, length, _ = tokens.shape
        return tokens.reshape(batch, length, self.num_heads, -1).transpose(1, 2)

    def project_memory(self, memory: Tensor, grid_position: Tensor) -> tuple[Tensor, ...]:
        width = self.embed_dim
        keys = F.linear(memory, self.in_proj_weight[width:2 * width], self.in_proj_bias[width:2 * width])
        values = F.linear(memory, self.in_proj_weight[2 * width:], self.in_proj_bias[2 * width:])
        positions = self._heads(self.key_position(grid_position).unsqueeze(0))
        return self._heads(keys), self._heads(values), positions

    def forward(
        self, query: Tensor, reference_position: Tensor, keys: Tensor,
        values: Tensor, key_positions: Tensor, owner_ids: Tensor,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        content = self._heads(F.linear(
            query, self.in_proj_weight[:self.embed_dim], self.in_proj_bias[:self.embed_dim],
        ))  # [P, 8, 1, 64]
        scale = 1 if self.query_scale is None else self.query_scale(query)
        position = self._heads(self.query_position(reference_position * scale))
        scores = content @ keys[owner_ids].transpose(-1, -2)
        scores = scores + position @ key_positions.transpose(-1, -2)
        weights = (scores / math.sqrt(2 * content.shape[-1])).float().softmax(-1)
        attention = F.dropout(weights.to(values.dtype), self.dropout, self.training)
        output = attention @ values[owner_ids]  # [P, 8, 1, 64]
        output = output.transpose(1, 2).reshape(query.shape[0], 1, self.embed_dim)
        return self.out_proj(output), weights if return_attention else None


class ConditionalIntersectionLayer(nn.Module):
    """Old pre-LN self-attention/FFN, with a conditional cross-attention slot."""

    def __init__(self, legacy: nn.TransformerDecoderLayer, first_layer: bool, cfg: Mapping):
        super().__init__()
        for name, module in legacy.named_children():
            self.add_module(name, module)
        self.multihead_attn = ConditionalCrossAttention(legacy.multihead_attn, first_layer)
        self.activation = legacy.activation
        self.chunk_size = int(cfg.get("candidate_chunk_size", 64))
        self.checkpoint_candidates = bool(cfg.get("checkpoint_candidates", True))

    def _run_chunk(
        self, query: Tensor, reference: Tensor, keys: Tensor, values: Tensor,
        key_positions: Tensor, owner_ids: Tensor, return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        normalized = self.norm1(query)
        update = self.self_attn(normalized, normalized, normalized, need_weights=False)[0]
        query = query + self.dropout1(update)
        update, weights = self.multihead_attn(
            self.norm2(query), reference, keys, values, key_positions, owner_ids,
            return_attention=return_attention,
        )
        query = query + self.dropout2(update)
        update = self.linear2(self.dropout(self.activation(self.linear1(self.norm3(query)))))
        return query + self.dropout3(update), weights

    def forward(
        self, query: Tensor, reference: Tensor, memory: Tensor,
        grid_position: Tensor, owner_ids: Tensor, return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        keys, values, positions = self.multihead_attn.project_memory(memory, grid_position)
        outputs, attention = [], []
        for start in range(0, query.shape[0], self.chunk_size):
            end = start + self.chunk_size
            args = (query[start:end], reference[start:end], keys, values, positions, owner_ids[start:end])
            if self.checkpoint_candidates and self.training and torch.is_grad_enabled() and not return_attention:
                output, weights = checkpoint(self._run_chunk, *args, use_reentrant=False)
            else:
                output, weights = self._run_chunk(*args, return_attention=return_attention)
            outputs.append(output)
            if return_attention:
                attention.append(weights)
        return torch.cat(outputs), torch.cat(attention) if return_attention else None


class ConditionalIntersection(nn.Module):
    """[N,32] faces + [N,96,16,16] maps + [P,2] pairs -> [P,64] edges.

    Each legacy parameter retains its name and value. The local memory starts
    at zero, giving the old eval output (up to floating-point summation). It
    does not preserve train-time dropout realizations after 1 -> 256 keys.
    """

    def __init__(self, legacy_inter: nn.Module, local_channels: int, cfg: Mapping):
        super().__init__()
        self.attn_proj_in = legacy_inter.attn_proj_in
        self.attn_proj_out = legacy_inter.attn_proj_out
        self.pos_encoding = legacy_inter.pos_encoding
        self.num_layers = len(legacy_inter.layers)
        self.edge_dim = self.attn_proj_out.out_features
        width = self.attn_proj_in.out_features
        self.spatial_memory = FaceSpatialMemory(local_channels, width)
        self.reference_uv_head = ReferenceUVHead(
            self.attn_proj_in.in_features, int(cfg.get("reference_hidden_dim", 128)),
        )
        self.uv_position = UVPositionEncoding(width)
        self.layers = nn.ModuleList([
            ConditionalIntersectionLayer(layer, index == 0, cfg)
            for index, layer in enumerate(legacy_inter.layers)
        ])

    def _inputs(self, faces: Tensor, local: Tensor, pairs: Tensor) -> tuple[Tensor, ...]:
        projected = self.attn_proj_in(faces)  # [N, 32] -> [N, 512]
        query = projected[pairs[:, 0], None] + self.pos_encoding[:, 0:1]
        memory = self.spatial_memory(projected + self.pos_encoding[:, 1], local)
        reference = self.reference_uv_head(faces[pairs])  # [P, 1, 2], fixed across layers
        reference_position = self.uv_position(reference).to(query.dtype)
        grid = self.uv_position.grid(*local.shape[-2:])
        grid_position = self.uv_position(grid).to(query.dtype)  # [256, 512]
        return query, reference, reference_position, memory, grid_position

    def forward(self, face_features: Tensor, local_features: Tensor, pair_indices: Tensor) -> Tensor:
        if pair_indices.shape[0] == 0:
            return self.attn_proj_out(self.attn_proj_in(face_features[:0]))
        query, _, reference_position, memory, grid_position = self._inputs(
            face_features, local_features, pair_indices,
        )
        for layer in self.layers:
            query, _ = layer(query, reference_position, memory, grid_position, pair_indices[:, 1])
        return self.attn_proj_out(query[:, 0])  # [P, 512] -> [P, 64]

    @torch.no_grad()
    def inspect_attention(
        self, face_features: Tensor, local_features: Tensor, pair_indices: Tensor, layer_index: int = -1,
    ) -> dict[str, Tensor]:
        """Inspect selected pairs in eval mode; no persistent training hooks."""
        layer_index = layer_index % self.num_layers
        query, reference, reference_position, memory, grid_position = self._inputs(
            face_features, local_features, pair_indices,
        )
        if pair_indices.shape[0] == 0:
            heads = self.layers[layer_index].multihead_attn.num_heads
            weights = query.new_empty((0, heads, 1, grid_position.shape[0]))
        else:
            for index, layer in enumerate(self.layers[:layer_index + 1]):
                query, weights = layer(
                    query, reference_position, memory, grid_position, pair_indices[:, 1],
                    return_attention=index == layer_index,
                )
        return {"reference_uv": reference, "attention": weights,
                "grid_uv": self.uv_position.grid(*local_features.shape[-2:])}
