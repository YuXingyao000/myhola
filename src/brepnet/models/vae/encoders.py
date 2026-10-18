"""Local geometry encoders, holistic graph fusion, and the face posterior."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn, Tensor
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_scatter import scatter_mean


class UVFaceEncoder(nn.Module):
    """Encode a 16 x 16 XYZ/normal grid without intermediate spatial pooling.

    This optional replacement for the face CNN keeps all 256 parameter-grid
    positions through its self-attention layers. A single learned query then
    reads the complete face into one vector. Faces are independent batch items;
    the existing downstream face attention/GAT performs cross-face fusion.

    The UV coordinates are normalized grid indices, not additional CAD samples
    or a claim that the samples have equal spacing in world coordinates.
    Input: ``[M, 6, 16, 16]``. Output: ``[M, out_channels]``.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 32,
        hidden_dim: int = 512,
        embedding_dim: int = 128,
        num_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(in_channels, bool) or not isinstance(in_channels, int) or in_channels != 6:
            raise ValueError("UVFaceEncoder requires in_channels=6 (XYZ and normal)")
        dimensions = {
            "out_channels": out_channels,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_dim % nhead:
            raise ValueError("hidden_dim must be divisible by nhead")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError("dropout must be a number in [0, 1)")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        self.geometry_embedding = nn.Sequential(
            nn.Linear(in_channels, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, hidden_dim),
        )
        self.uv_embedding = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, hidden_dim),
        )
        coordinates = torch.arange(16, dtype=torch.float32) / 15
        u, v = torch.meshgrid(coordinates, coordinates, indexing="ij")
        # Deterministic for every checkpoint; follows flatten(2)'s i-major order.
        self.register_buffer(
            "uv_grid", torch.stack((u, v), dim=-1).reshape(1, 256, 2), persistent=False
        )
        # Instantiate each layer separately: no copied/shared initial parameters.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.query, std=0.02)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.readout_attention = nn.MultiheadAttention(
            hidden_dim, nhead, dropout=dropout, batch_first=True
        )
        self.readout_dropout = nn.Dropout(dropout)
        self.readout_ffn_norm = nn.LayerNorm(hidden_dim)
        self.readout_ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, out_channels)

    def forward(self, face_points: Tensor) -> Tensor:
        if not isinstance(face_points, Tensor) or face_points.layout != torch.strided:
            raise ValueError("face_points must be a dense floating tensor [M, 6, 16, 16]")
        if (face_points.ndim != 4 or tuple(face_points.shape[1:]) != (6, 16, 16)
                or face_points.shape[0] == 0 or not face_points.is_floating_point()):
            raise ValueError("face_points must have floating shape [M, 6, 16, 16] with M > 0")

        # [M, 6, 16, 16] -> [M, 256, 6] -> [M, 256, hidden_dim].
        points = face_points.flatten(2).transpose(1, 2)
        tokens = self.geometry_embedding(points) + self.uv_embedding(self.uv_grid)
        for layer in self.layers:
            tokens = layer(tokens)
        memory = self.memory_norm(tokens)

        # Q: [M, 1, hidden_dim]; K/V: all [M, 256, hidden_dim] positions.
        query = self.query.expand(face_points.shape[0], -1, -1)
        readout, _ = self.readout_attention(
            self.query_norm(query), memory, memory, need_weights=False
        )
        summary = query + self.readout_dropout(readout)
        summary = summary + self.readout_ffn(self.readout_ffn_norm(summary))
        return self.output_projection(self.output_norm(summary)).squeeze(1)


class EdgeValueGATv2Conv(GATv2Conv):
    """Keep GATv2 attention and add an edge-conditioned message value.

    This experiment supports the encoder's dense, homogeneous, single-head
    graph. The original GATv2 output and parameter names are preserved. A new
    ``value_net`` reads ``[target_face, source_face, edge]`` and contributes an
    attention-weighted residual. Its final layer starts at zero, so importing
    the old GAT weights preserves the initial function exactly.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        *,
        edge_dim: int,
        value_hidden_dim: int = 64,
        **kwargs,
    ) -> None:
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError("EdgeValueGATv2Conv requires a positive homogeneous in_channels")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError("out_channels must be a positive integer")
        if heads != 1:
            raise ValueError("EdgeValueGATv2Conv supports heads=1 only")
        if not isinstance(edge_dim, int) or edge_dim <= 0:
            raise ValueError("edge_dim must be a positive integer")
        if not isinstance(value_hidden_dim, int) or value_hidden_dim <= 0:
            raise ValueError("value_hidden_dim must be a positive integer")
        if kwargs.get("flow", "source_to_target") != "source_to_target":
            raise ValueError("EdgeValueGATv2Conv requires source_to_target flow")
        if kwargs.get("aggr", "add") not in ("add", "sum"):
            raise ValueError("EdgeValueGATv2Conv requires sum aggregation")
        if kwargs.get("decomposed_layers", 1) != 1:
            raise ValueError("EdgeValueGATv2Conv requires decomposed_layers=1")
        super().__init__(
            in_channels, out_channels, heads=heads, edge_dim=edge_dim, **kwargs
        )
        self.value_hidden_dim = value_hidden_dim
        # Initializing the optional branch must not advance the old model's RNG.
        devices = [self.att.device.index] if self.att.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            factory_kwargs = {"device": self.att.device, "dtype": self.att.dtype}
            self.value_net = nn.Sequential(
                nn.Linear(2 * in_channels + edge_dim, value_hidden_dim, **factory_kwargs),
                nn.SiLU(),
                nn.Linear(value_hidden_dim, out_channels, **factory_kwargs),
            )
            nn.init.zeros_(self.value_net[-1].weight)
            nn.init.zeros_(self.value_net[-1].bias)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        # GATv2's constructor calls this before value_net exists.
        if hasattr(self, "value_net"):
            devices = [self.att.device.index] if self.att.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                self.value_net[0].reset_parameters()
                self.value_net[-1].reset_parameters()
                nn.init.zeros_(self.value_net[-1].weight)
                nn.init.zeros_(self.value_net[-1].bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        return_attention_weights: bool | None = None,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        if not isinstance(x, Tensor) or x.layout != torch.strided:
            raise ValueError("x must be a dense homogeneous node tensor, not a bipartite pair")
        if x.ndim != 2 or x.shape[1] != self.in_channels or not x.is_floating_point():
            raise ValueError(f"x must have floating shape [N, {self.in_channels}]")
        if not isinstance(edge_index, Tensor) or edge_index.layout != torch.strided:
            raise ValueError("edge_index must be a dense [2, E] index tensor; sparse graphs are unsupported")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
            raise ValueError("edge_index must have shape [2, E] and dtype torch.long")
        if not isinstance(edge_attr, Tensor) or edge_attr.layout != torch.strided:
            raise ValueError("edge_attr is required as a dense [E, edge_dim] tensor")
        if edge_attr.shape != (edge_index.shape[1], self.edge_dim) or not edge_attr.is_floating_point():
            raise ValueError(f"edge_attr must have floating shape [E, {self.edge_dim}]")
        if x.device != edge_index.device or x.device != edge_attr.device:
            raise ValueError("x, edge_index and edge_attr must be on the same device")

        out, (attention_edges, alpha) = super().forward(
            x, edge_index, edge_attr, return_attention_weights=True
        )
        # Match PyG's self-loop removal, ordering and mean-filled attributes.
        # Keep these tensors local so checkpoint recomputation is reentrant.
        if self.add_self_loops:
            loop_edges, edge_attr = remove_self_loops(edge_index, edge_attr)
            _, edge_attr = add_self_loops(
                loop_edges, edge_attr, fill_value=self.fill_value, num_nodes=x.shape[0]
            )
        source, target = attention_edges
        values = self.value_net(torch.cat([x[target], x[source], edge_attr], dim=-1))
        weighted_values = (values * alpha).to(dtype=out.dtype)
        residual = out.new_zeros(out.shape).index_add(0, target, weighted_values)
        out = out + residual
        # PyG returns attention for either boolean value (including False).
        if isinstance(return_attention_weights, bool):
            return out, (attention_edges, alpha)
        return out


class HolisticEncoder(nn.Module):
    def __init__(self, legacy: nn.Module, edge_value_cfg: Mapping | None = None):
        super().__init__()
        self.in_channels = legacy.in_channels
        self.face_coords = legacy.face_coords
        self.edge_coords = legacy.edge_coords
        self.face_attn_proj_in = legacy.face_attn_proj_in
        self.face_attn = legacy.face_attn
        self.face_attn_proj_out = legacy.face_attn_proj_out
        self.graph_face_edge = legacy.graph_face_edge
        self.global_feature1 = legacy.global_feature1
        self.global_feature2 = legacy.global_feature2
        # Added 2026-10-15: retain every GAT score/value parameter; only the
        # direct edge-content message is new. No change when disabled.
        if (edge_value_cfg or {}).get("enabled", False):
            hidden_dim = int(edge_value_cfg.get("hidden_dim", 64))
            if hidden_dim <= 0:
                raise ValueError("encoder_edge_value.hidden_dim must be positive")
            # Model construction must not shift subsequent dropout/sampling RNG.
            with torch.random.fork_rng(devices=[]):
                for index, layer in enumerate(self.graph_face_edge):
                    if not isinstance(layer, GATv2Conv):
                        continue
                    replacement = EdgeValueGATv2Conv(
                        layer.in_channels, layer.out_channels, heads=layer.heads,
                        edge_dim=layer.edge_dim, value_hidden_dim=hidden_dim,
                        concat=layer.concat, negative_slope=layer.negative_slope,
                        dropout=layer.dropout, add_self_loops=layer.add_self_loops,
                        fill_value=layer.fill_value, bias=layer.bias is not None,
                        share_weights=layer.share_weights,
                        residual=layer.res is not None,
                    )
                    fresh = {key: value for key, value in replacement.state_dict().items()
                             if key.startswith("value_net.")}
                    replacement.load_state_dict({**layer.state_dict(), **fresh}, strict=True)
                    self.graph_face_edge[index] = replacement

    def fuse_graph(self, faces: Tensor, edges: Tensor, connectivity: Tensor) -> Tensor:
        edge_index = connectivity[:, 1:].T
        edge_attributes = edges[connectivity[:, 0]]
        hidden = faces
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                hidden = hidden + layer(hidden, edge_index, edge_attributes)
            else:
                hidden = layer(hidden)
        return faces + hidden

    def fuse_global(self, faces: Tensor, face_counts: Tensor) -> Tensor:
        object_ids = torch.arange(len(face_counts), device=faces.device)
        object_ids = object_ids.repeat_interleave(face_counts)
        pooled = scatter_mean(faces, object_ids, dim=0)
        context = self.global_feature1(pooled).repeat_interleave(face_counts, dim=0)
        return faces + self.global_feature2(torch.cat((faces, context), dim=-1))

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        face_points = batch["face_points"][..., :self.in_channels].permute(0, 3, 1, 2).contiguous()
        edge_points = batch["edge_points"][..., :self.in_channels].permute(0, 2, 1).contiguous()
        faces = self.face_coords(face_points)
        # Empty edges are legitimate for a single-face input. Avoid asking CNN
        # backends to infer a zero-length batch's reshape dimensions.
        edges = self.edge_coords(edge_points) if len(edge_points) else faces.new_empty((0, faces.shape[-1] * 2))

        hidden = self.face_attn_proj_in(faces)
        hidden = self.face_attn(hidden, batch["attn_mask"])
        faces = faces + self.face_attn_proj_out(hidden)
        faces = self.fuse_graph(faces, edges, batch["edge_face_connectivity"])
        faces = self.fuse_global(faces, batch["face_counts"])
        return {"face_features": faces, "edge_features": edges}


class FacePosterior(nn.Module):
    def __init__(self, legacy: nn.Module):
        super().__init__()
        self.projection = legacy.gaussian_proj
        self.latent_dim = legacy.face_feature_dim
        self.kl_weight = legacy.gaussian_weights
        self.kl_reduction = legacy.kl_reduction

    def forward(self, face_features: Tensor, use_mean: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        projected = self.projection(face_features)
        if self.kl_weight <= 0:
            return projected, projected.new_zeros(()), projected
        statistics = projected.reshape(-1, self.latent_dim, 2)
        mean, log_variance = statistics[..., 0], statistics[..., 1]
        std = (0.5 * log_variance).exp()
        latents = mean if use_mean else mean + std * torch.randn_like(std)
        terms = 1 + log_variance - mean.square() - log_variance.exp()
        kl = -0.5 * (terms.mean() if self.kl_reduction == "mean" else terms.sum())
        return latents, kl * self.kl_weight, torch.cat((mean, std), dim=-1)
