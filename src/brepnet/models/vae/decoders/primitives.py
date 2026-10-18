"""Face features, primitive geometry heads, and optional curve auxiliary heads."""

from __future__ import annotations

import torch
from torch import nn, Tensor

from ..state import PrimitiveGeometry


class FaceFeatureDecoder(nn.Module):
    """Run the original face Transformer once with a packed object mask."""

    def __init__(self, input_projection: nn.Module, backbone: nn.Module, output_projection: nn.Module):
        super().__init__()
        self.input_projection = input_projection
        self.backbone = backbone
        self.output_projection = output_projection

    def forward(self, face_latents: Tensor, face_counts: Tensor) -> Tensor:
        object_ids = torch.repeat_interleave(
            torch.arange(face_counts.numel(), device=face_latents.device), face_counts
        )
        attention_mask = object_ids[:, None] != object_ids[None, :]
        hidden = self.input_projection(face_latents)
        hidden = self.backbone(hidden, mask=attention_mask)
        return self.output_projection(hidden)


class FacePairConditioning(nn.Module):
    """Condition the final curve heads; zero outputs preserve the old heads.

    Added for curve_conditioning: a shared MLP reads [E, F_i, F_j], then
    separate zero-initialized heads produce channel-wise scale and shift for
    the curve CNN and center/scale MLP. No normalization is inserted.
    """

    def __init__(self, condition_dim: int, hidden_dim: int, point_channels: int, bbox_channels: int):
        super().__init__()
        if min(condition_dim, hidden_dim, point_channels, bbox_channels) <= 0:
            raise ValueError("Curve conditioning dimensions must be positive.")
        self.condition_dim = condition_dim
        self.projection = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.SiLU())
        self.point_affine = nn.Linear(hidden_dim, 2 * point_channels)
        self.bbox_affine = nn.Linear(hidden_dim, 2 * bbox_channels)
        for head in (self.point_affine, self.bbox_affine):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, condition: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError(f"Expected curve condition [P,{self.condition_dim}].")
        hidden = self.projection(condition)
        point_scale, point_shift = self.point_affine(hidden).chunk(2, dim=-1)
        bbox_scale, bbox_shift = self.bbox_affine(hidden).chunk(2, dim=-1)
        return point_scale, point_shift, bbox_scale, bbox_shift


class PrimitiveDecoder(nn.Module):
    """Retained local point and center/scale heads, with explicit global XYZ."""

    def __init__(
        self,
        points_decoder: nn.Module,
        bbox_decoder: nn.Module,
        spatial_shape: tuple[int, ...],
        point_channels: int,
        condition_dim: int | None = None,
        condition_hidden_dim: int = 128,
    ):
        super().__init__()
        self.points_decoder = points_decoder
        self.bbox_decoder = bbox_decoder
        self.spatial_shape = spatial_shape
        self.point_channels = point_channels
        self.conditioning = None
        if condition_dim is not None:
            if spatial_shape != (16,) or not isinstance(points_decoder, nn.Sequential):
                raise ValueError("Curve conditioning requires the retained Sequential curve CNN.")
            indices = [i for i, layer in enumerate(points_decoder) if isinstance(layer, nn.Conv1d)]
            if not indices:
                raise ValueError("Curve conditioning requires a final Conv1d output layer.")
            self.point_condition_index = indices[-1]
            output_layer = points_decoder[self.point_condition_index]
            if output_layer.out_channels != point_channels or output_layer.kernel_size != (1,):
                raise ValueError("Curve conditioning expects the original final 1x1 Conv1d.")
            if not isinstance(bbox_decoder, nn.Sequential) or not isinstance(bbox_decoder[-1], nn.Linear):
                raise ValueError("Curve conditioning requires the retained Sequential bbox MLP.")
            if bbox_decoder[-1].out_features != 4:
                raise ValueError("The retained bbox head must output center 3 + scale 1.")
            self.bbox_condition_index = len(bbox_decoder) - 1
            self.conditioning = FacePairConditioning(
                condition_dim, condition_hidden_dim, output_layer.in_channels,
                bbox_decoder[-1].in_features,
            )

    def _empty_geometry(self, features: Tensor) -> PrimitiveGeometry:
        return PrimitiveGeometry(
            local_points=features.new_empty((0, *self.spatial_shape, self.point_channels)),
            center_scale=features.new_empty((0, 4)),
            points=features.new_empty((0, *self.spatial_shape, 3), dtype=torch.float32),
        )

    def _with_global_points(
        self, features: Tensor, local_points: Tensor, center_scale: Tensor | None = None,
    ) -> PrimitiveGeometry:
        if center_scale is None:
            center_scale = self.bbox_decoder(features)
        with torch.autocast(device_type=features.device.type, enabled=False):
            broadcast_shape = (features.shape[0],) + (1,) * len(self.spatial_shape)
            center = center_scale[:, :3].float().reshape(*broadcast_shape, 3)
            scale = center_scale[:, 3:].float().reshape(*broadcast_shape, 1)
            points = local_points[..., :3].float() * scale + center
        return PrimitiveGeometry(local_points=local_points, center_scale=center_scale, points=points)

    def _local_output_layer_index(self) -> int:
        if not isinstance(self.points_decoder, nn.Sequential):
            raise ValueError("Local intersection requires a Sequential face points decoder.")
        indices = [
            index for index, module in enumerate(self.points_decoder)
            if isinstance(module, nn.Conv2d)
        ]
        if not indices:
            raise ValueError("Local intersection requires a final Conv2d face output layer.")
        index = indices[-1]
        layer = self.points_decoder[index]
        if layer.out_channels != self.point_channels or layer.kernel_size != (1, 1):
            raise ValueError("Expected a final 1x1 Conv2d projecting local face features to points.")
        return index

    @property
    def local_feature_channels(self) -> int:
        return self.points_decoder[self._local_output_layer_index()].in_channels

    def forward_with_local_features(self, features: Tensor) -> tuple[PrimitiveGeometry, Tensor]:
        """Decode once and expose the map immediately before the output Conv2d.

        Iteration preserves all original module registrations and checkpoint
        names. The captured map stays connected to the face CNN's gradients.
        """
        output_index = self._local_output_layer_index()
        if features.shape[0] == 0:
            local_map = features.new_empty((0, self.local_feature_channels, *self.spatial_shape))
            return self._empty_geometry(features), local_map
        points = features
        local_map = None
        for index, module in enumerate(self.points_decoder):
            if index == output_index:
                local_map = points
            points = module(points)
        assert local_map is not None
        return self._with_global_points(features, points), local_map

    def forward(self, features: Tensor, condition: Tensor | None = None) -> PrimitiveGeometry:
        if features.shape[0] == 0:
            return self._empty_geometry(features)
        if condition is None:
            # Coarse geometry and the encoder-edge teacher keep this path.
            local_points = self.points_decoder(features)
            return self._with_global_points(features, local_points)
        if self.conditioning is None or condition.shape[0] != features.shape[0]:
            raise ValueError("Curve conditioning must be enabled and match the edge batch.")

        point_scale, point_shift, bbox_scale, bbox_shift = self.conditioning(condition)
        local_points = features
        for index, layer in enumerate(self.points_decoder):
            if index == self.point_condition_index:
                local_points = (1 + point_scale[..., None]) * local_points + point_shift[..., None]
            local_points = layer(local_points)
        center_scale = features
        for index, layer in enumerate(self.bbox_decoder):
            if index == self.bbox_condition_index:
                center_scale = (1 + bbox_scale) * center_scale + bbox_shift
            center_scale = layer(center_scale)
        return self._with_global_points(features, local_points, center_scale)


class UVPathHead(nn.Module):
    """Predict two owner UV paths without changing the retained XYZ decoder.

    Each ordered pair has one sequence of 16 curve-parameter tokens. The four
    outputs at each token are (u_i, v_i, u_j, v_j); they are not clamped because
    supervision uses an explicitly selected CAD parameter-chart branch.
    """

    def __init__(
        self, face_dim: int, edge_dim: int, hidden_dim: int = 128,
        num_layers: int = 2, nhead: int = 4, dim_feedforward: int = 256,
        dropout: float = 0.0, candidate_chunk_size: int = 256,
    ):
        super().__init__()
        if min(face_dim, edge_dim, hidden_dim, num_layers, nhead, dim_feedforward, candidate_chunk_size) <= 0:
            raise ValueError("UV path dimensions, layers and chunk size must be positive")
        if hidden_dim % nhead or not 0 <= dropout < 1:
            raise ValueError("UV hidden_dim must be divisible by nhead and dropout must be in [0,1)")
        self.face_dim, self.edge_dim = face_dim, edge_dim
        self.chunk_size = candidate_chunk_size
        self.context = nn.Sequential(nn.Linear(edge_dim + 2 * face_dim, hidden_dim), nn.SiLU())
        self.position = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, hidden_dim))
        # Construct layers independently, rather than copying identical initial weights.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                hidden_dim, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 4))
        self.register_buffer("sample_t", torch.linspace(0, 1, 16).reshape(16, 1), persistent=False)

    def forward(self, edges: Tensor, faces: Tensor, owners: Tensor) -> Tensor:
        if edges.ndim != 2 or edges.shape[1] != self.edge_dim:
            raise ValueError(f"UV edges must have shape [P,{self.edge_dim}]")
        if faces.ndim != 2 or faces.shape[1] != self.face_dim or owners.shape != (len(edges), 2):
            raise ValueError("UV faces or owner pair shape is invalid")
        if not len(edges):
            zero = edges.sum() * 0 + faces.sum() * 0
            for parameter in self.parameters():
                zero = zero + parameter.sum() * 0
            return edges.new_empty((0, 2, 16, 2)) + zero
        position = self.position(self.sample_t.to(dtype=edges.dtype))
        chunks = []
        for start in range(0, len(edges), self.chunk_size):
            stop = start + self.chunk_size
            condition = torch.cat((edges[start:stop], faces[owners[start:stop]].flatten(1)), dim=-1)
            tokens = self.context(condition)[:, None, :] + position[None]
            for layer in self.layers:
                tokens = layer(tokens)
            paths = self.output(tokens).reshape(-1, 16, 2, 2).permute(0, 2, 1, 3)
            chunks.append(paths)
        return torch.cat(chunks)
