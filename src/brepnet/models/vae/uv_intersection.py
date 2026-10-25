"""可独立阅读的 UV decoder：所有自定义 module 都在这个文件内。

从 UVIntersectionDecoder.forward 开始阅读；输入是已有 encoder 的 latent。
    z[N,64] -> 全局面特征 -> 面 CNN / center-scale -> 面采样点
                          -> 两面空间 token 拼接 -> curve query -> 双侧 UV
    面采样点 -> 固定可微 B-Spline -> 在双侧 UV 求值 -> 三维几何 loss

训练接入与旧权重迁移见 uv_intersection_training.py；OCC 构造尚未接入。
网格一律是 [v,u]；UV 向量是 (u,v)。每个无序面对最多一个边候选。
固定曲面为非周期三次插值；不保证周期接缝、连续边闭合或双侧轨迹相等。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class UVDecoderConfig:
    latent_dim: int = 64
    face_width: int = 768
    face_layers: int = 8
    face_heads: int = 16
    face_ffn: int = 2048
    curve_width: int = 512
    curve_layers: int = 8
    curve_heads: int = 8
    curve_ffn: int = 2048
    dropout: float = 0.1
    pair_chunk_size: int = 16


class UVIntersectionDecoder(nn.Module):
    """主入口：latent 和每个对象的输入 token 数；GT 只进入 loss。"""

    def __init__(self, cfg: UVDecoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or UVDecoderConfig()
        self.global_attention = GlobalFaceAttention(self.cfg)
        self.face_decoder = FaceDecoder(self.cfg)
        self.face_pairs = UnorderedFacePairs()
        self.face_tokens = FaceSpatialTokens(self.cfg.face_width // 8, self.cfg.curve_width)
        self.intersection = UVIntersection(self.cfg)
        self.surface = FixedBSplineSurface()

    def forward(self, latents: Tensor, face_counts: Tensor) -> dict[str, Tensor]:
        # packed N 是输入latent数；生成时必须先完成上游padding去重。
        object_ids = torch.repeat_interleave(torch.arange(len(face_counts), device=latents.device), face_counts)
        faces = self.global_attention(latents, object_ids)                   # [N,64]
        face = self.face_decoder(faces)                                     # [N,16,16,6], [N,96,16,16]
        owners = self.face_pairs(object_ids)                                # [P,2], i<j
        tokens = self.face_tokens(face["spatial"], face["xyz"])              # [N,256,512], 每面一次
        uv, logits = self._decode_candidates(faces, tokens, owners)
        poles = self.surface(face["xyz"])                                   # [N,16,16,3], 一面一次
        curves = self.surface.evaluate(poles[owners], uv)                    # [P,2,16,3]
        return {
            "face_local": face["local"], "face_center_scale": face["placement"],
            "face_xyz": face["xyz"], "surface_poles": poles,
            "edge_pairs": owners, "edge_logits": logits, "edge_uv": uv,
            "edge_xyz": curves,  # 两条面上轨迹；不是已保证相等的唯一3D边
        }

    def _decode_candidates(self, faces: Tensor, tokens: Tensor, owners: Tensor) -> tuple[Tensor, Tensor]:
        """这里只做候选分块；网络主线见 UVIntersection.forward。"""
        if not len(owners):
            zero = faces.sum() * 0 + tokens.sum() * 0
            return faces.new_empty((0, 2, 16, 2)) + zero, faces.new_empty(0) + zero
        uv_chunks, logit_chunks = [], []
        for pair_chunk in owners.split(self.cfg.pair_chunk_size):
            uv, logits = self.intersection(faces, tokens, pair_chunk)
            uv_chunks.append(uv)
            logit_chunks.append(logits)
        return torch.cat(uv_chunks), torch.cat(logit_chunks)


class ResidualLinear(nn.Module):
    """复制旧 ResBlockXD 的 Linear 分支，保留 LayerNorm 和投影 skip。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.skip = nn.Identity() if in_dim == out_dim else nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        hidden = F.relu(self.norm1(self.linear1(x)))
        hidden = self.norm2(self.linear2(hidden))
        return F.relu(hidden + self.skip(x))


class ResidualGrid(nn.Module):
    """复制旧 Conv2D 残差块；LN 覆盖整个 [C,V,U]，不是仅通道。"""

    def __init__(self, in_channels: int, out_channels: int, size: int):
        super().__init__()
        shape = (out_channels, size, size)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.LayerNorm(shape)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.LayerNorm(shape)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.LayerNorm(shape),
        )

    def forward(self, grid: Tensor) -> Tensor:
        hidden = F.relu(self.norm1(self.conv1(grid)))
        hidden = self.norm2(self.conv2(hidden))
        return F.relu(hidden + self.skip(grid))


class UpsampleFaceGrid(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, output_size: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.refine = ResidualGrid(out_channels, out_channels, output_size)

    def forward(self, grid: Tensor) -> Tensor:
        return self.refine(self.upsample(grid))  # [N,C,V,U] -> [N,C/2,2V,2U]


class GlobalFaceAttention(nn.Module):
    """保留原 8 层全局 Transformer；不同物体之间不可互相读取。"""

    def __init__(self, cfg: UVDecoderConfig):
        super().__init__()
        self.input = nn.Linear(cfg.latent_dim, cfg.face_width)
        layer = nn.TransformerEncoderLayer(
            cfg.face_width, cfg.face_heads, cfg.face_ffn, cfg.dropout,
            activation="relu", batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, cfg.face_layers, enable_nested_tensor=False)
        self.output = nn.Linear(cfg.face_width, cfg.latent_dim)

    def forward(self, latents: Tensor, object_ids: Tensor) -> Tensor:
        mask = object_ids[:, None] != object_ids[None, :]  # [N,N]
        hidden = self.input(latents)                     # [N,64] -> [N,768]
        hidden = self.layers(hidden, mask=mask)          # [N,768]
        return self.output(hidden)                      # [N,64]


class FacePointDecoder(nn.Module):
    """原面 CNN；同时暴露最终 1x1 Conv 前的空间特征。"""

    def __init__(self, latent_dim: int, width: int):
        super().__init__()
        self.stem = ResidualGrid(latent_dim // 4, width, 2)
        self.to_4 = UpsampleFaceGrid(width, width // 2, 4)
        self.to_8 = UpsampleFaceGrid(width // 2, width // 4, 8)
        self.to_16 = UpsampleFaceGrid(width // 4, width // 8, 16)
        self.output = nn.Conv2d(width // 8, 6, 1)

    def forward(self, faces: Tensor) -> tuple[Tensor, Tensor]:
        grid = faces.reshape(faces.shape[0], -1, 2, 2)  # [N,64] -> [N,16,2,2]
        grid = self.stem(grid)                         # [N,768,2,2]
        grid = self.to_4(grid)                         # [N,384,4,4]
        grid = self.to_8(grid)                         # [N,192,8,8]
        spatial = self.to_16(grid)                     # [N,96,16,16]
        local = self.output(spatial).permute(0, 2, 3, 1)  # [N,16,16,6]
        return local, spatial                         # local XYZ + normal


class FacePlacementHead(nn.Module):
    """原 center/scale 头：7 个残差 Linear block + Linear(768,4)。"""

    def __init__(self, latent_dim: int, width: int):
        super().__init__()
        self.input = ResidualLinear(latent_dim, width)
        self.blocks = nn.Sequential(*[ResidualLinear(width, width) for _ in range(6)])
        self.output = nn.Linear(width, 4)

    def forward(self, faces: Tensor) -> Tensor:
        return self.output(self.blocks(self.input(faces)))  # [N,64] -> [N,4]


class FaceDecoder(nn.Module):
    def __init__(self, cfg: UVDecoderConfig):
        super().__init__()
        self.points = FacePointDecoder(cfg.latent_dim, cfg.face_width)
        self.placement = FacePlacementHead(cfg.latent_dim, cfg.face_width)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        local, spatial = self.points(features)
        placement = self.placement(features)  # [N, center_xyz + scalar_scale]
        with torch.autocast(device_type=features.device.type, enabled=False):
            center = placement[:, None, None, :3].float()
            scale = placement[:, None, None, 3:].float()  # 保留原始标量，无 softplus
            xyz = local[..., :3].float() * scale + center
        return {"local": local, "spatial": spatial, "placement": placement, "xyz": xyz}


class UnorderedFacePairs(nn.Module):
    """全部物体内 i<j 候选；不读取 GT adjacency，不跨物体配对。"""

    def forward(self, object_ids: Tensor) -> Tensor:
        pairs = torch.triu_indices(len(object_ids), len(object_ids), 1, device=object_ids.device).T
        same_object = object_ids[pairs[:, 0]] == object_ids[pairs[:, 1]]
        return pairs[same_object]  # [P,2], P=sum_b N_b(N_b-1)/2


class FaceSpatialTokens(nn.Module):
    def __init__(self, channels: int, width: int):
        super().__init__()
        self.geometry = nn.Sequential(nn.Linear(channels + 3, width), nn.LayerNorm(width))
        self.position = nn.Sequential(nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, width))
        v, u = torch.meshgrid(torch.linspace(0, 1, 16), torch.linspace(0, 1, 16), indexing="ij")
        self.register_buffer("grid_uv", torch.stack((u, v), -1).reshape(256, 2))

    def forward(self, spatial: Tensor, xyz: Tensor) -> Tensor:
        cnn = spatial.flatten(2).transpose(1, 2)           # [N,256,96]
        points = xyz.flatten(1, 2).to(cnn.dtype)           # [N,256,3]
        tokens = self.geometry(torch.cat((cnn, points), -1))  # [N,256,d]
        return tokens + self.position(self.grid_uv.to(cnn.dtype))[None]


class PairMemory(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.side_embedding = nn.Parameter(torch.empty(2, 1, width))
        nn.init.normal_(self.side_embedding, std=0.02)

    def forward(self, face_tokens: Tensor, owners: Tensor) -> Tensor:
        memory = face_tokens[owners]                      # [P,2,256,d]
        memory = memory + self.side_embedding.to(memory.dtype)[None]
        return memory.flatten(1, 2)                       # [P,512,d], token维拼接


class CurveQueries(nn.Module):
    """每个无序面对只生成一组沿 t 展开的 query，不生成 h_ij/h_ji。"""

    def __init__(self, latent_dim: int, width: int):
        super().__init__()
        self.seed = nn.Sequential(nn.Linear(2 * latent_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.position = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, width))
        self.register_buffer("sample_t", torch.linspace(0, 1, 16).reshape(16, 1))

    def forward(self, faces: Tensor, owners: Tensor) -> Tensor:
        first, second = faces[owners[:, 0]], faces[owners[:, 1]]
        pair = torch.cat((first + second, (first - second).abs()), -1)  # [P,128]
        seed = self.seed(pair)                           # [P,d]
        position = self.position(self.sample_t.to(seed.dtype))
        return seed[:, None, :] + position[None]         # [P,16,d]


class CurveAttentionBlock(nn.Module):
    """展开 pre-LN TransformerDecoderLayer，直接显示 Q/K/V 与残差。"""

    def __init__(self, width: int, heads: int, ffn: int, dropout: float):
        super().__init__()
        self.norm_self = nn.LayerNorm(width)
        self.norm_cross = nn.LayerNorm(width)
        self.norm_ffn = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(width, ffn), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ffn, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, memory: Tensor) -> Tensor:
        q = self.norm_self(query)                         # [P,16,d]
        update, _ = self.self_attention(q, q, q, need_weights=False)
        query = query + self.dropout(update)              # 曲线16位置交流
        q = self.norm_cross(query)
        update, _ = self.cross_attention(q, memory, memory, need_weights=False)
        query = query + self.dropout(update)              # Q读两面512个token
        return query + self.dropout(self.ffn(self.norm_ffn(query)))


class CurveAttentionStack(nn.Module):
    """edge query 后接的主干：默认串联 8 个 CurveAttentionBlock。"""

    def __init__(self, cfg: UVDecoderConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            CurveAttentionBlock(cfg.curve_width, cfg.curve_heads, cfg.curve_ffn, cfg.dropout)
            for _ in range(cfg.curve_layers)
        ])

    def forward(self, edge_query: Tensor, face_memory: Tensor) -> Tensor:
        for block in self.blocks:
            edge_query = block(edge_query, face_memory)  # [p,16,512]，每层读取同一个 memory
        return edge_query                               # 更新后的曲线特征 [p,16,512]


class DualUVHead(nn.Module):
    """双侧 UV 输出头：一个 Linear(width,4) 同时输出 (u_i,v_i,u_j,v_j)。"""

    def __init__(self, width: int):
        super().__init__()
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4))

    def forward(self, curve_tokens: Tensor) -> Tensor:
        with torch.autocast(device_type=curve_tokens.device.type, enabled=False):
            values = self.output(curve_tokens.float())           # LN + Linear: [p,16,512] -> [p,16,4]
            uv = values.sigmoid()                               # FP32，四个坐标约束到 (0,1)
        uv = uv.reshape(-1, 16, 2, 2)                            # [p,位置,面侧,(u,v)]
        return uv.permute(0, 2, 1, 3)                            # [p,2,16,2]


class EdgeExistenceHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, curve_tokens: Tensor) -> Tensor:
        return self.output(curve_tokens.mean(dim=1)).squeeze(-1)  # [P], logits


class UVIntersection(nn.Module):
    """两面 memory + 一组 edge query -> attention -> UV / 存在性两个输出头。"""

    def __init__(self, cfg: UVDecoderConfig):
        super().__init__()
        self.pair_memory = PairMemory(cfg.curve_width)
        self.edge_queries = CurveQueries(cfg.latent_dim, cfg.curve_width)
        self.curve_attention = CurveAttentionStack(cfg)
        self.uv_head = DualUVHead(cfg.curve_width)
        self.existence_head = EdgeExistenceHead(cfg.curve_width)

    def forward(self, faces: Tensor, face_tokens: Tensor, owners: Tensor) -> tuple[Tensor, Tensor]:
        memory = self.pair_memory(face_tokens, owners)               # 两面 memory [p,512,512]
        edge_query = self.edge_queries(faces, owners)                # 16 个位置 query [p,16,512]
        edge_features = self.curve_attention(edge_query, memory)    # 8 层 attention -> [p,16,512]
        uv = self.uv_head(edge_features)                             # 双侧 UV [p,2,16,2]
        logits = self.existence_head(edge_features)                  # 存在性 logits [p]
        return uv, logits


class BSplineBasis(nn.Module):
    """Clamped open-uniform basis; UV must lie in [0, 1]."""

    def __init__(self, num_controls: int = 16, degree: int = 3):
        super().__init__()
        if not 0 <= degree < num_controls:
            raise ValueError("Require 0 <= degree < num_controls.")
        self.degree = degree
        breaks = torch.linspace(0, 1, num_controls - degree + 1, dtype=torch.float64)
        knots = torch.cat((breaks[:1].repeat(degree), breaks, breaks[-1:].repeat(degree)))
        self.register_buffer("knots", knots)

    def forward(self, parameter: Tensor) -> Tensor:
        """[...] parameters -> [..., num_controls] basis values, at least FP32."""
        dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=parameter.device.type, enabled=False):
            x = parameter.to(dtype).unsqueeze(-1)
            knots = self.knots.to(dtype=dtype, device=x.device)
            left, right = knots[:-1], knots[1:]
            # Use the last NONEMPTY span at x=1. Recurrence then retains the
            # actual left endpoint derivative; overwriting final values would not.
            last_span = (x == 1) & (right == 1) & (left < right)
            basis = (((x >= left) & (x < right)) | last_span).to(dtype)
            for degree in range(1, self.degree + 1):
                width = knots.numel() - degree - 1
                left_width = knots[degree:degree + width] - knots[:width]
                right_width = knots[degree + 1:] - knots[1:width + 1]
                left_den = torch.where(left_width > 0, left_width, torch.ones_like(left_width))
                right_den = torch.where(right_width > 0, right_width, torch.ones_like(right_width))
                left_weight = (x - knots[:width]) / left_den * (left_width > 0)
                right_weight = (knots[degree + 1:] - x) / right_den * (right_width > 0)
                basis = left_weight * basis[..., :-1] + right_weight * basis[..., 1:]
            return basis


class FixedBSplineSurface(nn.Module):
    """Interpolate the sample grid once; evaluate only requested UV positions.

    Fixed degree/knots and uniform sample UVs are a new explicit representation
    contract. This does not reproduce the adaptive fitting used by legacy OCC.
    It provides no periodic seam continuity and may overshoot between samples.
    """

    def __init__(self, grid_size: int = 16, degree: int = 3):
        super().__init__()
        self.basis = BSplineBasis(grid_size, degree)
        sample_uv = torch.linspace(0, 1, grid_size, dtype=torch.float64)
        sample_basis = self.basis(sample_uv)  # [sample_index, control_index]
        sample_to_control = torch.linalg.solve(sample_basis, torch.eye(grid_size, dtype=torch.float64))
        self.register_buffer("sample_to_control", sample_to_control)

    def fit(self, samples: Tensor) -> Tensor:
        """[F, V=16, U=16, 3] samples -> [F, B=16, A=16, 3] poles."""
        dtype = torch.float64 if samples.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=samples.device.type, enabled=False):
            samples = samples.to(dtype)
            inverse = self.sample_to_control.to(dtype=dtype, device=samples.device)
            # P = B_v^{-1} X (B_u^{-1})^T; fit each face once, not once per edge.
            return torch.einsum("bi,fijc,aj->fbac", inverse, samples, inverse)

    def evaluate(self, poles: Tensor, uv: Tensor) -> Tensor:
        """[..., B(v), A(u), 3] poles + [..., T, 2] UV -> [..., T, 3] XYZ.

        For edges, leading dimensions are [E, 2 owner faces].
        """
        dtype = torch.float64 if poles.dtype == torch.float64 or uv.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=poles.device.type, enabled=False):
            uv = uv.to(dtype)
            basis_u = self.basis(uv[..., 0])  # [..., T, A]
            basis_v = self.basis(uv[..., 1])  # [..., T, B]
            return torch.einsum("...ta,...bac,...tb->...tc", basis_u, poles.to(dtype), basis_v)

    def forward(self, samples: Tensor) -> Tensor:
        """Keep module forward explicit: sampled surface -> continuous poles."""
        return self.fit(samples)


class FaceReconstructionLoss(nn.Module):
    def forward(self, prediction: dict[str, Tensor], target: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            name: F.l1_loss(prediction[name].float(), target[name].float())
            for name in ("face_local", "face_center_scale", "face_xyz")
        }


class EdgeSupervision(nn.Module):
    """将GT边关联到所有候选；同一面对多重边/self边不在此表示范围。"""

    def forward(self, candidates: Tensor, target_pairs: Tensor, num_faces: int) -> tuple[Tensor, Tensor]:
        invalid_owner = (target_pairs < 0) | (target_pairs >= num_faces)
        if invalid_owner.any() or (target_pairs[:, 0] >= target_pairs[:, 1]).any():
            raise ValueError("GT owners must satisfy 0 <= i < j < num_faces.")
        lookup = candidates.new_full((num_faces, num_faces), -1)
        lookup[candidates[:, 0], candidates[:, 1]] = torch.arange(len(candidates), device=candidates.device)
        rows = lookup[target_pairs[:, 0], target_pairs[:, 1]]
        if (rows < 0).any() or rows.unique().numel() != rows.numel():
            raise ValueError("GT pairs must be unique supported i<j candidates within each object.")
        positive = torch.zeros(len(candidates), dtype=torch.bool, device=candidates.device)
        positive[rows] = True
        return positive, rows


class BalancedExistenceLoss(nn.Module):
    def forward(self, logits: Tensor, positive: Tensor) -> Tensor:
        values = F.binary_cross_entropy_with_logits(logits.float(), positive.float(), reduction="none")
        groups = [values[mask].mean() for mask in (positive, ~positive) if mask.any()]
        return torch.stack(groups).mean() if groups else logits.sum() * 0


class CurveGeometryLoss(nn.Module):
    """每侧都对GT监督；只允许两侧一起反向，防止各自选不同方向。"""

    def forward(self, predicted: Tensor, target: Tensor) -> dict[str, Tensor]:
        # predicted [R,2,16,3], target [R,16,3]；GT仅包含真实边。
        if not len(target):
            zero = predicted.sum() * 0
            return {"curve_xyz": zero, "curve_incidence": zero}
        predicted, target = predicted.float(), target.float()
        forward_error = (predicted - target[:, None]).square().mean((1, 2, 3))
        reverse_error = (predicted - target.flip(1)[:, None]).square().mean((1, 2, 3))
        return {
            "curve_xyz": torch.minimum(forward_error, reverse_error).mean(),
            "curve_incidence": F.mse_loss(predicted[:, 0], predicted[:, 1]),
        }


class UVDecoderLoss(nn.Module):
    """显式审阅目标；默认权重不是已经选定的正式训练协议。

    target: face_local[N,16,16,6], face_center_scale[N,4], face_xyz[N,16,16,3],
            edge_pairs[R,2] (i<j), edge_xyz[R,16,3] (共同点序)。
    旧UV标签不直接沿用；新曲面参数域不能假装等于旧OCC参数域。
    """

    def __init__(self, classification_weight: float = 0.1, incidence_weight: float = 0.1):
        super().__init__()
        self.face = FaceReconstructionLoss()
        self.match = EdgeSupervision()
        self.existence = BalancedExistenceLoss()
        self.geometry = CurveGeometryLoss()
        self.classification_weight = classification_weight
        self.incidence_weight = incidence_weight

    def forward(self, prediction: dict[str, Tensor], target: dict[str, Tensor]) -> dict[str, Tensor]:
        positive, rows = self.match(prediction["edge_pairs"], target["edge_pairs"], len(prediction["face_xyz"]))
        losses = self.face(prediction, target)
        losses["edge_existence"] = self.existence(prediction["edge_logits"], positive)
        losses.update(self.geometry(prediction["edge_xyz"][rows], target["edge_xyz"]))
        losses["total"] = (
            losses["face_local"] + losses["face_center_scale"] + losses["face_xyz"]
            + self.classification_weight * losses["edge_existence"]
            + losses["curve_xyz"] + self.incidence_weight * losses["curve_incidence"]
        )
        return losses
