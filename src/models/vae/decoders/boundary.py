"""Face32 boundary decoder: pair MLP -> geometry -> owner/endpoint exchange.

No spatial Intersection, face writeback, conditioning, shared-XYZ fusion or UV
head. Endpoint connection scores are computed once and read in both directions.
"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..state import DecoderStage, PrimitiveGeometry, RefinementOutput
from .intersection import enumerate_candidate_pairs
from .primitives import FaceFeatureDecoder, PrimitiveDecoder
from .refinement import GeometryRead


@dataclass
class EndpointOutput:
    vertices: Tensor                 # [P,2,3], object coordinates
    scores: Tensor                   # [N,H,L,L], end(row) -> start(column)
    candidate_rows: Tensor           # [N,L], -1 for padding


@dataclass
class BoundaryOutput(RefinementOutput):
    endpoints: EndpointOutput | None = None


def owner_layout(owners, face_count):
    """Map packed directed candidates to one padded sequence per owner face."""
    counts = torch.bincount(owners, minlength=face_count)
    length = int(counts.max()) if face_count else 0
    order = owners.argsort(stable=True)
    starts = counts.cumsum(0) - counts
    slots = torch.empty_like(order)
    slots[order] = torch.arange(len(owners), device=owners.device) - starts[owners[order]]
    rows = owners.new_full((face_count, length), -1)
    rows[owners, slots] = torch.arange(len(owners), device=owners.device)
    return rows, slots


def source_weights(scores, confidence):
    """Confidence gates source keys; a fixed null key has logit/value zero."""
    log_p = confidence.detach().float().clamp_min(torch.finfo(torch.float32).tiny).log()
    log_p = log_p.masked_fill(confidence == 0, -torch.inf)
    scores = scores.float() + log_p[:, None, None, :]
    scores = torch.cat((scores, scores.new_zeros((*scores.shape[:-1], 1))), -1)
    return scores.softmax(-1)[..., :-1]


class OwnerAttention(nn.Module):
    """One 128-dimensional, four-head attention over same-owner messages."""

    def __init__(self, width=128, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, messages, probabilities, owners, face_count):
        rows, slots = owner_layout(owners, face_count)
        n, length = rows.shape
        if not len(messages):
            return messages, rows, slots
        padded = messages.new_zeros((n, length, messages.shape[-1]))
        padded[owners, slots] = messages
        confidence = probabilities.new_zeros((n, length))
        confidence[owners, slots] = probabilities.detach()
        q, k, v = self.qkv(self.norm(padded)).reshape(
            n, length, 3, self.heads, -1,
        ).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q.float() @ k.float().transpose(-1, -2)) / q.shape[-1] ** 0.5
        weights = source_weights(scores, confidence).to(v.dtype)
        context = (weights @ v).transpose(1, 2).reshape_as(padded)
        return self.output(context)[owners, slots], rows, slots


class EndpointExchange(nn.Module):
    """One shared end/start score table; distinct start/end value projections."""

    def __init__(self, width=128, heads=4):
        super().__init__()
        self.width, self.heads = width, heads
        self.endpoint = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2 * width))
        self.start_key = nn.Linear(width, width)
        self.end_key = nn.Linear(width, width)
        self.start_value = nn.Linear(width, width)
        self.end_value = nn.Linear(width, width)
        self.start_output = nn.Linear(width, width, bias=False)
        self.end_output = nn.Linear(width, width, bias=False)
        self.update = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU())
        self.vertex = nn.Sequential(nn.Linear(width, width // 2), nn.SiLU(), nn.Linear(width // 2, 3))

    def forward(self, boundary, probabilities, owners, rows, slots):
        q = self.endpoint(boundary).reshape(-1, 2, self.width)  # [P,2,128]
        n, length = rows.shape
        padded = q.new_zeros((n, length, 2, self.width))
        padded[owners, slots] = q
        confidence = probabilities.new_zeros((n, length))
        confidence[owners, slots] = probabilities.detach()
        start, end = padded.unbind(2)

        def split(x):
            return x.reshape(n, length, self.heads, self.width // self.heads).transpose(1, 2)

        start_key, end_key = split(self.start_key(start)), split(self.end_key(end))
        scores = end_key.float() @ start_key.float().transpose(-1, -2)
        scores = scores / (self.width // self.heads) ** 0.5  # [N,4,L,L]
        next_weights = source_weights(scores, confidence)
        prev_weights = source_weights(scores.transpose(-1, -2), confidence)
        start_v, end_v = split(self.start_value(start)), split(self.end_value(end))
        next_message = (next_weights.to(start_v.dtype) @ start_v).transpose(1, 2).reshape_as(start)
        prev_message = (prev_weights.to(end_v.dtype) @ end_v).transpose(1, 2).reshape_as(end)
        context = torch.stack((self.start_output(prev_message), self.end_output(next_message)), 2)
        updated = self.update(torch.cat((q, context[owners, slots]), -1))
        with torch.autocast(boundary.device.type, enabled=False):
            vertices = self.vertex(updated.float())
        return updated, EndpointOutput(vertices, scores, rows)


class BoundaryExchange(nn.Module):
    """Common geometry/owner trunk, optionally followed by endpoint exchange."""

    def __init__(self, edge_dim, cfg):
        super().__init__()
        width, geometry = int(cfg['width']), int(cfg['geometry_dim'])
        self.geometry = GeometryRead(geometry, width, int(cfg['geometry_chunk_size']))
        self.message = nn.Sequential(nn.Linear(edge_dim + geometry, width), nn.LayerNorm(width), nn.SiLU())
        self.owner_attention = OwnerAttention(width, int(cfg['heads']))
        self.boundary = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU())
        # Construct common parameters identically in both variants.
        self.readout = nn.Linear(width, edge_dim)
        self.endpoints = None
        self.endpoint_readout = None
        if cfg['endpoints']:
            with torch.random.fork_rng(devices=[]):
                self.endpoints = EndpointExchange(width, int(cfg['heads']))
                self.endpoint_readout = nn.Linear(3 * width, edge_dim)
                # Same initial h1 as the basic variant, without a delta path.
                with torch.no_grad():
                    self.endpoint_readout.weight.zero_()
                    self.endpoint_readout.weight[:, :width].copy_(self.readout.weight)
                    self.endpoint_readout.bias.copy_(self.readout.bias)
            self.readout = None

    def forward(self, coarse, pairs):
        edges, faces = coarse.edge_features, coarse.face_features
        r = self.geometry(coarse.edge_geometry.points, coarse.face_geometry.points, pairs.indices)
        m = self.message(torch.cat((edges, r), -1))
        probabilities = coarse.edge_logits.float().sigmoid().detach()
        owners = pairs.indices[:, 0]
        a, rows, slots = self.owner_attention(m, probabilities, owners, len(faces))
        b = self.boundary(torch.cat((m, a), -1))  # [P,128]
        if self.endpoints is None:
            return self.readout(b), None
        endpoint_states, output = self.endpoints(b, probabilities, owners, rows, slots)
        features = self.endpoint_readout(torch.cat((b, endpoint_states.flatten(1)), -1))
        return features, output


class BoundaryDecoder(nn.Module):
    """Original face32 geometry; independent Teacher; shared Coarse/Final head."""

    def __init__(self, legacy, cfg):
        super().__init__()
        self.face_features = FaceFeatureDecoder(legacy.face_attn_proj_in2, legacy.face_attn2, legacy.face_attn_proj_out2)
        self.face_geometry = PrimitiveDecoder(legacy.face_points_decoder, legacy.face_center_scale_decoder,
                                              (16, 16), legacy.in_channels, force_fp32=True)
        self.edge_geometry = PrimitiveDecoder(legacy.edge_points_decoder, legacy.edge_center_scale_decoder,
                                              (16,), legacy.in_channels, force_fp32=True)
        self.teacher_geometry = self.edge_geometry.unconditioned_copy()
        self.classifier = legacy.classifier
        width, edge_dim = int(cfg['width']), legacy.edge_feature_dim
        self.pair = nn.Sequential(nn.Linear(2 * legacy.face_feature_dim, width), nn.SiLU(), nn.Linear(width, edge_dim))
        self.exchange = BoundaryExchange(edge_dim, cfg)
        self.chunk_size = int(cfg['candidate_chunk_size'])

    def decode_curves(self, features):
        """Checkpoint candidate CNN activations; do not retain all P grids."""
        if not len(features):
            return self.edge_geometry(features)

        def decode(x):
            geometry = self.edge_geometry(x)
            return geometry.local_points, geometry.center_scale, geometry.points

        chunks = [checkpoint(decode, x, use_reentrant=False) if self.training else decode(x)
                  for x in features.split(self.chunk_size)]
        return PrimitiveGeometry(*(torch.cat(parts) for parts in zip(*chunks)))

    def decode_teacher(self, features):
        return self.teacher_geometry(features)

    def forward(self, latents, face_counts):
        pairs = enumerate_candidate_pairs(face_counts.to(latents.device))
        faces = self.face_features(latents, face_counts)               # [N,32]
        surface = self.face_geometry(faces)                           # [N,16,16,3]
        h0 = self.pair(faces[pairs.indices].flatten(1))                 # [P,64]
        coarse = DecoderStage(faces, h0, surface, self.decode_curves(h0), self.classifier(h0).squeeze(-1))
        if len(h0):
            h1, endpoints = self.exchange(coarse, pairs)
        else:
            h1, endpoints = h0, None
        final = DecoderStage(faces, h1, surface, self.decode_curves(h1), self.classifier(h1).squeeze(-1))
        return BoundaryOutput(pairs, coarse, final, torch.zeros_like(faces), h1 - h0, endpoints=endpoints)
