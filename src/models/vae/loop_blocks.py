"""Ordered-loop encoding, owner-local AR events and reciprocal mate matching."""
import math

import networkx as nx
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_mean

from src.preprocess.loop_sequences import EDGE, CLOSE, END, BOS, PAD
from .pcurve_relations import feature_body, inverse_next


def position_encoding(length, width, device, dtype):
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    scale = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32) * (-math.log(10000) / width))
    values = position * scale
    return torch.stack([values.sin(), values.cos()], -1).flatten(1).to(dtype)


class OrderedLoopEncoder(nn.Module):
    def __init__(self, width=512, depth=4, loop_layers=4):
        super().__init__()
        self.uv = nn.Sequential(nn.Linear(32, width), nn.GELU(), nn.Linear(width, 64))
        self.relation = feature_body(320, width, depth)
        layer = nn.TransformerEncoderLayer(width, 8, 4 * width, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.loops = nn.TransformerEncoder(layer, loop_layers, enable_nested_tensor=False)
        self.length = nn.Linear(1, width)
        self.to_face = nn.Sequential(nn.Linear(width + 1, width), nn.GELU(), nn.Linear(width, 32))

    def forward(self, face, edge, data):
        owner, mate, nxt = data['owner'], data['mate'], data['next']
        previous = inverse_next(nxt)
        g = edge + self.uv(data['pcurve_uv'].flatten(1))
        message = self.relation(torch.cat([g, g[previous], g[nxt], g[mate],
                                          face[owner[mate[previous]]], face[owner[mate[nxt]]]], -1))
        ids = data['loop_edge_ids']
        valid = ids >= 0
        tokens = message[ids.clamp_min(0)]
        tokens = tokens + position_encoding(ids.shape[1], tokens.shape[-1], tokens.device, tokens.dtype)
        tokens = self.loops(tokens, src_key_padding_mask=~valid)
        lengths = valid.sum(1, keepdim=True)
        loop = (tokens * valid[..., None]).sum(1) / lengths
        loop = loop + self.length(lengths.float().log1p())
        pooled = scatter_mean(loop, data['loop_owner'], dim=0, dim_size=len(face))
        counts = torch.bincount(data['loop_owner'], minlength=len(face)).float().log1p()[:, None]
        return face + self.to_face(torch.cat([pooled, counts], -1))


def event_grammar(prefix):
    """Allowed next events, including safe values at ignored padded positions."""
    edge_count = (prefix == EDGE).cumsum(1)
    at_close = torch.where(prefix == CLOSE, edge_count, 0).cummax(1).values
    open_loop = edge_count > at_close
    closed = (prefix == CLOSE).cumsum(1) > 0
    ended = (prefix == END).cumsum(1) > 0
    return torch.stack([~ended, open_loop & ~ended, ((~open_loop) & closed) | ended], -1)


class OwnerBoundaryAR(nn.Module):
    """The published Intersection transformer, now reading a full owner chart.

    Prefix inputs are event types only. Geometry targets are never fed back as
    teacher inputs. EDGE-position states become directed coedge features.
    """
    def __init__(self, intersection, channels, max_events, face_chunk_size=128):
        super().__init__()
        self.face_projection = intersection.attn_proj_in
        self.layers = intersection.layers
        self.readout = intersection.attn_proj_out
        self.roles = intersection.pos_encoding
        width = self.face_projection.out_features
        self.events = nn.Embedding(5, width)
        nn.init.normal_(self.events.weight, std=0.02)
        self.memory = nn.Sequential(nn.Linear(channels + 5, width), nn.LayerNorm(width))
        self.event_head = nn.Linear(width, 3)
        self.max_events = max_events
        self.face_chunk_size = face_chunk_size

    def surface_memory(self, phi, grid):
        v, u = torch.meshgrid(torch.linspace(0, 1, grid.shape[1], device=grid.device),
                              torch.linspace(0, 1, grid.shape[2], device=grid.device), indexing='ij')
        uv = torch.stack([u, v], -1).expand(len(grid), -1, -1, -1)
        return self.memory(torch.cat([phi.permute(0, 2, 3, 1), grid, uv], -1).flatten(1, 2))

    def _decode(self, prefix, face, memory):
        conditioning = self.face_projection(face)
        x = self.events(prefix) + conditioning[:, None] + self.roles[:, 0:1]
        x = x + position_encoding(prefix.shape[1], x.shape[-1], x.device, x.dtype)
        memory = memory + conditioning[:, None] + self.roles[:, 1:2]
        causal = torch.ones(prefix.shape[1], prefix.shape[1], device=x.device, dtype=torch.bool).triu(1)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=causal, tgt_key_padding_mask=prefix == PAD)
        logits = self.event_head(x).float().masked_fill(~event_grammar(prefix), -torch.inf)
        return self.readout(x), logits

    def forward(self, prefix, face, memory):
        features, logits = [], []
        for p, f, m in zip(prefix.split(self.face_chunk_size), face.split(self.face_chunk_size),
                           memory.split(self.face_chunk_size)):
            if self.training and torch.is_grad_enabled():
                h, score = checkpoint(self._decode, p, f, m, use_reentrant=False)
            else:
                h, score = self._decode(p, f, m)
            features.append(h)
            logits.append(score)
        return torch.cat(features), torch.cat(logits)

    def teacher_forcing(self, targets, face, memory):
        previous = targets[:, :-1]
        prefix = torch.cat([torch.full_like(targets[:, :1], BOS),
                            torch.where(previous >= 0, previous, PAD)], 1)
        return self(prefix, face, memory)

    @torch.no_grad()
    def generate(self, face, memory):
        prefix = torch.full((len(face), 1), BOS, dtype=torch.long, device=face.device)
        finished = torch.zeros(len(face), dtype=torch.bool, device=face.device)
        events, states, confidence = [], [], []
        for _ in range(self.max_events):
            h, logits = self(prefix, face, memory)
            probability = logits[:, -1].softmax(-1)
            token = probability.argmax(-1)
            events.append(torch.where(finished, PAD, token))
            states.append(h[:, -1])
            confidence.append(probability[:, EDGE])
            finished = finished | (token == END)
            if finished.all():
                break
            prefix = torch.cat([prefix, token[:, None]], 1)
        return torch.stack(events, 1), torch.stack(states, 1), torch.stack(confidence, 1)


class MateDecoder(nn.Module):
    def __init__(self, width=512, depth=4):
        super().__init__()
        self.body = feature_body(96, width, depth)
        self.left = nn.Linear(width, width)
        self.right = nn.Linear(width, width)
        self.null = nn.Linear(width, 1)

    def forward(self, h, face, owner, objects):
        b = self.body(torch.cat([h, face[owner]], -1))
        left, right = self.left(b), self.right(b)
        with torch.autocast(h.device.type, enabled=False):
            r = left.float() @ right.float().T / math.sqrt(b.shape[-1])
            score = 0.5 * (r + r.T)
            forbidden = (objects[:, None] != objects[None, :]) | torch.eye(len(h), device=h.device, dtype=torch.bool)
            score = score.masked_fill(forbidden, -torch.inf)
        return score, self.null(b).squeeze(-1).float()


def mate_objective(scores, null, target, drop_probability, training):
    clean = F.cross_entropy(torch.cat([scores, null[:, None]], -1), target)
    if not training:
        return clean
    kept = torch.where(torch.rand(len(target), device=target.device) >= drop_probability)[0]
    if len(kept) == 0:
        kept = torch.arange(1, device=target.device)
    remap = torch.full_like(target, len(kept))
    remap[kept] = torch.arange(len(kept), device=target.device)
    corrupted = F.cross_entropy(torch.cat([scores[kept][:, kept], null[kept, None]], -1), remap[target[kept]])
    return 0.5 * (clean + corrupted)


@torch.no_grad()
def match_mates(scores, null):
    """Maximum-weight general-graph matching with an unmatched alternative.

    Relative to leaving both uses unmatched, pair(h,k) gains
    2*score[h,k] - null[h] - null[k]. No positive gain means no forced pair.
    """
    gain = (2 * scores - null[:, None] - null[None, :]).double().cpu().numpy()
    graph = nx.Graph()
    graph.add_nodes_from(range(len(null)))
    for h in range(len(null)):
        for k in range(h + 1, len(null)):
            if gain[h, k] > 0:
                graph.add_edge(h, k, weight=float(gain[h, k]))
    result = torch.full((len(null),), -1, dtype=torch.long, device=scores.device)
    for h, k in nx.max_weight_matching(graph, maxcardinality=False):
        result[h], result[k] = k, h
    return result
