"""Native relation encoding, discrete next pointers and shared UV corners."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


class FeatureBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, x):
        return x + self.layers(x)


def feature_body(input_dim, width, depth):
    return nn.Sequential(nn.Linear(input_dim, width), *[FeatureBlock(width) for _ in range(depth)])


def inverse_next(nxt):
    previous = torch.empty_like(nxt)
    previous[nxt] = torch.arange(len(nxt), device=nxt.device)
    return previous


class NativeRelationEncoder(nn.Module):
    def __init__(self, width=512, depth=4):
        super().__init__()
        self.uv = nn.Sequential(nn.Linear(32, width), nn.GELU(), nn.Linear(width, 64))
        self.relation = nn.Sequential(feature_body(320, width, depth), nn.Linear(width, 64))
        self.to_face = nn.Linear(64, 32)

    def forward(self, face, edge, uv, owner, mate, nxt):
        previous = inverse_next(nxt)
        g = edge + self.uv(uv.flatten(1))
        message = self.relation(
            torch.cat([g, g[previous], g[nxt], g[mate], face[owner[mate[previous]]], face[owner[mate[nxt]]]], -1)
        )
        total = message.new_zeros(len(face), message.shape[-1])
        total.index_add_(0, owner, message)
        mean = total / torch.bincount(owner, minlength=len(face)).clamp_min(1)[:, None]
        return face + self.to_face(mean)


class NextDecoder(nn.Module):
    def __init__(self, width=512, depth=4):
        super().__init__()
        self.body = feature_body(128, width, depth)
        self.start = nn.Linear(width, width)
        self.end = nn.Linear(width, width)
        self.start_null = nn.Linear(width, 1)
        self.end_null = nn.Linear(width, 1)

    def forward(self, h, face, owner, mate):
        b = self.body(torch.cat([h, face[owner], face[owner[mate]]], -1))
        end, start = self.end(b), self.start(b)
        with torch.autocast(h.device.type, enabled=False):
            scores = end.float() @ start.float().T / math.sqrt(b.shape[-1])
            scores = scores.masked_fill(owner[:, None] != owner[None, :], -torch.inf)
        return scores, self.end_null(b).squeeze(-1).float(), self.start_null(b).squeeze(-1).float()


def relation_loss(scores, end_null, start_null, nxt, previous):
    null = len(nxt)
    rows = torch.cat([scores, end_null[:, None]], 1)
    cols = torch.cat([scores.T, start_null[:, None]], 1)
    return 0.5 * (
        F.cross_entropy(rows, torch.where(nxt >= 0, nxt, null))
        + F.cross_entropy(cols, torch.where(previous >= 0, previous, null))
    )


@torch.no_grad()
def assign_next(scores, end_null, start_null, owner):
    nxt = torch.full_like(owner, -1)
    for face in owner.unique():
        ids = torch.where(owner == face)[0]
        n = len(ids)
        weight = scores.new_full((2 * n, 2 * n), -torch.inf)
        weight[:n, :n] = 2 * scores[ids][:, ids]
        diagonal = torch.arange(n, device=ids.device)
        weight[diagonal, n + diagonal] = end_null[ids]
        weight[n + diagonal, diagonal] = start_null[ids]
        weight[n:, n:] = 0
        rows, cols = linear_sum_assignment(-weight.double().cpu().numpy())
        real = (rows < n) & (cols < n)
        nxt[ids[torch.as_tensor(rows[real], device=ids.device)]] = ids[torch.as_tensor(cols[real], device=ids.device)]
    return nxt


class PcurveDecoder(nn.Module):
    def __init__(self, width=512, depth=4, controls=16):
        super().__init__()
        self.controls = controls
        self.body = feature_body(320, width, depth)
        self.corner = nn.Linear(width, 2)
        self.interior = nn.Linear(width, 2 * (controls - 2))

    def forward(self, h, face, owner, mate, nxt):
        if (nxt < 0).any():
            raise ValueError("Incomplete next: a null index is not a UV corner")
        previous = inverse_next(nxt)
        b = self.body(torch.cat([h, h[previous], h[nxt], h[mate], face[owner], face[owner[mate]]], -1))
        # Nonnegative partition-of-unity bases keep the whole pcurve in its
        # normalized UV rectangle when all poles lie in that rectangle.
        with torch.autocast(b.device.type, enabled=False):
            q = self.corner(b.float()).sigmoid()
            internal = self.interior(b.float()).sigmoid().reshape(len(h), self.controls - 2, 2)
        return torch.cat([q[:, None], internal, q[nxt, None]], 1)
