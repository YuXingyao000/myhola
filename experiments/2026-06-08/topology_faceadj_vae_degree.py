#!/usr/bin/env python3
"""Topology face-adj VAE with per-face degree tokens.

Variant of `experiments/2026-06-06/topology_faceadj_vae_edgecount.py`.

Sequence layout (length = 2 + max_faces + max_pairs):
    [N_FACE] [N_EDGE] [DEG_0] [DEG_1] ... [DEG_{max_faces-1}]
        [PAIR_0] [PAIR_1] ... [PAIR_{max_pairs-1}]

Where DEG_i = number of neighbors of face i in the canonical-ordered
adjacency matrix (i.e. face_adj.sum(axis=1) after canonical reorder).

For padding face slots (i >= num_faces), the degree token is set to 0 with
its mask flagged so the decoder only learns to predict valid degrees.

Training adds a degree CE loss with weight `degree-loss-weight`.

At generation time, the decoder uses degree-budget constrained decoding:
    - Each face has remaining_deg[i] tokens left to fulfill.
    - When deciding pair (i, j), if remaining_deg[i] == 0 or
      remaining_deg[j] == 0, force NO_EDGE.
    - If remaining_deg[i] >= remaining_active_pairs_with_i, force EDGE.
    - Otherwise sample / argmax from logits.
This is a hard constraint; it does not back-propagate. The global edge-count
budget is also kept as a fallback constraint (same as the 06-06 baseline).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import zipfile
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


DEFAULT_RAW_ROOT = "/mnt/d/data/deepcad_v7"
DEFAULT_TRAIN_LIST = "src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt"
DEFAULT_VAL_LIST = "src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt"
DEFAULT_TEST_LIST = "src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt"
DATA_READ_ERRORS = (OSError, EOFError, KeyError, zipfile.BadZipFile, zlib.error)

PAD = 0
NO_EDGE = 1
EDGE = 2
FACE_COUNT_OFFSET = 3


def face_count_token(num_faces): return FACE_COUNT_OFFSET + int(num_faces)
def token_to_face_count(token):
    if torch.is_tensor(token):
        return token - FACE_COUNT_OFFSET
    return int(token) - FACE_COUNT_OFFSET


def edge_count_offset(max_faces): return FACE_COUNT_OFFSET + max_faces + 1
def edge_count_token(edge_count, max_faces): return edge_count_offset(max_faces) + int(edge_count)
def token_to_edge_count(token, max_faces):
    if torch.is_tensor(token):
        return token - edge_count_offset(max_faces)
    return int(token) - edge_count_offset(max_faces)


def max_edge_count(max_faces): return max_faces * (max_faces - 1) // 2


def degree_offset(max_faces):
    """Token id range for per-face degree tokens.

    Degrees are in [0, max_faces - 1] (a face can be adjacent to at most
    max_faces - 1 others), so the range size is max_faces.
    """
    return edge_count_offset(max_faces) + max_edge_count(max_faces) + 1


def degree_token(degree, max_faces): return degree_offset(max_faces) + int(degree)
def token_to_degree(token, max_faces):
    if torch.is_tensor(token):
        return token - degree_offset(max_faces)
    return int(token) - degree_offset(max_faces)


def bos_token(max_faces): return degree_offset(max_faces) + max_faces


def vocab_size(max_faces): return bos_token(max_faces) + 1
def num_pair_tokens(max_faces): return max_edge_count(max_faces)


def sequence_length(max_faces):
    """[N_FACE] [N_EDGE] [DEG_0..DEG_{N-1}] [PAIR_0..PAIR_{K-1}]."""
    return 2 + max_faces + num_pair_tokens(max_faces)


def edge_to_index(i, j, max_faces):
    if i < 0 or j < 0 or i >= j or j >= max_faces:
        raise ValueError(f"Invalid upper-triangle edge ({i}, {j}) for max_faces={max_faces}")
    return i * (2 * max_faces - i - 1) // 2 + (j - i - 1)


def build_pair_index(max_faces):
    rows, cols = [], []
    for i in range(max_faces):
        for j in range(i + 1, max_faces):
            rows.append(i)
            cols.append(j)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)


def symmetrize_adj(face_adj):
    adj = np.asarray(face_adj).astype(np.float32)
    adj = ((adj > 0.5) | (adj.T > 0.5)).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj


def canonical_face_order(face_adj, mode, rounds=2):
    n = face_adj.shape[0]
    if mode == "none":
        return np.arange(n, dtype=np.int64)
    adj_bool = face_adj > 0.5
    degree = adj_bool.sum(axis=1).astype(np.int64)
    order = np.lexsort((np.arange(n), degree))
    if mode == "degree":
        return order.astype(np.int64)
    if mode != "wl":
        raise ValueError(f"Unknown canonical order mode: {mode}")
    labels = degree.copy()
    for _ in range(rounds):
        adj_ordered = adj_bool[:, order]
        keys = []
        for i in range(n):
            neighbor_labels = tuple(sorted(int(labels[j]) for j in np.where(adj_bool[i])[0]))
            row_bits = tuple(int(v) for v in adj_ordered[i])
            keys.append((int(degree[i]), neighbor_labels, row_bits, int(i)))
        order = np.array(sorted(range(n), key=lambda idx: keys[idx]), dtype=np.int64)
        labels = np.zeros(n, dtype=np.int64)
        last_key = None
        current = -1
        for idx in order:
            key = keys[int(idx)][:-1]
            if key != last_key:
                current += 1
                last_key = key
            labels[int(idx)] = current
    return order.astype(np.int64)


def adj_to_target(face_adj, max_faces, order_mode, wl_rounds):
    face_adj = symmetrize_adj(face_adj)
    num_faces = int(face_adj.shape[0])
    if num_faces > max_faces:
        raise ValueError(f"num_faces={num_faces} exceeds max_faces={max_faces}")
    order = canonical_face_order(face_adj, order_mode, wl_rounds)
    face_adj = face_adj[order][:, order]
    edge_count = int(np.triu(face_adj[:num_faces, :num_faces], 1).sum())
    degrees = face_adj[:num_faces, :num_faces].sum(axis=1).astype(np.int64)

    seq_len = sequence_length(max_faces)
    tokens = np.full(seq_len, PAD, dtype=np.int64)
    token_mask = np.zeros(seq_len, dtype=np.bool_)
    pair_targets = np.zeros(num_pair_tokens(max_faces), dtype=np.float32)
    pair_mask = np.zeros(num_pair_tokens(max_faces), dtype=np.bool_)
    degree_targets = np.zeros(max_faces, dtype=np.int64)
    degree_mask = np.zeros(max_faces, dtype=np.bool_)

    tokens[0] = face_count_token(num_faces)
    token_mask[0] = True
    tokens[1] = edge_count_token(edge_count, max_faces)
    token_mask[1] = True

    # Degree tokens occupy positions [2, 2 + max_faces).
    deg_start = 2
    for i in range(max_faces):
        pos = deg_start + i
        if i < num_faces:
            d = int(degrees[i])
            d = max(0, min(d, max_faces - 1))
            tokens[pos] = degree_token(d, max_faces)
            token_mask[pos] = True
            degree_targets[i] = d
            degree_mask[i] = True

    # Pair tokens occupy positions [2 + max_faces, ...].
    pair_start = 2 + max_faces
    for i in range(max_faces):
        for j in range(i + 1, max_faces):
            idx = edge_to_index(i, j, max_faces)
            pos = pair_start + idx
            if i < num_faces and j < num_faces:
                value = float(face_adj[i, j] > 0.5)
                tokens[pos] = EDGE if value else NO_EDGE
                token_mask[pos] = True
                pair_targets[idx] = value
                pair_mask[idx] = True

    return (tokens, token_mask, pair_targets, pair_mask,
            degree_targets, degree_mask, num_faces, edge_count)


def target_to_adj(tokens, max_faces, min_faces=1):
    """Decode generated tokens into adjacency matrix.

    tokens layout: [N_FACE, N_EDGE, DEG_0..DEG_{max_faces-1}, PAIR_0..]
    """
    device = tokens.device
    batch_size = tokens.shape[0]
    count = token_to_face_count(tokens[:, 0]).clamp(min_faces, max_faces)
    edge_count = token_to_edge_count(tokens[:, 1], max_faces).clamp(0, max_edge_count(max_faces))
    pair_start = 2 + max_faces
    pair_tokens = tokens[:, pair_start:]
    adj = torch.zeros(batch_size, max_faces, max_faces, device=device, dtype=torch.float32)
    rows, cols = build_pair_index(max_faces)
    rows = rows.to(device)
    cols = cols.to(device)
    for b in range(batch_size):
        n = int(count[b].item())
        active = (rows < n) & (cols < n) & (pair_tokens[b] == EDGE)
        adj[b, rows[active], cols[active]] = 1.0
        adj[b, cols[active], rows[active]] = 1.0
    return adj, count, edge_count


def adjacency_stats(adj, count):
    adj_np = adj.detach().cpu().numpy()
    count_np = count.detach().cpu().numpy().astype(np.int64)
    connected, no_isolated, valid_strict, edge_counts, densities = [], [], [], [], []
    for matrix, n in zip(adj_np, count_np):
        n = int(n)
        if n <= 1:
            connected.append(1.0); no_isolated.append(1.0); valid_strict.append(1.0)
            edge_counts.append(0.0); densities.append(0.0)
            continue
        sub = matrix[:n, :n] > 0.5
        ec = float(np.triu(sub, 1).sum())
        edge_counts.append(ec)
        densities.append(ec / (n * (n - 1) / 2))
        deg = sub.sum(axis=1)
        no_iso_flag = bool(np.all(deg > 0))
        no_isolated.append(float(no_iso_flag))
        seen = {0}; stack = [0]
        while stack:
            cur = stack.pop()
            for nxt in np.where(sub[cur])[0]:
                if int(nxt) not in seen:
                    seen.add(int(nxt))
                    stack.append(int(nxt))
        conn_flag = bool(len(seen) == n)
        connected.append(float(conn_flag))
        # Strict structural validity: connected AND no isolated face.
        # No empirical degree / density thresholds; pure structural check.
        valid_strict.append(float(conn_flag and no_iso_flag))
    return {
        "connected_ratio": float(np.mean(connected)) if connected else 0.0,
        "no_isolated_ratio": float(np.mean(no_isolated)) if no_isolated else 0.0,
        "valid_strict_ratio": float(np.mean(valid_strict)) if valid_strict else 0.0,
        "edge_count_mean": float(np.mean(edge_counts)) if edge_counts else 0.0,
        "density_mean": float(np.mean(densities)) if densities else 0.0,
    }


class FaceAdjDegreeVAEDataset(Dataset):
    def __init__(self, raw_root, model_list, max_faces, order_mode, wl_rounds, max_samples=0):
        self.raw_root = Path(raw_root)
        self.max_faces = max_faces
        self.order_mode = order_mode
        self.wl_rounds = wl_rounds
        ids = [
            line.strip()
            for line in Path(model_list).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        print(f"Filtering {len(ids)} model ids from {model_list}", flush=True)
        self.model_ids = []
        skipped_raw = skipped_faces = 0
        for model_id in ids:
            path = self.raw_root / model_id / "data.npz"
            if not path.is_file():
                skipped_raw += 1
                continue
            try:
                with np.load(path) as data:
                    n = int(data["face_adj"].shape[0])
                if n > max_faces:
                    skipped_faces += 1
                    continue
            except DATA_READ_ERRORS:
                skipped_raw += 1
                continue
            self.model_ids.append(model_id)
            if max_samples and len(self.model_ids) >= max_samples:
                break
        print(
            f"Loaded {len(self.model_ids)} samples (skipped raw/load={skipped_raw}, "
            f"faces>{max_faces}={skipped_faces})",
            flush=True,
        )

    def __len__(self): return len(self.model_ids)

    def __getitem__(self, index):
        last = None
        for offset in range(len(self.model_ids)):
            model_id = self.model_ids[(index + offset) % len(self.model_ids)]
            try:
                with np.load(self.raw_root / model_id / "data.npz") as data:
                    fa = data["face_adj"]
                (tokens, token_mask, pair_targets, pair_mask,
                 deg_targets, deg_mask, num_faces, edge_count) = adj_to_target(
                    face_adj=fa, max_faces=self.max_faces,
                    order_mode=self.order_mode, wl_rounds=self.wl_rounds,
                )
                return {
                    "model_id": model_id,
                    "tokens": torch.from_numpy(tokens),
                    "token_mask": torch.from_numpy(token_mask),
                    "pair_targets": torch.from_numpy(pair_targets),
                    "pair_mask": torch.from_numpy(pair_mask),
                    "degree_targets": torch.from_numpy(deg_targets),
                    "degree_mask": torch.from_numpy(deg_mask),
                    "num_faces": torch.tensor(num_faces, dtype=torch.long),
                    "edge_count": torch.tensor(edge_count, dtype=torch.long),
                }
            except DATA_READ_ERRORS as exc:
                last = exc
                print(f"Skip bad sample {model_id}: {type(exc).__name__}: {exc}", flush=True)
        raise RuntimeError("No readable sample found") from last


class FaceAdjDegreeTransformerVAE(nn.Module):
    def __init__(self, max_faces=30, d_model=256, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.max_faces = max_faces
        self.seq_len = sequence_length(max_faces)
        self.vocab_size = vocab_size(max_faces)
        self.d_model = d_model
        self.bos = bos_token(max_faces)

        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.register_buffer("positional_encoding", self._build_pe(self.seq_len, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                          dim_feedforward=dim_feedforward, dropout=dropout,
                                          batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_encoder_layers, norm=nn.LayerNorm(d_model))
        self.fc_mu = nn.Linear(d_model, d_model)
        self.fc_logvar = nn.Linear(d_model, d_model)
        dec = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                          dim_feedforward=dim_feedforward, dropout=dropout,
                                          batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, num_layers=num_decoder_layers, norm=nn.LayerNorm(d_model))
        self.output_layer = nn.Linear(d_model, self.vocab_size)

    @staticmethod
    def _build_pe(seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    @staticmethod
    def causal_mask(size, device):
        return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)

    def shifted_decoder_input(self, tokens, token_mask):
        bos = torch.full((tokens.shape[0], 1), self.bos, dtype=tokens.dtype, device=tokens.device)
        bos_mask = torch.ones((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device)
        decoder_input = torch.cat([bos, tokens[:, :-1]], dim=1)
        decoder_mask = torch.cat([bos_mask, token_mask[:, :-1]], dim=1)
        return decoder_input, decoder_mask

    def encode(self, tokens, token_mask):
        x = self.embedding(tokens) + self.positional_encoding[:, :tokens.shape[1]].to(tokens.device)
        memory = self.encoder(x, src_key_padding_mask=~token_mask)
        pooled = (memory * token_mask.unsqueeze(-1)).sum(dim=1) / token_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.fc_mu(pooled), self.fc_logvar(pooled)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, decoder_input, decoder_mask):
        tgt = self.embedding(decoder_input) + self.positional_encoding[:, :decoder_input.shape[1]].to(decoder_input.device)
        memory = z.unsqueeze(1)
        out = self.decoder(
            tgt=tgt, memory=memory,
            tgt_mask=self.causal_mask(decoder_input.shape[1], decoder_input.device),
            tgt_key_padding_mask=~decoder_mask,
        )
        return self.output_layer(out)

    def forward(self, tokens, token_mask, sample_posterior=True,
                decoder_input_override=None, decoder_mask_override=None):
        mu, logvar = self.encode(tokens, token_mask)
        z = self.reparameterize(mu, logvar) if sample_posterior else mu
        if decoder_input_override is None:
            decoder_input, decoder_mask = self.shifted_decoder_input(tokens, token_mask)
        else:
            decoder_input = decoder_input_override
            decoder_mask = decoder_mask_override
        logits = self.decode(z, decoder_input, decoder_mask)
        return logits, mu, logvar

    @torch.no_grad()
    def generate(self, z, min_faces=1, greedy=True, temperature=1.0):
        batch_size = z.shape[0]
        device = z.device
        max_pairs = num_pair_tokens(self.max_faces)
        decoder_input = torch.full((batch_size, 1), self.bos, dtype=torch.long, device=device)
        decoder_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        generated = []

        # ---- Step 1: face count ----
        logits = self.decode(z, decoder_input, decoder_mask)[:, -1]
        face_logits = logits[:, FACE_COUNT_OFFSET : FACE_COUNT_OFFSET + self.max_faces + 1]
        allowed = torch.full_like(face_logits, float("-inf"))
        allowed[:, min_faces : self.max_faces + 1] = face_logits[:, min_faces : self.max_faces + 1]
        face_cls = (allowed.argmax(dim=-1) if greedy
                    else torch.distributions.Categorical(logits=allowed / max(temperature, 1e-6)).sample())
        face_tok = face_cls + FACE_COUNT_OFFSET
        generated.append(face_tok)
        decoder_input = torch.cat([decoder_input, face_tok[:, None]], dim=1)
        decoder_mask = torch.cat([decoder_mask, torch.ones(batch_size, 1, dtype=torch.bool, device=device)], dim=1)
        counts = face_cls.clamp(min_faces, self.max_faces)

        # ---- Step 2: edge count ----
        logits = self.decode(z, decoder_input, decoder_mask)[:, -1]
        max_edges_per = counts * (counts - 1) // 2
        ec_logits = logits[:, edge_count_offset(self.max_faces) : edge_count_offset(self.max_faces) + max_edge_count(self.max_faces) + 1]
        allowed = torch.full_like(ec_logits, float("-inf"))
        for b in range(batch_size):
            allowed[b, : int(max_edges_per[b].item()) + 1] = ec_logits[b, : int(max_edges_per[b].item()) + 1]
        ec_cls = (allowed.argmax(dim=-1) if greedy
                  else torch.distributions.Categorical(logits=allowed / max(temperature, 1e-6)).sample())
        ec_tok = ec_cls + edge_count_offset(self.max_faces)
        generated.append(ec_tok)
        decoder_input = torch.cat([decoder_input, ec_tok[:, None]], dim=1)
        decoder_mask = torch.cat([decoder_mask, torch.ones(batch_size, 1, dtype=torch.bool, device=device)], dim=1)

        # ---- Step 3: per-face degree tokens ----
        # Each face's degree must be <= count[b] - 1 and respect remaining edge budget.
        deg_pred = torch.zeros(batch_size, self.max_faces, dtype=torch.long, device=device)
        for face_idx in range(self.max_faces):
            active = face_idx < counts
            next_tok = torch.full((batch_size,), PAD, dtype=torch.long, device=device)
            next_mask = active.clone()
            if active.any():
                logits = self.decode(z, decoder_input, decoder_mask)[:, -1]
                deg_logits = logits[:, degree_offset(self.max_faces) : degree_offset(self.max_faces) + self.max_faces]
                allowed = torch.full_like(deg_logits, float("-inf"))
                for b in range(batch_size):
                    if not active[b].item():
                        continue
                    n = int(counts[b].item())
                    # Degree of face_idx must be in [0, n-1].
                    allowed[b, : n] = deg_logits[b, : n]
                cls = (allowed.argmax(dim=-1) if greedy
                       else torch.distributions.Categorical(logits=allowed / max(temperature, 1e-6)).sample())
                cls = torch.where(active, cls, torch.zeros_like(cls))
                next_tok[active] = (cls + degree_offset(self.max_faces))[active]
                deg_pred[:, face_idx] = cls
            generated.append(next_tok)
            decoder_input = torch.cat([decoder_input, next_tok[:, None]], dim=1)
            decoder_mask = torch.cat([decoder_mask, next_mask[:, None]], dim=1)

        # ---- Step 4: pair tokens with degree-budget constrained decoding ----
        rows, cols = build_pair_index(self.max_faces)
        rows = rows.to(device)
        cols = cols.to(device)
        remaining_deg = deg_pred.clone()  # [B, max_faces]
        gen_edge_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        for pair_idx in range(max_pairs):
            i = rows[pair_idx]
            j = cols[pair_idx]
            active = (i < counts) & (j < counts)
            next_tok = torch.full((batch_size,), PAD, dtype=torch.long, device=device)
            next_mask = active.clone()
            if active.any():
                logits = self.decode(z, decoder_input, decoder_mask)[:, -1]
                pair_logits = torch.stack([logits[:, NO_EDGE], logits[:, EDGE]], dim=-1)
                # Edge-count budget (same as 06-06 baseline).
                rem_valid_edge = (
                    (rows[pair_idx:][None, :] < counts[:, None])
                    & (cols[pair_idx:][None, :] < counts[:, None])
                ).sum(dim=1)
                rem_needed_edge = (ec_cls - gen_edge_counts).clamp_min(0)
                # Degree budget for this pair.
                deg_i = remaining_deg.gather(1, i.expand(batch_size, 1))[:, 0]
                deg_j = remaining_deg.gather(1, j.expand(batch_size, 1))[:, 0]
                # remaining_pairs_with_i: how many more pairs (in upper triangle order) include face i and are within active range.
                pair_mask_with_i = (
                    ((rows[pair_idx:][None, :] == i[None]) | (cols[pair_idx:][None, :] == i[None]))
                    & (rows[pair_idx:][None, :] < counts[:, None])
                    & (cols[pair_idx:][None, :] < counts[:, None])
                )
                pair_mask_with_j = (
                    ((rows[pair_idx:][None, :] == j[None]) | (cols[pair_idx:][None, :] == j[None]))
                    & (rows[pair_idx:][None, :] < counts[:, None])
                    & (cols[pair_idx:][None, :] < counts[:, None])
                )
                rem_with_i = pair_mask_with_i.sum(dim=1)
                rem_with_j = pair_mask_with_j.sum(dim=1)

                force_no = active & (
                    (gen_edge_counts >= ec_cls)
                    | (deg_i <= 0) | (deg_j <= 0)
                )
                force_yes = active & (
                    (rem_needed_edge >= rem_valid_edge)
                    | (deg_i >= rem_with_i) | (deg_j >= rem_with_j)
                ) & ~force_no  # if both pull in opposite directions, prefer no-edge to keep budgets safe.

                if greedy:
                    cls = pair_logits.argmax(dim=-1)
                else:
                    cls = torch.distributions.Categorical(logits=pair_logits / max(temperature, 1e-6)).sample()
                cls = torch.where(force_no, torch.zeros_like(cls), cls)
                cls = torch.where(force_yes, torch.ones_like(cls), cls)
                cls = torch.where(active, cls, torch.zeros_like(cls))
                next_tok[active] = torch.where(cls[active].bool(), EDGE, NO_EDGE)
                # Update budgets.
                chose = active & cls.bool()
                gen_edge_counts += chose.long()
                # Decrement remaining_deg for both endpoints whenever an edge is chosen.
                if chose.any():
                    update_idx_i = i.expand(batch_size)
                    update_idx_j = j.expand(batch_size)
                    deltas = chose.long()
                    remaining_deg.scatter_add_(1, update_idx_i.unsqueeze(1), -deltas.unsqueeze(1))
                    remaining_deg.scatter_add_(1, update_idx_j.unsqueeze(1), -deltas.unsqueeze(1))
                    remaining_deg.clamp_min_(0)
            generated.append(next_tok)
            decoder_input = torch.cat([decoder_input, next_tok[:, None]], dim=1)
            decoder_mask = torch.cat([decoder_mask, next_mask[:, None]], dim=1)

        return torch.stack(generated, dim=1)


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def safe_model_forward(model, tokens, token_mask, **kwargs):
    if isinstance(model, nn.DataParallel) and tokens.shape[0] < len(model.device_ids):
        return model.module(tokens, token_mask, **kwargs)
    return model(tokens, token_mask, **kwargs)


def compute_loss(logits, mu, logvar,
                 pair_targets, pair_mask,
                 degree_targets, degree_mask,
                 num_faces, edge_count,
                 edge_pos_weight, face_count_loss_weight,
                 edge_count_loss_weight, degree_loss_weight, kl_beta):
    max_faces = degree_targets.shape[1]
    face_logits = logits[:, 0, FACE_COUNT_OFFSET : FACE_COUNT_OFFSET + max_faces + 1]
    edge_count_logits = logits[:, 1, edge_count_offset(max_faces) : edge_count_offset(max_faces) + max_edge_count(max_faces) + 1]
    deg_logits = logits[:, 2 : 2 + max_faces, degree_offset(max_faces) : degree_offset(max_faces) + max_faces]
    pair_logits_full = logits[:, 2 + max_faces :]

    face_count_loss = F.cross_entropy(face_logits, num_faces)
    edge_count_loss = F.cross_entropy(edge_count_logits, edge_count)

    deg_logits_flat = deg_logits.reshape(-1, max_faces)
    deg_targets_flat = degree_targets.reshape(-1)
    deg_mask_flat = degree_mask.reshape(-1)
    if deg_mask_flat.any():
        deg_loss_per = F.cross_entropy(
            deg_logits_flat[deg_mask_flat], deg_targets_flat[deg_mask_flat], reduction="mean",
        )
    else:
        deg_loss_per = torch.tensor(0.0, device=logits.device)

    edge_logit = pair_logits_full[:, :, EDGE] - pair_logits_full[:, :, NO_EDGE]
    bce = F.binary_cross_entropy_with_logits(
        edge_logit, pair_targets,
        pos_weight=torch.tensor(edge_pos_weight, device=logits.device),
        reduction="none",
    )
    pair_loss = (bce * pair_mask.float()).sum() / pair_mask.float().sum().clamp_min(1)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = (
        face_count_loss_weight * face_count_loss
        + edge_count_loss_weight * edge_count_loss
        + degree_loss_weight * deg_loss_per
        + pair_loss
        + kl_beta * kl
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "face_count_loss": float(face_count_loss.detach().cpu()),
        "edge_count_loss": float(edge_count_loss.detach().cpu()),
        "degree_loss": float(deg_loss_per.detach().cpu()),
        "pair_loss": float(pair_loss.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "kl_beta": float(kl_beta),
    }


def pair_metrics_from_logits(logits, pair_targets, pair_mask,
                              degree_targets, degree_mask,
                              num_faces, edge_count):
    max_faces = degree_targets.shape[1]
    face_logits = logits[:, 0, FACE_COUNT_OFFSET : FACE_COUNT_OFFSET + max_faces + 1]
    edge_count_logits = logits[:, 1, edge_count_offset(max_faces) : edge_count_offset(max_faces) + max_edge_count(max_faces) + 1]
    deg_logits = logits[:, 2 : 2 + max_faces, degree_offset(max_faces) : degree_offset(max_faces) + max_faces]
    pair_logits_full = logits[:, 2 + max_faces :]

    pred_count = face_logits.argmax(dim=-1)
    pred_edge_count = edge_count_logits.argmax(dim=-1)
    pred_deg = deg_logits.argmax(dim=-1)

    edge_logit = pair_logits_full[:, :, EDGE] - pair_logits_full[:, :, NO_EDGE]
    pred_edges = edge_logit > 0
    target_edges = pair_targets > 0.5
    mask = pair_mask.bool()
    tp = ((pred_edges & target_edges) & mask).sum().item()
    fp = ((pred_edges & ~target_edges) & mask).sum().item()
    fn = ((~pred_edges & target_edges) & mask).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)

    deg_correct = ((pred_deg == degree_targets) & degree_mask).sum().item()
    deg_total = degree_mask.sum().item()
    deg_acc = deg_correct / max(deg_total, 1)
    deg_diff = (pred_deg.float() - degree_targets.float()).abs() * degree_mask.float()
    deg_mae = deg_diff.sum().item() / max(deg_total, 1)
    return {
        "tf_edge_precision": precision, "tf_edge_recall": recall,
        "tf_edge_f1": f1, "tf_edge_iou": iou,
        "tf_face_count_acc": float((pred_count == num_faces).float().mean().item()),
        "tf_face_count_mae": float((pred_count.float() - num_faces.float()).abs().mean().item()),
        "tf_edge_count_acc": float((pred_edge_count == edge_count).float().mean().item()),
        "tf_edge_count_mae": float((pred_edge_count.float() - edge_count.float()).abs().mean().item()),
        "tf_degree_acc": deg_acc, "tf_degree_mae": deg_mae,
    }


def generated_metrics(generated, target_tokens,
                      target_pair_targets, target_pair_mask,
                      target_degree_targets, target_degree_mask,
                      target_num_faces, target_edge_count,
                      min_faces, prefix):
    max_faces = target_degree_targets.shape[1]
    pred_adj, pred_count, pred_edge_count_token = target_to_adj(generated, max_faces=max_faces, min_faces=min_faces)
    rows, cols = build_pair_index(max_faces)
    rows = rows.to(generated.device)
    cols = cols.to(generated.device)
    pred_pairs = pred_adj[:, rows, cols] > 0.5
    target_pairs = target_pair_targets > 0.5
    mask = target_pair_mask.bool()
    pred_edge_count_actual = (pred_pairs & mask).sum(dim=1)

    tp = ((pred_pairs & target_pairs) & mask).sum().item()
    fp = ((pred_pairs & ~target_pairs) & mask).sum().item()
    fn = ((~pred_pairs & target_pairs) & mask).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)

    pair_equal = ((pred_pairs == target_pairs) | ~mask).all(dim=1)
    count_equal = pred_count == target_num_faces
    edge_count_equal = pred_edge_count_actual == target_edge_count

    # Degree from generated: use the degree-token positions [2 : 2 + max_faces].
    pred_deg_tok = generated[:, 2 : 2 + max_faces]
    pred_deg = token_to_degree(pred_deg_tok, max_faces).clamp(0, max_faces - 1)
    deg_diff = (pred_deg.float() - target_degree_targets.float()).abs() * target_degree_mask.float()
    deg_total = target_degree_mask.sum().item()
    deg_mae = deg_diff.sum().item() / max(deg_total, 1)
    deg_acc = ((pred_deg == target_degree_targets) & target_degree_mask).sum().item() / max(deg_total, 1)
    # Actual degree from generated adjacency, for self-consistency.
    actual_deg = pred_adj.sum(dim=2).long()
    deg_self_consistent = ((actual_deg == pred_deg) & target_degree_mask).sum().item() / max(deg_total, 1)

    stats = adjacency_stats(pred_adj, pred_count)
    return {
        f"{prefix}_edge_precision": precision,
        f"{prefix}_edge_recall": recall,
        f"{prefix}_edge_f1": f1,
        f"{prefix}_edge_iou": iou,
        f"{prefix}_exact_adj_acc": float((pair_equal & count_equal).float().mean().item()),
        f"{prefix}_face_count_acc": float(count_equal.float().mean().item()),
        f"{prefix}_face_count_mae": float((pred_count.float() - target_num_faces.float()).abs().mean().item()),
        f"{prefix}_edge_count_acc": float(edge_count_equal.float().mean().item()),
        f"{prefix}_edge_count_mae": float((pred_edge_count_actual.float() - target_edge_count.float()).abs().mean().item()),
        f"{prefix}_edge_count_token_mae": float((pred_edge_count_token.float() - target_edge_count.float()).abs().mean().item()),
        f"{prefix}_degree_acc": deg_acc,
        f"{prefix}_degree_mae": deg_mae,
        f"{prefix}_degree_self_consistent": deg_self_consistent,
        f"{prefix}_connected_ratio": stats["connected_ratio"],
        f"{prefix}_no_isolated_ratio": stats["no_isolated_ratio"],
    }


@torch.no_grad()
def evaluate(model, dataloader, device, args, epoch):
    model.eval()
    core = unwrap_model(model)
    totals = Counter()
    tf_metric_totals = Counter()
    count = 0
    gen_batches = []
    gen_seen = 0
    for batch in tqdm(dataloader, desc="val", leave=False):
        tokens = batch["tokens"].to(device)
        token_mask = batch["token_mask"].to(device)
        pair_targets = batch["pair_targets"].to(device)
        pair_mask = batch["pair_mask"].to(device)
        deg_targets = batch["degree_targets"].to(device)
        deg_mask = batch["degree_mask"].to(device)
        num_faces = batch["num_faces"].to(device)
        edge_count = batch["edge_count"].to(device)

        logits, mu, logvar = safe_model_forward(model, tokens, token_mask, sample_posterior=False)
        kl_beta = args.kl_beta * min(1.0, epoch / max(args.kl_warmup_epochs, 1))
        _, parts = compute_loss(
            logits=logits, mu=mu, logvar=logvar,
            pair_targets=pair_targets, pair_mask=pair_mask,
            degree_targets=deg_targets, degree_mask=deg_mask,
            num_faces=num_faces, edge_count=edge_count,
            edge_pos_weight=args.edge_pos_weight,
            face_count_loss_weight=args.face_count_loss_weight,
            edge_count_loss_weight=args.edge_count_loss_weight,
            degree_loss_weight=args.degree_loss_weight,
            kl_beta=kl_beta,
        )
        bs = tokens.shape[0]
        for k, v in parts.items():
            totals[k] += v * bs
        m = pair_metrics_from_logits(logits, pair_targets, pair_mask, deg_targets, deg_mask, num_faces, edge_count)
        for k, v in m.items():
            tf_metric_totals[k] += v * bs
        count += bs
        if gen_seen < args.eval_generate_limit:
            take = min(bs, args.eval_generate_limit - gen_seen)
            gen = core.generate(mu[:take], min_faces=args.min_faces, greedy=True)
            gen_batches.append((
                gen, tokens[:take], pair_targets[:take], pair_mask[:take],
                deg_targets[:take], deg_mask[:take],
                num_faces[:take], edge_count[:take],
            ))
            gen_seen += take

    result = {k: v / max(count, 1) for k, v in totals.items()}
    result.update({k: v / max(count, 1) for k, v in tf_metric_totals.items()})
    if gen_batches:
        merged = []
        for i in range(8):
            merged.append(torch.cat([b[i] for b in gen_batches], dim=0))
        result.update(generated_metrics(*merged, min_faces=args.min_faces, prefix="ar_recon"))
    if args.prior_samples > 0:
        prior_z = torch.randn(args.prior_samples, core.d_model, device=device)
        prior_gen = core.generate(prior_z, min_faces=args.min_faces, greedy=False, temperature=args.prior_temperature)
        prior_adj, prior_count, prior_edge_count = target_to_adj(prior_gen, max_faces=core.max_faces, min_faces=args.min_faces)
        rows, cols = build_pair_index(core.max_faces)
        rows = rows.to(device)
        cols = cols.to(device)
        actual_ec = (prior_adj[:, rows, cols] > 0.5).sum(dim=1)
        stats = adjacency_stats(prior_adj, prior_count)
        result.update({f"prior_{k}": v for k, v in stats.items()})
        result["prior_count_mean"] = float(prior_count.float().mean().item())
        result["prior_count_std"] = float(prior_count.float().std(unbiased=False).item())
        result["prior_edge_count_token_mean"] = float(prior_edge_count.float().mean().item())
        result["prior_edge_count_actual_mean"] = float(actual_ec.float().mean().item())
    return result


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--train-list", default=DEFAULT_TRAIN_LIST)
    parser.add_argument("--val-list", default=DEFAULT_VAL_LIST)
    parser.add_argument("--test-list", default=DEFAULT_TEST_LIST)
    parser.add_argument("--output-dir", default="experiments/2026-06-08/outputs_degree_only")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--max-faces", type=int, default=30)
    parser.add_argument("--min-faces", type=int, default=7)
    parser.add_argument("--order-mode", choices=["none", "degree", "wl"], default="wl")
    parser.add_argument("--wl-rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kl-beta", type=float, default=0.001)
    parser.add_argument("--kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--edge-pos-weight", type=float, default=1.0)
    parser.add_argument("--face-count-loss-weight", type=float, default=1.0)
    parser.add_argument("--edge-count-loss-weight", type=float, default=0.2)
    parser.add_argument("--degree-loss-weight", type=float, default=0.2,
                        help="Weight for per-face degree CE loss.")
    parser.add_argument("--eval-generate-limit", type=int, default=256)
    parser.add_argument("--prior-samples", type=int, default=256)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--no-data-parallel", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def load_checkpoint_args(args):
    if not args.checkpoint:
        return args, None
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    for key in (
        "max_faces", "min_faces", "order_mode", "wl_rounds",
        "d_model", "nhead", "encoder_layers", "decoder_layers",
        "dim_feedforward", "dropout", "kl_beta", "kl_warmup_epochs",
        "edge_pos_weight", "face_count_loss_weight",
        "edge_count_loss_weight", "degree_loss_weight",
    ):
        if key in ckpt_args:
            setattr(args, key, ckpt_args[key])
    return args, checkpoint


def build_model(args, device):
    return FaceAdjDegreeTransformerVAE(
        max_faces=args.max_faces, d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.encoder_layers, num_decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
    ).to(device)


def build_dataset(args, split):
    if split == "test":
        model_list = args.test_list
        max_samples = args.max_test_samples
    elif split == "val":
        model_list = args.val_list
        max_samples = args.max_val_samples
    else:
        raise ValueError(f"Invalid split: {split}")
    return FaceAdjDegreeVAEDataset(
        raw_root=args.raw_root, model_list=model_list, max_faces=args.max_faces,
        order_mode=args.order_mode, wl_rounds=args.wl_rounds, max_samples=max_samples,
    )


def build_loader(dataset, args, shuffle):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=True, drop_last=False)


def run_eval_only(args, checkpoint):
    if checkpoint is None:
        raise ValueError("--eval-only requires --checkpoint")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args, args.eval_split)
    loader = build_loader(dataset, args, shuffle=False)
    model = build_model(args, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metrics_epoch = int(checkpoint.get("metrics", {}).get("epoch", args.epochs))
    metrics = evaluate(model, loader, device, args, metrics_epoch)
    payload = {"split": args.eval_split, "checkpoint": args.checkpoint,
               "num_samples": len(dataset), **metrics}
    save_json(output_dir / f"{args.eval_split}_metrics.json", payload)
    print(
        f"{args.eval_split} loss={metrics.get('loss', 0.0):.4f} "
        f"tf_f1={metrics.get('tf_edge_f1', 0.0):.4f} "
        f"ar_f1={metrics.get('ar_recon_edge_f1', 0.0):.4f} "
        f"ar_exact={metrics.get('ar_recon_exact_adj_acc', 0.0):.4f} "
        f"deg_acc={metrics.get('ar_recon_degree_acc', 0.0):.4f} "
        f"prior_conn={metrics.get('prior_connected_ratio', 0.0):.3f}",
        flush=True,
    )


def main():
    args = parse_args()
    args, checkpoint = load_checkpoint_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "args.json", vars(args))
    if args.eval_only:
        run_eval_only(args, checkpoint)
        return

    train_dataset = FaceAdjDegreeVAEDataset(
        raw_root=args.raw_root, model_list=args.train_list, max_faces=args.max_faces,
        order_mode=args.order_mode, wl_rounds=args.wl_rounds, max_samples=args.max_train_samples,
    )
    val_dataset = build_dataset(args, "val")
    train_loader = build_loader(train_dataset, args, shuffle=True)
    val_loader = build_loader(val_dataset, args, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    if torch.cuda.device_count() > 1 and not args.no_data_parallel:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    print(
        f"Model ready on {device}; trainable parameters={sum(p.numel() for p in params):,}; "
        f"train={len(train_dataset)} val={len(val_dataset)}; "
        f"degree_loss_weight={args.degree_loss_weight}",
        flush=True,
    )

    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        kl_beta = args.kl_beta * min(1.0, epoch / max(args.kl_warmup_epochs, 1))
        progress = tqdm(train_loader, desc=f"epoch {epoch:03d}/{args.epochs} train")
        running = Counter()
        seen = 0
        for batch in progress:
            tokens = batch["tokens"].to(device, non_blocking=True)
            token_mask = batch["token_mask"].to(device, non_blocking=True)
            pair_targets = batch["pair_targets"].to(device, non_blocking=True)
            pair_mask = batch["pair_mask"].to(device, non_blocking=True)
            deg_targets = batch["degree_targets"].to(device, non_blocking=True)
            deg_mask = batch["degree_mask"].to(device, non_blocking=True)
            num_faces = batch["num_faces"].to(device, non_blocking=True)
            edge_count = batch["edge_count"].to(device, non_blocking=True)

            logits, mu, logvar = safe_model_forward(model, tokens, token_mask, sample_posterior=True)
            loss, parts = compute_loss(
                logits=logits, mu=mu, logvar=logvar,
                pair_targets=pair_targets, pair_mask=pair_mask,
                degree_targets=deg_targets, degree_mask=deg_mask,
                num_faces=num_faces, edge_count=edge_count,
                edge_pos_weight=args.edge_pos_weight,
                face_count_loss_weight=args.face_count_loss_weight,
                edge_count_loss_weight=args.edge_count_loss_weight,
                degree_loss_weight=args.degree_loss_weight,
                kl_beta=kl_beta,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            bs = tokens.shape[0]
            seen += bs
            for k, v in parts.items():
                running[k] += v * bs
            progress.set_postfix({
                "loss": f"{running['loss'] / max(seen, 1):.4f}",
                "pair": f"{running['pair_loss'] / max(seen, 1):.4f}",
                "deg": f"{running['degree_loss'] / max(seen, 1):.4f}",
                "kl": f"{running['kl'] / max(seen, 1):.4f}",
            })

        train_metrics = {f"train_{k}": v / max(seen, 1) for k, v in running.items()}
        val_metrics = evaluate(model, val_loader, device, args, epoch)
        epoch_metrics = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(epoch_metrics)
        save_json(output_dir / "metrics.json", epoch_metrics)
        save_json(output_dir / "history.json", {"history": history})

        score = val_metrics.get("ar_recon_exact_adj_acc", val_metrics.get("ar_recon_edge_f1", 0.0))
        print(
            f"epoch {epoch:03d} train_loss={train_metrics['train_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"tf_f1={val_metrics.get('tf_edge_f1', 0.0):.4f} "
            f"ar_f1={val_metrics.get('ar_recon_edge_f1', 0.0):.4f} "
            f"ar_exact={val_metrics.get('ar_recon_exact_adj_acc', 0.0):.4f} "
            f"deg_acc={val_metrics.get('ar_recon_degree_acc', 0.0):.4f} "
            f"prior_conn={val_metrics.get('prior_connected_ratio', 0.0):.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            torch.save(
                {"model": unwrap_model(model).state_dict(),
                 "args": vars(args), "metrics": epoch_metrics},
                output_dir / "best.pt",
            )
            print(f"saved best checkpoint to {output_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
