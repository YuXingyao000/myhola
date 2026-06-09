#!/usr/bin/env python3
"""DTGBrepGen-style binary face-adjacency Transformer VAE.

This is a topology-prior experiment. It does not use images or HoLa latents.
The only supervision is GT `data.npz["face_adj"]`.

Representation:
    target[0] = N_FACE token
    target[1:] = fixed max-face upper-triangle binary adjacency tokens

The target sequence has fixed positions, so position 1 is always pair (0, 1),
position 2 is always pair (0, 2), and so on. Pair positions outside the
sample's face count are PAD and ignored by the loss.

Architecture:
    adjacency sequence -> Transformer encoder -> mu/logvar -> z
    shifted adjacency sequence + z -> causal Transformer decoder -> adjacency

Loss:
    count CE + binary pair BCE + beta * KL(q(z|topology) || N(0, I))
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
COUNT_OFFSET = 3
MASKED_LOGIT = -1.0e9


def count_token(num_faces: int) -> int:
    return COUNT_OFFSET + int(num_faces)


def token_to_count(token: int) -> int:
    if torch.is_tensor(token):
        return token - COUNT_OFFSET
    return int(token) - COUNT_OFFSET


def bos_token(max_faces: int) -> int:
    return COUNT_OFFSET + max_faces + 1


def vocab_size(max_faces: int) -> int:
    return bos_token(max_faces) + 1


def num_pair_tokens(max_faces: int) -> int:
    return max_faces * (max_faces - 1) // 2


def sequence_length(max_faces: int) -> int:
    return 1 + num_pair_tokens(max_faces)


def edge_to_index(i: int, j: int, max_faces: int) -> int:
    if i < 0 or j < 0 or i >= j or j >= max_faces:
        raise ValueError(f"Invalid upper-triangle edge ({i}, {j}) for max_faces={max_faces}")
    return i * (2 * max_faces - i - 1) // 2 + (j - i - 1)


def index_to_edge(index: int, max_faces: int) -> tuple[int, int]:
    if index < 0 or index >= num_pair_tokens(max_faces):
        raise ValueError(f"Invalid edge index {index} for max_faces={max_faces}")
    remaining = int(index)
    for i in range(max_faces - 1):
        row_len = max_faces - i - 1
        if remaining < row_len:
            return i, i + 1 + remaining
        remaining -= row_len
    raise ValueError(f"Invalid edge index {index} for max_faces={max_faces}")


def build_pair_index(max_faces: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    cols = []
    for i in range(max_faces):
        for j in range(i + 1, max_faces):
            rows.append(i)
            cols.append(j)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)


def symmetrize_adj(face_adj: np.ndarray) -> np.ndarray:
    adj = np.asarray(face_adj).astype(np.float32)
    adj = ((adj > 0.5) | (adj.T > 0.5)).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj


def canonical_face_order(face_adj: np.ndarray, mode: str, rounds: int = 2) -> np.ndarray:
    """Return a deterministic face order.

    `degree` follows DTGBrepGen's main idea. `wl` adds a small topology-only
    tie breaker using neighbor degree signatures and rows under the previous
    order. The original index is used only as the final deterministic fallback.
    """
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


def adj_to_target(
    face_adj: np.ndarray,
    max_faces: int,
    order_mode: str,
    wl_rounds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    face_adj = symmetrize_adj(face_adj)
    num_faces = int(face_adj.shape[0])
    if num_faces > max_faces:
        raise ValueError(f"num_faces={num_faces} exceeds max_faces={max_faces}")

    order = canonical_face_order(face_adj, order_mode, wl_rounds)
    face_adj = face_adj[order][:, order]

    seq_len = sequence_length(max_faces)
    tokens = np.full(seq_len, PAD, dtype=np.int64)
    token_mask = np.zeros(seq_len, dtype=np.bool_)
    pair_targets = np.zeros(num_pair_tokens(max_faces), dtype=np.float32)
    pair_mask = np.zeros(num_pair_tokens(max_faces), dtype=np.bool_)

    tokens[0] = count_token(num_faces)
    token_mask[0] = True

    for i in range(max_faces):
        for j in range(i + 1, max_faces):
            idx = edge_to_index(i, j, max_faces)
            pos = 1 + idx
            if i < num_faces and j < num_faces:
                value = float(face_adj[i, j] > 0.5)
                tokens[pos] = EDGE if value else NO_EDGE
                token_mask[pos] = True
                pair_targets[idx] = value
                pair_mask[idx] = True

    return tokens, token_mask, pair_targets, pair_mask, num_faces


def target_to_adj(tokens: torch.Tensor, max_faces: int, min_faces: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse generated target tokens into adjacency matrices.

    Args:
        tokens: [B, 1 + max_pairs], where tokens[:, 0] is a count token.
    """
    device = tokens.device
    batch_size = tokens.shape[0]
    count = token_to_count(tokens[:, 0]).clamp(min_faces, max_faces)
    adj = torch.zeros(batch_size, max_faces, max_faces, device=device, dtype=torch.float32)
    rows, cols = build_pair_index(max_faces)
    rows = rows.to(device)
    cols = cols.to(device)
    pair_tokens = tokens[:, 1:]
    for b in range(batch_size):
        n = int(count[b].item())
        active = (rows < n) & (cols < n) & (pair_tokens[b] == EDGE)
        adj[b, rows[active], cols[active]] = 1.0
        adj[b, cols[active], rows[active]] = 1.0
    return adj, count


def adjacency_stats(adj: torch.Tensor, count: torch.Tensor) -> dict[str, float]:
    adj_np = adj.detach().cpu().numpy()
    count_np = count.detach().cpu().numpy().astype(np.int64)
    connected = []
    no_isolated = []
    valid_strict = []
    edge_counts = []
    densities = []
    for matrix, n in zip(adj_np, count_np):
        n = int(n)
        if n <= 1:
            connected.append(1.0)
            no_isolated.append(1.0)
            valid_strict.append(1.0)
            edge_counts.append(0.0)
            densities.append(0.0)
            continue
        sub = matrix[:n, :n] > 0.5
        edge_count = float(np.triu(sub, 1).sum())
        edge_counts.append(edge_count)
        densities.append(edge_count / (n * (n - 1) / 2))
        degree = sub.sum(axis=1)
        no_iso_flag = bool(np.all(degree > 0))
        no_isolated.append(float(no_iso_flag))
        seen = {0}
        stack = [0]
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


class FaceAdjVAEDataset(Dataset):
    def __init__(
        self,
        raw_root: str | Path,
        model_list: str | Path,
        max_faces: int,
        order_mode: str,
        wl_rounds: int,
        max_samples: int = 0,
    ):
        self.raw_root = Path(raw_root)
        self.max_faces = max_faces
        self.order_mode = order_mode
        self.wl_rounds = wl_rounds

        model_ids = [
            line.strip()
            for line in Path(model_list).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        print(f"Filtering {len(model_ids)} model ids from {model_list}", flush=True)

        self.model_ids = []
        skipped_raw = 0
        skipped_faces = 0
        for model_id in model_ids:
            path = self.raw_root / model_id / "data.npz"
            if not path.is_file():
                skipped_raw += 1
                continue
            try:
                with np.load(path) as data:
                    num_faces = int(data["face_adj"].shape[0])
                if num_faces > max_faces:
                    skipped_faces += 1
                    continue
            except DATA_READ_ERRORS:
                skipped_raw += 1
                continue
            self.model_ids.append(model_id)
            if max_samples and len(self.model_ids) >= max_samples:
                break

        print(
            f"Loaded {len(self.model_ids)} samples "
            f"(skipped raw/load={skipped_raw}, faces>{max_faces}={skipped_faces})",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.model_ids)

    def __getitem__(self, index: int):
        last_error = None
        for offset in range(len(self.model_ids)):
            model_id = self.model_ids[(index + offset) % len(self.model_ids)]
            try:
                with np.load(self.raw_root / model_id / "data.npz") as data:
                    face_adj = data["face_adj"]
                tokens, token_mask, pair_targets, pair_mask, num_faces = adj_to_target(
                    face_adj=face_adj,
                    max_faces=self.max_faces,
                    order_mode=self.order_mode,
                    wl_rounds=self.wl_rounds,
                )
                return {
                    "model_id": model_id,
                    "tokens": torch.from_numpy(tokens),
                    "token_mask": torch.from_numpy(token_mask),
                    "pair_targets": torch.from_numpy(pair_targets),
                    "pair_mask": torch.from_numpy(pair_mask),
                    "num_faces": torch.tensor(num_faces, dtype=torch.long),
                }
            except DATA_READ_ERRORS as exc:
                last_error = exc
                print(f"Skip bad sample {model_id}: {type(exc).__name__}: {exc}", flush=True)
        raise RuntimeError("No readable sample found") from last_error


class FaceAdjTransformerVAE(nn.Module):
    def __init__(
        self,
        max_faces: int = 30,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_faces = max_faces
        self.seq_len = sequence_length(max_faces)
        self.vocab_size = vocab_size(max_faces)
        self.d_model = d_model
        self.bos = bos_token(max_faces)

        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.register_buffer("positional_encoding", self._build_positional_encoding(self.seq_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers, norm=nn.LayerNorm(d_model))

        self.fc_mu = nn.Linear(d_model, d_model)
        self.fc_logvar = nn.Linear(d_model, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers, norm=nn.LayerNorm(d_model))
        self.output_layer = nn.Linear(d_model, self.vocab_size)

    @staticmethod
    def _build_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    @staticmethod
    def causal_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)

    def shifted_decoder_input(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bos = torch.full((tokens.shape[0], 1), self.bos, dtype=tokens.dtype, device=tokens.device)
        bos_mask = torch.ones((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device)
        decoder_input = torch.cat([bos, tokens[:, :-1]], dim=1)
        decoder_mask = torch.cat([bos_mask, token_mask[:, :-1]], dim=1)
        return decoder_input, decoder_mask

    def encode(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(tokens) + self.positional_encoding[:, : tokens.shape[1]].to(tokens.device)
        memory = self.encoder(x, src_key_padding_mask=~token_mask)
        pooled = (memory * token_mask.unsqueeze(-1)).sum(dim=1) / token_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.fc_mu(pooled), self.fc_logvar(pooled)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, decoder_input: torch.Tensor, decoder_mask: torch.Tensor) -> torch.Tensor:
        tgt = self.embedding(decoder_input) + self.positional_encoding[:, : decoder_input.shape[1]].to(decoder_input.device)
        memory = z.unsqueeze(1)
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=self.causal_mask(decoder_input.shape[1], decoder_input.device),
            tgt_key_padding_mask=~decoder_mask,
        )
        return self.output_layer(output)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor, sample_posterior: bool = True):
        mu, logvar = self.encode(tokens, token_mask)
        z = self.reparameterize(mu, logvar) if sample_posterior else mu
        decoder_input, decoder_mask = self.shifted_decoder_input(tokens, token_mask)
        logits = self.decode(z, decoder_input, decoder_mask)
        return logits, mu, logvar

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        min_faces: int = 1,
        greedy: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        batch_size = z.shape[0]
        device = z.device
        max_pairs = num_pair_tokens(self.max_faces)
        decoder_input = torch.full((batch_size, 1), self.bos, dtype=torch.long, device=device)
        decoder_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)

        generated = []

        logits = self.decode(z, decoder_input, decoder_mask)[:, -1]
        count_logits = logits[:, COUNT_OFFSET : COUNT_OFFSET + self.max_faces + 1]
        allowed = torch.full_like(count_logits, MASKED_LOGIT)
        allowed[:, min_faces : self.max_faces + 1] = count_logits[:, min_faces : self.max_faces + 1]
        if greedy:
            count_cls = allowed.argmax(dim=-1)
        else:
            count_cls = torch.distributions.Categorical(logits=allowed / max(temperature, 1e-6)).sample()
        count_tok = count_cls + COUNT_OFFSET
        generated.append(count_tok)
        decoder_input = torch.cat([decoder_input, count_tok[:, None]], dim=1)
        decoder_mask = torch.cat([decoder_mask, torch.ones(batch_size, 1, dtype=torch.bool, device=device)], dim=1)
        counts = count_cls.clamp(min_faces, self.max_faces)

        rows, cols = build_pair_index(self.max_faces)
        rows = rows.to(device)
        cols = cols.to(device)
        for pair_idx in range(max_pairs):
            active = (rows[pair_idx] < counts) & (cols[pair_idx] < counts)
            next_tok = torch.full((batch_size,), PAD, dtype=torch.long, device=device)
            next_mask = active.clone()
            if active.any():
                logits = self.decode(z, decoder_input, decoder_mask)[:, -1]
                edge_logits = torch.stack([logits[:, NO_EDGE], logits[:, EDGE]], dim=-1)
                if greedy:
                    edge_cls = edge_logits.argmax(dim=-1)
                else:
                    edge_cls = torch.distributions.Categorical(
                        logits=edge_logits / max(temperature, 1e-6)
                    ).sample()
                next_tok[active] = torch.where(edge_cls[active].bool(), EDGE, NO_EDGE)
            generated.append(next_tok)
            decoder_input = torch.cat([decoder_input, next_tok[:, None]], dim=1)
            decoder_mask = torch.cat([decoder_mask, next_mask[:, None]], dim=1)

        return torch.stack(generated, dim=1)


def unwrap_model(model: nn.Module) -> FaceAdjTransformerVAE:
    return model.module if isinstance(model, nn.DataParallel) else model


def compute_loss(
    logits: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    tokens: torch.Tensor,
    pair_targets: torch.Tensor,
    pair_mask: torch.Tensor,
    num_faces: torch.Tensor,
    edge_pos_weight: float,
    count_loss_weight: float,
    kl_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    max_faces = pair_targets.shape[1]
    # Infer max_faces from pair count: m(m-1)/2 = num_pairs.
    max_faces = int((1 + math.sqrt(1 + 8 * max_faces)) / 2)
    count_logits = logits[:, 0, COUNT_OFFSET : COUNT_OFFSET + max_faces + 1]
    count_loss = F.cross_entropy(count_logits, num_faces)

    edge_logit = logits[:, 1:, EDGE] - logits[:, 1:, NO_EDGE]
    bce = F.binary_cross_entropy_with_logits(
        edge_logit,
        pair_targets,
        pos_weight=torch.tensor(edge_pos_weight, device=logits.device),
        reduction="none",
    )
    pair_loss = (bce * pair_mask.float()).sum() / pair_mask.float().sum().clamp_min(1)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = count_loss_weight * count_loss + pair_loss + kl_beta * kl
    return loss, {
        "loss": float(loss.detach().cpu()),
        "count_loss": float(count_loss.detach().cpu()),
        "pair_loss": float(pair_loss.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "kl_beta": float(kl_beta),
    }


def pair_metrics_from_logits(
    logits: torch.Tensor,
    pair_targets: torch.Tensor,
    pair_mask: torch.Tensor,
    num_faces: torch.Tensor,
) -> dict[str, float]:
    max_faces = int((1 + math.sqrt(1 + 8 * pair_targets.shape[1])) / 2)
    count_logits = logits[:, 0, COUNT_OFFSET : COUNT_OFFSET + max_faces + 1]
    pred_count = count_logits.argmax(dim=-1)
    edge_logit = logits[:, 1:, EDGE] - logits[:, 1:, NO_EDGE]
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
    return {
        "tf_edge_precision": precision,
        "tf_edge_recall": recall,
        "tf_edge_f1": f1,
        "tf_edge_iou": iou,
        "tf_count_acc": float((pred_count == num_faces).float().mean().item()),
        "tf_count_mae": float((pred_count.float() - num_faces.float()).abs().mean().item()),
    }


def generated_metrics(
    generated: torch.Tensor,
    target_tokens: torch.Tensor,
    target_pair_targets: torch.Tensor,
    target_pair_mask: torch.Tensor,
    target_num_faces: torch.Tensor,
    min_faces: int,
    prefix: str,
) -> dict[str, float]:
    max_faces = int((1 + math.sqrt(1 + 8 * target_pair_targets.shape[1])) / 2)
    pred_adj, pred_count = target_to_adj(generated, max_faces=max_faces, min_faces=min_faces)
    target_adj, _ = target_to_adj(target_tokens, max_faces=max_faces, min_faces=min_faces)
    rows, cols = build_pair_index(max_faces)
    rows = rows.to(generated.device)
    cols = cols.to(generated.device)
    pred_pairs = pred_adj[:, rows, cols] > 0.5
    target_pairs = target_pair_targets > 0.5
    mask = target_pair_mask.bool()

    tp = ((pred_pairs & target_pairs) & mask).sum().item()
    fp = ((pred_pairs & ~target_pairs) & mask).sum().item()
    fn = ((~pred_pairs & target_pairs) & mask).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)

    pair_equal = ((pred_pairs == target_pairs) | ~mask).all(dim=1)
    count_equal = pred_count == target_num_faces
    stats = adjacency_stats(pred_adj, pred_count)
    return {
        f"{prefix}_edge_precision": precision,
        f"{prefix}_edge_recall": recall,
        f"{prefix}_edge_f1": f1,
        f"{prefix}_edge_iou": iou,
        f"{prefix}_exact_adj_acc": float((pair_equal & count_equal).float().mean().item()),
        f"{prefix}_count_acc": float(count_equal.float().mean().item()),
        f"{prefix}_count_mae": float((pred_count.float() - target_num_faces.float()).abs().mean().item()),
        f"{prefix}_connected_ratio": stats["connected_ratio"],
        f"{prefix}_no_isolated_ratio": stats["no_isolated_ratio"],
        f"{prefix}_valid_strict_ratio": stats["valid_strict_ratio"],
    }


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, args, epoch: int) -> dict[str, float]:
    model.eval()
    core = unwrap_model(model)
    totals = Counter()
    tf_metric_totals = Counter()
    count = 0
    generated_batches = []
    generated_seen = 0

    for batch in tqdm(dataloader, desc="val", leave=False):
        tokens = batch["tokens"].to(device)
        token_mask = batch["token_mask"].to(device)
        pair_targets = batch["pair_targets"].to(device)
        pair_mask = batch["pair_mask"].to(device)
        num_faces = batch["num_faces"].to(device)

        logits, mu, logvar = model(tokens, token_mask, sample_posterior=False)
        kl_beta = args.kl_beta * min(1.0, epoch / max(args.kl_warmup_epochs, 1))
        loss, loss_parts = compute_loss(
            logits=logits,
            mu=mu,
            logvar=logvar,
            tokens=tokens,
            pair_targets=pair_targets,
            pair_mask=pair_mask,
            num_faces=num_faces,
            edge_pos_weight=args.edge_pos_weight,
            count_loss_weight=args.count_loss_weight,
            kl_beta=kl_beta,
        )
        batch_size = tokens.shape[0]
        for key, value in loss_parts.items():
            totals[key] += value * batch_size
        metrics = pair_metrics_from_logits(logits, pair_targets, pair_mask, num_faces)
        for key, value in metrics.items():
            tf_metric_totals[key] += value * batch_size
        count += batch_size

        if generated_seen < args.eval_generate_limit:
            take = min(batch_size, args.eval_generate_limit - generated_seen)
            generated = core.generate(mu[:take], min_faces=args.min_faces, greedy=True)
            generated_batches.append((
                generated,
                tokens[:take],
                pair_targets[:take],
                pair_mask[:take],
                num_faces[:take],
            ))
            generated_seen += take

    result = {key: value / max(count, 1) for key, value in totals.items()}
    result.update({key: value / max(count, 1) for key, value in tf_metric_totals.items()})

    if generated_batches:
        merged = []
        for item_idx in range(5):
            merged.append(torch.cat([batch[item_idx] for batch in generated_batches], dim=0))
        result.update(generated_metrics(*merged, min_faces=args.min_faces, prefix="ar_recon"))

    if args.prior_samples > 0:
        prior_z = torch.randn(args.prior_samples, core.d_model, device=device)
        prior_generated = core.generate(prior_z, min_faces=args.min_faces, greedy=False, temperature=args.prior_temperature)
        prior_adj, prior_count = target_to_adj(prior_generated, max_faces=core.max_faces, min_faces=args.min_faces)
        stats = adjacency_stats(prior_adj, prior_count)
        result.update({f"prior_{key}": value for key, value in stats.items()})
        result["prior_count_mean"] = float(prior_count.float().mean().item())
        result["prior_count_std"] = float(prior_count.float().std(unbiased=False).item())

    return result


def save_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--train-list", default=DEFAULT_TRAIN_LIST)
    parser.add_argument("--val-list", default=DEFAULT_VAL_LIST)
    parser.add_argument("--test-list", default=DEFAULT_TEST_LIST)
    parser.add_argument("--output-dir", default="experiments/2026-06-06/outputs_faceadj_vae")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--max-faces", type=int, default=30)
    parser.add_argument("--min-faces", type=int, default=7)
    parser.add_argument("--order-mode", choices=["none", "degree", "wl"], default="degree")
    parser.add_argument("--wl-rounds", type=int, default=2)
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
    parser.add_argument("--kl-beta", type=float, default=0.1)
    parser.add_argument("--kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--edge-pos-weight", type=float, default=1.0)
    parser.add_argument("--count-loss-weight", type=float, default=1.0)
    parser.add_argument("--eval-generate-limit", type=int, default=256)
    parser.add_argument("--prior-samples", type=int, default=256)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--no-data-parallel", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def load_checkpoint_args(args):
    if not args.checkpoint:
        return args, None
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    for key in (
        "max_faces",
        "min_faces",
        "order_mode",
        "wl_rounds",
        "d_model",
        "nhead",
        "encoder_layers",
        "decoder_layers",
        "dim_feedforward",
        "dropout",
        "kl_beta",
        "kl_warmup_epochs",
        "edge_pos_weight",
        "count_loss_weight",
    ):
        if key in ckpt_args:
            setattr(args, key, ckpt_args[key])
    return args, checkpoint


def build_model(args, device):
    model = FaceAdjTransformerVAE(
        max_faces=args.max_faces,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    return model


def build_dataset(args, split: str):
    if split == "test":
        model_list = args.test_list
        max_samples = args.max_test_samples
    elif split == "val":
        model_list = args.val_list
        max_samples = args.max_val_samples
    else:
        raise ValueError(f"Invalid split: {split}")
    return FaceAdjVAEDataset(
        raw_root=args.raw_root,
        model_list=model_list,
        max_faces=args.max_faces,
        order_mode=args.order_mode,
        wl_rounds=args.wl_rounds,
        max_samples=max_samples,
    )


def build_loader(dataset, args, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )


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
    payload = {
        "split": args.eval_split,
        "checkpoint": args.checkpoint,
        "num_samples": len(dataset),
        **metrics,
    }
    save_json(output_dir / f"{args.eval_split}_metrics.json", payload)
    print(
        f"{args.eval_split} "
        f"loss={metrics.get('loss', 0.0):.4f} "
        f"tf_f1={metrics.get('tf_edge_f1', 0.0):.4f} "
        f"ar_f1={metrics.get('ar_recon_edge_f1', 0.0):.4f} "
        f"ar_exact={metrics.get('ar_recon_exact_adj_acc', 0.0):.4f} "
        f"ar_count_mae={metrics.get('ar_recon_count_mae', 0.0):.2f} "
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

    train_dataset = FaceAdjVAEDataset(
        raw_root=args.raw_root,
        model_list=args.train_list,
        max_faces=args.max_faces,
        order_mode=args.order_mode,
        wl_rounds=args.wl_rounds,
        max_samples=args.max_train_samples,
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

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    print(
        f"Model ready on {device}; trainable parameters={sum(p.numel() for p in params):,}; "
        f"train={len(train_dataset)} val={len(val_dataset)}",
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
            num_faces = batch["num_faces"].to(device, non_blocking=True)

            logits, mu, logvar = model(tokens, token_mask, sample_posterior=True)
            loss, loss_parts = compute_loss(
                logits=logits,
                mu=mu,
                logvar=logvar,
                tokens=tokens,
                pair_targets=pair_targets,
                pair_mask=pair_mask,
                num_faces=num_faces,
                edge_pos_weight=args.edge_pos_weight,
                count_loss_weight=args.count_loss_weight,
                kl_beta=kl_beta,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            batch_size = tokens.shape[0]
            seen += batch_size
            for key, value in loss_parts.items():
                running[key] += value * batch_size
            progress.set_postfix({
                "loss": f"{running['loss'] / max(seen, 1):.4f}",
                "pair": f"{running['pair_loss'] / max(seen, 1):.4f}",
                "kl": f"{running['kl'] / max(seen, 1):.4f}",
                "beta": f"{kl_beta:.3f}",
            })

        train_metrics = {f"train_{key}": value / max(seen, 1) for key, value in running.items()}
        val_metrics = evaluate(model, val_loader, device, args, epoch)
        epoch_metrics = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(epoch_metrics)
        save_json(output_dir / "metrics.json", epoch_metrics)
        save_json(output_dir / "history.json", {"history": history})

        score = val_metrics.get("ar_recon_edge_f1", val_metrics.get("tf_edge_f1", 0.0))
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"tf_f1={val_metrics.get('tf_edge_f1', 0.0):.4f} "
            f"ar_f1={val_metrics.get('ar_recon_edge_f1', 0.0):.4f} "
            f"ar_count_mae={val_metrics.get('ar_recon_count_mae', 0.0):.2f} "
            f"prior_conn={val_metrics.get('prior_connected_ratio', 0.0):.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            checkpoint = {
                "model": unwrap_model(model).state_dict(),
                "args": vars(args),
                "metrics": epoch_metrics,
            }
            torch.save(checkpoint, output_dir / "best.pt")
            print(f"saved best checkpoint to {output_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
