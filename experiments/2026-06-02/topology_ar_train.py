"""Autoregressive topology predictor: image -> face-graph edge sequence.

V1 sequence representation:
    [BOS] [N_FACE=n] (i_0, j_0) (i_1, j_1) ... [EOS]
    where edge tokens are undirected face-adjacency pairs with i < j.

Vocabulary:
    PAD=0, BOS=1, EOS=2
    count_token(n) = 3 + n                         for 0 <= n <= max_faces
    edge_token(i,j) = edge_offset + upper_index(i,j) for i < j

Architecture:
    Frozen DINOv2 patch tokens -> projection -> image memory
    Standard Transformer decoder with causal self-attention and cross-attention
    to image memory -> LM head.

Loss:
    Standard next-token cross entropy, ignoring PAD.

Usage:
    cd /mnt/d/python && python experiments/2026-06-02/topology_ar_train.py \
        --output-dir experiments/2026-06-02/outputs_ar_v1
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm.auto import tqdm


DEFAULT_RAW_ROOT = "/mnt/d/data/deepcad_v7"
DEFAULT_COND_ROOT = "/mnt/d/data/deepcad_v7_cond"
DEFAULT_TRAIN_LIST = "src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt"
DEFAULT_VAL_LIST = "src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt"
DATA_READ_ERRORS = (OSError, EOFError, KeyError, zipfile.BadZipFile, zlib.error)

PAD, BOS, EOS = 0, 1, 2
NUM_SPECIAL = 3


def count_offset() -> int:
    return NUM_SPECIAL


def edge_offset(max_faces: int) -> int:
    return NUM_SPECIAL + max_faces + 1


def num_edge_tokens(max_faces: int) -> int:
    return max_faces * (max_faces - 1) // 2


def vocab_size(max_faces: int) -> int:
    return edge_offset(max_faces) + num_edge_tokens(max_faces)


def count_token(num_faces: int) -> int:
    return count_offset() + int(num_faces)


def token_to_count(token: int, max_faces: int) -> int | None:
    token = int(token)
    if count_offset() <= token < edge_offset(max_faces):
        count = token - count_offset()
        if 0 <= count <= max_faces:
            return count
    return None


def edge_to_index(i: int, j: int, max_faces: int) -> int:
    """Map upper-triangle edge (i,j), i<j, to compact index."""
    if i < 0 or j < 0 or i >= j or j >= max_faces:
        raise ValueError(f"Invalid edge ({i}, {j}) for max_faces={max_faces}")
    return i * (2 * max_faces - i - 1) // 2 + (j - i - 1)


def index_to_edge(index: int, max_faces: int) -> tuple[int, int]:
    index = int(index)
    if index < 0 or index >= num_edge_tokens(max_faces):
        raise ValueError(f"Invalid edge index {index} for max_faces={max_faces}")
    remaining = index
    for i in range(max_faces - 1):
        row = max_faces - i - 1
        if remaining < row:
            return i, i + 1 + remaining
        remaining -= row
    raise ValueError(f"Invalid edge index {index} for max_faces={max_faces}")


def edge_to_token(i: int, j: int, max_faces: int) -> int:
    return edge_offset(max_faces) + edge_to_index(i, j, max_faces)


def token_to_edge(token: int, max_faces: int) -> tuple[int, int] | None:
    token = int(token)
    idx = token - edge_offset(max_faces)
    if idx < 0 or idx >= num_edge_tokens(max_faces):
        return None
    return index_to_edge(idx, max_faces)


def adj_to_sequence(face_adj: np.ndarray, num_faces: int, max_faces: int, max_seq_len: int) -> np.ndarray:
    """Binary face adjacency -> [BOS] [N_FACE=n] sorted edge tokens [EOS] [PAD]."""
    seq = [BOS, count_token(num_faces)]
    for i in range(num_faces):
        for j in range(i + 1, num_faces):
            if face_adj[i, j] > 0.5:
                seq.append(edge_to_token(i, j, max_faces))
    seq.append(EOS)

    if len(seq) > max_seq_len:
        # This should not happen for the current DeepCAD 7-30 split with L=128.
        seq = seq[:max_seq_len]
        seq[-1] = EOS
    seq = seq + [PAD] * (max_seq_len - len(seq))
    return np.asarray(seq, dtype=np.int64)


def parse_sequence(tokens: list[int], max_faces: int) -> tuple[np.ndarray, int, set[tuple[int, int]]]:
    """Token list -> (adjacency, predicted_num_faces, edge_set)."""
    if tokens and tokens[0] == BOS:
        tokens = tokens[1:]

    pred_num_faces = None
    edge_set: set[tuple[int, int]] = set()
    for tok in tokens:
        if tok == PAD or tok == BOS:
            continue
        if tok == EOS:
            break
        count = token_to_count(tok, max_faces)
        if count is not None:
            if pred_num_faces is None:
                pred_num_faces = count
            continue
        edge = token_to_edge(tok, max_faces)
        if edge is None:
            continue
        i, j = edge
        if pred_num_faces is not None and (i >= pred_num_faces or j >= pred_num_faces):
            continue
        edge_set.add(edge)

    if pred_num_faces is None:
        pred_num_faces = max((max(edge) + 1 for edge in edge_set), default=0)
    pred_num_faces = int(max(0, min(pred_num_faces, max_faces)))

    adj = np.zeros((max_faces, max_faces), dtype=np.float32)
    for i, j in edge_set:
        if i < pred_num_faces and j < pred_num_faces:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    return adj, pred_num_faces, edge_set


def has_required_condition_files(cond_dir: Path) -> bool:
    if not cond_dir.exists():
        return False
    return (cond_dir / "svr.npz").is_file() or (cond_dir / "imgs.npz").is_file()


class TopologyARDataset(Dataset):
    def __init__(self, raw_root, cond_root, model_list, max_faces, max_seq_len,
                 view_id, max_samples=0):
        self.raw_root = Path(raw_root)
        self.cond_root = Path(cond_root)
        self.max_faces = max_faces
        self.max_seq_len = max_seq_len
        self.view_id = view_id
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        model_ids = [line.strip() for line in Path(model_list).read_text().splitlines()
                     if line.strip() and not line.startswith("#")]

        print(f"Filtering {len(model_ids)} model ids from {model_list}", flush=True)
        self.model_ids = []
        skipped_raw = skipped_cond = 0
        for mid in model_ids:
            if not (self.raw_root / mid / "data.npz").is_file():
                skipped_raw += 1
                continue
            if not has_required_condition_files(self.cond_root / mid):
                skipped_cond += 1
                continue
            self.model_ids.append(mid)
            if max_samples and len(self.model_ids) >= max_samples:
                break
        print(f"Loaded {len(self.model_ids)} samples (skipped raw={skipped_raw}, cond={skipped_cond})", flush=True)

    def __len__(self):
        return len(self.model_ids)

    def __getitem__(self, index):
        last_error = None
        for offset in range(len(self.model_ids)):
            mid = self.model_ids[(index + offset) % len(self.model_ids)]
            try:
                with np.load(self.raw_root / mid / "data.npz") as data:
                    face_adj = data["face_adj"].astype(np.float32)

                num_faces = int(face_adj.shape[0])
                if num_faces > self.max_faces:
                    raise ValueError(f"{mid} has {num_faces} faces, max {self.max_faces}")

                padded_adj = np.zeros((self.max_faces, self.max_faces), dtype=np.float32)
                padded_adj[:num_faces, :num_faces] = face_adj
                seq = adj_to_sequence(face_adj, num_faces, self.max_faces, self.max_seq_len)

                svr_path = self.cond_root / mid / "svr.npz"
                if svr_path.is_file():
                    with np.load(svr_path) as data:
                        images = data["images"]
                else:
                    with np.load(self.cond_root / mid / "imgs.npz") as data:
                        images = data["svr_imgs"]
                image = self.transform(images[self.view_id])

                return {
                    "model_id": mid,
                    "image": image,
                    "tokens": torch.from_numpy(seq),
                    "face_adj": torch.from_numpy(padded_adj),
                    "num_faces": torch.tensor(num_faces, dtype=torch.long),
                }
            except DATA_READ_ERRORS as exc:
                last_error = exc
                print(f"Skip {mid}: {type(exc).__name__}: {exc}", flush=True)
        raise RuntimeError("No readable sample") from last_error


class TopologyARTransformer(nn.Module):
    def __init__(self, vocab_size, max_faces, max_seq_len, d_model=512, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 dino_model="facebook/dinov2-large"):
        super().__init__()
        from transformers import Dinov2Model

        self.vocab_size = vocab_size
        self.max_faces = max_faces
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.dino = Dinov2Model.from_pretrained(dino_model)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.image_projection = nn.Sequential(
            nn.Linear(self.dino.config.hidden_size, d_model),
            nn.LayerNorm(d_model),
        )

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def encode_image(self, image):
        self.dino.eval()
        with torch.no_grad():
            patches = self.dino(image).last_hidden_state[:, 1:]
        return self.image_projection(patches)

    def forward(self, image, tokens):
        memory = self.encode_image(image)
        L = tokens.shape[1]
        pos = torch.arange(L, device=tokens.device)
        tgt = self.token_embed(tokens) + self.pos_embed(pos)[None]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tokens.device), diagonal=1)
        tgt_key_padding_mask = tokens == PAD
        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.lm_head(out)

    def _allowed_next_tokens(self, prefix: list[int], force_eos: bool,
                             sorted_edges: bool, require_min_edges: bool) -> torch.Tensor:
        allowed = torch.zeros(self.vocab_size, dtype=torch.bool)
        if force_eos:
            allowed[EOS] = True
            return allowed

        if prefix and prefix[-1] == EOS:
            allowed[EOS] = True
            return allowed

        counts = [token_to_count(tok, self.max_faces) for tok in prefix]
        counts = [count for count in counts if count is not None]
        if not counts:
            # Face count is the first generated token after BOS.
            for n in range(1, self.max_faces + 1):
                allowed[count_token(n)] = True
            return allowed

        n_face = counts[0]
        used_edge_indices = []
        for tok in prefix:
            edge = token_to_edge(tok, self.max_faces)
            if edge is None:
                continue
            i, j = edge
            if i < n_face and j < n_face:
                used_edge_indices.append(edge_to_index(i, j, self.max_faces))
        used = set(used_edge_indices)
        last_edge_idx = max(used_edge_indices) if (sorted_edges and used_edge_indices) else -1

        for i in range(n_face):
            for j in range(i + 1, n_face):
                idx = edge_to_index(i, j, self.max_faces)
                if idx in used:
                    continue
                if sorted_edges and idx <= last_edge_idx:
                    continue
                allowed[edge_offset(self.max_faces) + idx] = True

        min_edges = max(n_face - 1, 0) if require_min_edges else 0
        if len(used) >= min_edges:
            allowed[EOS] = True
        if not allowed.any():
            allowed[EOS] = True
        return allowed

    @torch.no_grad()
    def generate(self, image, max_len=None, decode="greedy", temperature=1.0, top_k=0,
                 sorted_edges=True, require_min_edges=True):
        max_len = max_len or self.max_seq_len
        memory = self.encode_image(image)
        B = image.shape[0]
        device = image.device

        tokens = torch.full((B, 1), BOS, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            L = tokens.shape[1]
            pos = torch.arange(L, device=device)
            tgt = self.token_embed(tokens) + self.pos_embed(pos)[None]
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
            out = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal)
            logits = self.lm_head(out[:, -1])

            allowed = []
            force_eos = L >= max_len - 1
            for b in range(B):
                allowed_b = self._allowed_next_tokens(
                    tokens[b].tolist(),
                    force_eos=force_eos or bool(finished[b]),
                    sorted_edges=sorted_edges,
                    require_min_edges=require_min_edges,
                )
                allowed.append(allowed_b)
            allowed_mask = torch.stack(allowed, dim=0).to(device)
            logits = logits.masked_fill(~allowed_mask, -torch.finfo(logits.dtype).max)

            if decode == "greedy":
                next_tok = logits.argmax(dim=-1, keepdim=True)
            elif decode == "sample":
                sample_logits = logits / max(float(temperature), 1e-6)
                if top_k > 0:
                    k = min(int(top_k), sample_logits.shape[-1])
                    values, _ = sample_logits.topk(k)
                    sample_logits = sample_logits.masked_fill(sample_logits < values[:, [-1]], -torch.finfo(sample_logits.dtype).max)
                probs = F.softmax(sample_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                raise ValueError(f"Unknown decode mode: {decode}")

            tokens = torch.cat([tokens, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(-1) == EOS)
            if finished.all():
                break

        return tokens


def compute_loss(logits, tokens):
    B, L, V = logits.shape
    pred = logits[:, :-1].reshape(B * (L - 1), V)
    target = tokens[:, 1:].reshape(B * (L - 1))
    return F.cross_entropy(pred, target, ignore_index=PAD)


def edge_set_from_adj(adj: np.ndarray, n: int) -> set[tuple[int, int]]:
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0.5:
                edges.add((i, j))
    return edges


def graph_counts(pred_adj: np.ndarray, pred_n: int, gt_adj: np.ndarray, gt_n: int) -> dict[str, float]:
    n = max(int(pred_n), int(gt_n), 1)
    n = min(n, pred_adj.shape[0], gt_adj.shape[0])
    pred = pred_adj[:n, :n] > 0.5
    gt = gt_adj[:n, :n] > 0.5
    tri = np.triu_indices(n, k=1)
    p = pred[tri]
    g = gt[tri]
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    exact = float(pred_n == gt_n and np.array_equal(p, g))
    return {"tp": tp, "fp": fp, "fn": fn, "exact": exact}


def is_connected(adj: np.ndarray, n: int) -> bool:
    n = int(n)
    if n <= 1:
        return n == 1
    mat = adj[:n, :n] > 0.5
    if not mat.any():
        return False
    seen = {0}
    stack = [0]
    while stack:
        cur = stack.pop()
        for nxt in np.flatnonzero(mat[cur]):
            nxt = int(nxt)
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == n


def edge_jaccard_distance(edge_sets: list[set[tuple[int, int]]]) -> float:
    if len(edge_sets) < 2:
        return 0.0
    distances = []
    for i in range(len(edge_sets)):
        for j in range(i + 1, len(edge_sets)):
            union = edge_sets[i] | edge_sets[j]
            if not union:
                distances.append(0.0)
            else:
                inter = edge_sets[i] & edge_sets[j]
                distances.append(1.0 - len(inter) / len(union))
    return float(np.mean(distances)) if distances else 0.0


def f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)
    return precision, recall, f1, iou


def evaluate(model, loader, device, max_faces, max_seq_len, sample_k=4, sample_temperature=0.8,
             sample_top_k=0, sorted_edges=True, require_min_edges=True, desc="eval"):
    model.eval()
    losses = []
    greedy_tp = greedy_fp = greedy_fn = 0
    exact_count = 0.0
    face_count_err = 0.0
    edge_count_err = 0.0
    connected_count = 0.0
    sample_oracle_f1 = []
    sample_diversity = []
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, dynamic_ncols=True, leave=False):
            image = batch["image"].to(device)
            tokens = batch["tokens"].to(device)
            adj_gt = batch["face_adj"].cpu().numpy()
            num_faces_gt = batch["num_faces"].cpu().numpy()

            logits = model(image, tokens)
            losses.append(float(compute_loss(logits, tokens).item()))

            generated = model.generate(
                image,
                max_len=max_seq_len,
                decode="greedy",
                sorted_edges=sorted_edges,
                require_min_edges=require_min_edges,
            )
            for b in range(generated.shape[0]):
                pred_adj, pred_n, pred_edges = parse_sequence(generated[b].cpu().tolist(), max_faces)
                gt = adj_gt[b]
                gt_n = int(num_faces_gt[b])
                counts = graph_counts(pred_adj, pred_n, gt, gt_n)
                greedy_tp += counts["tp"]
                greedy_fp += counts["fp"]
                greedy_fn += counts["fn"]
                exact_count += counts["exact"]
                face_count_err += abs(pred_n - gt_n)
                edge_count_err += abs(len(pred_edges) - len(edge_set_from_adj(gt, gt_n)))
                connected_count += float(is_connected(pred_adj, pred_n))
                n_samples += 1

            if sample_k > 0:
                per_item_f1 = [[] for _ in range(image.shape[0])]
                per_item_edges = [[] for _ in range(image.shape[0])]
                for _sample_idx in range(sample_k):
                    sampled = model.generate(
                        image,
                        max_len=max_seq_len,
                        decode="sample",
                        temperature=sample_temperature,
                        top_k=sample_top_k,
                        sorted_edges=sorted_edges,
                        require_min_edges=require_min_edges,
                    )
                    for b in range(sampled.shape[0]):
                        pred_adj, pred_n, pred_edges = parse_sequence(sampled[b].cpu().tolist(), max_faces)
                        gt = adj_gt[b]
                        gt_n = int(num_faces_gt[b])
                        counts = graph_counts(pred_adj, pred_n, gt, gt_n)
                        _, _, f1, _ = f1_from_counts(counts["tp"], counts["fp"], counts["fn"])
                        per_item_f1[b].append(f1)
                        per_item_edges[b].append(pred_edges)
                for b in range(image.shape[0]):
                    if per_item_f1[b]:
                        sample_oracle_f1.append(max(per_item_f1[b]))
                        sample_diversity.append(edge_jaccard_distance(per_item_edges[b]))

    precision, recall, f1, iou = f1_from_counts(greedy_tp, greedy_fp, greedy_fn)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "greedy_edge_precision": float(precision),
        "greedy_edge_recall": float(recall),
        "greedy_edge_f1": float(f1),
        "greedy_edge_iou": float(iou),
        "exact_matrix_acc": float(exact_count / max(n_samples, 1)),
        "face_count_mae": float(face_count_err / max(n_samples, 1)),
        "edge_count_mae": float(edge_count_err / max(n_samples, 1)),
        "connected_ratio": float(connected_count / max(n_samples, 1)),
        "sample_oracle_f1": float(np.mean(sample_oracle_f1)) if sample_oracle_f1 else 0.0,
        "sample_diversity": float(np.mean(sample_diversity)) if sample_diversity else 0.0,
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vsize = vocab_size(args.max_faces)
    train_set = TopologyARDataset(args.raw_root, args.cond_root, args.train_list,
                                   args.max_faces, args.max_seq_len, args.view_id, args.max_samples)
    val_set = TopologyARDataset(args.raw_root, args.cond_root, args.val_list,
                                 args.max_faces, args.max_seq_len, args.view_id, args.max_samples)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, vocab: {vsize}", flush=True)

    model = TopologyARTransformer(
        vocab_size=vsize,
        max_faces=args.max_faces,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_layers,
        dim_feedforward=args.feedforward_dim,
        dropout=args.dropout,
        dino_model=args.dino_model,
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=args.lr, weight_decay=args.weight_decay)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)

    sorted_edges = not args.disable_sorted_decoding
    require_min_edges = not args.allow_early_eos

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        progress = tqdm(train_loader, desc=f"epoch {epoch:03d}", dynamic_ncols=True)
        for batch in progress:
            image = batch["image"].to(device)
            tokens = batch["tokens"].to(device)
            logits = model(image, tokens)
            loss = compute_loss(logits, tokens)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{loss.item():.4f}")

        eval_model = model.module if isinstance(model, nn.DataParallel) else model
        metrics = evaluate(
            eval_model,
            val_loader,
            device,
            args.max_faces,
            args.max_seq_len,
            sample_k=args.sample_k,
            sample_temperature=args.sample_temperature,
            sample_top_k=args.sample_top_k,
            sorted_edges=sorted_edges,
            require_min_edges=require_min_edges,
            desc=f"epoch {epoch:03d} val",
        )
        metrics["train_loss"] = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print(
            f"epoch {epoch:03d} train={metrics['train_loss']:.4f} val={metrics['loss']:.4f} "
            f"greedy_f1={metrics['greedy_edge_f1']:.4f} sample@{args.sample_k}={metrics['sample_oracle_f1']:.4f} "
            f"conn={metrics['connected_ratio']:.4f} face_mae={metrics['face_count_mae']:.2f} "
            f"edge_mae={metrics['edge_count_mae']:.2f}",
            flush=True,
        )

        score = metrics["sample_oracle_f1"] if args.sample_k > 0 else metrics["greedy_edge_f1"]
        if score > best_score:
            best_score = score
            ckpt = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({"model": ckpt.state_dict(), "args": vars(args), "metrics": metrics},
                       output_dir / "best.pt")
            (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(f"saved best.pt (score={best_score:.4f})", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--cond-root", default=DEFAULT_COND_ROOT)
    parser.add_argument("--train-list", default=DEFAULT_TRAIN_LIST)
    parser.add_argument("--val-list", default=DEFAULT_VAL_LIST)
    parser.add_argument("--output-dir", default="experiments/2026-06-02/outputs_ar_v1")
    parser.add_argument("--dino-model", default="facebook/dinov2-large")
    parser.add_argument("--max-faces", type=int, default=30)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--view-id", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--feedforward-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-k", type=int, default=4)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--sample-top-k", type=int, default=0)
    parser.add_argument("--allow-early-eos", action="store_true")
    parser.add_argument("--disable-sorted-decoding", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.max_seq_len < 4:
        raise ValueError("--max-seq-len must leave room for BOS, count, at least one edge, and EOS")
    train(args)


if __name__ == "__main__":
    main()
