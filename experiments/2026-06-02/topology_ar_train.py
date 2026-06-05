"""Autoregressive topology predictor: image -> edge sequence.

Sequence representation:
    [BOS] (i_0, j_0) (i_1, j_1) ... [EOS]
    where (i, j) with i < j is an undirected edge in face_adj.

Vocabulary:
    PAD=0, BOS=1, EOS=2
    edge_token(i,j) = 3 + i * max_faces + j   for i < j

Notes:
    - No face type (HoLa-BRep assumes all B-spline)
    - No canonical face reordering: use raw face order from data.npz
    - Edges sorted by (i, j) lexicographic order in target sequence

Architecture:
    Frozen DINOv2 -> patch tokens [256, 1024] -> projection -> image memory
    Standard Transformer decoder (causal self-attn + cross-attn to memory)
    Cross-entropy loss

Usage:
    cd /mnt/d/python && python experiments/2026-06-02/topology_ar_train.py \
        --output-dir experiments/2026-06-02/outputs_ar
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


def edge_to_token(i: int, j: int, max_faces: int) -> int:
    return NUM_SPECIAL + i * max_faces + j


def token_to_edge(token: int, max_faces: int) -> tuple[int, int] | None:
    if token < NUM_SPECIAL:
        return None
    edge_idx = token - NUM_SPECIAL
    i, j = edge_idx // max_faces, edge_idx % max_faces
    if i >= j or i < 0 or j >= max_faces:
        return None
    return (i, j)


def adj_to_sequence(face_adj: np.ndarray, max_seq_len: int) -> np.ndarray:
    """face_adj [N, N] binary -> token sequence padded to max_seq_len."""
    max_faces = face_adj.shape[0]
    seq = [BOS]
    n = face_adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if face_adj[i, j] > 0.5:
                seq.append(edge_to_token(i, j, max_faces))
    seq.append(EOS)
    seq = seq[:max_seq_len]
    seq = seq + [PAD] * (max_seq_len - len(seq))
    return np.array(seq, dtype=np.int64)


def sequence_to_adj(tokens: list[int], max_faces: int) -> np.ndarray:
    """Token list -> face_adj [max_faces, max_faces] binary symmetric."""
    adj = np.zeros((max_faces, max_faces), dtype=np.float32)
    for tok in tokens:
        if tok == EOS:
            break
        edge = token_to_edge(tok, max_faces)
        if edge is None:
            continue
        i, j = edge
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return adj


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

                num_faces = face_adj.shape[0]
                if num_faces > self.max_faces:
                    raise ValueError(f"{mid} has {num_faces} faces, max {self.max_faces}")

                # Pad face_adj to max_faces
                padded_adj = np.zeros((self.max_faces, self.max_faces), dtype=np.float32)
                padded_adj[:num_faces, :num_faces] = face_adj
                seq = adj_to_sequence(padded_adj, self.max_seq_len)

                # Load image
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
                    "tokens": torch.from_numpy(seq),  # [max_seq_len]
                    "face_adj": torch.from_numpy(padded_adj),
                    "num_faces": torch.tensor(num_faces, dtype=torch.long),
                }
            except DATA_READ_ERRORS as exc:
                last_error = exc
                print(f"Skip {mid}: {type(exc).__name__}: {exc}", flush=True)
        raise RuntimeError("No readable sample") from last_error


class TopologyARTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model=512, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 dino_model="facebook/dinov2-large"):
        super().__init__()
        from transformers import Dinov2Model

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.dino = Dinov2Model.from_pretrained(dino_model)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.image_projection = nn.Linear(self.dino.config.hidden_size, d_model)

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers,
                                              norm=nn.LayerNorm(d_model))
        self.lm_head = nn.Linear(d_model, vocab_size)

    def encode_image(self, image):
        self.dino.eval()
        with torch.no_grad():
            patches = self.dino(image).last_hidden_state[:, 1:]  # [B, 256, dino_dim]
        return self.image_projection(patches)  # [B, 256, d_model]

    def forward(self, image, tokens):
        """Training forward with teacher forcing.

        Args:
            image: [B, 3, 224, 224]
            tokens: [B, L] target sequence (input)

        Returns:
            logits: [B, L, vocab_size]
        """
        memory = self.encode_image(image)
        L = tokens.shape[1]
        pos = torch.arange(L, device=tokens.device)
        tgt = self.token_embed(tokens) + self.pos_embed(pos)[None]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tokens.device), diagonal=1)

        # PAD token mask: don't attend to padded positions
        tgt_key_padding_mask = (tokens == PAD)

        out = self.decoder(
            tgt=tgt, memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.lm_head(out)

    @torch.no_grad()
    def generate(self, image, max_len=None, temperature=1.0, top_k=0):
        """Autoregressive generation. Returns list of token tensors."""
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
            logits = self.lm_head(out[:, -1])  # [B, V]

            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # [B, 1]
            tokens = torch.cat([tokens, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(-1) == EOS)
            if finished.all():
                break

        return tokens


def compute_loss(logits, tokens):
    """Standard LM cross-entropy.

    Args:
        logits: [B, L, V]
        tokens: [B, L]

    Predict tokens[1:] from logits[:-1] (next-token prediction).
    Ignore PAD positions.
    """
    B, L, V = logits.shape
    pred = logits[:, :-1].reshape(B * (L - 1), V)
    target = tokens[:, 1:].reshape(B * (L - 1))
    loss = F.cross_entropy(pred, target, ignore_index=PAD)
    return loss


def evaluate(model, loader, device, max_faces, max_seq_len, desc="eval"):
    model.eval()
    losses = []
    edge_correct, edge_total = 0, 0
    edge_tp = edge_fp = edge_fn = 0
    exact_count = 0
    face_count_err = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, dynamic_ncols=True, leave=False):
            image = batch["image"].to(device)
            tokens = batch["tokens"].to(device)
            adj_gt = batch["face_adj"].to(device).cpu().numpy()
            num_faces_gt = batch["num_faces"].cpu().numpy()

            logits = model(image, tokens)
            losses.append(compute_loss(logits, tokens).item())

            # Greedy generate (temperature -> 0)
            generated = model.generate(image, max_len=max_seq_len, temperature=1e-3)
            for b in range(generated.shape[0]):
                tokens_b = generated[b].cpu().tolist()
                # Strip BOS
                if tokens_b[0] == BOS:
                    tokens_b = tokens_b[1:]
                pred_adj = sequence_to_adj(tokens_b, max_faces)

                gt = adj_gt[b]
                # Restrict to actual num_faces region
                n = int(num_faces_gt[b])
                pred_n = pred_adj[:n, :n]
                gt_n = gt[:n, :n]
                tri = np.triu_indices(n, k=1)
                p = pred_n[tri] > 0.5
                g = gt_n[tri] > 0.5
                edge_tp += np.logical_and(p, g).sum()
                edge_fp += np.logical_and(p, ~g).sum()
                edge_fn += np.logical_and(~p, g).sum()
                edge_correct += (p == g).sum()
                edge_total += len(p)
                if np.array_equal(p, g):
                    exact_count += 1

                # Face count: count unique face indices appearing in predicted edges
                pred_faces_seen = set()
                for tok in tokens_b:
                    e = token_to_edge(tok, max_faces)
                    if e is not None:
                        pred_faces_seen.add(e[0])
                        pred_faces_seen.add(e[1])
                # Predicted num_faces = max index + 1 (or count of seen)
                pred_n_estimated = (max(pred_faces_seen) + 1) if pred_faces_seen else 0
                face_count_err += abs(pred_n_estimated - n)
                n_samples += 1

    precision = edge_tp / max(edge_tp + edge_fp, 1)
    recall = edge_tp / max(edge_tp + edge_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = edge_tp / max(edge_tp + edge_fp + edge_fn, 1)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "edge_precision": float(precision),
        "edge_recall": float(recall),
        "edge_f1": float(f1),
        "edge_iou": float(iou),
        "exact_matrix_acc": exact_count / max(n_samples, 1),
        "face_count_mae": face_count_err / max(n_samples, 1),
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = NUM_SPECIAL + args.max_faces * args.max_faces

    train_set = TopologyARDataset(args.raw_root, args.cond_root, args.train_list,
                                   args.max_faces, args.max_seq_len, args.view_id, args.max_samples)
    val_set = TopologyARDataset(args.raw_root, args.cond_root, args.val_list,
                                 args.max_faces, args.max_seq_len, args.view_id, args.max_samples)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, vocab: {vocab_size}", flush=True)

    model = TopologyARTransformer(
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model, nhead=args.nhead,
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

    best_f1 = -1.0
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
        metrics = evaluate(eval_model, val_loader, device, args.max_faces, args.max_seq_len,
                            desc=f"epoch {epoch:03d} val")
        metrics["train_loss"] = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print(
            f"epoch {epoch:03d} train={metrics['train_loss']:.4f} val={metrics['loss']:.4f} "
            f"f1={metrics['edge_f1']:.4f} iou={metrics['edge_iou']:.4f} "
            f"exact={metrics['exact_matrix_acc']:.4f} mae={metrics['face_count_mae']:.2f}",
            flush=True,
        )

        if metrics["edge_f1"] > best_f1:
            best_f1 = metrics["edge_f1"]
            ckpt = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({"model": ckpt.state_dict(), "args": vars(args), "metrics": metrics},
                       output_dir / "best.pt")
            (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(f"saved best.pt (f1={best_f1:.4f})", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--cond-root", default=DEFAULT_COND_ROOT)
    parser.add_argument("--train-list", default=DEFAULT_TRAIN_LIST)
    parser.add_argument("--val-list", default=DEFAULT_VAL_LIST)
    parser.add_argument("--output-dir", default="experiments/2026-06-02/outputs_ar")
    parser.add_argument("--dino-model", default="facebook/dinov2-large")
    parser.add_argument("--max-faces", type=int, default=30)
    parser.add_argument("--max-seq-len", type=int, default=128)  # 30 faces -> max 435 edges, but typical << 50
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
