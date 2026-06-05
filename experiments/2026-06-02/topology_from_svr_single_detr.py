"""DETR-style topology predictor: 1 image -> face adjacency matrix.

Key difference vs topology_from_svr_single.py:
    - svr_single uses CLS token [1024] -> MLP -> 30 face_tokens
    - This script uses patch tokens [256, 1024] + 30 learnable face queries
      -> cross-attention -> 30 face_tokens

Each face_query learns to attend to its corresponding image region. This is
the DETR set-prediction recipe (Carion et al. ECCV 2020).

Frozen DINOv2 backbone, only train face_queries + cross-attention + heads.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
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
DEFAULT_TEST_LIST = "src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt"
DATA_READ_ERRORS = (OSError, EOFError, KeyError, zipfile.BadZipFile, zlib.error)


def has_required_condition_files(cond_dir):
    if not cond_dir.exists():
        return False
    return (cond_dir / "svr.npz").is_file() or (cond_dir / "imgs.npz").is_file()


class TopologyDataset(Dataset):
    def __init__(self, raw_root, cond_root, model_list, max_faces, view_id, max_samples=0):
        self.raw_root = Path(raw_root)
        self.cond_root = Path(cond_root)
        self.max_faces = max_faces
        self.view_id = view_id
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        model_ids = [line.strip() for line in Path(model_list).read_text().splitlines()]
        model_ids = [item for item in model_ids if item and not item.startswith("#")]

        print(f"Filtering {len(model_ids)} model ids from {model_list}", flush=True)
        self.model_ids = []
        skipped_raw = 0
        skipped_condition = 0
        for model_id in model_ids:
            raw_path = self.raw_root / model_id / "data.npz"
            if not raw_path.is_file():
                skipped_raw += 1
                continue
            if not has_required_condition_files(self.cond_root / model_id):
                skipped_condition += 1
                continue
            self.model_ids.append(model_id)
            if max_samples and len(self.model_ids) >= max_samples:
                break

        print(f"Loaded {len(self.model_ids)} usable samples from {model_list}", flush=True)
        print(f"Filter out raw={skipped_raw}, condition={skipped_condition}", flush=True)

    def __len__(self):
        return len(self.model_ids)

    def __getitem__(self, index):
        last_error = None
        for offset in range(len(self.model_ids)):
            model_id = self.model_ids[(index + offset) % len(self.model_ids)]
            try:
                with np.load(self.raw_root / model_id / "data.npz") as data:
                    face_adj = data["face_adj"].astype(np.float32)

                num_faces = face_adj.shape[0]
                if num_faces > self.max_faces:
                    raise ValueError(f"{model_id} has {num_faces} faces, but max_faces={self.max_faces}")
                padded_adj = np.zeros((self.max_faces, self.max_faces), dtype=np.float32)
                padded_adj[:num_faces, :num_faces] = face_adj

                valid = np.zeros((self.max_faces,), dtype=np.float32)
                valid[:num_faces] = 1.0

                svr_path = self.cond_root / model_id / "svr.npz"
                if svr_path.is_file():
                    with np.load(svr_path) as data:
                        images = data["images"]
                else:
                    with np.load(self.cond_root / model_id / "imgs.npz") as data:
                        images = data["svr_imgs"]

                image_tensors = self.transform(images[self.view_id])

                return {
                    "model_id": model_id,
                    "images": image_tensors,
                    "face_adj": torch.from_numpy(padded_adj),
                    "valid": torch.from_numpy(valid),
                    "num_faces": torch.tensor(num_faces, dtype=torch.long),
                }
            except DATA_READ_ERRORS as exc:
                last_error = exc
                print(f"Skip bad sample {model_id}: {type(exc).__name__}: {exc}", flush=True)

        raise RuntimeError(f"No readable sample found after {len(self.model_ids)} attempts") from last_error


class TopologyPredictor(nn.Module):
    """DETR-style topology predictor.

    Frozen DINOv2 -> patch tokens [256, 1024]
    30 learnable face_queries [30, face_dim]
    Cross-attention(face_queries, patches) -> 30 face_tokens
    Pairwise edge head + valid head, same as svr_single.
    """

    def __init__(self, max_faces, dino_model, hidden_dim, face_dim, num_decoder_layers=2, nhead=8):
        super().__init__()
        from transformers import Dinov2Model

        self.max_faces = max_faces
        self.face_dim = face_dim

        self.dino = Dinov2Model.from_pretrained(dino_model)
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino_dim = self.dino.config.hidden_size

        # Project patch tokens to face_dim, so cross-attention runs in face_dim
        self.patch_projection = nn.Sequential(
            nn.Linear(self.dino_dim, face_dim),
            nn.LayerNorm(face_dim),
        )

        # Learnable face queries (DETR object queries)
        self.face_queries = nn.Parameter(torch.randn(max_faces, face_dim) * 0.02)

        # Transformer decoder: face_queries cross-attend to patches
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=face_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(face_dim),
        )

        # Heads (same as svr_single)
        self.valid_head = nn.Linear(face_dim, 1)
        self.edge_head = nn.Sequential(
            nn.Linear(face_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, images):
        self.dino.eval()
        batch_size = images.shape[0]
        with torch.no_grad():
            # Get all tokens including patches
            patch_tokens = self.dino(images).last_hidden_state[:, 1:]  # drop CLS, [B, 256, 1024]

        memory = self.patch_projection(patch_tokens)  # [B, 256, face_dim]

        # Expand face_queries to batch
        queries = self.face_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 30, face_dim]

        # Cross-attention: each face_query attends to image patches
        face_tokens = self.decoder(tgt=queries, memory=memory)  # [B, 30, face_dim]

        # Heads (identical to svr_single)
        valid_logits = self.valid_head(face_tokens).squeeze(-1)

        a = face_tokens.unsqueeze(2).expand(-1, -1, self.max_faces, -1)
        b = face_tokens.unsqueeze(1).expand(-1, self.max_faces, -1, -1)
        pair = torch.cat([a, b, torch.abs(a - b), a * b], dim=-1)
        edge_logits = self.edge_head(pair).squeeze(-1)
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))

        diag = torch.eye(self.max_faces, device=edge_logits.device, dtype=torch.bool)
        edge_logits = edge_logits.masked_fill(diag[None], -20.0)
        return valid_logits, edge_logits


def compute_loss(valid_logits, edge_logits, valid_target, adj_target):
    valid_loss = F.binary_cross_entropy_with_logits(valid_logits, valid_target)

    max_faces = adj_target.shape[1]
    upper = torch.triu(torch.ones(max_faces, max_faces, device=adj_target.device, dtype=torch.bool), diagonal=1)
    pair_mask = valid_target[:, :, None].bool() & valid_target[:, None, :].bool() & upper[None]

    raw_edge_loss = F.binary_cross_entropy_with_logits(edge_logits, adj_target, reduction="none")
    positive = ((adj_target > 0.5) & pair_mask).sum().float()
    negative = ((adj_target <= 0.5) & pair_mask).sum().float()
    pos_weight = (negative / (positive + 1.0)).clamp(min=1.0, max=50.0)
    weights = torch.where(adj_target > 0.5, pos_weight, torch.ones_like(adj_target))
    edge_loss = (raw_edge_loss * weights)[pair_mask].mean()

    total_loss = edge_loss + 0.5 * valid_loss
    return total_loss, edge_loss.detach(), valid_loss.detach()


def average_precision(scores, labels):
    labels = labels.astype(np.float32)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(len(sorted_labels)) + 1)
    return float((precision * sorted_labels).sum() / sorted_labels.sum())


def roc_auc(scores, labels):
    labels = labels.astype(np.float32)
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores)) + 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def edge_metrics(scores, labels):
    pred = scores >= 0.5
    truth = labels >= 0.5
    tp = np.logical_and(pred, truth).sum()
    fp = np.logical_and(pred, ~truth).sum()
    fn = np.logical_and(~pred, truth).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)
    return precision, recall, f1, iou


def evaluate(model, loader, device, desc="eval"):
    model.eval()
    losses = []
    all_scores = []
    all_labels = []
    count_errors = []
    exact = []
    recall_at_k = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, dynamic_ncols=True, leave=False):
            images = batch["images"].to(device)
            adj = batch["face_adj"].to(device)
            valid = batch["valid"].to(device)
            valid_logits, edge_logits = model(images)
            loss, _, _ = compute_loss(valid_logits, edge_logits, valid, adj)
            losses.append(float(loss.item()))

            probs = torch.sigmoid(edge_logits).cpu().numpy()
            adj_np = adj.cpu().numpy()
            valid_np = valid.cpu().numpy()
            valid_prob = torch.sigmoid(valid_logits).cpu().numpy()

            for item in range(probs.shape[0]):
                num_faces = int(valid_np[item].sum())
                pred_count = int((valid_prob[item] >= 0.5).sum())
                count_errors.append(abs(pred_count - num_faces))

                tri = np.triu_indices(num_faces, k=1)
                scores = probs[item, :num_faces, :num_faces][tri]
                labels = adj_np[item, :num_faces, :num_faces][tri]
                if len(scores) == 0:
                    continue
                all_scores.append(scores)
                all_labels.append(labels)
                exact.append(float(np.array_equal(scores >= 0.5, labels >= 0.5)))

                k = int(labels.sum())
                if k > 0:
                    top = np.argsort(-scores)[:k]
                    recall_at_k.append(float(labels[top].sum() / k))

    scores = np.concatenate(all_scores) if all_scores else np.zeros((0,), dtype=np.float32)
    labels = np.concatenate(all_labels) if all_labels else np.zeros((0,), dtype=np.float32)
    precision, recall, f1, iou = edge_metrics(scores, labels) if len(scores) else (0.0, 0.0, 0.0, 0.0)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "edge_ap": average_precision(scores, labels) if len(scores) else float("nan"),
        "edge_auc": roc_auc(scores, labels) if len(scores) else float("nan"),
        "edge_precision": float(precision),
        "edge_recall": float(recall),
        "edge_f1": float(f1),
        "edge_iou": float(iou),
        "recall_at_gt_edges": float(np.mean(recall_at_k)) if recall_at_k else 0.0,
        "exact_matrix_acc": float(np.mean(exact)) if exact else 0.0,
        "face_count_mae": float(np.mean(count_errors)) if count_errors else 0.0,
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set = TopologyDataset(args.raw_root, args.cond_root, args.train_list, args.max_faces, args.view_id, args.max_samples)
    val_set = TopologyDataset(args.raw_root, args.cond_root, args.val_list, args.max_faces, args.view_id, args.max_samples)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(
        f"Data ready: train={len(train_set)} samples/{len(train_loader)} batches, "
        f"val={len(val_set)} samples/{len(val_loader)} batches, batch_size={args.batch_size}",
        flush=True,
    )

    print(f"Loading frozen DINOv2 model: {args.dino_model}", flush=True)
    model = TopologyPredictor(
        args.max_faces, args.dino_model, args.hidden_dim, args.face_dim,
        num_decoder_layers=args.num_decoder_layers, nhead=args.nhead,
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} visible GPUs", flush=True)
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model ready on {device}; trainable parameters={trainable_params:,}", flush=True)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        progress = tqdm(train_loader, desc=f"epoch {epoch:03d}/{args.epochs} train", dynamic_ncols=True)
        for batch in progress:
            images = batch["images"].to(device)
            adj = batch["face_adj"].to(device)
            valid = batch["valid"].to(device)
            valid_logits, edge_logits = model(images)
            loss, edge_loss, valid_loss = compute_loss(valid_logits, edge_logits, valid, adj)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                edge=f"{edge_loss.item():.4f}",
                valid=f"{valid_loss.item():.4f}",
            )

        metrics = evaluate(model, val_loader, device, desc=f"epoch {epoch:03d}/{args.epochs} val")
        metrics["train_loss"] = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print(
            f"epoch {epoch:03d} "
            f"train_loss={metrics['train_loss']:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"f1={metrics['edge_f1']:.4f} "
            f"ap={metrics['edge_ap']:.4f} "
            f"auc={metrics['edge_auc']:.4f} "
            f"iou={metrics['edge_iou']:.4f} "
            f"face_mae={metrics['face_count_mae']:.2f}",
            flush=True,
        )

        if metrics["edge_f1"] > best_f1:
            best_f1 = metrics["edge_f1"]
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({"model": model_to_save.state_dict(), "args": vars(args), "metrics": metrics}, output_dir / "best.pt")
            (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(f"saved best checkpoint to {output_dir / 'best.pt'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--cond-root", default=DEFAULT_COND_ROOT)
    parser.add_argument("--train-list", default=DEFAULT_TRAIN_LIST)
    parser.add_argument("--val-list", default=DEFAULT_VAL_LIST)
    parser.add_argument("--output-dir", default="experiments/2026-06-02/outputs_svr_single_detr")
    parser.add_argument("--dino-model", default="facebook/dinov2-large")
    parser.add_argument("--max-faces", type=int, default=30)
    parser.add_argument("--view-id", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--face-dim", type=int, default=128)
    parser.add_argument("--num-decoder-layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.view_id < 0 or args.view_id >= 24:
        raise ValueError("--view-id must be in [0, 23]")

    train(args)


if __name__ == "__main__":
    main()
