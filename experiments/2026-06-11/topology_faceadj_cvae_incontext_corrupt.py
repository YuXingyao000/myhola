#!/usr/bin/env python3
"""Image-conditioned topology CVAE with in-context image memory tokens.

This is the first narrow CVAE experiment after the 2026-06-08
corruption-only topology VAE. It intentionally reuses the topology sequence,
WL ordering, loss terms, autoregressive generation, and random prefix
corruption idea from that script.

Code-level change:
  - q(z | topology) becomes q(z | topology, image)
  - N(0, I) prior becomes p(z | image)
  - decoder memory becomes concat([z_token, image_context_tokens])

The random corruption remains topology-only: NO_EDGE/EDGE tokens in the
decoder prefix are flipped during training while the image condition is kept
unchanged. Image token dropout is present as an experimental knob but defaults
to 0.0 for this first run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import zipfile
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.brepnet.models.condition_encoders import DINOv2ImageEncoder


BASE_SCRIPT = Path(__file__).resolve().parents[1] / "2026-06-08" / "topology_faceadj_vae_corrupt.py"
SPEC = importlib.util.spec_from_file_location("topology_faceadj_vae_corrupt_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(base)


DEFAULT_RAW_ROOT = "/mnt/d/data/deepcad_v7"
DEFAULT_CONDITION_ROOT = "/mnt/d/data/deepcad_v7_cond"
DEFAULT_TRAIN_LIST = "src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt"
DEFAULT_VAL_LIST = "src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt"
DEFAULT_TEST_LIST = "src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt"
DATA_READ_ERRORS = (OSError, EOFError, KeyError, zipfile.BadZipFile, zlib.error)

PAD = base.PAD
NO_EDGE = base.NO_EDGE
EDGE = base.EDGE
FACE_COUNT_OFFSET = base.FACE_COUNT_OFFSET
MASKED_LOGIT = base.MASKED_LOGIT

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_rgb_condition(cond_dir: Path, image_source: str, rotation_id: int) -> np.ndarray:
    if image_source in {"real_flux", "real_flux_masked", "real_blender"}:
        key = {
            "real_flux": "flux",
            "real_flux_masked": "flux_masked",
            "real_blender": "blender",
        }[image_source]
        with np.load(cond_dir / "real_photo.npz") as data:
            return data[key]
    if image_source in {"svr", "sketch"}:
        key = "svr_imgs" if image_source == "svr" else "sketch_imgs"
        with np.load(cond_dir / "imgs.npz") as data:
            images = data[key]
            return images[int(rotation_id) % images.shape[0]]
    raise ValueError(f"Unknown image_source={image_source}")


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image [H, W, 3], got {image.shape}")
    pil = Image.fromarray(image.astype(np.uint8), mode="RGB").resize((224, 224), Image.BICUBIC)
    array = np.asarray(pil).astype(np.float32) / 255.0
    array = (array - IMAGE_MEAN) / IMAGE_STD
    array = np.transpose(array, (2, 0, 1)).copy()
    return torch.from_numpy(array)


def has_condition_file(cond_dir: Path, image_source: str) -> bool:
    if image_source in {"real_flux", "real_flux_masked", "real_blender"}:
        return (cond_dir / "real_photo.npz").is_file()
    if image_source in {"svr", "sketch"}:
        return (cond_dir / "imgs.npz").is_file()
    raise ValueError(f"Unknown image_source={image_source}")


class FaceAdjImageDataset(Dataset):
    def __init__(
        self,
        raw_root,
        condition_root,
        model_list,
        max_faces,
        order_mode,
        wl_rounds,
        image_source,
        image_rotation_id,
        max_samples=0,
    ):
        self.raw_root = Path(raw_root)
        self.condition_root = Path(condition_root)
        self.max_faces = max_faces
        self.order_mode = order_mode
        self.wl_rounds = wl_rounds
        self.image_source = image_source
        self.image_rotation_id = image_rotation_id

        model_ids = [
            line.strip()
            for line in Path(model_list).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        print(f"Filtering {len(model_ids)} model ids from {model_list}", flush=True)
        self.model_ids = []
        skipped_raw = 0
        skipped_cond = 0
        skipped_faces = 0
        for model_id in model_ids:
            raw_path = self.raw_root / model_id / "data.npz"
            cond_dir = self.condition_root / model_id
            if not raw_path.is_file():
                skipped_raw += 1
                continue
            if not has_condition_file(cond_dir, image_source):
                skipped_cond += 1
                continue
            try:
                with np.load(raw_path) as data:
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
            f"(skipped raw/load={skipped_raw}, cond={skipped_cond}, faces>{max_faces}={skipped_faces})",
            flush=True,
        )

    def __len__(self):
        return len(self.model_ids)

    def __getitem__(self, index):
        last_error = None
        for offset in range(len(self.model_ids)):
            model_id = self.model_ids[(index + offset) % len(self.model_ids)]
            try:
                with np.load(self.raw_root / model_id / "data.npz") as data:
                    face_adj = data["face_adj"]
                tokens, token_mask, pair_targets, pair_mask, num_faces, edge_count = base.adj_to_target(
                    face_adj=face_adj,
                    max_faces=self.max_faces,
                    order_mode=self.order_mode,
                    wl_rounds=self.wl_rounds,
                )
                image = load_rgb_condition(
                    self.condition_root / model_id,
                    self.image_source,
                    self.image_rotation_id,
                )
                return {
                    "model_id": model_id,
                    "tokens": torch.from_numpy(tokens),
                    "token_mask": torch.from_numpy(token_mask),
                    "pair_targets": torch.from_numpy(pair_targets),
                    "pair_mask": torch.from_numpy(pair_mask),
                    "num_faces": torch.tensor(num_faces, dtype=torch.long),
                    "edge_count": torch.tensor(edge_count, dtype=torch.long),
                    "image": image_to_tensor(image),
                }
            except DATA_READ_ERRORS as exc:
                last_error = exc
                print(f"Skip bad sample {model_id}: {type(exc).__name__}: {exc}", flush=True)
        raise RuntimeError("No readable sample found") from last_error


class DINOImageContextEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        backbone: str = "dinov2",
        depth_anything_v2_ckpt: str | None = None,
    ):
        super().__init__()
        self.encoder = DINOv2ImageEncoder(
            projection_dim=d_model,
            backbone=backbone,
            depth_anything_v2_ckpt=depth_anything_v2_ckpt,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, images: torch.Tensor, token_dropout: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encoder(images)  # [B, 257, D], CLS + 16x16 patches.
        global_token = tokens[:, 0]
        context = tokens
        if self.training and token_dropout > 0:
            drop = torch.rand(context.shape[:2], device=context.device) < token_dropout
            drop[:, 0] = False
            context = torch.where(drop[..., None], self.mask_token.to(context.dtype), context)
        return global_token, context


class FaceAdjInContextCVAE(nn.Module):
    def __init__(
        self,
        max_faces=30,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        image_backbone="dinov2",
        depth_anything_v2_ckpt=None,
    ):
        super().__init__()
        self.max_faces = max_faces
        self.seq_len = base.sequence_length(max_faces)
        self.vocab_size = base.vocab_size(max_faces)
        self.d_model = d_model
        self.bos = base.bos_token(max_faces)

        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.register_buffer("positional_encoding", self._build_positional_encoding(self.seq_len, d_model))
        self.image_encoder = DINOImageContextEncoder(
            d_model=d_model,
            backbone=image_backbone,
            depth_anything_v2_ckpt=depth_anything_v2_ckpt,
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers, norm=nn.LayerNorm(d_model))
        self.posterior_hidden = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.posterior_mu = nn.Linear(d_model, d_model)
        self.posterior_logvar = nn.Linear(d_model, d_model)
        self.prior_hidden = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.prior_mu = nn.Linear(d_model, d_model)
        self.prior_logvar = nn.Linear(d_model, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers, norm=nn.LayerNorm(d_model))
        self.latent_memory_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.image_memory_type = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.latent_memory_type, std=0.02)
        nn.init.normal_(self.image_memory_type, std=0.02)
        self.output_layer = nn.Linear(d_model, self.vocab_size)

    @staticmethod
    def _build_positional_encoding(seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
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

    def encode_topology(self, tokens, token_mask):
        x = self.embedding(tokens) + self.positional_encoding[:, : tokens.shape[1]].to(tokens.device)
        memory = self.encoder(x, src_key_padding_mask=~token_mask)
        return (memory * token_mask.unsqueeze(-1)).sum(dim=1) / token_mask.sum(dim=1, keepdim=True).clamp_min(1)

    def encode_image(self, images, image_token_dropout=0.0):
        return self.image_encoder(images, token_dropout=image_token_dropout)

    def encode_posterior(self, tokens, token_mask, image_global):
        topo_global = self.encode_topology(tokens, token_mask)
        hidden = self.posterior_hidden(torch.cat([topo_global, image_global], dim=-1))
        return self.posterior_mu(hidden), self.posterior_logvar(hidden)

    def encode_prior(self, image_global):
        hidden = self.prior_hidden(image_global)
        return self.prior_mu(hidden), self.prior_logvar(hidden)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, decoder_input, decoder_mask, image_context):
        tgt = self.embedding(decoder_input) + self.positional_encoding[:, : decoder_input.shape[1]].to(decoder_input.device)
        latent_memory = z.unsqueeze(1) + self.latent_memory_type
        image_memory = image_context + self.image_memory_type
        memory = torch.cat([latent_memory, image_memory], dim=1)
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=self.causal_mask(decoder_input.shape[1], decoder_input.device),
            tgt_key_padding_mask=~decoder_mask,
        )
        return self.output_layer(output)

    def forward(
        self,
        tokens,
        token_mask,
        images,
        sample_posterior=True,
        decoder_input_override=None,
        decoder_mask_override=None,
        image_token_dropout=0.0,
    ):
        image_global, image_context = self.encode_image(images, image_token_dropout=image_token_dropout)
        q_mu, q_logvar = self.encode_posterior(tokens, token_mask, image_global)
        p_mu, p_logvar = self.encode_prior(image_global)
        z = self.reparameterize(q_mu, q_logvar) if sample_posterior else q_mu
        if decoder_input_override is None:
            decoder_input, decoder_mask = self.shifted_decoder_input(tokens, token_mask)
        else:
            decoder_input = decoder_input_override
            decoder_mask = decoder_mask_override
        logits = self.decode(z, decoder_input, decoder_mask, image_context)
        return logits, q_mu, q_logvar, p_mu, p_logvar, image_context

    @torch.no_grad()
    def generate(self, z, image_context, min_faces=1, greedy=True, temperature=1.0):
        batch_size = z.shape[0]
        device = z.device
        max_pairs = base.num_pair_tokens(self.max_faces)
        decoder_input = torch.full((batch_size, 1), self.bos, dtype=torch.long, device=device)
        decoder_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        generated = []

        logits = self.decode(z, decoder_input, decoder_mask, image_context)[:, -1]
        face_logits = logits[:, FACE_COUNT_OFFSET : FACE_COUNT_OFFSET + self.max_faces + 1]
        allowed_face = torch.full_like(face_logits, MASKED_LOGIT)
        allowed_face[:, min_faces : self.max_faces + 1] = face_logits[:, min_faces : self.max_faces + 1]
        face_cls = (
            allowed_face.argmax(dim=-1)
            if greedy
            else torch.distributions.Categorical(logits=allowed_face / max(temperature, 1e-6)).sample()
        )
        face_tok = face_cls + FACE_COUNT_OFFSET
        generated.append(face_tok)
        decoder_input = torch.cat([decoder_input, face_tok[:, None]], dim=1)
        decoder_mask = torch.cat([decoder_mask, torch.ones(batch_size, 1, dtype=torch.bool, device=device)], dim=1)
        counts = face_cls.clamp(min_faces, self.max_faces)

        logits = self.decode(z, decoder_input, decoder_mask, image_context)[:, -1]
        max_edges_per_sample = counts * (counts - 1) // 2
        edge_logits = logits[
            :,
            base.edge_count_offset(self.max_faces) : base.edge_count_offset(self.max_faces) + base.max_edge_count(self.max_faces) + 1,
        ]
        allowed_edge = torch.full_like(edge_logits, MASKED_LOGIT)
        for b in range(batch_size):
            allowed_edge[b, : int(max_edges_per_sample[b].item()) + 1] = edge_logits[b, : int(max_edges_per_sample[b].item()) + 1]
        edge_count_cls = (
            allowed_edge.argmax(dim=-1)
            if greedy
            else torch.distributions.Categorical(logits=allowed_edge / max(temperature, 1e-6)).sample()
        )
        edge_count_tok = edge_count_cls + base.edge_count_offset(self.max_faces)
        generated.append(edge_count_tok)
        decoder_input = torch.cat([decoder_input, edge_count_tok[:, None]], dim=1)
        decoder_mask = torch.cat([decoder_mask, torch.ones(batch_size, 1, dtype=torch.bool, device=device)], dim=1)

        rows, cols = base.build_pair_index(self.max_faces)
        rows = rows.to(device)
        cols = cols.to(device)
        generated_edge_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        for pair_idx in range(max_pairs):
            active = (rows[pair_idx] < counts) & (cols[pair_idx] < counts)
            next_tok = torch.full((batch_size,), PAD, dtype=torch.long, device=device)
            next_mask = active.clone()
            if active.any():
                logits = self.decode(z, decoder_input, decoder_mask, image_context)[:, -1]
                pair_logits = torch.stack([logits[:, NO_EDGE], logits[:, EDGE]], dim=-1)
                remaining_valid = (
                    (rows[pair_idx:][None, :] < counts[:, None])
                    & (cols[pair_idx:][None, :] < counts[:, None])
                ).sum(dim=1)
                remaining_needed = (edge_count_cls - generated_edge_counts).clamp_min(0)
                force_no_edge = active & (generated_edge_counts >= edge_count_cls)
                force_edge = active & (remaining_needed >= remaining_valid)
                edge_cls = (
                    pair_logits.argmax(dim=-1)
                    if greedy
                    else torch.distributions.Categorical(logits=pair_logits / max(temperature, 1e-6)).sample()
                )
                edge_cls = torch.where(force_no_edge, torch.zeros_like(edge_cls), edge_cls)
                edge_cls = torch.where(force_edge, torch.ones_like(edge_cls), edge_cls)
                edge_cls = torch.where(active, edge_cls, torch.zeros_like(edge_cls))
                next_tok[active] = torch.where(edge_cls[active].bool(), EDGE, NO_EDGE)
                generated_edge_counts += (active & edge_cls.bool()).long()
            generated.append(next_tok)
            decoder_input = torch.cat([decoder_input, next_tok[:, None]], dim=1)
            decoder_mask = torch.cat([decoder_mask, next_mask[:, None]], dim=1)

        return torch.stack(generated, dim=1)


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def safe_model_forward(model, tokens, token_mask, images, **kwargs):
    if isinstance(model, nn.DataParallel) and tokens.shape[0] < len(model.device_ids):
        return model.module(tokens=tokens, token_mask=token_mask, images=images, **kwargs)
    return model(tokens=tokens, token_mask=token_mask, images=images, **kwargs)


def gaussian_kl(q_mu, q_logvar, p_mu, p_logvar):
    q_var = q_logvar.exp()
    p_var = p_logvar.exp().clamp_min(1e-8)
    kl = 0.5 * (p_logvar - q_logvar + (q_var + (q_mu - p_mu).pow(2)) / p_var - 1.0)
    return kl.sum(dim=-1).mean()


def compute_loss(logits, q_mu, q_logvar, p_mu, p_logvar, pair_targets, pair_mask,
                 num_faces, edge_count, edge_pos_weight, face_count_loss_weight,
                 edge_count_loss_weight, kl_beta):
    max_faces = int((1 + math.sqrt(1 + 8 * pair_targets.shape[1])) / 2)
    face_logits = logits[:, 0, FACE_COUNT_OFFSET : FACE_COUNT_OFFSET + max_faces + 1]
    edge_count_logits = logits[
        :, 1, base.edge_count_offset(max_faces) : base.edge_count_offset(max_faces) + base.max_edge_count(max_faces) + 1
    ]
    face_count_loss = F.cross_entropy(face_logits, num_faces)
    edge_count_loss = F.cross_entropy(edge_count_logits, edge_count)
    edge_logit = logits[:, 2:, EDGE] - logits[:, 2:, NO_EDGE]
    bce = F.binary_cross_entropy_with_logits(
        edge_logit,
        pair_targets,
        pos_weight=torch.tensor(edge_pos_weight, device=logits.device),
        reduction="none",
    )
    pair_loss = (bce * pair_mask.float()).sum() / pair_mask.float().sum().clamp_min(1)
    kl = gaussian_kl(q_mu, q_logvar, p_mu, p_logvar)
    loss = face_count_loss_weight * face_count_loss + edge_count_loss_weight * edge_count_loss + pair_loss + kl_beta * kl
    return loss, {
        "loss": float(loss.detach().cpu()),
        "face_count_loss": float(face_count_loss.detach().cpu()),
        "edge_count_loss": float(edge_count_loss.detach().cpu()),
        "pair_loss": float(pair_loss.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "kl_beta": float(kl_beta),
    }


def structural_flags(adj, count):
    adj_np = adj.detach().cpu().numpy()
    count_np = count.detach().cpu().numpy().astype(np.int64)
    connected = []
    no_isolated = []
    valid = []
    for matrix, n in zip(adj_np, count_np):
        n = int(n)
        if n <= 1:
            connected.append(True)
            no_isolated.append(True)
            valid.append(True)
            continue
        sub = matrix[:n, :n] > 0.5
        degree = sub.sum(axis=1)
        no_iso_flag = bool(np.all(degree > 0))
        seen = {0}
        stack = [0]
        while stack:
            cur = stack.pop()
            for nxt in np.where(sub[cur])[0]:
                if int(nxt) not in seen:
                    seen.add(int(nxt))
                    stack.append(int(nxt))
        conn_flag = bool(len(seen) == n)
        connected.append(conn_flag)
        no_isolated.append(no_iso_flag)
        valid.append(conn_flag and no_iso_flag)
    device = adj.device
    return (
        torch.tensor(valid, dtype=torch.bool, device=device),
        torch.tensor(connected, dtype=torch.bool, device=device),
        torch.tensor(no_isolated, dtype=torch.bool, device=device),
    )


def sample_k_metrics(generated, target_pair_targets, target_pair_mask, target_num_faces, min_faces, prefix):
    # generated: [B, K, L]
    batch_size, sample_k, seq_len = generated.shape
    flat = generated.reshape(batch_size * sample_k, seq_len)
    max_faces = int((1 + math.sqrt(1 + 8 * target_pair_targets.shape[1])) / 2)
    pred_adj, pred_count, _ = base.target_to_adj(flat, max_faces=max_faces, min_faces=min_faces)
    rows, cols = base.build_pair_index(max_faces)
    rows = rows.to(flat.device)
    cols = cols.to(flat.device)
    pred_pairs = (pred_adj[:, rows, cols] > 0.5).reshape(batch_size, sample_k, -1)
    target_pairs = (target_pair_targets > 0.5)[:, None, :].expand_as(pred_pairs)
    mask = target_pair_mask.bool()[:, None, :].expand_as(pred_pairs)
    tp = ((pred_pairs & target_pairs) & mask).sum(dim=2).float()
    fp = ((pred_pairs & ~target_pairs) & mask).sum(dim=2).float()
    fn = ((~pred_pairs & target_pairs) & mask).sum(dim=2).float()
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    pair_equal = ((pred_pairs == target_pairs) | ~mask).all(dim=2)
    count_equal = pred_count.reshape(batch_size, sample_k) == target_num_faces[:, None]
    exact = pair_equal & count_equal
    valid, connected, no_isolated = structural_flags(pred_adj, pred_count)
    valid = valid.reshape(batch_size, sample_k)
    connected = connected.reshape(batch_size, sample_k)
    no_isolated = no_isolated.reshape(batch_size, sample_k)

    diversity = torch.zeros((), device=flat.device)
    if sample_k > 1:
        distances = []
        base_mask = target_pair_mask.bool()
        for i in range(sample_k):
            for j in range(i + 1, sample_k):
                diff = (pred_pairs[:, i] ^ pred_pairs[:, j]) & base_mask
                distances.append(diff.float().sum(dim=1) / base_mask.float().sum(dim=1).clamp_min(1))
        diversity = torch.stack(distances, dim=1).mean()

    return {
        f"{prefix}_valid_any": float(valid.any(dim=1).float().mean().item()),
        f"{prefix}_valid_mean": float(valid.float().mean().item()),
        f"{prefix}_connected_any": float(connected.any(dim=1).float().mean().item()),
        f"{prefix}_no_isolated_any": float(no_isolated.any(dim=1).float().mean().item()),
        f"{prefix}_best_edge_f1": float(f1.max(dim=1).values.mean().item()),
        f"{prefix}_mean_edge_f1": float(f1.mean().item()),
        f"{prefix}_best_exact": float(exact.any(dim=1).float().mean().item()),
        f"{prefix}_diversity": float(diversity.item()),
    }


@torch.no_grad()
def evaluate(model, dataloader, device, args, epoch):
    model.eval()
    core = unwrap_model(model)
    totals = Counter()
    tf_metric_totals = Counter()
    count = 0
    posterior_batches = []
    cond_prior_batches = []
    sample_k_batches = []
    generated_seen = 0
    for batch in tqdm(dataloader, desc="val", leave=False):
        tokens = batch["tokens"].to(device, non_blocking=True)
        token_mask = batch["token_mask"].to(device, non_blocking=True)
        pair_targets = batch["pair_targets"].to(device, non_blocking=True)
        pair_mask = batch["pair_mask"].to(device, non_blocking=True)
        num_faces = batch["num_faces"].to(device, non_blocking=True)
        edge_count = batch["edge_count"].to(device, non_blocking=True)
        images = batch["image"].to(device, non_blocking=True)

        logits, q_mu, q_logvar, p_mu, p_logvar, image_context = safe_model_forward(
            model, tokens, token_mask, images, sample_posterior=False
        )
        kl_beta = args.kl_beta * min(1.0, epoch / max(args.kl_warmup_epochs, 1))
        _, loss_parts = compute_loss(
            logits=logits,
            q_mu=q_mu,
            q_logvar=q_logvar,
            p_mu=p_mu,
            p_logvar=p_logvar,
            pair_targets=pair_targets,
            pair_mask=pair_mask,
            num_faces=num_faces,
            edge_count=edge_count,
            edge_pos_weight=args.edge_pos_weight,
            face_count_loss_weight=args.face_count_loss_weight,
            edge_count_loss_weight=args.edge_count_loss_weight,
            kl_beta=kl_beta,
        )
        batch_size = tokens.shape[0]
        for k, v in loss_parts.items():
            totals[k] += v * batch_size
        metrics = base.pair_metrics_from_logits(logits, pair_targets, pair_mask, num_faces, edge_count)
        for k, v in metrics.items():
            tf_metric_totals[k] += v * batch_size
        count += batch_size

        if generated_seen < args.eval_generate_limit:
            take = min(batch_size, args.eval_generate_limit - generated_seen)
            posterior_generated = core.generate(
                q_mu[:take], image_context[:take], min_faces=args.min_faces, greedy=True
            )
            cond_prior_generated = core.generate(
                p_mu[:take], image_context[:take], min_faces=args.min_faces, greedy=True
            )
            posterior_batches.append((
                posterior_generated, tokens[:take], pair_targets[:take], pair_mask[:take],
                num_faces[:take], edge_count[:take],
            ))
            cond_prior_batches.append((
                cond_prior_generated, tokens[:take], pair_targets[:take], pair_mask[:take],
                num_faces[:take], edge_count[:take],
            ))
            if args.eval_sample_k > 0:
                samples = []
                for _ in range(args.eval_sample_k):
                    z = core.reparameterize(p_mu[:take], p_logvar[:take])
                    samples.append(core.generate(
                        z,
                        image_context[:take],
                        min_faces=args.min_faces,
                        greedy=False,
                        temperature=args.prior_temperature,
                    ))
                sample_k_batches.append((
                    torch.stack(samples, dim=1),
                    pair_targets[:take],
                    pair_mask[:take],
                    num_faces[:take],
                ))
            generated_seen += take

    result = {k: v / max(count, 1) for k, v in totals.items()}
    result.update({k: v / max(count, 1) for k, v in tf_metric_totals.items()})
    if posterior_batches:
        merged = [torch.cat([b[i] for b in posterior_batches], dim=0) for i in range(6)]
        result.update(base.generated_metrics(*merged, min_faces=args.min_faces, prefix="ar_recon"))
    if cond_prior_batches:
        merged = [torch.cat([b[i] for b in cond_prior_batches], dim=0) for i in range(6)]
        result.update(base.generated_metrics(*merged, min_faces=args.min_faces, prefix="cond_prior_mu"))
    if sample_k_batches:
        generated = torch.cat([b[0] for b in sample_k_batches], dim=0)
        pair_targets = torch.cat([b[1] for b in sample_k_batches], dim=0)
        pair_mask = torch.cat([b[2] for b in sample_k_batches], dim=0)
        num_faces = torch.cat([b[3] for b in sample_k_batches], dim=0)
        result.update(sample_k_metrics(
            generated,
            pair_targets,
            pair_mask,
            num_faces,
            min_faces=args.min_faces,
            prefix=f"sample{args.eval_sample_k}",
        ))
    return result


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--condition-root", default=DEFAULT_CONDITION_ROOT)
    parser.add_argument("--train-list", default=DEFAULT_TRAIN_LIST)
    parser.add_argument("--val-list", default=DEFAULT_VAL_LIST)
    parser.add_argument("--test-list", default=DEFAULT_TEST_LIST)
    parser.add_argument("--output-dir", default="experiments/2026-06-11/outputs_cvae_incontext_corrupt")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--init-vae-checkpoint", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--max-faces", type=int, default=30)
    parser.add_argument("--min-faces", type=int, default=7)
    parser.add_argument("--order-mode", choices=["none", "degree", "wl"], default="wl")
    parser.add_argument("--wl-rounds", type=int, default=3)
    parser.add_argument("--image-source", choices=["real_flux", "real_flux_masked", "real_blender", "svr", "sketch"], default="real_flux")
    parser.add_argument("--image-rotation-id", type=int, default=0)
    parser.add_argument("--image-backbone", choices=["dinov2", "depth_anything_v2"], default="dinov2")
    parser.add_argument("--depth-anything-v2-ckpt", default=None)
    parser.add_argument("--image-token-dropout", type=float, default=0.0)
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
    parser.add_argument("--eval-generate-limit", type=int, default=256)
    parser.add_argument("--eval-sample-k", type=int, default=1)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--no-data-parallel", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--best-metric", default="cond_prior_mu_edge_f1")
    parser.add_argument("--corrupt-prob", type=float, default=0.15)
    parser.add_argument("--corrupt-warmup-epochs", type=int, default=10)
    return parser.parse_args()


def load_checkpoint_args(args):
    if not args.checkpoint:
        return args, None
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    for key in (
        "max_faces", "min_faces", "order_mode", "wl_rounds",
        "image_source", "image_rotation_id", "image_backbone", "depth_anything_v2_ckpt",
        "d_model", "nhead", "encoder_layers", "decoder_layers",
        "dim_feedforward", "dropout", "kl_beta", "kl_warmup_epochs",
        "edge_pos_weight", "face_count_loss_weight", "edge_count_loss_weight",
        "corrupt_prob", "corrupt_warmup_epochs",
    ):
        if key in ckpt_args:
            setattr(args, key, ckpt_args[key])
    return args, checkpoint


def build_model(args, device):
    return FaceAdjInContextCVAE(
        max_faces=args.max_faces,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        image_backbone=args.image_backbone,
        depth_anything_v2_ckpt=args.depth_anything_v2_ckpt,
    ).to(device)


def load_vae_initialization(model, checkpoint_path):
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint["model"]
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and tuple(target[key].shape) == tuple(value.shape)
    }
    model.load_state_dict(compatible, strict=False)
    print(f"Initialized {len(compatible)} tensors from {checkpoint_path}", flush=True)


def build_dataset(args, split):
    if split == "test":
        model_list = args.test_list
        max_samples = args.max_test_samples
    elif split == "val":
        model_list = args.val_list
        max_samples = args.max_val_samples
    else:
        raise ValueError(f"Invalid split: {split}")
    return FaceAdjImageDataset(
        raw_root=args.raw_root,
        condition_root=args.condition_root,
        model_list=model_list,
        max_faces=args.max_faces,
        order_mode=args.order_mode,
        wl_rounds=args.wl_rounds,
        image_source=args.image_source,
        image_rotation_id=args.image_rotation_id,
        max_samples=max_samples,
    )


def build_loader(dataset, args, shuffle):
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
        f"{args.eval_split} loss={metrics.get('loss', 0.0):.4f} "
        f"tf_f1={metrics.get('tf_edge_f1', 0.0):.4f} "
        f"ar_f1={metrics.get('ar_recon_edge_f1', 0.0):.4f} "
        f"ar_valid={metrics.get('ar_recon_valid_strict_ratio', 0.0):.4f} "
        f"cond_f1={metrics.get('cond_prior_mu_edge_f1', 0.0):.4f} "
        f"cond_valid={metrics.get('cond_prior_mu_valid_strict_ratio', 0.0):.4f}",
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

    train_dataset = FaceAdjImageDataset(
        raw_root=args.raw_root,
        condition_root=args.condition_root,
        model_list=args.train_list,
        max_faces=args.max_faces,
        order_mode=args.order_mode,
        wl_rounds=args.wl_rounds,
        image_source=args.image_source,
        image_rotation_id=args.image_rotation_id,
        max_samples=args.max_train_samples,
    )
    val_dataset = build_dataset(args, "val")
    train_loader = build_loader(train_dataset, args, shuffle=True)
    val_loader = build_loader(val_dataset, args, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    load_vae_initialization(model, args.init_vae_checkpoint)
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
        f"image_source={args.image_source}; image_backbone={args.image_backbone}; "
        f"corrupt_prob={args.corrupt_prob}; "
        f"image_token_dropout={args.image_token_dropout}",
        flush=True,
    )

    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        kl_beta = args.kl_beta * min(1.0, epoch / max(args.kl_warmup_epochs, 1))
        cur_corrupt = args.corrupt_prob * min(1.0, epoch / max(args.corrupt_warmup_epochs, 1))
        progress = tqdm(train_loader, desc=f"epoch {epoch:03d}/{args.epochs} train")
        running = Counter()
        seen = 0
        core = unwrap_model(model)
        for batch in progress:
            tokens = batch["tokens"].to(device, non_blocking=True)
            token_mask = batch["token_mask"].to(device, non_blocking=True)
            pair_targets = batch["pair_targets"].to(device, non_blocking=True)
            pair_mask = batch["pair_mask"].to(device, non_blocking=True)
            num_faces = batch["num_faces"].to(device, non_blocking=True)
            edge_count = batch["edge_count"].to(device, non_blocking=True)
            images = batch["image"].to(device, non_blocking=True)

            decoder_input, decoder_mask = core.shifted_decoder_input(tokens, token_mask)
            decoder_input = base.corrupt_pair_prefix(decoder_input, decoder_mask, cur_corrupt)
            logits, q_mu, q_logvar, p_mu, p_logvar, _ = safe_model_forward(
                model,
                tokens,
                token_mask,
                images,
                sample_posterior=True,
                decoder_input_override=decoder_input,
                decoder_mask_override=decoder_mask,
                image_token_dropout=args.image_token_dropout,
            )
            loss, loss_parts = compute_loss(
                logits=logits,
                q_mu=q_mu,
                q_logvar=q_logvar,
                p_mu=p_mu,
                p_logvar=p_logvar,
                pair_targets=pair_targets,
                pair_mask=pair_mask,
                num_faces=num_faces,
                edge_count=edge_count,
                edge_pos_weight=args.edge_pos_weight,
                face_count_loss_weight=args.face_count_loss_weight,
                edge_count_loss_weight=args.edge_count_loss_weight,
                kl_beta=kl_beta,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            batch_size = tokens.shape[0]
            seen += batch_size
            for k, v in loss_parts.items():
                running[k] += v * batch_size
            progress.set_postfix({
                "loss": f"{running['loss'] / max(seen, 1):.4f}",
                "pair": f"{running['pair_loss'] / max(seen, 1):.4f}",
                "kl": f"{running['kl'] / max(seen, 1):.4f}",
                "corrupt": f"{cur_corrupt:.2f}",
            })

        train_metrics = {f"train_{k}": v / max(seen, 1) for k, v in running.items()}
        val_metrics = evaluate(model, val_loader, device, args, epoch)
        epoch_metrics = {"epoch": epoch, "corrupt_prob_used": cur_corrupt, **train_metrics, **val_metrics}
        history.append(epoch_metrics)
        save_json(output_dir / "metrics.json", epoch_metrics)
        save_json(output_dir / "history.json", {"history": history})

        score = val_metrics.get(args.best_metric, val_metrics.get("cond_prior_mu_edge_f1", val_metrics.get("ar_recon_edge_f1", 0.0)))
        sample_summary = ""
        if args.eval_sample_k > 0:
            sample_summary = f" sample{args.eval_sample_k}_valid={val_metrics.get(f'sample{args.eval_sample_k}_valid_any', 0.0):.4f}"
        print(
            f"epoch {epoch:03d} train_loss={train_metrics['train_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"tf_f1={val_metrics.get('tf_edge_f1', 0.0):.4f} "
            f"ar_f1={val_metrics.get('ar_recon_edge_f1', 0.0):.4f} "
            f"ar_valid={val_metrics.get('ar_recon_valid_strict_ratio', 0.0):.4f} "
            f"cond_f1={val_metrics.get('cond_prior_mu_edge_f1', 0.0):.4f} "
            f"cond_valid={val_metrics.get('cond_prior_mu_valid_strict_ratio', 0.0):.4f} "
            f"{sample_summary} "
            f"corrupt={cur_corrupt:.2f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            torch.save(
                {"model": unwrap_model(model).state_dict(), "args": vars(args), "metrics": epoch_metrics},
                output_dir / "best.pt",
            )
            print(f"saved best checkpoint to {output_dir / 'best.pt'} by {args.best_metric}={score:.4f}", flush=True)


if __name__ == "__main__":
    main()
