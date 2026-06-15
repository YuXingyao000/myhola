#!/usr/bin/env python3
"""Train a zero-init gated image adapter for a frozen topology VAE decoder.

This experiment is a conservative follow-up to the 2026-06-11 in-context CVAE
and the 2026-06-13 frozen-prior negative result. It preserves the 2026-06-08
topology decoder path:

  decoder prefix -> cross-attn to z

and adds a separate image-conditioned logit residual:

  logits = frozen_decoder_logits + gate * zero_init_linear(CrossAttn(prefix_emb, image_tokens))

The adapter is initialized as an exact no-op, so the model starts equivalent to
the frozen 2026-06-08 topology VAE decoder and learns image control gradually.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

# PyTorch's fused CUDA SDPA kernels can hit a low-level misaligned-address
# failure when this experiment wraps the frozen TransformerDecoder in
# DataParallel. The topology sequences are short enough that the math kernel is
# the safer default for this one-off experiment.
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREV_SCRIPT = Path(__file__).resolve().parents[1] / "2026-06-11" / "topology_faceadj_cvae_incontext_corrupt.py"
SPEC = importlib.util.spec_from_file_location("topology_faceadj_cvae_0611", PREV_SCRIPT)
prev = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prev)
base = prev.base

DEFAULT_RAW_ROOT = "/mnt/d/data/deepcad_v7"
DEFAULT_CONDITION_ROOT = "/mnt/d/data/deepcad_v7_cond"
DEFAULT_TRAIN_LIST = "src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt"
DEFAULT_VAL_LIST = "src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt"
DEFAULT_TEST_LIST = "src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt"
DEFAULT_TOPOLOGY_CHECKPOINT = "experiments/2026-06-08/outputs_corrupt_only/best.pt"


class FrozenTopologyGatedImageAdapter(nn.Module):
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
        self.d_model = d_model
        self.topology = base.FaceAdjTransformerVAE(
            max_faces=max_faces,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.image_encoder = prev.DINOImageContextEncoder(
            d_model=d_model,
            backbone=image_backbone,
            depth_anything_v2_ckpt=depth_anything_v2_ckpt,
        )
        self.prior_hidden = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.prior_mu = nn.Linear(d_model, d_model)
        self.prior_logvar = nn.Linear(d_model, d_model)
        self.image_adapter_norm = nn.LayerNorm(d_model)
        self.image_adapter_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.image_adapter_out = nn.Linear(d_model, self.topology.vocab_size)
        nn.init.zeros_(self.image_adapter_out.weight)
        nn.init.zeros_(self.image_adapter_out.bias)
        # The branch is still a no-op at init because image_adapter_out is zero.
        # A nonzero scale keeps gradients flowing into the zero-init projection.
        self.image_gate = nn.Parameter(torch.tensor(1.0))

    def load_topology_checkpoint(self, checkpoint_path: str, train_topology: bool = False):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.topology.load_state_dict(checkpoint["model"], strict=True)
        if not train_topology:
            for param in self.topology.parameters():
                param.requires_grad = False
            self.topology.eval()
            print(f"Loaded frozen topology VAE from {checkpoint_path}", flush=True)
        else:
            print(f"Loaded trainable topology VAE from {checkpoint_path}", flush=True)

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(param.requires_grad for param in self.topology.parameters()):
            self.topology.eval()
        return self

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode_image_prior(self, images, image_token_dropout=0.0):
        image_global, _ = self.image_encoder(images, token_dropout=image_token_dropout)
        hidden = self.prior_hidden(image_global)
        return self.prior_mu(hidden), self.prior_logvar(hidden)

    def encode_topology_posterior(self, tokens, token_mask):
        with torch.no_grad():
            q_mu, q_logvar = self.topology.encode(tokens, token_mask)
        return q_mu.detach(), q_logvar.detach()

    def shifted_decoder_input(self, tokens, token_mask):
        return self.topology.shifted_decoder_input(tokens, token_mask)

    def encode_image(self, images, image_token_dropout=0.0):
        return self.image_encoder(images, token_dropout=image_token_dropout)

    def base_decode_logits(self, z, decoder_input, decoder_mask):
        tgt = (
            self.topology.embedding(decoder_input)
            + self.topology.positional_encoding[:, : decoder_input.shape[1]].to(decoder_input.device)
        )
        output = self.topology.decoder(
            tgt=tgt,
            memory=z.unsqueeze(1),
            tgt_mask=self.topology.causal_mask(decoder_input.shape[1], decoder_input.device),
            tgt_key_padding_mask=~decoder_mask,
            tgt_is_causal=True,
        )
        return self.topology.output_layer(output)

    def decode(self, z, decoder_input, decoder_mask, image_context):
        base_logits = self.base_decode_logits(z, decoder_input, decoder_mask)
        tgt = (
            self.topology.embedding(decoder_input)
            + self.topology.positional_encoding[:, : decoder_input.shape[1]].to(decoder_input.device)
        )
        query = self.image_adapter_norm(tgt)
        delta, _ = self.image_adapter_attn(
            query=query,
            key=image_context,
            value=image_context,
            need_weights=False,
        )
        delta = self.image_adapter_out(delta)
        return base_logits + self.image_gate * delta

    def forward(
        self,
        tokens,
        token_mask,
        images,
        sample_prior=False,
        decoder_input_override=None,
        decoder_mask_override=None,
        image_token_dropout=0.0,
        compute_posterior_logits=True,
    ):
        image_global, image_context = self.encode_image(images, image_token_dropout=image_token_dropout)
        q_mu, q_logvar = self.encode_topology_posterior(tokens, token_mask)
        hidden = self.prior_hidden(image_global)
        p_mu, p_logvar = self.prior_mu(hidden), self.prior_logvar(hidden)
        z = self.reparameterize(p_mu, p_logvar) if sample_prior else p_mu
        if decoder_input_override is None:
            decoder_input, decoder_mask = self.shifted_decoder_input(tokens, token_mask)
        else:
            decoder_input = decoder_input_override
            decoder_mask = decoder_mask_override
        prior_logits = self.decode(z, decoder_input, decoder_mask, image_context)
        posterior_logits = None
        if compute_posterior_logits:
            posterior_logits = self.decode(q_mu, decoder_input, decoder_mask, image_context)
        return prior_logits, posterior_logits, q_mu, q_logvar, p_mu, p_logvar, image_context

    @torch.no_grad()
    def generate(self, z, image_context, min_faces=1, greedy=True, temperature=1.0):
        batch_size = z.shape[0]
        device = z.device
        max_pairs = base.num_pair_tokens(self.max_faces)
        decoder_input = torch.full((batch_size, 1), self.topology.bos, dtype=torch.long, device=device)
        decoder_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        generated = []

        logits = self.decode(z, decoder_input, decoder_mask, image_context)[:, -1]
        face_logits = logits[:, base.FACE_COUNT_OFFSET : base.FACE_COUNT_OFFSET + self.max_faces + 1]
        allowed_face = torch.full_like(face_logits, base.MASKED_LOGIT)
        allowed_face[:, min_faces : self.max_faces + 1] = face_logits[:, min_faces : self.max_faces + 1]
        face_cls = (
            allowed_face.argmax(dim=-1)
            if greedy
            else torch.distributions.Categorical(logits=allowed_face / max(temperature, 1e-6)).sample()
        )
        face_tok = face_cls + base.FACE_COUNT_OFFSET
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
        allowed_edge = torch.full_like(edge_logits, base.MASKED_LOGIT)
        for b in range(batch_size):
            allowed_edge[b, : int(max_edges_per_sample[b].item()) + 1] = edge_logits[
                b, : int(max_edges_per_sample[b].item()) + 1
            ]
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
            next_tok = torch.full((batch_size,), base.PAD, dtype=torch.long, device=device)
            next_mask = active.clone()
            if active.any():
                logits = self.decode(z, decoder_input, decoder_mask, image_context)[:, -1]
                pair_logits = torch.stack([logits[:, base.NO_EDGE], logits[:, base.EDGE]], dim=-1)
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
                next_tok[active] = torch.where(edge_cls[active].bool(), base.EDGE, base.NO_EDGE)
                generated_edge_counts += (active & edge_cls.bool()).long()
            generated.append(next_tok)
            decoder_input = torch.cat([decoder_input, next_tok[:, None]], dim=1)
            decoder_mask = torch.cat([decoder_mask, next_mask[:, None]], dim=1)

        return torch.stack(generated, dim=1)

    @torch.no_grad()
    def generate_posterior_mu(self, tokens, token_mask, images, min_faces=1):
        q_mu, _ = self.encode_topology_posterior(tokens, token_mask)
        _, image_context = self.encode_image(images, image_token_dropout=0.0)
        return self.generate(q_mu, image_context, min_faces=min_faces, greedy=True)

    @torch.no_grad()
    def generate_prior_mu(self, images, min_faces=1):
        image_global, image_context = self.encode_image(images, image_token_dropout=0.0)
        hidden = self.prior_hidden(image_global)
        p_mu = self.prior_mu(hidden)
        return self.generate(p_mu, image_context, min_faces=min_faces, greedy=True)

    @torch.no_grad()
    def generate_prior_samples(self, images, sample_k, min_faces=1, temperature=1.0):
        image_global, image_context = self.encode_image(images, image_token_dropout=0.0)
        hidden = self.prior_hidden(image_global)
        p_mu, p_logvar = self.prior_mu(hidden), self.prior_logvar(hidden)
        samples = []
        for _ in range(sample_k):
            z = self.reparameterize(p_mu, p_logvar)
            generated = self.generate(
                z,
                image_context,
                min_faces=min_faces,
                greedy=False,
                temperature=temperature,
            )
            samples.append(generated)
        return torch.stack(samples, dim=1)


def unwrap_model(model):
    return model.module if isinstance(model, (nn.DataParallel, DDP)) else model


def safe_model_forward(model, tokens, token_mask, images, **kwargs):
    if isinstance(model, nn.DataParallel) and tokens.shape[0] < len(model.device_ids):
        return model.module(tokens=tokens, token_mask=token_mask, images=images, **kwargs)
    return model(tokens=tokens, token_mask=token_mask, images=images, **kwargs)


def setup_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def reduce_train_metrics(running, seen, device):
    keys = sorted(running.keys())
    values = torch.tensor([running[k] for k in keys] + [seen], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total_seen = max(float(values[-1].item()), 1.0)
    return {f"train_{key}": float(values[i].item() / total_seen) for i, key in enumerate(keys)}


def token_reconstruction_loss(logits, pair_targets, pair_mask, num_faces, edge_count, args):
    max_faces = int((1 + math.sqrt(1 + 8 * pair_targets.shape[1])) / 2)
    face_logits = logits[:, 0, base.FACE_COUNT_OFFSET : base.FACE_COUNT_OFFSET + max_faces + 1]
    edge_count_logits = logits[
        :, 1, base.edge_count_offset(max_faces) : base.edge_count_offset(max_faces) + base.max_edge_count(max_faces) + 1
    ]
    face_count_loss = F.cross_entropy(face_logits, num_faces)
    edge_count_loss = F.cross_entropy(edge_count_logits, edge_count)
    edge_logit = logits[:, 2:, base.EDGE] - logits[:, 2:, base.NO_EDGE]
    bce = F.binary_cross_entropy_with_logits(
        edge_logit,
        pair_targets,
        pos_weight=torch.tensor(args.edge_pos_weight, device=logits.device),
        reduction="none",
    )
    pair_loss = (bce * pair_mask.float()).sum() / pair_mask.float().sum().clamp_min(1)
    loss = (
        args.face_count_loss_weight * face_count_loss
        + args.edge_count_loss_weight * edge_count_loss
        + pair_loss
    )
    return loss, {
        "face_count_loss": float(face_count_loss.detach().cpu()),
        "edge_count_loss": float(edge_count_loss.detach().cpu()),
        "pair_loss": float(pair_loss.detach().cpu()),
        "token_loss": float(loss.detach().cpu()),
    }


def compute_adapter_loss(prior_logits, posterior_logits, q_mu, q_logvar, p_mu, p_logvar,
                         pair_targets, pair_mask, num_faces, edge_count, args, kl_beta):
    prior_loss, prior_parts = token_reconstruction_loss(
        prior_logits, pair_targets, pair_mask, num_faces, edge_count, args
    )
    if posterior_logits is None or args.posterior_loss_weight <= 0:
        posterior_loss = prior_loss.new_zeros(())
        posterior_parts = {
            "token_loss": 0.0,
            "pair_loss": 0.0,
            "face_count_loss": 0.0,
            "edge_count_loss": 0.0,
        }
    else:
        posterior_loss, posterior_parts = token_reconstruction_loss(
            posterior_logits, pair_targets, pair_mask, num_faces, edge_count, args
        )
    kl = prev.gaussian_kl(q_mu, q_logvar, p_mu, p_logvar)
    latent_mse = F.mse_loss(p_mu, q_mu)
    loss = (
        args.prior_loss_weight * prior_loss
        + args.posterior_loss_weight * posterior_loss
        + kl_beta * kl
        + args.latent_mse_weight * latent_mse
    )
    parts = {
        "loss": float(loss.detach().cpu()),
        "prior_token_loss": float(prior_loss.detach().cpu()),
        "posterior_token_loss": float(posterior_loss.detach().cpu()),
        "prior_pair_loss": prior_parts["pair_loss"],
        "posterior_pair_loss": posterior_parts["pair_loss"],
        "prior_face_count_loss": prior_parts["face_count_loss"],
        "posterior_face_count_loss": posterior_parts["face_count_loss"],
        "prior_edge_count_loss": prior_parts["edge_count_loss"],
        "posterior_edge_count_loss": posterior_parts["edge_count_loss"],
        "kl": float(kl.detach().cpu()),
        "kl_beta": float(kl_beta),
    }
    parts["latent_mse"] = float(latent_mse.detach().cpu())
    return loss, parts


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

        prior_logits, posterior_logits, q_mu, q_logvar, p_mu, p_logvar, image_context = safe_model_forward(
            model,
            tokens,
            token_mask,
            images,
            sample_prior=False,
            compute_posterior_logits=args.posterior_loss_weight > 0,
        )
        kl_beta = args.kl_beta * min(1.0, epoch / max(args.kl_warmup_epochs, 1))
        _, loss_parts = compute_adapter_loss(
            prior_logits, posterior_logits, q_mu, q_logvar, p_mu, p_logvar,
            pair_targets, pair_mask, num_faces, edge_count, args, kl_beta,
        )
        batch_size = tokens.shape[0]
        for k, v in loss_parts.items():
            totals[k] += v * batch_size
        metrics = base.pair_metrics_from_logits(prior_logits, pair_targets, pair_mask, num_faces, edge_count)
        for k, v in metrics.items():
            tf_metric_totals[k] += v * batch_size
        count += batch_size

        if generated_seen < args.eval_generate_limit:
            take = min(batch_size, args.eval_generate_limit - generated_seen)
            posterior_generated = core.generate_posterior_mu(
                tokens[:take],
                token_mask[:take],
                images[:take],
                min_faces=args.min_faces,
            )
            cond_prior_generated = core.generate_prior_mu(images[:take], min_faces=args.min_faces)
            posterior_batches.append((
                posterior_generated, tokens[:take], pair_targets[:take], pair_mask[:take],
                num_faces[:take], edge_count[:take],
            ))
            cond_prior_batches.append((
                cond_prior_generated, tokens[:take], pair_targets[:take], pair_mask[:take],
                num_faces[:take], edge_count[:take],
            ))
            if args.eval_sample_k > 0:
                sample_generated = core.generate_prior_samples(
                    images[:take],
                    sample_k=args.eval_sample_k,
                    min_faces=args.min_faces,
                    temperature=args.prior_temperature,
                )
                sample_k_batches.append((
                    sample_generated, pair_targets[:take], pair_mask[:take], num_faces[:take],
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
        result.update(prev.sample_k_metrics(
            generated,
            target_pair_targets=pair_targets,
            target_pair_mask=pair_mask,
            target_num_faces=num_faces,
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
    parser.add_argument("--output-dir", default="experiments/2026-06-14/outputs_cvae_gated_adapter")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--topology-checkpoint", default=DEFAULT_TOPOLOGY_CHECKPOINT)
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
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--latent-mse-weight", type=float, default=0.1)
    parser.add_argument("--prior-loss-weight", type=float, default=1.0)
    parser.add_argument("--posterior-loss-weight", type=float, default=1.0)
    parser.add_argument("--edge-pos-weight", type=float, default=1.0)
    parser.add_argument("--face-count-loss-weight", type=float, default=1.0)
    parser.add_argument("--edge-count-loss-weight", type=float, default=0.2)
    parser.add_argument("--eval-generate-limit", type=int, default=256)
    parser.add_argument("--eval-sample-k", type=int, default=0)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--no-data-parallel", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--best-metric", default="cond_prior_mu_edge_f1")
    parser.add_argument("--corrupt-prob", type=float, default=0.0)
    parser.add_argument("--corrupt-warmup-epochs", type=int, default=1)
    parser.add_argument("--train-topology", action="store_true")
    return parser.parse_args()


def load_checkpoint_args(args):
    if not args.checkpoint:
        return args, None
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    for key in (
        "topology_checkpoint", "max_faces", "min_faces", "order_mode", "wl_rounds",
        "image_source", "image_rotation_id", "image_backbone", "depth_anything_v2_ckpt",
        "d_model", "nhead", "encoder_layers", "decoder_layers",
        "dim_feedforward", "dropout", "kl_beta", "kl_warmup_epochs", "latent_mse_weight",
        "prior_loss_weight", "posterior_loss_weight", "train_topology",
        "edge_pos_weight", "face_count_loss_weight", "edge_count_loss_weight",
        "corrupt_prob", "corrupt_warmup_epochs",
    ):
        if key in ckpt_args:
            setattr(args, key, ckpt_args[key])
    return args, checkpoint


def build_model(args, device):
    model = FrozenTopologyGatedImageAdapter(
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
    model.load_topology_checkpoint(args.topology_checkpoint, train_topology=args.train_topology)
    return model


def build_dataset(args, split):
    if split == "test":
        model_list = args.test_list
        max_samples = args.max_test_samples
    elif split == "val":
        model_list = args.val_list
        max_samples = args.max_val_samples
    elif split == "train":
        model_list = args.train_list
        max_samples = args.max_train_samples
    else:
        raise ValueError(f"Invalid split: {split}")
    return prev.FaceAdjImageDataset(
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


def build_loader(dataset, args, shuffle, sampler=None):
    batch_size = args.batch_size
    if dist.is_available() and dist.is_initialized():
        batch_size = max(1, args.batch_size // dist.get_world_size())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def trainable_state_dict(model):
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key in trainable
    }


def run_eval_only(args, checkpoint):
    if checkpoint is None:
        raise ValueError("--eval-only requires --checkpoint")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args, args.eval_split)
    loader = build_loader(dataset, args, shuffle=False)
    model = build_model(args, device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    print(f"Loaded trainable checkpoint; missing={len(missing)} unexpected={len(unexpected)}", flush=True)
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
        f"cond_f1={metrics.get('cond_prior_mu_edge_f1', 0.0):.4f} "
        f"cond_valid={metrics.get('cond_prior_mu_valid_strict_ratio', 0.0):.4f}",
        flush=True,
    )


def main():
    distributed, rank, world_size, local_rank = setup_distributed()
    args = parse_args()
    args, checkpoint = load_checkpoint_args(args)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / "args.json", vars(args))
    if distributed:
        dist.barrier()
    if args.eval_only:
        run_eval_only(args, checkpoint)
        cleanup_distributed()
        return

    train_dataset = build_dataset(args, "train")
    val_dataset = build_dataset(args, "val") if is_main_process() else None
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
    )
    train_loader = build_loader(train_dataset, args, shuffle=True, sampler=train_sampler)
    val_loader = build_loader(val_dataset, args, shuffle=False) if is_main_process() else None

    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    if distributed:
        if is_main_process():
            print(f"Using DDP on {world_size} GPUs", flush=True)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    elif torch.cuda.device_count() > 1 and not args.no_data_parallel:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if is_main_process():
        per_rank_batch = max(1, args.batch_size // world_size) if distributed else args.batch_size
        print(
            f"Model ready on {device}; trainable parameters={sum(p.numel() for p in params):,}; "
            f"train={len(train_dataset)} val={len(val_dataset)}; "
            f"global_batch={args.batch_size} per_rank_batch={per_rank_batch}; "
            f"topology_checkpoint={args.topology_checkpoint}; "
            f"image_source={args.image_source}; image_backbone={args.image_backbone}; "
            f"corrupt_prob={args.corrupt_prob}",
            flush=True,
        )

    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        kl_beta = args.kl_beta * min(1.0, epoch / max(args.kl_warmup_epochs, 1))
        cur_corrupt = args.corrupt_prob * min(1.0, epoch / max(args.corrupt_warmup_epochs, 1))
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch:03d}/{args.epochs} train",
            disable=not is_main_process(),
        )
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
            prior_logits, posterior_logits, q_mu, q_logvar, p_mu, p_logvar, _ = safe_model_forward(
                model,
                tokens,
                token_mask,
                images,
                sample_prior=False,
                decoder_input_override=decoder_input,
                decoder_mask_override=decoder_mask,
                image_token_dropout=args.image_token_dropout,
                compute_posterior_logits=args.posterior_loss_weight > 0,
            )
            loss, loss_parts = compute_adapter_loss(
                prior_logits,
                posterior_logits,
                q_mu,
                q_logvar,
                p_mu,
                p_logvar,
                pair_targets, pair_mask, num_faces, edge_count, args, kl_beta,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            batch_size = tokens.shape[0]
            seen += batch_size
            for k, v in loss_parts.items():
                running[k] += v * batch_size
            if is_main_process():
                progress.set_postfix({
                    "loss": f"{running['loss'] / max(seen, 1):.4f}",
                    "prior_pair": f"{running['prior_pair_loss'] / max(seen, 1):.4f}",
                    "post_pair": f"{running['posterior_pair_loss'] / max(seen, 1):.4f}",
                    "kl": f"{running['kl'] / max(seen, 1):.4f}",
                    "mse": f"{running['latent_mse'] / max(seen, 1):.4f}",
                })

        train_metrics = reduce_train_metrics(running, seen, device)
        if is_main_process():
            val_metrics = evaluate(unwrap_model(model), val_loader, device, args, epoch)
            epoch_metrics = {"epoch": epoch, "corrupt_prob_used": cur_corrupt, **train_metrics, **val_metrics}
            history.append(epoch_metrics)
            save_json(output_dir / "metrics.json", epoch_metrics)
            save_json(output_dir / "history.json", {"history": history})

            score = val_metrics.get(args.best_metric, val_metrics.get("cond_prior_mu_edge_f1", 0.0))
            sample_summary = ""
            if args.eval_sample_k > 0:
                sample_summary = f" sample{args.eval_sample_k}_valid={val_metrics.get(f'sample{args.eval_sample_k}_valid_any', 0.0):.4f}"
            print(
                f"epoch {epoch:03d} train_loss={train_metrics['train_loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"tf_f1={val_metrics.get('tf_edge_f1', 0.0):.4f} "
                f"ar_f1={val_metrics.get('ar_recon_edge_f1', 0.0):.4f} "
                f"cond_f1={val_metrics.get('cond_prior_mu_edge_f1', 0.0):.4f} "
                f"cond_valid={val_metrics.get('cond_prior_mu_valid_strict_ratio', 0.0):.4f} "
                f"{sample_summary}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                core = unwrap_model(model)
                torch.save(
                    {
                        "model": trainable_state_dict(core),
                        "args": vars(args),
                        "metrics": epoch_metrics,
                        "topology_checkpoint": args.topology_checkpoint,
                        "state_format": "trainable_only",
                    },
                    output_dir / "best.pt",
                )
                print(f"saved best trainable checkpoint to {output_dir / 'best.pt'}", flush=True)
        if distributed:
            dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
