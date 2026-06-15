#!/usr/bin/env python3
"""Train an image-conditioned prior for a frozen topology VAE decoder.

This experiment is a conservative follow-up to the 2026-06-11 in-context CVAE.
Instead of changing the topology decoder memory to include image tokens, it
loads the 2026-06-08 corruption-only topology VAE, freezes its topology encoder
and decoder, and trains only:

  image -> p(z_topology | image)

The goal is to test whether the weak 06-11 reconstruction was caused by
disturbing the strong topology decoder, not by the image signal itself.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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


class FrozenTopologyImagePrior(nn.Module):
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

    def load_topology_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.topology.load_state_dict(checkpoint["model"], strict=True)
        for param in self.topology.parameters():
            param.requires_grad = False
        self.topology.eval()
        print(f"Loaded frozen topology VAE from {checkpoint_path}", flush=True)

    def train(self, mode: bool = True):
        super().train(mode)
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

    def decode(self, z, decoder_input, decoder_mask):
        return self.topology.decode(z, decoder_input, decoder_mask)

    def forward(
        self,
        tokens,
        token_mask,
        images,
        sample_prior=False,
        decoder_input_override=None,
        decoder_mask_override=None,
        image_token_dropout=0.0,
    ):
        q_mu, q_logvar = self.encode_topology_posterior(tokens, token_mask)
        p_mu, p_logvar = self.encode_image_prior(images, image_token_dropout=image_token_dropout)
        z = self.reparameterize(p_mu, p_logvar) if sample_prior else p_mu
        if decoder_input_override is None:
            decoder_input, decoder_mask = self.shifted_decoder_input(tokens, token_mask)
        else:
            decoder_input = decoder_input_override
            decoder_mask = decoder_mask_override
        logits = self.decode(z, decoder_input, decoder_mask)
        return logits, q_mu, q_logvar, p_mu, p_logvar

    @torch.no_grad()
    def generate_posterior_mu(self, tokens, token_mask, min_faces=1):
        q_mu, _ = self.encode_topology_posterior(tokens, token_mask)
        return self.topology.generate(q_mu, min_faces=min_faces, greedy=True)

    @torch.no_grad()
    def generate_prior_mu(self, images, min_faces=1):
        p_mu, _ = self.encode_image_prior(images, image_token_dropout=0.0)
        return self.topology.generate(p_mu, min_faces=min_faces, greedy=True)

    @torch.no_grad()
    def generate_prior_samples(self, images, sample_k, min_faces=1, temperature=1.0):
        p_mu, p_logvar = self.encode_image_prior(images, image_token_dropout=0.0)
        samples = []
        for _ in range(sample_k):
            z = self.reparameterize(p_mu, p_logvar)
            generated = self.topology.generate(z, min_faces=min_faces, greedy=False, temperature=temperature)
            samples.append(generated)
        return torch.stack(samples, dim=1)


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def safe_model_forward(model, tokens, token_mask, images, **kwargs):
    if isinstance(model, nn.DataParallel) and tokens.shape[0] < len(model.device_ids):
        return model.module(tokens=tokens, token_mask=token_mask, images=images, **kwargs)
    return model(tokens=tokens, token_mask=token_mask, images=images, **kwargs)


def compute_prior_loss(logits, q_mu, q_logvar, p_mu, p_logvar, pair_targets, pair_mask,
                       num_faces, edge_count, args, kl_beta):
    loss, parts = prev.compute_loss(
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
    latent_mse = F.mse_loss(p_mu, q_mu)
    loss = loss + args.latent_mse_weight * latent_mse
    parts["latent_mse"] = float(latent_mse.detach().cpu())
    parts["loss"] = float(loss.detach().cpu())
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

        logits, q_mu, q_logvar, p_mu, p_logvar = safe_model_forward(
            model, tokens, token_mask, images, sample_prior=False
        )
        kl_beta = args.kl_beta * min(1.0, epoch / max(args.kl_warmup_epochs, 1))
        _, loss_parts = compute_prior_loss(
            logits, q_mu, q_logvar, p_mu, p_logvar,
            pair_targets, pair_mask, num_faces, edge_count, args, kl_beta,
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
            posterior_generated = core.generate_posterior_mu(tokens[:take], token_mask[:take], min_faces=args.min_faces)
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
    parser.add_argument("--output-dir", default="experiments/2026-06-13/outputs_cvae_frozen_prior")
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
    parser.add_argument("--edge-pos-weight", type=float, default=1.0)
    parser.add_argument("--face-count-loss-weight", type=float, default=1.0)
    parser.add_argument("--edge-count-loss-weight", type=float, default=0.2)
    parser.add_argument("--eval-generate-limit", type=int, default=256)
    parser.add_argument("--eval-sample-k", type=int, default=0)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--no-data-parallel", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--best-metric", default="cond_prior_mu_edge_f1")
    parser.add_argument("--corrupt-prob", type=float, default=0.0)
    parser.add_argument("--corrupt-warmup-epochs", type=int, default=1)
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
        "edge_pos_weight", "face_count_loss_weight", "edge_count_loss_weight",
        "corrupt_prob", "corrupt_warmup_epochs",
    ):
        if key in ckpt_args:
            setattr(args, key, ckpt_args[key])
    return args, checkpoint


def build_model(args, device):
    model = FrozenTopologyImagePrior(
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
    model.load_topology_checkpoint(args.topology_checkpoint)
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


def build_loader(dataset, args, shuffle):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
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

    train_dataset = build_dataset(args, "train")
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
        f"topology_checkpoint={args.topology_checkpoint}; "
        f"image_source={args.image_source}; image_backbone={args.image_backbone}; "
        f"corrupt_prob={args.corrupt_prob}",
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
            logits, q_mu, q_logvar, p_mu, p_logvar = safe_model_forward(
                model,
                tokens,
                token_mask,
                images,
                sample_prior=False,
                decoder_input_override=decoder_input,
                decoder_mask_override=decoder_mask,
                image_token_dropout=args.image_token_dropout,
            )
            loss, loss_parts = compute_prior_loss(
                logits, q_mu, q_logvar, p_mu, p_logvar,
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
            progress.set_postfix({
                "loss": f"{running['loss'] / max(seen, 1):.4f}",
                "pair": f"{running['pair_loss'] / max(seen, 1):.4f}",
                "kl": f"{running['kl'] / max(seen, 1):.4f}",
                "mse": f"{running['latent_mse'] / max(seen, 1):.4f}",
            })

        train_metrics = {f"train_{k}": v / max(seen, 1) for k, v in running.items()}
        val_metrics = evaluate(model, val_loader, device, args, epoch)
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


if __name__ == "__main__":
    main()
