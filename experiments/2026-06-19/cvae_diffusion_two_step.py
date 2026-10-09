#!/usr/bin/env python3
"""Run the two-step pipeline: CVAE topology prediction + diffusion inference."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.brepnet.dataset import Diffusion_dataset
from src.brepnet.models import build_model


CVAE_SCRIPT = REPO_ROOT / "experiments" / "2026-06-18" / "topology_faceadj_cvae_visibility.py"
SPEC = importlib.util.spec_from_file_location("topology_faceadj_cvae_visibility", CVAE_SCRIPT)
cvae = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(cvae)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion-config", default="train_diffusion_baseline_real")
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--cvae-checkpoint", default="experiments/2026-06-18/outputs_cvae_visibility/best.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topology-cache-dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--raw-root", default="/mnt/d/data/deepcad_v7")
    parser.add_argument("--condition-root", default="/mnt/d/data/deepcad_v7_cond")
    parser.add_argument("--cached-latent-root", default="/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24")
    parser.add_argument("--ae-checkpoint", default="/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt")
    parser.add_argument("--train-list", default="src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt")
    parser.add_argument("--val-list", default="src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt")
    parser.add_argument("--test-list", default="src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--topology-bias-scale", type=float, default=2.0)
    parser.add_argument("--cvae-temperature", type=float, default=1.0)
    parser.add_argument("--sample-prior", action="store_true")
    parser.add_argument("--stochastic-topology", action="store_true")
    parser.add_argument("--seed", type=int, default=20260619)
    return parser.parse_args()


def move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(item, device) for item in value)
    return value


def default_cvae_args() -> SimpleNamespace:
    return SimpleNamespace(
        max_faces=30,
        min_faces=7,
        order_mode="wl",
        wl_rounds=3,
        image_backbone="dinov2",
        depth_anything_v2_ckpt=None,
        d_model=256,
        nhead=8,
        encoder_layers=4,
        decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
    )


def load_cvae_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cvae_args = default_cvae_args()
    for key, value in checkpoint.get("args", {}).items():
        if hasattr(cvae_args, key):
            setattr(cvae_args, key, value)
    model = cvae.build_model(cvae_args, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return cvae_args, model


def compose_diffusion_cfg(args: argparse.Namespace):
    split_list = args.test_list if args.split == "test" else args.val_list
    overrides = [
        f"trainer.batch_size={args.batch_size}",
        "trainer.devices=1",
        "trainer.wandb.enabled=false",
        f"dataset.raw_data_root={args.raw_root}",
        f"dataset.cached_latent_root={args.cached_latent_root}",
        f"dataset.condition_data_root={args.condition_root}",
        f"dataset.train_dataset={args.train_list}",
        f"dataset.val_dataset={args.val_list}",
        f"dataset.test_dataset={split_list}",
        "dataset.real_photo_ratio=1.0",
        "dataset.validation_real_photo_ratio=1.0",
        "dataset.is_aug=false",
        "dataset.is_overfit=false",
        "dataset.load_topology=false",
        "dataset.epoch_scale_factor=1",
        "model.topology_bias.enabled=true",
        "model.topology_bias.source=gt_adjacency",
        "model.topology_bias.mode=soft_bias",
        f"model.topology_bias.scale={args.topology_bias_scale}",
        f"model.autoencoder.checkpoint={args.ae_checkpoint}",
        "hydra.job.chdir=false",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name=args.diffusion_config, overrides=overrides)
    return cfg


def load_diffusion_model(args: argparse.Namespace, cfg, device: torch.device):
    model = build_model(cfg.model, cfg.condition).to(device)
    checkpoint = torch.load(args.diffusion_checkpoint, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state[key[len("model."):]] = value
        else:
            model_state[key] = value
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(
        f"Loaded diffusion checkpoint {args.diffusion_checkpoint} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})",
        flush=True,
    )
    if missing:
        print(f"Missing keys head: {missing[:10]}", flush=True)
    if unexpected:
        print(f"Unexpected keys head: {unexpected[:10]}", flush=True)
    model.eval()
    return model


def build_loader(cfg, args: argparse.Namespace):
    mode = "testing" if args.split == "test" else "validation"
    dataset = Diffusion_dataset(mode, cfg.dataset)
    if args.max_samples:
        dataset.data_folders = dataset.data_folders[: args.max_samples]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=Diffusion_dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader


def topology_cache_path(topology_cache_dir: Path, model_id: str) -> Path:
    return topology_cache_dir / str(model_id) / "topology.npz"


def generated_adjacency_to_original_order(
    generated: torch.Tensor,
    model_ids: list[str],
    raw_root: Path,
    cvae_args: SimpleNamespace,
) -> tuple[torch.Tensor, list[dict]]:
    pred_adj_ordered, pred_count, pred_edge_count = cvae.base.target_to_adj(
        generated,
        max_faces=cvae_args.max_faces,
        min_faces=cvae_args.min_faces,
    )
    pred_adj_ordered = pred_adj_ordered.detach().cpu()
    pred_count = pred_count.detach().cpu().numpy().astype(np.int64)
    pred_edge_count = pred_edge_count.detach().cpu().numpy().astype(np.int64)

    restored = []
    metadata = []
    for batch_idx, model_id in enumerate(model_ids):
        with np.load(raw_root / model_id / "data.npz") as data:
            gt_adj = cvae.base.symmetrize_adj(data["face_adj"])
        order = cvae.base.canonical_face_order(gt_adj, cvae_args.order_mode, cvae_args.wl_rounds)
        num_faces = min(int(gt_adj.shape[0]), cvae_args.max_faces)
        restored_adj = torch.zeros(cvae_args.max_faces, cvae_args.max_faces, dtype=torch.float32)
        order_used = torch.from_numpy(order[:num_faces]).long()
        restored_adj[order_used[:, None], order_used[None, :]] = pred_adj_ordered[batch_idx, :num_faces, :num_faces]
        restored_adj = torch.maximum(restored_adj, restored_adj.T)
        restored_adj.fill_diagonal_(0.0)
        restored.append(restored_adj)
        metadata.append({
            "model_id": model_id,
            "face_order": order.astype(np.int64),
            "gt_num_faces": int(gt_adj.shape[0]),
            "pred_num_faces": int(pred_count[batch_idx]),
            "pred_edge_count": int(pred_edge_count[batch_idx]),
        })
    return torch.stack(restored, dim=0), metadata


@torch.no_grad()
def predict_topology(cvae_model, cvae_args, batch: dict, args: argparse.Namespace, device: torch.device):
    images = batch["conditions"]["imgs"].to(device, non_blocking=True)
    image_global, image_context = cvae_model.encode_image(images, image_token_dropout=0.0)
    p_mu, p_logvar = cvae_model.encode_prior(image_global)
    z = cvae_model.reparameterize(p_mu, p_logvar) if args.sample_prior else p_mu
    generated = cvae_model.generate(
        z,
        image_context,
        min_faces=cvae_args.min_faces,
        greedy=not args.stochastic_topology,
        temperature=args.cvae_temperature,
    )
    face_adj, metadata = generated_adjacency_to_original_order(
        generated,
        list(batch["v_prefix"]),
        Path(args.raw_root),
        cvae_args,
    )
    return face_adj, generated.detach().cpu(), metadata


def save_diffusion_predictions(batch: dict, results: list[dict], output_dir: Path) -> None:
    prefixes = list(batch.get("v_prefix", [f"sample_{i}" for i in range(len(results))]))
    for idx, recon_data in enumerate(results):
        item_dir = output_dir / str(prefixes[idx])
        item_dir.mkdir(parents=True, exist_ok=True)
        save_dict = {}
        for key, value in recon_data.items():
            if isinstance(value, torch.Tensor):
                save_dict[key] = value.detach().cpu().numpy()
            elif isinstance(value, np.ndarray):
                save_dict[key] = value
        np.savez_compressed(str(item_dir / "data.npz"), **save_dict)


def save_topology_cache(
    batch: dict,
    face_adj: torch.Tensor,
    generated: torch.Tensor,
    metadata: list[dict],
    topology_cache_dir: Path,
) -> None:
    topology_cache_dir.mkdir(parents=True, exist_ok=True)
    for idx, model_id in enumerate(batch["v_prefix"]):
        item_dir = topology_cache_dir / str(model_id)
        item_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(item_dir / "topology.npz"),
            face_adj=face_adj[idx].numpy().astype(np.float32),
            generated_tokens=generated[idx].numpy().astype(np.int64),
            face_order=metadata[idx]["face_order"],
            gt_num_faces=np.array(metadata[idx]["gt_num_faces"], dtype=np.int64),
            pred_num_faces=np.array(metadata[idx]["pred_num_faces"], dtype=np.int64),
            pred_edge_count=np.array(metadata[idx]["pred_edge_count"], dtype=np.int64),
        )


def load_topology_cache(batch: dict, topology_cache_dir: Path) -> torch.Tensor:
    face_adj = []
    missing = []
    for model_id in batch["v_prefix"]:
        path = topology_cache_path(topology_cache_dir, str(model_id))
        if not path.exists():
            missing.append(str(model_id))
            continue
        with np.load(path) as data:
            face_adj.append(torch.from_numpy(data["face_adj"].astype(np.float32)))
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing topology cache for {len(missing)} models: {preview}")
    return torch.stack(face_adj, dim=0)


def release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def generate_topology_cache(
    args: argparse.Namespace,
    cfg,
    topology_cache_dir: Path,
    device: torch.device,
) -> None:
    cvae_args, cvae_model = load_cvae_model(args.cvae_checkpoint, device)
    _, loader = build_loader(cfg, args)
    for batch in tqdm(loader, desc=f"{args.split} CVAE topology cache"):
        face_adj_cpu, generated_cpu, metadata = predict_topology(cvae_model, cvae_args, batch, args, device)
        save_topology_cache(batch, face_adj_cpu, generated_cpu, metadata, topology_cache_dir)
    del cvae_model
    release_cuda_cache()


def run_diffusion_inference(
    args: argparse.Namespace,
    cfg,
    output_dir: Path,
    topology_cache_dir: Path,
    device: torch.device,
) -> None:
    _, loader = build_loader(cfg, args)
    diffusion_model = load_diffusion_model(args, cfg, device)
    for batch in tqdm(loader, desc=f"{args.split} topology-guided diffusion"):
        face_adj_cpu = load_topology_cache(batch, topology_cache_dir)
        batch = move_to_device(batch, device)
        batch["face_adj"] = face_adj_cpu.to(device, non_blocking=True)
        with torch.inference_mode():
            results = diffusion_model.inference(len(batch["v_prefix"]), device, v_data=batch, v_log=False)
        save_diffusion_predictions(batch, results, output_dir)
    del diffusion_model
    release_cuda_cache()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    topology_cache_dir = Path(args.topology_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    topology_cache_dir.mkdir(parents=True, exist_ok=True)

    diffusion_cfg = compose_diffusion_cfg(args)
    generate_topology_cache(args, diffusion_cfg, topology_cache_dir, device)
    run_diffusion_inference(args, diffusion_cfg, output_dir, topology_cache_dir, device)

    print(f"Saved diffusion outputs to {output_dir}", flush=True)
    print(f"Saved predicted topology cache to {topology_cache_dir}", flush=True)


if __name__ == "__main__":
    main()
