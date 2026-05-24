import argparse
import csv
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from lightning_fabric import seed_everything

from src.brepnet.dataset import Diffusion_dataset
from src.brepnet.models.diffusion import Diffusion
from src.brepnet.post.utils import export_edges


DEFAULT_RESULT_ROOT = "/mnt/d/data/new_cond_results/exp_plan_0502_depth"
DEFAULT_CHECKPOINT = "/mnt/d/data/new_cond_ckpt/0502_deepcad_flux_single_view_align_depth.ckpt"
DEFAULT_AUTOENCODER = "/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt"
DEFAULT_LATENT_ROOT = "/mnt/d/data/ae_cache/1119_deepcad_aug1_11k"
DEFAULT_COND_ROOT = "/mnt/d/data/deepcad_v6_cond"
DEFAULT_TEST_LIST = "src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt"


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--autoencoder-weights", default=DEFAULT_AUTOENCODER)
    parser.add_argument("--latent-root", default=DEFAULT_LATENT_ROOT)
    parser.add_argument("--cond-root", default=DEFAULT_COND_ROOT)
    parser.add_argument("--test-list", default=DEFAULT_TEST_LIST)
    parser.add_argument("--output-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=20260509)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")


def build_model_conf(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "name": "Diffusion",
        "stage": "diffusion",
        "loss": "l2",
        "latent": {
            "dim": 32,
            "use_cached_latents": True,
            "use_mean": True,
        },
        "padding": {
            "type": "zero",
            "max_faces": 30,
            "valid_loss_weight": 0.01,
        },
        "noise": {
            "prediction_type": "epsilon",
            "beta_schedule": "squaredcos_cap_v2",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "variance_type": "fixed_small",
            "num_train_timesteps": 1000,
        },
        "denoiser": {
            "hidden_dim": 768,
            "num_layers": 24,
            "nhead_divisor": 64,
            "feedforward_dim": 2048,
            "dropout": 0.1,
        },
        "condition_fuser": {
            "type": "cross_attention",
            "condition_dim": 1024,
            "hidden_dim": 1024,
            "num_layers": 4,
            "alignment": {
                "enabled": True,
                "weight": 1.0,
                "temperature": 0.07,
                "projection_dim": 256,
            },
        },
        "topology_bias": {
            "enabled": False,
            "source": "gt_adjacency",
            "target": "self_attention",
            "mode": "soft_bias",
            "timestep_weight": "linear_high_noise",
            "scale": 2.0,
        },
        "autoencoder": {
            "name": "AutoEncoder_light",
            "stage": "vae",
            "mode": "frozen_inference",
            "checkpoint": args.autoencoder_weights,
            "in_channels": 6,
            "latent_channels": 8,
            "hidden_channels": 768,
            "norm": "layer",
            "gaussian_weights": 1e-6,
            "sigmoid": False,
            "num_gat_layers": 5,
            "bottleneck_dim": 768,
            "num_encoder_layers": 8,
            "num_decoder_layers": 8,
            "nhead": 16,
            "with_intersection": True,
            "intersection_dim": 512,
            "intersection_layers": 8,
            "intersection_noise_std": 0.0,
            "trainable_scope": "all",
            "trainable_module_prefixes": ["inter", "classifier"],
            "loss": "l1",
        },
    }


def build_condition_conf() -> dict[str, Any]:
    return {
        "type": "single_img",
        "dataset_names": ["single_img"],
        "cached_features": False,
        "output_dim": 1024,
        "image": {
            "backbone": "dinov2",
            "depth_anything_v2_ckpt": None,
            "augment_probability": 0.0,
        },
        "point_cloud": {
            "encoder": "pointnet",
            "augment_probability": 0.0,
        },
    }


def build_dataset_conf(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "name": "Diffusion_dataset",
        "train_dataset": args.test_list,
        "val_dataset": args.test_list,
        "test_dataset": args.test_list,
        "data_root": None,
        "latent_root": args.latent_root,
        "padding": "zero",
        "cached_condition": False,
        "scale_factor": 200,
        "overfit": False,
        "is_overfit": False,
        "is_aug": 0,
        "length": 1000,
        "max_faces": 30,
        "load_topology": False,
        "condition_names": ["single_img"],
        "condition_root": args.cond_root,
        "num_points": 10000,
        "real_photo_ratio": 0.0,
        "augmentation": {
            "enabled": False,
            "random_rotate": False,
            "random_scale": False,
            "scale_range": [1.0, 1.0],
        },
    }


def ensure_exists(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def prepare_output_dir(path: str | Path, force: bool = False) -> Path:
    path = Path(path)
    if path.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {path}. Pass --force to overwrite it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_mkdir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def create_dataloader(args: argparse.Namespace) -> DataLoader:
    dataset_conf = build_dataset_conf(args)
    dataset = Diffusion_dataset("testing", dataset_conf)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=Diffusion_dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[Diffusion, dict[str, Any]]:
    ensure_exists(args.checkpoint, "diffusion checkpoint")
    ensure_exists(args.autoencoder_weights, "autoencoder checkpoint")
    model_conf = build_model_conf(args)
    model = Diffusion(model_conf, build_condition_conf())

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            stripped[key[len("model."):]] = value
        else:
            stripped[key] = value
    incompatible = model.load_state_dict(stripped, strict=False)
    model.to(device)
    model.eval()
    metadata = {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    return model, metadata


def condition_from_mode(condition: torch.Tensor, mode: str) -> torch.Tensor:
    if mode in ("normal", "none", ""):
        return condition
    if mode in ("zero", "zeros"):
        return torch.zeros_like(condition)
    if mode == "shuffle":
        if condition.shape[0] <= 1:
            return condition
        order = torch.randperm(condition.shape[0], device=condition.device)
        return condition[order]
    raise ValueError(f"Unknown condition mode: {mode}")


def install_condition_mode(model: Diffusion, mode: str) -> None:
    original_extract_condition = model.extract_condition

    def extract_condition_with_mode(v_data):
        condition = original_extract_condition(v_data)
        if condition is None:
            return condition
        return condition_from_mode(condition, mode)

    model.extract_condition = extract_condition_with_mode


def save_reconstruction(
    output_root: Path,
    prefix: str,
    recon_data: dict[str, Any],
    condition_batch: dict[str, Any] | None = None,
    batch_index: int | None = None,
) -> None:
    item_root = safe_mkdir(output_root / prefix)
    export_edges(recon_data["pred_edge"], str(item_root / f"{prefix}_edge.obj"))
    np.savez_compressed(
        str(item_root / "data.npz"),
        pred_face_adj_prob=recon_data["pred_face_adj_prob"],
        pred_face_adj=recon_data["pred_face_adj"].cpu().numpy()
        if torch.is_tensor(recon_data["pred_face_adj"])
        else recon_data["pred_face_adj"],
        pred_face=recon_data["pred_face"],
        pred_edge=recon_data["pred_edge"],
        pred_edge_face_connectivity=recon_data["pred_edge_face_connectivity"],
    )
    if condition_batch is None or batch_index is None:
        return
    if "ori_imgs" not in condition_batch:
        return
    imgs = condition_batch["ori_imgs"][batch_index].detach().cpu().numpy().astype(np.uint8)
    img_ids = condition_batch["img_id"][batch_index].detach().cpu().numpy()
    for img_offset in range(imgs.shape[0]):
        img_id = int(img_ids[img_offset]) if img_offset < len(img_ids) else img_offset
        Image.fromarray(imgs[img_offset]).save(item_root / f"{prefix}_img{img_id}.png")


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    safe_mkdir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    safe_mkdir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_prefixes(list_path: str | Path) -> list[str]:
    with Path(list_path).open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def git_status_short() -> str:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.stdout
    except Exception as exc:
        return f"failed to collect git status: {exc}"


def write_run_metadata(path: str | Path, args: argparse.Namespace, extra: dict[str, Any] | None = None) -> None:
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "args": vars(args),
        "cwd": os.getcwd(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "git_status_short": git_status_short(),
    }
    if extra:
        metadata.update(extra)
    write_json(path, metadata)


def load_eval_result(post_root: str | Path, folder: str) -> dict[str, Any] | None:
    eval_path = Path(post_root) / folder / "eval.npz"
    if not eval_path.exists():
        return None
    try:
        item = np.load(eval_path, allow_pickle=True)["results"].item()
    except Exception:
        return None
    result = {key: scalar_to_python(value) for key, value in item.items()}
    result["prefix"] = folder
    result["is_success"] = (Path(post_root) / folder / "success.txt").exists()
    result["eval_path"] = str(eval_path)
    return result


def scalar_to_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.tolist()
    return value


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            f_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f_value):
            values.append(f_value)
    return values


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(values))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_rows": len(rows),
        "num_success": int(sum(bool(row.get("is_success")) for row in rows)),
    }
    summary["valid_rate"] = summary["num_success"] / len(rows) if rows else None
    metric_keys = [
        "face_cd",
        "edge_cd",
        "vertex_cd",
        "face_fscore",
        "edge_fscore",
        "vertex_fscore",
        "fe_fscore",
        "ev_fscore",
        "num_recon_face",
        "num_gt_face",
        "abs_face_count_error",
    ]
    for key in metric_keys:
        values = finite_values(rows, key)
        summary[f"{key}_mean"] = mean_or_none(values)
        summary[f"{key}_median"] = median_or_none(values)
    return summary
