import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.brepnet.dataset import cube24_to_euler64
from src.brepnet.post.debug import export_edges


DEFAULT_NOISE_LEVELS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe AutoEncoder latent-space sensitivity by decoding noisy cached face latents."
    )
    parser.add_argument("--latent_root", type=str, default="/mnt/d/data/ae_cache/1119_deepcad_aug1_11k")
    parser.add_argument("--split", type=str,
                        default="src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt")
    parser.add_argument("--autoencoder-weights", dest="autoencoder_weights", type=str,
                        default="/mnt/d/data/new_cond_ckpt/20260521_intersection_noise_0p1_vae_1119_light.ckpt")
    parser.add_argument("--autoencoder", type=str, default="AutoEncoder_light")
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--noise_levels", nargs="+", type=float, default=list(DEFAULT_NOISE_LEVELS))
    parser.add_argument("--num_samples_per_level", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means use all split items.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--require_all_latents", action="store_true",
                        help="Fail before decoding if any split item is missing the requested latent cache.")

    parser.add_argument("--cube_id", type=int, default=0,
                        help="Cube-24 view id. It is mapped to the legacy Euler-64 cache suffix.")
    parser.add_argument("--euler_id", type=int, default=None,
                        help="Override the cache suffix directly. If set, cube_id is only metadata.")
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--feature_mode", choices=("mean", "sample", "raw"), default="mean",
                        help="mean uses first latent_dim dims; sample uses mean + std * eps if available.")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=None,
                        help="Decode in parallel with one worker per listed visible GPU id, e.g. --gpu_ids 0 1 2 3.")
    parser.add_argument("--save_edge_obj", action="store_true")

    parser.add_argument("--run_post", action="store_true")
    parser.add_argument("--post_only", action="store_true",
                        help="Skip decode and run post-processing from an existing decode_summary.csv.")
    parser.add_argument("--post_use_ray", action="store_true")
    parser.add_argument("--num_cpus", type=int, default=16)
    parser.add_argument("--drop_num", type=int, default=3)
    parser.add_argument("--post_timeout", type=int, default=120)
    parser.add_argument("--post_use_cuda", action="store_true")
    parser.add_argument("--max_optimize_iter", type=int, default=200)
    return parser.parse_args()


def sigma_name(sigma: float) -> str:
    return f"sigma_{sigma:.3f}".replace(".", "p").replace("-", "m")


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().to(torch.float32).cpu().numpy()
    return np.asarray(value)


def stable_int_hash(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:8], 16)


def read_split(path: Path, start: int, limit: int) -> List[str]:
    items = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    items.sort()
    if start > 0:
        items = items[start:]
    if limit > 0:
        items = items[:limit]
    return items


def check_latent_coverage(args, prefixes: Iterable[str], euler_id: int) -> List[str]:
    latent_root = Path(args.latent_root)
    prefixes = list(prefixes)
    missing = [
        prefix for prefix in prefixes
        if not (latent_root / f"{prefix}_{euler_id}" / "features.npy").is_file()
    ]
    print(
        f"[latent_sensitivity] latent coverage euler_id={euler_id}: "
        f"{len(prefixes) - len(missing)}/{len(prefixes)}"
    )
    if missing:
        print("[latent_sensitivity] first missing latents:", missing[:10])
    if missing and args.require_all_latents:
        raise FileNotFoundError(
            f"Missing {len(missing)} latent caches under {latent_root} for euler_id={euler_id}"
        )
    return missing


def build_autoencoder_conf(args) -> Dict:
    return {
        "name": args.autoencoder,
        "stage": "vae",
        "mode": "frozen_inference",
        "checkpoint": args.autoencoder_weights,
        "norm": "layer",
        "loss": "l1",
        "in_channels": 6,
        "latent_channels": 8,
        "hidden_channels": 768,
        "with_intersection": True,
        "sigmoid": False,
        "gaussian_weights": 1e-6,
        "intersection_noise_std": 0.0,
    }


def load_autoencoder(args, device: torch.device):
    from src.brepnet.models.vae import build_autoencoder

    model = build_autoencoder(build_autoencoder_conf(args))
    model.to(device)
    model.eval()
    return model


def load_latent(path: Path, args, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    raw = torch.from_numpy(np.load(path).astype(np.float32)).to(device)
    if raw.ndim != 2:
        raise ValueError(f"Expected [N,D] latent at {path}, got {tuple(raw.shape)}")

    if args.feature_mode == "raw":
        return raw

    if raw.shape[-1] < args.latent_dim:
        raise ValueError(f"{path} has dim={raw.shape[-1]}, smaller than latent_dim={args.latent_dim}")

    mean = raw[:, :args.latent_dim]
    if args.feature_mode == "mean":
        return mean

    if raw.shape[-1] < args.latent_dim * 2:
        raise ValueError(f"feature_mode=sample needs mean+std dims, got {tuple(raw.shape)} at {path}")
    std = raw[:, args.latent_dim:args.latent_dim * 2]
    eps = torch.randn(mean.shape, device=device, dtype=mean.dtype, generator=generator)
    return mean + std * eps


def save_reconstruction(item_dir: Path, recon: Dict, latent: torch.Tensor, noisy_latent: torch.Tensor,
                        meta: Dict, save_edge_obj: bool):
    item_dir.mkdir(parents=True, exist_ok=True)
    pred_face_adj = as_numpy(recon["pred_face_adj"])
    pred_edge_face_connectivity = as_numpy(recon["pred_edge_face_connectivity"])
    pred_face_adj_prob = as_numpy(recon.get("pred_face_adj_prob", pred_edge_face_connectivity.reshape(-1)))
    pred_face = as_numpy(recon["pred_face"])
    pred_edge = as_numpy(recon["pred_edge"])

    latent_np = latent.detach().to(torch.float32).cpu().numpy()
    noisy_np = noisy_latent.detach().to(torch.float32).cpu().numpy()
    np.savez_compressed(
        item_dir / "data.npz",
        pred_face_adj_prob=pred_face_adj_prob,
        pred_face_adj=pred_face_adj,
        pred_face=pred_face,
        pred_edge=pred_edge,
        pred_edge_face_connectivity=pred_edge_face_connectivity,
        latent=latent_np,
        noisy_latent=noisy_np,
        latent_delta=noisy_np - latent_np,
        **meta,
    )
    if save_edge_obj:
        export_edges(pred_edge, str(item_dir / "edge.obj"))


def decode_noisy_latents(args, model, device: torch.device, prefixes: Iterable[str], euler_id: int,
                         output_root: Path, desc: str = "decode") -> List[Dict]:
    latent_root = Path(args.latent_root)
    network_root = output_root / "network_pred"
    rows = []

    for prefix in tqdm(list(prefixes), desc=desc):
        latent_path = latent_root / f"{prefix}_{euler_id}" / "features.npy"
        if not latent_path.is_file():
            for sigma in args.noise_levels:
                reps = max(1, args.num_samples_per_level)
                for rep in range(reps):
                    rows.append({
                        "prefix": prefix,
                        "sample_name": f"{prefix}_cube{args.cube_id:02d}_euler{euler_id:02d}_s{rep:02d}",
                        "noise_sigma": float(sigma),
                        "sample_id": rep,
                        "cube_id": args.cube_id,
                        "euler_id": euler_id,
                        "latent_path": str(latent_path),
                        "status": "missing_latent",
                        "error": f"Missing {latent_path}",
                    })
            continue

        base_generator = torch.Generator(device=device)
        base_generator.manual_seed(args.seed)
        latent = load_latent(latent_path, args, device, base_generator)

        for sigma in args.noise_levels:
            reps = max(1, args.num_samples_per_level)
            for rep in range(reps):
                sample_seed = (
                    args.seed
                    + int(round(sigma * 1000)) * 100000
                    + rep * 1009
                    + stable_int_hash(prefix) % 100000
                )
                generator = torch.Generator(device=device)
                generator.manual_seed(sample_seed)
                if sigma == 0:
                    noisy_latent = latent.clone()
                else:
                    noise = torch.randn(latent.shape, device=device, dtype=latent.dtype, generator=generator)
                    noisy_latent = latent + float(sigma) * noise

                sample_name = f"{prefix}_cube{args.cube_id:02d}_euler{euler_id:02d}_s{rep:02d}"
                item_dir = network_root / sigma_name(float(sigma)) / sample_name
                row = {
                    "prefix": prefix,
                    "sample_name": sample_name,
                    "noise_sigma": float(sigma),
                    "sample_id": rep,
                    "seed": sample_seed,
                    "cube_id": args.cube_id,
                    "euler_id": euler_id,
                    "latent_path": str(latent_path),
                    "num_faces_in": int(latent.shape[0]),
                    "latent_dim": int(latent.shape[1]),
                    "delta_l2_mean": float(torch.linalg.norm(noisy_latent - latent, dim=-1).mean().item()),
                    "delta_linf": float((noisy_latent - latent).abs().max().item()),
                    "status": "decode_ok",
                    "error": "",
                }
                try:
                    with torch.no_grad():
                        recon = model.inference(noisy_latent)
                    row["num_faces_out"] = int(as_numpy(recon["pred_face"]).shape[0])
                    row["num_edges_out"] = int(as_numpy(recon["pred_edge"]).shape[0])
                    save_reconstruction(
                        item_dir,
                        recon,
                        latent,
                        noisy_latent,
                        {
                            "source_prefix": np.array(prefix),
                            "noise_sigma": np.array(float(sigma), dtype=np.float32),
                            "sample_id": np.array(rep, dtype=np.int32),
                            "cube_id": np.array(args.cube_id, dtype=np.int32),
                            "euler_id": np.array(euler_id, dtype=np.int32),
                            "latent_path": np.array(str(latent_path)),
                        },
                        args.save_edge_obj,
                    )
                except Exception as exc:
                    row["status"] = "decode_failed"
                    row["error"] = repr(exc)
                    item_dir.mkdir(parents=True, exist_ok=True)
                    (item_dir / "error.txt").write_text(traceback.format_exc())
                rows.append(row)
    return rows


def split_round_robin(items: List[str], num_parts: int) -> List[List[str]]:
    return [items[i::num_parts] for i in range(num_parts)]


def decode_worker(args_dict: Dict, gpu_id: int, worker_id: int, prefixes: List[str],
                  euler_id: int, output_root: str) -> List[Dict]:
    args = SimpleNamespace(**args_dict)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed + worker_id)
    np.random.seed(args.seed + worker_id)

    print(f"[decode worker {worker_id}] gpu={gpu_id} device={device} items={len(prefixes)}")
    model = load_autoencoder(args, device)
    rows = decode_noisy_latents(
        args,
        model,
        device,
        prefixes,
        euler_id,
        Path(output_root),
        desc=f"decode gpu{gpu_id}",
    )
    for row in rows:
        row["decode_worker"] = worker_id
        row["decode_gpu"] = gpu_id
    return rows


def decode_all(args, prefixes: List[str], euler_id: int, output_root: Path) -> List[Dict]:
    gpu_ids = list(args.gpu_ids or [])
    if len(gpu_ids) > 0:
        chunks = split_round_robin(prefixes, len(gpu_ids))
        jobs = [
            (vars(args), gpu_id, worker_id, chunk, euler_id, str(output_root))
            for worker_id, (gpu_id, chunk) in enumerate(zip(gpu_ids, chunks))
            if len(chunk) > 0
        ]
        if len(jobs) == 1:
            return decode_worker(*jobs[0])

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(jobs)) as pool:
            nested_rows = pool.starmap(decode_worker, jobs)
        return [row for rows in nested_rows for row in rows]

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = load_autoencoder(args, device)
    return decode_noisy_latents(args, model, device, prefixes, euler_id, output_root)


def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def run_post_for_sigma(args, data_root: Path, out_root: Path, folders: List[str]) -> List[Dict]:
    from src.brepnet.post.construct_brep import construct_brep_for_sample

    rows = []
    if args.post_use_ray:
        import ray

        ray.init(dashboard_host="0.0.0.0", dashboard_port=8080, num_cpus=args.num_cpus,
                 ignore_reinit_error=True)
        remote_fn = ray.remote(num_gpus=0.1 if args.post_use_cuda else 0, max_retries=0)(construct_brep_for_sample)
        tasks = []
        for folder in folders:
            tasks.append((
                folder,
                remote_fn.remote(
                    data_root, out_root, folder,
                    v_drop_num=args.drop_num,
                    use_cuda=args.post_use_cuda,
                    from_scratch=True,
                    is_log=False,
                    is_ray=True,
                    is_optimize_geom=True,
                    isdebug=False,
                    v_max_optimize_iter=args.max_optimize_iter,
                ),
            ))
        for folder, task in tqdm(tasks, desc=f"post {data_root.name}"):
            status = "post_ok"
            error = ""
            try:
                ray.get(task, timeout=args.post_timeout)
            except Exception as exc:
                status = "post_failed"
                error = repr(exc)
                try:
                    ray.cancel(task)
                except Exception:
                    pass
            rows.append(post_status_row(folder, out_root / folder, status, error))
        return rows

    for folder in tqdm(folders, desc=f"post {data_root.name}"):
        status = "post_ok"
        error = ""
        try:
            construct_brep_for_sample(
                data_root, out_root, folder,
                v_drop_num=args.drop_num,
                use_cuda=args.post_use_cuda,
                from_scratch=True,
                is_log=False,
                is_ray=False,
                is_optimize_geom=True,
                isdebug=False,
                v_max_optimize_iter=args.max_optimize_iter,
            )
        except Exception as exc:
            status = "post_failed"
            error = repr(exc)
            sample_dir = out_root / folder
            sample_dir.mkdir(parents=True, exist_ok=True)
            (sample_dir / "post_error.txt").write_text(traceback.format_exc())
        rows.append(post_status_row(folder, out_root / folder, status, error))
    return rows


def post_status_row(folder: str, sample_dir: Path, status: str, error: str) -> Dict:
    return {
        "sample_name": folder,
        "post_status": status,
        "post_error": error,
        "success": int((sample_dir / "success.txt").is_file()),
        "step_exists": int((sample_dir / "recon_brep.step").is_file()),
        "stl_exists": int((sample_dir / "recon_brep.stl").is_file()),
    }


def run_post(args, output_root: Path, decode_rows: List[Dict]) -> List[Dict]:
    post_rows = []
    for sigma in args.noise_levels:
        sigma_dir = sigma_name(float(sigma))
        data_root = output_root / "network_pred" / sigma_dir
        out_root = output_root / "after_post" / sigma_dir
        if not data_root.is_dir():
            continue
        folders = sorted(
            row["sample_name"]
            for row in decode_rows
            if row.get("status") == "decode_ok" and float(row.get("noise_sigma", -1)) == float(sigma)
        )
        post_rows.extend([
            {"noise_sigma": float(sigma), **row}
            for row in run_post_for_sigma(args, data_root, out_root, folders)
        ])
    return post_rows


def summarize(decode_rows: List[Dict], post_rows: Optional[List[Dict]]) -> List[Dict]:
    sigmas = sorted({float(row["noise_sigma"]) for row in decode_rows if "noise_sigma" in row})
    post_by_sample = {}
    if post_rows is not None:
        post_by_sample = {(float(row["noise_sigma"]), row["sample_name"]): row for row in post_rows}

    summary = []
    for sigma in sigmas:
        rows = [row for row in decode_rows if float(row.get("noise_sigma", -1)) == sigma]
        decode_ok = [row for row in rows if row.get("status") == "decode_ok"]
        item = {
            "noise_sigma": sigma,
            "total": len(rows),
            "decode_ok": len(decode_ok),
            "decode_success_rate": len(decode_ok) / max(1, len(rows)),
        }
        if post_rows is not None:
            post_items = [post_by_sample.get((sigma, row["sample_name"])) for row in decode_ok]
            post_items = [row for row in post_items if row is not None]
            success = sum(int(row["success"]) for row in post_items)
            step_exists = sum(int(row["step_exists"]) for row in post_items)
            item.update({
                "post_total": len(post_items),
                "post_success": success,
                "post_success_rate": success / max(1, len(post_items)),
                "step_exists": step_exists,
                "step_exists_rate": step_exists / max(1, len(post_items)),
            })
        summary.append(item)
    return summary


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    euler_id = int(args.euler_id) if args.euler_id is not None else int(cube24_to_euler64(args.cube_id))
    prefixes = read_split(Path(args.split), args.start, args.limit)

    config = vars(args).copy()
    config["resolved_device"] = (
        "multi_gpu:" + ",".join(map(str, args.gpu_ids))
        if args.gpu_ids else
        str(torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"))
    )
    config["resolved_euler_id"] = euler_id
    config["num_prefixes"] = len(prefixes)
    config_name = "post_config.json" if args.post_only else "config.json"
    (output_root / config_name).write_text(json.dumps(config, indent=2))

    print(f"[latent_sensitivity] split items={len(prefixes)}")
    print(f"[latent_sensitivity] cube_id={args.cube_id} -> euler_id={euler_id}")
    print(f"[latent_sensitivity] output={output_root}")

    if args.post_only:
        decode_summary_path = output_root / "decode_summary.csv"
        if not decode_summary_path.is_file():
            raise FileNotFoundError(f"--post_only requires {decode_summary_path}")
        decode_rows = read_csv(decode_summary_path)
        args.run_post = True
    else:
        check_latent_coverage(args, prefixes, euler_id)
        decode_rows = decode_all(args, prefixes, euler_id, output_root)
        write_csv(output_root / "decode_summary.csv", decode_rows)

    post_rows = None
    if args.run_post:
        post_rows = run_post(args, output_root, decode_rows)
        write_csv(output_root / "post_summary.csv", post_rows)

    summary = summarize(decode_rows, post_rows)
    write_csv(output_root / "success_summary.csv", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
