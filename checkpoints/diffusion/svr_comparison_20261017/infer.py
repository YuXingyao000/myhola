"""Fixed-list, one-proposal SVR inference; each worker owns one GPU/model.

No GT tensors enter the model. A saved success/error is terminal unless an
infrastructure repair is explicitly recorded outside this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("current", "legacy", "alignment"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--condition-root", type=Path, default=Path("/mnt/d/data/deepcad_v7_cond"))
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--only-id", help="Interface check on one frozen-list member; same seed/output contract.")
    return parser.parse_args()


def main():
    args = arguments()
    if args.shards < 1 or not 0 <= args.rank < args.shards:
        raise ValueError("Require 0 <= rank < shards.")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TQDM_DISABLE"] = "1"

    import numpy as np
    import torch
    import torchvision.transforms as T

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    split = args.root / "validation100.txt"
    ids = split.read_text().splitlines()
    if sha256(split) != "6c952c3556860c0a2782a0bedbf139b48bc2769b385788baf0fcb82fac1be587" or len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("The frozen comparison requires exactly 100 unique IDs.")
    if args.only_id and args.only_id not in ids:
        raise ValueError("The interface sample must belong to the frozen list.")
    selected = [args.only_id] if args.only_id else ids[args.rank::args.shards]
    status_root = args.root / "metadata" / "inference" / args.method
    output_root = args.root / "raw_predictions" / args.method
    checkpoint_stat = args.checkpoint.stat()
    identity = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "backend": args.backend,
        "split_sha256": sha256(split),
        "condition_root": str(args.condition_root.resolve()),
        "config_sha256": sha256(args.config) if args.config is not None else None,
        "worker_source_sha256": sha256(Path(__file__)),
        "sampling": {"steps": 1000, "proposals_per_image": 1, "batch_size": 1, "precision": "float32", "tf32": False},
    }
    if args.backend == "current":
        source_files = list((ROOT / "src/brepnet/diffusion").glob("*.py"))
        source_files += [ROOT / "src/brepnet/models" / name for name in ("vae.py", "blocks.py", "utils.py")]
        identity["source_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in source_files if path.is_file()}
    else:
        runtime = Path(__file__).with_name(f"{args.backend}_runtime.py")
        identity["runtime_sha256"] = sha256(runtime)
    pending = []
    for sample_id in selected:
        saved = status_root / f"{sample_id}.json"
        if not saved.exists():
            if (output_root / sample_id / "data.npz").exists():
                raise ValueError(f"Untracked raw prediction for {sample_id}; preserve and audit it before recovery.")
            pending.append(sample_id)
            continue
        record = json.loads(saved.read_text())
        if record["identity"] != identity:
            raise ValueError(f"Refusing mixed inference protocols at {saved}")
        with np.load(args.condition_root / sample_id / "imgs.npz") as archive:
            pixels = archive["svr_imgs"][0]
        if hashlib.sha256(pixels.tobytes()).hexdigest() != record.get("image_sha256"):
            raise ValueError(f"Condition image changed since {saved}")
        if record["status"] == "ok":
            raw = output_root / sample_id / "data.npz"
            if not raw.is_file() or sha256(raw) != record["output_sha256"]:
                raise ValueError(f"Saved prediction no longer matches {saved}")
    if not pending:
        print(f"{args.method} shard{args.rank}: all records already terminal", flush=True)
        return

    load_started = time.perf_counter()
    if args.backend in ("legacy", "alignment"):
        spec = importlib.util.spec_from_file_location(f"svr_{args.backend}_runtime", Path(__file__).with_name(f"{args.backend}_runtime.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model, provenance = getattr(module, f"load_{args.backend}_model")(args.checkpoint)
    else:
        if args.config is None:
            raise ValueError("Current checkpoints require their saved resolved config.")
        from omegaconf import OmegaConf
        from src.brepnet.diffusion.model import Diffusion

        cfg = OmegaConf.load(args.config)
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        # A full checkpoint supplies all frozen AE weights as well as diffusion.
        model_cfg["autoencoder"]["checkpoint"] = None
        checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
        state = checkpoint["state_dict"]
        if not all(key.startswith("model.") for key in state):
            raise ValueError("Expected a complete UnifiedTrainer model state.")
        model = Diffusion(model_cfg, OmegaConf.to_container(cfg.condition, resolve=True))
        result = model.load_state_dict({key[6:]: value for key, value in state.items()}, strict=True, assign=True)
        provenance = {
            "checkpoint_global_step": int(checkpoint["global_step"]),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "state_key_count": len(state), "strict_load": str(result),
            "config_path": str(args.config.resolve()), "config_sha256": sha256(args.config),
            "autoencoder_source": "full diffusion checkpoint, including alias state keys",
            "source_sha256": {name: sha256(ROOT / "src/brepnet/diffusion" / name)
                              for name in ("model.py", "denoiser.py", "image_encoder.py", "fusion.py", "sampling.py", "padding.py")},
        }
        del checkpoint, state
    model = model.float().to(device).eval().requires_grad_(False)
    if len(model.noise_scheduler.timesteps) != 1000:
        raise ValueError("This comparison requires the native 1000-step DDPM schedule.")
    torch.cuda.synchronize(device)
    provenance.update({
        "identity": identity, "rank": args.rank, "gpu": torch.cuda.get_device_name(device),
        "load_seconds": time.perf_counter() - load_started,
        "torch": torch.__version__, "numpy": np.__version__,
        "python": sys.version, "worker_source_sha256": sha256(Path(__file__)),
    })
    write_json(status_root / f"worker{args.rank:02d}.json", provenance)
    transform = T.Compose([
        T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    for sample_id in pending:
        started = time.perf_counter()
        seed = int.from_bytes(hashlib.sha256(f"svr-compare-20261017-seed42:{sample_id}".encode()).digest()[:4], "little")
        record = {"sample_id": sample_id, "method": args.method, "seed": seed, "identity": identity}
        try:
            with np.load(args.condition_root / sample_id / "imgs.npz") as archive:
                pixels = archive["svr_imgs"][0]
            record["image_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
            image_tensor = transform(pixels).unsqueeze(0).to(device)
            record["normalized_image_sha256"] = hashlib.sha256(image_tensor.cpu().numpy().tobytes()).hexdigest()
            if args.backend in ("legacy", "alignment"):
                batch = {"conditions": {"imgs": image_tensor.unsqueeze(1), "img_id": torch.zeros((1, 1), dtype=torch.long, device=device)}}
            else:
                batch = {"conditions": {"imgs": image_tensor}}
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.synchronize(device)
            inference_started = time.perf_counter()
            with torch.no_grad():
                prediction = model.inference(1, device, v_data=batch, v_log=False)[0]
            torch.cuda.synchronize(device)
            record["inference_seconds"] = time.perf_counter() - inference_started
            arrays = {}
            for key in ("pred_face", "pred_edge", "pred_edge_face_connectivity", "pred_face_adj", "pred_face_adj_prob"):
                if key in prediction:
                    value = prediction[key]
                    arrays[key] = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            for key in ("pred_face", "pred_edge", "pred_edge_face_connectivity"):
                if key not in arrays or not np.isfinite(arrays[key]).all():
                    raise ValueError(f"Missing/nonfinite native prediction: {key}")
            destination = output_root / sample_id
            destination.mkdir(parents=True, exist_ok=True)
            raw = destination / "data.npz"
            if raw.exists():
                raise FileExistsError(f"Refusing to overwrite untracked prediction: {raw}")
            np.savez_compressed(raw, **arrays)
            record.update(status="ok", output_sha256=sha256(raw), shapes={key: list(value.shape) for key, value in arrays.items()})
        except Exception as error:
            record.update(status="error", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        record["total_seconds"] = time.perf_counter() - started
        write_json(status_root / f"{sample_id}.json", record)
        print(json.dumps({key: record.get(key) for key in ("sample_id", "method", "status", "inference_seconds", "error")}), flush=True)


if __name__ == "__main__":
    main()
