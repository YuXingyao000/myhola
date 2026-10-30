"""固定旧seed采样新版last/best；复用当前网络与DDPM实现。"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BASE = ROOT.parent / "uncond_baseline_20261018"
WEIGHTS = REPO / "checkpoints/diffusion_uncond/20261018T085204.415712Z/weights"
CHECKPOINTS = {"new_last": WEIGHTS / "last.ckpt",
               "new_best": WEIGHTS / "epoch=0090-step=00276185-val_loss=0.027705.ckpt"}
sys.path.insert(0, str(REPO))

from checkpoints.diffusion.uncond_baseline_20261018.infer import (
    pending_samples, sample_ids, sample_seed, sha256, utc_now, write_json,
)


def prepare():
    import torch
    base_manifest = json.loads((BASE / "metadata/manifest.json").read_text())
    for name, path in CHECKPOINTS.items():
        root = ROOT / name
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        manifest = {**base_manifest, "checkpoint": {
            "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
            "global_step": checkpoint["global_step"], "epoch": checkpoint["epoch"]},
            "plan_sha256": sha256(ROOT / "PLAN.md")}
        target = root / "metadata/manifest.json"
        if target.exists():
            assert json.loads(target.read_text()) == manifest
        write_json(target, manifest)
        identity = {"manifest_sha256": sha256(target), "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                    "sample_source_sha256": sha256(Path(__file__))}
        write_json(root / "metadata/inference/protocol.json", {"identity": identity})
    references = Path(REPO / "src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt").read_text().splitlines()
    references = sorted(references, key=lambda name: hashlib.sha256(("uncond-ref-20261019:" + name).encode()).hexdigest())[:1000]
    (ROOT / "reference_ids.txt").write_text("\n".join(references) + "\n")


def load_model(path, torch):
    from diffusers import DDPMScheduler
    from omegaconf import OmegaConf
    from src.brepnet.models import build_model
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    cfg = OmegaConf.create(checkpoint["hyper_parameters"])
    cfg.model.autoencoder.checkpoint = None  # 完整checkpoint已包含AE。
    with torch.device("meta"):
        model = build_model(cfg.model, cfg.condition)
    model.load_state_dict({k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}, strict=True, assign=True)
    model.noise_scheduler = DDPMScheduler.from_config(model.noise_scheduler.config)
    reference = torch.load("/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt", map_location="cpu", mmap=True, weights_only=False)
    ae = model.autoencoder.state_dict()
    assert len(ae) == 704 and all(torch.equal(ae[k.removeprefix("model.")], v) for k, v in reference["state_dict"].items())
    assert model.condition_encoder is None and len(model.denoiser.cross_attention_layers) == 0
    return model.float().cuda().eval().requires_grad_(False)


def worker(args):
    import numpy as np
    import torch
    from src.brepnet.diffusion.sampling import sample_latent_sequence
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    root = ROOT / args.method
    identity = json.loads((root / "metadata/inference/protocol.json").read_text())["identity"]
    selected = sample_ids(args.limit)[args.rank::args.shards]
    pending = pending_samples(root, selected, identity)
    if not pending:
        return
    model = load_model(CHECKPOINTS[args.method], torch)
    for name in pending:
        record = {"schema_version": 1, "sample_id": name, "seed": sample_seed(name), "identity": identity,
                  "status": "running", "stage": "sampling", "started_at_utc": utc_now()}
        status_path = root / "metadata/inference/samples" / f"{name}.json"
        write_json(status_path, record)
        started = time.perf_counter()
        try:
            random.seed(record["seed"]); np.random.seed(record["seed"])
            torch.manual_seed(record["seed"]); torch.cuda.manual_seed_all(record["seed"])
            # 与旧模型共享CPU初始noise和后续CUDA RNG起点。
            initial = torch.randn(1, 30, 32)
            noise_hash = hashlib.sha256(initial.numpy().tobytes()).hexdigest()
            old_record = json.loads((BASE / "metadata/inference/samples" / f"{name}.json").read_text())
            assert noise_hash == old_record["initial_noise"]["sha256"]
            record["initial_noise"] = {"sha256": noise_hash, "shape": [1,30,32], "dtype": "float32", "device": "cpu"}
            torch.cuda.synchronize()
            sample_started = time.perf_counter()
            with torch.inference_mode():
                latents = sample_latent_sequence(model.diffuse, model.noise_scheduler, initial.cuda(), None,
                    model.oracle_topology_self_attention, None, show_progress=False)
                latents = model.face_padder.postprocess(latents[0])
                prediction = model.autoencoder.decode_latents(latents)
            torch.cuda.synchronize()
            record["inference_seconds"] = time.perf_counter() - sample_started
            arrays = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v) for k,v in prediction.items()}
            raw = root / "raw_predictions" / name / "data.npz"
            raw.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(raw, **arrays)
            record.update(output_path=str(raw), output_sha256=sha256(raw), raw_face_count=len(arrays["pred_face"]),
                          raw_directed_edge_count=len(arrays["pred_edge"]),
                          shapes={k:list(v.shape) for k,v in arrays.items()}, dtypes={k:str(v.dtype) for k,v in arrays.items()})
            assert all(np.isfinite(arrays[k]).all() for k in ("pred_face", "pred_edge", "pred_edge_face_connectivity"))
            record.update(status="ok", stage="complete")
        except Exception as error:
            record.update(status="error", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        record.update(total_seconds=time.perf_counter()-started, finished_at_utc=utc_now())
        write_json(status_path, record)
        print(json.dumps({k:record.get(k) for k in ("sample_id","status","raw_face_count","inference_seconds","error")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=list(CHECKPOINTS))
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.launch:
        jobs = []
        for name, offset in (("new_last",0),("new_best",4)):
            for rank in range(4):
                log = ROOT / name / "metadata" / f"inference_gpu{offset+rank}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open("a")
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(offset+rank), OMP_NUM_THREADS="1",
                                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                command = [sys.executable, str(Path(__file__)), "--method", name, "--rank",str(rank),"--shards","4"]
                jobs.append((subprocess.Popen(command, cwd=REPO, env=environment, stdout=handle, stderr=subprocess.STDOUT),handle))
        codes=[]
        for process, handle in jobs:
            codes.append(process.wait()); handle.close()
        assert not any(codes), codes
    else:
        worker(args)


if __name__ == "__main__":
    main()
