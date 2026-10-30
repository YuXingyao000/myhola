"""Freeze original DINO photo features and validate original-AE latent provenance.

Each GPU writes an isolated shard. The final manifest is published only after all
shards pass; training never reads a partially prepared cache.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.datasets.photo import photo_transform
from src.datasets.preference import read_split, sha256_file
from src.models.diffusion.legacy import load_legacy_svr


class InputDataset(Dataset):
    def __init__(self, records, condition_root, latent_root):
        self.records = records
        self.condition_root = Path(condition_root)
        self.latent_root = Path(latent_root)
        self.transform = photo_transform()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        mid = record["id"]
        photo_path = self.condition_root / mid / "real_photo.npz"
        payload = photo_path.read_bytes()
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            photo = arrays["flux"]
        if photo.ndim != 3 or photo.shape[-1] != 3 or photo.dtype != np.uint8 or min(photo.shape[:2]) < 1:
            raise ValueError(f"Invalid flux photo: {photo_path}, {photo.shape}, {photo.dtype}")
        if photo.shape[0] != photo.shape[1]:
            raise ValueError(f"Baseline expects square source photos; refusing anisotropic resizing: {mid}")
        latent_dir = self.latent_root / f"{mid}_0"
        latent_bytes = (latent_dir / "features.npy").read_bytes()
        source_bytes = (latent_dir / "source.json").read_bytes()
        source = json.loads(source_bytes)
        latent_hash = hashlib.sha256(latent_bytes).hexdigest()
        if (source.get("precision") != "float32" or source.get("layout") != "mean32,std32"
                or source["output_sha256"]["features.npy"] != latent_hash):
            raise ValueError(f"Original AE cache provenance mismatch: {mid}")
        stats = np.load(io.BytesIO(latent_bytes), allow_pickle=False)
        if (stats.ndim != 2 or stats.shape[1] != 64 or not 1 <= len(stats) <= 30
                or stats.dtype != np.float32 or not np.isfinite(stats).all() or (stats[:, 32:] < 0).any()):
            raise ValueError(f"Invalid mean/std cache: {mid}")
        mean = np.zeros((30, 32), dtype=np.float32)
        mean[:len(stats)] = stats[:, :32]
        audit = {**record, "photo_sha256": hashlib.sha256(payload).hexdigest(),
                 "photo_shape": list(photo.shape), "latent_sha256": latent_hash,
                 "latent_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                 "gt_npz_sha256": source["gt_npz_sha256"]}
        return self.transform(photo), mean, len(stats), audit


def collate(items):
    images, means, counts, audit = zip(*items)
    return torch.stack(images), np.stack(means), np.asarray(counts, dtype=np.int16), list(audit)


def worker(args):
    root = Path(args.output)
    protocol = json.loads((root / "preparation.json").read_text())
    records = protocol["records"][args.rank::args.gpus]
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    legacy, provenance = load_legacy_svr(args.checkpoint, source_cache=root / f"legacy_source_{args.rank}")
    encoder = legacy.img_model.requires_grad_(False).eval().cuda()
    del legacy
    dataset = InputDataset(records, args.conditions, Path(args.source_run) / "gt_cache")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2,
                        pin_memory=True, collate_fn=collate, persistent_workers=True)
    features, means, counts, audit = [], [], [], []
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, (images, batch_mean, batch_counts, batch_audit) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            value = encoder(images).float()
            if value.shape != (len(images), 1024) or not torch.isfinite(value).all():
                raise ValueError("DINO produced malformed/nonfinite CLS")
            if batch_index == 0:
                online = encoder(images[:1]).float()
                error = (online-value[:1]).abs().max().item()
                if not torch.allclose(online, value[:1], atol=2e-4, rtol=2e-4):
                    raise ValueError(f"DINO batch/single discrepancy: {error}")
                (root / f"worker_{args.rank}_validation.json").write_text(json.dumps({
                    "max_abs_batch_vs_single": error, "dtype": str(value.dtype),
                    "strict_source_load": provenance["strict_load"],
                    "legacy_model_config": provenance["model_config"],
                    "source_commit": provenance["source_commit"],
                }, indent=2)+"\n")
            features.append(value.cpu().numpy())
            means.append(batch_mean)
            counts.append(batch_counts)
            audit.extend(batch_audit)
            if batch_index % 25 == 0:
                print(f"rank={args.rank} completed={len(audit)}/{len(records)} seconds={time.monotonic()-started:.1f}", flush=True)
    np.savez(root / f"shard_{args.rank}.npz", features=np.concatenate(features),
             means=np.concatenate(means), counts=np.concatenate(counts))
    (root / f"shard_{args.rank}.json").write_text(json.dumps(audit)+"\n")
    print(f"rank={args.rank} complete {len(records)} in {time.monotonic()-started:.1f}s", flush=True)


def prepare(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    train, val = read_split(args.train_split), read_split(args.val_split)
    if not train or not val or set(train) & set(val):
        raise ValueError("Invalid or overlapping splits")
    source_path = Path(args.source_run) / "metadata/protocol.json"
    source = json.loads(source_path.read_text())
    if source["version"] != "svr_preference_generation_v1":
        raise ValueError("Expected original SVR latent cache protocol")
    checkpoint_hash = sha256_file(Path(args.checkpoint))
    if checkpoint_hash != source["checkpoint_sha256"]:
        raise ValueError("Original AE cache and requested checkpoint differ")
    if train != source["splits"]["train"] or val != source["splits"]["validation"]:
        raise ValueError("Requested splits differ from original cached latent protocol")
    records = [{"split": name, "index": i, "id": mid}
               for name, ids in [("training", train), ("validation", val)] for i, mid in enumerate(ids)]
    protocol = {"args": vars(args), "checkpoint_sha256": checkpoint_hash,
                "source_protocol_sha256": sha256_file(source_path), "records": records}
    (root / "preparation.json").write_text(json.dumps(protocol, indent=2)+"\n")
    processes = []
    for rank in range(args.gpus):
        log = (root / f"worker_{rank}.log").open("w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS="2")
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--worker", "--rank", str(rank)]
        processes.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    codes = [process.wait() for process, _ in processes]
    for _, log in processes:
        log.close()
    if any(codes):
        raise RuntimeError(f"Photo cache preparation failed; inspect worker logs: {codes}")
    merge_shards(args)


def merge_shards(args):
    """Validate and pack completed GPU shards without repeating image encoding."""
    root = Path(args.output).resolve()
    if (root / "manifest.json").exists():
        raise ValueError("Refusing to overwrite a completed cache")
    protocol = json.loads((root / "preparation.json").read_text())
    for field in ("checkpoint", "conditions", "source_run", "train_split", "val_split", "gpus", "batch_size"):
        if getattr(args, field) != protocol["args"][field]:
            raise ValueError(f"Merge configuration differs: {field}")
    checkpoint_hash = sha256_file(Path(args.checkpoint))
    if checkpoint_hash != protocol["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed since cache preparation")
    if sha256_file(Path(args.source_run) / "metadata/protocol.json") != protocol["source_protocol_sha256"]:
        raise ValueError("Source protocol changed since preparation")
    records = protocol["records"]
    train, val = read_split(args.train_split), read_split(args.val_split)
    expected = [{"split": name, "index": i, "id": mid}
                for name, ids in [("training", train), ("validation", val)] for i, mid in enumerate(ids)]
    if records != expected:
        raise ValueError("Split changed since preparation")
    manifest = {"version": "legacy_photo_rgb_v1", "complete": True,
                "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": checkpoint_hash,
                "image_key": "flux", "rotation_id": 0, "feature_precision": "float32",
                "preprocessing": "PIL bilinear Resize224, RGB/255, ImageNet normalize; no augmentation",
                "source_protocol_sha256": protocol["source_protocol_sha256"], "splits": {}}
    arrays = {}
    for name, ids in [("training", train), ("validation", val)]:
        arrays[name] = {"features": np.empty((len(ids), 1024), np.float32),
                        "means": np.empty((len(ids), 30, 32), np.float32),
                        "counts": np.empty(len(ids), np.int16)}
    seen = set()
    with (root / "inputs.jsonl").open("w") as audit_out:
        for rank in range(args.gpus):
            audit = json.loads((root / f"shard_{rank}.json").read_text())
            if [{k: record[k] for k in ("split", "index", "id")} for record in audit] != records[rank::args.gpus]:
                raise ValueError(f"Incomplete or reordered shard {rank}")
            with np.load(root / f"shard_{rank}.npz", allow_pickle=False) as shard:
                # NPZ indexing reloads an entire member: materialize each once.
                values = {field: shard[field] for field in ("features", "means", "counts")}
                if any(len(value) != len(audit) for value in values.values()):
                    raise ValueError(f"Shard row count mismatch: {rank}")
                for i, record in enumerate(audit):
                    key = record["split"], record["index"]
                    if key in seen:
                        raise ValueError("Duplicate shard index")
                    seen.add(key)
                    for field in ("features", "means", "counts"):
                        arrays[key[0]][field][key[1]] = values[field][i]
                    audit_out.write(json.dumps(record)+"\n")
    if len(seen) != len(records):
        raise ValueError("Incomplete cache coverage")
    for name, ids in [("training", train), ("validation", val)]:
        files, hashes = {}, {}
        for field, value in arrays[name].items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite final {name}/{field}")
            path = root / f"{name}_{field}.npy"
            np.save(path, value)
            files[field], hashes[field] = path.name, sha256_file(path)
        manifest["splits"][name] = {"ids": ids, "files": files, "sha256": hashes}
    manifest["inputs_sha256"] = sha256_file(root / "inputs.jsonl")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"complete": True, "train": len(train), "validation": len(val), "output": str(root)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt")
    parser.add_argument("--conditions", default="/mnt/d/data/deepcad_v7_cond")
    parser.add_argument("--source-run", default="/mnt/d/python/checkpoints/diffusion/preference_data/20261020T134930Z")
    parser.add_argument("--train-split", default="src/preprocess/splits/deduplicated_deepcad_training_7_30.txt")
    parser.add_argument("--val-split", default="src/preprocess/splits/deepcad_v7_validation100.txt")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--merge-existing", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    if args.gpus < 1 or args.batch_size < 1:
        parser.error("gpus and batch-size must be positive")
    if args.worker:
        worker(args)
    elif args.merge_existing:
        merge_shards(args)
    else:
        prepare(args)


if __name__ == "__main__":
    main()
