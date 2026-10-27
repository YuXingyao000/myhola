"""Fixed final-checkpoint reconstruction evaluation; no OCC adaptation or training."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from omegaconf import OmegaConf
from scipy.interpolate import BSpline
import torch

from src.brepnet.dataset import AutoEncoder_joint_dataset
from src.brepnet.models.vae.uv_intersection_training import AutoEncoder_uv_intersection, _training_targets
from metrics import measure

CHECKPOINT_SHA = "c4f409b1a904d431cf82cf4754001ca590b59dc44abbaf987181819977a32318"
SPLIT_SHA = "ae607bfe5499e919d1e68a5110c0c211daf1158aedbe57a4756eda99f6f3f0ed"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def move(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict):
        return {key: move(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [move(item) for item in value]
    return value


def seed(identifier="initialization"):
    value = int(hashlib.sha256(f"42:{identifier}:0".encode()).hexdigest()[:8], 16)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def verify_export(arrays):
    """Independent float64 SciPy evaluation of the saved surface and UV."""
    for key, value in arrays.items():
        if np.asarray(value).dtype.hasobject:
            raise ValueError(f"Object array in {key}")
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite output in {key}")
    poles = arrays["pred_surface_poles"].astype(np.float64)
    basis_u = BSpline(arrays["surface_knots_u"], np.eye(16), int(arrays["surface_degree"]))
    basis_v = BSpline(arrays["surface_knots_v"], np.eye(16), int(arrays["surface_degree"]))
    sample = np.linspace(0, 1, 16)
    grid = np.einsum("vb,fbac,ua->fvuc", basis_v(sample), poles, basis_u(sample))
    grid_error = float(np.max(np.abs(grid - arrays["pred_face"])))
    uv = arrays["pred_edge_uv"]
    if uv.size:
        owners = arrays["pred_edge_owner_pairs"]
        xyz = np.einsum("...ta,...bac,...tb->...tc", basis_u(uv[..., 0]), poles[owners], basis_v(uv[..., 1]))
        xyz_error = float(np.max(np.abs(xyz - arrays["pred_edge_xyz_by_face"])))
    else:
        xyz_error = 0.0
    # FP32 fitting/evaluation is compared with an independent FP64 computation.
    if grid_error > 1e-4 or xyz_error > 1e-4:
        raise ValueError(f"Surface roundtrip mismatch: grid={grid_error}, curve={xyz_error}")
    return {"surface_grid_roundtrip_max_abs": grid_error, "surface_uv_roundtrip_max_abs": xyz_error}


def predict(model, batch):
    encoded = model.encode(batch)
    latents, kl, statistics = model.sample(encoded["face_features"], v_is_test=True)
    prediction = model.decode(latents, batch["face_counts"])
    losses = model.criterion(prediction, _training_targets(batch))
    losses["weighted_kl_loss"] = kl
    losses["total_loss"] = losses.pop("total") + kl
    arrays = model.export_training_items(prediction, latents, statistics, batch)["items"][0]
    lookup = {tuple(pair): index for index, pair in enumerate(arrays["all_candidate_pairs"].tolist())}
    rows = [lookup[tuple(pair)] for pair in arrays["gt_edge_owner_pairs"].tolist()]
    arrays["diagnostic_gt_pair_xyz"] = prediction["edge_xyz"][rows].cpu().numpy()
    arrays["diagnostic_note"] = np.asarray("GT pair selection is measurement only; raw predictions use probability > 0.5.")
    values = {f"loss_{key}": float(value) for key, value in losses.items()}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Nonfinite loss")
    return arrays, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Implementation smoke only; total IDs before sharding")
    args = parser.parse_args()
    ids = args.split.read_text().split()
    if len(ids) != 3495 or len(set(ids)) != 3495 or sha256(args.split) != SPLIT_SHA:
        raise ValueError("Frozen validation list changed")
    if sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Final checkpoint identity changed")
    if args.limit is not None:
        ids = ids[:args.limit]
    selected = ids[args.shard_index::args.shard_count]
    out = args.output_root / "shards" / f"shard_{args.shard_index:03d}"
    out.mkdir(parents=True, exist_ok=False)
    split_path = out / "intended_ids.txt"
    split_path.write_text("\n".join(selected) + "\n")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint["global_step"] != 30000 or checkpoint["hyper_parameters"]["model"]["freeze_encoder"]:
        raise ValueError("Expected final full-training 30k checkpoint")
    cfg = OmegaConf.to_container(OmegaConf.create(checkpoint["hyper_parameters"]), resolve=True)
    model_cfg = dict(cfg["model"])
    model_cfg.update(init_uv_decoder_checkpoint=None, checkpoint=str(args.checkpoint), mode="frozen_inference")
    model = AutoEncoder_uv_intersection(model_cfg).cuda().eval()
    available = [identifier for identifier in selected if (args.data_root / identifier / "data.npz").is_file()]
    available_path = out / "available_ids.txt"
    available_path.write_text("\n".join(available) + "\n")
    data_cfg = dict(cfg["dataset"])
    data_cfg.update(raw_data_root=str(args.data_root), val_dataset=str(available_path), is_aug=0,
                    is_overfit=False, epoch_scale_factor=1, edge_uv_root=None)
    dataset = AutoEncoder_joint_dataset("validation", data_cfg) if available else None
    locations = {name: index for index, name in enumerate(dataset.data_folders)} if dataset else {}
    sources = [Path(__file__), HERE / "metrics.py", HERE / "SPEC.md", ROOT / "tools/vae_compare_metrics.py",
               ROOT / "src/brepnet/dataset.py", ROOT / "src/brepnet/data/shared_boundaries.py"]
    sources += sorted((ROOT / "src/brepnet/models/vae").rglob("*.py"))
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "checkpoint_sha256": CHECKPOINT_SHA,
        "checkpoint": str(args.checkpoint), "checkpoint_global_step": checkpoint["global_step"],
        "checkpoint_epoch": checkpoint["epoch"], "split_sha256": SPLIT_SHA,
        "intended_ids": selected, "shard_index": args.shard_index, "shard_count": args.shard_count,
        "smoke_limit": args.limit, "model_config": model_cfg, "dataset_config": data_cfg,
        "initialization_report": model.initialization_report, "command": sys.argv,
        "precision": "float32 model; float64 metrics", "tf32": False, "autocast": False,
        "latent": "posterior_mean", "batch_size": 1, "seed": 42, "threshold": model.edge_threshold,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "occ_status": "not_evaluated_native_dual_uv_constructor_not_implemented",
    }
    write_json(out / "protocol.json", protocol)
    (out / "git_status.txt").write_text(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True))
    del checkpoint
    counts, started = Counter(), time.monotonic()
    with (out / "metrics.jsonl").open("x") as stream, torch.inference_mode():
        for ordinal, identifier in enumerate(selected):
            row = {"id": identifier, "status": "failed"}
            tick = time.monotonic()
            try:
                source = args.data_root / identifier / "data.npz"
                if identifier not in locations:
                    raise FileNotFoundError(source)
                seed(identifier)
                source_digest = sha256(source)
                batch = move(dataset.collate_fn([dataset[locations[identifier]]]))
                arrays, losses = predict(model, batch)
                checks = verify_export(arrays)
                values = measure(arrays)
                if values["gt_pair_curve_xyz_mse"] is not None:
                    if not np.isclose(values["gt_pair_curve_xyz_mse"], losses["loss_curve_xyz"], rtol=1e-5, atol=1e-7):
                        raise ValueError("Independent GT-pair metric differs from curve loss")
                    if not np.isclose(values["gt_pair_gap_rms"] ** 2 / 3, losses["loss_curve_incidence"], rtol=1e-5, atol=1e-7):
                        raise ValueError("Independent dual-side gap differs from incidence loss")
                arrays.update(input_source_sha256=np.asarray(source_digest),
                              encoder_checkpoint_sha256=np.asarray(CHECKPOINT_SHA))
                path = args.output_root / "raw_predictions" / identifier / "data.npz"
                path.parent.mkdir(parents=True, exist_ok=False)
                with path.open("xb") as target:
                    np.savez_compressed(target, **arrays)
                with np.load(path, allow_pickle=False) as saved:
                    if any(not np.array_equal(saved[key], value) for key, value in arrays.items()):
                        raise ValueError("NPZ roundtrip changed exported values")
                if sha256(source) != source_digest:
                    raise ValueError("Input changed during evaluation")
                row.update(values, **losses, **checks, status="ok", input_source_sha256=source_digest,
                           raw_path=str(path), raw_sha256=sha256(path))
            except Exception as error:
                row.update(error=repr(error), traceback=traceback.format_exc())
            row["seconds"] = time.monotonic() - tick
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            counts[row["status"]] += 1
            if ordinal < 2 or (ordinal + 1) % 25 == 0:
                print(json.dumps({"completed": ordinal + 1, "total": len(selected), "counts": counts,
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
    completion = {"status": "completed", "intended_count": len(selected), "counts": dict(counts),
                  "elapsed_seconds": time.monotonic() - started, "checkpoint_sha256": CHECKPOINT_SHA}
    write_json(out / "completion.json", completion)
    print(json.dumps(completion), flush=True)
    if counts["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
