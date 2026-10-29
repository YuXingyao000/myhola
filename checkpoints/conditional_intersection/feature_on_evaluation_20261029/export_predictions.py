"""Frozen validation export for feature-on conditional attention checkpoints."""
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
import torch
from src.brepnet.dataset import AutoEncoder_joint_dataset
from src.brepnet.models.vae import AutoEncoder_intersection
from tools.vae_compare_metrics import evaluate_prediction

SPLIT_SHA = "ae607bfe5499e919d1e68a5110c0c211daf1158aedbe57a4756eda99f6f3f0ed"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def seed(identifier):
    value = int(hashlib.sha256(f"42:{identifier}:0".encode()).hexdigest()[:8], 16)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def move(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict):
        return {key: move(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [move(item) for item in value]
    return value


def verify(arrays, count):
    assert arrays["face_latents"].shape == (count, 32)
    assert arrays["face_latents"].dtype == np.float32
    assert arrays["pred_face"].shape == (count, 16, 16, 3)
    n = len(arrays["pred_edge"])
    assert arrays["pred_edge"].shape == (n, 16, 3)
    conn = arrays["pred_edge_face_connectivity"]
    assert conn.shape == (n, 3) and np.issubdtype(conn.dtype, np.integer)
    assert np.array_equal(conn[:, 0], np.arange(n))
    assert np.all((conn[:, 1:] >= 0) & (conn[:, 1:] < count))
    assert str(arrays["representation"].item()) == "legacy_half_edges"
    for key, value in arrays.items():
        assert not np.asarray(value).dtype.hasobject, key
        if np.issubdtype(value.dtype, np.floating):
            assert np.isfinite(value).all(), key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/d/data/deepcad_v7"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    split = ROOT / "checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt"
    ids = split.read_text().split()
    assert sha(split) == SPLIT_SHA and len(ids) == len(set(ids)) == 3495
    assert sha(args.checkpoint) == args.checkpoint_sha256
    if args.limit:
        ids = ids[:args.limit]
    metadata = args.output_root / "metadata/export" / args.method
    metadata.mkdir(parents=True, exist_ok=False)
    selected = metadata / "ids.txt"
    selected.write_text("\n".join(ids) + "\n")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed("initialization")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    assert checkpoint["global_step"] == args.step
    cfg = OmegaConf.to_container(OmegaConf.create(checkpoint["hyper_parameters"]), resolve=True)
    model_cfg = dict(cfg["model"])
    assert model_cfg["intersection"]["kind"] == "conditional_attention"
    assert model_cfg["loss_weights"]["feature_weight"] == 1.0
    for key in tuple(model_cfg):
        if key.startswith("init_") and key.endswith("checkpoint"):
            model_cfg[key] = None
    model_cfg.update(checkpoint=None, restoring_checkpoint=True, mode="frozen_inference")
    model = AutoEncoder_intersection(model_cfg)
    model.load_state_dict({k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}, strict=True)
    model = model.float().cuda().eval().requires_grad_(False)
    assert model.face_feature_dim == 32 and model.edge_threshold == 0.5
    data_cfg = dict(cfg["dataset"])
    for alias in ("data_root", "condition_names", "condition_root", "scale_factor"):
        data_cfg.pop(alias, None)
    data_cfg.update(raw_data_root=str(args.data_root), edge_uv_root=None,
                    val_dataset=str(selected), condition_name=[], condition_data_root=None,
                    cached_condition=False, is_aug=False, is_overfit=False, epoch_scale_factor=1)
    dataset = AutoEncoder_joint_dataset("validation", data_cfg)
    locations = {name: index for index, name in enumerate(dataset.data_folders)}
    sources = [Path(__file__), HERE / "SPEC.md", ROOT / "src/brepnet/dataset.py",
               ROOT / "tools/vae_compare_metrics.py"] + sorted((ROOT / "src/brepnet/models/vae").rglob("*.py"))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    protocol = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=args.checkpoint_sha256,
        step=args.step, epoch=checkpoint["epoch"], method=args.method, split_sha256=SPLIT_SHA,
        count=len(ids), model_config=model_cfg, dataset_config=data_cfg, command=sys.argv,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        torch=torch.__version__, source_hashes=hashes, precision="FP32; metrics FP64", tf32=False,
        autocast=False, posterior="mean", threshold=0.5, face_deduplication=False,
        seed="sha256(42:id:0) first eight hex digits", started_utc=datetime.now(timezone.utc).isoformat())
    write(metadata / "protocol.json", protocol)
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    del checkpoint
    started = time.monotonic()
    counts = Counter()
    with (metadata / "records.jsonl").open("x") as stream, torch.inference_mode():
        for ordinal, identifier in enumerate(ids):
            row = dict(id=identifier, method=args.method, status="failed")
            tick = time.monotonic()
            try:
                seed(identifier)
                source = args.data_root / identifier / "data.npz"
                source_sha = sha(source)
                batch = move(dataset.collate_fn([dataset[locations[identifier]]]))
                encoded = model.encode(batch)
                latents, _, _ = model.sample(encoded["face_features"], True)
                prediction = model.decode_state(latents, batch["face_counts"])
                arrays = model.export_items(prediction, latents)[0]
                verify(arrays, len(batch["face_points"]))
                boundary = batch["boundary_targets"][0]
                measured = evaluate_prediction(arrays, batch["face_points"][..., :3].cpu().numpy(),
                    boundary["curves"].cpu().numpy(), boundary["face_pairs"].cpu().numpy(),
                    method="intersection", mode="posterior_mean")
                provenance = dict(method=args.method, comparison_mode="posterior_mean",
                    input_source_path=str(source), input_source_sha256=source_sha,
                    input_source_kind="checkpoint_posterior_mean", encoder_checkpoint_sha256=args.checkpoint_sha256,
                    protocol_sha256=digest)
                arrays.update({k: np.asarray(v) for k, v in provenance.items()})
                target = args.output_root / "raw_predictions/posterior_mean" / args.method / identifier / "data.npz"
                target.parent.mkdir(parents=True, exist_ok=False)
                with target.open("xb") as output:
                    np.savez_compressed(output, **arrays)
                with np.load(target, allow_pickle=False) as saved:
                    assert all(np.array_equal(saved[k], v) for k, v in arrays.items())
                assert sha(source) == source_sha
                row.update(measured, **provenance, status="ok", raw_path=str(target), raw_sha256=sha(target))
                del batch, encoded, latents, prediction
            except Exception as error:
                row.update(error=repr(error), traceback=traceback.format_exc())
            row["seconds"] = time.monotonic() - tick
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            counts[row["status"]] += 1
            if ordinal < 2 or (ordinal + 1) % 100 == 0:
                progress = dict(completed=ordinal + 1, total=len(ids), counts=dict(counts), elapsed_seconds=time.monotonic()-started)
                write(metadata / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    result = dict(count=len(ids), counts=dict(counts), elapsed_seconds=time.monotonic()-started,
                  source_unchanged=all(sha(ROOT / k) == v for k, v in hashes.items()),
                  checkpoint_unchanged=sha(args.checkpoint) == args.checkpoint_sha256,
                  completed_utc=datetime.now(timezone.utc).isoformat())
    write(metadata / "completion.json", result)
    print(json.dumps(result), flush=True)
    return int(counts["failed"] > 0 or not result["source_unchanged"] or not result["checkpoint_unchanged"])


if __name__ == "__main__":
    raise SystemExit(main())
