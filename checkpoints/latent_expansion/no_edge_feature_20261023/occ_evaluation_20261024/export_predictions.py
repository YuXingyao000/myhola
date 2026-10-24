"""Export the fixed no-feature 64-D checkpoint for unchanged OCC processing."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DIAGNOSIS = ROOT / "checkpoints/latent_expansion/diagnosis_20261021"
PREVIOUS = HERE.parent / "evaluation_20261024"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DIAGNOSIS))

import numpy as np
from run_diagnostics import (
    AutoEncoder_joint_dataset, as_numpy, dataset_config, evaluate_prediction,
    item_seed, load_model, move, seed_all, sha256, torch, write_json,
)

CHECKPOINT_SHA = "4262f4998fe047e19c3ca21a835d74c183b7430f7a656c709ec8ed045262ada4"
SPLIT_SHA = "ae607bfe5499e919d1e68a5110c0c211daf1158aedbe57a4756eda99f6f3f0ed"
METHOD = "no_edge_feature64"
CD_FIELDS = ("face_cd_sq", "edge_cd_sq", "face_pred_to_gt_sq", "face_gt_to_pred_sq",
             "edge_pred_to_gt_sq", "edge_gt_to_pred_sq")
ATOL, RTOL = 1e-9, 1e-3


def array_digest(array):
    digest = hashlib.sha256(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def previous_rows():
    rows = {}
    for path in sorted((PREVIOUS / "runs/full/no_edge_feature_last64").glob("shard_*/metrics.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["id"] in rows or row["status"] != "ok":
                raise ValueError("Prior mean result is duplicate or failed")
            rows[row["id"]] = row
    if len(rows) != 3495:
        raise ValueError("Expected 3495 completed prior mean results")
    return rows


def verify_prediction(arrays, face_count):
    if arrays["face_latents"].dtype != np.float32 or arrays["face_latents"].shape != (face_count, 64):
        raise ValueError("Incorrect 64-D posterior shape/dtype")
    if arrays["pred_face"].shape != (face_count, 16, 16, 3):
        raise ValueError("Incorrect face geometry shape")
    edge_count = len(arrays["pred_edge"])
    if arrays["pred_edge"].shape != (edge_count, 16, 3):
        raise ValueError("Incorrect curve geometry shape")
    connectivity = arrays["pred_edge_face_connectivity"]
    if connectivity.shape != (edge_count, 3) or not np.issubdtype(connectivity.dtype, np.integer):
        raise ValueError("Incorrect connectivity")
    if not np.array_equal(connectivity[:, 0], np.arange(edge_count)):
        raise ValueError("Nonsequential exported edge indices")
    if (connectivity[:, 1:] < 0).any() or (connectivity[:, 1:] >= face_count).any():
        raise ValueError("Invalid predicted owner face indices")
    if str(arrays["representation"].item()) != "legacy_half_edges":
        raise ValueError("Incorrect representation")
    for name, array in arrays.items():
        if np.issubdtype(np.asarray(array).dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"Nonfinite {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--uv-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.variant = "no_edge_feature_last64"
    args.source_checkpoint = args.checkpoint
    args.late_checkpoint = args.checkpoint
    ids = args.split.read_text().split()
    if sha256(args.split) != SPLIT_SHA or len(ids) != 3495 or len(set(ids)) != 3495:
        raise ValueError("Validation split changed")
    if sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Checkpoint changed")
    if args.limit is not None:
        ids = ids[:args.limit]
    selected = ids[args.shard_index::args.shard_count]
    metadata = args.output_root / "metadata/export" / f"shard_{args.shard_index:03d}"
    metadata.mkdir(parents=True, exist_ok=False)
    split_path = metadata / "selected_ids.txt"
    split_path.write_text("\n".join(selected) + "\n")
    references = previous_rows()
    prior_audit = json.loads((PREVIOUS / "execution_validation.json").read_text())
    prior_sources = prior_audit["source_hashes"]
    relevant = [name for name in prior_sources if name.startswith("src/brepnet/models/vae/")
                or name in {"src/brepnet/dataset.py", "tools/vae_compare_metrics.py",
                            "checkpoints/latent_expansion/diagnosis_20261021/run_diagnostics.py"}]
    hashes = {name: sha256(ROOT / name) for name in relevant}
    if any(hashes[name] != prior_sources[name] for name in relevant):
        raise ValueError("Model/dataset/diagnostic code differs from previous mean evaluation")
    for path in (Path(__file__), HERE / "launch_export.py", HERE / "SPEC.md"):
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "method": METHOD,
        "schema_version": 1, "mode": "posterior_mean", "input_contract": "same_geometry_source",
        "checkpoint_sha256": CHECKPOINT_SHA, "split_sha256": SPLIT_SHA,
        "selected_ids": selected, "shard_index": args.shard_index, "shard_count": args.shard_count,
        "smoke_limit": args.limit, "command": sys.argv, "source_hashes": hashes,
        "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(args.device),
        "seed": 42, "item_seed": "sha256(42:id:0) first eight hex digits",
        "precision": "float32", "tf32": False, "autocast": False, "augmentation": False,
        "adjacency_threshold": 0.5, "face_deduplication": False,
        "gt_usage": "encoder geometry/topology and measurement labels only; decoder receives own posterior mean",
        "previous_metric_comparison": {"atol": ATOL, "rtol": RTOL, "counts": "exact"},
    }
    (metadata / "git_status.txt").write_text(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True))
    (metadata / "git.diff").write_text(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT, text=True))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed_all(42)
    model, original_data, model_audit = load_model(args)
    model.requires_grad_(False)
    cfg = model_audit["checkpoint_model_config"]
    if model.face_feature_dim != 64 or model_audit["checkpoint_global_step"] != 30000:
        raise ValueError("Wrong latent width or checkpoint step")
    if cfg["freeze_encoder"] or cfg["loss_weights"]["feature_weight"] != 0 or model.edge_threshold != 0.5:
        raise ValueError("Wrong ablation configuration")
    effective_data = dataset_config(original_data, args, split_path)
    dataset = AutoEncoder_joint_dataset("validation", effective_data)
    locations = {name: index for index, name in enumerate(dataset.data_folders)}
    protocol.update(model_audit=model_audit, effective_dataset_config=effective_data)
    protocol_digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    write_json(metadata / "protocol.json", protocol)
    counts = Counter()
    started = time.monotonic()
    with (metadata / "records.jsonl").open("x") as handle:
        for index, identifier in enumerate(selected):
            row = {"id": identifier, "method": METHOD, "status": "failed"}
            try:
                seed_all(item_seed(identifier))
                source = args.data_root / identifier / "data.npz"
                source_digest = sha256(source)
                uv_source = args.uv_root / identifier / "edge_uv.npz"
                batch = move(dataset.collate_fn([dataset[locations[identifier]]]), args.device)
                with torch.no_grad():
                    encoding = model.encode(batch)
                    z, _, _ = model.sample(encoding["face_features"], True)
                    prediction = model.decode_state(z, batch["face_counts"])
                    arrays = model.export_items(prediction, z)[0]
                verify_prediction(arrays, len(batch["face_points"]))
                boundary = batch["boundary_targets"][0]
                measured = evaluate_prediction(arrays, as_numpy(batch["face_points"][..., :3]),
                    as_numpy(boundary["curves"]), boundary["face_pairs"].cpu().numpy(),
                    method="intersection", mode="posterior_mean")
                prior = references[identifier]
                metric_comparison = {}
                for key in CD_FIELDS:
                    actual, expected = measured[key], prior[f"raw_{key}"]
                    matches = bool(np.isclose(actual, expected, atol=ATOL, rtol=RTOL))
                    metric_comparison[key] = {"current": actual, "previous": expected,
                        "abs_difference": abs(actual-expected), "within_tolerance": matches}
                metric_comparison["face_mae"] = {"current": measured["face_mae"],
                    "previous": prior["raw_face_xyz_l1"],
                    "abs_difference": abs(measured["face_mae"]-prior["raw_face_xyz_l1"]),
                    "within_tolerance": bool(np.isclose(measured["face_mae"], prior["raw_face_xyz_l1"], atol=ATOL, rtol=RTOL))}
                count_checks = {key: measured[key] == prior[key] for key in ("topology_tp", "topology_fp", "topology_fn")}
                count_checks.update(face_count=len(arrays["pred_face"]) == prior["face_count"],
                                    edge_count=len(arrays["pred_edge"]) == prior["pred_edge_count"])
                row.update(metric_comparison=metric_comparison, count_checks=count_checks,
                           all_metrics_match=all(value["within_tolerance"] for value in metric_comparison.values()),
                           all_counts_match=all(count_checks.values()))
                # Retain a genuine prediction even if diagnostic reproduction differs;
                # the coordinator prevents OCC launch until the discrepancy is resolved.
                provenance = {"method": METHOD, "comparison_mode": "posterior_mean",
                    "input_source_path": str(source.resolve()), "input_source_sha256": source_digest,
                    "input_source_kind": "checkpoint_posterior_mean", "encoder_checkpoint_sha256": CHECKPOINT_SHA,
                    "input_latent_sha256": array_digest(arrays["face_latents"]), "protocol_sha256": protocol_digest,
                    "uv_source_sha256": sha256(uv_source)}
                if sha256(source) != source_digest:
                    raise ValueError("Source CAD changed during inference")
                arrays.update({key: np.asarray(value) for key, value in provenance.items()})
                path = args.output_root / "raw_predictions/posterior_mean" / METHOD / identifier / "data.npz"
                path.parent.mkdir(parents=True, exist_ok=False)
                with path.with_suffix(".tmp").open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                path.with_suffix(".tmp").replace(path)
                with np.load(path, allow_pickle=False) as saved:
                    if any(not np.array_equal(saved[key], value) for key, value in arrays.items()):
                        raise ValueError("NPZ roundtrip changed exported values")
                row.update(status="ok", raw_path=str(path), raw_sha256=sha256(path), **provenance,
                           faces=len(arrays["pred_face"]), edges=len(arrays["pred_edge"]))
                del batch, encoding, z, prediction
            except Exception as exc:
                row.update(error=repr(exc), traceback=traceback.format_exc())
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            counts[row["status"]] += 1
            if (index+1) % 100 == 0 or index+1 == len(selected):
                progress = {"processed": index+1, "total": len(selected), "counts": dict(counts),
                            "elapsed_seconds": time.monotonic()-started}
                write_json(metadata / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    write_json(metadata / "completion.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "counts": dict(counts), "elapsed_seconds": time.monotonic()-started,
        "source_unchanged": all(sha256(ROOT/name) == digest for name, digest in hashes.items())})
    return int(counts.get("failed", 0) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
