"""Frozen-mean validation and curve decomposition for the no-feature ablation.

This reuses the original diagnosis metrics and decomposition without modifying
the model, dataset, loss, or post-processing. See SPEC.md for the protocol.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DIAGNOSIS = ROOT / "checkpoints/latent_expansion/diagnosis_20261021"
sys.path.insert(0, str(DIAGNOSIS))
sys.path.insert(0, str(DIAGNOSIS / "shape_placement"))

from run_diagnostics import (
    AutoEncoder_joint_dataset, FeatureMoments, append_row, dataset_config,
    item_seed, load_model, measure_prediction, move, seed_all, sha256, torch,
    write_json,
)
from run_decomposition import path_decomposition


def source_hashes():
    paths = [
        Path(__file__), HERE / "SPEC.md", DIAGNOSIS / "run_diagnostics.py",
        DIAGNOSIS / "shape_placement/run_decomposition.py",
        DIAGNOSIS / "shape_placement/metrics.py",
        ROOT / "src/brepnet/dataset.py", ROOT / "tools/vae_compare_metrics.py",
    ]
    paths += sorted((ROOT / "src/brepnet/models/vae").rglob("*.py"))
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("source32", "no_edge_feature_last64"), required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--late-checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--uv-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Per-shard implementation smoke only")
    args = parser.parse_args()
    out = args.out / f"shard_{args.shard_index:03d}"
    out.mkdir(parents=True, exist_ok=False)
    all_ids = [line.strip() for line in args.split.read_text().splitlines() if line.strip()]
    selected = all_ids[args.shard_index::args.shard_count]
    if args.limit is not None:
        selected = selected[:args.limit]
    (out / "intended_ids.txt").write_text("\n".join(selected) + "\n")
    protocol = {
        "status": "initializing", "variant": args.variant, "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": str(args.split.resolve()), "split_sha256": sha256(args.split),
        "split_count": len(all_ids), "intended_ids": selected,
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "smoke_limit": args.limit, "seed": 42, "batch_size": 1,
        "precision": "float32 model/composition; float64 direct-difference metric",
        "tf32": False, "model_mode": "eval", "latent": "posterior_mean",
        "source_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes": source_hashes(), "torch": torch.__version__,
        "command": sys.argv,
        "raw_cd_definition": "bidirectional NN squared mean sum across full point sets; CAD macro",
        "matched_cd_definition": "bidirectional NN squared mean sum per GT edge; edge mean per CAD",
        "partial_alignment": "valid-entity-only case means with counts and same-subset baseline",
    }
    write_json(out / "protocol.json", protocol)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed_all(42)
    model, data_cfg, metadata = load_model(args)
    protocol.update(metadata)
    protocol["gpu"] = torch.cuda.get_device_name(args.device)
    available, missing = [], {}
    for identifier in selected:
        paths = (args.data_root / identifier / "data.npz", args.uv_root / identifier / "edge_uv.npz")
        absent = [str(path) for path in paths if not path.is_file()]
        if absent:
            missing[identifier] = "Missing inputs: " + ", ".join(absent)
        else:
            available.append(identifier)
    available_split = out / "available_ids.txt"
    available_split.write_text("\n".join(available) + "\n")
    effective_data = dataset_config(data_cfg, args, available_split)
    protocol["effective_dataset_config"] = effective_data
    dataset = AutoEncoder_joint_dataset("validation", effective_data) if available else None
    locations = {name: index for index, name in enumerate(dataset.data_folders)} if dataset else {}
    protocol["status"] = "running"
    write_json(out / "protocol.json", protocol)
    counts, moments, edge_total = Counter(), FeatureMoments(), 0
    tick = time.monotonic()
    with (out / "metrics.jsonl").open("x") as handle:
        for ordinal, identifier in enumerate(selected):
            row = {"id": identifier, "variant": args.variant, "status": "ok"}
            try:
                if identifier in missing:
                    raise FileNotFoundError(missing[identifier])
                seed_all(item_seed(identifier))
                batch = move(dataset.collate_fn([dataset[locations[identifier]]]), args.device)
                with torch.no_grad():
                    encoding = model.encode(batch)
                    z, _, statistics = model.sample(encoding["face_features"], True)
                    mean, std = statistics.chunk(2, dim=-1)
                    teacher = model.decoder.edge_geometry(encoding["edge_features"])
                    prediction = model.decode_state(z, batch["face_counts"])
                    values, targets = measure_prediction(model, batch, encoding, teacher, z, prediction)
                    row.update(values)
                    rows, edge_ids = targets.candidate_rows, targets.edge_ids
                    refined_batch = model.criterion._shared_curve_batch(prediction, batch, targets)
                    row.update(
                        edge_count=len(edge_ids), teacher_input_edge_count=len(batch["edge_points"]),
                        posterior_std_rms=float(std.square().mean().sqrt()),
                        posterior_mean_rms=float(mean.square().mean().sqrt()),
                        paths={
                            "teacher": path_decomposition(teacher, batch, edge_ids, edge_ids),
                            "coarse": path_decomposition(prediction.coarse.edge_geometry, batch, edge_ids, rows),
                            "refined": path_decomposition(prediction.refined.edge_geometry, refined_batch, edge_ids, rows),
                        },
                    )
                    json.dumps(row, allow_nan=False)
                    feature_sets = {
                        "teacher": encoding["edge_features"][edge_ids],
                        "coarse": prediction.coarse.edge_features[rows],
                        "encoder_face": encoding["face_features"], "z_mean": mean, "z_std": std,
                    }
                    if not all(torch.isfinite(value).all() for value in feature_sets.values()):
                        raise ValueError("Nonfinite feature statistics")
                    for name, tensor in feature_sets.items():
                        moments.update(name, tensor)
                    edge_total += len(edge_ids)
                del batch, encoding, z, statistics, mean, std, teacher, prediction, targets, refined_batch, feature_sets
            except Exception as exc:
                row = {"id": identifier, "variant": args.variant, "status": "failed",
                       "error": repr(exc), "traceback": traceback.format_exc()}
            append_row(handle, row)
            counts[row["status"]] += 1
            if ordinal % 25 == 0 or ordinal + 1 == len(selected):
                moments.save(out / "moments.npz")
                progress = {"processed": ordinal + 1, "intended": len(selected),
                            "status_counts": dict(counts), "gt_edge_count": edge_total,
                            "elapsed_seconds": time.monotonic() - tick}
                write_json(out / "progress.json", progress)
                print(json.dumps({"variant": args.variant, "shard": args.shard_index, **progress}), flush=True)
    protocol.update(status="complete", status_counts=dict(counts), gt_edge_count=edge_total,
                    elapsed_seconds=time.monotonic() - tick,
                    completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / "protocol.json", protocol)


if __name__ == "__main__":
    main()
