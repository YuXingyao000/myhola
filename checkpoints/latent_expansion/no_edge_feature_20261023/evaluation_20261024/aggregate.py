"""Aggregate the frozen 30k no-feature ablation; never execute a model."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DIAG = ROOT / "checkpoints/latent_expansion/diagnosis_20261021"
VARIANT = "no_edge_feature_last64"
SPLIT_SHA = "ae607bfe5499e919d1e68a5110c0c211daf1158aedbe57a4756eda99f6f3f0ed"
SOURCE_SHA = "ce749416c1635dda1563161b971bd4260d064cfdb0c31f1fa9d06ad0e70961ff"
EXPECTED_CADS, EXPECTED_EDGES, EXPECTED_SHARDS = 3495, 199092, 8


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


shape = module("ablation_shape_aggregation", DIAG / "shape_placement/aggregate.py")
old = module("ablation_raw_aggregation", DIAG / "aggregate.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(paths):
    return [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_ids(rows, ids, label):
    counts = Counter(row["id"] for row in rows)
    require(counts == Counter(ids), f"{label}: missing, unexpected or repeated CAD IDs")


def pair(before, after, preferred="lower"):
    result = shape.paired_stats(before, after)
    # Preserve a complete schema even when there are no valid paired metrics.
    if not result["count"]:
        result.update(improved=0, worsened=0, equal=0, mean_ratio=None)
    result["relative_change"] = result["mean_ratio"] - 1 if result.get("mean_ratio") is not None else None
    result["decreased"] = result.pop("improved")
    result["increased"] = result.pop("worsened")
    result["preferred_direction"] = preferred
    if preferred == "lower":
        result.update(improved=result["decreased"], worsened=result["increased"])
    elif preferred == "higher":
        result.update(improved=result["increased"], worsened=result["decreased"])
    return result


def direction(key):
    if key in ("feature_cosine", "topology_precision", "topology_recall", "topology_f1", "uv_valid_fraction"):
        return "higher"
    if key.endswith(("_l1", "_sq", "_relative_error")) or key in ("uv", "uv_anchor"):
        return "lower"
    return "unspecified"


def topology(rows):
    values = {name: int(sum(row.get("topology_" + name, 0) for row in rows))
              for name in ("tp", "fp", "fn")}
    tp, fp, fn = (values[name] for name in ("tp", "fp", "fn"))
    values.update(precision=tp / (tp + fp) if tp + fp else None,
                  recall=tp / (tp + fn) if tp + fn else None,
                  f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
    return values


def load_source(ids, previous, inputs):
    directory = DIAG / "shape_placement/runs/20261021T123732Z/source32"
    paths = sorted(directory.glob("shard_*/metrics.jsonl"))
    require(bool(paths), "Missing source32 shape/placement records")
    path_rows = read_rows(paths)
    check_ids(path_rows, ids, "source32 shape/placement")
    by_id = {row["id"]: row for row in path_rows}
    for path in paths:
        protocol = json.loads(path.with_name("protocol.json").read_text())
        require(protocol["status"] == "complete" and protocol["variant"] == "source32",
                "Incomplete or wrong source32 shape/placement protocol")
        require(protocol["split_sha256"] == SPLIT_SHA and protocol["checkpoint_sha256"] == SOURCE_SHA,
                "Source32 shape/placement split/checkpoint mismatch")
        inputs[str(path.relative_to(ROOT))] = digest(path)
    source_csv = DIAG / "per_object.csv"
    source = []
    with source_csv.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["variant"] != "source32":
                continue
            item = {key: value if key in ("id", "variant", "status", "face_status", "edge_status")
                    else float(value) if value else None for key, value in row.items()}
            path_row = by_id[row["id"]]
            require(item["status"] == path_row["status"], "Source raw/path status disagreement")
            if item["status"] == "ok":
                item.update(paths=path_row["paths"], edge_count=path_row["edge_count"])
            source.append(item)
    check_ids(source, ids, "source32 raw CSV")
    protocols = previous["models"]["source32"]["protocols"]
    require(all(p["checkpoint_sha256"] == SOURCE_SHA and p["split_sha256"] == SPLIT_SHA
                and p["status"] == "complete" for p in protocols), "Source32 raw protocol mismatch")
    require(sum(row.get("edge_count", 0) for row in source) == EXPECTED_EDGES,
            "Source32 does not cover the fixed 199092 GT edges")
    inputs[str(source_csv.relative_to(ROOT))] = digest(source_csv)
    return source


def load_new(run, ids, expected_sha, inputs):
    directory = run / VARIANT
    shards = sorted(directory.glob("shard_*"))
    require(len(shards) == EXPECTED_SHARDS and all(path.is_dir() for path in shards),
            "Expected exactly eight full-evaluation shard directories")
    protocols, records = [], []
    for shard in shards:
        protocol_file, metrics_file = shard / "protocol.json", shard / "metrics.jsonl"
        protocol = json.loads(protocol_file.read_text())
        require(protocol["status"] == "complete", f"Incomplete shard: {shard}")
        require(protocol["variant"] == VARIANT and protocol["split_sha256"] == SPLIT_SHA,
                f"Wrong variant/split: {shard}")
        require(protocol["checkpoint_global_step"] == 30000, f"Not the fixed 30k checkpoint: {shard}")
        require(protocol["shard_count"] == EXPECTED_SHARDS and protocol["split_count"] == EXPECTED_CADS,
                f"Wrong shard/split count: {shard}")
        require(protocol.get("smoke_limit") is None, f"Smoke result passed to full aggregation: {shard}")
        selected = ids[protocol["shard_index"]::EXPECTED_SHARDS]
        require(protocol["intended_ids"] == selected, f"Shard CAD assignment mismatch: {shard}")
        rows = read_rows([metrics_file])
        check_ids(rows, selected, str(shard))
        require(all(row["variant"] == VARIANT for row in rows), f"Wrong row variant: {shard}")
        require(dict(Counter(r["status"] for r in rows)) == protocol["status_counts"],
                f"Shard status counts disagree: {shard}")
        require(sum(r.get("edge_count", 0) for r in rows if r["status"] == "ok") == protocol["gt_edge_count"],
                f"Shard edge count disagrees: {shard}")
        for path in (protocol_file, metrics_file, shard / "moments.npz"):
            inputs[str(path.resolve())] = digest(path)
        protocols.append(protocol)
        records.extend(rows)
    require(sorted(p["shard_index"] for p in protocols) == list(range(EXPECTED_SHARDS)),
            "Missing/repeated shard index")
    # Compare nested source hashes and model config as well as flat metadata.
    for key in ("checkpoint_sha256", "checkpoint_global_step", "source_hashes", "effective_model_config",
                "precision", "tf32", "model_mode", "latent", "batch_size", "seed"):
        first = protocols[0].get(key)
        require(first is not None and all(p.get(key) == first for p in protocols),
                f"Missing or inconsistent shard field: {key}")
    actual_sha = protocols[0]["checkpoint_sha256"]
    require(len(actual_sha) == 64 and all(c in "0123456789abcdef" for c in actual_sha), "Invalid checkpoint SHA256")
    if expected_sha is not None:
        require(actual_sha == expected_sha, "Checkpoint SHA256 differs from CLI expectation")
    require(protocols[0]["effective_model_config"]["loss_weights"]["feature_weight"] == 0,
            "Feature matching is not disabled in the evaluated model")
    require(protocols[0]["model_mode"] == "eval" and protocols[0]["latent"] == "posterior_mean"
            and protocols[0]["tf32"] is False and protocols[0]["batch_size"] == 1
            and protocols[0]["seed"] == 42, "Evaluation protocol changed")
    check_ids(records, ids, VARIANT)
    return records, protocols, directory


def alignment_subset(lookups, ids):
    """Both models must align every matched edge of a CAD for this comparison."""
    result = {}
    for path in shape.PATHS:
        common = [i for i in ids if all(i in rows and shape.complete_case_metric(
            rows[i], path, "aligned_translation_scale", "cd_sq") is not None for rows in lookups.values())]
        result[path] = {
            "count": len(common), "intended_count": len(ids), "excluded_ids": sorted(set(ids) - set(common)),
            "criterion": "both models have valid translation/positive-scale alignment for every matched edge of the CAD",
            "variants": {variant: {"cases": {case: {metric: shape.distribution([
                rows[i]["paths"][path]["cases"][case][metric] for i in common])
                for metric in ("cd_sq", "xyz_l1")} for case in (
                    "pred_all", "aligned_translation", "aligned_translation_scale")}}
                for variant, rows in lookups.items()},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "runs/full")
    parser.add_argument("--checkpoint-sha256", help="Expected new checkpoint hash; otherwise require all eight shards agree")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    require(not any((args.out / name).exists() for name in ("summary.json", "per_object.csv.gz")),
            "Output exists; choose a new --out directory rather than overwrite a result")
    split = ROOT / "checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt"
    ids = [line.strip() for line in split.read_text().splitlines() if line.strip()]
    require(len(ids) == len(set(ids)) == EXPECTED_CADS and digest(split) == SPLIT_SHA,
            "The frozen validation split changed")
    inputs = {str(split.relative_to(ROOT)): digest(split)}
    previous_file = DIAG / "summary.json"
    previous = json.loads(previous_file.read_text())
    require(previous["split_sha256"] == SPLIT_SHA, "Historical summary split mismatch")
    inputs[str(previous_file.relative_to(ROOT))] = digest(previous_file)
    previous_shape_file = DIAG / "shape_placement/summary.json"
    previous_shape = json.loads(previous_shape_file.read_text())
    require(previous_shape["split_sha256"] == SPLIT_SHA, "Historical shape summary split mismatch")
    inputs[str(previous_shape_file.relative_to(ROOT))] = digest(previous_shape_file)
    source = load_source(ids, previous, inputs)
    new, protocols, directory = load_new(args.run, ids, args.checkpoint_sha256, inputs)
    rows_by_model = {"source32": source, VARIANT: new}
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "split_sha256": SPLIT_SHA,
        "intended_count": EXPECTED_CADS, "expected_gt_edge_count": EXPECTED_EDGES,
        "evaluation": "own encoder; eval FP32, no TF32, no augmentation; posterior mean; no OCC/test",
        "units": "CD squared: sum of directional NN means; CAD macro; normalized dataset XYZ",
        "feature_moments_weighting": "GT edge vectors pooled across CADs; population covariance, not CAD macro",
        "same_budget_64_mean_comparison_available": False,
        "checkpoint_sha256": protocols[0]["checkpoint_sha256"], "checkpoint_global_step": 30000,
        "checkpoint_hash_source": "CLI plus all shards" if args.checkpoint_sha256 else "all eight shard protocols",
        "models": {}, "paired": {}, "protocols": protocols,
        "input_sha256": inputs, "script_sha256": digest(__file__), "spec_sha256": digest(HERE / "SPEC.md"),
    }
    lookups = {}
    for variant, rows in rows_by_model.items():
        good = [row for row in rows if row["status"] == "ok"]
        lookups[variant] = {row["id"]: row for row in good}
        numeric = sorted({key for row in good for key, value in row.items()
                          if isinstance(value, (int, float)) and not isinstance(value, bool)})
        paths_summary = shape.summarize(rows, ids)
        if len(good) == EXPECTED_CADS:
            require(paths_summary["edge_count"] == EXPECTED_EDGES, f"{variant}: incorrect GT edge total")
        moments = (previous["models"]["source32"]["moments"] if variant == "source32" else
                   old.moment_summary(sorted(directory.glob("shard_*/moments.npz")), True))
        result["models"][variant] = {
            "coverage": {"intended": len(ids), "ok": len(good),
                         "status_counts": dict(Counter(r["status"] for r in rows)),
                         "failures": [r for r in rows if r["status"] != "ok"]},
            "edge_count": paths_summary["edge_count"],
            "metrics": {key: shape.distribution([row.get(key) for row in good]) for key in numeric},
            "paths": paths_summary["paths"], "moments": moments, "topology_micro": topology(good),
        }
    a, b = lookups["source32"], lookups[VARIANT]
    common = [i for i in ids if i in a and i in b]
    metrics = set(result["models"]["source32"]["metrics"]) & set(result["models"][VARIANT]["metrics"])
    result["paired"] = {
        "before": "source32", "after": VARIANT, "count": len(common), "intended": len(ids),
        "excluded_ids": sorted(set(ids) - set(common)),
        "metrics": {key: pair([a[i].get(key) for i in common], [b[i].get(key) for i in common], direction(key))
                    for key in sorted(metrics)},
        "paths": shape.compare_models(a, b)["paths"],
    }
    result["interventions"] = {v: shape.intervention_summary(list(rows.values())) for v, rows in lookups.items()}
    result["alignment_common_subset"] = alignment_subset(lookups, ids)
    reproduction = {}
    for path in shape.PATHS:
        keys = result["models"]["source32"]["paths"][path]["stats"]
        result["paired"]["paths"][path]["stats"] = {
            key: pair([a[i]["paths"][path]["stats"].get(key) for i in common],
                      [b[i]["paths"][path]["stats"].get(key) for i in common], direction(key)) for key in keys}
        reproduction[path] = {}
        for variant, lookup in lookups.items():
            deltas = [abs(r["paths"][path]["cases"]["pred_all"]["cd_sq"] - r[f"path_{path}_matched_cd_sq"])
                      for r in lookup.values()]
            reproduction[path][variant] = {"count": len(deltas), "max_abs_diff": max(deltas, default=None)}
            if variant == "source32":
                # The two historical passes already had recorded FP32 differences;
                # verify that exact historical discrepancy, without changing its tolerance.
                expected = previous_shape["baseline_reproduction"]["source32"][path]
                require(len(deltas) == expected["count"] and np.isclose(
                    max(deltas), expected["max_object_abs_diff"], rtol=1e-12, atol=1e-15),
                    f"Historical decomposition discrepancy changed: {path}")
                reproduction[path][variant]["historical_reproduction_unchanged"] = True
            else:
                require(all(np.isclose(r["paths"][path]["cases"]["pred_all"]["cd_sq"],
                                       r[f"path_{path}_matched_cd_sq"], rtol=1e-6, atol=1e-12)
                            for r in lookup.values()), f"{variant}/{path}: decomposition differs from matched CD")
    baseline_checks = {}
    for key in metrics:
        expected = previous["models"]["source32"]["metrics"].get(key, {}).get("mean")
        actual = result["models"]["source32"]["metrics"][key]["mean"]
        if expected is not None and actual is not None:
            require(np.isclose(actual, expected, rtol=1e-12, atol=1e-14), f"Historical mean changed: {key}")
            baseline_checks[key] = abs(actual - expected)
    # Context only: these older terminal weights have a different training budget.
    late = previous["models"]["late_last64"]
    late_steps = {p["checkpoint_global_step"] for p in late["protocols"]}
    require(len(late_steps) == 1, "Historical late64 checkpoint steps disagree")
    result["background"] = {"late_last64": {
        "role": "different-budget historical context, not a matched-budget ablation",
        "global_step": next(iter(late_steps)), "same_training_budget": False,
        "checkpoint_sha256": late["protocols"][0]["checkpoint_sha256"],
        **{key: late[key] for key in ("coverage", "metrics", "moments", "topology_micro")},
    }}
    result["validation"] = {
        "baseline_csv_means_reproduced": True, "baseline_metric_count": len(baseline_checks),
        "paired_counts_valid": all(v["count"] == v["decreased"] + v["increased"] + v["equal"]
                                   for v in result["paired"]["metrics"].values()),
        "all_intended_ids_exactly_once": True, "all_shards_same_checkpoint_step_and_hash": True,
        "eight_full_shards": True, "decomposition_reproduction": reproduction,
        "all_3495_successful": all(len(v) == EXPECTED_CADS for v in lookups.values()),
        "matched_gt_edges_199092": all(v["edge_count"] == EXPECTED_EDGES for v in result["models"].values()),
    }
    require(result["validation"]["paired_counts_valid"], "Paired direction counts disagree")
    result["complete"] = result["validation"]["all_3495_successful"] and result["validation"]["matched_gt_edges_199092"]
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    flat = [shape.flatten(row) for rows in rows_by_model.values() for row in rows]
    fields = sorted({key for row in flat for key in row})
    with gzip.open(args.out / "per_object.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"complete": result["complete"], "checkpoint_sha256": result["checkpoint_sha256"],
                      "models": {v: {"coverage": m["coverage"], "edge_count": m["edge_count"],
                                     "face_cd": m["metrics"].get("raw_face_cd_sq"),
                                     "edge_cd": m["metrics"].get("raw_edge_cd_sq")}
                                 for v, m in result["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
