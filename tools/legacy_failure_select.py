"""Freeze 50 observationally stratified legacy VAE failure cases, without inference.

Inputs are the completed comparison CSVs. No CAD, predictions, or existing
comparison artifacts are modified. Case sampling is for diagnostic coverage,
not an unbiased estimate of failure mechanism prevalence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


METHODS = ("legacy", "joint")
INTENDED = 2417
VALID = 2048
FAILED = 369
SELECTED = 50
SHAPE_NAMES = {0: "compound", 2: "invalid_solid", 3: "shell"}
TEXT_FIELDS = {
    "model_id", "sample_id", "method", "mode", "status", "error", "stage", "representation",
    "prediction_file", "input_latent_sha256", "protocol_fingerprint", "raw_sha256", "compatibility_error",
}
RAW_METRICS = (
    "face_mae", "face_cd_sq", "edge_cd_sq", "edge_pred_to_gt_sq", "edge_gt_to_pred_sq",
    "topology_precision", "topology_recall", "topology_f1", "unique_pair_count_error",
    "legacy_unilateral_pair_count", "legacy_reciprocal_cd_sq_mean", "legacy_reciprocal_cd_sq_max",
    "gt_face_count", "gt_boundary_count", "raw_edge_count",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def typed_value(key: str, value: str):
    if value == "":
        return None
    if key in TEXT_FIELDS or key.endswith("_status"):
        return value
    if value in {"True", "False"}:
        return value == "True"
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    try:
        numeric = float(value)
    except ValueError:
        return value
    if not math.isfinite(numeric):
        raise ValueError(f"Nonfinite source field {key}={value}")
    return numeric


def read_table(path: Path, id_key: str) -> dict[tuple[str, str], dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Missing or repeated CSV columns in {path}")
        rows = {}
        for original in reader:
            if None in original or any(value is None for value in original.values()):
                raise ValueError(f"Malformed row in {path}")
            row = {key: typed_value(key, value) for key, value in original.items()}
            item, method = row[id_key], row["method"]
            if not isinstance(item, str) or not re.fullmatch(r"\d{8}", item) or method not in METHODS:
                raise ValueError(f"Invalid identity {item!r}/{method!r} in {path}")
            key = (method, item)
            if key in rows:
                raise ValueError(f"Duplicate identity {key} in {path}")
            rows[key] = row
    return rows


def face_bin(count: int) -> str:
    if 1 <= count <= 9:
        return "1-9"
    if 10 <= count <= 19:
        return "10-19"
    if 20 <= count <= 30:
        return "20-30"
    raise ValueError(f"Unexpected GT face count {count}")


def describe_stratum(post: dict, raw: dict) -> tuple[str, dict]:
    if post["shape_type"] not in SHAPE_NAMES or type(post["native_brepcheck_valid"]) is not bool:
        raise ValueError("Failed case lacks an audited native shape type/check result")
    exact = raw["topology_fp"] == 0 and raw["topology_fn"] == 0
    dimensions = {
        "shape_type": post["shape_type"], "shape_name": SHAPE_NAMES[post["shape_type"]],
        "native_brepcheck_valid": post["native_brepcheck_valid"], "raw_pair_topology_exact": exact,
    }
    key = "|".join((dimensions["shape_name"], "brepcheck_valid" if dimensions["native_brepcheck_valid"] else "brepcheck_invalid",
                    "pair_exact" if exact else "pair_inexact"))
    return key, dimensions


def allocate_quotas(capacities: dict[str, int], count: int) -> dict[str, int]:
    """One per nonempty stratum, then exact-integer Hamilton allocation."""
    if any(n < 1 for n in capacities.values()) or not len(capacities) <= count <= sum(capacities.values()):
        raise ValueError("Sample count cannot satisfy nonempty strata and capacities")
    remaining = count - len(capacities)
    total_capacity = sum(n - 1 for n in capacities.values())
    if not remaining:
        return {key: 1 for key in capacities}
    allocations = {key: divmod(remaining * (n - 1), total_capacity) for key, n in capacities.items()}
    quotas = {key: 1 + value[0] for key, value in allocations.items()}
    for key in sorted(capacities, key=lambda key: (-allocations[key][1], key))[:count - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def select_cases(failures: list[str], raw: dict, post: dict, count: int = SELECTED) -> tuple[list[str], list[dict]]:
    groups = defaultdict(list)
    dimensions = {}
    for item in failures:
        key, description = describe_stratum(post["legacy", item], raw["legacy", item])
        groups[key].append(item)
        dimensions[key] = description
    quotas = allocate_quotas({key: len(values) for key, values in groups.items()}, count)
    selected, strata = [], []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda item: (
            raw["legacy", item]["gt_face_count"], hashlib.sha256(f"legacy-failure-v1:{item}".encode()).hexdigest(), item))
        ranks = [((2 * index + 1) * len(ordered)) // (2 * quotas[key]) for index in range(quotas[key])]
        chosen = [ordered[rank] for rank in ranks]
        selected.extend(chosen)
        strata.append({"stratum": key, **dimensions[key], "population_count": len(ordered),
                       "selected_count": len(chosen), "selected_ids": chosen, "selected_zero_based_ranks": ranks})
    if len(selected) != count or len(set(selected)) != count:
        raise AssertionError("Selection is not the requested unique size")
    return selected, strata


def statistics(values: list) -> dict:
    import numpy as np
    values = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite descriptive metric")
    return {"count": len(values), "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "p90": float(np.percentile(values, 90)) if len(values) else None}


def raw_group_summary(ids: list[str], raw: dict) -> dict:
    return {"object_count": len(ids), "metrics": {
        metric: statistics([raw["legacy", item].get(metric) for item in ids]) for metric in RAW_METRICS}}


def population_comparison(ids: list[str], raw: dict, post: dict) -> dict:
    groups = {"native_valid": [item for item in ids if post["legacy", item]["native_valid_solid"]],
              "native_failed": [item for item in ids if not post["legacy", item]["native_valid_solid"]]}
    all_counts = sorted({raw["legacy", item]["gt_face_count"] for item in ids})
    exact, binned = {}, {}
    for count in all_counts:
        exact[str(count)] = {name: raw_group_summary([item for item in members if raw["legacy", item]["gt_face_count"] == count], raw)
                             for name, members in groups.items()}
    for bin_name in ("1-9", "10-19", "20-30"):
        binned[bin_name] = {name: raw_group_summary([item for item in members if face_bin(raw["legacy", item]["gt_face_count"]) == bin_name], raw)
                            for name, members in groups.items()}
    standardized = {}
    for metric in RAW_METRICS:
        supported = [count for count in all_counts if all(exact[str(count)][name]["metrics"][metric]["count"] for name in groups)]
        failures_supported = sum(exact[str(count)]["native_failed"]["object_count"] for count in supported)
        weights = {str(count): exact[str(count)]["native_failed"]["object_count"] / failures_supported for count in supported} if failures_supported else {}
        means = {name: sum(weights[str(count)] * exact[str(count)][name]["metrics"][metric]["mean"] for count in supported) if weights else None
                 for name in groups}
        standardized[metric] = {"exact_face_count_weights": weights, "failure_objects_in_common_support": failures_supported,
                                "failure_total": len(groups["native_failed"]), "weighted_means": means,
                                "failed_minus_valid": means["native_failed"] - means["native_valid"] if weights else None}
    return {"scope": "all 2417 accepted posterior-mean legacy cases, not the selected 50",
            "unadjusted": {name: raw_group_summary(members, raw) for name, members in groups.items()},
            "by_gt_face_bin": binned, "by_exact_gt_face_count": exact,
            "face_count_standardized": {"method": "Within each exact GT face count, compare group means; weight both groups by the failed population face-count distribution on common finite-metric support.",
                                        "causal_claim": False, "metrics": standardized},
            "limitations": ["Conditioning on final validity is descriptive, not evidence of causal failure mechanisms.",
                            "Controlling face count does not control geometry type, size, curvature, holes, or training difficulty.",
                            "Topology exactness concerns merged distinct-face adjacency, not complete B-Rep loops, seams, orientation or manifoldness."]}


def observations(ids: list[str], raw: dict, post: dict) -> dict:
    rows = [post["legacy", item] for item in ids]
    return {"count": len(ids), "shape_counts": dict(Counter(SHAPE_NAMES[row["shape_type"]] for row in rows)),
            "brepcheck_counts": dict(Counter("valid" if row["native_brepcheck_valid"] else "invalid" for row in rows)),
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "geometry_status_counts": dict(Counter(row["geometry_status"] for row in rows)),
            "pair_exact_counts": dict(Counter("exact" if raw["legacy", item]["topology_fp"] == raw["legacy", item]["topology_fn"] == 0 else "inexact" for item in ids)),
            "gt_face_bin_counts": dict(Counter(face_bin(raw["legacy", item]["gt_face_count"]) for item in ids))}


def make_paths(item: str, comparison: Path, data_root: Path) -> dict:
    return {"gt_npz": str(data_root / item / "data.npz"), "gt_step": str(data_root / item / "normalized_shape.step"),
            "raw_legacy_npz": str(comparison / "raw_predictions/posterior_mean/legacy" / item / "data.npz"),
            "raw_joint_npz": str(comparison / "raw_predictions/posterior_mean/joint" / item / "data.npz"),
            "post_legacy_step": str(comparison / "post/optimized/legacy" / item / "recon_brep.step"),
            "post_joint_step": str(comparison / "post/optimized/joint" / item / "recon_brep.step"),
            "fit_legacy_step": str(comparison / "post/fit_only/legacy" / item / "recon_brep.step"),
            "post_legacy_fitted_faces": str(comparison / "post/optimized/legacy" / item / "separate_faces.ply"),
            "post_legacy_log": str(comparison / "metadata/post/optimized/logs/legacy" / (item + ".log")),
            "post_legacy_status": str(comparison / "metadata/post/optimized/samples/legacy" / (item + ".json"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, default=Path("checkpoints/joint_boundary/comparison_20261011"))
    parser.add_argument("--output-root", type=Path, default=Path("checkpoints/joint_boundary/legacy_failures_20261011"))
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/d/data/deepcad_v7"))
    args = parser.parse_args()
    comparison, output, data_root = args.comparison_root.resolve(), args.output_root.resolve(), args.data_root.resolve()
    names = ("selected_cases.json", "selection_summary.json", "selected_ids.txt")
    if any((output / name).exists() for name in names):
        parser.error("Selection artifacts already exist; preserve the frozen selection instead of overwriting")
    sources = {"raw_metrics": comparison / "raw_metrics/posterior_mean/per_object.csv",
               "optimized": comparison / "post/optimized/samples.csv", "fit_only": comparison / "post/fit_only/samples.csv"}
    raw = read_table(sources["raw_metrics"], "model_id")
    post = read_table(sources["optimized"], "sample_id")
    fit = read_table(sources["fit_only"], "sample_id")
    ids = sorted(item for method, item in post if method == "legacy")
    expected = {(method, item) for method in METHODS for item in ids}
    if len(ids) != INTENDED or any(set(table) != expected for table in (raw, post, fit)):
        raise ValueError("Tables must contain the same 2417 unique identities for both methods")
    for item in ids:
        row = post["legacy", item]
        if type(row["native_valid_solid"]) is not bool:
            raise ValueError(f"Missing native validity for {item}")
        face_bin(raw["legacy", item]["gt_face_count"])
    failed = [item for item in ids if not post["legacy", item]["native_valid_solid"]]
    if len(failed) != FAILED or len(ids) - len(failed) != VALID:
        raise ValueError("Expected the frozen 2048 valid / 369 failed population")
    if any(not post["legacy", item]["step_readable"] or not post["legacy", item]["step_exists"] for item in failed):
        raise ValueError("The audited failure population must have readable STEP artifacts")
    selected, strata = select_cases(failed, raw, post)
    if "00480996" not in selected or len(strata) != 9:
        raise ValueError("Frozen observed strata/rare invalid-solid case changed")
    metadata = {"version": "legacy_failure_selection_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "comparison_root": str(comparison), "data_root": str(data_root),
                "sources": {key: {"path": str(path), "sha256": file_sha256(path)} for key, path in sources.items()},
                "selector_sha256": file_sha256(Path(__file__)),
                "population": {"accepted_test": INTENDED, "original_test": 2424, "contract_unsupported": 7,
                               "native_valid": VALID, "native_failed": FAILED, "selected": SELECTED},
                "selection_protocol": {"dimensions": ["shape_type", "native_brepcheck_valid", "raw_pair_topology_exact"],
                                       "quota": "min 1 per nonempty stratum; remaining slots proportional to n-1 with exact-integer Hamilton remainders; tie by stratum name",
                                       "within_stratum": "sort by GT face count, SHA256('legacy-failure-v1:{id}'), ID; select zero-based floor((j+.5)*n/q) ranks",
                                       "frozen_before_in_depth_case_analysis": True,
                                       "prevalence_estimator": False}}
    cases = []
    for item in selected:
        key, dimensions = describe_stratum(post["legacy", item], raw["legacy", item])
        paths = make_paths(item, comparison, data_root)
        case = {"id": item, "stratum": key, "stratum_dimensions": dimensions,
                "gt_face_count": raw["legacy", item]["gt_face_count"], "gt_face_bin": face_bin(raw["legacy", item]["gt_face_count"]),
                "raw_legacy": raw["legacy", item], "raw_joint": raw["joint", item],
                "post_legacy": post["legacy", item], "post_joint": post["joint", item], "fit_legacy": fit["legacy", item],
                "paths": paths, "path_exists": {key: Path(path).is_file() for key, path in paths.items()}}
        cases.append(case)
    text_ids = "\n".join(sorted(selected)) + "\n"
    metadata["selected_ids_sha256"] = hashlib.sha256(text_ids.encode()).hexdigest()
    manifest = {**metadata, "cases": cases}
    summary = {**metadata, "strata": strata, "failure_population": observations(failed, raw, post),
               "selected_population": observations(selected, raw, post), "raw_population_comparison": population_comparison(ids, raw, post)}
    serialized = {"selected_cases.json": json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                  "selection_summary.json": json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                  "selected_ids.txt": text_ids}
    output.mkdir(parents=True, exist_ok=True)
    for name, content in serialized.items():
        with (output / name).open("x", encoding="utf-8") as handle:
            handle.write(content)
    print(json.dumps({"output": str(output), "population": metadata["population"],
                      "selected_ids_sha256": metadata["selected_ids_sha256"], "strata": len(strata)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
