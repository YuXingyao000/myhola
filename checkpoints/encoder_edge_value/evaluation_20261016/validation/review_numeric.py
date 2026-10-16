"""Independent CSV/statistics review; imports no production summary functions."""

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / "evaluation_20261015"
CASES = {
    "conditioned": (OLD / "candidate", "conditioned"),
    "candidate_1089": (OLD / "candidate", "encoder_edge_value"),
    "candidate_14520": (OLD / "endpoint_14520/candidate", "encoder_edge_value"),
    "resume_best_27588": (ROOT / "resume_best", "encoder_edge_value"),
    "last_28677": (ROOT / "last", "encoder_edge_value"),
}
NEW = ("resume_best_27588", "last_28677")
PAIRS = [(n, o) for n in NEW for o in list(CASES)[:3]] + [("last_28677", "resume_best_27588")]
IDS = (ROOT.parents[2] / "checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt").read_text().splitlines()
CHECKS = 0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(actual, expected, label):
    global CHECKS
    if isinstance(actual, dict):
        assert set(actual) <= set(expected), (label, "schema", set(actual) - set(expected))
        for key, value in actual.items():
            equal(value, expected[key], label + "/" + key)
    elif isinstance(actual, float):
        assert expected is not None and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-15), (label, actual, expected)
        CHECKS += 1
    else:
        assert actual == expected, (label, actual, expected)
        CHECKS += 1


def read_rows(path, method, id_key):
    rows = [r for r in csv.DictReader(path.open()) if r["method"] == method]
    result = {r[id_key]: r for r in rows}
    assert len(result) == len(rows) == 2417 and set(result) == set(IDS), path
    return result


def number(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    value = float(value)
    assert math.isfinite(value), (key, value)
    return value


def stats(values, count):
    values = [v for v in values if v is not None]
    return {"count": len(values), "missing": count - len(values), "denominator": count,
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            **{f"p{q}": float(np.percentile(values, q)) if values else None for q in (90, 95, 99)},
            "max": max(values) if values else None}


def solid(row):
    value = row.get("native_valid_solid")
    assert value in ("True", "False", "", None), value
    if value in ("", None):
        return None
    if value == "True":
        assert row["native_is_solid_type"] == row["native_brepcheck_valid"] == "True", row["sample_id"]
    return value == "True"


def exact(row):
    return row["topology_status"] == "ok" and number(row, "topology_fp") == number(row, "topology_fn") == 0


def paired(rows, ids, new, old, expected, tie, label):
    equal(len(ids), expected["subset_count"], label + "/subset")
    for key, target in expected["metrics"].items():
        a, b = [], []
        for item in ids:
            x, y = number(rows[new][item], key), number(rows[old][item], key)
            if x is not None and y is not None:
                a.append(x)
                b.append(y)
        changes = [x - y for x, y in zip(a, b)]
        improved, regressed = sum(v < -tie for v in changes), sum(v > tie for v in changes)
        actual = {"common_count": len(a), "missing_count": len(ids) - len(a), "new": stats(a, len(ids)),
                  "reference": stats(b, len(ids)), "delta_new_minus_reference": stats(changes, len(ids)),
                  "improved": improved, "regressed": regressed, "tied": len(a) - improved - regressed,
                  "improved_fraction_computable": improved / len(a) if a else None,
                  "relative_mean_change_percent": (statistics.mean(a) / statistics.mean(b) - 1) * 100 if b and statistics.mean(b) else None,
                  "relative_median_change_percent": (statistics.median(a) / statistics.median(b) - 1) * 100 if b and statistics.median(b) else None}
        equal(actual, target, label + "/" + key)


def main():
    diag_path = ROOT / "diagnostics.json"
    d = json.loads(diag_path.read_text())
    assert d["methods"] == list(CASES) and d["intended_count"] == 2417
    raw, post = {}, {mode: {} for mode in ("fit_only", "optimized")}
    for alias, (directory, method) in CASES.items():
        raw[alias] = read_rows(directory / "raw_metrics/posterior_mean/per_object.csv", method, "model_id")
        target = d["raw"]["methods"][alias]
        for key, expected in target["metrics"].items():
            equal(stats([number(r, key) for r in raw[alias].values()], 2417), expected, alias + "/raw/" + key)
        equal(sum(exact(r) for r in raw[alias].values()), target["exact_adjacency_count"], alias + "/exact_adjacency")
        for mode in post:
            group = read_rows(directory / "post" / mode / "samples.csv", method, "sample_id")
            post[mode][alias] = group
            counts = Counter(solid(r) for r in group.values())
            target = d["post"][mode]["methods"][alias]
            equal({"valid": counts[True], "failed": counts[False], "unknown": counts[None]}, target["native_solid_counts"], alias + "/" + mode)
            equal(counts[True] / 2417, target["valid_fraction_intended"], alias + "/" + mode + "/fraction")
            for key, expected in target["geometry_metrics"].items():
                equal(stats([number(r, key) for r in group.values() if r["geometry_status"] == "ok"], 2417), expected, alias + "/" + mode + "/geometry/" + key)
    tie = d["paired_change_tolerance"]
    for new, old in PAIRS:
        name = new + "_vs_" + old
        target = d["raw"]["pairs"][name]
        paired(raw, IDS, new, old, target["all_intended"], tie, "raw/" + name)
        common = [i for i in IDS if exact(raw[new][i]) and exact(raw[old][i])]
        equal(common, target["both_exact_adjacency_ids"], name + "/exact_ids")
        paired(raw, common, new, old, target["both_exact_adjacency"], tie, "raw_exact/" + name)
        for mode in post:
            observed = {key: [] for key in ("both_valid", "rescued", "regressed", "both_failed", "unknown")}
            for item in IDS:
                a, b = solid(post[mode][new][item]), solid(post[mode][old][item])
                key = "unknown" if a is None or b is None else {(True, True): "both_valid", (True, False): "rescued", (False, True): "regressed", (False, False): "both_failed"}[a, b]
                observed[key].append(item)
            target = d["post"][mode]["pairs"][name]
            equal(observed, target["native_solid_transitions"]["ids"], mode + "/" + name + "/ids")
            equal({k: len(v) for k, v in observed.items()}, target["native_solid_transitions"]["counts"], mode + "/" + name + "/counts")
            common = [i for i in IDS if all(post[mode][m][i]["geometry_status"] == "ok" for m in (new, old))]
            equal(common, target["common_geometry_ids"], mode + "/" + name + "/common_ids")
            paired(post[mode], common, new, old, target["common_geometry"], tie, mode + "/" + name + "/geometry")
    for key, snapshots in d["raw"]["top_one_percent"].items():
        for alias, expected in snapshots.items():
            values = [(i, number(raw[alias][i], key)) for i in IDS]
            values = sorted((p for p in values if p[1] is not None), key=lambda p: (-p[1], p[0]))
            count = math.ceil(len(values) / 100)
            top, total = math.fsum(v for _, v in values[:count]), math.fsum(v for _, v in values)
            equal({"finite_count": len(values), "missing_count": 2417 - len(values), "top_count": count, "top_ids": [i for i, _ in values[:count]], "top_error_sum": top, "total_error_sum": total, "top_error_share": top / total if total else None}, expected, alias + "/tail/" + key)
    for item, details in d["raw"]["tail_case_details"].items():
        for alias, values in details["metrics"].items():
            for key, value in values.items():
                equal(number(raw[alias][item], key), value, item + "/tail_detail/" + alias + "/" + key)
        for new, old in PAIRS:
            delta = number(raw[new][item], "edge_cd_sq") - number(raw[old][item], "edge_cd_sq")
            changes = [(number(raw[new][i], "edge_cd_sq"), number(raw[old][i], "edge_cd_sq")) for i in IDS]
            net = math.fsum(a - b for a, b in changes if a is not None and b is not None)
            equal({"object_delta": delta, "delta_divided_by_full_intended_count": delta / 2417, "fraction_of_net_error_change": delta / net if net else None},
                  details["edge_delta_contributions"][new + "_vs_" + old], item + "/tail_delta/" + new + "_vs_" + old)
    owner_path = ROOT / "owner_per_object.csv"
    owner = {alias: read_rows(owner_path, alias, "model_id") for alias in CASES}
    for alias, expected in d["owner"]["methods"].items():
        for key, values in expected["metrics"].items():
            equal(stats([number(r, key) for r in owner[alias].values()], 2417), values, alias + "/owner/" + key)
        for key in ("missing_gt_directed_pairs", "extra_directed_pairs"):
            equal(sum(int(r[key]) for r in owner[alias].values()), expected[key + "_total"], alias + "/owner/" + key)
    for new, old in PAIRS:
        paired(owner, IDS, new, old, d["owner"]["pairs"][new + "_vs_" + old], tie, "owner/" + new + "_vs_" + old)
    sample = json.loads((ROOT / "validation/owner_independent_samples.json").read_text())
    for record in sample["inputs"]:
        equal(sha(record["path"]), record["sha256"], "owner raw unchanged")
    for record in sample["checks"]:
        row = owner[record["method"]][record["model_id"]]
        for key, value in record.items():
            if key in ("method", "model_id"):
                equal(value, row[key], "owner_identity")
            else:
                equal(value, float(row[key]) if isinstance(value, float) else int(row[key]), "independent_KDTree/" + record["method"] + "/" + record["model_id"] + "/" + key)
    result = {"status": "passed", "diagnostics_sha256": sha(diag_path), "owner_csv_sha256": sha(owner_path), "review_script_sha256": sha(__file__),
              "checks": CHECKS, "independence": "No production summary functions imported. Python statistics/NumPy quantiles recompute all CSV summaries; original NPZ owner samples independently used SciPy KDTree.",
              "all_2417_csv_statistics_recomputed": True, "five_snapshots_both_post_modes_recounted": True, "seven_paired_comparisons_recomputed": True,
              "tail_statistics_recomputed": True, "owner_npz_objects": 27, "owner_npz_method_checks": len(sample["checks"]),
              "native_solids": {mode: {m: sum(solid(r) is True for r in group.values()) for m, group in groups.items()} for mode, groups in post.items()},
              "optimized_transitions": {n + "_vs_" + o: d["post"]["optimized"]["pairs"][n + "_vs_" + o]["native_solid_transitions"]["counts"] for n, o in PAIRS}}
    (ROOT / "validation/final_numeric_review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
