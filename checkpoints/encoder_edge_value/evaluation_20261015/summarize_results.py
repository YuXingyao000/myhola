"""Three-way fixed-cohort summary and owner-pair geometry diagnostics.

Contract: DIAGNOSTICS_SPEC.md. This script reads completed evaluation outputs;
it never invokes a model, resamples geometry, or runs the CAD postprocessor.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
DATA = Path("/mnt/d/data/deepcad_v7")
BASE = REPO / "checkpoints/intersection_refinement/curve_conditioning/evaluation_20261014"
SPLIT = REPO / "checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt"
SPLIT_SHA = "2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3"
METHODS = ("conditioned", "encoder_gat_control", "encoder_edge_value")
ARMS = {"candidate": "encoder_edge_value", "control": "encoder_gat_control"}
LABELS = {"conditioned": "Frozen source", "encoder_gat_control": "GAT control", "encoder_edge_value": "Edge-value GAT"}
PAIRS = (("encoder_edge_value", "conditioned"), ("encoder_gat_control", "conditioned"), ("encoder_edge_value", "encoder_gat_control"))
RAW_METRICS = ("face_mae", "face_cd_sq", "face_pred_to_gt_sq", "face_gt_to_pred_sq", "edge_cd_sq", "edge_pred_to_gt_sq", "edge_gt_to_pred_sq", "legacy_reciprocal_cd_sq_mean", "legacy_reciprocal_cd_sq_max")
POST_METRICS = ("face_cd", "face_accuracy", "face_completeness", "edge_cd", "edge_accuracy", "edge_completeness")
TERMINAL = {"ok", "missing_raw", "invalid_raw", "post_error", "no_step", "step_read_error", "geometry_error", "timeout", "crash", "interrupted"}
TIE = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Inputs:
    def __init__(self):
        self.records = {}

    def read(self, path):
        path = Path(path).resolve()
        before = sha256(path)
        content = path.read_bytes()
        require(before == hashlib.sha256(content).hexdigest() == sha256(path), f"Input changed while reading: {path}")
        old = self.records.get(str(path))
        require(old is None or old["sha256"] == before, f"Input changed between reads: {path}")
        self.records[str(path)] = {"path": str(path), "sha256": before, "bytes": len(content), "verified_before_after_read": True}
        return content

    def json(self, path):
        return json.loads(self.read(path))

    def csv(self, path):
        return list(csv.DictReader(io.StringIO(self.read(path).decode())))

    def arrays(self, path):
        content = self.read(path)
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}


def number(row, key, *, nonnegative=True):
    value = row.get(key)
    if value in (None, ""):
        return None
    value = float(value)
    require(math.isfinite(value) and (not nonnegative or value >= 0), f"Invalid metric {key}: {value}")
    return value


def boolean(value):
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    if value in (None, ""):
        return None
    raise ValueError(f"Invalid boolean field: {value!r}")


def stats(values, denominator=None):
    values = [float(value) for value in values if value is not None]
    require(all(math.isfinite(value) for value in values), "Nonfinite summary input")
    a = np.asarray(values, dtype=np.float64)
    denominator = len(values) if denominator is None else denominator
    require(len(values) <= denominator, "Coverage exceeds denominator")
    return {"count": len(values), "missing": denominator - len(values), "denominator": denominator,
            "mean": float(a.mean()) if len(a) else None,
            "median": float(np.median(a)) if len(a) else None,
            **{f"p{q}": float(np.quantile(a, q / 100)) if len(a) else None for q in (90, 95, 99)},
            "max": float(a.max()) if len(a) else None}


def verify_stats(actual, expected):
    require(actual["count"] == expected["count"], "Upstream count differs")
    for key in ("mean", "median", "p90"):
        a, b = actual[key], expected[key]
        require((a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-16)), f"Upstream {key} differs: {a} / {b}")


def frame(rows, methods, ids, id_key):
    require(len(rows) == len(methods) * len(ids), "Unexpected row count")
    result = {method: {} for method in methods}
    for row in rows:
        method, item = row["method"], row[id_key]
        require(method in result and item in ids and item not in result[method], "Wrong method/ID or duplicate row")
        result[method][item] = row
    require(all(set(group) == set(ids) for group in result.values()), "Incomplete fixed cohort")
    return result


def same_baseline(first, second, *, raw):
    ignored = {"prediction_file"} if raw else set()
    require(set(first) == set(second), "Baseline ID mismatch")
    for item in first:
        # CSV schemas are unions of all rows in each arm. A timeout elsewhere
        # may add a blank column to unchanged baseline rows in only one file.
        a = {k: v for k, v in first[item].items() if k not in ignored and v not in (None, "")}
        b = {k: v for k, v in second[item].items() if k not in ignored and v not in (None, "")}
        require(a == b, f"Baseline records differ between pair roots: {item}")
        if raw:
            require(Path(first[item]["prediction_file"]).resolve() == Path(second[item]["prediction_file"]).resolve(), "Different baseline raw files")


def paired(rows, selected_ids, new, old, metrics):
    output = {"new_method": new, "reference_method": old, "subset_count": len(selected_ids), "metrics": {}}
    for metric in metrics:
        comparable = [(item, number(rows[new][item], metric), number(rows[old][item], metric)) for item in selected_ids]
        valid = [(item, a, b) for item, a, b in comparable if a is not None and b is not None]
        a, b = [r[1] for r in valid], [r[2] for r in valid]
        delta = [x - y for x, y in zip(a, b)]
        output["metrics"][metric] = {
            "common_count": len(valid), "missing_count": len(selected_ids) - len(valid),
            "new": stats(a, len(selected_ids)), "reference": stats(b, len(selected_ids)),
            "delta_new_minus_reference": stats(delta, len(selected_ids)),
            "improved": sum(x < -TIE for x in delta), "tied": sum(abs(x) <= TIE for x in delta), "regressed": sum(x > TIE for x in delta),
            "improved_fraction_computable": sum(x < -TIE for x in delta) / len(delta) if delta else None,
            "relative_mean_change_percent": (np.mean(a) / np.mean(b) - 1) * 100 if b and np.mean(b) else None,
            "relative_median_change_percent": (np.median(a) / np.median(b) - 1) * 100 if b and np.median(b) else None,
        }
        require(output["metrics"][metric]["improved"] + output["metrics"][metric]["tied"] + output["metrics"][metric]["regressed"] == len(valid), "Pair accounting failed")
    return output


def topology_exact(row):
    return row.get("topology_status") == "ok" and number(row, "topology_fp") == number(row, "topology_fn") == 0


def tails(rows, ids, metric):
    result = {}
    for method in METHODS:
        values = [(item, number(rows[method][item], metric)) for item in ids]
        values = [(item, value) for item, value in values if value is not None]
        values.sort(key=lambda pair: (-pair[1], pair[0]))
        k = math.ceil(len(values) * .01)
        total, top = sum(v for _, v in values), sum(v for _, v in values[:k])
        result[method] = {"finite_count": len(values), "missing_count": len(ids) - len(values), "top_count": k,
                          "top_ids": [item for item, _ in values[:k]], "top_error_sum": top, "total_error_sum": total,
                          "top_error_share": top / total if total else None}
    return result


def load_raw(inputs, ids):
    merged, checks = {}, {}
    for arm, method in ARMS.items():
        path = ROOT / arm / "raw_metrics/posterior_mean"
        summary = inputs.json(path / "summary.json")
        require(summary["input_contract"] == "same_geometry_source" and summary["protocol_valid"], "Invalid raw input contract")
        require(summary["intended_count"] == 2417 and summary["original_count"] == 2424, "Wrong raw denominator")
        source_counts = summary["shared_input_status_counts"]
        require(sum(source_counts.values()) == 2417 and not any(source_counts.get(k, 0) for k in ("mismatch", "missing_digest")), "Invalid raw source accounting")
        require(set(source_counts) <= {"verified", "not_verified"}, "Unknown source status")
        rows = frame(inputs.csv(path / "per_object.csv"), ("conditioned", method), ids, "model_id")
        if "conditioned" in merged:
            same_baseline(merged["conditioned"], rows["conditioned"], raw=True)
        merged.update(rows)
        for name in rows:
            for metric in RAW_METRICS:
                verify_stats(stats([number(row, metric) for row in rows[name].values()], len(ids)), summary["methods"][name]["metrics"][metric])
        checks[arm] = {"summary_path": str(path / "summary.json"), "independently_verified_metrics": list(RAW_METRICS)}
    historical = inputs.csv(BASE / "raw_metrics/posterior_mean/per_object.csv")
    historical = {r["model_id"]: r for r in historical if r["method"] == "conditioned"}
    require(set(historical) == set(ids), "Historical baseline missing IDs")
    for item in ids:
        for key in (*RAW_METRICS, "status", "prediction_status", "topology_fp", "topology_fn"):
            require(historical[item][key] == merged["conditioned"][item][key], f"Historical raw baseline changed: {item}/{key}")
    output = {"methods": {}, "pairs": {}, "checks": checks}
    for method in METHODS:
        output["methods"][method] = {"status_counts": dict(Counter(row["status"] for row in merged[method].values())),
                                      "exact_adjacency_count": sum(topology_exact(row) for row in merged[method].values()),
                                      "metrics": {key: stats([number(row, key) for row in merged[method].values()], len(ids)) for key in RAW_METRICS}}
    for new, old in PAIRS:
        key = new + "_vs_" + old
        exact = [item for item in ids if topology_exact(merged[new][item]) and topology_exact(merged[old][item])]
        output["pairs"][key] = {"all_intended": paired(merged, ids, new, old, RAW_METRICS),
                                "both_exact_adjacency": paired(merged, exact, new, old, RAW_METRICS),
                                "both_exact_adjacency_ids": exact}
    output["top_one_percent"] = {key: tails(merged, ids, key) for key in ("edge_cd_sq", "legacy_reciprocal_cd_sq_mean", "legacy_reciprocal_cd_sq_max")}
    outlier_ids = output["top_one_percent"]["edge_cd_sq"]["encoder_edge_value"]["top_ids"]
    net_change = {old: sum(number(merged["encoder_edge_value"][item], "edge_cd_sq") - number(merged[old][item], "edge_cd_sq")
                           for item in ids if number(merged["encoder_edge_value"][item], "edge_cd_sq") is not None and number(merged[old][item], "edge_cd_sq") is not None)
                  for old in ("conditioned", "encoder_gat_control")}
    details = []
    for item in outlier_ids:
        record = {"model_id": item, "methods": {method: {key: number(merged[method][item], key) for key in ("edge_cd_sq", "edge_pred_to_gt_sq", "edge_gt_to_pred_sq", "topology_fp", "topology_fn", "unique_pair_count", "gt_boundary_count")} for method in METHODS}, "candidate_difference_contribution": {}}
        for old in net_change:
            a, b = number(merged["encoder_edge_value"][item], "edge_cd_sq"), number(merged[old][item], "edge_cd_sq")
            delta = a - b if a is not None and b is not None else None
            record["candidate_difference_contribution"][old] = {"object_delta": delta, "delta_divided_by_full_intended_count": delta / len(ids) if delta is not None else None, "fraction_of_net_error_change": delta / net_change[old] if delta is not None and net_change[old] else None}
        details.append(record)
    output["candidate_edge_top_one_percent_details"] = details
    output["contribution_note"] = "Per-object signed delta divided by 2417; fraction of net change may exceed 1 or be negative because other objects offset it. No object is removed from primary results."
    return merged, output


def solid_transitions(rows, ids, new, old):
    result = {key: [] for key in ("both_valid", "rescued", "regressed", "both_failed", "unknown")}
    for item in ids:
        a, b = boolean(rows[new][item].get("native_valid_solid")), boolean(rows[old][item].get("native_valid_solid"))
        category = "unknown" if a is None or b is None else ("both_valid" if a and b else "rescued" if a else "regressed" if b else "both_failed")
        result[category].append(item)
    require(sum(map(len, result.values())) == len(ids), "Solid transition accounting failed")
    return {"new_method": new, "reference_method": old, "counts": {k: len(v) for k, v in result.items()}, "ids": result}


def load_post(inputs, ids, mode):
    merged = {}
    for arm, method in ARMS.items():
        path = ROOT / arm / "post" / mode
        summary = inputs.json(path / "summary.json")
        require(summary["intended_count_per_method"] == 2417 and summary["original_test_count"] == 2424, "Wrong post denominator")
        rows = frame(inputs.csv(path / "samples.csv"), ("conditioned", method), ids, "sample_id")
        require(all(r["status"] in TERMINAL for group in rows.values() for r in group.values()), "Postprocessing still pending")
        if "conditioned" in merged:
            same_baseline(merged["conditioned"], rows["conditioned"], raw=False)
        merged.update(rows)
        for name, group in rows.items():
            require(sum(boolean(r.get("native_valid_solid")) is True for r in group.values()) == summary["methods"][name]["counts"]["native_valid_solid"], "Post solid count differs")
            for metric in POST_METRICS:
                observed = [number(r, metric) for r in group.values() if r.get("geometry_status") == "ok"]
                verify_stats(stats(observed, len(ids)), summary["methods"][name]["geometry_all_computable"][metric])
    output = {"methods": {}, "pairs": {}}
    historical = inputs.csv(BASE / "post" / mode / "samples.csv")
    historical = {row["sample_id"]: row for row in historical if row["method"] == "conditioned"}
    require(set(historical) == set(ids), "Historical post baseline missing IDs")
    for item in ids:
        for key in (*POST_METRICS, "status", "geometry_status", "native_valid_solid", "post_status", "step_exists", "step_readable", "raw_sha256"):
            require(historical[item].get(key, "") == merged["conditioned"][item].get(key, ""), f"Historical post baseline changed: {mode}/{item}/{key}")
    for method, group in merged.items():
        solids = Counter("unknown" if boolean(r.get("native_valid_solid")) is None else "valid" if boolean(r.get("native_valid_solid")) else "failed" for r in group.values())
        output["methods"][method] = {"native_solid_counts": {key: solids[key] for key in ("valid", "failed", "unknown")},
                                     "valid_fraction_intended": solids["valid"] / len(ids), "status_counts": dict(Counter(r["status"] for r in group.values())),
                                     "geometry_metrics": {metric: stats([number(r, metric) for r in group.values() if r.get("geometry_status") == "ok"], len(ids)) for metric in POST_METRICS}}
    for new, old in PAIRS:
        common = [item for item in ids if all(merged[m][item].get("geometry_status") == "ok" for m in (new, old))]
        output["pairs"][new + "_vs_" + old] = {"native_solid_transitions": solid_transitions(merged, ids, new, old),
                                                  "common_geometry": paired(merged, common, new, old, POST_METRICS), "common_geometry_ids": common,
                                                  "common_geometry_missing_count": len(ids) - len(common)}
    output["all_three_common_geometry_ids"] = [item for item in ids if all(merged[m][item].get("geometry_status") == "ok" for m in METHODS)]
    return merged, output


def text(array):
    require(np.asarray(array).ndim == 0, "Expected scalar provenance")
    return str(array.item())


def curve_map(curves, connectivity, face_count):
    require(curves.ndim == 3 and curves.shape[1:] == (16, 3) and np.isfinite(curves).all(), "Invalid sampled curves")
    require(connectivity.ndim == 2 and connectivity.shape[1] == 3 and np.issubdtype(connectivity.dtype, np.integer), "Invalid owner connectivity")
    result, seen = {}, set()
    for edge, source, target in connectivity:
        edge, source, target = int(edge), int(source), int(target)
        require(0 <= edge < len(curves) and edge not in seen, "Wrong/repeated edge reference")
        require(0 <= source < face_count and 0 <= target < face_count and source != target, "Invalid owner face index")
        require((source, target) not in result, "Duplicate directed owner pair")
        result[source, target] = curves[edge].astype(np.float64)
        seen.add(edge)
    require(len(seen) == len(curves), "Unreferenced sampled curve")
    return result


def owner_cd(predicted, truth):
    require(predicted.shape == truth.shape and predicted.ndim == 3 and predicted.shape[1:] == (16, 3), "Owner CD needs matching [P,16,3]")
    if not len(predicted):
        return np.empty(0, dtype=np.float64)
    distances = np.square(predicted[:, :, None, :] - truth[:, None, :, :]).sum(axis=-1)
    return distances.min(axis=2).mean(axis=1) + distances.min(axis=1).mean(axis=1)


def owner_diagnostics(inputs, ids, raw, post_rows, training, outlier_ids):
    protocols = {"conditioned": inputs.json(BASE / "metadata/inference/posterior_mean/protocol.json")}
    for arm, method in ARMS.items():
        protocols[method] = inputs.json(ROOT / arm / "metadata/inference/posterior_mean/protocol.json")
        require(protocols[method]["checkpoint_sha256"] == training["arms"][arm]["selected"]["sha256"], "Wrong own-encoder checkpoint")
    digests = {method: hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest() for method, protocol in protocols.items()}
    rows, source_records, outlier_records = [], [], {}
    for index, item in enumerate(ids):
        gt_path = DATA / item / "data.npz"
        gt = inputs.arrays(gt_path)
        gt_sha = inputs.records[str(gt_path.resolve())]["sha256"]
        faces = gt["sample_points_faces"]
        require(faces.ndim == 4 and faces.shape[1:] == (16, 16, 6) and np.isfinite(faces).all(), "Invalid canonical GT faces")
        truth = curve_map(gt["sample_points_lines"][..., :3], gt["edge_face_connectivity"], len(faces))
        maps, errors, probabilities = {}, {}, {}
        for method in METHODS:
            file = Path(raw[method][item]["prediction_file"])
            if not file.is_file():
                errors[method] = "missing_prediction"
                continue
            arrays = inputs.arrays(file)
            raw_sha = inputs.records[str(file.resolve())]["sha256"]
            require(text(arrays["input_source_sha256"]) == gt_sha, "Wrong raw CAD source")
            require(text(arrays["protocol_sha256"]) == digests[method], "Raw protocol mismatch")
            require(text(arrays["comparison_mode"]) == "posterior_mean", "Wrong reconstruction mode")
            latents = arrays["face_latents"]
            require(latents.shape == (len(faces), 32) and latents.dtype == np.float32 and np.isfinite(latents).all(), "Wrong canonical face count/latent array")
            latent = hashlib.sha256(str(latents.shape).encode())
            latent.update(np.ascontiguousarray(latents).tobytes())
            require(latent.hexdigest() == text(arrays["input_latent_sha256"]) == raw[method][item]["input_latent_sha256"], "Wrong stored mean digest")
            require(raw[method][item]["input_source_sha256"] == gt_sha and raw[method][item]["latent_digest_status"] == "verified", "CSV source contract mismatch")
            if method != "conditioned":
                require(text(arrays["encoder_checkpoint_sha256"]) == protocols[method]["checkpoint_sha256"] and text(arrays["input_source_kind"]) == "checkpoint_posterior_mean", "Wrong encoder provenance")
            for mode, records in post_rows.items():
                require(records[method][item].get("raw_sha256") == raw_sha, f"Post raw source mismatch: {mode}/{method}/{item}")
            require(arrays["pred_face"].shape == (len(faces), 16, 16, 3), "Wrong predicted face shape")
            try:
                maps[method] = curve_map(arrays["pred_edge"], arrays["pred_edge_face_connectivity"], len(faces))
            except ValueError as exc:
                errors[method] = str(exc)
            if item in outlier_ids:
                pairs, probs = arrays["all_candidate_pairs"], arrays["all_candidate_prob"]
                require(pairs.shape == (len(probs), 2) and np.isfinite(probs).all(), "Invalid stored candidate probabilities")
                probabilities[method] = {tuple(int(x) for x in pair): float(p) for pair, p in zip(pairs, probs)}
                require(len(probabilities[method]) == len(pairs), "Duplicate stored candidate pair")
            if method in maps and raw[method][item].get("topology_status") == "ok":
                actual, expected = {tuple(sorted(p)) for p in maps[method]}, {tuple(sorted(p)) for p in truth}
                require(len(actual - expected) == number(raw[method][item], "topology_fp") and len(expected - actual) == number(raw[method][item], "topology_fn"), "Owner map contradicts topology CSV")
        common = sorted(set(truth).intersection(*(set(maps[m]) for m in METHODS))) if len(maps) == 3 else []
        gt_curves = np.stack([truth[p] for p in common]) if common else np.empty((0, 16, 3))
        if item in outlier_ids:
            outlier_records[item] = {"common_directed_pair_count": len(common), "gt_directed_pair_count": len(truth), "methods": {}}
        for method in METHODS:
            curves = np.stack([maps[method][p] for p in common]) if common else np.empty((0, 16, 3))
            distances = owner_cd(curves, gt_curves)
            row = {"model_id": item, "method": method, "owner_status": "invalid_or_missing_input" if errors else "ok" if common else "empty_common_pairs", "error": json.dumps(errors, sort_keys=True) if errors else "", "gt_directed_pair_count": len(truth), "predicted_directed_pair_count": len(maps[method]) if method in maps else None, "missing_gt_directed_pairs": len(set(truth) - set(maps[method])) if method in maps else None, "extra_directed_pairs": len(set(maps[method]) - set(truth)) if method in maps else None, "common_directed_pair_count": len(common), "common_gt_pair_fraction": len(common) / len(truth) if truth else None, "owner_cd_sq_mean": float(distances.mean()) if len(distances) else None, "owner_cd_sq_max": float(distances.max()) if len(distances) else None}
            rows.append(row)
            if item in outlier_ids:
                largest = sorted(range(len(distances)), key=lambda i: (-distances[i], common[i]))[:3]
                outlier_records[item]["methods"][method] = {"owner_cd_sq_mean": row["owner_cd_sq_mean"], "owner_cd_sq_max": row["owner_cd_sq_max"], "missing_directed_pairs": [{"pair": pair, "existence_probability": probabilities.get(method, {}).get(pair)} for pair in sorted(set(truth) - set(maps.get(method, {})))], "extra_directed_pairs": sorted(set(maps.get(method, {})) - set(truth)), "largest_common_owner_errors": [{"pair": common[i], "cd_sq": float(distances[i])} for i in largest]}
        source_records.append({"model_id": item, "gt_sha256": gt_sha, "common_directed_pair_count": len(common), "common_directed_pairs_sha256": hashlib.sha256(json.dumps(common, separators=(",", ":")).encode()).hexdigest()})
        if (index + 1) % 250 == 0:
            print(json.dumps({"owner_objects": index + 1, "intended": len(ids)}), flush=True)
    frames = {method: {row["model_id"]: row for row in rows if row["method"] == method} for method in METHODS}
    output = {"definition": "Common GT directed face-pairs present in all three methods; 16-point bidirectional squared nearest-neighbor means summed, per-object mean/max then macro summary.", "intended_objects": len(ids), "objects_with_common_pairs": sum(frames[METHODS[0]][item]["common_directed_pair_count"] > 0 for item in ids), "all_gt_directed_pairs": sum(frames[METHODS[0]][item]["gt_directed_pair_count"] for item in ids), "common_directed_pairs": sum(frames[METHODS[0]][item]["common_directed_pair_count"] for item in ids), "methods": {}, "pairs": {}, "source_object_records": source_records, "candidate_edge_top_one_percent_owner_details": outlier_records}
    output["common_pair_fraction"] = output["common_directed_pairs"] / output["all_gt_directed_pairs"] if output["all_gt_directed_pairs"] else None
    for method in METHODS:
        output["methods"][method] = {"status_counts": dict(Counter(row["owner_status"] for row in frames[method].values())), "metrics": {key: stats([number(row, key) for row in frames[method].values()], len(ids)) for key in ("owner_cd_sq_mean", "owner_cd_sq_max", "missing_gt_directed_pairs", "extra_directed_pairs", "common_gt_pair_fraction")}, "missing_gt_directed_pairs_total": sum(row["missing_gt_directed_pairs"] or 0 for row in frames[method].values()), "extra_directed_pairs_total": sum(row["extra_directed_pairs"] or 0 for row in frames[method].values())}
    for new, old in PAIRS:
        output["pairs"][new + "_vs_" + old] = paired(frames, ids, new, old, ("owner_cd_sq_mean", "owner_cd_sq_max"))
    return rows, output


def figure(report, destination):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    colors = ("#64748b", "#2877a9", "#d6691c")
    x = np.arange(3)
    for offset, mode in ((-.18, "fit_only"), (.18, "optimized")):
        counts = [report["post"][mode]["methods"][m]["native_solid_counts"]["valid"] for m in METHODS]
        bars = axes[0, 0].bar(x + offset, np.asarray(counts) / 2417 * 100, width=.34, label=mode, alpha=.9)
        axes[0, 0].bar_label(bars, labels=[str(c) for c in counts], padding=3, fontsize=9)
    axes[0, 0].set(title="Native valid solids / 2,417", ylabel="Valid fraction (%)", ylim=(0, 103), xticks=x, xticklabels=[LABELS[m] for m in METHODS])
    axes[0, 0].legend()
    for ax, key, title, section in ((axes[0, 1], "face_cd_sq", "Raw face CD squared", "raw"), (axes[0, 2], "edge_cd_sq", "Raw pooled-edge CD squared", "raw"), (axes[1, 0], "owner_cd_sq_mean", "Owner-matched curve CD squared\nSame directed pairs across all three", "owner")):
        for offset, statistic in ((-.18, "mean"), (.18, "median")):
            values = [report[section]["methods"][m]["metrics"][key][statistic] for m in METHODS]
            ax.bar(x + offset, [v if v is not None else np.nan for v in values], width=.34, label=statistic, alpha=.9)
        ax.set(title=title, ylabel="Squared normalized distance", xticks=x, xticklabels=[LABELS[m] for m in METHODS])
        ax.set_yscale("log")
        ax.legend()
    labels = ["Value vs source", "Control vs source", "Value vs control"]
    for offset, field, color in ((-.18, "rescued", "#2f8b70"), (.18, "regressed", "#bd4d39")):
        values = [report["post"]["optimized"]["pairs"][new + "_vs_" + old]["native_solid_transitions"]["counts"][field] for new, old in PAIRS]
        bars = axes[1, 1].bar(x + offset, values, width=.34, label=field, color=color)
        axes[1, 1].bar_label(bars, padding=3)
    axes[1, 1].set(title="Optimized solid transitions", ylabel="Objects", xticks=x, xticklabels=labels)
    axes[1, 1].legend()
    shares = [report["raw"]["top_one_percent"]["edge_cd_sq"][m]["top_error_share"] for m in METHODS]
    bars = axes[1, 2].bar(x, [100 * v if v is not None else np.nan for v in shares], color=colors)
    axes[1, 2].bar_label(bars, fmt="%.1f%%", padding=3)
    axes[1, 2].set(title="Worst 1% share of total raw edge error\nEach method ranks its own finite predictions", ylabel="Error share (%)", ylim=(0, 105), xticks=x, xticklabels=[LABELS[m] for m in METHODS])
    fig.suptitle("Encoder edge-content experiment: three-way fixed-cohort results\nBoth trainable models selected at epoch 2 / step 1,089 by validation; 7 unsupported CAD remain separate.", fontsize=14)
    for suffix in ("png", "svg"):
        fig.savefig(destination / ("summary_figure." + suffix), dpi=160)
    plt.close(fig)


def self_check():
    curve = np.stack((np.linspace(0, 1, 16), np.zeros(16), np.zeros(16)), axis=1)[None]
    require(np.array_equal(owner_cd(curve, curve), np.zeros(1)), "Identical curve check")
    moved = curve.copy()
    moved[..., 1] += .125
    require(np.allclose(owner_cd(moved, curve), 2 * .125 ** 2, rtol=0, atol=1e-16), "Known translation check")
    require(boolean("False") is False and boolean("") is None, "Strict boolean check")
    require(stats([], 3)["missing"] == 3 and stats([], 3)["mean"] is None, "Missing metric check")
    require(len(owner_cd(np.empty((0, 16, 3)), np.empty((0, 16, 3)))) == 0, "Empty pair check")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print(json.dumps({"self_checks": "passed", "final_outputs_written": False}))
        return
    targets = [ROOT / "diagnostics.json", ROOT / "owner_per_object.csv", ROOT / "input_hashes.jsonl", ROOT / "visualizations/summary_figure.png", ROOT / "visualizations/summary_figure.svg"]
    require(not any(path.exists() for path in targets), "Refuse to overwrite existing diagnostics")
    inputs = Inputs()
    split = inputs.read(SPLIT)
    require(hashlib.sha256(split).hexdigest() == SPLIT_SHA, "Fixed split changed")
    ids = split.decode().splitlines()
    require(len(ids) == len(set(ids)) == 2417, "Invalid fixed cohort")
    inputs.read(ROOT / "PLAN.md")
    inputs.read(ROOT / "DIAGNOSTICS_SPEC.md")
    inputs.read(Path(__file__))
    training = inputs.json(ROOT / "training_audit.json")
    raw, raw_summary = load_raw(inputs, ids)
    post, post_summary = {}, {}
    for mode in ("fit_only", "optimized"):
        post[mode], post_summary[mode] = load_post(inputs, ids, mode)
    owner_rows, owner_summary = owner_diagnostics(inputs, ids, raw, post, training,
                                                 set(raw_summary["top_one_percent"]["edge_cd_sq"]["encoder_edge_value"]["top_ids"]))
    report = {"intended_count": 2417, "original_count": 2424, "unsupported_count": 7, "split_sha256": SPLIT_SHA,
              "methods": list(METHODS), "paired_change_tolerance": TIE, "raw": raw_summary, "post": post_summary, "owner": owner_summary,
              "selected_checkpoints": {arm: {key: training["arms"][arm]["selected"][key] for key in ("path", "sha256", "epoch", "global_step")} for arm in ARMS},
              "checks": {"numeric_self_checks": "passed", "baseline_consistency_between_pair_roots": True, "upstream_summary_statistics_recomputed": True, "all_post_rows_terminal": True, "cad_and_latent_digests_verified": True, "source_files_checked_before_after_read": True},
              "limitations": ["All means are macro averages over computable objects; missing counts retain the fixed intended cohort.", "Geometry metrics are squared sampled distances, not maximum CAD tolerances or physical-edge matching.", "Owner geometry covers only common GT directed face-pairs predicted by all three; missed/extra pairs and coverage are reported separately.", "Exact-adjacency subsets are descriptive and do not establish loop/vertex topology correctness or causal explanations.", "Historical source remains a prior frozen-encoder model; candidate/control checkpoint exposure matches but full run budgets and capacity do not.", "Post geometry common subsets may contain invalid solids; geometry and native validity are separate."]}
    # Recheck scalar/CSV/spec inputs after the expensive owner pass. Array files
    # were hashed immediately before and after each complete CRC-checked read.
    for path, record in inputs.records.items():
        if not path.endswith(".npz"):
            require(sha256(path) == record["sha256"], f"Summary input changed during execution: {path}")
    with tempfile.TemporaryDirectory(prefix=".diagnostics-build-", dir=ROOT) as temp:
        stage = Path(temp)
        with (stage / "owner_per_object.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(owner_rows[0]))
            writer.writeheader()
            writer.writerows(owner_rows)
        with (stage / "input_hashes.jsonl").open("w") as handle:
            for path in sorted(inputs.records):
                handle.write(json.dumps(inputs.records[path], ensure_ascii=False) + "\n")
        report["input_hash_manifest"] = {"path": str(ROOT / "input_hashes.jsonl"), "sha256": sha256(stage / "input_hashes.jsonl"), "file_count": len(inputs.records)}
        report["owner_csv_sha256"] = sha256(stage / "owner_per_object.csv")
        (stage / "diagnostics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        figure(json.loads((stage / "diagnostics.json").read_text()), stage)
        require(not any(path.exists() for path in targets), "Output appeared during computation")
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            (stage / target.name).replace(target)
    print(json.dumps({"output": str(ROOT / "diagnostics.json"), "intended_count": len(ids), "owner_objects": owner_summary["objects_with_common_pairs"], "optimized_solids": {m: post_summary["optimized"]["methods"][m]["native_solid_counts"] for m in METHODS}}))


if __name__ == "__main__":
    main()
