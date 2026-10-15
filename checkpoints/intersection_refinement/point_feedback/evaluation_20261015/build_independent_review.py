"""Independent checks of point-feedback evaluation against conditioned records; no pipeline imports.

Run after raw inference/metrics, then rerun after post processing and optionally
with --census-root once complete. The only output is independent_review.json.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip())
TERMINAL = {"ok", "missing_raw", "invalid_raw", "post_error", "no_step", "step_read_error",
            "geometry_error", "timeout", "crash", "interrupted"}
THRESHOLDS = {"0.002", "0.006", "0.01", "0.08"}
OBSERVATIONS = {
    "face_without_boundary", "never_wire", "never_closed_admitted",
    "closed_admitted_still_failed", "strict_closed_candidate_failed",
    "final_success_discarded_wire", "raw_extra_gt_pair", "raw_missing_gt_pair",
    "raw_duplicate_directed_pair", "raw_same_face_pair", "face_count_mismatch",
    "selection_removed_gt_pair", "reciprocal_excess_disagreement",
    "raw_new_cross_pair_curve_nearness", "raw_new_face_grid_nearness",
    "selected_endpoint_gap", "optimized_endpoint_gap",
    "selected_new_open_edge_endpoint_collapse", "optimized_new_open_edge_endpoint_collapse",
    "fit_face_residual", "fit_curve_residual", "owner_independent", "owner_after",
    "closed_failed_owner_gap", "optimization_face_away", "optimization_edge_away",
    "sewing_free_edges", "saved_near_free_edges",
}



def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def close(a, b, label: str) -> None:
    if a is None or b is None:
        assert a is b, label
    else:
        assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-15), (label, a, b)


def quantile90(values: list[float]) -> float:
    values = sorted(values)
    location = (len(values) - 1) * 0.9
    low = math.floor(location)
    high = math.ceil(location)
    return values[low] + (values[high] - values[low]) * (location - low)


def verify_stats(values, expected: dict, label: str) -> None:
    numbers = [float(v) for v in values if v not in (None, "") and math.isfinite(float(v))]
    assert len(numbers) == expected["count"], label
    close(math.fsum(numbers) / len(numbers) if numbers else None, expected["mean"], label + ":mean")
    close(statistics.median(numbers) if numbers else None, expected["median"], label + ":median")
    close(quantile90(numbers) if numbers else None, expected["p90"], label + ":p90")


def source_review() -> dict:
    """Verify the exact method-only extension, including new init-source clearing.

    Inspect complete module ASTs, not just a permissive function-name allowlist.
    Numeric inference/metric/post functions must remain identical to the commit
    used for training. Explicit identity validation is checked independently.
    """
    result = {}
    for filename in ("tools/intersection_evaluate_inference.py", "tools/vae_compare_metrics.py", "tools/vae_compare_post.py"):
        before = subprocess.check_output(["git", "show", BASE_COMMIT + ":" + filename], text=True, cwd=REPO)
        after = (REPO / filename).read_text()
        previous_tree, current_tree = ast.parse(before), ast.parse(after)
        normalized = deepcopy(current_tree)
        previous = {node.name: node for node in previous_tree.body if isinstance(node, ast.FunctionDef)}
        current = {node.name: node for node in current_tree.body if isinstance(node, ast.FunctionDef)}
        dumped = lambda node: ast.dump(node, include_attributes=False)
        changed = {name for name in set(previous) | set(current) if dumped(previous.get(name)) != dumped(current.get(name))}
        if filename.endswith("intersection_evaluate_inference.py"):
            assert changed == {"load_intersection", "build_parser", "validate_method"}, changed
            loader = deepcopy(current["load_intersection"])
            removed_init = 0
            for node in ast.walk(loader):
                if isinstance(node, ast.Dict):
                    for index in reversed(range(len(node.keys))):
                        key = node.keys[index]
                        if isinstance(key, ast.Constant) and key.value == "init_curve_feedback_checkpoint":
                            assert isinstance(node.values[index], ast.Constant) and node.values[index].value is None
                            del node.keys[index], node.values[index]
                            removed_init += 1
            assert removed_init == 1 and dumped(loader) == dumped(previous["load_intersection"])
            parser = deepcopy(current["build_parser"])
            extended_choices = 0
            for node in ast.walk(parser):
                if isinstance(node, ast.keyword) and node.arg == "choices" and isinstance(node.value, ast.Tuple):
                    values = [item.value for item in node.value.elts if isinstance(item, ast.Constant)]
                    if "point_feedback" in values:
                        assert values == ["intersection", "local", "conditioned", "point_feedback"]
                        node.value.elts.pop()
                        extended_choices += 1
            assert extended_choices == 1 and dumped(parser) == dumped(previous["build_parser"])
            expected_validator = ast.parse('''
def validate_method(model: torch.nn.Module, method: str) -> None:
    """Keep experimental identity explicit while using the same strict loader."""
    kind = model.decoder.intersection_kind
    conditioned = model.decoder.curve_conditioning_enabled
    feedback = model.decoder.curve_point_feedback_enabled
    expected = {"intersection": ("legacy", False, False), "local": ("local_attention", False, False),
                "conditioned": ("legacy", True, False), "point_feedback": ("legacy", True, True)}
    if method not in expected or (kind, conditioned, feedback) != expected[method]:
        raise ValueError(f"Method {method} does not match checkpoint intersection={kind}, "
                         f"curve_conditioning={conditioned}, curve_point_feedback={feedback}")
    if method in {"conditioned", "point_feedback"} and not model.decoder.refinement_enabled:
        raise ValueError(f"Method {method} requires enabled refinement")
''').body[0]
            assert dumped(current["validate_method"]) == dumped(expected_validator)
        if filename == "tools/vae_compare_metrics.py":
            assert changed == {"summarize"}
            summarizer = deepcopy(current["summarize"])
            changed_description = 0
            for node in ast.walk(summarizer):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and "legacy, intersection, local, conditioned and point_feedback" in node.value:
                    node.value = node.value.replace("legacy, intersection, local, conditioned and point_feedback", "legacy, intersection, local and conditioned")
                    changed_description += 1
            assert changed_description == 1 and dumped(summarizer) == dumped(previous["summarize"])
        if filename == "tools/vae_compare_post.py":
            assert not changed, "Post functions changed"
        for index, node in enumerate(normalized.body):
            if isinstance(node, ast.FunctionDef) and node.name in changed:
                normalized.body[index] = deepcopy(previous[node.name])
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in {"CANDIDATE_METHODS", "BASELINE_METHODS"}:
                    previous_node = next(item for item in previous_tree.body if isinstance(item, ast.Assign)
                                         and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name) and item.targets[0].id == name)
                    old_values, new_values = ast.literal_eval(previous_node.value), ast.literal_eval(node.value)
                    assert new_values == old_values + (("point_feedback",) if name == "CANDIDATE_METHODS" else ("conditioned",))
                    normalized.body[index] = deepcopy(previous_node)
        assert dumped(normalized) == dumped(previous_tree), f"Unexpected top-level/source change in {filename}"
        result[filename] = {"sha256": digest(REPO / filename), "changed_functions_vs_base": sorted(changed),
                            "unchanged_function_count": len(set(previous) & set(current)) - len(changed)}
    unchanged_post = not subprocess.check_output(["git", "diff", BASE_COMMIT, "--", "src/brepnet/post", "shared"], cwd=REPO)
    assert unchanged_post
    return {"base_commit": BASE_COMMIT, "files": result,
            "production_post_and_shared_unchanged_vs_base": True,
            "manual_review": "Full-module AST checked: method enums extend explicitly; inference additionally clears init_curve_feedback_checkpoint for strict full-load and verifies feedback identity. All numeric inference, raw metric, post worker, validity, sampling and timeout computations are unchanged."}



def input_review(mode: str) -> dict:
    metadata = ROOT / "metadata/inference" / mode
    protocol = read(metadata / "protocol.json")
    assert Path(protocol["checkpoint"]) == SELECTED and protocol["checkpoint_sha256"] == SELECTED_SHA
    checkpoint_audit = protocol["model_audit"]
    assert checkpoint_audit["strict_load"] is True
    assert checkpoint_audit["intersection"]["epoch"] == SELECTED_RECORD["epoch"]
    assert checkpoint_audit["intersection"]["global_step"] == SELECTED_RECORD["global_step"]
    cfg = checkpoint_audit["checkpoint_model_config"]
    assert cfg["intersection"]["kind"] == "legacy"
    assert cfg["intersection_layers"] == 8 and cfg["intersection_dim"] == 512
    assert cfg["curve_conditioning"] == {"enabled": True, "hidden_dim": 128}
    assert cfg["curve_point_feedback"] == {"enabled": True, "hidden_dim": 128, "feature_dim": 64}
    assert cfg["refinement"]["enabled"] is True
    assert protocol["method"] == "point_feedback" and protocol["intended_count"] == 2417
    assert protocol["split_sha256"] == EXPECTED_SHA
    assert protocol["edge_threshold"] == 0.5
    assert protocol["decoder_gt_topology"] is False and protocol["decoder_face_deduplication"] is False
    assert protocol["model_audit"]["encoder_equal_tensor_count"] == 243
    numeric = protocol["numeric_settings"]
    assert numeric["dtype"] == "float32" and numeric["seed"] == 42
    assert numeric["autocast"] is False and numeric["matmul_allow_tf32"] is False and numeric["cudnn_allow_tf32"] is False
    for path, expected in protocol["source_sha256"].items():
        assert digest(REPO / path) == expected, path
    rows = [json.loads(line) for path in sorted(metadata.glob("shard-???.jsonl"))
            for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == len(IDS) and {row["model_id"] for row in rows} == set(IDS)
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["method"] == "point_feedback" and row["mode"] == mode for row in rows)
    by_id = {row["model_id"]: row for row in rows}
    means = [read(path) for path in metadata.glob("shard-???-mean-audit-*.json")]
    assert len(means) == protocol["num_shards"] and all(item["passed"] for item in means)
    reference_protocol = LATENT_SOURCE / "metadata/inference" / mode / "protocol.json"
    assert protocol["reference_root"] == str(LATENT_SOURCE)
    assert protocol["reference_protocol_file_sha256"] == digest(reference_protocol)
    assert protocol["reference_protocol_sha256"] == fingerprint(read(reference_protocol))
    baseline_folder = ROOT / "raw_predictions" / mode / "conditioned"
    assert baseline_folder.is_symlink()
    assert baseline_folder.resolve() == (OLD / "raw_predictions" / mode / "conditioned").resolve()
    input_hashes = {}
    for model_id in IDS:
        candidate_path = ROOT / "raw_predictions" / mode / "point_feedback" / model_id / "data.npz"
        baseline_path = ROOT / "raw_predictions" / mode / "conditioned" / model_id / "data.npz"
        source_path = LATENT_SOURCE / "raw_predictions" / mode / "legacy" / model_id / "data.npz"
        with np.load(candidate_path, allow_pickle=False) as candidate, np.load(baseline_path, allow_pickle=False) as baseline, np.load(source_path, allow_pickle=False) as source:
            assert str(candidate["comparison_mode"]) == mode
            values = candidate["face_latents"]
            assert values.dtype == np.float32 and np.isfinite(values).all()
            assert np.array_equal(values, baseline["face_latents"]) and np.array_equal(values, source["face_latents"])
            h = hashlib.sha256(str(values.shape).encode())
            h.update(np.ascontiguousarray(values).tobytes())
            expected = h.hexdigest()
            assert expected == str(candidate["input_latent_sha256"]) == str(baseline["input_latent_sha256"])
            assert expected == by_id[model_id]["input_latent_sha256"]
            assert str(candidate["reference_raw_sha256"]) == digest(source_path) == by_id[model_id]["reference_raw_sha256"]
            assert str(candidate["protocol_sha256"]) == fingerprint(protocol)
            assert str(candidate["representation"]) == "legacy_half_edges"
            assert np.isfinite(candidate["pred_face"]).all() and np.isfinite(candidate["pred_edge"]).all()
            assert candidate["pred_face"].dtype == candidate["pred_edge"].dtype == np.float32
            probability = candidate["all_candidate_prob"]
            pairs = candidate["all_candidate_pairs"]
            assert np.array_equal(candidate["pred_edge_face_connectivity"][:, 1:], pairs[probability > 0.5])
            count = len(values)
            assert len(pairs) == count * (count - 1)
            assert not np.any(pairs[:, 0] == pairs[:, 1])
            expected_pairs = np.asarray([(first, second) for first in range(count) for second in range(count)
                                         if first != second], dtype=pairs.dtype).reshape(-1, 2)
            assert np.array_equal(pairs, expected_pairs)
            assert probability.shape == (len(pairs),) and np.isfinite(probability).all()
            assert ((probability >= 0) & (probability <= 1)).all()
            input_hashes[model_id] = expected
    return {"intended": len(IDS), "point_feedback_prediction_count": len(rows), "status_counts": dict(Counter(row["status"] for row in rows)),
            "validation_selected_checkpoint_verified": str(SELECTED),
            "exact_input_array_comparisons": len(IDS), "input_hash_manifest_sha256": fingerprint(input_hashes),
            "source_hashes_unchanged": True, "all_candidate_threshold_exports_verified": True,
            "mean_audit_shards": len(means), "max_reencoding_mean_abs_error": max(item["max_abs_error"] for item in means)}


def raw_review(mode: str) -> dict:
    directory = ROOT / "raw_metrics" / mode
    summary = read(directory / "summary.json")
    old_summary = read(OLD / "raw_metrics" / mode / "summary.json")
    rows = list(csv.DictReader((directory / "per_object.csv").open()))
    old_rows = list(csv.DictReader((OLD / "raw_metrics" / mode / "per_object.csv").open()))
    assert len(rows) == 4834
    maps = {method: {r["model_id"]: r for r in rows if r["method"] == method} for method in ("conditioned", "point_feedback")}
    assert all(set(group) == set(IDS) for group in maps.values())
    old_map = {r["model_id"]: r for r in old_rows if r["method"] == "conditioned"}
    assert summary["methods"]["conditioned"] == old_summary["methods"]["conditioned"]
    for model_id, row in maps["conditioned"].items():
        previous = old_map[model_id]
        assert {k: v for k, v in row.items() if k != "prediction_file"} == {k: v for k, v in previous.items() if k != "prediction_file"}
    checked_stats = 0
    for method, group in maps.items():
        for metric, values in summary["methods"][method]["metrics"].items():
            verify_stats([row.get(metric) for row in group.values()], values, mode + ":" + method + ":" + metric)
            checked_stats += 1
        micro = summary["methods"][method]["topology_micro"]
        selected = [row for row in group.values() if row.get("topology_f1")]
        assert micro["objects"] == len(selected)
        if selected:
            for kind in ("tp", "fp", "fn"):
                assert micro[kind] == sum(int(row["topology_" + kind]) for row in selected)
            tp, fp, fn = (micro[kind] for kind in ("tp", "fp", "fn"))
            close(tp / (tp + fp) if tp + fp else 1.0, micro["precision"], mode + method + ":micro_precision")
            close(tp / (tp + fn) if tp + fn else 1.0, micro["recall"], mode + method + ":micro_recall")
            close(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0, micro["f1"], mode + method + ":micro_f1")
        else:
            assert all(micro[key] is None for key in ("precision", "recall", "f1"))
    for metric, paired in summary["paired"].items():
        matched = [(maps["conditioned"][model_id].get(metric), maps["point_feedback"][model_id].get(metric)) for model_id in IDS
                   if maps["conditioned"][model_id].get("shared_input_status") == "verified"
                   and maps["conditioned"][model_id].get(metric) not in (None, "")
                   and maps["point_feedback"][model_id].get(metric) not in (None, "")]
        differences = [float(b) - float(a) for a, b in matched]
        assert paired["common_count"] == len(matched)
        verify_stats(differences, paired["point_feedback_minus_conditioned"], mode + ":paired:" + metric)
    return {"row_count": len(rows), "stats_blocks_independently_recalculated": checked_stats,
            "objects_with_exact_adjacency": {
                method: sum(row.get("topology_f1") not in (None, "") and float(row["topology_f1"]) == 1.0 for row in group.values())
                if mode == "posterior_mean" else None for method, group in maps.items()
            },
            "baseline_metrics_exactly_equal_previous": True,
            "baseline_csv_exact_except_prediction_path": True, "paired_differences_verified": True,
            "summary_sha256": digest(directory / "summary.json"), "csv_sha256": digest(directory / "per_object.csv")}


def distribution_review() -> dict:
    directory = ROOT / "diagnostics"
    names = ("raw_distribution.json", "adjacency_subset.json", "reciprocal_distribution.json", "outlier_pair_probe.json", "reuse_provenance.json")
    documents = {name: read(directory / name) for name in names}
    sources_checked = 0
    for name, document in documents.items():
        methods = list(document["methods"])
        assert methods == ["conditioned", "point_feedback"], (name, methods)
        for path, expected in document.get("sources_sha256", document.get("source_sha256", {})).items():
            assert digest(Path(path)) == expected, (name, path)
            sources_checked += 1
    for name, script, spec in (("raw_distribution.json", "raw_distribution_analysis.py", "RAW_DISTRIBUTION_SPEC.md"),
                               ("reciprocal_distribution.json", "reciprocal_distribution_analysis.py", "RECIPROCAL_SPEC.md")):
        assert documents[name]["script_sha256"] == digest(OLD / "diagnostics" / script)
        assert documents[name]["spec_sha256"] == digest(directory / spec)
    maps = {}
    for mode in ("posterior_mean", "archived_latents"):
        records = csv_rows(ROOT / "raw_metrics" / mode / "per_object.csv")
        maps[mode] = {method: {row["model_id"]: row for row in records if row["method"] == method} for method in ("conditioned", "point_feedback")}
    exact = {sid for sid in IDS if all(maps["posterior_mean"][method][sid]["topology_fp"] == "0"
                                     and maps["posterior_mean"][method][sid]["topology_fn"] == "0" for method in maps["posterior_mean"])}
    assert exact == set(documents["adjacency_subset.json"]["cohorts"]["both_adjacency_exact"]["ids"])
    assert exact == set(documents["reciprocal_distribution.json"]["both_adjacency_exact_ids"])
    checked = 0

    def stats(values, expected, label):
        nonlocal checked
        verify_stats(values, expected, label)
        ordered = sorted(values)
        close(max(values) if values else None, expected["max"], label + ":max")
        for percentile in (95, 99):
            value = None
            if values:
                location = (len(values) - 1) * percentile / 100
                left, right = math.floor(location), math.ceil(location)
                value = ordered[left] + (ordered[right] - ordered[left]) * (location - left)
            close(value, expected[f"p{percentile}"], label + f":p{percentile}")
        checked += 1

    def pair_stats(mode, ids, metric, block):
        common = [sid for sid in sorted(ids) if all(maps[mode][method][sid].get(metric) not in (None, "") for method in maps[mode])]
        first, second = ([float(maps[mode][method][sid][metric]) for sid in common] for method in ("conditioned", "point_feedback"))
        assert len(common) == block["common_count"]
        stats(first, block["conditioned"], mode + metric + ":baseline")
        stats(second, block["point_feedback"], mode + metric + ":candidate")
        delta = [b - a for a, b in zip(first, second)]
        stats(delta, block["paired_delta"], mode + metric + ":delta")
        stats([b / a for a, b in zip(first, second) if a > 0], block["paired_ratio"], mode + metric + ":ratio")
        assert block["win_count"] == sum(value < -1e-12 for value in delta)
        assert block["loss_count"] == sum(value > 1e-12 for value in delta)
        assert block["tie_count"] == sum(abs(value) <= 1e-12 for value in delta)

    for mode, value in documents["raw_distribution.json"]["modes"].items():
        for metric, block in value["metrics"].items():
            pair_stats(mode, IDS, metric, block)
    for mode, value in documents["reciprocal_distribution.json"]["modes"].items():
        for cohort, data in value["cohorts"].items():
            subset = IDS if cohort == "all" else exact if cohort == "both_adjacency_exact" else set(IDS) - exact
            assert len(subset) == data["count"]
            for metric, block in data["metrics"].items():
                pair_stats(mode, subset, metric, block)
    candidate = maps["posterior_mean"]["point_feedback"]
    baseline = maps["posterior_mean"]["conditioned"]
    highest_error = min(IDS, key=lambda sid: (-float(candidate[sid]["edge_cd_sq"]), sid))
    highest_reciprocal_increase = min(IDS, key=lambda sid: (-(float(candidate[sid]["legacy_reciprocal_cd_sq_mean"]) - float(baseline[sid]["legacy_reciprocal_cd_sq_mean"])), sid))
    probe = documents["outlier_pair_probe.json"]
    assert probe["sample_ids"] == [highest_error, highest_reciprocal_increase] == ["00335203", "00326127"]
    revision = read(ROOT / "metadata/diagnostic_spec_revision/outlier_pair_probe.json")
    assert revision["sample_ids"] == probe["sample_ids"] and revision["methods"] == probe["methods"]
    assert revision["samples"] == probe["samples"], "Corrected specification rerun changed numerical evidence"
    return {"method_identity_verified": True, "source_hashes_checked": sources_checked,
            "distribution_stats_blocks_recalculated": checked, "both_exact_adjacency_count": len(exact),
            "extreme_ids_independently_selected_from_csv": probe["sample_ids"],
            "specification_revision_numerical_evidence_unchanged": True,
            "artifact_sha256": {name: digest(directory / name) for name in names}}


def historical_review() -> dict:
    report_path = ROOT / "historical_reference.json"
    report = read(report_path)
    original_root = Path("/mnt/d/python/checkpoints/intersection_refinement/evaluation_20261011")
    methods = ("intersection", "conditioned", "point_feedback")
    assert report["methods"] == list(methods)
    assert report["intended_count"] == 2417 and report["original_count"] == 2424
    for path, expected in report["sources_sha256"].items():
        assert digest(Path(path)) == expected, path
    numeric_blocks = 0
    for mode in ("posterior_mean", "archived_latents"):
        tables = {}
        for method in methods:
            root = original_root if method == "intersection" else ROOT
            rows = csv_rows(root / "raw_metrics" / mode / "per_object.csv")
            selected = [row for row in rows if row["method"] == method]
            table = {row["model_id"]: row for row in selected}
            assert len(table) == len(selected) == 2417 and set(table) == set(IDS)
            tables[method] = table
        for sid in IDS:
            hashes = {tables[method][sid]["input_latent_sha256"] for method in methods}
            assert len(hashes) == 1 and "" not in hashes
        for metric, block in report["raw"][mode].items():
            common = [sid for sid in IDS if all(tables[method][sid].get(metric) not in (None, "") for method in methods)]
            assert len(common) == block["common_count"] and 2417 - len(common) == block["missing_count"]
            for method in methods:
                numbers = [float(tables[method][sid][metric]) for sid in common]
                assert all(math.isfinite(value) and value >= 0 for value in numbers)
                close(math.fsum(numbers) / len(numbers) if numbers else None, block[method]["mean"], metric + method)
                close(statistics.median(numbers) if numbers else None, block[method]["median"], metric + method)
                numeric_blocks += 1
    post_counts = {}
    for mode in ("fit_only", "optimized"):
        first, current = [read(root / "metadata/post" / mode / "protocol.json") for root in (original_root, ROOT)]
        for field in ("split_sha256", "timeout_seconds", "max_optimize_iter", "adaptive_max_optimize_iter",
                      "drop_num", "use_cuda", "threads_per_process", "post_algorithm_files"):
            assert first[field] == current[field], (mode, field)
        outcomes = {}
        for method in methods:
            root = original_root if method == "intersection" else ROOT
            selected = [row for row in csv_rows(root / "post" / mode / "samples.csv") if row["method"] == method]
            assert len(selected) == 2417 and {row["sample_id"] for row in selected} == set(IDS)
            assert all(row["native_valid_solid"] in ("True", "False") for row in selected)
            outcomes[method] = {row["sample_id"]: row["native_valid_solid"] == "True" for row in selected}
            count = sum(outcomes[method].values())
            assert report["post"][mode][method]["native_valid_solid"] == count
            close(count / 2417, report["post"][mode][method]["rate_over_accepted"], mode + method)
        transitions = report["post"][mode]["point_feedback_vs_original"]
        assert transitions["rescued_ids"] == sorted(sid for sid in IDS if outcomes["point_feedback"][sid] and not outcomes["intersection"][sid])
        assert transitions["regressed_ids"] == sorted(sid for sid in IDS if outcomes["intersection"][sid] and not outcomes["point_feedback"][sid])
        post_counts[mode] = {method: sum(values.values()) for method, values in outcomes.items()}
    return {"sources_rehashed": len(report["sources_sha256"]), "shared_input_hashes_verified_per_mode": 2417,
            "three_method_numeric_blocks_recalculated": numeric_blocks, "post_counts": post_counts,
            "original_to_point_feedback_transition_ids_verified": True, "report_sha256": digest(report_path)}


def post_review(mode: str) -> dict:
    directory = ROOT / "post" / mode
    metadata = ROOT / "metadata/post" / mode
    summary = read(directory / "summary.json")
    old_summary = read(OLD / "post" / mode / "summary.json")
    protocol = read(metadata / "protocol.json")
    reference = protocol["reference_baseline"]
    assert summary["intended_count_per_method"] == 2417 and summary["original_test_count"] == 2424
    assert summary["methods"]["conditioned"] == old_summary["methods"]["conditioned"]
    old_protocol = read(OLD / "metadata/post" / mode / "protocol.json")
    assert protocol["baseline_method"] == "conditioned" and protocol["candidate_method"] == "point_feedback"
    assert reference["method"] == "conditioned" and reference["result_origin"] == "reference_reused"
    assert Path(reference["reference_root"]).resolve() == OLD.resolve()
    assert protocol["post_algorithm_files"] == old_protocol["post_algorithm_files"]
    assert protocol["timeout_seconds"] == 180 and protocol["drop_num"] == 0
    assert protocol["threads_per_process"] == 1 and protocol["use_cuda"] is False
    for field in reference["verified_contract_fields"]:
        assert protocol[field] == old_protocol[field], field
    assert digest(OLD / "metadata/post" / mode / "protocol.json") == reference["source_protocol_sha256"]
    assert digest(REPO / "tools/vae_compare_post.py") == protocol["runner_sha256"]
    for path, sha in protocol["post_algorithm_files"].items():
        assert digest(REPO / path) == sha
    rows = list(csv.DictReader((directory / "samples.csv").open()))
    assert len(rows) == 4834
    csv_maps = {method: {r["sample_id"]: r for r in rows if r["method"] == method} for method in ("conditioned", "point_feedback")}
    maps = {method: {} for method in csv_maps}
    record_hashes, step_hashes = {}, {}
    for method in maps:
        assert set(csv_maps[method]) == set(IDS)
        for model_id in IDS:
            base = OLD if method == "conditioned" else ROOT
            path = base / "metadata/post" / mode / "samples" / method / (model_id + ".json")
            row = read(path)
            assert row["sample_id"] == model_id and row["method"] == method and row["mode"] == mode
            assert row["status"] in TERMINAL
            assert row["raw_sha256"] == digest(base / "raw_predictions/posterior_mean" / method / model_id / "data.npz")
            assert row["protocol_fingerprint"] == fingerprint(old_protocol if method == "conditioned" else protocol)
            for field in ("native_valid_solid", "compat_valid_solid", "step_exists", "step_readable", "success_marker"):
                assert row.get(field) in (None, True, False), (method, model_id, field)
                value = csv_maps[method][model_id].get(field, "")
                assert value == ("" if row.get(field) is None else str(row[field]))
            assert csv_maps[method][model_id]["result_origin"] == ("reference_reused" if method == "conditioned" else "current_run")
            if method == "conditioned":
                assert Path(csv_maps[method][model_id]["source_record_path"]).resolve() == path.resolve()
                record_hashes[model_id] = digest(path)
                assert csv_maps[method][model_id]["source_record_sha256"] == record_hashes[model_id]
                step_path = base / "post" / mode / method / model_id / "recon_brep.step"
                if step_path.is_file():
                    step_hashes[model_id] = digest(step_path)
                    assert csv_maps[method][model_id]["source_step_sha256"] == step_hashes[model_id]
            maps[method][model_id] = row
    assert fingerprint(record_hashes) == reference["record_hash_manifest_sha256"]
    assert fingerprint(step_hashes) == reference["step_hash_manifest_sha256"]
    for method, group in maps.items():
        recorded = summary["methods"][method]
        for field, count in recorded["counts"].items():
            if field == "geometry_computable":
                calculated = sum(row.get("geometry_status") == "ok" for row in group.values())
            elif field == "terminal_records":
                calculated = sum(row["status"] in TERMINAL for row in group.values())
            else:
                calculated = sum(row.get(field) is True for row in group.values())
            assert calculated == count
            close(calculated / 2417, recorded["rates_over_intended"][field], method + field)
            close(calculated / 2424, recorded["rates_over_original_test"][field], method + field)
        for metric, expected in recorded["geometry_all_computable"].items():
            verify_stats([row.get(metric) for row in group.values() if row.get("geometry_status") == "ok"], expected, mode + method + metric)
    transitions = {name: [] for name in summary["native_validity_transitions"]["counts"]}
    for model_id in sorted(IDS):
        old, new = maps["conditioned"][model_id], maps["point_feedback"][model_id]
        if old.get("native_valid_solid") is True:
            name = "both_valid" if new.get("native_valid_solid") is True else "conditioned_valid_to_point_feedback_failed"
        else:
            name = "conditioned_failed_to_point_feedback_valid" if new.get("native_valid_solid") is True else "both_failed"
        transitions[name].append(model_id)
    assert transitions == summary["native_validity_transitions"]["ids"]
    assert sum(map(len, transitions.values())) == 2417
    common = [model_id for model_id in IDS if all(maps[method][model_id].get("geometry_status") == "ok" for method in maps)]
    assert len(common) == summary["paired_common_geometry"]["count"]
    for metric, values in summary["paired_common_geometry"]["metrics"].items():
        for method in maps:
            verify_stats([maps[method][item][metric] for item in common], values[method], mode + metric + method)
        verify_stats([maps["point_feedback"][item][metric] - maps["conditioned"][item][metric] for item in common], values["point_feedback_minus_conditioned"], mode + metric + ":difference")
    return {"terminal_records_per_method": {method: len(group) for method, group in maps.items()},
            "transition_counts": {name: len(group) for name, group in transitions.items()},
            "counts_and_fixed_denominator_rates_verified": True, "common_geometry_count": len(common),
            "baseline_summary_exactly_equal_previous": True, "baseline_record_hashes_unchanged_since_reuse": len(record_hashes),
            "baseline_step_hashes_unchanged_since_reuse": len(step_hashes), "post_algorithms_unchanged": True,
            "source_step_hash_limitation": "Checks protect assets since this evaluation's reuse snapshot, not against unrecorded historical mutations.",
            "summary_sha256": digest(directory / "summary.json"), "csv_sha256": digest(directory / "samples.csv")}


def csv_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def frozen_identities():
    manifest = read(CENSUS / "manifest.json")
    baseline = read(CENSUS / "baseline_manifest.json")
    controls = read(CENSUS / "baseline_control_manifest.json")
    reuse = read(CENSUS / "baseline_reuse.json")
    ids = set(SPLIT.read_text().split())
    assert len(ids) == 2417 and digest(SPLIT) == EXPECTED_SHA
    rows = csv_rows(EVAL / "post/optimized/samples.csv")
    old = {row["sample_id"]: row for row in rows if row["method"] == "conditioned"}
    new = {row["sample_id"]: row for row in rows if row["method"] == "point_feedback"}
    assert set(old) == set(new) == ids
    groups = {name: [] for name in ("both_failed", "new_failure", "rescued", "both_valid")}
    for sid in sorted(ids):
        assert old[sid]["native_valid_solid"] in ("True", "False") and new[sid]["native_valid_solid"] in ("True", "False")
        a, b = old[sid]["native_valid_solid"] == "True", new[sid]["native_valid_solid"] == "True"
        name = "both_valid" if a and b else "new_failure" if a else "rescued" if b else "both_failed"
        groups[name].append(sid)
    assert groups == manifest["cohorts"]
    failed = sorted(groups["both_failed"] + groups["new_failure"])
    selected = set(failed + groups["rescued"])
    assert failed == manifest["failed_ids"]
    for document, method in ((manifest, "point_feedback"), (baseline, "conditioned")):
        assert len(document["cases"]) == len(selected) and {case["id"] for case in document["cases"]} == selected
        assert all(case["method"] == method for case in document["cases"])
        outcomes = new if method == "point_feedback" else old
        assert all(case["post_record"]["native_valid_solid"] is (outcomes[case["id"]]["native_valid_solid"] == "True") for case in document["cases"])
    reuse_ids = {case["id"] for case in reuse["cases"] if case["eligible"]}
    control_ids = {case["id"] for case in controls["cases"]}
    assert len(control_ids) == len(controls["cases"]) and not (reuse_ids & control_ids) and reuse_ids | control_ids == selected
    assert reuse["eligible_count"] == len(reuse_ids)
    historical = read(HISTORICAL / "manifest.json")
    historical_ids = {case["id"] for case in historical["cases"]}
    assert reuse_ids <= historical_ids and all(case["method"] == "conditioned" for case in historical["cases"])
    checked_reused_files = 0
    for case in reuse["cases"]:
        if case["eligible"]:
            assert case["method"] == "conditioned" and not case["reasons"]
            for kind, source in case["sources"].items():
                path = Path(source["path"])
                assert path.is_relative_to(HISTORICAL) and digest(path) == source["sha256"]
                if kind != "stage_geometry":
                    assert read(path)["method"] == "conditioned"
                checked_reused_files += 1
    fit = {row["sample_id"]: row for row in csv_rows(EVAL / "post/fit_only/samples.csv") if row["method"] == "point_feedback"}
    regressions = sorted(sid for sid in ids if fit[sid]["native_valid_solid"] == "True" and new[sid]["native_valid_solid"] == "False")
    assert regressions == manifest["optimization_regression_ids"]
    contract = REPO / "checkpoints/joint_boundary/validation/test_contract_audit"
    contract_summary = read(contract / "summary.json")
    original = set(Path(contract_summary["split_list"]).read_text().split())
    assert digest(Path(contract_summary["split_list"])) == contract_summary["split_sha256"]
    excluded = {row["model_id"] for row in csv_rows(contract / "rejected.csv")}
    assert len(original) == 2424 and original - ids == excluded and len(excluded) == 7
    result = {"accepted": len(ids), "original": len(original), "contract_excluded_ids": sorted(excluded),
              "contract_exclusion_source": str(contract / "rejected.csv"), "contract_exclusion_sha256": digest(contract / "rejected.csv"),
              "cohorts": {name: len(values) for name, values in groups.items()}, "candidate_failure_denominator": len(failed),
              "selected_paired_ids": len(selected), "excluded_both_valid_ids": len(groups["both_valid"]),
              "baseline_reuse": len(reuse_ids), "baseline_new_controls": len(control_ids),
              "historical_reuse_files_checked": checked_reused_files, "reuse_partition_verified": True,
              "optimization_regression_ids": regressions}
    baseline_cases = {case["id"]: case for case in baseline["cases"]}
    result["raw_topology_profile"] = {}
    for label, selected_cases in (
        ("all_candidate_failures", [case for case in manifest["cases"] if not case["post_record"]["native_valid_solid"]]),
        ("regressed", [case for case in manifest["cases"] if case["cohort"] == "new_failure"]),
        ("persistent", [case for case in manifest["cases"] if case["cohort"] == "both_failed"]),
        ("rescued", [case for case in manifest["cases"] if case["cohort"] == "rescued"]),
    ):
        result["raw_topology_profile"][label] = {
            "denominator": len(selected_cases),
            "candidate_exact": sum(case["raw_record"]["topology_fp"] == case["raw_record"]["topology_fn"] == 0 for case in selected_cases),
            "baseline_exact": sum(baseline_cases[case["id"]]["raw_record"]["topology_fp"] == baseline_cases[case["id"]]["raw_record"]["topology_fn"] == 0 for case in selected_cases),
            "candidate_extra": sum(case["raw_record"]["topology_fp"] > 0 for case in selected_cases),
            "candidate_missing": sum(case["raw_record"]["topology_fn"] > 0 for case in selected_cases),
            "candidate_extra_and_missing": sum(case["raw_record"]["topology_fp"] > 0 and case["raw_record"]["topology_fn"] > 0 for case in selected_cases),
        }
    return result, manifest, baseline, reuse_ids


def protected_sources():
    sources = read(CENSUS / "source_asset_audit.json")["source_assets"]
    checked = 0
    for item in sources:
        path = Path(item["path"])
        assert path.is_file() == item["exists"], str(path)
        assert (digest(path) if path.is_file() else None) == item["sha256"], str(path)
        checked += 1
    result = {"count": checked, "changed": 0, "source_audit_sha256": digest(CENSUS / "source_asset_audit.json")}
    report_path = CENSUS / "source_preservation.json"
    if report_path.is_file():
        report = read(report_path)
        assert report["asset_count"] == checked and report["mismatch_count"] == 0 and report["changed_paths"] == []
        assert report["source_audit_sha256"] == result["source_audit_sha256"]
        result["report_sha256"] = digest(report_path)
        result["independent_rehash_matches_report"] = True
    return result


def verified(trace):
    return trace.get("status") == "ok" and not trace.get("instrumentation_errors") and trace.get("comparison", {}).get("same_recorded_outcome") is True and trace.get("step_data_comparison", {}).get("equal") is True


def step_data_digest(path):
    data = Path(path).read_bytes()
    start = data.index(b"DATA;") + len(b"DATA;")
    stop = data.index(b"ENDSEC;", start)
    return hashlib.sha256(data[start:stop]).hexdigest()


def terminal(trace, valid):
    if valid:
        return "native_valid"
    if not verified(trace):
        return "unknown"
    checks = [item.get("legacy_saved_step_check") for item in trace.get("save_attempts", [])]
    if True in checks:
        return "native_readback_invalid" if trace.get("replay", {}).get("native_valid_solid") is False else "unresolved"
    if False in checks:
        return "saved_step_rejected"
    if trace.get("trim_gate", {}).get("solid_attempt_allowed") is False:
        return "trim_gate"
    attempts = trace.get("solid_attempts", [])
    return "solid_construction" if attempts and all(item.get("result", {}).get("present") is False for item in attempts) else "unresolved"


def owner_flag(trace, stage, threshold):
    if not verified(trace):
        return None
    fit = trace.get("fit_residuals", {})
    count = trace.get("fitting", {}).get("num_curves")
    rows = [(edge.get("edge_index"), owner) for edge in fit.get("curve_owner", []) for owner in edge.get(stage, [])]
    identities = [(edge, owner.get("face_index")) for edge, owner in rows]
    if len(set(identities)) != len(identities) or any(type(e) is not int or type(f) is not int for e, f in identities):
        return None
    finite = lambda value: isinstance(value, (int, float)) and math.isfinite(value)
    if any(finite(owner.get("max")) and owner["max"] > threshold for _, owner in rows):
        return True
    complete = bool(rows) and type(count) is int and count >= 0 and len(rows) == 2 * count and fit.get("status") in ("ok", "partial")
    complete = complete and all(owner.get("status") == "ok" and owner.get("count", 0) > 0 and owner["count"] == owner.get("finite_count") and finite(owner.get("max")) and not owner.get("errors") for _, owner in rows)
    return False if complete else None


def trim_flags(trace):
    names = ("never_wire", "never_closed_admitted", "closed_admitted_still_failed", "strict_closed_candidate_failed")
    unknown = {name: None for name in names}
    if not verified(trace):
        return unknown
    runs = [run for run in trace.get("trim_runs", []) if run.get("context") == "main"]
    if len(runs) != 1 or not isinstance(runs[0].get("failed_face_indices"), list):
        return unknown
    run = runs[0]
    categories, strict_values = [], []
    for face in set(run["failed_face_indices"]):
        attempts = [attempt for attempt in run.get("attempts", []) if attempt.get("face_index") == face]
        complete, admitted, strict = bool(attempts), False, False
        for attempt in attempts:
            if attempt.get("wire_returned") is False:
                continue
            detail = attempt.get("trim_detail", {})
            wires = detail.get("wire_events")
            if not isinstance(wires, list) or detail.get("outcome", "unknown") == "unknown":
                complete = False
                continue
            ok = [wire.get("added") is True and wire.get("repaired_closed") is True for wire in wires]
            admitted |= any(ok)
            strict |= bool(ok) and all(ok) and attempt.get("trim_returned") is False
        category = "closed_admitted_still_failed" if admitted else "unknown" if not complete else "never_wire" if all(a.get("wire_returned") is False for a in attempts) else "never_closed_admitted"
        categories.append(category)
        strict_values.append(True if strict else False if complete else None)
    result = {name: True if name in categories else None if "unknown" in categories else False for name in names[:3]}
    result[names[3]] = True if True in strict_values else None if None in strict_values else False
    return result


def all_observation_flags(trace, upstream, saved, raw, trace_sha, threshold):
    """Rebuild all 28 predicates from trace/measurement primitives, not report flags.

    Geometry is not recomputed here. Distances/counts are the frozen measured
    evidence; this independently checks thresholding, stage provenance, and
    three-state completeness rules used in the report.
    """
    finite = lambda value: type(value) in (int, float) and math.isfinite(value)
    result = {key: None for key in OBSERVATIONS}
    stage_ok = verified(trace) and upstream.get("sources", {}).get("trace", {}).get("sha256") == trace_sha
    structure = upstream.get("structure", {})
    original = structure.get("raw", {})
    if original.get("status") == "ok":
        result.update(
            raw_extra_gt_pair=bool(original["fp_pairs"]),
            raw_missing_gt_pair=bool(original["fn_pairs"]),
            raw_duplicate_directed_pair=original["exact_directed_duplicate_count"] > 0,
            raw_same_face_pair=bool(original["same_face_row_indices"]),
        )
    selected = structure.get("selected", {})
    if stage_ok and selected.get("status") == "ok":
        result["selection_removed_gt_pair"] = bool(selected["removed_gt_pairs"])
    if finite(raw.get("pred_face_count")) and finite(raw.get("gt_face_count")):
        result["face_count_mismatch"] = raw["pred_face_count"] != raw["gt_face_count"]

    reciprocal = upstream.get("reciprocal", {})
    if reciprocal.get("status") == "ok":
        count = reciprocal["thresholds"][threshold]
        result["reciprocal_excess_disagreement"] = True if count["excess_over_gt_pairs"] else None if count["gt_unknown_pairs"] else False
    stage = upstream.get("stages", {}).get("raw", {})
    for module, flag in (("cross_pair_near_curves", "raw_new_cross_pair_curve_nearness"),
                         ("face_collapse_proxy", "raw_new_face_grid_nearness")):
        values = stage.get(module, {})
        if values.get("status") == "ok":
            count = values["thresholds"][threshold]
            result[flag] = True if count["new_nearness_count"] else None if count["gt_baseline_unknown_count"] else False
    if stage_ok:
        for name in ("selected", "optimized"):
            values = upstream.get("stages", {}).get(name, {}).get("endpoint_proxy", {})
            if values.get("status") == "ok":
                count = values.get("thresholds", {}).get(threshold, {})
                uncertain = values.get("unknown_closure_endpoint_count", 0)
                missing_faces = values.get("faces_without_incident_edges", [])
                result[name + "_endpoint_gap"] = True if count.get("endpoints_with_residual", 0) else None if uncertain or missing_faces else False
                result[name + "_new_open_edge_endpoint_collapse"] = True if count["new_open_endpoint_collapse_edges"] else None if uncertain else False
        changes = upstream.get("stage_deltas", {})
        for flag, value in (("optimization_face_away", changes.get("face_xyz_mae_change")),
                            ("optimization_edge_away", changes.get("edge_mean_distance_change", {}).get("mean"))):
            result[flag] = value > 1e-12 if finite(value) else None

    saved_count = saved.get("free_edge_proximity", {}).get("thresholds", {}).get(threshold, {}).get("pair_count")
    saved_complete = saved.get("status") == "ok" and not saved.get("sampling_errors")
    result["saved_near_free_edges"] = True if finite(saved_count) and saved_count > 0 else False if saved_complete and saved_count == 0 else None
    result.update(trim_flags(trace))
    result["owner_independent"] = owner_flag(trace, "independent_curve", float(threshold))
    result["owner_after"] = owner_flag(trace, "after_periodic_replacement", float(threshold))
    if not verified(trace):
        return result
    result["sewing_free_edges"] = any(item.get("sewing", {}).get("free_edges", 0) > 0 for item in trace.get("solid_attempts", []))
    fit = trace.get("fit_residuals", {})
    fitting = trace.get("fitting", {})

    def complete_fit(rows, expected):
        return (fit.get("status") in {"ok", "partial"} and expected is not None and len(rows) == expected and bool(rows)
                and all(item.get("status") == "ok" and item.get("count", 0) > 0
                        and item.get("count") == item.get("finite_count") and finite(item.get("max"))
                        and not item.get("errors") for item in rows))

    for name, flag, expected in (("faces", "fit_face_residual", len(fitting.get("faces", []))),
                                 ("curves", "fit_curve_residual", fitting.get("num_curves"))):
        rows = fit.get(name, [])
        result[flag] = True if any(finite(item.get("max")) and item["max"] > float(threshold) for item in rows) else False if complete_fit(rows, expected) else None

    runs = [run for run in trace.get("trim_runs", []) if run.get("context") == "main"]
    if len(runs) != 1 or not isinstance(runs[0].get("failed_face_indices"), list):
        return result
    run = runs[0]
    failed = set(run["failed_face_indices"])
    count, connectivity = run.get("num_faces"), run.get("connectivity")
    if (type(count) is int and isinstance(connectivity, list)
            and all(type(value) is int and 0 <= value < count for value in failed)
            and all(isinstance(row, list) and len(row) == 3 and all(type(value) is int and 0 <= value < count for value in row[1:]) for row in connectivity)):
        no_edge = failed - {value for row in connectivity for value in row[1:]}
        if not no_edge.intersection(attempt.get("face_index") for attempt in run.get("attempts", [])):
            result["face_without_boundary"] = bool(no_edge)
    discarded, uncertain = False, False
    for face_id in set(range(count)) - failed:
        attempts = [attempt for attempt in run.get("attempts", []) if attempt.get("face_index") == face_id and attempt.get("trim_returned") is True]
        if len(attempts) != 1 or not isinstance(attempts[0].get("trim_detail", {}).get("wire_events"), list):
            uncertain = True
        else:
            discarded |= any(wire.get("skip_reason") for wire in attempts[0]["trim_detail"]["wire_events"])
    result["final_success_discarded_wire"] = True if discarded else None if uncertain else False
    closed_faces = {face_id for face_id in failed if any(
        wire.get("added") is True and wire.get("repaired_closed") is True
        for attempt in run.get("attempts", []) if attempt.get("face_index") == face_id and attempt.get("wire_returned") is not False
        and attempt.get("trim_detail", {}).get("outcome", "unknown") != "unknown"
        for wire in attempt.get("trim_detail", {}).get("wire_events", [])
    )}
    if not closed_faces:
        result["closed_failed_owner_gap"] = None if result["closed_admitted_still_failed"] is None else False
    else:
        owners = [owner for edge in fit.get("curve_owner", []) for owner in edge.get("after_periodic_replacement", [])]
        local = [owner for owner in owners if owner.get("face_index") in closed_faces]
        expected = 2 * fitting["num_curves"] if finite(fitting.get("num_curves")) else None
        result["closed_failed_owner_gap"] = True if any(finite(owner.get("max")) and owner["max"] > float(threshold) for owner in local) else False if complete_fit(owners, expected) and local else None
    return result


def stratified_failure_mechanisms(pairs):
    """Post-hoc description only; preserve the original evaluation denominators."""
    failed = [pair for pair in pairs if not pair["candidate"]["original_native_valid"]]
    exact = lambda row: row["raw_metrics"]["topology_fp"] == row["raw_metrics"]["topology_fn"] == 0
    strata = {
        "all_failed_adjacency_exact": [pair["candidate"] for pair in failed if exact(pair["candidate"])],
        "all_failed_adjacency_inexact": [pair["candidate"] for pair in failed if not exact(pair["candidate"])],
        "regressed_adjacency_exact": [pair["candidate"] for pair in failed if pair["cohort"] == "regressed" and exact(pair["candidate"])],
    }
    assert len(strata["all_failed_adjacency_exact"]) + len(strata["all_failed_adjacency_inexact"]) == len(failed)
    assert {row["id"] for row in strata["regressed_adjacency_exact"]} <= {row["id"] for row in strata["all_failed_adjacency_exact"]}
    result = {
        "interpretation": "Post-hoc descriptive stratification, not a preregistered causal test; full evaluation metrics and the dynamically derived main failure denominator are unchanged. Exact adjacency checks only face-pair adjacency, not complete B-Rep topology or boundary geometry.",
        "main_failure_denominator": len(failed),
        "threshold": 0.006,
        "strata": {},
    }
    for label, rows in strata.items():
        observations = {}
        for key in ("owner_after", "strict_closed_candidate_failed", "never_closed_admitted", "final_success_discarded_wire"):
            counts = {"positive": 0, "negative": 0, "unknown": 0}
            for row in rows:
                value = row["flags"]["0.006"][key]
                counts["positive" if value is True else "negative" if value is False else "unknown"] += 1
            assert sum(counts.values()) == len(rows)
            observations[key] = counts
        result["strata"][label] = {
            "denominator": len(rows),
            "ids": sorted(row["id"] for row in rows),
            "terminal_counts": dict(sorted(Counter(row["terminal_stage"] for row in rows).items())),
            "observations": observations,
        }
    return result


def optimization_counterfactual_review(manifest, pairs):
    document = read(CENSUS / "optimization_regressions.json")
    expected_ids = manifest["optimization_regression_ids"]
    assert document["denominator"] == len(expected_ids)
    assert [row["id"] for row in document["cases"]] == expected_ids
    cases = {case["id"]: case for case in manifest["cases"]}
    candidates = {pair["id"]: pair["candidate"] for pair in pairs}
    fit_root = CENSUS / "fit_only_counterfactual"
    evidence = []
    predicate_comparisons = 0
    for row in document["cases"]:
        sid = row["id"]
        case = cases[sid]
        assert case["fit_record"]["native_valid_solid"] is True
        assert case["post_record"]["native_valid_solid"] is False
        assert row["optimized"] == candidates[sid]
        fit = row["fit_only"]
        trace = read(fit_root / "traces" / (sid + ".json"))
        assert fit["method"] == trace["method"] == "point_feedback"
        assert trace["post_mode"] == "fit_only"
        assert trace["original"] == case["fit_record"]
        assert fit["original_native_valid"] is True
        assert fit["trace_verified"] is bool(verified(trace))
        assert fit["terminal_stage"] == terminal(trace, True)
        assert set(fit["flags"]) == THRESHOLDS
        assert all(set(flags) == OBSERVATIONS for flags in fit["flags"].values())
        assert all(value is None or type(value) is bool for flags in fit["flags"].values() for value in flags.values())
        for kind in ("trace", "upstream"):
            directory = "traces" if kind == "trace" else "upstream"
            path = fit_root / directory / (sid + ".json")
            assert Path(fit["source_files"][kind]["path"]) == path
            assert fit["source_files"][kind]["sha256"] == digest(path)
        if verified(trace):
            original_hash = step_data_digest(case["paths"]["fit_step"])
            replay_hash = step_data_digest(fit_root / "replay" / sid / "recon_brep.step")
            assert original_hash == replay_hash == trace["step_data_comparison"]["original"] == trace["step_data_comparison"]["replay"]
        upstream = read(fit_root / "upstream" / (sid + ".json"))
        saved_path = fit_root / "saved_step" / (sid + ".json")
        saved = read(saved_path) if saved_path.is_file() else {}
        for threshold, flags in fit["flags"].items():
            calculated = all_observation_flags(trace, upstream, saved, case["raw_record"], digest(fit_root / "traces" / (sid + ".json")), threshold)
            assert calculated.keys() == flags.keys()
            for flag, value in calculated.items():
                assert flags[flag] is value, (sid, "fit_only", flag, threshold, flags[flag], value)
                predicate_comparisons += 1
        equality = {key: None for key in ("selected_faces", "selected_edges", "selected_connectivity")}
        if fit["trace_verified"] and candidates[sid]["trace_verified"]:
            paths = [location / "replay" / sid / "trace_geometry.npz" for location in (fit_root, CENSUS)]
            if all(path.is_file() for path in paths):
                with np.load(paths[0], allow_pickle=False) as first, np.load(paths[1], allow_pickle=False) as second:
                    equality = {key: bool(np.array_equal(first[key], second[key])) if key in first and key in second else None for key in equality}
        assert row["same_selected_arrays"] == equality
        same_input = case["fit_record"]["raw_sha256"] == case["post_record"]["raw_sha256"]
        assert row["same_raw_input_sha256"] is same_input
        evidence.append({"id": sid, "fit_trace_verified": fit["trace_verified"],
                         "optimized_trace_verified": candidates[sid]["trace_verified"],
                         "same_raw_input_sha256": same_input, "same_selected_arrays": equality})
    return {"denominator": len(expected_ids), "ids": expected_ids, "all_records_verified_against_source": True,
            "fit_verified_trace_count": sum(row["fit_trace_verified"] for row in evidence),
            "all_observation_predicate_comparisons": predicate_comparisons,
            "cases": evidence, "report_sha256": digest(CENSUS / "optimization_regressions.json")}


def final_results(manifest, baseline, reuse_ids):
    pairs = read(CENSUS / "cases.json")["pairs"]
    summary = read(CENSUS / "summary.json")
    compact = read(CENSUS / "compact_summary.json")
    assert compact["full_summary_sha256"] == digest(CENSUS / "summary.json")
    assert set(summary["cohorts"]) == {"persistent", "regressed", "rescued"}
    assert set(compact["definitions"]) == OBSERVATIONS
    assert compact["population"] == summary["population"] == manifest["population"]
    assert compact["columns"]["observations"] == ["positive", "negative", "unknown"]
    assert compact["columns"]["transitions"] == ["present_to_absent", "absent_to_present", "both_present", "both_absent", "unknown"]
    cases = {"candidate": {case["id"]: case for case in manifest["cases"]}, "baseline": {case["id"]: case for case in baseline["cases"]}}
    assert len(pairs) == len(cases["candidate"]) and {pair["id"] for pair in pairs} == set(cases["candidate"])
    trace_count, predicates = 0, 0
    for pair in pairs:
        sid = pair["id"]
        for role in ("candidate", "baseline"):
            location = CENSUS if role == "candidate" else HISTORICAL if sid in reuse_ids else CENSUS / "baseline_controls"
            path = location / "traces" / (sid + ".json")
            trace = read(path)
            row, case = pair[role], cases[role][sid]
            assert trace.get("method") == row["method"] == case["method"]
            assert row["original_native_valid"] is case["post_record"]["native_valid_solid"]
            for key in ("sample_id", "method", "mode", "status", "post_status", "raw_sha256", "native_valid_solid", "step_exists", "step_readable"):
                assert trace["original"].get(key) == case["post_record"].get(key), (sid, role, key)
            assert row["trace_verified"] == bool(verified(trace))
            assert row["terminal_stage"] == terminal(trace, case["post_record"]["native_valid_solid"])
            assert set(row["flags"]) == THRESHOLDS
            assert all(set(flags) == OBSERVATIONS for flags in row["flags"].values())
            assert all(value is None or type(value) is bool for flags in row["flags"].values() for value in flags.values())
            # Historical reuse preserves the original path; current baseline
            # manifests point at a read-only symlink to the identical NPZ.
            assert {key: value for key, value in row["raw_metrics"].items() if key != "prediction_file"} == {
                key: value for key, value in case["raw_record"].items() if key != "prediction_file"
            }
            assert row["source_files"]["trace"]["sha256"] == digest(path)
            if verified(trace):
                original_hash = step_data_digest(case["paths"]["post_step"])
                replay_hash = step_data_digest(location / "replay" / sid / "recon_brep.step")
                assert original_hash == replay_hash == trace["step_data_comparison"]["original"] == trace["step_data_comparison"]["replay"]
            diagnostic_data = {}
            for diagnostic, directory in (("upstream", "upstream"), ("saved_step", "saved_step")):
                assert row["source_files"][diagnostic]["sha256"] == digest(location / directory / (sid + ".json"))
                diagnostic_data[diagnostic] = read(location / directory / (sid + ".json"))
            for threshold, flags in row["flags"].items():
                calculated = all_observation_flags(trace, diagnostic_data["upstream"], diagnostic_data["saved_step"],
                                                   case["raw_record"], digest(path), threshold)
                assert calculated.keys() == flags.keys()
                for flag, value in calculated.items():
                    assert flags[flag] is value, (sid, role, flag, threshold, flags[flag], value)
                    predicates += 1
            trace_count += 1
    group_lists = [("main", [p["candidate"] for p in pairs if not p["candidate"]["original_native_valid"]], summary["main"]),
                   ("baseline_failures", [p["baseline"] for p in pairs if not p["baseline"]["original_native_valid"]], summary["baseline_failures"])]
    for cohort, values in summary["cohorts"].items():
        subset = [pair for pair in pairs if pair["cohort"] == cohort]
        assert len(subset) == values["count"]
        for role in ("candidate", "baseline"):
            group_lists.append((cohort + ":" + role, [p[role] for p in subset], values[role]))
        for threshold, observations in values["transitions"].items():
            assert set(values["transitions"]) == THRESHOLDS and set(observations) == OBSERVATIONS
            for key, recorded in observations.items():
                actual = {name: [] for name in recorded["counts"]}
                for pair in subset:
                    a, b = (pair[role]["flags"][threshold][key] for role in ("baseline", "candidate"))
                    name = "unknown" if type(a) is not bool or type(b) is not bool else "both_present" if a and b else "both_absent" if not a and not b else "present_to_absent" if a else "absent_to_present"
                    actual[name].append(pair["id"])
                assert actual == recorded["ids"] and {name: len(ids) for name, ids in actual.items()} == recorded["counts"]
                assert compact["cohorts"][cohort]["transitions"][threshold][key] == [len(actual[name]) for name in compact["columns"]["transitions"]]
    groups_review = {}
    for label, records, group in group_lists:
        compact_group = compact[label] if ":" not in label else compact["cohorts"][label.split(":")[0]][label.split(":")[1]]
        assert len(records) == group["denominator"]
        assert compact_group["denominator"] == len(records)
        assert set(group["observations"]) == THRESHOLDS
        assert all(set(observations) == OBSERVATIONS for observations in group["observations"].values())
        assert sum(row["trace_verified"] for row in records) == group["verified_traces"]
        assert dict(Counter(row["raw_topology"] for row in records)) == group["raw_topology_counts"]
        for name, recorded in group["terminal"].items():
            ids = [row["id"] for row in records if row["terminal_stage"] == name]
            assert ids == recorded["ids"] and len(ids) == recorded["count"]
            assert compact_group["terminal"][name] == len(ids)
        for threshold, observations in group["observations"].items():
            for key, expected in observations.items():
                actual = {"positive": [], "negative": [], "unknown": []}
                for row in records:
                    value = row["flags"][threshold][key]
                    actual["positive" if value is True else "negative" if value is False else "unknown"].append(row["id"])
                for category, ids in actual.items():
                    assert len(ids) == expected[category] and sorted(ids) == sorted(expected["ids"][category])
                assert compact_group["observations"][threshold][key] == [len(actual[name]) for name in compact["columns"]["observations"]]
        context = group["failed_face_context"]
        faces = [face for row in records for face in row["failed_face_context"]["faces"] or []]
        assert len(faces) == context["counted_final_failed_faces"]
        for code, expected in context["codes"].items():
            for category, field in (("face_count", "codes"), ("direct_face_count", "direct_codes"), ("context_face_count", "context_codes")):
                assert expected[category] == sum(code in face[field] for face in faces)
        groups_review[label] = {"denominator": len(records), "verified_traces": group["verified_traces"],
                                "terminal": {key: item["count"] for key, item in group["terminal"].items()},
                                "raw_topology_counts": group["raw_topology_counts"],
                                "key_observations_at_0.006": {key: {name: values[name] for name in ("positive", "negative", "unknown")} for key, values in group["observations"]["0.006"].items()}}
    return {"paired_cases": len(pairs), "direct_trace_records_checked": trace_count,
            "primary_display_tally_grid": {"groups": 4, "thresholds": len(THRESHOLDS), "observations": len(OBSERVATIONS),
                                           "combinations_verified": 4 * len(THRESHOLDS) * len(OBSERVATIONS)},
            "direct_primary_trace_predicate_comparisons": predicates, "all_group_threshold_tallies_and_pair_transitions_verified": True,
            "compact_report_tallies_match_independent_counts": True,
            "failed_face_context_count_aggregation_verified": True, "groups": groups_review,
            "stratified_failure_mechanisms": stratified_failure_mechanisms(pairs),
            "optimization_counterfactual": optimization_counterfactual_review(manifest, pairs),
            "cases_sha256": digest(CENSUS / "cases.json"), "summary_sha256": digest(CENSUS / "summary.json"),
            "compact_summary_sha256": digest(CENSUS / "compact_summary.json")}


def main() -> None:
    global ROOT, OLD, LATENT_SOURCE, SPLIT, IDS, EXPECTED_SHA, SELECTED, SELECTED_SHA
    global SELECTED_RECORD, BASE_COMMIT, CENSUS, HISTORICAL, EVAL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--latent-source", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--census-root", type=Path)
    parser.add_argument("--historical-census", type=Path)
    args = parser.parse_args()
    ROOT, OLD, LATENT_SOURCE = (path.resolve() for path in (args.root, args.reference_root, args.latent_source))
    EVAL, BASE_COMMIT = ROOT, args.base_commit
    audit = read(ROOT / "training_audit.json")
    SELECTED_RECORD = audit["checkpoint_recommendation"]
    SELECTED, SELECTED_SHA = Path(SELECTED_RECORD["path"]), SELECTED_RECORD["sha256"]
    SPLIT = Path(audit["splits"]["test"]["path"])
    IDS, EXPECTED_SHA = SPLIT.read_text().split(), audit["splits"]["test"]["sha256"]
    assert len(IDS) == len(set(IDS)) == 2417 and digest(SPLIT) == EXPECTED_SHA
    assert digest(SELECTED) == SELECTED_SHA
    assert SELECTED_RECORD["epoch"] == min(audit["validation_history"], key=lambda r: r["val/loss"])["epoch"]
    assert audit["state_compatibility"]["frozen_keys_bitwise_equal_selected_and_last"] is True
    review = {"created_utc": datetime.now(timezone.utc).isoformat(), "reviewer": "test_point_feedback",
              "created_display_time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "display_timezone": "Asia/Shanghai", "checkpoint_sha256": SELECTED_SHA,
              "review_builder_sha256": digest(Path(__file__).resolve()),
              "independent_method": "No pipeline helpers imported. Statistics reconstructed from CSV; identity, latent arrays and all record hashes checked; all 28 census predicates independently reconstructed from trace/upstream/saved-step measurement primitives, including unknown and provenance gating. Geometry measurements themselves are not rerun.",
              "code": source_review(), "inputs": {}, "raw": {}, "post": {}, "pending": []}
    for mode in ("posterior_mean", "archived_latents"):
        review["inputs"][mode] = input_review(mode)
        if (ROOT / "raw_metrics" / mode / "summary.json").is_file():
            review["raw"][mode] = raw_review(mode)
        else:
            review["pending"].append("raw:" + mode)
    for mode in ("fit_only", "optimized"):
        if not args.raw_only and (ROOT / "post" / mode / "summary.json").is_file():
            review["post"][mode] = post_review(mode)
        else:
            review["pending"].append("post:" + mode)
    if (ROOT / "diagnostics/outlier_pair_probe.json").is_file():
        review["distributions"] = distribution_review()
    else:
        review["pending"].append("diagnostics:distributions")
    if (ROOT / "historical_reference.json").is_file():
        review["historical_reference"] = historical_review()
    else:
        review["pending"].append("historical_reference")
    if args.census_root:
        if not args.historical_census:
            parser.error("--historical-census is required with --census-root")
        CENSUS, HISTORICAL = args.census_root.resolve(), args.historical_census.resolve()
        identities, manifest, baseline, reused = frozen_identities()
        review["census"] = {"frozen_identities": identities, "source_preservation": protected_sources()}
        if (CENSUS / "cases.json").is_file() and (CENSUS / "summary.json").is_file():
            review["census"]["diagnostics"] = final_results(manifest, baseline, reused)
        else:
            review["pending"].append("census:diagnostics")
    else:
        review["pending"].append("census:not_requested")
    review["status"] = "pending" if review["pending"] else "passed"
    (ROOT / "independent_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": review["status"], "pending": review["pending"], "post": review["post"],
                      "census_checked": "diagnostics" in review.get("census", {})}, indent=2))


if __name__ == "__main__":
    main()
