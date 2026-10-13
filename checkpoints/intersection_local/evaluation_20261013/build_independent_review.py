"""Independently check frozen evaluation records without importing pipeline code."""

from __future__ import annotations

import ast
import argparse
from collections import Counter
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
REPO = ROOT.parents[2]
OLD = REPO / "checkpoints/intersection_refinement/evaluation_20261011"
LATENT_SOURCE = REPO / "checkpoints/joint_boundary/comparison_20261011"
SPLIT = REPO / "checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt"
IDS = [line.strip() for line in SPLIT.read_text().splitlines() if line.strip()]
EXPECTED_SHA = "2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3"
SELECTED = REPO / "checkpoints/intersection_local/full_capacity/20261012T141134.581695Z/weights/epoch=0042-step=00015609-val_loss=0.098798.ckpt"
SELECTED_SHA = "c16d9ebe829659f3f4b8eb631e115fb10d03a28e9319bdb6bb5968b8d4bdb8f3"
TERMINAL = {"ok", "missing_raw", "invalid_raw", "post_error", "no_step", "step_read_error",
            "geometry_error", "timeout", "crash", "interrupted"}


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
    allowed = {
        "tools/intersection_evaluate_inference.py": {"load_intersection", "build_parser", "main"},
        "tools/vae_compare_metrics.py": {"comparison_methods", "evaluate_sample", "summarize", "main"},
        "tools/vae_compare_post.py": {"comparison_methods", "summarize", "write_summary", "load_reference_rows", "build_parser", "main"},
    }
    result = {}
    for filename, names in allowed.items():
        before = subprocess.check_output(["git", "show", "HEAD:" + filename], text=True, cwd=REPO)
        after = (REPO / filename).read_text()
        functions = lambda source: {node.name: ast.dump(node, include_attributes=False)
                                   for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        previous, current = functions(before), functions(after)
        changed = {name for name in set(previous) | set(current) if previous.get(name) != current.get(name)}
        assert changed <= names, (filename, changed - names)
        result[filename] = {"sha256": digest(REPO / filename), "changed_functions_vs_head": sorted(changed),
                            "unchanged_function_count": len(set(previous) & set(current)) - len(changed)}
    unchanged_post = not subprocess.check_output(["git", "diff", "HEAD", "--", "src/brepnet/post", "shared"], cwd=REPO)
    assert unchanged_post
    previous_audit = read(ROOT.parent / "evaluation_20261012/independent_review.json")
    assert all(item["sha256"] == previous_audit["code"]["files"][name]["sha256"] for name, item in result.items())
    return {"files": result, "production_post_and_shared_unchanged_vs_head": unchanged_post,
            "inference_raw_metrics_post_runner_exactly_match_previous_evaluation": True,
            "manual_review": "Changed functions implement method identities, output/reference paths, provenance and aggregation. Raw distance/topology formulas and post worker/validity/sampling/timeout logic are unchanged."}


def input_review(mode: str) -> dict:
    metadata = ROOT / "metadata/inference" / mode
    protocol = read(metadata / "protocol.json")
    assert Path(protocol["checkpoint"]) == SELECTED and protocol["checkpoint_sha256"] == SELECTED_SHA
    checkpoint_audit = protocol["model_audit"]
    assert checkpoint_audit["strict_load"] is True
    assert checkpoint_audit["intersection"]["epoch"] == 42 and checkpoint_audit["intersection"]["global_step"] == 15609
    cfg = checkpoint_audit["checkpoint_model_config"]["intersection"]
    assert all(cfg[key] == value for key, value in {
        "block_type": "transformer_decoder", "hidden_dim": 512, "num_layers": 8,
        "num_heads": 8, "ffn_dim": 2048, "dropout": 0.1, "activation": "relu",
    }.items())
    assert protocol["method"] == "local" and protocol["intended_count"] == 2417
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
    means = [read(path) for path in metadata.glob("shard-???-mean-audit-*.json")]
    assert len(means) == 8 and all(item["passed"] for item in means)
    input_hashes = {}
    for model_id in IDS:
        local_path = ROOT / "raw_predictions" / mode / "local" / model_id / "data.npz"
        baseline_path = ROOT / "raw_predictions" / mode / "intersection" / model_id / "data.npz"
        source_path = LATENT_SOURCE / "raw_predictions" / mode / "legacy" / model_id / "data.npz"
        with np.load(local_path, allow_pickle=False) as local, np.load(baseline_path, allow_pickle=False) as baseline, np.load(source_path, allow_pickle=False) as source:
            values = local["face_latents"]
            assert values.dtype == np.float32 and np.isfinite(values).all()
            assert np.array_equal(values, baseline["face_latents"]) and np.array_equal(values, source["face_latents"])
            h = hashlib.sha256(str(values.shape).encode())
            h.update(np.ascontiguousarray(values).tobytes())
            expected = h.hexdigest()
            assert expected == str(local["input_latent_sha256"]) == str(baseline["input_latent_sha256"])
            assert str(local["reference_raw_sha256"]) == digest(source_path)
            assert str(local["protocol_sha256"]) == fingerprint(protocol)
            assert str(local["representation"]) == "legacy_half_edges"
            probability = local["all_candidate_prob"]
            pairs = local["all_candidate_pairs"]
            assert np.array_equal(local["pred_edge_face_connectivity"][:, 1:], pairs[probability > 0.5])
            count = len(values)
            assert len(pairs) == count * (count - 1)
            assert not np.any(pairs[:, 0] == pairs[:, 1])
            input_hashes[model_id] = expected
    return {"intended": len(IDS), "local_prediction_count": len(rows), "status_counts": dict(Counter(row["status"] for row in rows)),
            "selected_full_capacity_checkpoint_verified": str(SELECTED),
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
    maps = {method: {r["model_id"]: r for r in rows if r["method"] == method} for method in ("intersection", "local")}
    assert all(set(group) == set(IDS) for group in maps.values())
    old_map = {r["model_id"]: r for r in old_rows if r["method"] == "intersection"}
    assert summary["methods"]["intersection"] == old_summary["methods"]["intersection"]
    for model_id, row in maps["intersection"].items():
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
        matched = [(maps["intersection"][model_id].get(metric), maps["local"][model_id].get(metric)) for model_id in IDS
                   if maps["intersection"][model_id].get("shared_input_status") == "verified"
                   and maps["intersection"][model_id].get(metric) not in (None, "")
                   and maps["local"][model_id].get(metric) not in (None, "")]
        differences = [float(b) - float(a) for a, b in matched]
        assert paired["common_count"] == len(matched)
        verify_stats(differences, paired["local_minus_intersection"], mode + ":paired:" + metric)
    return {"row_count": len(rows), "stats_blocks_independently_recalculated": checked_stats,
            "objects_with_exact_adjacency": {
                method: sum(row.get("topology_f1") not in (None, "") and float(row["topology_f1"]) == 1.0 for row in group.values())
                if mode == "posterior_mean" else None for method, group in maps.items()
            },
            "baseline_metrics_exactly_equal_previous": True,
            "baseline_csv_exact_except_prediction_path": True, "paired_differences_verified": True,
            "summary_sha256": digest(directory / "summary.json"), "csv_sha256": digest(directory / "per_object.csv")}


def post_review(mode: str) -> dict:
    directory = ROOT / "post" / mode
    metadata = ROOT / "metadata/post" / mode
    summary = read(directory / "summary.json")
    old_summary = read(OLD / "post" / mode / "summary.json")
    protocol = read(metadata / "protocol.json")
    reference = protocol["reference_baseline"]
    assert summary["intended_count_per_method"] == 2417 and summary["original_test_count"] == 2424
    assert summary["methods"]["intersection"] == old_summary["methods"]["intersection"]
    old_protocol = read(OLD / "metadata/post" / mode / "protocol.json")
    for field in reference["verified_contract_fields"]:
        assert protocol[field] == old_protocol[field], field
    assert digest(OLD / "metadata/post" / mode / "protocol.json") == reference["source_protocol_sha256"]
    assert digest(REPO / "tools/vae_compare_post.py") == protocol["runner_sha256"]
    for path, sha in protocol["post_algorithm_files"].items():
        assert digest(REPO / path) == sha
    rows = list(csv.DictReader((directory / "samples.csv").open()))
    assert len(rows) == 4834
    csv_maps = {method: {r["sample_id"]: r for r in rows if r["method"] == method} for method in ("intersection", "local")}
    maps = {method: {} for method in csv_maps}
    record_hashes, step_hashes = {}, {}
    for method in maps:
        assert set(csv_maps[method]) == set(IDS)
        for model_id in IDS:
            base = OLD if method == "intersection" else ROOT
            path = base / "metadata/post" / mode / "samples" / method / (model_id + ".json")
            row = read(path)
            assert row["sample_id"] == model_id and row["method"] == method and row["mode"] == mode
            assert row["status"] in TERMINAL
            assert row["raw_sha256"] == digest(base / "raw_predictions/posterior_mean" / method / model_id / "data.npz")
            assert row["protocol_fingerprint"] == fingerprint(old_protocol if method == "intersection" else protocol)
            for field in ("native_valid_solid", "compat_valid_solid", "step_exists", "step_readable", "success_marker"):
                assert row.get(field) in (None, True, False), (method, model_id, field)
                value = csv_maps[method][model_id].get(field, "")
                assert value == ("" if row.get(field) is None else str(row[field]))
            if method == "intersection":
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
        old, new = maps["intersection"][model_id], maps["local"][model_id]
        if old.get("native_valid_solid") is True:
            name = "both_valid" if new.get("native_valid_solid") is True else "intersection_valid_to_local_failed"
        else:
            name = "intersection_failed_to_local_valid" if new.get("native_valid_solid") is True else "both_failed"
        transitions[name].append(model_id)
    assert transitions == summary["native_validity_transitions"]["ids"]
    assert sum(map(len, transitions.values())) == 2417
    common = [model_id for model_id in IDS if all(maps[method][model_id].get("geometry_status") == "ok" for method in maps)]
    assert len(common) == summary["paired_common_geometry"]["count"]
    for metric, values in summary["paired_common_geometry"]["metrics"].items():
        for method in maps:
            verify_stats([maps[method][item][metric] for item in common], values[method], mode + metric + method)
        verify_stats([maps["local"][item][metric] - maps["intersection"][item][metric] for item in common], values["local_minus_intersection"], mode + metric + ":difference")
    return {"terminal_records_per_method": {method: len(group) for method, group in maps.items()},
            "transition_counts": {name: len(group) for name, group in transitions.items()},
            "counts_and_fixed_denominator_rates_verified": True, "common_geometry_count": len(common),
            "baseline_summary_exactly_equal_previous": True, "baseline_record_hashes_unchanged_since_reuse": len(record_hashes),
            "baseline_step_hashes_unchanged_since_reuse": len(step_hashes), "post_algorithms_unchanged": True,
            "source_step_hash_limitation": "Checks protect assets since this evaluation's reuse snapshot, not against unrecorded historical mutations.",
            "summary_sha256": digest(directory / "summary.json"), "csv_sha256": digest(directory / "samples.csv")}


def supplementary_raw_review() -> dict:
    """Recount descriptive subsets and paired trimming from the primary CSVs."""
    result = {"scope": "Descriptive checks only; the full 2417-object primary denominator is unchanged."}
    subset_path = ROOT / "diagnostics/adjacency_subset.json"
    distribution_path = ROOT / "diagnostics/raw_distribution.json"
    if subset_path.exists():
        subset = read(subset_path)
        path = ROOT / "raw_metrics/posterior_mean/per_object.csv"
        assert digest(path) == subset["source_sha256"]
        rows = list(csv.DictReader(path.open()))
        table = {(row["method"], row["model_id"]): row for row in rows}
        exact = [sid for sid in sorted(IDS) if all(
            float(table[method, sid]["topology_fp"]) == 0 and float(table[method, sid]["topology_fn"]) == 0
            for method in ("intersection", "local")
        )]
        identities = {"both_adjacency_exact": exact,
                      "at_least_one_adjacency_inexact": sorted(set(IDS) - set(exact))}
        for name, group in subset["cohorts"].items():
            assert group["ids"] == identities[name] and group["count"] == len(identities[name])
            for metric, values in group["metrics"].items():
                samples = {method: [float(table[method, sid][metric]) for sid in identities[name]] for method in ("intersection", "local")}
                for method, numbers in samples.items():
                    verify_stats(numbers, values[method], name + metric + method)
                close(math.fsum(samples["local"]) / math.fsum(samples["intersection"]), values["local_mean_over_intersection"], name + metric + ":ratio")
                assert sum(b < a - 1e-12 for a, b in zip(samples["intersection"], samples["local"])) == values["local_improved_count"]
        result["adjacency_subset"] = {"counts": {name: len(ids) for name, ids in identities.items()},
                                      "ids_and_metric_counts_means_medians_p90_ratios_wins_verified": True,
                                      "sha256": digest(subset_path)}
    if distribution_path.exists():
        distribution = read(distribution_path)
        result["distribution"] = {"sha256": digest(distribution_path), "modes": {}}
        for mode, values in distribution["modes"].items():
            path = ROOT / "raw_metrics" / mode / "per_object.csv"
            assert digest(path) == values["source_csv_sha256"]
            rows = list(csv.DictReader(path.open()))
            table = {(row["method"], row["model_id"]): row for row in rows}
            checked = 0
            for metric, record in values["metrics"].items():
                selected = [sid for sid in sorted(IDS) if all(table[method, sid].get(metric) not in (None, "") for method in ("intersection", "local"))]
                assert record["common_count"] == len(selected) and record["missing_count"] == len(IDS) - len(selected)
                if not selected:
                    continue
                samples = {method: {sid: float(table[method, sid][metric]) for sid in selected} for method in ("intersection", "local")}
                for method, numbers in samples.items():
                    verify_stats(list(numbers.values()), record[method], mode + metric + method)
                deltas = {sid: samples["local"][sid] - samples["intersection"][sid] for sid in selected}
                assert record["win_count"] == sum(delta < -1e-12 for delta in deltas.values())
                assert record["loss_count"] == sum(delta > 1e-12 for delta in deltas.values())
                assert record["tie_count"] == sum(abs(delta) <= 1e-12 for delta in deltas.values())
                ranking = sorted(selected, key=lambda sid: (-samples["local"][sid], sid))
                for fraction in (0.01, 0.05, 0.1):
                    tail = record["tail"][f"top_{int(fraction * 100)}pct"]
                    count = math.ceil(fraction * len(selected))
                    removed, retained = ranking[:count], ranking[count:]
                    assert tail["count"] == count and tail["retained_count"] == len(retained)
                    if fraction == 0.01:
                        assert tail["removed_ids"] == removed
                    for method in ("intersection", "local"):
                        close(math.fsum(samples[method][sid] for sid in retained) / len(retained), tail[f"trimmed_{method}_mean"], mode + metric + method + ":paired_trim")
                    close(math.fsum(samples["local"][sid] for sid in removed) / math.fsum(samples["local"].values()), tail["local_error_share"], mode + metric + ":tail_share")
                checked += 1
            result["distribution"]["modes"][mode] = {"metric_blocks_checked": checked,
                "counts_means_medians_p90_wins_and_same_id_1_5_10pct_trimming_verified": True}
    return result


def capacity_review() -> dict:
    import torch
    path = ROOT / "diagnostics/intersection_capacity.json"
    source = read(path)
    result = {}
    for method, expected in source.items():
        checkpoint = torch.load(expected["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
        tensors = {key: value for key, value in checkpoint["state_dict"].items() if key.startswith("model.decoder.inter.")}
        count = sum(value.numel() for value in tensors.values())
        assert len(tensors) == expected["state_keys"] and count == expected["state_elements"]
        result[method] = {"state_keys": len(tensors), "state_elements": count}
    return {"scope": "Only model.decoder.inter; not total VAE capacity", "methods": result, "artifact_sha256": digest(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-only", action="store_true", help="Audit completed raw outputs while post processing is still active.")
    args = parser.parse_args()
    assert len(IDS) == len(set(IDS)) == 2417 and digest(SPLIT) == EXPECTED_SHA
    assert digest(SELECTED) == SELECTED_SHA
    review = {"created_utc": datetime.now(timezone.utc).isoformat(), "reviewer": "audit_training_local",
              "created_display_time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "display_timezone": "Asia/Shanghai", "checkpoint_sha256": SELECTED_SHA,
              "review_builder_sha256": digest(Path(__file__).resolve()),
              "independent_method": "No pipeline metric or summarizer imports; raw CSV statistics, terminal JSONs, NPZ arrays and hashes checked directly.",
              "code": source_review(), "inputs": {}, "raw": {}, "post": {}, "pending": []}
    if (ROOT / "diagnostics/intersection_capacity.json").exists():
        review["intersection_capacity"] = capacity_review()
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
    review["supplementary_raw"] = supplementary_raw_review()
    review["status"] = "pending" if review["pending"] else "passed"
    (ROOT / "independent_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": review["status"], "pending": review["pending"], "post": review["post"]}, indent=2))


if __name__ == "__main__":
    main()
