"""Freeze a method-explicit paired failure census without executing geometry."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from tools.failure_census_contract import SUPPORTED_METHODS, case_method, case_path, case_record
from tools.intersection_failure_manifest import (
    AssetAudit, REPO, SPLIT_SHA256, baseline_protocol_audit, read_json,
    step_data_hash, typed_value, verify_reuse,
)
from tools.vae_compare_post import REFERENCE_CONTRACT_FIELDS, TERMINAL, fingerprint, sha256_file

VERSION = "paired_failure_manifest_v1"
METHODS = ("intersection", "local")
CANDIDATE_METHODS = ("local", "conditioned", "point_feedback")
BASELINE_METHODS = ("intersection", "conditioned")
COHORTS = ("both_failed", "new_failure", "rescued", "both_valid")
LABELS = {"intersection": "上一版 Intersection refinement", "local": "本版局部注意力",
          "conditioned": "曲线条件调制", "point_feedback": "逐点曲线反馈"}


def paired_methods(candidate_method="local", baseline_method="intersection"):
    if candidate_method not in CANDIDATE_METHODS:
        raise ValueError(f"Unsupported paired candidate {candidate_method!r}")
    if baseline_method not in BASELINE_METHODS or baseline_method == candidate_method:
        raise ValueError(f"Unsupported paired baseline {baseline_method!r}/{candidate_method!r}")
    return baseline_method, candidate_method


def read_table(path: Path, id_key: str, candidate_method="local", baseline_method="intersection") -> dict:
    methods = paired_methods(candidate_method, baseline_method)
    rows = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Missing/repeated CSV columns: {path}")
        for source in reader:
            if None in source or any(value is None for value in source.values()):
                raise ValueError(f"Malformed CSV row: {path}")
            row = {key: typed_value(key, value) for key, value in source.items()}
            sid, method = row.get(id_key), row.get("method")
            if not isinstance(sid, str) or not re.fullmatch(r"\d{8}", sid) or method not in methods:
                raise ValueError(f"Invalid comparison identity/method: {sid!r}/{method!r}")
            if (method, sid) in rows:
                raise ValueError(f"Duplicate identity: {method}/{sid}")
            rows[method, sid] = row
    return rows


def partition_population(raw: dict, post: dict, fit: dict, ids: list[str], candidate_method="local", baseline_method="intersection") -> dict:
    methods = paired_methods(candidate_method, baseline_method)
    expected = {(method, sid) for method in methods for sid in ids}
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Intended IDs must be nonempty and unique")
    if any(set(table) != expected for table in (raw, post, fit)):
        raise ValueError("Raw/post/fit must contain exactly the intended IDs for both methods")
    groups = {key: [] for key in COHORTS}
    transitions = {method: Counter() for method in methods}
    regressions = []
    for sid in sorted(ids):
        for method in methods:
            r = raw[method, sid]
            if r.get("status") != "ok" or r.get("mode") != "posterior_mean" or r.get("shared_input_status") != "verified":
                raise ValueError(f"Raw contract not verified: {method}/{sid}")
            for mode, table in (("optimized", post), ("fit_only", fit)):
                row = table[method, sid]
                if row.get("status") not in TERMINAL or row.get("mode") != mode:
                    raise ValueError(f"Nonterminal/wrong-mode source: {method}/{sid}/{mode}")
                if type(row.get("native_valid_solid")) is not bool:
                    raise ValueError(f"Unknown native validity: {method}/{sid}/{mode}")
                if row.get("representation") != "legacy_half_edges":
                    raise ValueError(f"Wrong representation: {method}/{sid}")
            a, b = fit[method, sid]["native_valid_solid"], post[method, sid]["native_valid_solid"]
            transitions[method][f"fit_{'valid' if a else 'failed'}_optimized_{'valid' if b else 'failed'}"] += 1
            if method == candidate_method and a and not b:
                regressions.append(sid)
        hashes = [raw[m, sid].get("input_latent_sha256") for m in methods]
        if hashes[0] != hashes[1] or not isinstance(hashes[0], str) or not re.fullmatch(r"[a-f0-9]{64}", hashes[0]):
            raise ValueError(f"Paired input latent hashes differ/absent: {sid}")
        old, new = [post[m, sid]["native_valid_solid"] for m in methods]
        group = "both_valid" if old and new else "rescued" if new else "new_failure" if old else "both_failed"
        groups[group].append(sid)
    return {"cohorts": groups, "optimization_regression_ids": regressions,
            "fit_only_vs_optimized_all_accepted": {m: dict(v) for m, v in transitions.items()}}


def validate_protocols(comparison, reference, data_root, split, audit, candidate_method="local", baseline_method="intersection"):
    paired_methods(candidate_method, baseline_method)
    protocols = {}
    for mode in ("fit_only", "optimized"):
        current_path = comparison / "metadata/post" / mode / "protocol.json"
        old_path = reference / "metadata/post" / mode / "protocol.json"
        current, old = read_json(current_path), read_json(old_path)
        differences = [key for key in REFERENCE_CONTRACT_FIELDS if current.get(key) != old.get(key)]
        if differences:
            raise ValueError(f"Reference post protocol differs: {mode}/{differences}")
        required = {"source": "posterior_mean", "mode": mode, "data_root": str(data_root),
                    "split_sha256": sha256_file(split), "intended_count": 2417,
                    "original_test_count": 2424, "timeout_seconds": 180,
                    "max_optimize_iter": 200, "adaptive_max_optimize_iter": 600 if mode == "optimized" else 0,
                    "drop_num": 0, "use_cuda": False, "threads_per_process": 1,
                    "baseline_method": baseline_method, "candidate_method": candidate_method}
        for key, value in required.items():
            if current.get(key) != value:
                raise ValueError(f"Frozen post protocol differs: {mode}/{key}")
        source = current.get("reference_baseline", {})
        if (source.get("method") != baseline_method or source.get("source_protocol") != old
                or source.get("source_protocol_fingerprint") != fingerprint(old)
                or source.get("source_protocol_sha256") != sha256_file(old_path)
                or Path(source.get("reference_root", "")).resolve() != reference):
            raise ValueError(f"Baseline provenance differs: {mode}")
        algorithms = current.get("post_algorithm_files", {})
        if set(algorithms) != {str(p.relative_to(REPO)) for p in (REPO / "src/brepnet/post").glob("*.py")}:
            raise ValueError("Post algorithm inventory differs")
        for name, digest in algorithms.items():
            if not audit.add(REPO / name, digest)["exists"]:
                raise ValueError(f"Missing production algorithm: {name}")
        protocols[mode] = {"current": current, "baseline": old,
            "current_source": audit.add(current_path), "baseline_source": audit.add(old_path),
            "current_fingerprint": fingerprint(current), "baseline_fingerprint": fingerprint(old)}
    return protocols


def make_case(sid, method, cohort, raw, post, fit, comparison, reference, data_root, protocols, audit, baseline_method="intersection"):
    source_root = reference if method == baseline_method else comparison
    paths = {"raw_npz": comparison / "raw_predictions/posterior_mean" / method / sid / "data.npz",
             "post_step": source_root / "post/optimized" / method / sid / "recon_brep.step",
             "fit_step": source_root / "post/fit_only" / method / sid / "recon_brep.step",
             "gt_npz": data_root / sid / "data.npz", "gt_step": data_root / sid / "normalized_shape.step"}
    case = {"id": sid, "method": method, "cohort": cohort, "gt_face_count": raw[method, sid]["gt_face_count"],
            "raw_record": raw[method, sid], "post_record": post[method, sid], "fit_record": fit[method, sid],
            "paths": {k: str(v.resolve()) for k, v in paths.items()}, "source_records": {}}
    raw_hash = None
    for mode, stage in (("optimized", "post"), ("fit_only", "fit")):
        row = case_record(case, stage)
        is_baseline = method == baseline_method
        expected = protocols[mode]["baseline_fingerprint" if is_baseline else "current_fingerprint"]
        if row.get("protocol_fingerprint") != expected or row.get("result_origin") != ("reference_reused" if is_baseline else "current_run"):
            raise ValueError(f"Wrong post provenance: {method}/{sid}/{mode}")
        path = source_root / "metadata/post" / mode / "samples" / method / f"{sid}.json"
        asset = audit.add(path, row.get("source_record_sha256") if is_baseline else None)
        if is_baseline and Path(row.get("source_record_path", "")).resolve() != path.resolve():
            raise ValueError(f"Historical source record points elsewhere: {sid}/{mode}")
        if not asset["exists"]:
            raise ValueError(f"Missing source post record: {path}")
        original = read_json(path)
        for key in ("sample_id", "method", "mode", "status", "native_valid_solid", "shape_type",
                    "native_brepcheck_valid", "num_faces", "num_edges", "num_vertices", "success_marker",
                    "raw_sha256", "protocol_fingerprint"):
            if row.get(key) != original.get(key):
                raise ValueError(f"CSV differs from source record: {method}/{sid}/{mode}/{key}")
        if raw_hash is not None and raw_hash != row.get("raw_sha256"):
            raise ValueError(f"Fit/optimized raw differs: {method}/{sid}")
        raw_hash = row.get("raw_sha256")
        if not isinstance(raw_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", raw_hash):
            raise ValueError(f"Missing raw fingerprint: {method}/{sid}")
        case["source_records"][stage] = asset
    assets = {key: audit.add(path, raw_hash if key == "raw_npz" else None) for key, path in paths.items()}
    case["path_exists"] = {key: item["exists"] for key, item in assets.items()}
    case["asset_sha256"] = {key: item["sha256"] for key, item in assets.items()}
    return case


def audit_reuse(case, historical_case, baseline_root, protocols, audit, baseline_method="intersection"):
    """Audit exact historical method; no renamed diagnostic records or fingerprints."""
    if case_method(case) != baseline_method or case_method(historical_case) != baseline_method:
        raise ValueError(f"Only exact historical {baseline_method} diagnostics may be reused")
    result = verify_reuse(case, historical_case, baseline_root, protocols, audit)
    # The historical verifier predates configurable methods; its measurements
    # are identical, while identity is explicitly checked here.
    result["method"] = baseline_method
    for name in ("trace", "upstream", "saved_step"):
        source = result["sources"][name]
        if source["exists"] and read_json(Path(source["path"])).get("method") != baseline_method:
            result["reasons"].append(f"historical {name} method differs")
    result["eligible"] = not result["reasons"]
    return result


def build_manifests(comparison, reference, baseline, data_root, split, candidate_method="local", baseline_method="intersection"):
    methods = paired_methods(candidate_method, baseline_method)
    audit = AssetAudit()
    ids = split.read_text().split()
    if len(ids) != 2417 or sha256_file(split) != SPLIT_SHA256:
        raise ValueError("Expected the frozen 2417-ID accepted split")
    paths = {"raw_metrics": comparison / "raw_metrics/posterior_mean/per_object.csv",
             "optimized": comparison / "post/optimized/samples.csv", "fit_only": comparison / "post/fit_only/samples.csv"}
    raw, post, fit = [read_table(paths[key], id_key, candidate_method, baseline_method) for key, id_key in
                      (("raw_metrics", "model_id"), ("optimized", "sample_id"), ("fit_only", "sample_id"))]
    partition = partition_population(raw, post, fit, ids, candidate_method, baseline_method)
    protocols = validate_protocols(comparison, reference, data_root, split, audit, candidate_method, baseline_method)
    raw_summary_path = comparison / "raw_metrics/posterior_mean/summary.json"
    raw_summary = read_json(raw_summary_path)
    if (raw_summary.get("split_sha256") != SPLIT_SHA256 or raw_summary.get("baseline_method") != baseline_method
            or raw_summary.get("candidate_method") != candidate_method or raw_summary.get("protocol_valid") is not True
            or raw_summary.get("shared_input_status_counts") != {"verified": 2417}):
        raise ValueError("Raw summary input protocol differs")
    sources = {key: audit.add(path) for key, path in paths.items()}
    sources.update(raw_summary=audit.add(raw_summary_path), split=audit.add(split))
    groups = partition["cohorts"]
    failed = sorted(groups["both_failed"] + groups["new_failure"])
    selected = sorted(failed + groups["rescued"])
    cohort_of = {sid: key for key, values in groups.items() for sid in values}
    cases = {method: [make_case(sid, method, cohort_of[sid], raw, post, fit, comparison, reference,
                               data_root, protocols, audit, baseline_method) for sid in selected] for method in methods}
    historical_path = baseline / "manifest.json"
    historical = read_json(historical_path)
    if historical.get("method") != baseline_method:
        raise ValueError("Baseline census has wrong method")
    by_id = {case["id"]: case for case in historical["cases"]}
    if len(by_id) != len(historical["cases"]):
        raise ValueError("Duplicate historical cases")
    old_protocols, protocol_audit = baseline_protocol_audit(baseline, historical, protocols["optimized"]["current"], audit)
    reuse_entries = []
    for case in cases[baseline_method]:
        sid = case["id"]
        if sid in by_id:
            reuse_entries.append(audit_reuse(case, by_id[sid], baseline, old_protocols, audit, baseline_method))
        else:
            reuse_entries.append({"id": sid, "method": baseline_method, "cohort": case["cohort"],
                                  "eligible": False, "reasons": ["not in historical diagnostic census"], "sources": {}})
    eligible = {entry["id"] for entry in reuse_entries if entry["eligible"]}
    controls = [case for case in cases[baseline_method] if case["id"] not in eligible]
    valid = len(ids) - len(failed)
    population = {"original_test": 2424, "contract_unsupported": 7, "accepted_test": len(ids),
                  "native_valid": valid, "native_failed": len(failed), "main_failure_denominator": len(failed),
                  "baseline_native_valid": len(groups["both_valid"]) + len(groups["new_failure"]),
                  "baseline_native_failed": len(groups["both_failed"]) + len(groups["rescued"]),
                  "new_diagnostic_cases": len(selected), "baseline_additional_controls": len(controls)}
    common = {"schema": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "comparison_root": str(comparison), "reference_root": str(reference), "baseline_census_root": str(baseline),
              "baseline_method": baseline_method, "candidate_method": candidate_method,
              "method_labels": {method: LABELS[method] for method in methods},
              "data_root": str(data_root), "sources": sources, "post_protocols": protocols, "population": population,
              **partition, "failed_ids": failed,
              "face_identity_contract": {"gt_order_preserved": True, "face_deduplication": "not_applied",
                    "evidence": "Frozen shared posterior-mean inputs preserve GT face slots; not STEP subshape numbering."},
              "selection_protocol": {"sampling": False,
                    "main_failure_rule": f"{candidate_method} optimized native_valid_solid is False; missing diagnosis stays in denominator",
                    "control_scope": "Outcome-selected rescued/failed cohorts, not representative of all successes"}}
    manifest = dict(common, method=candidate_method, cases=cases[candidate_method],
                    scope=f"All {candidate_method} failures plus all rescued controls")
    baseline_controls = dict(common, method=baseline_method, cases=controls, scope="All selected baseline IDs lacking verified reusable diagnosis")
    baseline_cases = dict(common, method=baseline_method, cases=cases[baseline_method], scope="Identical paired baseline IDs; includes historical reuse")
    reuse = {"schema": "paired_failure_baseline_reuse_v1", "baseline_manifest": audit.add(historical_path),
             "protocol_audit": protocol_audit, "intended_count": len(selected), "eligible_count": len(eligible),
             "note": "Historical diagnostics remain at their original paths with original method/protocol fingerprints.", "cases": reuse_entries}
    assets = {"schema": "failure_census_source_assets_v1", "created_at_utc": common["created_at_utc"],
              "source_assets": sorted(audit.entries.values(), key=lambda x: x["path"]),
              "missing_assets": [x["path"] for x in audit.entries.values() if not x["exists"]]}
    return {"manifest.json": manifest, "baseline_control_manifest.json": baseline_controls,
            "baseline_manifest.json": baseline_cases, "baseline_reuse.json": reuse, "source_asset_audit.json": assets}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("comparison-root", "reference-root", "baseline-census-root", "data-root", "split-list", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--candidate-method", choices=CANDIDATE_METHODS, default="local")
    parser.add_argument("--baseline-method", choices=BASELINE_METHODS, default="intersection")
    args = parser.parse_args(argv)
    comparison, reference, baseline, data_root, split, root = [p.resolve() for p in
        (args.comparison_root, args.reference_root, args.baseline_census_root, args.data_root, args.split_list, args.output_root)]
    if any(root == source or root.is_relative_to(source) for source in (comparison, reference, baseline, data_root)):
        raise ValueError("Output must be independent of source roots")
    names = ["manifest.json", "baseline_manifest.json", "baseline_control_manifest.json", "baseline_reuse.json", "source_asset_audit.json",
             "failed_ids.txt", "new_failure_ids.txt", "both_failed_ids.txt", "rescued_ids.txt", "optimization_regression_ids.txt"]
    if any((root / name).exists() for name in names):
        raise FileExistsError("Frozen census files already exist; select a new root")
    documents = build_manifests(comparison, reference, baseline, data_root, split, args.candidate_method, args.baseline_method)
    root.mkdir(parents=True, exist_ok=True)
    for name, value in documents.items():
        (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    manifest = documents["manifest.json"]
    lists = {"failed_ids": manifest["failed_ids"], "optimization_regression_ids": manifest["optimization_regression_ids"],
             **{name + "_ids": manifest["cohorts"][name] for name in ("new_failure", "both_failed", "rescued")}}
    for name, values in lists.items():
        (root / (name + ".txt")).write_text("".join(value + "\n" for value in values))
    print(json.dumps({"candidate_cases": len(manifest["cases"]), "baseline_controls": len(documents["baseline_control_manifest.json"]["cases"]),
                      "baseline_reuse_eligible": documents["baseline_reuse.json"]["eligible_count"],
                      "missing_assets": documents["source_asset_audit.json"]["missing_assets"]}, indent=2))


if __name__ == "__main__":
    main()
