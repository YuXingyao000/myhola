"""Freeze the intersection failure census and audit historical diagnostic reuse.

This is metadata/file-hash work only: no inference, OCC execution, repairs, or
baseline writes. Missing assets stay visible; conflicting source/protocol hashes
are rejected. Existing diagnostic fingerprints are preserved, never rebadged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from tools.failure_census_contract import METHODS, case_path, case_record
from tools.vae_compare_post import REFERENCE_CONTRACT_FIELDS, TERMINAL, fingerprint, sha256_file


VERSION = "intersection_failure_census_manifest_v1"
SPLIT_SHA256 = "2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3"
EXPECTED_COHORTS = {"both_failed": 262, "new_failure": 42, "rescued": 107, "both_valid": 2006}
REPO = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def typed_value(key: str, value: str):
    if not value:
        return None
    if (key in {"model_id", "sample_id", "method", "mode", "status", "stage", "error", "representation", "result_origin", "compatibility_error"}
            or key.endswith(("sha256", "fingerprint", "_path", "_file", "_status"))):
        return value
    if value in ("True", "False"):
        return value == "True"
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    try:
        result = float(value)
    except ValueError:
        return value
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite source value {key}={value}")
    return result


def read_table(path: Path, id_key: str) -> dict:
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
            if not isinstance(sid, str) or not re.fullmatch(r"\d{8}", sid) or method not in METHODS:
                raise ValueError(f"Invalid identity/method: {sid!r}/{method!r}")
            if (method, sid) in rows:
                raise ValueError(f"Duplicate identity: {method}/{sid}")
            rows[method, sid] = row
    return rows


def partition_population(raw: dict, post: dict, fit: dict, intended_ids: list[str]) -> dict:
    expected = {(method, sid) for method in METHODS for sid in intended_ids}
    if not intended_ids or len(set(intended_ids)) != len(intended_ids):
        raise ValueError("Intended IDs must be nonempty and unique")
    if any(set(table) != expected for table in (raw, post, fit)):
        raise ValueError("Raw/post/fit must have identical intended IDs for both methods")
    groups = {key: [] for key in EXPECTED_COHORTS}
    regressions = []
    fit_transitions = {method: Counter() for method in METHODS}
    for sid in sorted(intended_ids):
        for method in METHODS:
            r = raw[method, sid]
            if r.get("status") != "ok" or r.get("mode") != "posterior_mean" or r.get("shared_input_status") != "verified":
                raise ValueError(f"Raw contract not verified: {method}/{sid}")
            for mode, table in (("optimized", post), ("fit_only", fit)):
                row = table[method, sid]
                if row.get("status") not in TERMINAL or row.get("mode") != mode:
                    raise ValueError(f"Nonterminal or wrong-mode source: {method}/{sid}/{mode}")
                if type(row.get("native_valid_solid")) is not bool:
                    raise ValueError(f"Unknown native validity: {method}/{sid}/{mode}")
                if row.get("representation") != "legacy_half_edges":
                    raise ValueError(f"Wrong half-edge representation: {method}/{sid}")
            a, b = fit[method, sid]["native_valid_solid"], post[method, sid]["native_valid_solid"]
            fit_transitions[method][f"fit_{'valid' if a else 'failed'}_optimized_{'valid' if b else 'failed'}"] += 1
            if method == "intersection" and a and not b:
                regressions.append(sid)
        hashes = [raw[m, sid].get("input_latent_sha256") for m in METHODS]
        if not hashes[0] or hashes[0] != hashes[1] or not re.fullmatch(r"[a-f0-9]{64}", hashes[0]):
            raise ValueError(f"Paired input latent hashes differ/absent: {sid}")
        old, new = [post[m, sid]["native_valid_solid"] for m in METHODS]
        cohort = "both_valid" if old and new else "rescued" if new else "new_failure" if old else "both_failed"
        groups[cohort].append(sid)
    return {"cohorts": groups, "optimization_regression_ids": regressions,
            "fit_only_vs_optimized_all_accepted": {m: dict(c) for m, c in fit_transitions.items()}}


def step_data_hash(path: Path) -> str:
    # Match the original observer exactly; STEP headers contain volatile timestamps.
    data = path.read_bytes()
    start = data.find(b"DATA;")
    end = data.find(b"ENDSEC;", start)
    if start < 0 or end < 0:
        raise ValueError(f"Missing STEP DATA section: {path}")
    return hashlib.sha256(data[start + len(b"DATA;"):end]).hexdigest()


class AssetAudit:
    def __init__(self):
        self.entries = {}

    def add(self, path: Path, expected_sha256: str | None = None) -> dict:
        path = path.resolve()
        key = str(path)
        if key not in self.entries:
            self.entries[key] = {"path": key, "exists": path.is_file(),
                                 "sha256": sha256_file(path) if path.is_file() else None}
        entry = self.entries[key]
        if expected_sha256 and entry["exists"] and entry["sha256"] != expected_sha256:
            raise ValueError(f"Source asset SHA256 changed: {path}")
        if expected_sha256:
            entry["expected_sha256"] = expected_sha256
            entry["expected_hash_matches"] = entry["sha256"] == expected_sha256 if entry["exists"] else None
        return dict(entry)


def validate_protocols(comparison: Path, reference: Path, data_root: Path, split: Path, audit: AssetAudit) -> dict:
    protocols = {}
    for mode in ("fit_only", "optimized"):
        current_path = comparison / "metadata/post" / mode / "protocol.json"
        old_path = reference / "metadata/post" / mode / "protocol.json"
        current, old = read_json(current_path), read_json(old_path)
        differences = [key for key in REFERENCE_CONTRACT_FIELDS if current.get(key) != old.get(key)]
        if differences:
            raise ValueError(f"Reference post protocol differs: {mode}/{differences}")
        expected = {"source": "posterior_mean", "mode": mode, "data_root": str(data_root),
                    "split_sha256": sha256_file(split), "intended_count": 2417, "original_test_count": 2424,
                    "timeout_seconds": 180, "max_optimize_iter": 200,
                    "adaptive_max_optimize_iter": 600 if mode == "optimized" else 0,
                    "drop_num": 0, "use_cuda": False, "threads_per_process": 1}
        for key, value in expected.items():
            if current.get(key) != value:
                raise ValueError(f"Frozen post protocol differs: {mode}/{key}")
        if current.get("candidate_method") != "intersection":
            raise ValueError("Current comparison must explicitly name intersection")
        source = current.get("reference_legacy", {})
        if (source.get("source_protocol_fingerprint") != fingerprint(old)
                or source.get("source_protocol") != old
                or source.get("source_protocol_sha256") != sha256_file(old_path)
                or Path(source.get("reference_root", "")).resolve() != reference):
            raise ValueError(f"Reference provenance does not match actual protocol: {mode}")
        algorithms = current.get("post_algorithm_files", {})
        if set(algorithms) != {str(p.relative_to(REPO)) for p in (REPO / "src/brepnet/post").glob("*.py")}:
            raise ValueError("Post algorithm file inventory differs")
        for name, digest in algorithms.items():
            asset = audit.add(REPO / name, digest)
            if not asset["exists"]:
                raise ValueError(f"Missing production algorithm: {name}")
        protocols[mode] = {"current": current, "legacy": old,
                           "current_source": audit.add(current_path), "legacy_source": audit.add(old_path),
                           "current_fingerprint": fingerprint(current), "legacy_fingerprint": fingerprint(old)}
    return protocols


def make_case(sid, method, cohort, raw, post, fit, comparison, reference, data_root, protocols, audit):
    post_root = reference if method == "legacy" else comparison
    paths = {"raw_npz": comparison / "raw_predictions/posterior_mean" / method / sid / "data.npz",
             "post_step": post_root / "post/optimized" / method / sid / "recon_brep.step",
             "fit_step": post_root / "post/fit_only" / method / sid / "recon_brep.step",
             "gt_npz": data_root / sid / "data.npz", "gt_step": data_root / sid / "normalized_shape.step"}
    case = {"id": sid, "method": method, "cohort": cohort, "gt_face_count": raw[method, sid]["gt_face_count"],
            "raw_record": raw[method, sid], "post_record": post[method, sid], "fit_record": fit[method, sid],
            "paths": {k: str(v.resolve()) for k, v in paths.items()}, "source_records": {}}
    raw_sha = None
    for mode, stage in (("optimized", "post"), ("fit_only", "fit")):
        record = case_record(case, stage)
        expected_key = protocols[mode]["legacy_fingerprint" if method == "legacy" else "current_fingerprint"]
        expected_origin = "reference_reused" if method == "legacy" else "current_run"
        if record.get("protocol_fingerprint") != expected_key or record.get("result_origin") != expected_origin:
            raise ValueError(f"Wrong post provenance: {method}/{sid}/{mode}")
        record_path = post_root / "metadata/post" / mode / "samples" / method / f"{sid}.json"
        record_asset = audit.add(record_path, record.get("source_record_sha256") if method == "legacy" else None)
        if method == "legacy" and Path(record.get("source_record_path", "")).resolve() != record_path:
            raise ValueError(f"Historical record points elsewhere: {sid}/{mode}")
        if not record_asset["exists"]:
            raise ValueError(f"Missing source post record: {record_path}")
        original = read_json(record_path)
        for key in ("sample_id", "method", "mode", "status", "native_valid_solid", "shape_type",
                    "native_brepcheck_valid", "num_faces", "num_edges", "num_vertices", "success_marker",
                    "raw_sha256", "protocol_fingerprint"):
            if record.get(key) != original.get(key):
                raise ValueError(f"CSV differs from source record: {method}/{sid}/{mode}/{key}")
        if raw_sha is not None and raw_sha != record.get("raw_sha256"):
            raise ValueError(f"Fit/optimized raw input differs: {method}/{sid}")
        raw_sha = record.get("raw_sha256")
        if not isinstance(raw_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", raw_sha):
            raise ValueError(f"Missing raw fingerprint: {method}/{sid}")
        case["source_records"][stage] = record_asset
    assets = {key: audit.add(path, raw_sha if key == "raw_npz" else None) for key, path in paths.items()}
    case["path_exists"] = {key: row["exists"] for key, row in assets.items()}
    case["asset_sha256"] = {key: row["sha256"] for key, row in assets.items()}
    return case


def verify_reuse(case, old_case, baseline_root, protocols, audit):
    """Return eligibility with reasons; missing/unverified diagnostics remain unknown."""
    sid = case["id"]
    result = {"id": sid, "method": "legacy", "cohort": case["cohort"],
              "result_origin": "historical_diagnostic_reused", "eligible": False, "reasons": [], "sources": {}}
    reasons = result["reasons"]
    for key in ("raw_npz", "post_step", "fit_step", "gt_npz", "gt_step"):
        if case_path(case, key).resolve() != case_path(old_case, key).resolve():
            reasons.append(f"baseline asset path differs: {key}")
    for stage in ("raw", "post", "fit"):
        old_record = case_record(old_case, stage)
        keys = ("input_latent_sha256",) if stage == "raw" else ("raw_sha256", "protocol_fingerprint", "native_valid_solid")
        if any(case_record(case, stage).get(k) != old_record.get(k) for k in keys):
            reasons.append(f"baseline source record differs: {stage}")
    diagnostics = {}
    for name, directory in (("trace", "traces"), ("upstream", "upstream"), ("saved_step", "saved_step")):
        path = baseline_root / directory / f"{sid}.json"
        result["sources"][name] = audit.add(path)
        if not path.is_file():
            reasons.append(f"missing {name}")
            continue
        data = diagnostics[name] = read_json(path)
        identity_key = "id" if name == "saved_step" else "sample_id"
        if data.get(identity_key) != sid or data.get("status") != "ok":
            reasons.append(f"unverified {name} identity/status")
        result.setdefault("original_protocol_fingerprints", {})[name] = data.get("protocol_fingerprint")
        if data.get("protocol_fingerprint") != fingerprint(protocols[name]):
            reasons.append(f"{name} protocol fingerprint differs")
    trace = diagnostics.get("trace", {})
    if (trace.get("instrumentation_errors") or trace.get("comparison", {}).get("same_recorded_outcome") is not True
            or trace.get("step_data_comparison", {}).get("equal") is not True):
        reasons.append("original trace is not a complete identical replay")
    if trace.get("raw_sha256") != case["asset_sha256"]["raw_npz"]:
        reasons.append("trace raw hash differs")
    step = case_path(case, "post_step")
    if step.is_file():
        digest = step_data_hash(step)
        result["original_step_data_sha256"] = digest
        hashes = trace.get("step_data_comparison", {})
        if hashes.get("original") != digest or hashes.get("replay") != digest:
            reasons.append("original STEP DATA differs from frozen trace")
    else:
        reasons.append("missing original STEP")
    upstream = diagnostics.get("upstream", {})
    for name, asset_key in (("raw", "raw_npz"), ("gt", "gt_npz")):
        if upstream.get("sources", {}).get(name, {}).get("sha256") != case["asset_sha256"][asset_key]:
            reasons.append(f"upstream {name} source differs")
    if upstream.get("sources", {}).get("trace", {}).get("sha256") != result["sources"]["trace"]["sha256"]:
        reasons.append("upstream trace source differs")
    geometry = baseline_root / "replay" / sid / "trace_geometry.npz"
    expected = upstream.get("sources", {}).get("stage_geometry", {}).get("sha256")
    result["sources"]["stage_geometry"] = audit.add(geometry, expected)
    if not expected or not geometry.is_file():
        reasons.append("stage geometry unavailable/unfingerprinted")
    if diagnostics.get("saved_step", {}).get("step_sha256") != case["asset_sha256"]["post_step"]:
        reasons.append("saved-step diagnostic source differs")
    if not all(case["path_exists"].values()):
        reasons.append("one or more source assets missing")
    result["eligible"] = not reasons
    return result


def baseline_protocol_audit(baseline, old_manifest, current_post, audit):
    protocols, result = {}, {}
    for name, directory in (("trace", "provenance"), ("upstream", "upstream"), ("saved_step", "saved_step")):
        path = baseline / "metadata" / directory / "protocol.json"
        data = protocols[name] = read_json(path)
        if data.get("manifest_sha256") != sha256_file(baseline / "manifest.json"):
            raise ValueError(f"Historical diagnostic manifest changed: {name}")
        result[name] = {"source": audit.add(path), "original_fingerprint": fingerprint(data), "protocol": data}
    production = current_post["post_algorithm_files"]
    frozen_sources = protocols["trace"]["source_sha256"]
    for name, digest in frozen_sources.items():
        if name.startswith("src/"):
            if name in production and production[name] != digest:
                raise ValueError(f"Historical trace post source differs: {name}")
            if audit.add(REPO / name, digest)["exists"] is not True:
                raise ValueError(f"Missing historical production source: {name}")
    result["production_contract"] = {"current_full_post_algorithm_files": production,
        "historical_trace_production_files": {k:v for k,v in frozen_sources.items() if k.startswith("src/")},
        "runner_hash_may_differ": True,
        "note": "Historical observer fingerprints remain original. Method/dispatch extensions are not production post changes; measurement settings are preserved below."}
    # Import only pure metadata; none of these modules executes OCC on import.
    from tools.legacy_failure_provenance import SAMPLING as fit_sampling
    from tools.legacy_failure_upstream import SETTINGS as upstream_settings
    from tools.legacy_failure_saved_step import THRESHOLDS as saved_thresholds
    if fingerprint(protocols["trace"].get("sampling")) != fingerprint(fit_sampling):
        raise ValueError("Historical fit measurement settings differ")
    settings_delta = measurement_settings_delta(protocols["upstream"].get("settings", {}), upstream_settings)
    result["allowed_description_delta"] = settings_delta
    saved = protocols["saved_step"]
    if saved.get("curve_samples") != 128 or saved.get("read_precision_mode") != 0 or saved.get("thresholds") != list(saved_thresholds):
        raise ValueError("Historical saved-step measurement settings differ")
    for key,value in {"post_mode":"optimized", "optimization_initial_budget":200,
                      "optimization_adaptive_limit":600,"drop_num":0,"cpu_threads":1}.items():
        if protocols["trace"].get(key) != value:
            raise ValueError(f"Historical replay budget differs: {key}")
    return protocols, result


def measurement_settings_delta(historical: dict, current: dict) -> dict:
    """Only the no-dedup description may change during method generalization."""
    differences = {key: {"historical": historical.get(key), "current": current.get(key)}
                   for key in historical.keys() | current.keys()
                   if fingerprint(historical.get(key)) != fingerprint(current.get(key))}
    if differences:
        if set(differences) != {"dedup"} or not all(
                isinstance(value, str) and value.startswith("not_applied")
                for value in differences["dedup"].values()):
            raise ValueError(f"Historical upstream measurement settings differ: {sorted(differences)}")
    return differences


def build_manifests(comparison, reference, baseline, data_root, split):
    audit = AssetAudit()
    ids = split.read_text().split()
    if len(ids) != 2417 or sha256_file(split) != SPLIT_SHA256:
        raise ValueError("This study requires the frozen 2417-ID accepted split")
    source_paths = {"raw_metrics":comparison / "raw_metrics/posterior_mean/per_object.csv",
                    "optimized":comparison / "post/optimized/samples.csv",
                    "fit_only":comparison / "post/fit_only/samples.csv"}
    raw, post, fit = [read_table(source_paths[k], id_key) for k,id_key in
                      (("raw_metrics","model_id"),("optimized","sample_id"),("fit_only","sample_id"))]
    partition = partition_population(raw, post, fit, ids)
    groups = partition["cohorts"]
    if {key:len(value) for key,value in groups.items()} != EXPECTED_COHORTS or len(partition["optimization_regression_ids"]) != 6:
        raise ValueError("Frozen cohort population differs from 304/107/42/262 and six optimization regressions")
    protocols = validate_protocols(comparison, reference, data_root, split, audit)
    raw_summary_path = comparison / "raw_metrics/posterior_mean/summary.json"
    raw_summary = read_json(raw_summary_path)
    if (raw_summary.get("split_sha256") != SPLIT_SHA256 or raw_summary.get("candidate_method") != "intersection"
            or raw_summary.get("protocol_valid") is not True or raw_summary.get("shared_input_status_counts") != {"verified":2417}):
        raise ValueError("Raw summary input protocol differs")
    source_metadata = {key:audit.add(path) for key,path in source_paths.items()}
    source_metadata.update(raw_summary=audit.add(raw_summary_path), split=audit.add(split))
    failed = sorted(groups["both_failed"] + groups["new_failure"])
    selected = sorted(failed + groups["rescued"])
    cohort_of = {sid:key for key,values in groups.items() for sid in values}
    current_cases = [make_case(sid,"intersection",cohort_of[sid],raw,post,fit,comparison,reference,data_root,protocols,audit) for sid in selected]
    old_cases = [make_case(sid,"legacy",cohort_of[sid],raw,post,fit,comparison,reference,data_root,protocols,audit) for sid in selected]
    old_by_id = {case["id"]:case for case in old_cases}
    original_manifest_path = baseline / "manifest.json"
    original_manifest = read_json(original_manifest_path)
    historical_cases = {case["id"]:case for case in original_manifest["cases"]}
    reused = sorted(groups["both_failed"] + groups["rescued"])
    if set(historical_cases) != set(reused) or len(original_manifest["cases"]) != 369:
        raise ValueError("Historical 369 census does not equal common-failure plus rescued IDs")
    baseline_protocols, protocol_audit = baseline_protocol_audit(baseline, original_manifest, protocols["optimized"]["current"], audit)
    reuse_entries = [verify_reuse(old_by_id[sid],historical_cases[sid],baseline,baseline_protocols,audit) for sid in reused]
    common = {"schema":VERSION,"created_at_utc":datetime.now(timezone.utc).isoformat(),
              "comparison_root":str(comparison),"reference_root":str(reference),"baseline_census_root":str(baseline),"data_root":str(data_root),
              "sources":source_metadata,"post_protocols":protocols,
              "population":{"original_test":2424,"contract_unsupported":7,"accepted_test":2417,
                            "native_valid":2113,"native_failed":304,"main_failure_denominator":304,
                            "new_diagnostic_cases":411,"legacy_additional_controls":42},
              **partition, "failed_ids":failed,
              "face_identity_contract":{"gt_order_preserved":True,"face_deduplication":"not_applied",
                                        "evidence":"Frozen posterior-mean exports share input latent hashes; original GT face slots are retained. This does not map STEP subshape numbering."},
              "selection_protocol":{"sampling":False,"main_failure_rule":"intersection optimized native_valid_solid is False; diagnostic/geometry failures stay in denominator",
                                    "control_scope":"107 rescued are outcome-selected paired controls, not representative of all 2113 successes"}}
    manifest = dict(common, method="intersection", cases=current_cases, scope="304 intersection failures plus 107 rescued controls; main denominator stays 304")
    controls = dict(common, method="legacy", cases=[old_by_id[sid] for sid in groups["new_failure"]], scope="42 original-success controls for new failures; new replay only, original outputs read-only")
    reuse = {"schema":"failure_census_baseline_reuse_v1","baseline_manifest":audit.add(original_manifest_path),
             "protocol_audit":protocol_audit,"intended_count":369,"eligible_count":sum(x["eligible"] for x in reuse_entries),
             "note":"References only. Preserve the source diagnostic protocol fingerprints; no old result is presented as a new execution.","cases":reuse_entries}
    assets = {"schema":"failure_census_source_assets_v1","created_at_utc":common["created_at_utc"],
              "source_assets":sorted(audit.entries.values(),key=lambda x:x["path"]),
              "missing_assets":[x["path"] for x in audit.entries.values() if not x["exists"]],
              "note":"SHA256 inventory before diagnosis; rerun file hashing after diagnosis to confirm sources stayed unchanged. No model, raw or original STEP writes are performed."}
    return {"manifest.json":manifest,"legacy_control_manifest.json":controls,"baseline_reuse.json":reuse,"source_asset_audit.json":assets}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("comparison-root","reference-root","baseline-census-root","data-root","split-list","output-root"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args(argv)
    comparison,reference,baseline,data_root,split,root = [p.resolve() for p in
        (args.comparison_root,args.reference_root,args.baseline_census_root,args.data_root,args.split_list,args.output_root)]
    if any(root == source or root.is_relative_to(source) for source in (comparison,reference,baseline,data_root)):
        raise ValueError("Output must be independent of comparison, baseline and data roots")
    names = ["manifest.json","legacy_control_manifest.json","baseline_reuse.json","source_asset_audit.json",
             "failed_ids.txt","new_failure_ids.txt","both_failed_ids.txt","rescued_ids.txt","optimization_regression_ids.txt"]
    if any((root/name).exists() for name in names):
        raise FileExistsError("Frozen census outputs already exist; select a new root")
    documents = build_manifests(comparison,reference,baseline,data_root,split)
    root.mkdir(parents=True,exist_ok=True)
    for name,data in documents.items():
        (root/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    manifest = documents["manifest.json"]
    lists = {"failed_ids":manifest["failed_ids"],"optimization_regression_ids":manifest["optimization_regression_ids"],
             **{name+"_ids":manifest["cohorts"][name] for name in ("new_failure","both_failed","rescued")}}
    for name,values in lists.items():
        (root/(name+".txt")).write_text("\n".join(values)+"\n")
    print(json.dumps({"new_cases":len(manifest["cases"]),"legacy_controls":len(documents["legacy_control_manifest.json"]["cases"]),
                      "baseline_reuse_eligible":documents["baseline_reuse.json"]["eligible_count"],
                      "missing_assets":documents["source_asset_audit.json"]["missing_assets"]},indent=2))


if __name__ == "__main__":
    main()
