"""Independent provenance and population audit; see the frozen SPEC.md.

This deliberately recomputes counts from terminal records without importing the
report implementation. It writes only validation/independent_audit.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BASE = REPO / "checkpoints/shared_curve_uv/evaluation_20261020/best"
SPLIT = REPO / "checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt"
SPLIT_SHA = "ae607bfe5499e919d1e68a5110c0c211daf1158aedbe57a4756eda99f6f3f0ed"
CHECKPOINT_SHA = "4262f4998fe047e19c3ca21a835d74c183b7430f7a656c709ec8ed045262ada4"
METHOD = "no_edge_feature64"
ROLES = {"source32": (BASE, "shared_curve_uv"), METHOD: (HERE, METHOD)}
TERMINAL = {"ok", "missing_raw", "invalid_raw", "post_error", "no_step", "step_read_error",
            "geometry_error", "timeout", "crash", "interrupted"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fixed_ids():
    ids = SPLIT.read_text().split()
    require(sha(SPLIT) == SPLIT_SHA and len(ids) == len(set(ids)) == 3495, "Fixed split differs")
    return ids


def raw_phase():
    ids = fixed_ids()
    export = HERE / "metadata/export"
    coordinator = read(export / "validation.json")
    require(coordinator.get("passed") is True, "Export coordinator did not pass")
    records, protocols = {}, {}
    shards = sorted(export.glob("shard_*/protocol.json"))
    require(len(shards) == 8, "Expected eight complete export shards")
    for path in shards:
        protocol = read(path)
        index = protocol["shard_index"]
        require(protocol["shard_count"] == 8 and protocol["selected_ids"] == ids[index::8],
                f"Shard population differs: {path}")
        require(protocol["method"] == METHOD and protocol["checkpoint_sha256"] == CHECKPOINT_SHA
                and protocol["split_sha256"] == SPLIT_SHA and protocol["mode"] == "posterior_mean",
                f"Shard model identity differs: {path}")
        audit = protocol["model_audit"]
        require(audit["checkpoint_global_step"] == 30000, "Not the fixed 30k checkpoint")
        model = audit["checkpoint_model_config"]
        require(model["freeze_encoder"] is False and model["loss_weights"]["feature_weight"] == 0,
                "Wrong ablation settings")
        require(protocol["precision"] == "float32" and not protocol["tf32"] and not protocol["autocast"]
                and protocol["seed"] == 42 and protocol["adjacency_threshold"] == .5
                and not protocol["face_deduplication"] and not protocol["augmentation"],
                "Inference contract differs")
        digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        protocols[digest] = protocol
        require(read(path.parent / "completion.json")["source_unchanged"] is True, "Export sources changed")
        selected = []
        for line in (path.parent / "records.jsonl").read_text().splitlines():
            row = json.loads(line)
            sid = row["id"]
            require(sid not in records, f"Duplicate export record {sid}")
            require(row["status"] == "ok" and row["all_metrics_match"] and row["all_counts_match"],
                    f"Failed export/reproduction: {sid}")
            require(row["protocol_sha256"] == digest, f"Record protocol differs: {sid}")
            records[sid] = row
            selected.append(sid)
        require(selected == protocol["selected_ids"], f"Incomplete or reordered export shard {index}")
    require(set(records) == set(ids), "Export record population differs")
    raw_root = HERE / f"raw_predictions/posterior_mean/{METHOD}"
    require({p.parent.name for p in raw_root.glob("*/data.npz")} == set(ids), "Raw file population differs")

    def check(sid):
        row = records[sid]
        path = raw_root / sid / "data.npz"
        digest = sha(path)
        require(digest == row["raw_sha256"], f"Transferred prediction differs from 142 record: {sid}")
        gt_digest = sha(Path("/mnt/d/data/deepcad_v7") / sid / "data.npz")
        with np.load(BASE / "raw_predictions/posterior_mean/shared_curve_uv" / sid / "data.npz",
                     allow_pickle=False) as old, np.load(path, allow_pickle=False) as new:
            require(old["input_source_sha256"].item() == new["input_source_sha256"].item()
                    == row["input_source_sha256"] == gt_digest, f"CAD bytes differ across models/hosts: {sid}")
            require(new["method"].item() == row["method"] == METHOD
                    and new["encoder_checkpoint_sha256"].item() == row["encoder_checkpoint_sha256"] == CHECKPOINT_SHA
                    and new["comparison_mode"].item() == "posterior_mean"
                    and new["protocol_sha256"].item() == row["protocol_sha256"], f"NPZ model metadata differs: {sid}")
            z = new["face_latents"]
            require(z.dtype == np.float32 and z.shape == (row["faces"], 64)
                    and len(old["pred_face"]) == row["faces"], f"Wrong face count/latent width: {sid}")
            latent_digest = hashlib.sha256(str(z.shape).encode() + np.ascontiguousarray(z).tobytes()).hexdigest()
            require(latent_digest == new["input_latent_sha256"].item() == row["input_latent_sha256"],
                    f"Latent SHA differs: {sid}")
            require(new["uv_source_sha256"].item() == row["uv_source_sha256"],
                    f"UV identity differs between NPZ and 142 record: {sid}")
            require(new["pred_face"].shape == (row["faces"], 16, 16, 3)
                    and new["pred_edge"].shape == (row["edges"], 16, 3), f"Geometry shape differs: {sid}")
            for name in new.files:
                value = new[name]
                if np.issubdtype(value.dtype, np.floating):
                    require(np.isfinite(value).all(), f"Nonfinite prediction {sid}/{name}")
        return sid, digest, gt_digest

    with ThreadPoolExecutor(max_workers=8) as pool:
        manifests = list(pool.map(check, ids))
    return {"passed": True, "intended_count": len(ids), "checked_count": len(manifests),
            "export_shards": len(shards), "split_sha256": sha(SPLIT), "checkpoint_sha256": CHECKPOINT_SHA,
            "prediction_and_gt_manifest_sha256": canonical_digest(manifests),
            "checks": ["3495 exact IDs", "eight complete disjoint shards", "fixed 30k full-unfreeze no-feature model",
                       "FP32 posterior mean inference contract", "all previous raw metrics and counts reproduced",
                       "transferred NPZ SHA equals 142 record", "source32/new64/local CAD input SHA identical",
                       "UV SHA in NPZ equals 142 export record", "64-D float32 latent and its SHA", "NPZ metadata and finite geometry"],
            "uv_audit_scope": "187 does not hold the UV dataset; its hash is checked against 142 records, not local bytes. OCC consumes XYZ only."}


def results_phase():
    ids = fixed_ids()
    summary = read(HERE / "failure_analysis/summary.json")
    require(summary["complete"] and summary["population"] == {
        "accepted_validation": 3495, "original_validation": 3505, "unsupported": 10}, "Wrong report population")
    require(summary["checkpoint_sha256"][METHOD] == CHECKPOINT_SHA, "Wrong report checkpoint")
    post_summary = read(HERE / "post/optimized/summary.json")
    raw_audit = read(HERE / "validation/independent_audit.json")["raw"]
    require(raw_audit["passed"], "Independent raw audit has not passed")
    rows, protocols, details = {}, {}, {}
    for role, (root, method) in ROLES.items():
        protocol = read(root / "metadata/post/optimized/protocol.json")
        protocols[role] = protocol
        paths = root / f"metadata/post/optimized/samples/{method}"
        require({p.stem for p in paths.glob("*.json")} == set(ids), f"Incomplete terminal population: {role}")
        rows[role] = {sid: read(paths / f"{sid}.json") for sid in ids}
        digest = canonical_digest(protocol)
        for sid, row in rows[role].items():
            require(row["sample_id"] == sid and row["method"] == method and row["status"] in TERMINAL
                    and row["protocol_fingerprint"] == digest, f"Invalid terminal record: {role}/{sid}")
            require(row["raw_sha256"] == sha(root / f"raw_predictions/posterior_mean/{method}" / sid / "data.npz"),
                    f"Post used different raw input: {role}/{sid}")
            if row.get("native_valid_solid"):
                require(row.get("step_readable") and row.get("native_is_solid_type") and row.get("native_brepcheck_valid"),
                        f"Native-valid flag is inconsistent: {role}/{sid}")
        valid = [sid for sid in ids if rows[role][sid].get("native_valid_solid")]
        measurable = [sid for sid in valid if geometry_ok(rows[role][sid])]
        observed = summary["methods"][role]
        require(observed["valid"] == len(valid) and observed["failed"] == len(ids)-len(valid)
                and observed["native_valid_geometry_count"] == len(measurable)
                and observed["native_valid_geometry_missing"] == len(valid)-len(measurable),
                f"Validity/geometry counts differ: {role}")
        require(math.isclose(observed["valid_fraction"], len(valid)/3495)
                and math.isclose(observed["valid_original_fraction"], len(valid)/3505), f"Denominator error: {role}")
        require(sum(observed["stages"].values()) == 3495-len(valid), f"Failure stages do not partition failures: {role}")
        require(observed["runner_statuses"] == dict(Counter(r["status"] for r in rows[role].values())),
                f"Runner status count differs: {role}")
        counts = post_summary["methods"][method]["counts"]
        require(counts["native_valid_solid"] == len(valid) and counts["terminal_records"] == 3495,
                f"Runner summary differs: {role}")
        details[role] = {"native_valid": len(valid), "geometry_measurable_when_valid": len(measurable),
                         "geometry_missing_when_valid": len(valid)-len(measurable),
                         "native_valid_despite_timeout": sum(rows[role][sid]["status"] == "timeout" for sid in valid)}
    require(details["source32"]["native_valid"] == 3180, "Historical baseline differs")
    contract = ("source", "mode", "data_root", "split_sha256", "intended_count", "original_test_count", "timeout_seconds",
                "timeout_scope", "max_optimize_iter", "adaptive_max_optimize_iter", "drop_num", "use_cuda",
                "threads_per_process", "sampling", "native_validity", "compatibility_validity", "post_algorithm_files", "post_policy")
    require(all(protocols["source32"][key] == protocols[METHOD][key] for key in contract), "Historical post contract differs")
    require(all(sha(REPO / path) == digest for path, digest in protocols[METHOD]["post_algorithm_files"].items()),
            "Production post source changed")
    groups = {key: [] for key in ("both_valid", "rescued", "regressed", "both_failed")}
    for sid in ids:
        old, new = (bool(rows[role][sid].get("native_valid_solid")) for role in ROLES)
        key = "both_valid" if old and new else "rescued" if new else "regressed" if old else "both_failed"
        groups[key].append(sid)
    for key, members in groups.items():
        require(summary["paired"]["counts"][key] == len(members)
                and set(summary["paired"]["ids"][key]) == set(members), f"Paired partition differs: {key}")
    require(summary["paired"]["net_valid_change"] == len(groups["rescued"])-len(groups["regressed"]),
            "Net validity change differs")
    for name, intended in (("common_computable", ids), ("common_native_valid", groups["both_valid"])):
        selected = [sid for sid in intended if all(geometry_ok(rows[role][sid]) for role in ROLES)]
        cohort = summary["paired"]["post_geometry"][name]
        require(cohort["intended_count"] == len(intended) and cohort["computable_count"] == len(selected)
                and cohort["missing_count"] == len(intended)-len(selected) and set(cohort["ids"]) == set(selected),
                f"Matched geometry cohort differs: {name}")
        for metric in ("face_cd", "edge_cd"):
            for role in ROLES:
                values = np.asarray([rows[role][sid][metric] for sid in selected], dtype=float)
                stats = cohort["metrics"][metric][role]
                require(stats["count"] == len(values), f"Geometry metric count differs: {name}/{metric}/{role}")
                if len(values):
                    for statistic, actual in (("mean", values.mean()), ("median", np.median(values)), ("p90", np.quantile(values, .9))):
                        require(np.isclose(stats[statistic], actual, rtol=1e-12, atol=1e-15),
                                f"Geometry statistic differs: {name}/{metric}/{role}/{statistic}")
    with (HERE / "failure_analysis/samples.csv").open() as handle:
        table = list(csv.DictReader(handle))
    require(len(table) == 6990 and len({(r["role"], r["id"]) for r in table}) == 6990, "Analysis CSV population differs")
    for row in table:
        actual = rows[row["role"]][row["id"]]
        require((row["valid"] == "True") == bool(actual.get("native_valid_solid")), "CSV native validity differs")
    completion = read(HERE / "metadata/post_launch/completion.json")
    require(completion["returncode"] == 0 and completion["source_unchanged"]
            and all(completion["source_unchanged"].values()), "Post sources changed or run failed")
    return {"passed": True, "intended_count": 3495, "original_count": 3505, "method_counts": details,
            "paired_counts": {key: len(value) for key, value in groups.items()},
            "common_native_valid_geometry": {key: summary["paired"]["post_geometry"]["common_native_valid"][key]
                                             for key in ("intended_count", "computable_count", "missing_count")},
            "summary_sha256": sha(HERE / "failure_analysis/summary.json"), "csv_sha256": sha(HERE / "failure_analysis/samples.csv"),
            "checks": ["3495 terminal records per method", "raw SHA and protocol fingerprint", "native validity flags",
                       "fixed baseline 3180", "same post algorithms and contract", "3495/3505 denominators",
                       "native validity independent of geometry availability", "four-way paired partition",
                       "exclusive failure stages sum", "common-native-valid and both-measurable cohort",
                       "independent mean/median/P90", "6990 CSV rows and native flags"]}


def geometry_ok(row):
    return row.get("geometry_status") == "ok" and all(
        isinstance(row.get(key), (int, float)) and math.isfinite(row[key]) for key in ("face_cd", "edge_cd"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("raw", "results"))
    args = parser.parse_args()
    path = HERE / "validation/independent_audit.json"
    output = read(path) if path.is_file() else {}
    try:
        result = raw_phase() if args.phase == "raw" else results_phase()
    except Exception as exc:
        result = {"passed": False, "error": repr(exc), "traceback": traceback.format_exc()}
    result.update(completed_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__))
    output[args.phase] = result
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    temporary.replace(path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
