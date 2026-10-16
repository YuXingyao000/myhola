"""Audit completed continuation evaluations without rerunning any predictions."""
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SPLIT = REPO / "checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt"
IDS = SPLIT.read_text().splitlines()
EXPECTED = {
    "last": "040c05a45dd7edae825f2e6dd8e2cb3489a418146f9a7b2122cc54d38070618c",
    "resume_best": "e1e579e7bc6bdf64b3c078d6b5414e33630222de989b180ca8f3eb22b59c6ccf",
}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    destination = ROOT / "validation/main_validation.json"
    require(not destination.exists(), "Refuse to overwrite validation")
    require(len(IDS) == len(set(IDS)) == 2417, "Wrong fixed cohort")
    source = load(ROOT / "validation/source_audit.json")
    require(sha(ROOT / "PLAN.md") == source["plan_sha256"], "Frozen plan changed")
    for record in source["validated_source_files"]:
        require(sha(REPO / record["path"]) == record["sha256"], "Evaluation algorithm changed")
    result = {"status": "passed", "intended_count": len(IDS), "original_count": 2424,
              "source_files_unchanged": len(source["validated_source_files"]), "production_post_unchanged": 11,
              "plan_sha256": source["plan_sha256"], "selections": {}, "jobs": []}
    for selection, digest in EXPECTED.items():
        directory = ROOT / selection
        protocol = load(directory / "metadata/inference/posterior_mean/protocol.json")
        require(protocol["checkpoint_sha256"] == digest == sha(protocol["checkpoint"]), "Wrong evaluated checkpoint")
        require(protocol["input_source"] == "checkpoint_posterior" and protocol["input_contract"] == "same_geometry_source", "Wrong encoder input")
        require(protocol["num_shards"] == 8 and protocol["edge_threshold"] == .5 and not protocol["decoder_gt_topology"] and not protocol["decoder_face_deduplication"], "Changed decoding protocol")
        require(protocol["intended_count"] == len(IDS) and protocol["split_sha256"] == sha(SPLIT) and protocol["encoder_gt_topology"] and protocol["canonical_face_order"], "Wrong cohort or encoder contract")
        numeric = protocol["numeric_settings"]
        require(numeric["dtype"] == "float32" and numeric["seed"] == 42 and not numeric["autocast"] and not numeric["matmul_allow_tf32"] and not numeric["cudnn_allow_tf32"], "Changed numeric protocol")
        statuses = [json.loads(line) for path in sorted((directory / "metadata/inference/posterior_mean").glob("shard-???.jsonl")) for line in path.read_text().splitlines()]
        require(len(statuses) == len(IDS) and {row["model_id"] for row in statuses} == set(IDS), "Wrong inference cohort")
        require(all(row["status"] == "ok" and row["encoder_checkpoint_sha256"] == digest for row in statuses), "Inference failure or wrong encoder")
        raw = load(directory / "raw_metrics/posterior_mean/summary.json")
        require(raw["protocol_valid"] and raw["input_contract"] == "same_geometry_source", "Raw metric contract failed")
        detail = {"checkpoint_sha256": digest, "inference_status_counts": dict(Counter(r["status"] for r in statuses)),
                  "inference_total_object_seconds": sum(r["seconds"] for r in statuses), "raw_protocol_valid": raw["protocol_valid"], "numeric_settings": numeric, "post": {}}
        inference_jobs = load(ROOT / "metadata/orchestration" / f"inference-{selection}_jobs.json")
        starts = [datetime.datetime.fromisoformat(r["started_utc"]) for r in inference_jobs]
        ends = [datetime.datetime.fromisoformat(r["completed_utc"]) for r in inference_jobs]
        detail["inference_resources"] = {"gpu_count": 8, "shards": 8, "supervisor_wall_seconds": (max(ends) - min(starts)).total_seconds(),
                                         "note": "Supervisor completion includes up to 30 seconds of polling delay."}
        for mode in ("fit_only", "optimized"):
            post = load(directory / "post" / mode / "summary.json")
            recorded = post["protocol"]
            require(recorded["timeout_seconds"] == 180 and recorded["drop_num"] == 0 and recorded["max_optimize_iter"] == 200 and recorded["threads_per_process"] == 1 and not recorded["use_cuda"], "Changed post budget")
            counts = post["methods"]["encoder_edge_value"]["counts"]
            require(counts["terminal_records"] == len(IDS), "Incomplete post run")
            entries = [load(path) for path in sorted((directory / "metadata/post" / mode / "samples/encoder_edge_value").glob("*.json"))]
            require(len(entries) == len(IDS) and {r["sample_id"] for r in entries} == set(IDS), "Wrong post per-object cohort")
            detail["post"][mode] = {"counts": counts, "status_counts": dict(Counter(r["status"] for r in entries)),
                                   "timeouts": [{k: r.get(k) for k in ("sample_id", "stage", "native_valid_solid", "geometry_status", "wall_seconds")} for r in entries if r["status"] == "timeout"],
                                   "total_worker_wall_seconds": sum(r.get("wall_seconds", 0) for r in entries)}
            invocation = load(directory / "metadata/post" / mode / "invocation.json")
            start = datetime.datetime.fromisoformat(invocation["time_utc"])
            end = datetime.datetime.fromtimestamp((directory / "post" / mode / "summary.json").stat().st_mtime, datetime.timezone.utc)
            detail["post"][mode]["resources"] = {"workers": invocation["workers"], "threads_per_worker": 1,
                "wall_seconds": (end - start).total_seconds(), "timing_definition": "Saved dispatch invocation time to final summary write time; no rerun or summary-only invocation.",
                "started_utc": start.isoformat(), "summary_written_utc": end.isoformat()}
        result["selections"][selection] = detail
    for path in sorted((ROOT / "metadata/orchestration").glob("*_jobs.json")):
        for job in load(path):
            require(job["exit_code"] == 0, "Orchestration job did not succeed")
            start = datetime.datetime.fromisoformat(job["started_utc"])
            end = datetime.datetime.fromisoformat(job["completed_utc"])
            result["jobs"].append({"name": job["name"], "exit_code": job["exit_code"], "wall_seconds": (end - start).total_seconds(), "manifest": str(path)})
    diagnostic = load(ROOT / "diagnostics.json")
    require(diagnostic["intended_count"] == 2417 and len(diagnostic["methods"]) == 5, "Incomplete final comparison")
    manifest = diagnostic["input_hash_manifest"]
    require(sha(manifest["path"]) == manifest["sha256"], "Input hash manifest changed")
    require(sha(ROOT / "owner_per_object.csv") == diagnostic["owner_csv_sha256"], "Owner table changed")
    result["full_provenance_audit"] = {"input_files": manifest["file_count"], "diagnostics_sha256": sha(ROOT / "diagnostics.json"),
                                      "owner_table_sha256": diagnostic["owner_csv_sha256"], "checks": diagnostic["checks"]}
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "output": str(destination), "input_files": manifest["file_count"]}))


if __name__ == "__main__":
    main()
