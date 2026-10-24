"""Run only the fixed OCC raw-export smoke/full stage on 142."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PYTHON = "/mnt/d/anaconda3/envs/torch/bin/python"
CHECKPOINT = ROOT / "checkpoints/latent_expansion/d64_no_edge_feature/20261023T034042Z/weights/last.ckpt"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "full"))
    args = parser.parse_args()
    output = HERE / "smoke" if args.phase == "smoke" else HERE
    if args.phase == "full":
        smoke = json.loads((HERE / "smoke/metadata/export/validation.json").read_text())
        if not smoke["passed"] or smoke["count"] != 8:
            raise ValueError("Eight-item export smoke must pass before full inference")
    metadata = output / "metadata/export"
    metadata.mkdir(parents=True, exist_ok=False)
    shards = 1 if args.phase == "smoke" else 8
    processes, logs, commands = [], [], []
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    started = time.monotonic()
    for shard in range(shards):
        command = [PYTHON, "-u", str(HERE / "export_predictions.py"), "--checkpoint", str(CHECKPOINT),
            "--data-root", "/mnt/d/data/deepcad_v7", "--uv-root", "/mnt/d/data/deepcad_v7_uv/v1",
            "--split", str(ROOT / "checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt"),
            "--output-root", str(output), "--device", f"cuda:{shard}",
            "--shard-index", str(shard), "--shard-count", str(shards)]
        if args.phase == "smoke":
            command += ["--limit", "8"]
        handle = (metadata / f"shard_{shard:03d}.log").open("x")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        processes.append(process); logs.append(handle)
        commands.append({"pid": process.pid, "argv": command})
    write(metadata / "launch.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                                     "phase": args.phase, "processes": commands})
    while any(process.poll() is None for process in processes):
        time.sleep(5)
    for handle in logs:
        handle.close()
    records = []
    for path in sorted(metadata.glob("shard_*/records.jsonl")):
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    expected = (ROOT / "checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt").read_text().split()
    if args.phase == "smoke":
        expected = expected[:8]
    ids = [row["id"] for row in records]
    code_checks = [json.loads(path.read_text())["source_unchanged"] for path in metadata.glob("shard_*/completion.json")]
    result = {"finished_utc": datetime.now(timezone.utc).isoformat(), "count": len(records),
        "expected_count": len(expected), "unique": len(ids) == len(set(ids)),
        "ids_match": set(ids) == set(expected), "status_counts": dict(Counter(row["status"] for row in records)),
        "all_metric_comparisons_match": all(row.get("all_metrics_match", False) for row in records),
        "all_count_comparisons_match": all(row.get("all_counts_match", False) for row in records),
        "source_unchanged": len(code_checks) == shards and all(code_checks),
        "returncodes": [process.returncode for process in processes],
        "elapsed_seconds": time.monotonic()-started,
        "metric_max_abs_difference": {key: max((row.get("metric_comparison", {}).get(key, {}).get("abs_difference", 0) for row in records), default=0)
            for key in ("face_cd_sq", "edge_cd_sq", "face_mae")}}
    result["passed"] = (result["count"] == len(expected) and result["unique"] and result["ids_match"]
        and result["status_counts"] == {"ok": len(expected)} and result["all_metric_comparisons_match"]
        and result["all_count_comparisons_match"] and result["source_unchanged"] and not any(result["returncodes"]))
    write(metadata / "validation.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
