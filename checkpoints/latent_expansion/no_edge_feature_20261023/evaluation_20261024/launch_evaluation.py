"""Launch eight independent fixed-weight validation shards on 142."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=("source32", "no_edge_feature_last64"), default="no_edge_feature_last64")
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Per-shard smoke only")
    parser.add_argument("--split", type=Path, help="Explicit split; a smoke subset is never a full validation result")
    args = parser.parse_args()
    if not 1 <= args.shards <= 8:
        parser.error("--shards must be between 1 and 8 for this eight-GPU launcher")
    here = Path(__file__).resolve().parent
    root = here.parents[3]
    runner = here / "evaluate_ablation.py"
    source = root / "checkpoints/shared_curve_uv/candidate/20261019T064906.049992Z/weights/epoch=0116-step=00042471-val_loss=0.036517.ckpt"
    last = root / "checkpoints/latent_expansion/d64_no_edge_feature/20261023T034042Z/weights/last.ckpt"
    split = args.split or root / "checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt"
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "jobs": []}
    jobs = []
    for shard in range(args.shards):
        command = [sys.executable, str(runner), "--variant", args.variant,
                   "--source-checkpoint", str(source), "--late-checkpoint", str(last),
                   "--data-root", "/mnt/d/data/deepcad_v7", "--uv-root", "/mnt/d/data/deepcad_v7_uv/v1",
                   "--split", str(split.resolve()),
                   "--out", str(args.out / args.variant), "--device", "cuda:0",
                   "--shard-index", str(shard), "--shard-count", str(args.shards)]
        if args.limit is not None:
            command += ["--limit", str(args.limit)]
        log = (args.out / f"{args.variant}_{shard}.log").open("w")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        record = {"variant": args.variant, "shard": shard, "physical_gpu": shard,
                  "pid": process.pid, "command": command, "exit_code": None}
        receipt["jobs"].append(record)
        jobs.append((process, log, record))
    receipt_path = args.out / "launch_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    for process, log, record in jobs:
        record["exit_code"] = process.wait()
        log.close()
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt.update(status="complete" if all(r["exit_code"] == 0 for r in receipt["jobs"]) else "failed",
                   completed_utc=datetime.now(timezone.utc).isoformat())
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
