"""Run disjoint single-GPU evaluation shards, preserving every process outcome."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    args.output_root.mkdir(parents=True, exist_ok=False)
    gpus = args.gpus.split(",")
    commands, processes = [], []
    try:
        for index, gpu in enumerate(gpus):
            command = [sys.executable, str(here / "evaluate.py"), "--checkpoint", args.checkpoint,
                       "--split", args.split, "--data-root", args.data_root,
                       "--output-root", str(args.output_root), "--shard-index", str(index),
                       "--shard-count", str(len(gpus))]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "2",
                   "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(root)}
            log = (args.output_root / f"shard_{index:03d}.log").open("x")
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append((process, log))
            commands.append({"shard": index, "physical_gpu": gpu, "pid": process.pid, "command": command})
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "commands": commands,
                    "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
        (args.output_root / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
        outcomes = []
        for index, (process, log) in enumerate(processes):
            outcomes.append({"shard": index, "exit_code": process.wait()})
            log.close()
        (args.output_root / "process_exit_codes.json").write_text(json.dumps(outcomes, indent=2) + "\n")
        if any(item["exit_code"] for item in outcomes):
            raise SystemExit(2)
    finally:
        for _, log in processes:
            log.close()


if __name__ == "__main__":
    main()
