"""Reproduce the frozen comparison with one inference process per GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]
CONFIG = ROOT / "report/current_resolved_config.yaml"
BACKENDS = {
    "hola_1500k": "legacy",
    "current_best_54639": "current",
    "current_last_182130": "current",
    "alignment_150k": "alignment",
    "alignment_best_250k": "alignment",
}


def verify_checkpoint(record: dict) -> None:
    """Verify the full checkpoint, not a name-derived model identity."""
    path = Path(record["path"])
    if path.stat().st_size != record["bytes"]:
        raise ValueError(f"Checkpoint size changed: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != record["sha256"]:
        raise ValueError(f"Checkpoint contents changed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=[str(i) for i in range(8)])
    parser.add_argument("--methods", nargs="+", choices=list(BACKENDS), default=list(BACKENDS))
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or len(set(args.methods)) != len(args.methods):
        parser.error("GPU IDs and methods must be unique")
    checkpoints = json.loads((ROOT / "report/checkpoints.json").read_text())
    logs = ROOT / "metadata/logs"
    logs.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        verify_checkpoint(checkpoints[method])
        workers = []
        for rank, gpu in enumerate(args.gpus):
            command = [sys.executable, str(ROOT / "infer.py"), "--backend", BACKENDS[method],
                       "--method", method, "--checkpoint", checkpoints[method]["path"],
                       "--rank", str(rank), "--shards", str(len(args.gpus))]
            if BACKENDS[method] == "current":
                command.extend(["--config", str(CONFIG)])
            handle = (logs / f"{method}_gpu{gpu}.log").open("a")
            process = subprocess.Popen(command, cwd=REPOSITORY,
                                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                                       stdout=handle, stderr=subprocess.STDOUT)
            workers.append((process, handle))
        # Finish every launched process before reporting any worker failure.
        codes = []
        for process, handle in workers:
            codes.append(process.wait())
            handle.close()
        if any(codes):
            raise RuntimeError(f"Inference worker failed for {method}: {codes}; inspect {logs}")
        print(f"Inference terminal: {method}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "evaluate.py"), "all", "--methods", *args.methods],
                   cwd=REPOSITORY, check=True)
    displayed = [m for m in args.methods if m != "alignment_150k" or "alignment_best_250k" not in args.methods]
    subprocess.run([sys.executable, str(ROOT / "visualize.py"), "--methods", *displayed, "--steps",
                    *(f"{m}={checkpoints[m]['global_step']}" for m in displayed)],
                   cwd=REPOSITORY, check=True)


if __name__ == "__main__":
    main()
