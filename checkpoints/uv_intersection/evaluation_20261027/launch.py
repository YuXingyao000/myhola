"""Run disjoint single-GPU evaluation shards, preserving every process outcome."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sources(output_root, root, checkpoint, launch, outcomes):
    """Record end-of-run identities independently of the shard's start receipts."""
    protocol_path = output_root / "shards/shard_000/protocol.json"
    if not protocol_path.is_file():
        receipt = {"source_unchanged": False, "error": "First shard protocol missing", "exit_codes": outcomes}
    else:
        protocol = json.loads(protocol_path.read_text())
        checks = {name: (root / name).is_file() and sha256(root / name) == digest
                  for name, digest in protocol["source_hashes"].items()}
        checkpoint_digest = sha256(checkpoint)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        receipt = {
            "source_unchanged": (all(checks.values())
                                 and commit == launch["source_git_commit"] == protocol["source_git_commit"]
                                 and checkpoint_digest == protocol["checkpoint_sha256"]),
            "source_checks": checks, "checkpoint_sha256": checkpoint_digest,
            "source_git_commit": commit,
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
            "launcher_sha256": sha256(__file__), "exit_codes": outcomes,
        }
    (output_root / "source_verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt["source_unchanged"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", help="Override evaluate.py's frozen UV 30k checkpoint identity")
    parser.add_argument("--expected-step", type=int, help="Override evaluate.py's default step 30000")
    parser.add_argument("--spec", type=Path, help="Override evaluate.py's original frozen specification")
    parser.add_argument("--split", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    args.output_root.mkdir(parents=True, exist_ok=False)
    gpus = args.gpus.split(",")
    commands, processes = [], []
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "commands": commands,
                "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    try:
        for index, gpu in enumerate(gpus):
            command = [sys.executable, str(here / "evaluate.py"), "--checkpoint", args.checkpoint,
                       "--split", args.split, "--data-root", args.data_root,
                       "--output-root", str(args.output_root), "--shard-index", str(index),
                       "--shard-count", str(len(gpus))]
            for option, value in (("--checkpoint-sha256", args.checkpoint_sha256),
                                  ("--expected-step", args.expected_step), ("--spec", args.spec)):
                if value is not None:
                    command.extend((option, str(value)))
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "2",
                   "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(root)}
            log = (args.output_root / f"shard_{index:03d}.log").open("x")
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append((process, log))
            commands.append({"shard": index, "physical_gpu": gpu, "pid": process.pid, "command": command})
        (args.output_root / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
        outcomes = []
        for index, (process, log) in enumerate(processes):
            outcomes.append({"shard": index, "exit_code": process.wait()})
            log.close()
        (args.output_root / "process_exit_codes.json").write_text(json.dumps(outcomes, indent=2) + "\n")
        unchanged = verify_sources(args.output_root, root, args.checkpoint, manifest, outcomes)
        if any(item["exit_code"] for item in outcomes) or not unchanged:
            raise SystemExit(2)
    finally:
        for _, log in processes:
            log.close()


if __name__ == "__main__":
    main()
