"""Frozen own-posterior evaluation; no training or archived-latent replay."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PYTHON = "/mnt/d/anaconda3/envs/torch/bin/python"
SPLIT = REPO / "checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt"
REFERENCE = REPO / "checkpoints/intersection_refinement/curve_conditioning/evaluation_20261014"
LATENT_REFERENCE = Path("/mnt/d/python/checkpoints/joint_boundary/comparison_20261011")
SELECTIONS = {
    "candidate": {
        "method": "encoder_edge_value",
        # Same evaluated bytes, preserved before continuation can rotate top-k files.
        # The original invocation metadata retains the original checkpoint path.
        "checkpoint": REPO / "checkpoints/encoder_edge_value/candidate/20261015T073112.923985Z/weights/evaluated_snapshots/d9e911e23a5dfc0e45fb90735a806706c9c89329898f0873df06927c8d2fb62d.ckpt",
        "sha256": "d9e911e23a5dfc0e45fb90735a806706c9c89329898f0873df06927c8d2fb62d",
    },
    "control": {
        "method": "encoder_gat_control",
        "checkpoint": REPO / "checkpoints/encoder_edge_value/control/20261015T073112.923985Z/weights/epoch=0002-step=00001089-val_loss=0.040526.ckpt",
        "sha256": "f1de7f11fee749e51921e0fb45d32ae7b6b7d7899316b248560bab4501d6c279",
    },
}
ENV = {**os.environ, **{name: "1" for name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
ORCHESTRATION = ROOT / "metadata/orchestration"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(arm):
    selection = SELECTIONS[arm]
    if sha256(selection["checkpoint"]) != selection["sha256"]:
        raise ValueError(f"Selected {arm} checkpoint changed")
    pair_root = ROOT / arm
    target = pair_root / "raw_predictions/posterior_mean/conditioned"
    source = REFERENCE / "raw_predictions/posterior_mean/conditioned"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError(f"Historical baseline link points elsewhere: {target}")
    else:
        target.symlink_to(source, target_is_directory=True)
    return pair_root


def run_jobs(label, commands):
    ORCHESTRATION.mkdir(parents=True, exist_ok=True)
    manifest = ORCHESTRATION / (label + "_jobs.json")
    if manifest.exists():
        raise FileExistsError(manifest)
    jobs = []
    for name, command in commands:
        log = ORCHESTRATION / (name + ".log")
        handle = log.open("x")
        process = subprocess.Popen(command, cwd=REPO, env=ENV, stdout=handle, stderr=subprocess.STDOUT)
        record = {"name": name, "command": command, "log": str(log),
                  "pid": process.pid, "started_utc": now()}
        jobs.append((process, handle, record))
    manifest.write_text(json.dumps([record for _, _, record in jobs], indent=2) + "\n")
    while any(process.poll() is None for process, _, _ in jobs):
        print(json.dumps({"phase": label, "utc": now(),
                          "running": sum(process.poll() is None for process, _, _ in jobs)}), flush=True)
        time.sleep(30)
    for process, handle, record in jobs:
        handle.close()
        record.update(exit_code=process.returncode, completed_utc=now())
    manifest.write_text(json.dumps([record for _, _, record in jobs], indent=2) + "\n")
    if any(process.returncode for process, _, _ in jobs):
        raise RuntimeError(f"{label} failed; inspect recorded jobs, no automatic retry")


def inference(arm, smoke=False):
    selection = SELECTIONS[arm]
    pair_root = prepare(arm)
    base = [PYTHON, "-m", "tools.intersection_evaluate_inference",
            "--method", selection["method"], "--checkpoint", str(selection["checkpoint"]),
            "--legacy-checkpoint", "/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt",
            "--reference-root", str(LATENT_REFERENCE), "--data-root", "/mnt/d/data/deepcad_v7",
            "--split-list", str(SPLIT), "--mode", "posterior_mean",
            "--input-source", "checkpoint_posterior", "--output-root",
            str(pair_root / "validation/inference_smoke" if smoke else pair_root)]
    return [(f'{"smoke" if smoke else "inference"}-{arm}-{shard}', base + [
        "--num-shards", str(1209 if smoke else 8), "--shard-index", str(shard),
        "--device", f"cuda:{shard}"]) for shard in range(1 if smoke else 8)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "inference", "metrics", "post"))
    parser.add_argument("--arm", choices=("both", "candidate", "control"), default="both")
    parser.add_argument("--post-mode", choices=("both", "fit_only", "optimized"), default="both")
    args = parser.parse_args()
    arms = list(SELECTIONS) if args.arm == "both" else [args.arm]
    if args.phase in {"smoke", "inference"}:
        # Both arms use all eight GPUs in sequence, never sixteen workers/GPU overlap.
        for arm in arms:
            run_jobs(f"{args.phase}-{arm}", inference(arm, smoke=args.phase == "smoke"))
        return
    commands = []
    for arm in arms:
        pair_root = prepare(arm)
        method = SELECTIONS[arm]["method"]
        common = ["--baseline-method", "conditioned", "--candidate-method", method,
                  "--comparison-root", str(pair_root), "--data-root", "/mnt/d/data/deepcad_v7",
                  "--split-list", str(SPLIT)]
        if args.phase == "metrics":
            commands.append((f"metrics-{arm}", [PYTHON, "-m", "tools.vae_compare_metrics", *common,
                             "--mode", "posterior_mean", "--input-contract", "same_geometry_source",
                             "--workers", "16"]))
        else:
            modes = ("fit_only", "optimized") if args.post_mode == "both" else (args.post_mode,)
            for mode in modes:
                commands.append((f"post-{arm}-{mode}", [PYTHON, "-m", "tools.vae_compare_post", *common,
                                 "--methods", method, "--reference-root", str(REFERENCE),
                                 "--mode", mode, "--workers", "32", "--timeout", "180",
                                 "--max-optimize-iter", "200", "--original-test-count", "2424"]))
    label = f"{args.phase}-{args.arm}" + (f"-{args.post_mode}" if args.phase == "post" else "")
    run_jobs(label, commands)


if __name__ == "__main__":
    main()
