"""Run the frozen paired-decoder / post-policy experiment without changing algorithms."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PYTHON = "/mnt/d/anaconda3/envs/torch/bin/python"
LEGACY = Path("/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt")
CANDIDATE = Path("/mnt/d/python/checkpoints/intersection_refinement/curve_conditioning/20261013T083729.740439Z/weights/epoch=0157-step=00057354-val_loss=0.038204.ckpt")
SPLITS = {"validation": ("val", 3495, 3505), "test": ("test", 2417, 2424)}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def split_file(split):
    prefix, count, _ = SPLITS[split]
    path = REPO / f"checkpoints/joint_boundary/validation/{prefix}_contract_audit/accepted.txt"
    ids = path.read_text().split()
    assert len(ids) == len(set(ids)) == count
    return path


def supervise(stage, commands, sources, extra):
    meta = ROOT / "metadata" / stage
    meta.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": str(REPO)}
    frozen = {str(p.relative_to(REPO)): sha(p) for p in sources}
    for path in sources:
        dest = meta / "source" / path.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    launch = {"started_utc": datetime.now(timezone.utc).isoformat(), "supervisor_pid": os.getpid(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
              "plan_sha256": sha(ROOT / "PLAN.md"), "source_sha256": frozen,
              "processes": [], **extra}
    (meta / "git.diff").write_text(subprocess.check_output(["git", "diff", "--binary"], text=True))
    children, handles = [], []
    start = time.monotonic()
    try:
        for label, argv in commands:
            log = meta / f"{label}.log"
            handle = log.open("x")
            handles.append(handle)
            child = subprocess.Popen(argv, cwd=REPO, env=env, stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            launch["processes"].append({"label": label, "pid": child.pid, "argv": argv,
                                         "log": str(log), "start_ticks": Path(f"/proc/{child.pid}/stat").read_text().split()[21]})
        write(meta / "launch.json", launch)
        print(json.dumps({"stage": stage, "started": [p.pid for p in children]}), flush=True)
        while any(p.poll() is None for p in children):
            time.sleep(10)
            print(json.dumps({"stage": stage, "elapsed_seconds": round(time.monotonic()-start, 1),
                              "exit_codes": {label: p.poll() for (label, _), p in zip(commands, children)}}), flush=True)
    except BaseException:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        raise
    finally:
        for handle in handles:
            handle.close()
    result = {"finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic()-start,
              "returncodes": {label: p.returncode for (label, _), p in zip(commands, children)},
              "source_unchanged": {path: sha(REPO/path) == digest for path, digest in frozen.items()}}
    write(meta / "completion.json", result)
    print(json.dumps(result), flush=True)
    if any(p.returncode != 0 for p in children) or not all(result["source_unchanged"].values()):
        raise RuntimeError(f"{stage} failed; inspect recorded logs")


def inference():
    sources = [ROOT / "infer_pairs.py", ROOT / "run_ablation.py",
               *[REPO/p for p in ("src/brepnet/models/vae.py", "src/brepnet/models/blocks.py",
                  "tools/intersection_evaluate_inference.py", "tools/vae_compare_inference.py")]]
    for split in SPLITS:
        commands = []
        for shard in range(8):
            argv = [PYTHON, "-u", str(ROOT / "infer_pairs.py"),
                    "--legacy-checkpoint", str(LEGACY), "--checkpoint", str(CANDIDATE),
                    "--data-root", "/mnt/d/data/deepcad_v7", "--split-list", str(split_file(split)),
                    "--reference-root", str(ROOT.parent/f"{split}_20261016"),
                    "--output-root", str(ROOT/split), "--num-shards", "8", "--shard-index", str(shard),
                    "--device", f"cuda:{shard}"]
            commands.append((f"shard-{shard:03d}", argv))
        supervise(f"inference_{split}", commands, sources,
                  {"split_sha256": sha(split_file(split)), "legacy_sha256": sha(LEGACY),
                   "candidate_sha256": sha(CANDIDATE)})


def post():
    import numpy as np
    sources = list((REPO/"src/brepnet/post").glob("*.py")) + [REPO/"tools/vae_compare_post.py", ROOT/"run_ablation.py"]
    previous = json.loads((ROOT.parent/"test_20261016/metadata/post_launch/launch.json").read_text())["source_sha256"]
    for path, digest in previous.items():
        assert sha(REPO/path) == digest, f"Frozen post source changed: {path}"
    commands, manifests = [], {}
    for split in SPLITS:
        ids = split_file(split).read_text().split()
        raw = ROOT/split/"raw_predictions/posterior_mean"
        records = []
        for sample_id in ids:
            hashes, latent_hashes = {}, []
            for method in ("legacy", "conditioned"):
                path = raw/method/sample_id/"data.npz"
                hashes[method] = sha(path)
                with np.load(path, allow_pickle=False) as data:
                    latent_hashes.append(str(data["input_latent_sha256"].item()))
            assert latent_hashes[0] == latent_hashes[1], sample_id
            records.append({"model_id": sample_id, "raw_sha256": hashes, "latent_sha256": latent_hashes[0]})
        manifest = ROOT/split/"metadata/raw_manifest.json"
        write(manifest, {"records": records, "intended_count": len(ids)})
        manifests[split] = sha(manifest)
        cache = ROOT/split/"metadata/gt_step_samples"
        cache.mkdir(parents=True, exist_ok=True)
        for policy in ("legacy", "recover_v1"):
            variant = ROOT/split/"post_variants"/policy
            parent = variant/"raw_predictions/posterior_mean"
            parent.mkdir(parents=True, exist_ok=False)
            for method in ("legacy", "conditioned"):
                (parent/method).symlink_to(raw/method, target_is_directory=True)
            (variant/"metadata").mkdir()
            (variant/"metadata/gt_step_samples").symlink_to(cache, target_is_directory=True)
            argv = [PYTHON, "-u", "-m", "tools.vae_compare_post", "--comparison-root", str(variant),
                    "--data-root", "/mnt/d/data/deepcad_v7", "--split-list", str(split_file(split)),
                    "--baseline-method", "legacy", "--candidate-method", "conditioned",
                    "--methods", "legacy", "conditioned", "--mode", "optimized", "--post-policy", policy,
                    "--workers", "32", "--timeout", "180", "--max-optimize-iter", "200",
                    "--original-test-count", str(SPLITS[split][2])]
            commands.append((f"{split}_{policy}", argv))
    supervise("post", commands, sources, {"raw_manifests": manifests, "workers_total": 128,
                                         "per_object_budget_seconds": 180})


def raw_metrics():
    import csv
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    sys.path.insert(0, str(REPO))
    from tools.vae_compare_metrics import evaluate_sample, summarize
    for split, (_, count, original) in SPLITS.items():
        output = ROOT/split/"raw_metrics/posterior_mean"
        output.mkdir(parents=True, exist_ok=False)
        ids = split_file(split).read_text().split()
        evaluate = partial(evaluate_sample, data_root=Path("/mnt/d/data/deepcad_v7"),
                           prediction_root=ROOT/split/"raw_predictions/posterior_mean",
                           mode="posterior_mean", candidate_method="conditioned", baseline_method="legacy",
                           input_contract="identical_latents")
        rows = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index, pair in enumerate(pool.map(evaluate, ids), 1):
                rows.extend(pair)
                if index % 500 == 0:
                    print(json.dumps({"split": split, "raw_metrics": index, "count": count}), flush=True)
        summary = summarize(rows, ids, mode="posterior_mean", original_count=original,
                            candidate_method="conditioned", baseline_method="legacy", input_contract="identical_latents")
        assert summary["protocol_valid"]
        summary.update(split_sha256=sha(split_file(split)),
                       metric_source_sha256=sha(REPO/"tools/vae_compare_metrics.py"),
                       inference_protocol_sha256=sha(ROOT/split/"metadata/inference/posterior_mean/protocol.json"),
                       coordinate_frame="canonical; no alignment or scaling", workers=4)
        columns = sorted(set().union(*(row.keys() for row in rows)))
        with (output/"per_object.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        write(output/"summary.json", summary)
        print(json.dumps({"split": split, "protocol_valid": True, "raw_rows": len(rows)}), flush=True)


if __name__ == "__main__":
    os.chdir(REPO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inference", "post", "raw_metrics"))
    args = parser.parse_args()
    {"inference": inference, "post": post, "raw_metrics": raw_metrics}[args.stage]()
