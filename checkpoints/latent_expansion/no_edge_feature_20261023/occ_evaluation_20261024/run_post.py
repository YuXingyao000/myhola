"""Run the frozen 2026-10-20 OCC protocol with explicit new model identities."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
REFERENCE = REPO / "checkpoints/shared_curve_uv/evaluation_20261020/best"
SPLIT = REPO / "checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt"
SPLIT_SHA = "ae607bfe5499e919d1e68a5110c0c211daf1158aedbe57a4756eda99f6f3f0ed"
sys.path.insert(0, str(REPO))
from tools import vae_compare_post as post


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else post.sha256_file(Path(path))


def link_directory(target, link):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.resolve() != target.resolve():
            raise ValueError(f"Existing link differs: {link}")
    else:
        link.symlink_to(target, target_is_directory=True)


def prepare():
    ids = SPLIT.read_text().split()
    if sha(SPLIT) != SPLIT_SHA or len(ids) != 3495 or len(set(ids)) != 3495:
        raise ValueError("Frozen validation split differs")
    previous = json.loads((REFERENCE / "metadata/post/optimized/protocol.json").read_text())
    files = {str(path.relative_to(REPO)): sha(path) for path in sorted((REPO / "src/brepnet/post").glob("*.py"))}
    if files != previous["post_algorithm_files"] or len(files) != 11:
        raise ValueError("Production OCC algorithms differ from the frozen baseline")
    for suffix in ("raw_predictions/posterior_mean/shared_curve_uv", "post/optimized/shared_curve_uv"):
        link_directory(REFERENCE / suffix, HERE / suffix)
    cache = HERE / "metadata/gt_step_samples"
    cache.mkdir(parents=True, exist_ok=True)
    count = 0
    for source in (REFERENCE / "metadata/gt_step_samples").glob("*.npz"):
        target = cache / source.name
        if not target.exists():
            os.link(source, target)
        elif sha(target) != sha(source):
            raise ValueError(f"Existing GT sample cache differs: {target}")
        count += 1
    candidate = HERE / "raw_predictions/posterior_mean/no_edge_feature64"
    found = {path.parent.name for path in candidate.glob("*/data.npz")}
    result = {"split_sha256": sha(SPLIT), "intended": len(ids), "baseline_root": str(REFERENCE),
              "post_algorithm_files": files, "reused_gt_cache_entries": count,
              "candidate_raw_count": len(found), "missing_ids": sorted(set(ids) - found),
              "unexpected_ids": sorted(found - set(ids)), "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    post.atomic_json(HERE / "validation/post_preflight.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "summary"))
    args = parser.parse_args()
    os.chdir(REPO)
    preflight = prepare()
    if args.phase == "prepare":
        print(json.dumps({key: value for key, value in preflight.items() if key not in ("missing_ids", "post_algorithm_files")}, indent=2))
        return
    if preflight["missing_ids"] or preflight["unexpected_ids"]:
        raise ValueError("Full candidate raw export must be complete before OCC")
    export_validation = json.loads((HERE / "validation/export_validation.json").read_text())
    if not export_validation.get("passed") or export_validation.get("count") != 3495:
        raise ValueError("Full exported geometry must reproduce the previous mean evaluation")
    post.BASELINE_METHODS = (*post.BASELINE_METHODS, "shared_curve_uv")
    post.CANDIDATE_METHODS = (*post.CANDIDATE_METHODS, "no_edge_feature64")
    argv = [str(Path(__file__)), "--comparison-root", str(HERE), "--data-root", "/mnt/d/data/deepcad_v7",
            "--split-list", str(SPLIT), "--baseline-method", "shared_curve_uv",
            "--candidate-method", "no_edge_feature64", "--methods", "no_edge_feature64",
            "--reference-root", str(REFERENCE), "--mode", "optimized", "--post-policy", "recover_v1",
            "--workers", "32", "--timeout", "180", "--max-optimize-iter", "200",
            "--original-test-count", "3505"]
    if args.phase == "summary":
        argv.append("--summary-only")
    metadata = HERE / "metadata" / ("post_launch" if args.phase == "run" else "post_summary")
    metadata.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), HERE / "SPEC.md", REPO / "tools/vae_compare_post.py"]
    sources += sorted((REPO / "src/brepnet/post").glob("*.py"))
    hashes = {str(path.relative_to(REPO)): sha(path) for path in sources}
    for flag, name in ((["diff", "--binary"], "git.diff"), (["status", "--short"], "git_status.txt")):
        (metadata / name).write_text(subprocess.check_output(["git", *flag], cwd=REPO, text=True))
    import OCC
    post.atomic_json(metadata / "launch.json", {
        "argv": argv, "source_sha256": hashes, "python": sys.version, "pythonocc": OCC.VERSION,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "note": "Only method identity enums extended; post algorithms and sampling unchanged."})
    start = time.monotonic()
    sys.argv = argv
    code = post.main()
    unchanged = {name: sha(REPO / name) == value for name, value in hashes.items()}
    post.atomic_json(metadata / "completion.json", {"returncode": code, "elapsed_seconds": time.monotonic() - start,
        "source_unchanged": unchanged, "finished_utc": datetime.now(timezone.utc).isoformat()})
    if code or not all(unchanged.values()):
        raise RuntimeError("Post run failed or source changed during execution")


if __name__ == "__main__":
    main()
