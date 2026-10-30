"""Rescore saved K-candidate SVR pools and select observation-aware DPO pairs.

CPU only: no diffusion sampling, VAE decoding, or OCC reconstruction is repeated.
Run from the repository root; existing runs require explicit ``--resume``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version as package_version
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

from tools.vae_compare_post import atomic_json, fingerprint, read_json, sha256_file
from src.brepnet.data.preference_observation_selection import SelectionConfig, select_pair_with_report


VERSION = "svr_observation_preference_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS = {"train": "pairs.jsonl", "validation": "validation_pairs.jsonl"}


def verified_protocol(path: Path) -> dict:
    protocol = read_json(path)
    expected = protocol.get("fingerprint")
    if not expected or expected != fingerprint({k: v for k, v in protocol.items() if k != "fingerprint"}):
        raise ValueError(f"Protocol fingerprint mismatch: {path}")
    return protocol


def validate_splits(splits: dict) -> None:
    for split in MANIFESTS:
        ids = splits.get(split)
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Expected nonempty distinct {split} identities")
        if any(not isinstance(x, str) or not x or x.strip() != x or
               Path(x).name != x or x in {".", ".."} or "\\" in x for x in ids):
            raise ValueError(f"Unsafe identity in {split}")
    if set(splits["train"]) & set(splits["validation"]):
        raise ValueError("Source training and validation identities overlap")


def build_protocol(args: argparse.Namespace) -> dict:
    source = args.source_run.resolve()
    source_path = source / "metadata/protocol.json"
    old = verified_protocol(source_path)
    if old["version"] != "svr_preference_generation_v1":
        raise ValueError("Only the original SVR generation pool is supported")
    if read_json(source / "status.json").get("status") != "complete":
        raise ValueError("Source generation must be complete")
    validate_splits(old["splits"])
    if old["candidates"] != 8:
        raise ValueError("This frozen protocol requires the original K=8 pool")
    splits = {name: (ids[:args.limit] if args.limit else ids) if name in args.splits else []
              for name, ids in old["splits"].items() if name in MANIFESTS}
    sources = [
        "src/brepnet/data/preference_relabel.py",
        "src/brepnet/data/preference_observation_worker.py",
        "src/brepnet/data/preference_observation_selection.py",
        "src/brepnet/data/extract_visibility.py", "src/brepnet/data/rotations.py",
        "src/brepnet/diffusion/image_loading.py", "src/brepnet/eval/metrics/observation.py",
        "shared/occ_utils.py", "shared/common_utils.py", "tools/vae_compare_post.py",
    ]
    from OCC import VERSION as occ_version
    protocol = {
        "version": VERSION, "source_run": str(source),
        "source_protocol_fingerprint": old["fingerprint"],
        "source_protocol_sha256": sha256_file(source_path),
        "source_status_sha256": sha256_file(source / "status.json"),
        "source_manifest_sha256": {name: sha256_file(source / name) for name in MANIFESTS.values()},
        "checkpoint_sha256": old["checkpoint_sha256"],
        "condition_data_root": old["condition_data_root"], "raw_data_root": old["raw_data_root"],
        "gt_cache_root": str(source / "gt_cache"), "candidates": old["candidates"],
        "splits": splits, "partial": splits != old["splits"], "seed": args.seed,
        "observation": {"resolution": args.resolution, "mesh_deflection": args.mesh_deflection,
                        "mask_threshold": args.mask_threshold, "view_id": 0, "image_key": "svr_imgs"},
        "selection": asdict(SelectionConfig()), "image_timeout_seconds": args.image_timeout,
        "source_sha256": {name: sha256_file(REPO_ROOT / name) for name in sources},
        "runtime": {"python": sys.version, "pythonocc": occ_version,
                    "packages": {name: package_version(name) for name in ("numpy", "scipy", "Pillow")}},
    }
    protocol["fingerprint"] = fingerprint(protocol)
    return protocol


def prepare_run(output: Path, protocol: dict, args: argparse.Namespace) -> None:
    path = output / "metadata/protocol.json"
    if args.resume:
        if verified_protocol(path) != protocol:
            raise ValueError("Resume requires unchanged source data, code, metric settings and selection protocol")
    else:
        atomic_json(path, protocol)
        for name, ids in protocol["splits"].items():
            (output / f"metadata/{name}.txt").write_text("".join(f"{x}\n" for x in ids))
        for name in protocol["source_sha256"]:
            target = output / "metadata/source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / name, target)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    launch = output / "metadata/launches" / stamp
    launch.mkdir(parents=True)
    git = lambda *parts: subprocess.check_output(["git", *parts], cwd=REPO_ROOT, timeout=60)
    atomic_json(launch / "launch.json", {
        "command": [sys.executable, *sys.argv], "git_commit": git("rev-parse", "HEAD").decode().strip(),
        "created_utc": stamp, "workers": args.workers, "cuda_visible_devices": "",
        "protocol_fingerprint": protocol["fingerprint"], "resume": args.resume,
    })
    (launch / "working_tree.diff").write_bytes(git("diff", "--", "src/brepnet", "configs", "checkpoints/diffusion/fidelity_uncertainty"))


def pair_change(previous: dict | None, current: dict | None) -> str:
    if not previous:
        return "added" if current else "neither"
    if not current:
        return "abstained"
    old = (previous["winner_id"], previous["loser_id"])
    new = (current["winner_id"], current["loser_id"])
    return "retained" if new == old else "flipped" if new == old[::-1] else "replaced"


def read_old_pairs(source: Path, splits: dict) -> dict:
    result = {}
    for split, name in MANIFESTS.items():
        ids = set(splits[split])
        with (source / name).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["image_id"] not in ids:
                    continue
                key = (split, row["image_id"])
                if key in result:
                    raise ValueError(f"Duplicate source preference: {key}")
                result[key] = {name: row[name] for name in
                               ("image_id", "winner_id", "loser_id", "pair_type")}
    return result


def _child(source: Path, output: Path, split: str, image_id: str, config: dict) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    folder = output / "cases" / split / image_id
    with (folder / "worker.log").open("a", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            from src.brepnet.data.preference_observation_worker import evaluate_image
            evaluate_image(source, output, split, image_id, config)
        except Exception as error:
            atomic_json(folder / "error.json", {"error": f"{type(error).__name__}: {error}",
                                                 "traceback": traceback.format_exc()})
            raise


def _bounded(source: Path, output: Path, split: str, image_id: str, config: dict, previous: dict | None) -> dict:
    config = {**config, "attempt_id": uuid.uuid4().hex}
    folder = output / "cases" / split / image_id
    folder.mkdir(parents=True, exist_ok=True)
    error_path = folder / "error.json"
    error_path.unlink(missing_ok=True)
    started = time.monotonic()
    process = mp.get_context("fork").Process(target=_child, args=(source, output, split, image_id, config))
    process.start()
    expired = False
    try:
        process.join(config["image_timeout_seconds"])
        expired = process.is_alive()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(3)
            if process.is_alive():
                process.kill()
                process.join()
    if error_path.exists():
        raise RuntimeError(f"{split}/{image_id}: {read_json(error_path)['error']}")
    row = read_json(folder / "metrics.json")
    candidates = row.get("candidates", [])
    if (row.get("attempt_id") != config["attempt_id"] or
            row.get("protocol_fingerprint") != config["fingerprint"] or
            row.get("split") != split or row.get("image_id") != image_id or
            [item.get("candidate_id") for item in candidates] !=
            [f"c{index:02d}" for index in range(config["candidates"])] or
            any(item.get("image_id") != image_id for item in candidates)):
        raise RuntimeError(f"{split}/{image_id}: worker stopped before verifying source inputs (exit={process.exitcode})")
    if expired or process.exitcode != 0:
        status = "timeout" if expired else "crash"
        row["status"] = status
        for candidate in row["candidates"]:
            if candidate.get("observation", {}).get("status") in {None, "running", "pending"}:
                candidate["observation"] = {"status": status, "error": "CAD worker stopped before measurement completed"}
                atomic_json(output / "candidates" / split / image_id / candidate["candidate_id"] / "observation.json", candidate)
    row.update(worker_exit_code=process.exitcode, elapsed_seconds=time.monotonic() - started)
    atomic_json(folder / "metrics.json", row)
    pair, report = select_pair_with_report(row["candidates"], seed=config["seed"],
                                           config=SelectionConfig(**config["selection"]))
    if pair is not None:
        lookup = {item["candidate_id"]: item for item in row["candidates"]}
        winner, loser = lookup[pair["winner_id"]], lookup[pair["loser_id"]]
        pair.update(split=split, winner_path=winner["latent_path"], loser_path=loser["latent_path"],
                    protocol_fingerprint=config["fingerprint"],
                    metrics_evidence={role: {k: candidate.get(k) for k in
                                            ("validity", "raw", "observation", "source_fingerprint")}
                                      for role, candidate in (("winner", winner), ("loser", loser))})
    change = pair_change(previous, pair)
    atomic_json(folder / "selection.json", {"image_id": image_id, "split": split,
                "protocol_fingerprint": config["fingerprint"], "pair": pair, "report": report,
                "old_pair": previous, "change": change})
    counts = Counter({"images": 1, "candidates": len(row["candidates"]), "pairs": int(pair is not None),
                      f"change_{change}": 1, f"image_status_{row['status']}": 1})
    if pair:
        counts[f"pairs_{pair['pair_type']}"] += 1
    for candidate in row["candidates"]:
        counts[f"validity_{candidate.get('validity', 'unknown')}"] += 1
        counts[f"observation_{candidate.get('observation', {}).get('status', 'missing')}"] += 1
    return {"split": split, "image_id": image_id, "counts": dict(counts)}


def _stop_worker(signum, frame) -> None:
    raise SystemExit(128 + signum)


def _worker_loop(tasks, results, source: Path, output: Path, config: dict) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _stop_worker)
    # Import once per CPU worker; forked CAD children never inherit a CUDA context.
    from src.brepnet.data import preference_observation_worker  # noqa: F401
    from src.brepnet.data import extract_visibility  # noqa: F401
    while True:
        task = tasks.get()
        if task is None:
            return
        split, image_id, previous = task
        try:
            results.put({"result": _bounded(source, output, split, image_id, config, previous)})
        except Exception:
            results.put({"error": traceback.format_exc()})
            return


def run_jobs(source: Path, output: Path, protocol: dict, workers: int, old_pairs: dict) -> dict:
    jobs = [(split, image_id, old_pairs.get((split, image_id)))
            for split in ("validation", "train") for image_id in protocol["splits"][split]]
    config = {key: value for key, value in protocol.items()
              if key not in {"splits", "source_sha256", "source_manifest_sha256", "runtime"}}
    workers = min(workers, len(jobs))
    context = mp.get_context("spawn")
    tasks, results = context.Queue(maxsize=workers * 2), context.Queue()
    processes = [context.Process(target=_worker_loop, args=(tasks, results, source, output, config))
                 for _ in range(workers)]
    counts = {name: Counter() for name in MANIFESTS}
    started = time.monotonic()
    completed = submitted = 0

    def write_status(status: str) -> None:
        atomic_json(output / "status.json", {"status": status, "partial": protocol["partial"],
            "protocol_fingerprint": protocol["fingerprint"], "completed": completed, "total": len(jobs),
            "elapsed_seconds": time.monotonic() - started,
            **{key: dict(value) for key, value in counts.items()}})

    try:
        for process in processes:
            process.start()
        write_status("running")
        while submitted < min(len(jobs), workers * 2):
            tasks.put(jobs[submitted])
            submitted += 1
        last_update = time.monotonic()
        while completed < len(jobs):
            try:
                message = results.get(timeout=1)
            except queue.Empty:
                if any(process.exitcode is not None for process in processes):
                    raise RuntimeError("A coordinator worker exited unexpectedly; see per-CAD logs")
                if time.monotonic() - last_update >= 10:
                    write_status("running")
                    last_update = time.monotonic()
                continue
            if "error" in message:
                raise RuntimeError(message["error"])
            row = message["result"]
            counts[row["split"]].update(row["counts"])
            completed += 1
            if submitted < len(jobs):
                tasks.put(jobs[submitted])
                submitted += 1
            if completed % 100 == 0 or completed == len(jobs):
                write_status("running")
                print(f"Scored {completed}/{len(jobs)} CADs in {time.monotonic()-started:.1f}s", flush=True)
        for _ in processes:
            tasks.put(None)
        for process in processes:
            process.join(10)
        write_status("finalizing")
    except BaseException as error:
        write_status("interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed")
        raise
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join()
        tasks.cancel_join_thread()
        tasks.close()
        results.close()
    return read_json(output / "status.json")


def finalize(output: Path, protocol: dict, status: dict) -> None:
    source = Path(protocol["source_run"])
    source_digests = {
        source / "metadata/protocol.json": protocol["source_protocol_sha256"],
        source / "status.json": protocol["source_status_sha256"],
        **{source / name: digest for name, digest in protocol["source_manifest_sha256"].items()},
        **{REPO_ROOT / name: digest for name, digest in protocol["source_sha256"].items()},
    }
    for path, expected in source_digests.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen source changed during this run: {path}")
    hashes = {}
    for split, name in MANIFESTS.items():
        if status[split].get("images", 0) != len(protocol["splits"][split]):
            raise ValueError(f"Incomplete image denominator: {split}")
        with tempfile.NamedTemporaryFile(mode="w", dir=output, suffix=".jsonl.tmp", delete=False) as stream:
            temporary = Path(stream.name)
            for image_id in protocol["splits"][split]:
                row = read_json(output / "cases" / split / image_id / "selection.json")
                if (row.get("protocol_fingerprint") != protocol["fingerprint"] or
                        row.get("image_id") != image_id or row.get("split") != split):
                    raise ValueError(f"Missing or stale selection: {split}/{image_id}")
                if row["pair"] is not None:
                    if row["pair"].get("image_id") != image_id or row["pair"].get("split") != split:
                        raise ValueError(f"Pair identity mismatch: {split}/{image_id}")
                    stream.write(json.dumps(row["pair"], allow_nan=False, sort_keys=True) + "\n")
        os.replace(temporary, output / name)
        hashes[name] = sha256_file(output / name)
    summary = {"protocol_fingerprint": protocol["fingerprint"], "partial": protocol["partial"],
               "source_run": protocol["source_run"], "splits": {}}
    for split in MANIFESTS:
        counts = status[split]
        count = len(protocol["splits"][split])
        summary["splits"][split] = {**counts, "intended_images": count,
            "intended_candidates": count * protocol["candidates"],
            "pair_coverage": counts.get("pairs", 0) / count if count else None}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "status.json", {**status, "status": "complete", "manifest_sha256": hashes})


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-run", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--workers", type=int, default=32, help="CPU coordinators; each processes one CAD at a time")
    result.add_argument("--image-timeout", type=float, default=600, help="Seconds per CAD including all 8 candidates")
    result.add_argument("--splits", nargs="+", choices=list(MANIFESTS), default=["validation", "train"])
    result.add_argument("--limit", type=int, help="First N CADs per selected split; marks dataset partial, unsuitable for training")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--resolution", type=int, default=224)
    result.add_argument("--mesh-deflection", type=float, default=.002)
    result.add_argument("--mask-threshold", type=int, default=245)
    result.add_argument("--resume", action="store_true", help="Reuse matching terminal measurements; change worker count freely")
    return result


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    signal.signal(signal.SIGTERM, _stop_worker)
    if args.workers < 1 or not math.isfinite(args.image_timeout) or args.image_timeout <= 0:
        cli.error("workers and image-timeout must be positive")
    if args.limit is not None and args.limit < 1:
        cli.error("limit must be positive")
    if len(set(args.splits)) != len(args.splits):
        cli.error("splits must be distinct")
    if args.resolution < 2 or not math.isfinite(args.mesh_deflection) or args.mesh_deflection <= 0 or not 0 < args.mask_threshold <= 255:
        cli.error("Invalid observation settings")
    # Set before importing numerical/OCC modules, including spawned workers.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    source, output = args.source_run.resolve(), args.output_dir.resolve()
    if output == source or source in output.parents or output in source.parents:
        cli.error("Output must be independent of the source run")
    protocol = build_protocol(args)
    output.mkdir(parents=True, exist_ok=args.resume)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_run(output, protocol, args)
        old_pairs = read_old_pairs(source, protocol["splits"])
        try:
            status = run_jobs(source, output, protocol, args.workers, old_pairs)
            finalize(output, protocol, status)
        except BaseException as error:
            atomic_json(output / "failure.json", {"error": f"{type(error).__name__}: {error}",
                "created_utc": datetime.now(timezone.utc).isoformat(), "traceback": traceback.format_exc()})
            current = read_json(output / "status.json")
            atomic_json(output / "status.json", {**current,
                "status": "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                "error": f"{type(error).__name__}: {error}"})
            raise
    print(f"Complete{' (partial validation run)' if protocol['partial'] else ''}: {output}", flush=True)


if __name__ == "__main__":
    main()
