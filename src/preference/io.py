"""偏好数据的运行记录；协议和输入变化不能混入已有 run。"""

from __future__ import annotations

from datetime import datetime, timezone
import difflib
from importlib.metadata import distributions
import json
from pathlib import Path
import shutil
import subprocess
import sys

from tools.vae_compare_post import atomic_json, fingerprint, read_json, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEDULING_SOURCES = {
    "src/preference/generate.py",
    "src/preference/io.py",
}


def read_ids(path: str | Path) -> list[str]:
    values = Path(path).read_text().splitlines()
    if not values or any(not value or value.strip() != value or "/" in value for value in values):
        raise ValueError(f"Expected nonempty CAD identities, one per line: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate CAD identities: {path}")
    return values


def source_hashes() -> dict[str, str]:
    paths = [
        *REPO_ROOT.glob("src/preference/*.py"),
        REPO_ROOT / "src/models/diffusion/legacy.py",
        REPO_ROOT / "src/models/diffusion/image_loading.py",
        REPO_ROOT / "src/datasets/photo.py",
        REPO_ROOT / "src/models/diffusion/photo_baseline.py",
        REPO_ROOT / "src/models/diffusion/preference.py",
        *REPO_ROOT.glob("src/models/diffusion/preference_geometry.py"),
        REPO_ROOT / "src/eval/preference.py",
        *REPO_ROOT.glob("src/postprocess/**/*.py"),
        REPO_ROOT / "tools/vae_compare_post.py",
        REPO_ROOT / "src/common/occ_utils.py",
        REPO_ROOT / "src/common/common_utils.py",
    ]
    # Training can be edited after generation starts without changing its protocol.
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in sorted(paths) if path.name != "preference_dataset.py"}


def _record_runtime_revision(run: Path, config: dict, previous: dict, current: dict,
                             accept_scheduling_update: bool) -> None:
    """保留原始协议；恢复所用调度源码和资源另存为有序记录。"""
    created = datetime.now(timezone.utc)
    relative_root = Path("metadata/runtime") / created.strftime("%Y%m%dT%H%M%S.%fZ")
    root = run / relative_root
    root.mkdir(parents=True)
    changes = {
        name: {"before": previous["source_sha256"][name], "after": digest}
        for name, digest in current["source_sha256"].items()
        if previous["source_sha256"][name] != digest
    }
    diff = []
    for name in sorted(SCHEDULING_SOURCES):
        source = REPO_ROOT / name
        target = root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        original = run / "metadata/source" / name
        diff.extend(difflib.unified_diff(
            original.read_text().splitlines(keepends=True), source.read_text().splitlines(keepends=True),
            fromfile=f"original/{name}", tofile=f"runtime/{name}"))
    (root / "scheduling.diff").write_text("".join(diff))
    git = lambda *args: subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)
    record = {
        "created_utc": created.isoformat(),
        "protocol_fingerprint": previous["fingerprint"],
        "accept_scheduling_update": accept_scheduling_update,
        "source_changes": changes,
        "source_sha256": current["source_sha256"],
        "runtime_directory": str(relative_root),
        "git_commit": git("rev-parse", "HEAD").strip(),
        "git_status": git("status", "--short"),
        "command": [sys.executable, *sys.argv],
        "resources": {key: config[key] for key in (
            "gpus", "workers_per_gpu", "gpu_workers_per_gpu", "cpu_workers", "queue_size",
            "gpu_cpu_threads") if key in config},
        "config": config,
    }
    atomic_json(root / "runtime.json", record)
    path = run / "metadata/runtime_revisions.json"
    revisions = read_json(path) or {"version": "svr_preference_runtime_v1", "revisions": []}
    revisions["revisions"].append(record)
    atomic_json(path, revisions)


def prepare_run(config: dict, run: Path, resume: bool,
                accept_scheduling_update: bool = False) -> dict:
    from OCC import VERSION as occ_version
    from src.preference.source import source_protocol

    splits = {name: read_ids(config[f"{name}_list"]) for name in ("validation", "train")}
    if set(splits["validation"]) & set(splits["train"]):
        raise ValueError("Training and validation identities overlap")
    for name in splits:
        limit = config.get(f"{name}_limit")
        if limit is not None:
            splits[name] = splits[name][:int(limit)]
    if not any(splits.values()):
        raise ValueError("No requested images")
    protocol = {
        "version": "svr_preference_generation_v1",
        "checkpoint": str(Path(config["checkpoint"]).resolve()),
        "checkpoint_sha256": sha256_file(Path(config["checkpoint"])),
        "raw_data_root": str(Path(config["raw_data_root"]).resolve()),
        "condition_data_root": str(Path(config["condition_data_root"]).resolve()),
        "dino_repo": str(Path(config["dino_repo"]).resolve()),
        "dino_source_sha256": {
            str(path.relative_to(config["dino_repo"])): sha256_file(path)
            for path in sorted(Path(config["dino_repo"]).rglob("*.py"))},
        "seed": config["seed"], "candidates": config["candidates"],
        "gpu_cpu_threads": config["gpu_cpu_threads"],
        "post": config["post"], "selection": config["selection"],
        "software": {
            distribution.metadata["Name"]: distribution.version
            for distribution in distributions()
            if distribution.metadata["Name"].lower() in
            {"torch", "torchvision", "diffusers", "numpy", "scipy", "trimesh", "pythonocc-core"}},
        "python": sys.version,
        "pythonocc": occ_version,
        "source_sha256": source_hashes(), "splits": splits,
        "sampling": "original DDPM1000 / FP32 / batch=1 / svr_imgs[0]",
        "gt_cache": "original embedded AE / FP32 / mean32+std32 / rotation0",
    }
    source = source_protocol(config)
    if source:
        protocol.update(source)
        protocol["version"] = "photo_preference_generation_v1"
        protocol["sampling"] = "original DDPM1000 / FP32 / batch=1 / real_photo.npz[flux]"
    protocol["fingerprint"] = fingerprint(protocol)
    path = run / "metadata/protocol.json"
    if resume:
        previous = read_json(path)
        if not previous:
            raise ValueError(f"Resume protocol is missing: {path}")
        if previous != protocol:
            # 仅显式放行编排变更；模型、指标、预算和数据协议仍逐项严格相等。
            unchanged_protocol = {
                **protocol, "source_sha256": previous["source_sha256"],
                "fingerprint": previous["fingerprint"],
            }
            changed_sources = {
                name for name in previous["source_sha256"].keys() | protocol["source_sha256"].keys()
                if previous["source_sha256"].get(name) != protocol["source_sha256"].get(name)
            }
            if (not accept_scheduling_update or unchanged_protocol != previous or
                    previous["source_sha256"].keys() != protocol["source_sha256"].keys() or
                    not changed_sources <= SCHEDULING_SOURCES or
                    previous["fingerprint"] != fingerprint({
                        key: value for key, value in previous.items() if key != "fingerprint"})):
                raise ValueError("Resume protocol, source files, checkpoint or splits differ")
        _record_runtime_revision(run, config, previous, protocol, accept_scheduling_update)
        return previous
    if run.exists():
        raise FileExistsError(f"Existing run requires --resume: {run}")
    (run / "metadata").mkdir(parents=True)
    atomic_json(path, protocol)
    atomic_json(run / "metadata/config.json", config)
    git = lambda *args: subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)
    atomic_json(run / "metadata/run.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD").strip(),
        "git_status": git("status", "--short"),
        "command": [sys.executable, *sys.argv],
        "gpu_inventory": subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"], text=True),
    })
    (run / "metadata/working_tree.diff").write_text(git("diff", "--", "src", "configs"))
    for relative in protocol["source_sha256"]:
        target = run / "metadata/source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, target)
    for name, ids in splits.items():
        (run / f"metadata/{name}.txt").write_text("".join(f"{value}\n" for value in ids))
    return protocol


def append_json(stream, value: dict) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
