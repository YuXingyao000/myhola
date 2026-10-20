"""偏好数据的运行记录；协议和输入变化不能混入已有 run。"""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import distributions
import json
from pathlib import Path
import shutil
import subprocess
import sys

from tools.vae_compare_post import atomic_json, fingerprint, read_json, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]


def read_ids(path: str | Path) -> list[str]:
    values = Path(path).read_text().splitlines()
    if not values or any(not value or value.strip() != value or "/" in value for value in values):
        raise ValueError(f"Expected nonempty CAD identities, one per line: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate CAD identities: {path}")
    return values


def source_hashes() -> dict[str, str]:
    paths = [
        *REPO_ROOT.glob("src/brepnet/data/preference_*.py"),
        REPO_ROOT / "src/brepnet/diffusion/legacy.py",
        REPO_ROOT / "src/brepnet/diffusion/image_loading.py",
        REPO_ROOT / "src/brepnet/eval/preference.py",
        *REPO_ROOT.glob("src/brepnet/post/**/*.py"),
        REPO_ROOT / "tools/vae_compare_post.py",
        REPO_ROOT / "shared/occ_utils.py",
        REPO_ROOT / "shared/common_utils.py",
    ]
    # Training can be edited after generation starts without changing its protocol.
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in sorted(paths) if path.name != "preference_dataset.py"}


def prepare_run(config: dict, run: Path, resume: bool) -> dict:
    from OCC import VERSION as occ_version

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
    protocol["fingerprint"] = fingerprint(protocol)
    path = run / "metadata/protocol.json"
    if resume:
        previous = read_json(path)
        if previous != protocol:
            raise ValueError("Resume protocol, source files, checkpoint or splits differ")
        return protocol
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
    (run / "metadata/working_tree.diff").write_text(git("diff", "--", "src/brepnet", "configs"))
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
