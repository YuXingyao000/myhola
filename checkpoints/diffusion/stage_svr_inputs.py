"""将指定 train/validation 的 SVR 输入按原字节暂存，不解码图像或 latent。

仅复制 imgs.npz 和 rotation 0 的 features.npy；不读取照片、test 或其他
rotation 的 latent。目标必须不存在，失败后保留 INCOMPLETE 供检查。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import stat
import time


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def checked_stat(path: Path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Expected a regular file: {path}")
    return info


def file_version(info) -> tuple[int, int]:
    return info.st_size, info.st_mtime_ns


def read_split(path: Path) -> tuple[list[str], dict]:
    before = checked_stat(path)
    content = path.read_bytes()
    if file_version(before) != file_version(checked_stat(path)) or len(content) != before.st_size:
        raise RuntimeError(f"Source changed while reading split: {path}")
    model_ids = []
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), start=1):
        model_id = line.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", model_id):
            raise ValueError(f"Invalid model ID on {path}:{line_number}: {model_id!r}")
        model_ids.append(model_id)
    if not model_ids:
        raise ValueError(f"Empty split: {path}")
    return model_ids, {
        "source": str(path), "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
        "sha256": hashlib.sha256(content).hexdigest(),
        "entries": len(model_ids), "unique_ids": len(set(model_ids)),
    }


def inspect_file(source: Path, destination: Path) -> dict:
    info = checked_stat(source)
    return {
        "source": str(source.resolve()), "destination": destination.as_posix(),
        "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
    }


def copy_checked(source: Path, destination: Path, dst_root: Path, expected: dict | None = None) -> dict:
    """流式复制并哈希；源文件在复制前后的 size/mtime 必须一致。"""
    before = checked_stat(source)
    if expected is not None and file_version(before) != (expected["size_bytes"], expected["mtime_ns"]):
        raise RuntimeError(f"Source changed before copy: {source}")
    target = dst_root / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    copied_bytes = 0
    with source.open("rb") as reader, target.open("xb") as writer:
        while chunk := reader.read(1024 * 1024):
            writer.write(chunk)
            digest.update(chunk)
            copied_bytes += len(chunk)
    after = checked_stat(source)
    if file_version(before) != file_version(after) or copied_bytes != before.st_size:
        raise RuntimeError(f"Source changed during copy: {source}")
    checksum = digest.hexdigest()
    if expected is not None and checksum != expected["sha256"]:
        raise RuntimeError(f"Split contents changed before copy: {source}")
    return {
        "source": str(source.resolve()), "destination": destination.as_posix(),
        "size_bytes": copied_bytes, "mtime_ns": before.st_mtime_ns, "sha256": checksum,
    }


def stage_inputs(
    *, train_split: str | Path, val_split: str | Path,
    condition_root: str | Path, latent_root: str | Path, dst_root: str | Path,
    workers: int = 4, dry_run: bool = False,
) -> dict:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    train_split, val_split = Path(train_split).resolve(), Path(val_split).resolve()
    condition_root, latent_root = Path(condition_root).resolve(), Path(latent_root).resolve()
    dst_root = Path(dst_root).absolute()
    if dst_root.exists() or dst_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing destination: {dst_root}")
    started = time.monotonic()
    created_at = utc_now()
    if not dry_run:
        dst_root.mkdir(parents=True, exist_ok=False)
        (dst_root / "INCOMPLETE").write_text(f"SVR staging started at {created_at}\n", encoding="utf-8")

    # split 的排序、重复行、换行符和空白按原字节保留；只对待复制对象去重。
    train_ids, train_info = read_split(train_split)
    val_ids, val_info = read_split(val_split)
    model_ids = sorted(set(train_ids) | set(val_ids))
    splits = {"train": train_info, "validation": val_info}
    files = []
    for name, source in (("train", train_split), ("validation", val_split)):
        relative_path = Path("splits") / f"{name}.txt"
        splits[name]["destination"] = relative_path.as_posix()
        if dry_run:
            files.append({key: value for key, value in splits[name].items() if key not in ("entries", "unique_ids")})
        else:
            files.append(copy_checked(source, relative_path, dst_root, expected=splits[name]))

    def stage_object(model_id: str) -> list[dict]:
        paths = (
            (condition_root / model_id / "imgs.npz", Path("conditions") / model_id / "imgs.npz"),
            (latent_root / f"{model_id}_0" / "features.npy", Path("latents") / f"{model_id}_0" / "features.npy"),
        )
        if dry_run:
            return [inspect_file(source, relative) for source, relative in paths]
        return [copy_checked(source, relative, dst_root) for source, relative in paths]

    action = "Scanned" if dry_run else "Copied"
    print(f"SVR staging: {len(model_ids)} unique objects, {workers} workers, dry_run={dry_run}", flush=True)
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        for completed, records in enumerate(executor.map(stage_object, model_ids), start=1):
            files.extend(records)
            if completed % 2000 == 0 or completed == len(model_ids):
                print(f"{action} {completed}/{len(model_ids)} objects in {time.monotonic() - started:.1f}s", flush=True)
    finally:
        # 出错时取消尚未启动的任务，保留已复制内容和 INCOMPLETE。
        executor.shutdown(wait=True, cancel_futures=True)

    manifest = {
        "schema_version": 1, "dry_run": dry_run,
        "created_at_utc": created_at, "completed_at_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - started, "destination_root": str(dst_root.resolve()),
        "source_roots": {"condition_root": str(condition_root), "latent_root": str(latent_root)},
        "protocol": {
            "condition_name": "single_img", "image_member": "svr_imgs", "rotation_id": 0,
            "is_aug": False, "real_photo_ratio": 0.0, "validation_real_photo_ratio": 0.0,
            "image_storage": "Original imgs.npz bytes; its other members/views are unchanged.",
            "scope": "Only supplied train/validation splits; no test, real_photo.npz or other latent rotations.",
        },
        "splits": splits,
        "id_counts": {
            "train_entries": len(train_ids), "validation_entries": len(val_ids),
            "train_unique": len(set(train_ids)), "validation_unique": len(set(val_ids)),
            "train_validation_overlap": len(set(train_ids) & set(val_ids)), "unique_total": len(model_ids),
        },
        "file_count": len(files), "total_bytes": sum(item["size_bytes"] for item in files),
        "files": sorted(files, key=lambda item: item["destination"]),
    }
    if not dry_run:
        with (dst_root / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        (dst_root / "INCOMPLETE").unlink()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Read splits and inspect file metadata without copying inputs or creating dst-root.")
    manifest = stage_inputs(**vars(parser.parse_args()))
    print(json.dumps({key: manifest[key] for key in (
        "dry_run", "destination_root", "id_counts", "file_count", "total_bytes", "elapsed_seconds",
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
