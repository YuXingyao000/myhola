"""整理 DeepCAD 渲染输入：保留模型编号、既有坐标及原始 split。

优先将已有 mesh.ply 导出为 STL；缺少有效网格时才读取 normalized_shape.step。
完整记录缺失、冲突和转换失败，不修改源目录，也不重新划分数据。
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from threading import Lock
import time

import numpy as np
import trimesh


STEP_LOCK = Lock()
STL_TRIANGLE = np.dtype([
    ("normal", "<f4", (3,)),
    ("vertices", "<f4", (3, 3)),
    ("attribute", "<u2"),
])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_mesh(mesh) -> None:
    """仅检查能否忠实导出三角网格，不在这里修复或认证 B-Rep。"""
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("No triangle mesh")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Non-finite vertex coordinates")
    if mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices):
        raise ValueError("Invalid triangle vertex indices")


def mesh_from_step(path: Path, output_dir: Path):
    """在已有 PythonOCC 环境中细分 STEP；临时 STL 只用于读回网格。"""
    # 大多数模型直接复用 PLY。少量 STEP 回退串行执行，避免 OCC 共享状态干扰。
    with STEP_LOCK:
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.StlAPI import StlAPI_Writer
        from OCC.Extend.DataExchange import read_step_file

        shape = read_step_file(str(path), verbosity=False)
        mesher = BRepMesh_IncrementalMesh(shape, 0.001, False, 0.1)
        if not mesher.IsDone():
            raise ValueError("STEP triangulation failed")
        with tempfile.TemporaryDirectory(prefix="step-", dir=output_dir) as temporary:
            stl_path = Path(temporary) / "mesh.stl"
            writer = StlAPI_Writer()
            writer.ASCIIMode = False
            if not writer.Write(shape, str(stl_path)):
                raise ValueError("STEP STL export failed")
            return trimesh.load_mesh(stl_path, process=False)


def convert_model(model_id: str, *, folders: dict, output_dir: Path) -> dict:
    """一个编号生成一个 STL；每种结果都返回一条可追溯记录。"""
    model_dirs = folders.get(model_id, [])
    record = {"model_id": model_id, "source_directories": [str(p) for p in model_dirs]}
    if not model_dirs:
        return {**record, "status": "missing_source"}

    try:
        # 同一编号可能同时出现在原始和条件目录。比较实际几何文件，而非目录名。
        sources = {}
        for filename in ("mesh.ply", "normalized_shape.step"):
            paths = [folder / filename for folder in model_dirs if (folder / filename).is_file()]
            sources[filename] = [{"path": str(path), "sha256": file_sha256(path)} for path in paths]
        record["source_geometry"] = sources
        if any(len({item["sha256"] for item in items}) > 1 for items in sources.values()):
            return {**record, "status": "conflicting_sources"}

        ply_sources = sources["mesh.ply"]
        step_sources = sources["normalized_shape.step"]
        if not ply_sources and not step_sources:
            return {**record, "status": "missing_geometry"}

        mesh = None
        if ply_sources:
            source = ply_sources[0]
            try:
                mesh = trimesh.load_mesh(source["path"], process=False)
                validate_mesh(mesh)
            except Exception as error:
                record["ply_error"] = f"{type(error).__name__}: {error}"
                mesh = None
                if not step_sources:
                    raise
        if mesh is None:
            source = step_sources[0]
            mesh = mesh_from_step(Path(source["path"]), output_dir)
            validate_mesh(mesh)
            record["conversion"] = "step_to_stl"
            record["meshing"] = {"linear_deflection": 0.001, "angular_deflection_radians": 0.1}
        else:
            record["conversion"] = "ply_to_stl"

        # STL 保存 float32 三角形。核对全部顶点和绕序，禁止导出时偷偷改变几何。
        payload = mesh.export(file_type="stl")
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        if len(payload) != 84 + 50 * triangle_count or triangle_count != len(mesh.faces):
            raise ValueError("Unexpected binary STL size or triangle count")
        triangles = np.frombuffer(payload, dtype=STL_TRIANGLE, offset=84)
        if not np.array_equal(triangles["vertices"], mesh.triangles.astype("<f4")):
            raise ValueError("Export changed triangle coordinates or ordering")
        if not np.isfinite(triangles["vertices"]).all() or not np.isfinite(triangles["normal"]).all():
            raise ValueError("Non-finite STL coordinates or normals")

        stl_path = output_dir / f"{model_id}.stl"
        stl_path.write_bytes(payload)
        return {
            **record,
            "status": "success",
            "selected_source": source,
            "stl_file": stl_path.name,
            "stl_sha256": hashlib.sha256(payload).hexdigest(),
            "stl_bytes": len(payload),
            "vertex_count": len(mesh.vertices),
            "triangle_count": triangle_count,
            "bounds": {"minimum": mesh.bounds[0].tolist(), "maximum": mesh.bounds[1].tolist()},
        }
    except Exception as error:
        return {**record, "status": "failed", "error": f"{type(error).__name__}: {error}"}


def prepare(data_roots: list[Path], split_files: dict[str, Path], output_dir: Path, workers: int) -> dict:
    """读取编号 → 核对 split → 转换网格 → 保存全量逐件记录和汇总。"""
    started = time.monotonic()
    data_roots = [path.resolve() for path in data_roots]
    split_files = {name: path.resolve() for name, path in split_files.items()}
    output_dir = output_dir.resolve()

    # 1. 目录重叠是源资产关系；清单内和跨清单的重复另外统计。
    folders = defaultdict(list)
    root_counts = {}
    for root in data_roots:
        model_dirs = [Path(entry.path) for entry in os.scandir(root) if entry.is_dir()]
        root_counts[str(root)] = len(model_dirs)
        for folder in model_dirs:
            folders[folder.name].append(folder)
    split_rows = {name: path.read_text(encoding="utf-8").split() for name, path in split_files.items()}
    split_sets = {name: set(rows) for name, rows in split_rows.items()}
    split_duplicates = {
        name: {mid: count for mid, count in Counter(rows).items() if count > 1}
        for name, rows in split_rows.items()
    }
    memberships = defaultdict(list)
    for name, ids in split_sets.items():
        for model_id in ids:
            memberships[model_id].append(name)
    source_ids = set(folders)
    requested_ids = set(memberships)
    model_ids = sorted(source_ids | requested_ids)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python_version": sys.version,
        "trimesh_version": trimesh.__version__,
        "numpy_version": np.__version__,
        "workers": workers,
        "output_dir": str(output_dir),
        "source_roots": root_counts,
        "source_union_count": len(source_ids),
        "source_overlap_count": sum(len(paths) > 1 for paths in folders.values()),
        "requested_unique_count": len(requested_ids),
        "unassigned_count": len(source_ids - requested_ids),
        "split_files": {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in split_files.items()},
        "split_rows": {name: len(rows) for name, rows in split_rows.items()},
        "split_unique_counts": {name: len(ids) for name, ids in split_sets.items()},
        "split_duplicate_ids": split_duplicates,
        "cross_split_ids": {mid: names for mid, names in memberships.items() if len(names) > 1},
        "missing_source_ids_by_split": {name: sorted(ids - source_ids) for name, ids in split_sets.items()},
    }

    # 2. 原清单原样存档，不用成功子集覆盖既有 train/validation/test。
    output_dir.mkdir(parents=True, exist_ok=False)
    split_dir = output_dir / "splits"
    split_dir.mkdir()
    for path in split_files.values():
        shutil.copyfile(path, split_dir / path.name)
    (split_dir / "unassigned.txt").write_text(
        "".join(mid + "\n" for mid in sorted(source_ids - requested_ids)), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"源模型 {len(source_ids):,}；split 请求 {len(requested_ids):,}；工作线程 {workers}", flush=True)

    # 3. 每个编号都留下结果。工作线程只操作自己的文件，不改变源资产。
    statuses = Counter()
    conversions = Counter()
    split_statuses = {name: Counter() for name in split_files}
    exact_groups = defaultdict(list)
    successful_ids = set()
    unsuccessful = []
    convert = partial(convert_model, folders=folders, output_dir=output_dir)
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, record in enumerate(executor.map(convert, model_ids), start=1):
                model_id = record["model_id"]
                record["splits"] = memberships.get(model_id, [])
                statuses[record["status"]] += 1
                for name in record["splits"]:
                    split_statuses[name][record["status"]] += 1
                if record["status"] == "success":
                    successful_ids.add(model_id)
                    conversions[record["conversion"]] += 1
                    exact_groups[record["stl_sha256"]].append(model_id)
                else:
                    unsuccessful.append({"model_id": model_id, "status": record["status"], "error": record.get("error")})
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                if index % 1000 == 0 or index == len(model_ids):
                    manifest.flush()
                    print(f"[{index:,}/{len(model_ids):,}] 成功 {statuses['success']:,}；其他 {index-statuses['success']:,}", flush=True)

    # 4. 精确字节重复只报告，不删模型、不改变 split。另核对最终文件编号和长度。
    duplicates = [{"sha256": digest, "model_ids": ids} for digest, ids in exact_groups.items() if len(ids) > 1]
    cross_split_groups = [
        group for group in duplicates
        if len({name for mid in group["model_ids"] for name in memberships.get(mid, [])}) > 1
    ]
    disk_ids = set()
    invalid_sizes = []
    for entry in os.scandir(output_dir):
        if entry.name.endswith(".stl"):
            disk_ids.add(Path(entry.name).stem)
            size = entry.stat().st_size
            if size < 134 or (size - 84) % 50:
                invalid_sizes.append(entry.name)
    summary.update({
        "status_counts": dict(statuses),
        "conversion_counts": dict(conversions),
        "split_status_counts": {name: dict(counts) for name, counts in split_statuses.items()},
        "unsuccessful_models": unsuccessful,
        "exact_duplicate_stl_group_count": len(duplicates),
        "exact_duplicate_stl_model_count": sum(len(group["model_ids"]) for group in duplicates),
        "cross_split_exact_duplicate_stl_groups": cross_split_groups,
        "output_file_count": len(disk_ids),
        "output_ids_match_manifest": disk_ids == successful_ids,
        "invalid_stl_sizes": invalid_sizes,
        "split_rendering_ready_by_id": requested_ids <= successful_ids and not any(split_duplicates.values()) and not summary["cross_split_ids"],
        "elapsed_seconds": time.monotonic() - started,
    })
    (output_dir / "duplicate_stl_groups.json").write_text(json.dumps(duplicates, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("status_counts", "split_status_counts", "output_file_count", "split_rendering_ready_by_id", "elapsed_seconds")}, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_roots", type=Path, nargs="+", required=True)
    parser.add_argument("--training_list", type=Path, required=True)
    parser.add_argument("--validation_list", type=Path, required=True)
    parser.add_argument("--testing_list", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    prepare(args.data_roots, {
        "training": args.training_list,
        "validation": args.validation_list,
        "testing": args.testing_list,
    }, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
