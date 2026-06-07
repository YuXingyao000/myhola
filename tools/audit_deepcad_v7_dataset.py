"""Audit migrated DeepCAD v7 assets for diffusion/topology training.

The script is intentionally read-only. By default it performs a fast missing-file
and NPZ-key audit. Use --check-npz-arrays to decompress arrays and catch zlib/zip
corruption, and --check-latent-shape to read every features.npy and validate
latent stat shapes.

Example:
    python tools/audit_deepcad_v7_dataset.py --max-models 100
    python tools/audit_deepcad_v7_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATA_ROOT = Path("/mnt/d/data/deepcad_v7")
DEFAULT_COND_ROOT = Path("/mnt/d/data/deepcad_v7_cond")
DEFAULT_LATENT_ROOT = Path("/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24")
DEFAULT_OUT_DIR = Path("experiments/2026-06-07/dataset_audit")

DEFAULT_LISTS = (
    ("train", Path("src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt")),
    ("val", Path("src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt")),
    ("test", Path("src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt")),
)

DATA_NPZ_KEYS = (
    "sample_points_lines",
    "sample_points_faces",
    "edge_face_connectivity",
    "face_adj",
    "zero_positions",
)

NUM_VIEWS = 24
EXPECTED_IMAGE_DIM = 3
EXPECTED_CONDITION_FEATURE_ROWS = 48
EXPECTED_CONDITION_FEATURE_DIM = 1024
EXPECTED_LATENT_MODEL_DIM = 32
EXPECTED_LATENT_CACHE_DIM = 64
MAX_FACES = 30

NP_LOAD_ERRORS = (
    OSError,
    EOFError,
    KeyError,
    zipfile.BadZipFile,
    zlib.error,
    ValueError,
)


@dataclass(frozen=True)
class ModelEntry:
    split: str
    model_id: str


@dataclass(frozen=True)
class Issue:
    severity: str
    split: str
    model_id: str
    asset: str
    code: str
    detail: str
    path: str


@dataclass
class ModelAudit:
    split: str
    model_id: str
    status: str
    required_issues: int
    warnings: int
    issue_codes: list[str]


def parse_list_spec(spec: str) -> tuple[str, Path]:
    if ":" not in spec:
        raise ValueError(f"--list expects split:path, got {spec!r}")
    split, path = spec.split(":", 1)
    split = split.strip()
    if not split:
        raise ValueError(f"--list has an empty split name: {spec!r}")
    return split, Path(path)


def load_entries(list_specs: list[tuple[str, Path]]) -> list[ModelEntry]:
    entries: list[ModelEntry] = []
    for split, path in list_specs:
        if not path.is_file():
            raise FileNotFoundError(f"Model list not found: {path}")
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            model_id = raw_line.strip()
            if not model_id or model_id.startswith("#"):
                continue
            entries.append(ModelEntry(split=split, model_id=model_id))
    return entries


def add_issue(
    issues: list[Issue],
    *,
    severity: str,
    entry: ModelEntry,
    asset: str,
    code: str,
    detail: str,
    path: Path,
) -> None:
    issues.append(
        Issue(
            severity=severity,
            split=entry.split,
            model_id=entry.model_id,
            asset=asset,
            code=code,
            detail=detail,
            path=str(path),
        )
    )


def read_npz_arrays(
    path: Path,
    keys: tuple[str, ...],
    issues: list[Issue],
    entry: ModelEntry,
    asset: str,
    corrupt_code: str,
    missing_key_code: str,
    severity: str,
    load_arrays: bool = True,
) -> dict[str, np.ndarray | None] | None:
    arrays: dict[str, np.ndarray | None] = {}
    try:
        with np.load(path) as npz_data:
            available = set(npz_data.files)
            for key in keys:
                if key not in available:
                    add_issue(
                        issues,
                        severity=severity,
                        entry=entry,
                        asset=asset,
                        code=missing_key_code,
                        detail=f"missing key {key!r}; available={sorted(available)}",
                        path=path,
                    )
                    continue
                arrays[key] = npz_data[key] if load_arrays else None
    except NP_LOAD_ERRORS as exc:
        add_issue(
            issues,
            severity=severity,
            entry=entry,
            asset=asset,
            code=corrupt_code,
            detail=f"{type(exc).__name__}: {exc}",
            path=path,
        )
        return None
    return arrays


def read_npy(
    path: Path,
    issues: list[Issue],
    entry: ModelEntry,
    asset: str,
    corrupt_code: str,
    severity: str,
) -> np.ndarray | None:
    try:
        return np.load(path)
    except NP_LOAD_ERRORS as exc:
        add_issue(
            issues,
            severity=severity,
            entry=entry,
            asset=asset,
            code=corrupt_code,
            detail=f"{type(exc).__name__}: {exc}",
            path=path,
        )
        return None


def check_image_stack(
    arr: np.ndarray,
    *,
    expected_views: int,
    code: str,
    key: str,
    issues: list[Issue],
    entry: ModelEntry,
    asset: str,
    path: Path,
    severity: str,
) -> None:
    if arr.ndim != 4 or arr.shape[0] != expected_views or arr.shape[-1] != EXPECTED_IMAGE_DIM:
        add_issue(
            issues,
            severity=severity,
            entry=entry,
            asset=asset,
            code=code,
            detail=f"{key} shape {arr.shape}, expected [24,H,W,3]",
            path=path,
        )


def check_real_photo_flux(
    arr: np.ndarray,
    *,
    issues: list[Issue],
    entry: ModelEntry,
    path: Path,
    severity: str,
) -> None:
    valid_single = arr.ndim == 3 and arr.shape[-1] == EXPECTED_IMAGE_DIM
    valid_stack = arr.ndim == 4 and arr.shape[0] == NUM_VIEWS and arr.shape[-1] == EXPECTED_IMAGE_DIM
    if not (valid_single or valid_stack):
        add_issue(
            issues,
            severity=severity,
            entry=entry,
            asset="condition.real_photo",
            code="invalid_real_photo_shape",
            detail=f"flux shape {arr.shape}, expected [H,W,3] or [24,H,W,3]",
            path=path,
        )


def check_data_root(
    entry: ModelEntry,
    data_root: Path,
    issues: list[Issue],
    *,
    check_npz_arrays: bool,
) -> int | None:
    data_dir = data_root / entry.model_id
    if not data_dir.is_dir():
        add_issue(
            issues,
            severity="required",
            entry=entry,
            asset="raw.dir",
            code="missing_raw_dir",
            detail="raw model directory is missing",
            path=data_dir,
        )
        return None

    for filename, code in (
        ("normalized_shape.step", "missing_normalized_step"),
        ("mesh.ply", "missing_mesh_ply"),
    ):
        path = data_dir / filename
        if not path.is_file():
            add_issue(
                issues,
                severity="warning",
                entry=entry,
                asset=f"raw.{filename}",
                code=code,
                detail=f"{filename} is missing",
                path=path,
            )

    data_npz = data_dir / "data.npz"
    if not data_npz.is_file():
        add_issue(
            issues,
            severity="required",
            entry=entry,
            asset="raw.data_npz",
            code="missing_data_npz",
            detail="data.npz is missing",
            path=data_npz,
        )
        return None

    arrays = read_npz_arrays(
        data_npz,
        DATA_NPZ_KEYS,
        issues,
        entry,
        "raw.data_npz",
        "corrupt_data_npz",
        "missing_data_key",
        "required",
        load_arrays=check_npz_arrays,
    )
    if arrays is None:
        return None

    face_adj = arrays.get("face_adj")
    if face_adj is None:
        return None
    if face_adj.ndim != 2 or face_adj.shape[0] != face_adj.shape[1]:
        add_issue(
            issues,
            severity="required",
            entry=entry,
            asset="raw.data_npz",
            code="invalid_face_adj_shape",
            detail=f"face_adj shape {face_adj.shape}, expected square [F,F]",
            path=data_npz,
        )
        return None
    if face_adj.shape[0] > MAX_FACES:
        add_issue(
            issues,
            severity="required",
            entry=entry,
            asset="raw.data_npz",
            code="too_many_faces",
            detail=f"face_adj has {face_adj.shape[0]} faces, max_faces={MAX_FACES}",
            path=data_npz,
        )
    return int(face_adj.shape[0])


def check_condition_root(
    entry: ModelEntry,
    cond_root: Path,
    issues: list[Issue],
    *,
    check_npz_arrays: bool,
    check_cached_condition_shape: bool,
    require_imgs: bool,
    require_sketch: bool,
    require_real_photo: bool,
    require_cached_condition: bool,
    require_pc: bool,
) -> None:
    cond_dir = cond_root / entry.model_id
    if not cond_dir.is_dir():
        severity = "required" if (require_imgs or require_real_photo or require_cached_condition or require_pc) else "warning"
        add_issue(
            issues,
            severity=severity,
            entry=entry,
            asset="condition.dir",
            code="missing_condition_dir",
            detail="condition directory is missing",
            path=cond_dir,
        )
        return

    imgs_npz = cond_dir / "imgs.npz"
    if require_imgs and not imgs_npz.is_file():
        add_issue(
            issues,
            severity="required",
            entry=entry,
            asset="condition.imgs",
            code="missing_imgs_npz",
            detail="imgs.npz is missing",
            path=imgs_npz,
        )
    if imgs_npz.is_file():
        keys = ("svr_imgs", "sketch_imgs")
        arrays = read_npz_arrays(
            imgs_npz,
            keys,
            issues,
            entry,
            "condition.imgs",
            "corrupt_imgs_npz",
            "missing_imgs_key",
            "required" if require_imgs else "warning",
            load_arrays=check_npz_arrays,
        )
        if arrays is not None:
            if "svr_imgs" in arrays and arrays["svr_imgs"] is not None:
                check_image_stack(
                    arrays["svr_imgs"],
                    expected_views=NUM_VIEWS,
                    code="invalid_svr_imgs_shape",
                    key="svr_imgs",
                    issues=issues,
                    entry=entry,
                    asset="condition.imgs",
                    path=imgs_npz,
                    severity="required" if require_imgs else "warning",
                )
            if "sketch_imgs" in arrays and arrays["sketch_imgs"] is not None:
                check_image_stack(
                    arrays["sketch_imgs"],
                    expected_views=NUM_VIEWS,
                    code="invalid_sketch_imgs_shape",
                    key="sketch_imgs",
                    issues=issues,
                    entry=entry,
                    asset="condition.imgs",
                    path=imgs_npz,
                    severity="required" if require_sketch else "warning",
                )
            elif require_sketch:
                add_issue(
                    issues,
                    severity="required",
                    entry=entry,
                    asset="condition.imgs",
                    code="missing_sketch_imgs",
                    detail="sketch_imgs is required but missing",
                    path=imgs_npz,
                )

    real_photo_npz = cond_dir / "real_photo.npz"
    if require_real_photo and not real_photo_npz.is_file():
        add_issue(
            issues,
            severity="required",
            entry=entry,
            asset="condition.real_photo",
            code="missing_real_photo_npz",
            detail="real_photo.npz is missing",
            path=real_photo_npz,
        )
    if real_photo_npz.is_file():
        arrays = read_npz_arrays(
            real_photo_npz,
            ("flux",),
            issues,
            entry,
            "condition.real_photo",
            "corrupt_real_photo_npz",
            "missing_real_photo_key",
            "required" if require_real_photo else "warning",
            load_arrays=check_npz_arrays,
        )
        if arrays is not None and "flux" in arrays and arrays["flux"] is not None:
            check_real_photo_flux(
                arrays["flux"],
                issues=issues,
                entry=entry,
                path=real_photo_npz,
                severity="required" if require_real_photo else "warning",
            )

    feature_path = cond_dir / "img_feature_dinov2.npy"
    if require_cached_condition and not feature_path.is_file():
        add_issue(
            issues,
            severity="required",
            entry=entry,
            asset="condition.cached_feature",
            code="missing_cached_condition_feature",
            detail="img_feature_dinov2.npy is missing",
            path=feature_path,
        )
    if feature_path.is_file() and (require_cached_condition or check_cached_condition_shape):
        arr = read_npy(
            feature_path,
            issues,
            entry,
            "condition.cached_feature",
            "corrupt_cached_condition_feature",
            "required" if require_cached_condition else "warning",
        )
        if arr is not None:
            if (
                arr.ndim != 2
                or arr.shape[0] != EXPECTED_CONDITION_FEATURE_ROWS
                or arr.shape[1] != EXPECTED_CONDITION_FEATURE_DIM
            ):
                add_issue(
                    issues,
                    severity="required" if require_cached_condition else "warning",
                    entry=entry,
                    asset="condition.cached_feature",
                    code="invalid_cached_condition_shape",
                    detail=(
                        f"img_feature_dinov2.npy shape {arr.shape}, "
                        f"expected [{EXPECTED_CONDITION_FEATURE_ROWS},{EXPECTED_CONDITION_FEATURE_DIM}]"
                    ),
                    path=feature_path,
                )

    rotation_meta = cond_dir / "rotation_meta.json"
    if not rotation_meta.is_file():
        add_issue(
            issues,
            severity="warning",
            entry=entry,
            asset="condition.rotation_meta",
            code="missing_rotation_meta",
            detail="rotation_meta.json is missing",
            path=rotation_meta,
        )
    else:
        try:
            meta = json.loads(rotation_meta.read_text(encoding="utf-8"))
            if meta.get("rotation_basis") != "identity_first_cube24" or meta.get("num_views") != NUM_VIEWS:
                add_issue(
                    issues,
                    severity="warning",
                    entry=entry,
                    asset="condition.rotation_meta",
                    code="invalid_rotation_meta",
                    detail=(
                        "expected rotation_basis='identity_first_cube24' and num_views=24, "
                        f"got rotation_basis={meta.get('rotation_basis')!r}, num_views={meta.get('num_views')!r}"
                    ),
                    path=rotation_meta,
                )
        except (OSError, json.JSONDecodeError) as exc:
            add_issue(
                issues,
                severity="warning",
                entry=entry,
                asset="condition.rotation_meta",
                code="corrupt_rotation_meta",
                detail=f"{type(exc).__name__}: {exc}",
                path=rotation_meta,
            )

    if require_pc and not (cond_dir / "pc.ply").is_file():
        add_issue(
            issues,
            severity="required",
            entry=entry,
            asset="condition.pc",
            code="missing_pc_ply",
            detail="pc.ply is missing",
            path=cond_dir / "pc.ply",
        )


def check_latent_root(
    entry: ModelEntry,
    latent_root: Path,
    issues: list[Issue],
    *,
    latent_views: str,
    raw_face_count: int | None,
    check_latent_shape: bool,
    latent_model_dim: int,
    latent_cache_dim: int,
) -> None:
    view_ids = range(NUM_VIEWS) if latent_views == "all" else (0,)
    face_counts: dict[int, int] = {}
    for view_id in view_ids:
        latent_dir = latent_root / f"{entry.model_id}_{view_id}"
        if not latent_dir.is_dir():
            add_issue(
                issues,
                severity="required",
                entry=entry,
                asset="latent.dir",
                code="missing_latent_dir",
                detail=f"latent cache directory for rotation {view_id} is missing",
                path=latent_dir,
            )
            continue

        feature_path = latent_dir / "features.npy"
        if not feature_path.is_file():
            add_issue(
                issues,
                severity="required",
                entry=entry,
                asset="latent.features",
                code="missing_latent_features",
                detail=f"features.npy for rotation {view_id} is missing",
                path=feature_path,
            )
        elif check_latent_shape:
            arr = read_npy(
                feature_path,
                issues,
                entry,
                "latent.features",
                "corrupt_latent_features",
                "required",
            )
            if arr is not None:
                if arr.ndim != 2 or arr.shape[1] != latent_cache_dim:
                    add_issue(
                        issues,
                        severity="required",
                        entry=entry,
                        asset="latent.features",
                        code="invalid_latent_shape",
                        detail=f"rotation {view_id} features shape {arr.shape}, expected [F,{latent_cache_dim}]",
                        path=feature_path,
                    )
                elif arr.shape[1] < latent_model_dim * 2:
                    add_issue(
                        issues,
                        severity="required",
                        entry=entry,
                        asset="latent.features",
                        code="latent_stats_too_narrow",
                        detail=(
                            f"rotation {view_id} features shape {arr.shape}; "
                            f"model latent dim {latent_model_dim} requires at least {latent_model_dim * 2} stats columns"
                        ),
                        path=feature_path,
                    )
                else:
                    face_counts[view_id] = int(arr.shape[0])
                    if arr.shape[0] > MAX_FACES:
                        add_issue(
                            issues,
                            severity="required",
                            entry=entry,
                            asset="latent.features",
                            code="latent_too_many_faces",
                            detail=f"rotation {view_id} has {arr.shape[0]} faces, max_faces={MAX_FACES}",
                            path=feature_path,
                        )

        cache_data = latent_dir / "data.npz"
        if not cache_data.is_file():
            add_issue(
                issues,
                severity="warning",
                entry=entry,
                asset="latent.data_npz",
                code="missing_latent_data_npz",
                detail=f"data.npz for rotation {view_id} is missing",
                path=cache_data,
            )

    unique_face_counts = sorted(set(face_counts.values()))
    if len(unique_face_counts) > 1:
        add_issue(
            issues,
            severity="required",
            entry=entry,
            asset="latent.features",
            code="latent_face_count_mismatch_across_rotations",
            detail=f"face counts by rotation differ: {face_counts}",
            path=latent_root / entry.model_id,
        )
    if raw_face_count is not None and unique_face_counts and raw_face_count not in unique_face_counts:
        add_issue(
            issues,
            severity="required",
            entry=entry,
            asset="latent.features",
            code="raw_latent_face_count_mismatch",
            detail=f"raw face_adj has {raw_face_count} faces, latent counts={unique_face_counts}",
            path=latent_root / entry.model_id,
        )


def audit_entry(entry: ModelEntry, args: argparse.Namespace) -> tuple[ModelAudit, list[Issue]]:
    issues: list[Issue] = []
    raw_face_count = check_data_root(entry, args.data_root, issues, check_npz_arrays=args.check_npz_arrays)
    check_condition_root(
        entry,
        args.condition_root,
        issues,
        check_npz_arrays=args.check_npz_arrays,
        check_cached_condition_shape=args.check_cached_condition_shape,
        require_imgs=args.require_imgs,
        require_sketch=args.require_sketch,
        require_real_photo=args.require_real_photo,
        require_cached_condition=args.require_cached_condition,
        require_pc=args.require_pc,
    )
    check_latent_root(
        entry,
        args.latent_root,
        issues,
        latent_views=args.latent_views,
        raw_face_count=raw_face_count,
        check_latent_shape=args.check_latent_shape,
        latent_model_dim=args.latent_model_dim,
        latent_cache_dim=args.latent_cache_dim,
    )

    required_count = sum(1 for issue in issues if issue.severity == "required")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    status = "pass" if required_count == 0 else "fail"
    audit = ModelAudit(
        split=entry.split,
        model_id=entry.model_id,
        status=status,
        required_issues=required_count,
        warnings=warning_count,
        issue_codes=sorted({issue.code for issue in issues}),
    )
    return audit, issues


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_reports(
    out_dir: Path,
    audits: list[ModelAudit],
    issues: list[Issue],
    entries: list[ModelEntry],
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    issue_rows = [asdict(issue) for issue in issues]
    write_tsv(
        out_dir / "issues.tsv",
        issue_rows,
        ["severity", "split", "model_id", "asset", "code", "detail", "path"],
    )

    model_rows = []
    for audit in audits:
        row = asdict(audit)
        row["issue_codes"] = ",".join(audit.issue_codes)
        model_rows.append(row)
    write_tsv(
        out_dir / "models.tsv",
        model_rows,
        ["split", "model_id", "status", "required_issues", "warnings", "issue_codes"],
    )

    failed_lines = [
        f"{audit.split}\t{audit.model_id}\t{','.join(audit.issue_codes)}"
        for audit in audits
        if audit.status == "fail"
    ]
    (out_dir / "failed_models.tsv").write_text(
        "split\tmodel_id\tissue_codes\n" + "\n".join(failed_lines) + ("\n" if failed_lines else ""),
        encoding="utf-8",
    )

    ok_lines = [f"{audit.split}\t{audit.model_id}" for audit in audits if audit.status == "pass"]
    (out_dir / "ok_models.tsv").write_text(
        "split\tmodel_id\n" + "\n".join(ok_lines) + ("\n" if ok_lines else ""),
        encoding="utf-8",
    )

    by_issue_dir = out_dir / "by_issue"
    by_issue_dir.mkdir(exist_ok=True)
    grouped: dict[str, list[Issue]] = defaultdict(list)
    for issue in issues:
        grouped[issue.code].append(issue)
    for code, code_issues in grouped.items():
        lines = [
            f"{issue.severity}\t{issue.split}\t{issue.model_id}\t{issue.asset}\t{issue.detail}\t{issue.path}"
            for issue in code_issues
        ]
        (by_issue_dir / f"{code}.tsv").write_text(
            "severity\tsplit\tmodel_id\tasset\tdetail\tpath\n" + "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    issue_counter = Counter(issue.code for issue in issues)
    severity_counter = Counter(issue.severity for issue in issues)
    split_counter = Counter(entry.split for entry in entries)
    split_status: dict[str, Counter[str]] = defaultdict(Counter)
    for audit in audits:
        split_status[audit.split][audit.status] += 1

    duplicate_ids = {
        model_id: count
        for model_id, count in Counter(entry.model_id for entry in entries).items()
        if count > 1
    }

    summary = {
        "inputs": {
            "data_root": str(args.data_root),
            "condition_root": str(args.condition_root),
            "latent_root": str(args.latent_root),
            "lists": [{"split": split, "path": str(path)} for split, path in args.list_specs],
            "latent_views": args.latent_views,
            "latent_model_dim": args.latent_model_dim,
            "latent_cache_dim": args.latent_cache_dim,
            "check_npz_arrays": args.check_npz_arrays,
            "check_latent_shape": args.check_latent_shape,
            "check_cached_condition_shape": args.check_cached_condition_shape,
            "require_imgs": args.require_imgs,
            "require_sketch": args.require_sketch,
            "require_real_photo": args.require_real_photo,
            "require_cached_condition": args.require_cached_condition,
            "require_pc": args.require_pc,
            "max_models": args.max_models,
        },
        "totals": {
            "entries": len(entries),
            "unique_model_ids": len({entry.model_id for entry in entries}),
            "passed": sum(1 for audit in audits if audit.status == "pass"),
            "failed": sum(1 for audit in audits if audit.status == "fail"),
            "issues": len(issues),
            "required_issues": severity_counter.get("required", 0),
            "warnings": severity_counter.get("warning", 0),
            "duplicate_model_ids": len(duplicate_ids),
        },
        "by_split": {
            split: {
                "entries": split_counter[split],
                "passed": split_status[split].get("pass", 0),
                "failed": split_status[split].get("fail", 0),
            }
            for split in sorted(split_counter)
        },
        "issues_by_code": dict(issue_counter.most_common()),
        "duplicate_examples": dict(list(sorted(duplicate_ids.items()))[:50]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    summary_lines = [
        f"entries: {summary['totals']['entries']}",
        f"unique_model_ids: {summary['totals']['unique_model_ids']}",
        f"passed: {summary['totals']['passed']}",
        f"failed: {summary['totals']['failed']}",
        f"required_issues: {summary['totals']['required_issues']}",
        f"warnings: {summary['totals']['warnings']}",
        "",
        "issues_by_code:",
    ]
    summary_lines.extend(f"  {code}: {count}" for code, count in issue_counter.most_common())
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--condition-root", type=Path, default=DEFAULT_COND_ROOT)
    parser.add_argument("--latent-root", type=Path, default=DEFAULT_LATENT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--list",
        action="append",
        dest="list_specs_raw",
        help="Model list as split:path. Repeat for multiple splits. Defaults to deduplicated DeepCAD train/val/test lists.",
    )
    parser.add_argument("--max-models", type=int, default=None, help="Limit total entries after loading lists.")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--latent-views", choices=("all", "identity"), default="all")
    parser.add_argument("--latent-model-dim", type=int, default=EXPECTED_LATENT_MODEL_DIM)
    parser.add_argument("--latent-cache-dim", type=int, default=EXPECTED_LATENT_CACHE_DIM)
    parser.add_argument("--check-npz-arrays", action="store_true", help="Decompress required NPZ arrays and run shape checks. Slower but catches zlib/zip corruption.")
    parser.add_argument("--check-latent-shape", action="store_true", help="Read every required features.npy and validate latent stat shapes. Slower on full datasets.")
    parser.add_argument("--check-cached-condition-shape", action="store_true", help="Read optional img_feature_dinov2.npy files and validate shape.")

    parser.set_defaults(
        require_imgs=True,
        require_sketch=False,
        require_real_photo=True,
        require_cached_condition=False,
        require_pc=False,
    )
    parser.add_argument("--require-imgs", dest="require_imgs", action="store_true")
    parser.add_argument("--no-require-imgs", dest="require_imgs", action="store_false")
    parser.add_argument("--require-sketch", dest="require_sketch", action="store_true")
    parser.add_argument("--require-real-photo", dest="require_real_photo", action="store_true")
    parser.add_argument("--no-require-real-photo", dest="require_real_photo", action="store_false")
    parser.add_argument("--require-cached-condition", dest="require_cached_condition", action="store_true")
    parser.add_argument("--require-pc", dest="require_pc", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_specs_raw:
        args.list_specs = [parse_list_spec(spec) for spec in args.list_specs_raw]
    else:
        args.list_specs = list(DEFAULT_LISTS)

    entries = load_entries(args.list_specs)
    if args.max_models is not None:
        entries = entries[: args.max_models]

    print(f"Audit entries: {len(entries)}", flush=True)
    print(f"Data root: {args.data_root}", flush=True)
    print(f"Condition root: {args.condition_root}", flush=True)
    print(f"Latent root: {args.latent_root}", flush=True)
    print(f"Output dir: {args.out_dir}", flush=True)
    print(
        "Required: "
        f"imgs={args.require_imgs}, real_photo={args.require_real_photo}, "
        f"sketch={args.require_sketch}, cached_condition={args.require_cached_condition}, pc={args.require_pc}",
        flush=True,
    )

    audits: list[ModelAudit] = []
    all_issues: list[Issue] = []
    for index, entry in enumerate(entries, start=1):
        audit, issues = audit_entry(entry, args)
        audits.append(audit)
        all_issues.extend(issues)
        if args.progress_every and index % args.progress_every == 0:
            failed = sum(1 for item in audits if item.status == "fail")
            print(f"  scanned {index}/{len(entries)} | failed={failed} | issues={len(all_issues)}", flush=True)

    write_reports(args.out_dir, audits, all_issues, entries, args)
    failed = sum(1 for audit in audits if audit.status == "fail")
    print(f"Done. passed={len(audits) - failed}, failed={failed}, issues={len(all_issues)}", flush=True)
    print(f"Wrote: {args.out_dir / 'summary.txt'}", flush=True)
    print(f"Wrote: {args.out_dir / 'issues.tsv'}", flush=True)


if __name__ == "__main__":
    main()
