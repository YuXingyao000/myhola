"""Explicit method-aware case access, with the original legacy manifest supported.

An intersection case must use generic fields. Old ``*_legacy`` fields are never
silently relabelled as another method, and present-but-invalid generic fields do
not fall back to historical aliases.
"""
from __future__ import annotations

from pathlib import Path


METHODS = ("legacy", "intersection")
SUPPORTED_METHODS = (*METHODS, "local", "conditioned", "point_feedback",
                     "encoder_edge_value", "encoder_gat_control")
_RECORD_ALIASES = {"raw": "raw_legacy", "post": "post_legacy", "fit": "fit_legacy"}
_PATH_ALIASES = {
    "raw_npz": "raw_legacy_npz",
    "post_step": "post_legacy_step",
    "fit_step": "fit_legacy_step",
    "gt_npz": "gt_npz",
    "gt_step": "gt_step",
}


def case_method(case: dict) -> str:
    method = case.get("method", "legacy")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported census method {method!r} for {case.get('id')}")
    return method


def manifest_method(cases: list[dict]) -> str:
    methods = {case_method(case) for case in cases}
    if len(methods) != 1:
        raise ValueError("Census cases must be nonempty and have exactly one method")
    return methods.pop()


def case_record(case: dict, stage: str) -> dict:
    if stage not in _RECORD_ALIASES:
        raise ValueError(f"Unknown case record stage {stage!r}")
    method = case_method(case)
    key = f"{stage}_record"
    if key not in case and method == "legacy":
        key = _RECORD_ALIASES[stage]
    record = case.get(key)
    if not isinstance(record, dict):
        raise ValueError(f"Missing/invalid {stage} record for {method}/{case.get('id')}")
    if record.get("method", method) != method:
        raise ValueError(f"Record method differs from case: {method}/{case.get('id')}/{stage}")
    identity_key = "model_id" if stage == "raw" else "sample_id"
    if identity_key in record and record[identity_key] != case.get("id"):
        raise ValueError(f"Record identity differs from case: {method}/{case.get('id')}/{stage}")
    return record


def case_path(case: dict, name: str) -> Path:
    if name not in _PATH_ALIASES:
        raise ValueError(f"Unknown case path name {name!r}")
    method = case_method(case)
    paths = case.get("paths")
    if not isinstance(paths, dict):
        raise ValueError(f"Missing paths for {method}/{case.get('id')}")
    key = name
    if key not in paths and method == "legacy":
        key = _PATH_ALIASES[name]
    value = paths.get(key)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"Missing/invalid {name} path for {method}/{case.get('id')}")
    return Path(value)
