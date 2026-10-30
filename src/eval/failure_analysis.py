"""Describe saved diffusion outcomes without rerunning generation or repair.

The mutually exclusive stage is an execution outcome, not a causal attribution.
Multiple observations may apply to one CAD. Missing records remain pending.
"""
from __future__ import annotations

from collections import Counter
from typing import Any


PRIMARY_STAGES = (
    "pending", "generation_failure", "raw_input_failure", "budget_exhausted",
    "execution_error", "incomplete_trimmed_faces", "no_solid",
    "native_readback_or_boundary_rejected", "valid_solid", "unresolved",
)
UNAVAILABLE_DIAGNOSTICS = {
    "fitting_residuals": "not_recorded",
    "optimization_displacement": "not_recorded",
    "sewing_internal_trace": "not_recorded",
    "diffusion_vs_vae_causal_attribution": "not_identified_by_saved_outputs",
}
_STAGE_ORDER = {"optimization": 0, "fitting": 1, "trimming": 2,
                "solid_candidates": 3, "native_readback": 4}


def _summarize_path(path: dict[str, Any], index: int) -> dict[str, Any]:
    faces = path.get("faces", [])
    solid_attempts = path.get("solid_attempts", [])
    return {
        "index": index, "label": path.get("label"), "status": path.get("status"),
        "stage": path.get("stage"), "input_face_count": path.get("input_face_count"),
        "used_face_count": path.get("used_face_count"),
        "selected_edge_count": path.get("selected_edge_count"),
        "failed_face_indices": [face["face_index"] for face in faces
                                if face.get("accepted_attempt") is None]
                               if path.get("status") == "incomplete_faces" else [],
        "unaccepted_face_indices": [face["face_index"] for face in faces
                                    if face.get("accepted_attempt") is None],
        "solid_attempt_counts": dict(Counter(item.get("status", "unknown") for item in solid_attempts)),
        "elapsed_seconds": path.get("elapsed_seconds"),
        "evidence": f"diagnostics.paths[{index}]",
    }


def _trace_observations(paths: list[dict[str, Any]]) -> dict[str, bool | None]:
    face_attempts = [attempt for path in paths for face in path.get("faces", [])
                     for attempt in face.get("attempts", [])]
    solid_attempts = [attempt for path in paths for attempt in path.get("solid_attempts", [])]
    face_tests = {
        "wire_assembly_failed": lambda item: item.get("status") == "no_wires",
        "open_wire_rejected": lambda item: any(value in ("open_wire", "single_open_edge")
                                               for value in item.get("skipped_wires", [])),
        "wire_count_decreased_during_face_fix": lambda item: item.get("rejection") == "wire_count_decreased_during_face_fix",
        "edge_omitted_during_wire_assembly": lambda item: item.get("rejection") == "edge_omitted_during_wire_assembly",
        "trim_rejected": lambda item: item.get("status") == "trim_rejected",
        "natural_wire_replacement": lambda item: item.get("natural_wire_replacement") is True,
    }
    result = {name: any(check(item) for item in face_attempts) if face_attempts else None
              for name, check in face_tests.items()}
    result.update({
        "incomplete_trimmed_faces": any(item.get("status") == "incomplete_faces" for item in paths) if paths else None,
        "solid_not_returned": any(item.get("status") == "no_solid" for item in solid_attempts) if solid_attempts else None,
        "native_readback_rejected": any(item.get("status") == "native_rejected" for item in [*paths, *solid_attempts]) if paths else None,
        "boundary_counts_rejected": any(item.get("status") == "boundary_rejected" for item in solid_attempts) if solid_attempts else None,
        "post_execution_error": any(item.get("status") == "error" for item in [*paths, *face_attempts, *solid_attempts]) if paths else None,
    })
    return result


def trace_outcome(
    inference: dict[str, Any], evaluation: dict[str, Any] | None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify one saved result, preserving valid solids with later metric errors.

    ``diagnostics`` is the saved ``post_diagnostics.json``. If omitted, the copy
    embedded in the evaluation record is used. No GT correspondence is assumed.
    """
    evaluation = evaluation or {}
    diagnostics = diagnostics if diagnostics is not None else evaluation.get("post_diagnostics", {})
    paths = diagnostics.get("paths", [])
    status = evaluation.get("status")
    validity = evaluation.get("validity", "unknown")
    native = evaluation.get("native", {})
    observed = [path["stage"] for path in paths if path.get("stage") in _STAGE_ORDER]
    if any(item.get("status") in ("native_rejected", "boundary_rejected", "accepted")
           for path in paths for item in path.get("solid_attempts", [])):
        observed.append("native_readback")
    flags = _trace_observations(paths)
    flags["geometry_metric_error"] = status == "evaluation_error" and evaluation.get("stage") in ("raw", "geometry", "metrics", "post_metrics")
    flags["evaluation_error_after_validity"] = validity == "valid" and status not in (None, "ok", "running")
    flags["budget_exhausted"] = status == "timeout" or diagnostics.get("status") == "budget_exhausted"

    stage, evidence = "unresolved", {"source": "evaluation", "status": status}
    if validity == "valid" or native.get("native_valid_solid") is True:
        stage, evidence = "valid_solid", {"source": "evaluation.validity/native.native_valid_solid", "validity": validity}
    elif inference.get("status") not in (None, "ok", "running", "pending"):
        stage, evidence = "generation_failure", {"source": "inference.status", "status": inference["status"]}
    elif not inference or not evaluation or status in (None, "running", "pending", "not_run"):
        stage, evidence = "pending", {"source": "inference/evaluation.status", "status": status}
    elif status in ("input_error", "missing_raw", "invalid_raw") or (status == "evaluation_error" and evaluation.get("stage") == "raw"):
        stage, evidence = "raw_input_failure", {"source": "evaluation.status/stage", "status": status, "stage": evaluation.get("stage")}
    elif flags["budget_exhausted"]:
        stage, evidence = "budget_exhausted", {"source": "evaluation.status/diagnostics.status", "status": status, "diagnostic_status": diagnostics.get("status")}
    elif paths:
        last = paths[-1]
        path_status = last.get("status")
        evidence = {"source": f"diagnostics.paths[{len(paths) - 1}]", "label": last.get("label"), "status": path_status,
                    "stage": last.get("stage")}
        if path_status == "incomplete_faces":
            stage = "incomplete_trimmed_faces"
        elif path_status == "no_accepted_solid":
            attempt_statuses = {item.get("status") for item in last.get("solid_attempts", [])}
            if attempt_statuses & {"native_rejected", "boundary_rejected"}:
                stage = "native_readback_or_boundary_rejected"
            elif "no_solid" in attempt_statuses:
                stage = "no_solid"
            elif "error" in attempt_statuses:
                stage = "execution_error"
        elif path_status in ("native_rejected", "boundary_rejected"):
            stage = "native_readback_or_boundary_rejected"
        elif path_status == "budget_exhausted":
            stage = "budget_exhausted"
        elif path_status == "error":
            stage = "execution_error"
    elif status in ("evaluation_error", "post_error", "crash", "interrupted", "step_read_error"):
        stage = "execution_error"
    return {
        "primary_stage": stage, "stage_evidence": evidence,
        "native_validity": "valid" if stage == "valid_solid" else validity,
        "evaluation_status": status, "evaluation_error": evaluation.get("error", ""),
        "path_summaries": [_summarize_path(path, index) for index, path in enumerate(paths)],
        "furthest_recovery_stage": max(observed, key=_STAGE_ORDER.get) if observed else "not_observed",
        "observation_flags": flags, "unavailable_diagnostics": dict(UNAVAILABLE_DIAGNOSTICS),
        "interpretation": "Recorded stages and co-occurring observations; not established root causes.",
    }
