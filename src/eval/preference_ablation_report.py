"""Validate and report original SVR, old-label DPO and observation-label DPO.

This module reads completed runs only. Missing geometry is never zero-filled;
paired deltas always mean new DPO minus old DPO on the common CAD population.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np

from src.eval.preference_report import (
    RAW_METRICS, POST_METRICS, VALIDITIES, finite_number, load_rows, paired_statistics,
)
from tools.vae_compare_post import atomic_json, fingerprint, sha256_file


VERSION = "svr_observation_dpo_ablation_report_v1"
METHODS = ("baseline", "old_dpo", "new_dpo")
OBSERVATION_METRICS = (
    "silhouette_iou", "visible_face_cd_sq", "silhouette_boundary_distance", "depth_mae",
    "overlap_coverage", "gt_input_silhouette_iou", "gt_input_silhouette_boundary_distance",
)
METRICS = tuple(f"raw_{key}" for key in RAW_METRICS) + tuple(
    f"post_{key}" for key in POST_METRICS) + tuple(f"observation_{key}" for key in OBSERVATION_METRICS)
COMPARISON_ORCHESTRATION = {"src/eval/preference_compare.py", "src/eval/preference_report.py"}
AUDIT_ORCHESTRATION_NAMES = {"diffusion_audit.py", "diffusion_audit_report.py", "preference_report.py"}
AUDIT_TERMINAL = {"ok", "partial", "timeout", "crash", "interrupted"}
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_protocol(path: Path, *, audit: bool = False) -> dict:
    """Verify a saved protocol, including the historical audit-v1 alias format.

    Audit v1 inserted its fingerprint into the shared config dict after hashing
    the protocol. Accept that exact documented format, not arbitrary fallbacks.
    """
    value = json.loads(Path(path).read_text())
    expected = value.get("fingerprint")
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    if audit and "fingerprint" in body.get("config", {}):
        if body["config"]["fingerprint"] != expected:
            raise ValueError(f"Nested audit fingerprint differs: {path}")
        if fingerprint(body) != expected:
            if value.get("version") != "single_view_diffusion_audit_v1":
                raise ValueError(f"Unsupported audit fingerprint format: {path}")
            body = {**body, "config": {key: item for key, item in body["config"].items()
                                      if key != "fingerprint"}}
    if not expected or fingerprint(body) != expected:
        raise ValueError(f"Protocol fingerprint mismatch: {path}")
    return value


def compare_protocols(old: dict, new: dict) -> None:
    """Check comparison semantics while allowing a new policy checkpoint/run."""
    for key in ("version", "identities", "split_sha256", "sampling", "post"):
        if old.get(key) != new.get(key):
            raise ValueError(f"Comparison protocol differs: {key}")
    if old.get("version") != "svr_dpo_fixed_validation_v1":
        raise ValueError("Unsupported comparison version")
    if old["checkpoint_sha256"]["original_checkpoint"] != new["checkpoint_sha256"]["original_checkpoint"]:
        raise ValueError("Original SVR checkpoint differs")
    for key in ("raw_data_root", "condition_data_root", "seed"):
        if old["config"][key] != new["config"][key]:
            raise ValueError(f"Comparison input setting differs: {key}")
    a, b = old["source_sha256"], new["source_sha256"]
    changed = {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
    if not a or changed - COMPARISON_ORCHESTRATION:
        raise ValueError(f"Generation/metric source differs: {sorted(changed)}")


class SourceLedger:
    """Hash each resolved input once, preserving missing files explicitly."""

    def __init__(self) -> None:
        self.files: dict[str, str | None] = {}

    def capture(self, path: Path) -> str | None:
        path = Path(path).resolve()
        key = str(path)
        if key not in self.files:
            self.files[key] = sha256_file(path) if path.is_file() else None
        return self.files[key]

    def read(self, path: Path, *, optional: bool = False) -> dict:
        digest = self.capture(path)
        if digest is None:
            if optional:
                return {}
            raise ValueError(f"Missing required record: {path}")
        return json.loads(Path(path).read_text())

    def verify(self, path: Path, expected: str | None) -> None:
        if self.capture(path) != expected:
            raise ValueError(f"Source SHA256 mismatch: {path}")


def _safe_ids(ids: list[str]) -> None:
    if not ids or len(set(ids)) != len(ids) or any(
        not isinstance(value, str) or not value or Path(value).name != value or
        value in (".", "..") or "\\" in value for value in ids
    ):
        raise ValueError("Frozen identities must be distinct safe CAD IDs")


def _case_paths(run: Path, protocol: dict, method: str, image_id: str) -> dict[str, Path]:
    raw = run / "raw_predictions" / method / image_id
    evaluation = run / "evaluations" / method / image_id
    return {
        "inference_path": raw / "inference.json", "raw_path": raw / "data.npz",
        "evaluation_path": evaluation / "metrics.json",
        "diagnostics_path": evaluation / "post/c00/post_diagnostics.json",
        "step_path": evaluation / "post/c00/recon_brep.step",
        "image_path": Path(protocol["config"]["condition_data_root"]) / image_id / "imgs.npz",
        "gt_step_path": Path(protocol["config"]["raw_data_root"]) / image_id / "normalized_shape.step",
        "gt_npz_path": Path(protocol["config"]["raw_data_root"]) / image_id / "data.npz",
    }


def _verify_case(run: Path, protocol: dict, method: str, image_id: str,
                 ledger: SourceLedger) -> tuple[dict, dict, dict | None]:
    paths = _case_paths(run, protocol, method, image_id)
    inference = ledger.read(paths["inference_path"])
    if inference.get("image_id") != image_id or inference.get("method") != method:
        raise ValueError(f"Inference identity differs: {paths['inference_path']}")
    seed = int.from_bytes(hashlib.sha256(
        f"{protocol['config']['seed']}:{image_id}:diffusion:0".encode()).digest()[:4], "little")
    if inference.get("seed") != seed:
        raise ValueError(f"Inference seed differs: {method}/{image_id}")
    inputs = {"image": paths["image_path"], "gt_npz": paths["gt_npz_path"], "gt_step": paths["gt_step_path"]}
    recorded_inputs = inference.get("input_sha256", {})
    if inference["status"] == "ok" and set(recorded_inputs) != set(inputs):
        raise ValueError(f"Successful inference lacks input hashes: {method}/{image_id}")
    for key, digest in recorded_inputs.items():
        if key not in inputs:
            raise ValueError(f"Unexpected inference input: {key}")
        ledger.verify(inputs[key], digest)
    outputs = inference.get("output_sha256", {})
    if inference["status"] == "ok" and not all(outputs.get(key) for key in ("latent.npy", "data.npz")):
        raise ValueError(f"Successful inference lacks output hashes: {method}/{image_id}")
    for name, digest in outputs.items():
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("Unsafe inference output name")
        ledger.verify(paths["raw_path"].parent / name, digest)
    # Even a failed inference may leave an unmeasured partial output. Its
    # presence contributes to coverage, so preserve that input state as well.
    ledger.capture(paths["raw_path"])
    ledger.capture(paths["step_path"])
    evaluation = ledger.read(paths["evaluation_path"], optional=inference["status"] != "ok")
    semantic = None
    if evaluation:
        if evaluation.get("image_id") != image_id or evaluation.get("candidate_id") != "c00":
            raise ValueError(f"Evaluation identity differs: {method}/{image_id}")
        ep = ledger.read(paths["evaluation_path"].with_name("protocol.json"))
        expected = ep.get("protocol_fingerprint")
        if expected != fingerprint({key: value for key, value in ep.items() if key != "protocol_fingerprint"}):
            raise ValueError(f"Evaluation protocol fingerprint mismatch: {method}/{image_id}")
        if evaluation.get("protocol_fingerprint") != expected:
            raise ValueError(f"Evaluation record protocol differs: {method}/{image_id}")
        for key, path in (("raw_path", paths["raw_path"]), ("gt_npz", paths["gt_npz_path"]),
                          ("gt_step", paths["gt_step_path"])):
            if Path(ep[key]).resolve() != path.resolve():
                raise ValueError(f"Evaluation source path differs: {method}/{image_id}/{key}")
            ledger.verify(path, ep["input_sha256"][key])
        if ep.get("image_id") != image_id or ep.get("candidate_id") != "c00" or ep.get("seed") != protocol["config"]["seed"]:
            raise ValueError(f"Evaluation identity/seed differs: {method}/{image_id}")
        excluded = {"raw_path", "gt_npz", "gt_step", "output_dir", "image_id", "candidate_id",
                    "gt_cache_root", "input_sha256", "protocol_fingerprint"}
        semantic = {key: value for key, value in ep.items() if key not in excluded}
        if evaluation.get("step_sha256"):
            if Path(evaluation["step_path"]).resolve() != paths["step_path"].resolve():
                raise ValueError(f"Evaluation STEP path differs: {method}/{image_id}")
            ledger.verify(paths["step_path"], evaluation["step_sha256"])
    return inference, evaluation, semantic


def _audit_protocol(audit: Path, run: Path, methods: set[str], ids: list[str], ledger: SourceLedger) -> dict:
    ledger.capture(audit / "protocol.json")
    protocol = load_protocol(audit / "protocol.json", audit=True)
    if protocol.get("version") != "single_view_diffusion_audit_v1":
        raise ValueError("Unsupported observation audit version")
    config = protocol["config"]
    actual = config["methods"]
    if len(actual) != len(set(actual)) or not methods <= set(actual) or not set(actual) <= {"baseline", "dpo_5000"}:
        raise ValueError("Audit methods do not cover the requested methods")
    if Path(config["comparison_run"]).resolve() != run.resolve():
        raise ValueError("Audit comparison source path differs")
    ledger.verify(run / "metadata/protocol.json", protocol["comparison_protocol_sha256"])
    status = ledger.read(audit / "status.json")
    expected = len(ids) * len(actual)
    if status.get("status") != "complete" or status.get("completed") != expected or status.get("total") != expected:
        raise ValueError("Audit is incomplete or has the wrong denominator")
    manifest = ledger.read(audit / "manifest.json")
    identities = [(case["method"], case["image_id"], case.get("candidate_id")) for case in manifest["cases"]]
    expected_keys = {(method, image_id, "c00") for method in actual for image_id in ids}
    if len(identities) != expected or set(identities) != expected_keys or manifest.get("intended_count") != expected:
        raise ValueError("Audit manifest has missing or duplicate identities")
    return protocol


def _compare_audits(old: dict, new: dict) -> None:
    ignored = {"comparison_run", "output_dir", "methods", "workers", "fingerprint"}
    configs = [{key: value for key, value in item["config"].items() if key not in ignored} for item in (old, new)]
    if configs[0] != configs[1] or old["runtime"] != new["runtime"]:
        raise ValueError("Audit measurement settings/runtime differ")
    sources = [{key: value for key, value in item["source_sha256"].items()
                if Path(key).name not in AUDIT_ORCHESTRATION_NAMES} for item in (old, new)]
    if not sources[0] or sources[0] != sources[1]:
        raise ValueError("Audit measurement source differs")


def _verify_audit_row(audit: Path, audit_protocol: dict, run: Path, comparison: dict,
                      method: str, image_id: str, evaluation: dict, ledger: SourceLedger) -> dict:
    path = audit / "cases" / method / image_id / "audit.json"
    row = ledger.read(path)
    if (row.get("method"), row.get("image_id"), row.get("candidate_id")) != (method, image_id, "c00"):
        raise ValueError(f"Audit row identity differs: {path}")
    if row.get("audit_fingerprint") != audit_protocol["fingerprint"] or row.get("status") not in AUDIT_TERMINAL:
        raise ValueError(f"Unfinished, stale or source-mismatch audit: {path}")
    if row.get("validity", "unknown") != evaluation.get("validity", "unknown"):
        raise ValueError(f"Audit validity differs from saved evaluation: {path}")
    paths = _case_paths(run, comparison, method, image_id)
    sources = row.get("sources", {})
    observation = row.get("observation", {})
    if observation.get("status") == "ok" and set(sources) != set(paths):
        raise ValueError(f"Measured observation has incomplete source provenance: {path}")
    for name, source in sources.items():
        if name not in paths or Path(source["path"]).resolve() != paths[name].resolve():
            raise ValueError(f"Audit source path differs: {path}: {name}")
        ledger.verify(paths[name], source["sha256"])
    if observation.get("status") == "ok":
        op = observation.get("protocol", {})
        for name in ("resolution", "mesh_deflection", "mask_threshold"):
            if op.get(name) != audit_protocol["config"][name]:
                raise ValueError(f"Observation row settings differ: {path}: {name}")
        if op.get("view_id") != 0 or op.get("image_key") != "svr_imgs":
            raise ValueError(f"Observation view differs: {path}")
    return row


def load_study(old_run: Path, new_run: Path, old_audit: Path, new_audit: Path,
               *, expected_count: int = 3505) -> tuple[list[str], dict[str, list[dict]], dict]:
    """Validate all records and source links before returning report-ready rows."""
    roots = [Path(path).resolve() for path in (old_run, new_run, old_audit, new_audit)]
    old_run, new_run, old_audit, new_audit = roots
    ledger = SourceLedger()
    protocols, loaded = [], []
    for run in (old_run, new_run):
        ledger.capture(run / "metadata/protocol.json")
        protocol = load_protocol(run / "metadata/protocol.json")
        status = ledger.read(run / "status.json")
        if status.get("status") != "complete":
            raise ValueError(f"Comparison is incomplete: {run}")
        _safe_ids(protocol["identities"])
        ids, rows = load_rows(run, expected_count=expected_count)
        if ids != protocol["identities"]:
            raise ValueError("Comparison split differs from protocol")
        for name in ("validation.txt", "validation100.txt"):
            ledger.capture(run / name)
        ledger.verify(Path(protocol["config"]["split_list"]), protocol["split_sha256"])
        protocols.append(protocol)
        loaded.append(rows)
    compare_protocols(*protocols)
    ids = protocols[0]["identities"]
    aps = [_audit_protocol(old_audit, old_run, {"baseline", "dpo_5000"}, ids, ledger),
           _audit_protocol(new_audit, new_run, {"dpo_5000"}, ids, ledger)]
    _compare_audits(*aps)
    rows = {"baseline": loaded[0]["baseline"], "old_dpo": loaded[0]["dpo_5000"],
            "new_dpo": loaded[1]["dpo_5000"]}
    semantics = observation_protocol = None
    missing_provenance = Counter()
    for index, image_id in enumerate(ids):
        records = {}
        for label, run, cp, method in (
            ("baseline", old_run, protocols[0], "baseline"),
            ("old_dpo", old_run, protocols[0], "dpo_5000"),
            ("new_baseline", new_run, protocols[1], "baseline"),
            ("new_dpo", new_run, protocols[1], "dpo_5000"),
        ):
            inference, evaluation, semantic = _verify_case(run, cp, method, image_id, ledger)
            records[label] = (inference, evaluation)
            if semantic is not None:
                if semantics is not None and semantic != semantics:
                    raise ValueError(f"Per-object geometry measurement protocol differs: {label}/{image_id}")
                semantics = semantic
        for name in ("inference_path", "evaluation_path"):
            a = _case_paths(old_run, protocols[0], "baseline", image_id)[name]
            b = _case_paths(new_run, protocols[1], "baseline", image_id)[name]
            if ledger.capture(a) != ledger.capture(b):
                raise ValueError(f"Reused baseline record differs: {image_id}/{name}")
        nonempty_inputs = [inference.get("input_sha256") for inference, _ in records.values()
                           if inference.get("input_sha256")]
        if nonempty_inputs and any(item != nonempty_inputs[0] for item in nonempty_inputs[1:]):
            raise ValueError(f"Methods received different input hashes: {image_id}")
        for label, audit, ap, run, cp, method in (
            ("baseline", old_audit, aps[0], old_run, protocols[0], "baseline"),
            ("old_dpo", old_audit, aps[0], old_run, protocols[0], "dpo_5000"),
            ("new_dpo", new_audit, aps[1], new_run, protocols[1], "dpo_5000"),
        ):
            audit_row = _verify_audit_row(audit, ap, run, cp, method, image_id, records[label][1], ledger)
            row = rows[label][index]
            row["method"] = label
            obs = audit_row.get("observation", {})
            row.update(audit_status=audit_row["status"], observation_status=obs.get("status", "not_reached"),
                       audit_seconds=audit_row.get("elapsed_seconds"))
            if obs.get("status") == "ok":
                if observation_protocol is not None and observation_protocol != obs["protocol"]:
                    raise ValueError(f"Per-object observation protocol differs: {label}/{image_id}")
                observation_protocol = obs["protocol"]
            elif not audit_row.get("sources"):
                missing_provenance[label] += 1
            for key in OBSERVATION_METRICS:
                value = obs.get(key)
                row[f"observation_{key}"] = value if obs.get("status") == "ok" and finite_number(value) else None
            row["nonfinite_metrics"] = []
            raw = records[label][1].get("raw", {})
            if any(raw.get(key) is not None and not finite_number(raw[key])
                   for key in ("pred_face_count", "gt_face_count")):
                row["nonfinite_metrics"].append("raw_face_count_abs_error")
            for key in POST_METRICS:
                if raw.get(key) is not None and not finite_number(raw[key]):
                    row["nonfinite_metrics"].append(f"raw_{key}")
                if any(item.get(key) is not None and not finite_number(item[key])
                       for item in records[label][1].get("post", [])):
                    row["nonfinite_metrics"].append(f"post_{key}")
            for key in OBSERVATION_METRICS:
                if obs.get(key) is not None and not finite_number(obs[key]):
                    row["nonfinite_metrics"].append(f"observation_{key}")
    provenance = {
        "sources": dict(zip(("old_run", "new_run", "old_audit", "new_audit"), map(str, roots))),
        "comparison_fingerprints": [item["fingerprint"] for item in protocols],
        "audit_fingerprints": [item["fingerprint"] for item in aps],
        "checkpoint_sha256": {"baseline": protocols[0]["checkpoint_sha256"]["original_checkpoint"],
                              "old_dpo": protocols[0]["checkpoint_sha256"]["preference_checkpoint"],
                              "new_dpo": protocols[1]["checkpoint_sha256"]["preference_checkpoint"]},
        "comparison_sampling": protocols[0]["sampling"], "comparison_post": protocols[0]["post"],
        "geometry_protocol": semantics, "observation_protocol": observation_protocol,
        "failed_audits_without_source_records": dict(missing_provenance),
        "input_sha256": ledger.files,
    }
    return ids, rows, provenance


def _distribution(values: list[float]) -> dict:
    return {"n": len(values), "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "p05": float(np.quantile(values, .05)) if values else None,
            "p95": float(np.quantile(values, .95)) if values else None}


def _paired(old: dict, new: dict, repeats: int) -> dict:
    result = paired_statistics(old, new, bootstrap_repeats=repeats, seed=42)
    result["old_dpo_mean"] = result.pop("baseline_mean")
    result["new_dpo_mean"] = result.pop("dpo_5000_mean")
    result["new_minus_old"] = result.pop("dpo_minus_baseline")
    return result


def summarize(ids: list[str], rows: dict[str, list[dict]], *, bootstrap_repeats: int = 10000) -> dict:
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be positive")
    _safe_ids(ids)
    summary = {"version": VERSION, "intended_count": len(ids), "methods": {}, "metrics": {},
               "comparison": "new DPO minus old DPO; fixed step5000, K=1; no checkpoint selection",
               "bootstrap": {"repeats": bootstrap_repeats, "seed": 42, "interval": "paired CAD percentile 95%"}}
    for method in METHODS:
        records = rows[method]
        if [row["image_id"] for row in records] != ids:
            raise ValueError(f"Rows must match frozen ordered identities: {method}")
        counts = {"inference_success": sum(row["inference_status"] == "ok" for row in records),
                  "raw_outputs": sum(row["raw_exists"] for row in records),
                  "step_outputs": sum(row["step_exists"] for row in records),
                  "readable_steps": sum(row["step_readable"] for row in records),
                  "post_complete": sum(row["post_complete"] for row in records),
                  "partial_post": sum(0 < row["post_repeat_count"] < 3 for row in records),
                  "valid_without_complete_post": sum(row["validity"] == "valid" and not row["post_complete"] for row in records),
                  "inference_errors": sum(row["inference_status"] != "ok" for row in records),
                  "evaluation_errors": sum(row["evaluation_status"] not in ("ok", "not_run") for row in records),
                  "timeouts": sum(row["evaluation_status"] == "timeout" for row in records),
                  "observation_ok": sum(row["observation_status"] == "ok" for row in records),
                  "observation_available": sum(all(finite_number(row.get(f"observation_{key}")) for key in
                                                     ("silhouette_iou", "visible_face_cd_sq", "silhouette_boundary_distance"))
                                               for row in records),
                  **{f"validity_{value}": sum(row["validity"] == value for row in records) for value in VALIDITIES}}
        summary["methods"][method] = {
            "counts": counts, "rates_over_intended": {key: value / len(ids) for key, value in counts.items()},
            **{key + "_counts": dict(Counter(row[key] for row in records)) for key in
               ("inference_status", "evaluation_status", "audit_status", "observation_status")},
            "nonfinite_metric_counts": dict(Counter(key for row in records for key in row.get("nonfinite_metrics", []))),
            "timing": {key: _distribution([row[key] for row in records if finite_number(row.get(key))])
                       for key in ("inference_seconds", "evaluation_seconds", "audit_seconds")},
        }
    for key in METRICS:
        values = {method: {row["image_id"]: row[key] for row in rows[method] if finite_number(row.get(key))}
                  for method in METHODS}
        summary["metrics"][key] = {
            "own": {method: _distribution(list(values[method].values())) for method in METHODS},
            "paired_old_new_common": _paired(values["old_dpo"], values["new_dpo"], bootstrap_repeats),
            "availability": {"old_only": sorted(values["old_dpo"].keys() - values["new_dpo"].keys()),
                             "new_only": sorted(values["new_dpo"].keys() - values["old_dpo"].keys()),
                             "neither": sorted(set(ids) - values["old_dpo"].keys() - values["new_dpo"].keys())},
        }
    maps = {method: {row["image_id"]: row["validity"] for row in rows[method]} for method in METHODS}
    summary["validity_paired"] = _paired(
        {key: float(value == "valid") for key, value in maps["old_dpo"].items()},
        {key: float(value == "valid") for key, value in maps["new_dpo"].items()}, bootstrap_repeats)
    summary["validity_transitions"] = {
        f"{left}_to_{right}": [image_id for image_id in ids if maps["old_dpo"][image_id] == left and maps["new_dpo"][image_id] == right]
        for left in VALIDITIES for right in VALIDITIES}
    return summary


def _write_csv(output: Path, ids: list[str], rows: dict[str, list[dict]]) -> None:
    columns = ("image_id", "method", "seed", "inference_status", "evaluation_status", "validity",
               "post_complete", "observation_status", "audit_status", "inference_seconds",
               "evaluation_seconds", "audit_seconds", *METRICS)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for index, _ in enumerate(ids):
            for method in METHODS:
                writer.writerow({key: rows[method][index].get(key) for key in columns})


def _plot(summary: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in ("Archivo", "Jost", "DejaVu Sans") if name in available), "DejaVu Sans")
    labels = ["Original SVR", "Old-label DPO", "Observation DPO"]
    colors = ["#71839B", "#BB784C", "#168578"]
    with plt.rc_context({"font.family": family, "font.size": 11, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.7))
        rates = [100 * summary["methods"][method]["rates_over_intended"]["validity_valid"] for method in METHODS]
        axes[0].bar(labels, rates, color=colors)
        axes[0].set(ylim=(0, 100), ylabel="Native valid solid (%)", title=f"All {summary['intended_count']} CADs")
        for index, value in enumerate(rates):
            axes[0].text(index, value + 1, f"{value:.2f}%", ha="center")
        for axis, key, label in ((axes[1], "observation_silhouette_iou", "Silhouette IoU"),
                                 (axes[2], "observation_visible_face_cd_sq", "Visible surface CD squared")):
            paired = summary["metrics"][key]["paired_old_new_common"]
            if paired["n"]:
                axis.bar(labels[1:], [paired["old_dpo_mean"], paired["new_dpo_mean"]], color=colors[1:])
            else:
                axis.text(.5, .5, "No common measurements", transform=axis.transAxes, ha="center")
            axis.set(ylabel=label, title=f"Same measurable CADs: n={paired['n']}")
        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="x", labelrotation=15)
        fig.suptitle("Fixed step 5000 | K=1 | matched inputs, seeds and post-processing")
        fig.text(.02, .015, "Missing observations are not zero. Paired deltas and 95% bootstrap intervals are in summary.json and README.md.", fontsize=9)
        fig.tight_layout(rect=(0, .055, 1, .93))
        fig.savefig(output / "comparison.png", dpi=180)
        fig.savefig(output / "comparison.svg")
        plt.close(fig)


def _readme(summary: dict, provenance: dict) -> str:
    def number(value):
        return "—" if value is None else f"{value:.6g}"
    lines = ["# 新旧 DPO 固定 validation 对照", "",
             f"预定 {summary['intended_count']} 个 CAD；原 SVR、旧标签 DPO 和观测标签 DPO 使用同一输入、K=1、seed 与后处理预算。两种 DPO 均固定 step5000。",
             "", "| 方法 | 合法 solid | 非法 | unknown | 三轮 post 完整 | 观测可测 |", "|---|---:|---:|---:|---:|---:|"]
    for method, label in zip(METHODS, ("原 SVR", "旧标签 DPO", "观测标签 DPO")):
        counts = summary["methods"][method]["counts"]
        lines.append(f"| {label} | {counts['validity_valid']}/{summary['intended_count']} | {counts['validity_invalid']} | {counts['validity_unknown']} | {counts['post_complete']} | {counts['observation_available']} |")
    lines += ["", "## 新 DPO 减旧 DPO", "", "几何与观测均值只比较两者共同可测 CAD；合法率始终用完整预定分母。缺测保留为 null，各自可测集分布与缺测方向见 summary.json。", "",
              "| 指标 | 共同 n | 旧 DPO 均值 | 新 DPO 均值 | 新减旧 | 95% 配对 bootstrap CI |",
              "|---|---:|---:|---:|---:|---|"]
    comparisons = [("validity_rate", summary["validity_paired"])] + [
        (key, summary["metrics"][key]["paired_old_new_common"]) for key in METRICS]
    for key, item in comparisons:
        interval = "—" if item["ci95"] is None else f"[{number(item['ci95'][0])}, {number(item['ci95'][1])}]"
        lines.append(f"| {key} | {item['n']} | {number(item['old_dpo_mean'])} | {number(item['new_dpo_mean'])} | {number(item['new_minus_old'])} | {interval} |")
    lines += ["", f"CAD 配对 bootstrap {summary['bootstrap']['repeats']:,} 次，seed42，百分位95%区间。IoU、F1、合法率增大更好；CD²、边界距离、深度误差减小更好。post 先对每 CAD 的完整三轮测量取均值。",
              "", "可见表面距离与深度依赖 GT，属于评估监督；K=1 不能评估多样性。观测可测不要求 native solid 合法，二者独立报告；unknown 只在合法率二值统计中记为未获证实合法，不能据此称为确认非法。",
              "", "本报告不按单个指标或置信区间自动判定研究方向；应同时检查实体有效性、观测忠实度和完整几何的取舍。",
              "", "![比较](comparison.png)", "", "## 来源", ""]
    lines += [f"- {name}: `{path}`" for name, path in provenance["sources"].items()]
    if provenance["checkpoint_sha256"]["old_dpo"] == provenance["checkpoint_sha256"]["new_dpo"]:
        lines += ["", "**旧新 DPO checkpoint 摘要相同：这是同权重接口核对，不能作为新模型实验结果。**"]
    return "\n".join(lines) + "\n"


def write_report(old_run: Path, new_run: Path, old_audit: Path, new_audit: Path,
                 output_dir: Path, *, expected_count: int = 3505,
                 bootstrap_repeats: int = 10000, resume: bool = False) -> dict:
    output = Path(output_dir).resolve()
    if expected_count < 1 or bootstrap_repeats < 1:
        raise ValueError("Expected count and bootstrap repeats must be positive")
    for source in (old_run, new_run, old_audit, new_audit):
        source = Path(source).resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Report output must be independent of source runs")
    output.mkdir(parents=True, exist_ok=resume)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        ids, rows, provenance = load_study(old_run, new_run, old_audit, new_audit, expected_count=expected_count)
        provenance.update(version=VERSION, expected_count=expected_count, bootstrap_repeats=bootstrap_repeats,
                          source_sha256={str(path): sha256_file(path) for path in
                                         (Path(__file__), Path(__file__).with_name("preference_report.py"))})
        provenance["fingerprint"] = fingerprint(provenance)
        previous = output / "provenance.json"
        if resume and previous.is_file():
            if json.loads(previous.read_text()) != provenance:
                raise ValueError("Resume requires unchanged report settings and all source artifacts")
        else:
            if resume and any(path.name != ".lock" for path in output.iterdir()):
                raise ValueError("Cannot resume existing artifacts without source provenance")
            atomic_json(previous, provenance)
        atomic_json(output / "status.json", {"status": "writing", "fingerprint": provenance["fingerprint"]})
        summary = summarize(ids, rows, bootstrap_repeats=bootstrap_repeats)
        summary["provenance_fingerprint"] = provenance["fingerprint"]
        atomic_json(output / "summary.json", summary)
        _write_csv(output / "per_object.csv", ids, rows)
        (output / "README.md").write_text(_readme(summary, provenance))
        _plot(summary, output)
        atomic_json(output / "status.json", {"status": "complete", "fingerprint": provenance["fingerprint"],
                    "intended_count": len(ids), "completed_utc": datetime.now(timezone.utc).isoformat()})
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-run", "new-run", "old-audit", "new-audit", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=3505)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = write_report(**vars(args))
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "intended_count": summary["intended_count"],
                      "validity_paired": summary["validity_paired"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
