"""Read-only ablation reporting: identity, provenance, failures and paired CIs."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.eval import preference_ablation_report as report
from src.common.io import atomic_json, fingerprint, sha256_file


IDS = ["cad_b", "cad_a", "cad_c"]


def _protocol(path, value, *, nested_audit=False):
    value = deepcopy(value)
    value.pop("fingerprint", None)
    if nested_audit:
        value["config"].pop("fingerprint", None)
    value["fingerprint"] = fingerprint(value)
    if nested_audit:
        value["config"]["fingerprint"] = value["fingerprint"]
    atomic_json(path, value)
    return value


@pytest.fixture
def study(tmp_path):
    raw, images = tmp_path / "raw", tmp_path / "images"
    split = tmp_path / "split.txt"
    split.write_text("\n".join(IDS) + "\n")
    for image_id in IDS:
        for path in (raw / image_id / "data.npz", raw / image_id / "normalized_shape.step", images / image_id / "imgs.npz"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"immutable {image_id} {path.name}".encode())
    old, new = tmp_path / "old", tmp_path / "new"
    old_audit, new_audit = tmp_path / "old_audit", tmp_path / "new_audit"
    protocols = {}
    for run, checkpoint in ((old, "old-policy"), (new, "new-policy")):
        protocol = _protocol(run / "metadata/protocol.json", {
            "version": "svr_dpo_fixed_validation_v1", "config": {
                "run_dir": str(run), "raw_data_root": str(raw), "condition_data_root": str(images),
                "split_list": str(split), "seed": 42,
            }, "identities": IDS, "split_sha256": sha256_file(split),
            "checkpoint_sha256": {"original_checkpoint": "same-svr", "preference_checkpoint": checkpoint},
            "source_sha256": {"src/eval/preference.py": "same-measurement"},
            "sampling": "DDPM1000 FP32 K1", "post": "recover_v1 repeats3",
        })
        protocols[run] = protocol
        atomic_json(run / "status.json", {"status": "complete", "intended_per_model": 3})
        (run / "validation.txt").write_text(split.read_text())
        for method in ("baseline", "dpo_5000"):
            for image_id in IDS:
                raw_dir = run / "raw_predictions" / method / image_id
                evaluation_dir = run / "evaluations" / method / image_id
                if run == new and method == "baseline":
                    for destination, origin in ((raw_dir, old / "raw_predictions" / method / image_id),
                                                (evaluation_dir, old / "evaluations" / method / image_id)):
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.symlink_to(origin, target_is_directory=True)
                    continue
                seed = int.from_bytes(hashlib.sha256(f"42:{image_id}:diffusion:0".encode()).digest()[:4], "little")
                inference = {"method": method, "image_id": image_id, "seed": seed, "seconds": 1., "status": "ok"}
                if run == new and image_id == "cad_c":
                    atomic_json(raw_dir / "inference.json", {**inference, "status": "generation_error", "error": "fixture failure"})
                    continue
                raw_dir.mkdir(parents=True)
                for name in ("data.npz", "latent.npy"):
                    (raw_dir / name).write_bytes(f"{run.name}-{method}-{image_id}-{name}".encode())
                inference["input_sha256"] = {"image": sha256_file(images / image_id / "imgs.npz"),
                    "gt_npz": sha256_file(raw / image_id / "data.npz"), "gt_step": sha256_file(raw / image_id / "normalized_shape.step")}
                inference["output_sha256"] = {name: sha256_file(raw_dir / name) for name in ("data.npz", "latent.npy")}
                atomic_json(raw_dir / "inference.json", inference)
                ep = {
                    "version": "svr_preference_metrics_v1", "image_id": image_id, "candidate_id": "c00",
                    "seed": 42, "repeats": 3, "timeout_seconds": 180, "max_optimize_iter": 200,
                    "raw_path": str(raw_dir / "data.npz"), "gt_npz": str(raw / image_id / "data.npz"),
                    "gt_step": str(raw / image_id / "normalized_shape.step"), "output_dir": str(evaluation_dir),
                    "gt_cache_root": str(run / "cache"), "sampling": {"version": "same"}, "face_match_threshold": .1,
                    "code_sha256": {"metric.py": "same-metric"},
                    "input_sha256": {"raw_path": inference["output_sha256"]["data.npz"],
                                      "gt_npz": inference["input_sha256"]["gt_npz"], "gt_step": inference["input_sha256"]["gt_step"]},
                }
                ep["protocol_fingerprint"] = fingerprint(ep)
                atomic_json(evaluation_dir / "protocol.json", ep)
                step = evaluation_dir / "post/c00/recon_brep.step"
                step.parent.mkdir(parents=True)
                step.write_bytes(f"step {run.name} {method} {image_id}".encode())
                atomic_json(step.with_name("post_diagnostics.json"), {"status": "fixture"})
                valid = not (run == old and method == "dpo_5000" and image_id == "cad_b")
                metric = {"cad_a": 1., "cad_b": 2., "cad_c": 3.}[image_id]
                if run == new:
                    metric = {"cad_a": .8, "cad_b": 1.5}[image_id]
                measured = {"face_cd_sq": metric, "edge_cd_sq": 2 * metric, "face_f1": .8,
                            "pred_face_count": 2, "gt_face_count": 2}
                atomic_json(evaluation_dir / "metrics.json", {
                    "image_id": image_id, "candidate_id": "c00", "protocol_fingerprint": ep["protocol_fingerprint"],
                    "status": "ok", "validity": "valid" if valid else "invalid", "raw": measured,
                    "post": [{"repeat": index, **measured} for index in range(3)] if valid else [],
                    "native": {"step_readable": True}, "step_path": str(step), "step_sha256": sha256_file(step), "wall_seconds": 2.,
                })
    observation_protocol = {"version": "fixture-observation", "resolution": 224, "mesh_deflection": .002,
                            "mask_threshold": 245, "view_id": 0, "image_key": "svr_imgs", "renderer_dependencies": {"version": "same"}}
    for audit, run, methods in ((old_audit, old, ["baseline", "dpo_5000"]), (new_audit, new, ["dpo_5000"])):
        ap = _protocol(audit / "protocol.json", {
            "version": "single_view_diffusion_audit_v1", "config": {"comparison_run": str(run), "output_dir": str(audit),
                "methods": methods, "workers": 2 if run == old else 4, "timeout": 120., "distance_threshold": .01,
                "resolution": 224, "mesh_deflection": .002, "mask_threshold": 245},
            "runtime": {"pythonocc": "fixture"}, "comparison_protocol_sha256": sha256_file(run / "metadata/protocol.json"),
            "source_sha256": {"/fixed/observation.py": "same-observation"},
        }, nested_audit=True)
        cases = []
        for method in methods:
            for image_id in IDS:
                paths = report._case_paths(run, protocols[run], method, image_id)
                evaluation = json.loads(paths["evaluation_path"].read_text()) if paths["evaluation_path"].exists() else {}
                sources = {name: {"path": str(path.resolve()), "sha256": sha256_file(path) if path.exists() else None} for name, path in paths.items()}
                obs = {"status": "ok", "silhouette_iou": .7 if run == old else .8,
                       "visible_face_cd_sq": .1 if run == old else .08, "silhouette_boundary_distance": .02,
                       "depth_mae": None if image_id == "cad_b" and run == old else .05,
                       "overlap_coverage": .9, "gt_input_silhouette_iou": .99,
                       "gt_input_silhouette_boundary_distance": .001, "protocol": observation_protocol}
                if not evaluation:
                    obs = {"status": "missing_step"}
                atomic_json(audit / "cases" / method / image_id / "audit.json", {
                    "method": method, "image_id": image_id, "candidate_id": "c00", "status": "ok",
                    "audit_fingerprint": ap["fingerprint"], "validity": evaluation.get("validity", "unknown"),
                    "sources": sources, "observation": obs, "elapsed_seconds": .1,
                })
                cases.append({"method": method, "image_id": image_id, "candidate_id": "c00"})
        atomic_json(audit / "manifest.json", {"cases": cases, "intended_count": len(cases)})
        atomic_json(audit / "status.json", {"status": "complete", "completed": len(cases), "total": len(cases)})
    return {"old_run": old, "new_run": new, "old_audit": old_audit, "new_audit": new_audit}


def test_complete_three_method_load_preserves_failures_common_sets_and_direction(study):
    ids, rows, provenance = report.load_study(**study, expected_count=3)
    assert ids == IDS
    summary = report.summarize(ids, rows, bootstrap_repeats=200)
    assert summary["methods"]["new_dpo"]["counts"]["validity_unknown"] == 1
    assert summary["methods"]["new_dpo"]["rates_over_intended"]["validity_valid"] == 2 / 3
    assert summary["methods"]["old_dpo"]["counts"]["observation_ok"] == 3  # Invalid solid is still observable.
    raw = summary["metrics"]["raw_face_cd_sq"]
    assert raw["own"]["old_dpo"]["n"] == 3 and raw["own"]["new_dpo"]["n"] == 2
    assert raw["paired_old_new_common"]["new_minus_old"] == pytest.approx(-.35)
    assert raw["paired_old_new_common"]["ci95"][1] < 0
    assert summary["metrics"]["post_face_cd_sq"]["paired_old_new_common"]["n"] == 1
    assert summary["metrics"]["observation_depth_mae"]["paired_old_new_common"]["n"] == 1
    assert raw["availability"]["old_only"] == ["cad_c"]
    assert summary["validity_transitions"]["invalid_to_valid"] == ["cad_b"]
    assert summary["validity_transitions"]["valid_to_unknown"] == ["cad_c"]
    assert summary == report.summarize(ids, rows, bootstrap_repeats=200)
    assert provenance["checkpoint_sha256"]["old_dpo"] != provenance["checkpoint_sha256"]["new_dpo"]


def test_report_outputs_csv_nulls_resume_and_changed_inputs_rejected(study, tmp_path):
    output = tmp_path / "report"
    summary = report.write_report(**study, output_dir=output, expected_count=3, bootstrap_repeats=20)
    assert json.loads((output / "status.json").read_text())["status"] == "complete"
    assert (output / "comparison.png").stat().st_size > 1000
    assert (output / "comparison.svg").is_file()
    csv_rows = list(csv.DictReader((output / "per_object.csv").open()))
    assert len(csv_rows) == 9
    failed = next(row for row in csv_rows if row["method"] == "new_dpo" and row["image_id"] == "cad_c")
    assert failed["raw_face_cd_sq"] == "" and failed["observation_silhouette_iou"] == ""
    assert summary == report.write_report(**study, output_dir=output, expected_count=3, bootstrap_repeats=20, resume=True)
    path = study["new_audit"] / "cases/dpo_5000/cad_a/audit.json"
    row = json.loads(path.read_text());row["observation"]["silhouette_iou"] = .9;atomic_json(path, row)
    with pytest.raises(ValueError, match="unchanged report settings"):
        report.write_report(**study, output_dir=output, expected_count=3, bootstrap_repeats=20, resume=True)


@pytest.mark.parametrize("field", ["sampling", "post", "split_sha256"])
def test_cross_run_protocol_drift_rejected(study, field):
    path = study["new_run"] / "metadata/protocol.json"
    protocol = json.loads(path.read_text());protocol[field] = "changed";_protocol(path, protocol)
    with pytest.raises(ValueError):
        report.load_study(**study, expected_count=3)


def test_only_orchestration_source_and_policy_checkpoint_may_differ(study):
    old = report.load_protocol(study["old_run"] / "metadata/protocol.json")
    new = report.load_protocol(study["new_run"] / "metadata/protocol.json")
    new["source_sha256"]["src/eval/preference_compare.py"] = "new-scheduler"
    report.compare_protocols(old, new)
    new["source_sha256"]["src/eval/preference.py"] = "changed-metric"
    with pytest.raises(ValueError, match="metric source differs"):
        report.compare_protocols(old, new)
    new = deepcopy(old);new["checkpoint_sha256"]["original_checkpoint"] = "other-svr"
    with pytest.raises(ValueError, match="Original SVR"):
        report.compare_protocols(old, new)


@pytest.mark.parametrize("target", ["seed", "input", "identity", "raw_bytes", "audit_source", "evaluation", "unfinished"])
def test_per_object_source_and_terminal_failures_are_fatal(study, target):
    inference_path = study["new_run"] / "raw_predictions/dpo_5000/cad_a/inference.json"
    if target in ("seed", "input", "identity"):
        row = json.loads(inference_path.read_text())
        if target == "seed":row["seed"] += 1
        elif target == "input":row["input_sha256"]["image"] = "wrong"
        else:row["image_id"] = "different"
        atomic_json(inference_path, row)
    elif target == "raw_bytes":
        inference_path.with_name("data.npz").write_bytes(b"changed raw")
    elif target == "evaluation":
        path = study["new_run"] / "evaluations/dpo_5000/cad_a/metrics.json"
        row = json.loads(path.read_text());row["raw"]["face_cd_sq"] = .01;atomic_json(path, row)
    else:
        path = study["new_audit"] / "cases/dpo_5000/cad_a/audit.json"
        row = json.loads(path.read_text())
        if target == "audit_source":row["sources"]["step_path"]["sha256"] = "wrong"
        else:row["status"] = "running"
        atomic_json(path, row)
    with pytest.raises(ValueError):
        report.load_study(**study, expected_count=3)


def test_audit_v1_nested_fingerprint_compatibility_is_narrow(study):
    path = study["old_audit"] / "protocol.json"
    value = report.load_protocol(path, audit=True)
    value["config"]["fingerprint"] = "wrong"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="Nested audit fingerprint"):
        report.load_protocol(path, audit=True)


def test_duplicate_audit_manifest_and_comparison_split_rejected(study):
    path = study["new_audit"] / "manifest.json"
    value = json.loads(path.read_text());value["cases"][0] = value["cases"][1];atomic_json(path, value)
    with pytest.raises(ValueError, match="missing or duplicate"):
        report.load_study(**study, expected_count=3)
    (study["new_run"] / "validation.txt").write_text("cad_a\ncad_a\ncad_b\n")
    with pytest.raises(ValueError, match="distinct IDs"):
        report.load_study(**study, expected_count=3)


def test_empty_common_set_and_nonfinite_observation_do_not_impute_zero(study):
    ids, rows, _ = report.load_study(**study, expected_count=3)
    for row in rows["new_dpo"]:
        row["observation_depth_mae"] = float("nan")
    summary = report.summarize(ids, rows, bootstrap_repeats=20)
    paired = summary["metrics"]["observation_depth_mae"]["paired_old_new_common"]
    assert paired["n"] == 0 and paired["new_minus_old"] is None and paired["ci95"] is None
    assert summary["metrics"]["observation_depth_mae"]["own"]["new_dpo"]["mean"] is None
    json.dumps(summary, allow_nan=False)


def test_report_never_writes_inside_source_tree(study):
    with pytest.raises(ValueError, match="independent"):
        report.write_report(**study, output_dir=study["old_run"] / "report", expected_count=3)


def test_resume_after_interruption_during_initial_validation(study, tmp_path):
    output = tmp_path / "interrupted_report"
    output.mkdir()
    (output / ".lock").touch()
    summary = report.write_report(**study, output_dir=output, expected_count=3,
                                  bootstrap_repeats=10, resume=True)
    assert summary["intended_count"] == 3


def test_measured_observation_without_source_links_is_rejected(study):
    path = study["new_audit"] / "cases/dpo_5000/cad_a/audit.json"
    row = json.loads(path.read_text());row["sources"].pop("evaluation_path");atomic_json(path, row)
    with pytest.raises(ValueError, match="incomplete source provenance"):
        report.load_study(**study, expected_count=3)


def test_nonfinite_post_and_face_counts_are_missing_and_explicitly_reported(study):
    evaluation_path = study["new_run"] / "evaluations/dpo_5000/cad_a/metrics.json"
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["raw"]["pred_face_count"] = float("nan")
    evaluation["post"][0]["edge_cd_sq"] = float("inf")
    evaluation_path.write_text(json.dumps(evaluation))
    audit_path = study["new_audit"] / "cases/dpo_5000/cad_a/audit.json"
    audit = json.loads(audit_path.read_text())
    audit["sources"]["evaluation_path"]["sha256"] = sha256_file(evaluation_path)
    atomic_json(audit_path, audit)
    ids, rows, _ = report.load_study(**study, expected_count=3)
    summary = report.summarize(ids, rows, bootstrap_repeats=10)
    counts = summary["methods"]["new_dpo"]["nonfinite_metric_counts"]
    assert counts == {"raw_face_count_abs_error": 1, "post_edge_cd_sq": 1}
    assert summary["metrics"]["post_edge_cd_sq"]["paired_old_new_common"]["n"] == 0
    assert summary["metrics"]["raw_face_count_abs_error"]["own"]["new_dpo"]["n"] == 1
    json.dumps(summary, allow_nan=False)
