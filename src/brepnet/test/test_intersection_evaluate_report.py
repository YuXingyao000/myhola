"""Report input contracts; fixtures are synthetic and never research results."""

import csv
import json

import pytest

from tools import intersection_evaluate_report as report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def statistics(value, count=4):
    return {"count": count, "mean": value, "median": value}


@pytest.fixture
def completed_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "EXPECTED_INTENDED", 4)
    monkeypatch.setattr(report, "EXPECTED_ORIGINAL", 7)
    ids = ["a", "b", "c", "d"]
    for mode in ("posterior_mean", "archived_latents"):
        metrics = ("face_cd_sq", "edge_cd_sq")
        if mode == "posterior_mean":
            metrics += ("face_mae", "topology_precision", "topology_recall", "topology_f1")
        rows = [dict(model_id=item, method=method, status="ok", shared_input_status="verified") for method in report.METHODS for item in ids]
        summary = {
            "intended_count": 4, "original_count": 7, "mode": mode,
            "protocol_valid": True, "split_sha256": "fixture-only-digest",
            "shared_input_status_counts": {"verified": 4},
            "methods": {method: {"status_counts": {"ok": 4}, "metrics": {
                "legacy_reciprocal_cd_sq_mean": statistics(1.0),
                "legacy_reciprocal_cd_sq_max": statistics(2.0),
                "legacy_unilateral_pair_count": statistics(0.0),
            }} for method in report.METHODS},
            "paired": {metric: {"common_count": 4, "legacy": statistics(0.5), "intersection": statistics(0.4)} for metric in metrics},
        }
        directory = tmp_path / "raw_metrics" / mode
        write_json(directory / "summary.json", summary)
        write_csv(directory / "per_object.csv", rows)

    for mode in report.POST_MODES:
        rows, methods = [], {}
        for method in report.METHODS:
            successful = {"a", "b"} if method == "legacy" else {"a", "c"}
            value = 1.0 if method == "legacy" else 2.0
            for item in ids:
                rows.append({
                    "sample_id": item, "method": method, "status": "ok",
                    "native_valid_solid": item in successful, "compat_valid_solid": item in successful,
                    "step_exists": True, "step_readable": True, "success_marker": item in successful,
                    "geometry_status": "ok", "face_cd": value, "edge_cd": value,
                })
            counts = {"native_valid_solid": 2, "compat_valid_solid": 2, "step_exists": 4, "step_readable": 4, "success_marker": 2, "geometry_computable": 4, "terminal_records": 4}
            methods[method] = {
                "counts": counts, "status_counts": {"ok": 4},
                "rates_over_intended": {key: value / 4 for key, value in counts.items()},
                "rates_over_original_test": {key: value / 7 for key, value in counts.items()},
            }
        summary = {
            "intended_count_per_method": 4, "original_test_count": 7,
            "methods": methods, "protocol": {"mode": mode, "split_sha256": "fixture-only-digest"},
            "paired_common_geometry": {"count": 4, "metrics": {
                name: {"legacy": statistics(1.0), "intersection": statistics(2.0)} for name in ("face_cd", "edge_cd")
            }},
        }
        directory = tmp_path / "post" / mode
        write_json(directory / "summary.json", summary)
        write_csv(directory / "samples.csv", rows)
    write_json(tmp_path / "training_audit.json", {"checkpoint_recommendation": {"path": "validation-best.ckpt"}, "validation_history": [{"epoch": 0}]})
    return tmp_path


def numerical_fixture():
    return {
        "sample_ids": ["a", "b"], "repeats_per_method_per_id": 5,
        "encoder_equal_tensor_count": 243,
        "cpu_algebra_check": [{"exact_equal": True, "max_abs": 0.0}],
        "summary": {
            "max_legacy_repeat": 2.89e-6, "max_legacy_vs_cache": 4.29e-6, "max_legacy_vs_new": 3.34e-6,
            "legacy_runs_pass_atol1e6": 6, "new_runs_pass_atol1e6": 6,
            "legacy_runs_pass_atol1e5": 10, "new_runs_pass_atol1e5": 10,
        },
        "numerical_audit_tolerance": {"previous_atol": 1e-6, "atol": 1e-5, "rtol": 1e-5},
    }


def test_success_transitions_use_same_ids_and_keep_unsupported_separate(completed_fixture):
    data = report.load_comparison(completed_fixture)
    for mode in report.POST_MODES:
        transition = data["transitions"][mode]
        assert transition["counts"] == {name: 1 for name in ("old_failure_new_success", "old_success_new_failure", "both_success", "both_failure")}
        assert transition["ids"]["old_failure_new_success"] == ["c"]
        assert transition["ids"]["old_success_new_failure"] == ["b"]
        assert transition["contract_unsupported_count"] == 3


def test_geometry_timeout_does_not_erase_a_solid_already_confirmed(completed_fixture):
    directory = completed_fixture / "post/optimized"
    with (directory / "samples.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["stage"] = "done"
        if row["method"] == "intersection" and row["sample_id"] == "a":
            row.update(status="timeout", stage="geometry", geometry_status="running", face_cd="", edge_cd="")
    write_csv(directory / "samples.csv", rows)
    summary = report.read_json(directory / "summary.json")
    summary["methods"]["intersection"]["status_counts"] = {"ok": 3, "timeout": 1}
    summary["methods"]["intersection"]["counts"]["geometry_computable"] = 3
    summary["paired_common_geometry"]["count"] = 3
    for metric in ("face_cd", "edge_cd"):
        for method in report.METHODS:
            summary["paired_common_geometry"]["metrics"][metric][method]["count"] = 3
    write_json(directory / "summary.json", summary)
    data = report.load_comparison(completed_fixture)
    assert data["timeout_diagnostics"]["optimized"]["intersection"] == {"all_stages": 1, "already_valid_solid": 1, "geometry_stage": 1}
    assert data["transitions"]["optimized"]["ids"]["both_success"] == ["a"]


@pytest.mark.parametrize("fault", ["missing_file", "pending_row", "duplicate_id", "denominator", "mean_mismatch", "median_mismatch", "nonfinite"])
def test_incomplete_or_inconsistent_results_are_rejected(completed_fixture, fault):
    directory = completed_fixture / "post/fit_only"
    summary_path = directory / "summary.json"
    if fault == "missing_file":
        summary_path.unlink()
    elif fault in ("pending_row", "duplicate_id"):
        csv_path = directory / "samples.csv"
        with csv_path.open() as handle:
            rows = list(csv.DictReader(handle))
        if fault == "pending_row":
            rows[0]["status"] = "pending"
        else:
            rows.append(rows[0].copy())
        write_csv(csv_path, rows)
    else:
        data = report.read_json(summary_path)
        if fault == "denominator":
            data["original_test_count"] += 1
        elif fault == "median_mismatch":
            data["paired_common_geometry"]["metrics"]["face_cd"]["legacy"]["median"] = 3.0
        else:
            data["paired_common_geometry"]["metrics"]["face_cd"]["legacy"]["mean"] = float("nan") if fault == "nonfinite" else 3.0
        write_json(summary_path, data)
    with pytest.raises((ValueError, FileNotFoundError)):
        report.load_comparison(completed_fixture)
    assert not (completed_fixture / "REPORT.md").exists()


def test_report_refuses_overwrite_before_reading_or_rendering(tmp_path):
    destination = tmp_path / "REPORT.md"
    destination.write_text("existing report")
    with pytest.raises(FileExistsError, match="overwrite"):
        report.generate_report(tmp_path)
    assert destination.read_text() == "existing report"
    assert not (tmp_path / "visualizations").exists()


def test_absent_numerical_audit_is_optional(tmp_path):
    assert report.load_numerical_audit(tmp_path) is None
    assert report.numerical_audit_paragraphs(None) == []


def test_numerical_audit_distinguishes_reencoding_tolerance_from_exact_latent_inputs(tmp_path):
    write_json(tmp_path / "numerical_audit.json", numerical_fixture())
    write_json(tmp_path / "validation_summary.json", {"first_full_inference": {"exit_codes": [1, 0, 1]}})
    (tmp_path / "metadata/initial_mean_audit_failed").mkdir(parents=True)
    record = report.load_numerical_audit(tmp_path)
    text = "\n".join(report.numerical_audit_paragraphs(record))
    assert "2/3 个分片" in text
    assert "4.29e-06" in text and "3.34e-06" in text
    assert "缓存 latent 数组和 SHA 校验始终严格相同" in text
    assert "initial_mean_audit_failed" in text
    assert "初轮成功子集没有混入最终统计" in text
    assert "1/1 件" in text


@pytest.mark.parametrize("fault", ["duplicate_ids", "bad_count", "bad_tolerance"])
def test_corrupt_present_numerical_audit_is_not_silently_ignored(tmp_path, fault):
    data = numerical_fixture()
    if fault == "duplicate_ids":
        data["sample_ids"] = ["a", "a"]
    elif fault == "bad_count":
        data["summary"]["new_runs_pass_atol1e5"] = 11
    else:
        data["numerical_audit_tolerance"]["atol"] = -1
    write_json(tmp_path / "numerical_audit.json", data)
    with pytest.raises(ValueError):
        report.load_numerical_audit(tmp_path)


def test_reciprocal_table_does_not_claim_direction_or_loop_correctness(completed_fixture):
    raw = report.load_comparison(completed_fixture)["raw"]
    raw["posterior_mean"]["methods"]["intersection"]["metrics"]["legacy_reciprocal_cd_sq_mean"]["mean"] = 0.8
    raw["posterior_mean"]["methods"]["intersection"]["metrics"]["legacy_reciprocal_cd_sq_max"]["mean"] = 2.1
    text = "\n".join(report.reciprocal_comparison(raw))
    assert "误差降低 20.00%" in text
    assert "误差增加 5.00%" in text
    assert "不检查半边方向" in text
    assert "不是固定相同 pair 的消融" in text
    assert "显式对称化" in text and "不能解读为模型已经学会拓扑闭合" in text
