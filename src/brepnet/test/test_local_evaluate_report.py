"""Keep new baseline/candidate identities, completeness gates and artifact content."""
from pathlib import Path
import csv
import json

import pytest

from tools import intersection_evaluate_report as generic
from tools import local_evaluate_report as local
from src.brepnet.test.test_intersection_evaluate_report import (
    completed_fixture, statistics, write_csv, write_json,
)


def rename_methods(value):
    mapping = {"legacy": "intersection", "intersection": "local"}
    if isinstance(value, dict):
        return {mapping.get(key, key): rename_methods(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rename_methods(item) for item in value]
    return mapping.get(value, value) if isinstance(value, str) else value


@pytest.fixture
def local_fixture(completed_fixture, monkeypatch):
    root = completed_fixture
    monkeypatch.setattr(local, "EXPECTED_INTENDED", 4)
    monkeypatch.setattr(local, "EXPECTED_ORIGINAL", 7)
    for path in root.rglob("*.json"):
        value = rename_methods(json.loads(path.read_text()))
        if path.name == "summary.json":
            value.update(baseline_method="intersection", candidate_method="local")
        write_json(path, value)
    for path in root.rglob("*.csv"):
        with path.open() as handle:
            rows = [rename_methods(row) for row in csv.DictReader(handle)]
        write_csv(path, rows)
    for mode in ("posterior_mean", "archived_latents"):
        path = root / "raw_metrics" / mode / "summary.json"
        value = generic.read_json(path)
        for method in local.METHODS:
            value["methods"][method]["coverage"] = {name: {"count": 4} for name in
                ("files_present", "face_geometry_computable", "edge_geometry_computable", "topology_scores_computable")}
        write_json(path, value)
    for mode in local.POST_MODES:
        path = root / "post" / mode / "summary.json"
        value = generic.read_json(path)
        for method in local.METHODS:
            value["methods"][method]["geometry_all_computable"] = {
                metric: value["paired_common_geometry"]["metrics"][metric][method]
                for metric in ("face_cd", "edge_cd")}
        write_json(path, value)
    write_json(root / "training_audit.json", {
        "checkpoint_recommendation": {"path": "val_best.ckpt", "epoch": 19, "global_step": 7260, "best_model_score": .5},
        "last_checkpoint": {"epoch": 30, "global_step": 10000}, "validation_history": [{"epoch": 19}]})
    return root


def test_refinement_local_contract_preserves_transition_direction_and_labels(local_fixture):
    data = generic.load_comparison(local_fixture, methods=local.METHODS)
    transition = data["transitions"]["optimized"]
    assert transition["ids"]["old_failure_new_success"] == ["c"]
    assert transition["ids"]["old_success_new_failure"] == ["b"]
    text = local.build_report(data)
    assert "上一版 Intersection refinement（`intersection`）" in text
    assert "本版局部注意力（`local`）" in text
    assert "原 HoLa 仅用于" in text
    assert "各自可计算" in text and "共同可计算" in text
    assert "等预算" in text and "STEP SHA 本次补记" in text


def test_local_report_rejects_old_comparison_identity(completed_fixture):
    with pytest.raises(ValueError, match="Unexpected method|Wrong mode/methods"):
        generic.load_comparison(completed_fixture, methods=local.METHODS)


def test_local_report_refuses_overwrite_and_renders_chinese_artifacts(local_fixture):
    font = Path("/root/.cache/hola-browser-libs/fonts/NotoSansCJKsc-Regular.otf")
    if not font.is_file():
        pytest.skip("Optional system CJK font unavailable")
    result = local.generate_report(local_fixture, font)
    assert Path(result["png"]).stat().st_size > 1000
    assert Path(result["svg"]).stat().st_size > 1000
    assert "本版局部注意力" in Path(result["svg"]).read_text()
    assert json.loads(Path(result["transitions"]).read_text())["optimized"]["counts"]["old_success_new_failure"] == 1
    with pytest.raises(FileExistsError, match="overwrite"):
        local.generate_report(local_fixture, font)
