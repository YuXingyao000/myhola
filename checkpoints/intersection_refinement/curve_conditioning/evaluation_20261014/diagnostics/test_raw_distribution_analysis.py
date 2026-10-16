"""Focused checks for missing cohorts, paired denominators and source validation."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tools.vae_compare_metrics import statistics

spec = importlib.util.spec_from_file_location("curve_raw_diagnostic", Path(__file__).with_name("raw_distribution_analysis.py"))
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def fixture_data():
    ids = ["a", "b", "c"]
    table = {}
    for method, values in zip(diagnostic.METHODS, ((1., None, 0.), (2., 3., 0.))):
        for item, value in zip(ids, values):
            table[method, item] = {
                **dict.fromkeys(diagnostic.METRICS, value),
                **dict.fromkeys(diagnostic.COMPLEXITY, 4.),
                "topology_fp": 0., "topology_fn": 0., "topology_f1": 1., "topology_status": "ok"}
    table["conditioned", "b"]["topology_fp"] = 1.
    table["conditioned", "c"].update(topology_fp=None, topology_fn=None, topology_status="not_computable")
    summary = {"methods": {m: {"metrics": {}} for m in diagnostic.METHODS}, "paired": {}}
    for key in diagnostic.METRICS:
        common = [i for i in ids if all(table[m, i][key] is not None for m in diagnostic.METHODS)]
        summary["paired"][key] = {"common_count": len(common)}
        for method in diagnostic.METHODS:
            summary["methods"][method]["metrics"][key] = statistics([table[method, i][key] for i in ids if table[method, i][key] is not None])
            summary["paired"][key][method] = statistics([table[method, i][key] for i in common])
    return ids, table, summary


def test_independent_statistics_and_empty_set():
    values = [-2., 0., 0.001, 1., 23.]
    actual = diagnostic.stats(values)
    assert actual["mean"] == pytest.approx(np.mean(values))
    assert actual["median"] == np.median(values)
    assert actual["p95"] == pytest.approx(np.percentile(values, 95))
    assert diagnostic.stats([])["mean"] is None
    assert diagnostic.ratio(1, 0) is None
    with pytest.raises(ValueError, match="Invalid"):
        diagnostic.value({"edge_cd_sq": "nan"}, "edge_cd_sq")


def test_independent_means_do_not_mix_own_and_paired_denominators():
    ids, table, summary = fixture_data()
    result, rows, extremes = diagnostic.analyze_mode(table, summary, ids, "posterior_mean")
    edge = result["metrics"]["edge_cd_sq"]
    assert edge["common_count"] == 2 and edge["missing_count"] == 1
    assert edge["conditioned"]["mean"] == 1.
    assert edge["all_computable"]["conditioned"]["mean"] == pytest.approx(5 / 3)
    assert edge["loss_count"] == 1 and edge["tie_count"] == 1
    assert edge["tail"]["top_1pct"]["removed_ids"] == ["a"]
    assert len(rows) == 3 and len(result["summary_checks"]) == 20
    summary["methods"]["conditioned"]["metrics"]["edge_cd_sq"]["mean"] = 1.
    with pytest.raises(ValueError, match="CSV/summary mismatch"):
        diagnostic.analyze_mode(table, summary, ids, "posterior_mean")


def test_adjacency_unknown_remains_separate_and_empty_subsets_are_valid():
    ids, table, _ = fixture_data()
    result = diagnostic.adjacency_cohorts(table, ids)
    assert result["cohorts"]["both_adjacency_exact"]["ids"] == ["a"]
    assert result["cohorts"]["at_least_one_adjacency_inexact"]["ids"] == ["b"]
    assert result["cohorts"]["unknown"]["ids"] == ["c"]
    for item in ids:
        table["conditioned", item]["topology_status"] = "not_computable"
    result = diagnostic.adjacency_cohorts(table, ids)
    assert result["cohorts"]["both_adjacency_exact"]["metrics"]["edge_cd_sq"]["mean_ratio"] is None
    assert result["cohorts"]["unknown"]["count"] == 3


def test_report_retains_counts_with_provenance_and_does_not_assert_a_result():
    ids, table, summary = fixture_data()
    record, _, extremes = diagnostic.analyze_mode(table, summary, ids, "posterior_mean")
    adjacency = diagnostic.adjacency_cohorts(table, ids)
    adjacency.update(sources_sha256={}, script_sha256="test", method_names=diagnostic.METHODS)
    text = diagnostic.build_report({"modes": {mode: record for mode in diagnostic.MODES}}, adjacency, extremes)
    assert "本版曲线条件调制：整件邻接全对 1/2417" in text
    assert "至少一方不可计算" in text
    assert "局部注意力" not in text and "明显改善" not in text


def test_read_mode_rejects_duplicate_ids_and_latent_mismatch(tmp_path):
    ids, table, summary = fixture_data()
    summary.update(mode="posterior_mean", baseline_method="intersection", candidate_method="conditioned",
                   protocol_valid=True, split_sha256=diagnostic.SPLIT_SHA, intended_count=3)
    directory = tmp_path / "raw_metrics/posterior_mean"
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(json.dumps(summary))
    rows = [{**values, "method": method, "model_id": item, "shared_input_status": "verified",
             "input_latent_sha256": item * 64} for (method, item), values in table.items()]

    def write(rows):
        with (directory / "per_object.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write(rows)
    actual, _, sources = diagnostic.read_mode(tmp_path, "posterior_mean", ids)
    assert actual == table and len(sources) == 2
    write([*rows, rows[0]])
    with pytest.raises(ValueError, match="coverage"):
        diagnostic.read_mode(tmp_path, "posterior_mean", ids)
    rows[0]["input_latent_sha256"] = "different"
    write(rows)
    with pytest.raises(ValueError, match="Unverified"):
        diagnostic.read_mode(tmp_path, "posterior_mean", ids)
