"""Selection and descriptive-adjustment contracts; no model-quality assertions."""
import csv
import json

import pytest

from tools.legacy_failure_select import (
    RAW_METRICS, allocate_quotas, population_comparison, read_table, select_cases, typed_value,
)


def test_hamilton_minimum_rare_stratum_capacity_and_remainder_tie():
    assert allocate_quotas({"rare": 1, "common": 9}, 5) == {"rare": 1, "common": 4}
    assert allocate_quotas({"b": 3, "a": 3}, 3) == {"b": 1, "a": 2}
    assert allocate_quotas({"a": 2, "b": 4}, 6) == {"a": 2, "b": 4}
    assert allocate_quotas({"a": 1, "b": 1}, 2) == {"a": 1, "b": 1}
    with pytest.raises(ValueError):
        allocate_quotas({"a": 2, "b": 1}, 1)
    with pytest.raises(ValueError):
        allocate_quotas({"a": 2}, 3)


def test_selection_is_order_independent_spans_face_count_and_preserves_rare_type():
    ids = [f"{index:08}" for index in range(10)]
    raw = {("legacy", item): {"gt_face_count": index + 1, "topology_fp": 0, "topology_fn": 0}
           for index, item in enumerate(ids)}
    post = {("legacy", item): {"shape_type": 2 if index == 0 else 0, "native_brepcheck_valid": False}
            for index, item in enumerate(ids)}
    chosen, strata = select_cases(ids, raw, post, count=5)
    repeated, other = select_cases(list(reversed(ids)), raw, post, count=5)
    assert chosen == repeated and strata == other
    assert len(chosen) == len(set(chosen)) == 5 and "00000000" in chosen
    common = next(row for row in strata if row["shape_type"] == 0)
    assert common["selected_zero_based_ranks"] == [1, 3, 5, 7]
    assert common["selected_ids"] == ["00000002", "00000004", "00000006", "00000008"]


def test_csv_types_preserve_identity_hash_null_and_boolean_and_reject_duplicates(tmp_path):
    path = tmp_path / "rows.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "method", "native_valid_solid", "shape_type", "face_cd", "input_latent_sha256", "error"])
        writer.writeheader()
        writer.writerow({"sample_id": "00000001", "method": "legacy", "native_valid_solid": False,
                         "shape_type": 3, "face_cd": "1e-5", "input_latent_sha256": "1" * 64, "error": ""})
    row = read_table(path, "sample_id")["legacy", "00000001"]
    assert row["sample_id"] == "00000001" and row["input_latent_sha256"] == "1" * 64
    assert row["native_valid_solid"] is False and row["shape_type"] == 3
    assert row["face_cd"] == 1e-5 and row["error"] is None
    assert json.loads(json.dumps(row)) == row
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines + lines[1:]) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_table(path, "sample_id")
    with pytest.raises(ValueError, match="Nonfinite"):
        typed_value("face_cd", "NaN")


def test_exact_face_count_standardization_removes_composition_only_difference():
    # Identical within-face-count means, different population composition.
    data = [("a", True, 2, 1), ("b", True, 2, 3), ("c", True, 10, 100),
            ("d", False, 2, 2), ("e", False, 10, 100), ("f", False, 10, 100),
            ("g", False, 30, 999)]  # no successful control at this complexity
    raw, post = {}, {}
    for item, valid, faces, metric in data:
        raw["legacy", item] = {key: None for key in RAW_METRICS}
        raw["legacy", item].update(gt_face_count=faces, face_mae=metric)
        post["legacy", item] = {"native_valid_solid": valid}
    summary = population_comparison([item[0] for item in data], raw, post)
    result = summary["face_count_standardized"]["metrics"]["face_mae"]
    assert result["failure_total"] == 4 and result["failure_objects_in_common_support"] == 3
    assert result["exact_face_count_weights"] == {"2": 1 / 3, "10": 2 / 3}
    assert result["failed_minus_valid"] == 0
    assert summary["by_exact_gt_face_count"]["30"]["native_valid"]["object_count"] == 0
    assert summary["unadjusted"]["native_failed"]["metrics"]["face_mae"]["mean"] > summary["unadjusted"]["native_valid"]["metrics"]["face_mae"]["mean"]
    json.dumps(summary, allow_nan=False)
