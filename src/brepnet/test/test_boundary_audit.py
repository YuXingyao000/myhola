"""Audit failures must remain visible when materializing a training subset."""

import csv
import hashlib

from tools.audit_shared_boundaries import audit_sample, write_accepted_list


def test_missing_sample_is_a_failure_row_not_a_dropped_id(tmp_path):
    row = audit_sample("missing", data_root=tmp_path, reciprocal_tolerance=.01,
                       closed_tolerance=1e-5, num_queries=128)
    assert row["model_id"] == "missing"
    assert row["status"] == "failed"
    assert "FileNotFoundError" in row["error"]
    assert row["num_boundaries"] is None


def test_explicit_subset_preserves_rejections_and_order(tmp_path):
    rows = [
        {"model_id": "b", "status": "ok", "error": ""},
        {"model_id": "invalid", "status": "failed", "error": "reciprocal disagreement"},
        {"model_id": "a", "status": "ok", "error": ""},
    ]
    result = write_accepted_list(rows, tmp_path)
    text = (tmp_path / "accepted.txt").read_text()
    assert text == "b\na\n"
    assert result["accepted_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    with (tmp_path / "rejected.csv").open() as handle:
        assert list(csv.DictReader(handle)) == [rows[1]]
