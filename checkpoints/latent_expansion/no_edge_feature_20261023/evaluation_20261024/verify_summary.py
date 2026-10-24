"""Independently recompute headline statistics from the persisted paired CSV."""
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
VARIANT = "no_edge_feature_last64"
METRICS = (
    "raw_face_cd_sq", "raw_edge_cd_sq", "path_teacher_matched_cd_sq",
    "path_coarse_matched_cd_sq", "path_refined_matched_cd_sq",
)


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14), (actual, expected)


def main():
    summary = json.loads((HERE / "summary.json").read_text())
    table = HERE / "per_object.csv.gz"
    with gzip.open(table, "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped = {}
    for variant in ("source32", VARIANT):
        selected = [row for row in rows if row["variant"] == variant]
        assert len(selected) == 3495
        assert all(row["status"] == "ok" for row in selected)
        assert len({row["id"] for row in selected}) == 3495
        grouped[variant] = {row["id"]: row for row in selected}
    assert set(grouped["source32"]) == set(grouped[VARIANT])
    checks = {}
    for key in METRICS:
        values = {variant: [float(row[key]) for row in group.values()]
                  for variant, group in grouped.items()}
        for variant, numbers in values.items():
            expected = summary["models"][variant]["metrics"][key]
            close(statistics.mean(numbers), expected["mean"])
            close(statistics.median(numbers), expected["median"])
            close(percentile(numbers, .9), expected["p90"])
        differences = [float(grouped[VARIANT][identifier][key]) - float(before[key])
                       for identifier, before in grouped["source32"].items()]
        counts = Counter("improved" if value < 0 else "worsened" if value > 0 else "equal"
                         for value in differences)
        expected_pair = summary["paired"]["metrics"][key]
        for name in ("improved", "worsened", "equal"):
            assert counts[name] == expected_pair[name]
        close(statistics.mean(values[VARIANT]) / statistics.mean(values["source32"]) - 1,
              expected_pair["relative_change"])
        checks[key] = dict(counts)
    result = {
        "status": "passed", "rows": len(rows), "paired_cads": 3495,
        "checks": "independent standard-library mean, median, linear P90, direction counts, ratio of means",
        "metrics": checks,
        "summary_sha256": hashlib.sha256((HERE / "summary.json").read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
    }
    (HERE / "summary_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
