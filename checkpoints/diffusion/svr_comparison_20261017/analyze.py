"""Paired bootstrap for the frozen validation100 reports; no model/data loading.

Only finalized reports are accepted. Resampling CAD IDs estimates variation
across validation objects at K=1; it does not estimate generation-seed variation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
from itertools import combinations
import json
import math
from pathlib import Path
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parent
METHODS = ("hola_1500k", "current_best_54639", "current_last_182130")
SPLIT_SHA256 = "6c952c3556860c0a2782a0bedbf139b48bc2769b385788baf0fcb82fac1be587"
SEED = 42
RESAMPLES = 10_000
RAW_TERMINAL = {"ok", "missing_raw", "unreadable_raw", "invalid_raw"}
# Frozen report contract from tools.vae_compare_post; no OCC dependency/import.
POST_TERMINAL = {
    "ok", "missing_raw", "invalid_raw", "post_error", "no_step",
    "step_read_error", "geometry_error", "timeout", "crash", "interrupted",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finite_number(value: str | None) -> float | None:
    """A missing/nonfinite metric has no numeric fallback."""
    if value is None or not value.strip():
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"Malformed numeric metric: {value!r}") from exc
    return number if math.isfinite(number) else None


def native_valid(value: str | None) -> int:
    """Use the native-validity stage result, even if later geometry timed out."""
    if value is None or not value.strip():
        return 0
    value = value.strip().lower()
    if value in {"true", "1"}:
        return 1
    if value in {"false", "0"}:
        return 0
    raise ValueError(f"Malformed native_valid_solid value: {value!r}")


def index_rows(
    payload: bytes, ids: list[str], terminal: set[str], stage: str,
) -> dict[str, dict[str, dict]]:
    """Require one terminal record per frozen ID for each primary model."""
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    if not {"sample_id", "method", "status"}.issubset(reader.fieldnames or []):
        raise ValueError(f"{stage}: missing ID/method/status columns")
    indexed = {method: {} for method in METHODS}
    intended = set(ids)
    for row in reader:
        method = row["method"]
        if method not in indexed:
            continue  # Extra alignment candidates are outside this comparison.
        sample_id = row["sample_id"]
        if sample_id not in intended or sample_id in indexed[method]:
            raise ValueError(f"{stage}: unexpected/duplicate ID {method}/{sample_id}")
        if row["status"] not in terminal:
            raise ValueError(f"{stage}: nonterminal/unknown status {method}/{sample_id}: {row['status']!r}")
        indexed[method][sample_id] = row
    for method, records in indexed.items():
        if set(records) != intended:
            raise ValueError(f"{stage}: incomplete frozen roster for {method}: {len(records)}/{len(ids)}")
    return indexed


def paired_estimate(
    ids: list[str], left: dict[str, float], right: dict[str, float], *, higher_is_better: bool,
) -> dict:
    """Resample paired differences, never the two methods independently."""
    common = [sample_id for sample_id in ids if sample_id in left and sample_id in right]
    n = len(common)
    result = {
        "n": n, "ids": common, "excluded_ids": [i for i in ids if i not in common],
        "left_available_n": len(left), "right_available_n": len(right),
        "difference": "right_minus_left", "higher_is_better": higher_is_better,
        "favorable_difference_for_right": "positive" if higher_is_better else "negative",
        "left_mean": None, "right_mean": None, "mean_difference": None,
        "ci95_percentile": [None, None], "differences_in_id_order": [],
    }
    if not n:
        result["status"] = "no_common_computable_objects"
        return result
    left_values = np.asarray([left[i] for i in common], dtype=np.float64)
    right_values = np.asarray([right[i] for i in common], dtype=np.float64)
    difference = right_values - left_values
    # Reset seed for every comparison: adding another metric cannot alter a CI.
    indices = np.random.default_rng(SEED).integers(0, n, size=(RESAMPLES, n))
    bootstrap_means = difference[indices].mean(axis=1)
    interval = np.percentile(bootstrap_means, [2.5, 97.5], method="linear")
    result.update(
        status="ok", left_mean=float(left_values.mean()), right_mean=float(right_values.mean()),
        mean_difference=float(difference.mean()), ci95_percentile=interval.tolist(),
        differences_in_id_order=difference.tolist(),
    )
    return result


def analyze_rows(ids: list[str], raw: dict, processed: dict) -> dict:
    metrics = {"raw_face_cd_sq": {}, "raw_face_f1": {}, "native_valid_solid": {}}
    for method in METHODS:
        for field, name in (("face_cd_sq", "raw_face_cd_sq"), ("face_f1", "raw_face_f1")):
            values = {i: finite_number(raw[method][i].get(field)) for i in ids}
            values = {i: value for i, value in values.items() if value is not None}
            if any(v < 0 or (field == "face_f1" and v > 1) for v in values.values()):
                raise ValueError(f"Out-of-range {method}/{field}")
            metrics[name][method] = values
        # Explicit missing/failed native validity stays in the fixed denominator.
        metrics["native_valid_solid"][method] = {
            i: native_valid(processed[method][i].get("native_valid_solid")) for i in ids
        }
    pairs = {}
    for left, right in combinations(METHODS, 2):
        item = {"left": left, "right": right, "metrics": {}}
        for metric, methods in metrics.items():
            result = paired_estimate(ids, methods[left], methods[right], higher_is_better=metric != "raw_face_cd_sq")
            if metric == "native_valid_solid":
                transitions = Counter((methods[left][i], methods[right][i]) for i in ids)
                result["transitions"] = {
                    "both_valid": transitions[1, 1], "both_invalid": transitions[0, 0],
                    "left_valid_right_invalid": transitions[1, 0],
                    "left_invalid_right_valid": transitions[0, 1],
                }
                result["mean_difference_percentage_points"] = 100 * result["mean_difference"]
                result["ci95_percentage_points"] = [100 * v for v in result["ci95_percentile"]]
                result["denominator"] = "All 100 frozen objects, including terminal failures/missing outputs."
            else:
                result["denominator"] = "Metric-specific common finite values; failed/missing values are not zero-imputed."
            item["metrics"][metric] = result
        pairs[f"{left}__{right}"] = item
    return pairs


def analyze_report(root: Path) -> tuple[dict, dict[Path, bytes]]:
    """Read finalized CSVs and their completion evidence as one checked snapshot."""
    report = root / "report"
    files = [root / "validation100.txt", report / "summary.json"]
    files += [report / f"{stage}_per_object.csv" for stage in ("raw", "post", "inference")]
    snapshots = {path: path.read_bytes() for path in files}
    split = snapshots[root / "validation100.txt"]
    ids = split.decode().splitlines()
    if sha256(split) != SPLIT_SHA256 or len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("This analysis requires the frozen validation100.txt")
    summary = json.loads(snapshots[report / "summary.json"])
    if (summary.get("split_sha256") != SPLIT_SHA256
            or summary.get("all_post_terminal") is not True
            or summary.get("all_inference_terminal") is not True):
        raise ValueError("summary.json does not certify a complete terminal report")
    raw = index_rows(snapshots[report / "raw_per_object.csv"], ids, RAW_TERMINAL, "raw")
    processed = index_rows(snapshots[report / "post_per_object.csv"], ids, POST_TERMINAL, "post")
    index_rows(snapshots[report / "inference_per_object.csv"], ids, {"ok", "error"}, "inference")
    analysis = {
        "methods": list(METHODS), "intended_count": 100, "split_sha256": SPLIT_SHA256,
        "bootstrap": {
            "unit": "paired CAD ID", "resamples": RESAMPLES, "seed": SEED,
            "rng": "numpy.random.default_rng (PCG64), reset for each pair/metric",
            "interval": "percentile 95%", "quantiles": [2.5, 97.5], "quantile_method": "linear",
            "statistic": "mean(right - left)", "numpy_version": np.__version__,
        },
        "limitations": [
            "Validation100, K=1: object-resampling uncertainty conditional on these saved predictions.",
            "The intervals do not include generation-seed variation, retraining variation or model-selection uncertainty.",
            "Marginal 95% intervals are not corrected for multiple model/metric comparisons.",
            "Raw continuous metrics condition on each pair's common computable IDs; coverage must be read separately.",
            "Native validity uses all 100 IDs. A recorded True remains valid after a later geometry_error/timeout; this is not whole-pipeline success.",
            "Checkpoint/system comparison, not an equal-training-budget architecture ablation or test-set result.",
        ],
        "input_sha256": {str(path.relative_to(root)): sha256(data) for path, data in snapshots.items()},
        "analysis_code_sha256": sha256(Path(__file__).read_bytes()),
        "status_counts": {
            stage: {method: dict(Counter(row["status"] for row in records.values()))
                    for method, records in grouped.items()}
            for stage, grouped in (("raw", raw), ("post", processed))
        },
        "pairs": analyze_rows(ids, raw, processed),
    }
    return analysis, snapshots


def self_check() -> None:
    """Synthetic fixtures verify pairing, stage-specific validity and rejection."""
    constant = paired_estimate(["a", "b"], {"a": 1, "b": 100}, {"a": 3, "b": 102}, higher_is_better=False)
    assert constant["mean_difference"] == 2 and constant["ci95_percentile"] == [2, 2]
    balanced = paired_estimate(["a", "b"], {"a": 0, "b": 1}, {"a": 1, "b": 0}, higher_is_better=True)
    assert balanced["mean_difference"] == 0 and balanced["ci95_percentile"] == [-1, 1]
    empty = paired_estimate(["a", "b"], {"a": 1}, {"b": 1}, higher_is_better=False)
    assert empty["n"] == 0 and empty["ci95_percentile"] == [None, None]
    assert native_valid("False") == 0 and native_valid("True") == 1 and native_valid("") == 0

    split = (ROOT / "validation100.txt").read_bytes()  # IDs only, no prediction/model/data assets.
    ids = split.decode().splitlines()
    with tempfile.TemporaryDirectory(prefix="svr-paired-analysis-selfcheck-") as directory:
        root = Path(directory)
        report = root / "report"
        report.mkdir()
        (root / "validation100.txt").write_bytes(split)
        (report / "summary.json").write_text(json.dumps({
            "split_sha256": SPLIT_SHA256, "all_post_terminal": True, "all_inference_terminal": True,
        }))
        raw, processed, inference = [], [], []
        for method_index, method in enumerate(METHODS):
            for index, sample_id in enumerate(ids):
                base = {"method": method, "sample_id": sample_id, "status": "ok"}
                raw.append({**base, "face_cd_sq": index + 2 * method_index, "face_f1": .2 + .1 * method_index})
                processed.append({**base, "native_valid_solid": index < (50 if method_index == 0 else 100)})
                inference.append(base)
        # Missing metric changes only its own common set; post missing stays /100.
        raw[100]["face_cd_sq"] = ""
        processed[0]["status"] = "timeout"  # Its completed native-validity True survives.
        processed[100].update(status="missing_raw", native_valid_solid="")
        inference[100] = {**inference[100], "status": "error"}
        for name, rows in (("raw", raw), ("post", processed), ("inference", inference)):
            with (report / f"{name}_per_object.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(reversed(rows))  # Pair by ID, not CSV position.
        result, _ = analyze_report(root)
        pair = result["pairs"][f"{METHODS[0]}__{METHODS[1]}"]["metrics"]
        assert pair["raw_face_cd_sq"]["n"] == 99 and pair["raw_face_cd_sq"]["ci95_percentile"] == [2, 2]
        assert pair["raw_face_f1"]["n"] == 100 and math.isclose(pair["raw_face_f1"]["mean_difference"], .1)
        valid = pair["native_valid_solid"]
        assert valid["n"] == 100 and valid["left_mean"] == .5 and valid["right_mean"] == .99
        assert math.isclose(valid["mean_difference"], .49) and valid["transitions"]["left_valid_right_invalid"] == 1
        assert analyze_report(root)[0] == result
        payload = (report / "post_per_object.csv").read_bytes()
        for malformed in (
            payload.replace(b"timeout", b"running"),
            b"\n".join(payload.splitlines()[:-1]) + b"\n",
            payload + payload.splitlines()[1] + b"\n",
        ):
            try:
                index_rows(malformed, ids, POST_TERMINAL, "post")
            except ValueError:
                pass
            else:
                raise AssertionError("Nonterminal, missing or duplicate record was accepted")
        summary = json.loads((report / "summary.json").read_text())
        summary["all_inference_terminal"] = False
        (report / "summary.json").write_text(json.dumps(summary))
        try:
            analyze_report(root)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete inference report was accepted")
    print("Self-check passed: paired CIs, metric-specific common IDs, /100 validity, terminal guards and deterministic output.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--self-check", action="store_true", help="Use synthetic temporary CSVs; never analyze the experiment outputs.")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    analysis, snapshots = analyze_report(args.root)
    for path, original in snapshots.items():
        if path.read_bytes() != original:
            raise RuntimeError(f"Report changed during analysis: {path}")
    destination = args.root / "report" / "paired_analysis.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(destination)
    print(f"Paired analysis written: {destination}")


if __name__ == "__main__":
    main()
