"""Compare fixed checkpoints on all IDs, with explicit diagnostic coverage.

This task-local reader never relabels or modifies historical observations.
Missing historical traces remain unknown, including for previously valid cases.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SPLIT = REPO / "checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt"
SPLIT_SHA = "2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3"
THRESHOLDS = ("0.002", "0.006", "0.01", "0.08")
COHORTS = ("both_failed", "new_failure", "rescued", "both_valid")
LABELS = {"both_failed": "持续失败", "new_failure": "新增失败", "rescued": "救回", "both_valid": "持续成功"}
EXPECTED = {
    "early_1089": "d9e911e23a5dfc0e45fb90735a806706c9c89329898f0873df06927c8d2fb62d",
    "previous_14520": "e9120c78254d9b94aaaef88333c0f0656fd6a01346225d092fb8341297bff2bf",
    "resume_best_27588": "e1e579e7bc6bdf64b3c078d6b5414e33630222de989b180ca8f3eb22b59c6ccf",
    "last_28677": "040c05a45dd7edae825f2e6dd8e2cb3489a418146f9a7b2122cc54d38070618c",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def boolean(value: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(f"Not a recorded boolean: {value!r}")
    return value == "True"


def load_snapshot(name: str, evaluation: Path, census: Path, ids: set[str]) -> dict:
    post_path = evaluation / "post/optimized/samples.csv"
    raw_path = evaluation / "raw_metrics/posterior_mean/per_object.csv"
    post = [row for row in csv.DictReader(post_path.open()) if row["method"] == "encoder_edge_value"]
    raw = [row for row in csv.DictReader(raw_path.open()) if row["method"] == "encoder_edge_value"]
    if len(post) != len(ids) or len(raw) != len(ids):
        raise ValueError(f"Incomplete fixed population: {name}")
    if {row["sample_id"] for row in post} != ids or {row["model_id"] for row in raw} != ids:
        raise ValueError(f"Wrong or duplicate population: {name}")
    if any(row["encoder_checkpoint_sha256"] != EXPECTED[name] for row in raw):
        raise ValueError(f"Wrong checkpoint: {name}")
    outcome = {row["sample_id"]: boolean(row["native_valid_solid"]) for row in post}
    case_path, summary_path = census / "cases.json", census / "summary.json"
    review_path = census / "independent_review.json"
    review = read(review_path)
    if review["status"] != "passed" or review["cases_sha256"] != digest(case_path) or review["summary_sha256"] != digest(summary_path):
        raise ValueError(f"Independent review no longer matches: {name}")
    cases = [pair["candidate"] for pair in read(case_path)["pairs"]]
    records = {row["id"]: row for row in cases}
    if len(cases) != len(records) or not records.keys() <= ids:
        raise ValueError(f"Diagnostic ID mismatch: {name}")
    for sid, row in records.items():
        if row["method"] != "encoder_edge_value" or row["original_native_valid"] is not outcome[sid]:
            raise ValueError(f"Historical identity or outcome changed: {name}/{sid}")
        for source in row["source_files"].values():
            if source["sha256"] != digest(Path(source["path"])):
                raise ValueError(f"Historical source changed: {name}/{sid}/{source['path']}")
    return {
        "name": name, "outcome": outcome, "records": records,
        "source": {"checkpoint_sha256": EXPECTED[name], "diagnostic_count": len(records),
                   "native_valid": sum(outcome.values()), "native_failed": len(ids) - sum(outcome.values()),
                   "assets": [{"path": str(path), "sha256": digest(path)} for path in (post_path, raw_path, case_path, summary_path, review_path)]},
    }


def cohort(old: bool, new: bool) -> str:
    return "both_valid" if old and new else "rescued" if new else "new_failure" if old else "both_failed"


def comparison(old: dict, new: dict, ids: set[str], keys: tuple[str, ...]) -> dict:
    groups = {group: sorted(sid for sid in ids if cohort(old["outcome"][sid], new["outcome"][sid]) == group) for group in COHORTS}
    result = {"old": old["name"], "new": new["name"], "fixed_population": len(ids), "groups": {}}
    for group, selected in groups.items():
        old_available = [sid for sid in selected if sid in old["records"]]
        new_available = [sid for sid in selected if sid in new["records"]]
        common = sorted(set(old_available) & set(new_available))
        verified = [sid for sid in common if old["records"][sid]["trace_verified"] and new["records"][sid]["trace_verified"]]
        verified_set = set(verified)
        observations = {}
        for threshold in THRESHOLDS:
            observations[threshold] = {}
            for key in keys:
                counts = Counter({category: 0 for category in ("both_present", "both_absent", "present_to_absent", "absent_to_present", "unknown")})
                for sid in selected:
                    a, b = (snapshot["records"][sid]["flags"][threshold][key] if sid in verified_set else None for snapshot in (old, new))
                    category = "unknown" if type(a) is not bool or type(b) is not bool else "both_present" if a and b else "both_absent" if not a and not b else "present_to_absent" if a else "absent_to_present"
                    counts[category] += 1
                if sum(counts.values()) != len(selected):
                    raise AssertionError("Observation denominator lost")
                observations[threshold][key] = dict(counts)
        result["groups"][group] = {
            "count": len(selected), "ids": selected,
            "coverage": {"old_available": len(old_available), "new_available": len(new_available),
                         "common_available": len(common), "common_verified": len(verified),
                         "missing_or_unverified": len(selected) - len(verified), "common_verified_ids": verified},
            "terminal_transitions_common_verified": dict(Counter(f"{old['records'][sid]['terminal_stage']} -> {new['records'][sid]['terminal_stage']}" for sid in verified)),
            "observations": observations,
        }
    if sum(group["count"] for group in result["groups"].values()) != len(ids):
        raise AssertionError("Cohort denominator lost")
    result["solid_net_change"] = result["groups"]["rescued"]["count"] - result["groups"]["new_failure"]["count"]
    return result


def main() -> None:
    if digest(SPLIT) != SPLIT_SHA:
        raise ValueError("Fixed split changed")
    ids = set(SPLIT.read_text().split())
    if len(ids) != 2417:
        raise ValueError("Wrong fixed population")
    topic = ROOT.parent
    sources = {
        "early_1089": (topic / "evaluation_20261015/candidate", topic / "failure_census_20261015"),
        "previous_14520": (topic / "evaluation_20261015/endpoint_14520/candidate", topic / "failure_census_20261015/endpoint_14520_comparison"),
        "resume_best_27588": (topic / "evaluation_20261016/resume_best", ROOT / "resume_best"),
        "last_28677": (topic / "evaluation_20261016/last", ROOT / "last"),
    }
    snapshots = {name: load_snapshot(name, evaluation, census, ids) for name, (evaluation, census) in sources.items()}
    keys = tuple(next(iter(snapshots["early_1089"]["records"].values()))["flags"]["0.006"])
    if len(keys) != 28:
        raise ValueError("Observation inventory differs")
    comparisons = [comparison(snapshots[old], snapshots[new], ids, keys)
                   for old, new in (("early_1089", "last_28677"), ("previous_14520", "last_28677"),
                                    ("early_1089", "resume_best_27588"), ("previous_14520", "resume_best_27588"),
                                    ("resume_best_27588", "last_28677"))]
    output = {"schema": "encoder_checkpoint_temporal_census_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "split": {"path": str(SPLIT), "sha256": SPLIT_SHA, "count": len(ids)},
              "reader_sha256": digest(Path(__file__)), "snapshots": {name: snapshot["source"] for name, snapshot in snapshots.items()},
              "interpretation": "Full 2417-ID outcomes determine cohorts. Only same-ID verified traces compare observations; unavailable/unverified history is unknown, never negative. No causal architecture conclusion or representative-success claim.",
              "comparisons": comparisons}
    (ROOT / "temporal_comparison.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# 固定 checkpoint 的同 ID 时间比较", "", "所有成败分组从完整2,417件重新计算；诊断范围仍由各权重相对conditioned的失败/救回决定，因此阶段观察只在两时点均有核验轨迹的共同ID比较。未覆盖或未核验记为未知，不能当作告警消失。", "",
             "| 旧→新 | 持续失败 | 新增失败 | 救回 | 持续成功 | solid净变化 |", "|---|---:|---:|---:|---:|---:|"]
    for item in comparisons:
        counts = [str(item["groups"][group]["count"]) for group in COHORTS]
        lines.append("| " + item["old"] + " → " + item["new"] + " | " + " | ".join(counts) + f" | {item['solid_net_change']:+d} |")
    lines += ["", "| 旧→新 | 组别 | 对象数 | 共同核验轨迹 | 轨迹缺失/未核验 |", "|---|---|---:|---:|---:|"]
    for item in comparisons:
        for group in COHORTS:
            row = item["groups"][group]
            lines.append(f"| {item['old']} → {item['new']} | {LABELS[group]} | {row['count']} | {row['coverage']['common_verified']} | {row['coverage']['missing_or_unverified']} |")
    lines += ["", "每项观察的四阈值配对转换、完整共同ID、互斥末端转换和原始来源SHA见 [temporal_comparison.json](temporal_comparison.json)。所有旧记录只读且重新核验其来源摘要；这里不填补旧轨迹、不改写旧cohort，不将成功样本中的局部覆盖当作全部成功件的统计。", ""]
    (ROOT / "temporal_REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"status": "complete", "comparisons": len(comparisons), "snapshot_counts": {name: len(snapshot["records"]) for name, snapshot in snapshots.items()}}))


if __name__ == "__main__":
    main()
