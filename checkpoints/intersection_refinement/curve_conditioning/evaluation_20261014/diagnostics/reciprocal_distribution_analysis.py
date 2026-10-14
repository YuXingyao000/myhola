"""Audit existing reciprocal scalar metrics, without recomputing curve geometry."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from raw_distribution_analysis import METHODS, ROOT, OUT, SPLIT, SPLIT_SHA, digest, stats, value, paired, verify_stats, number

METRICS = ("legacy_reciprocal_cd_sq_mean", "legacy_reciprocal_cd_sq_max")


def ratio(a, b):
    return float(a / b) if b else None


def measure(table, ids, metric):
    record, common, old, new = paired(table, ids, metric)
    delta = new - old
    positive = np.maximum(delta, 0)
    count = math.ceil(.01 * len(common))
    rankings = {"candidate_error": sorted(range(len(common)), key=lambda i: (-new[i], common[i])),
                "positive_delta": sorted(range(len(common)), key=lambda i: (-positive[i], common[i]))}
    record["tail"] = {}
    for name, ranked in rankings.items():
        selected, remain = ranked[:count], ranked[count:]
        a, b = stats(old[remain]), stats(new[remain])
        record["tail"][name + "_top1pct"] = {
            "count": count, "ids": [common[i] for i in selected],
            "candidate_error_share": ratio(math.fsum(new[selected]), math.fsum(new)),
            "baseline_same_id_error_share": ratio(math.fsum(old[selected]), math.fsum(old)),
            "positive_delta_share": ratio(math.fsum(positive[selected]), math.fsum(positive)),
            "remaining_count": len(remain), "remaining_baseline_mean": a["mean"], "remaining_candidate_mean": b["mean"],
            "remaining_mean_ratio": ratio(b["mean"], a["mean"]) if a["mean"] is not None else None,
            "top10": [{"model_id": common[i], "baseline": float(old[i]), "candidate": float(new[i]),
                       "delta": float(delta[i]), "reciprocal_pair_count_baseline": table[METHODS[0], common[i]]["legacy_reciprocal_pair_count"],
                       "reciprocal_pair_count_candidate": table[METHODS[1], common[i]]["legacy_reciprocal_pair_count"]} for i in ranked[:10]]}
    record["positive_delta_total"] = math.fsum(positive)
    record["net_delta_total"] = math.fsum(delta)
    return record


def report(data):
    names = {METRICS[0]: "每件平均互反差距", METRICS[1]: "每件最差互反差距"}
    labels = {"all": "全体", "both_adjacency_exact": "双方 GT 邻接全对", "remaining": "其余对象"}
    lines = ["# 互反半边差距：分布、尾部与邻接子集", "",
             "这里复算现有 CSV 的逐对象标量，不重算底层几何。每件平均/最差都是该 CAD 内无向面对之间两份半边的平方 CD，再在 CAD 集合上汇总；不检查曲线方向。", "",
             "## 分布与同 ID 变化", "",
             "| 输入/集合 | 指标 | 共同数/缺测 | 均值倍数 | baseline 中位数 → 本版 | P90 倍数 | P95 倍数 | 改善/退步/持平 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for mode, groups in data["modes"].items():
        for cohort, metrics in groups["cohorts"].items():
            for key, row in metrics["metrics"].items():
                a, b = row[METHODS[0]], row[METHODS[1]]
                lines.append(f"| {mode}/{labels[cohort]} | {names[key]} | {row['common_count']}/{row['missing_count']} | {number(row['mean_ratio'])} | {number(a['median'])} → {number(b['median'])} | {number(ratio(b['p90'], a['p90']))} | {number(ratio(b['p95'], a['p95']))} | {row['win_count']}/{row['loss_count']}/{row['tie_count']} |")
    lines += ["", "倍数为候选/基线的对应统计量；整体中位数之比与逐对象倍数的中位数不同。低于1表示该统计量下降。", "",
              "## 全体的尾部贡献", "",
              "候选最大1%按候选误差排序；正增量最大1%按候选−基线的正值排序。排除时两方法移除相同ID，仅作敏感性分析。", "",
              "| 输入 | 指标 | 选取 | 件数 | 候选总误差份额 | 全部正增量份额 | 剩余均值倍数 |", "|---|---|---|---:|---:|---:|---:|"]
    for mode, groups in data["modes"].items():
        for key, row in groups["cohorts"]["all"]["metrics"].items():
            for name, tail in row["tail"].items():
                label = "候选最大1%" if name.startswith("candidate") else "正增量最大1%"
                lines.append(f"| {mode} | {names[key]} | {label} | {tail['count']} | {number(tail['candidate_error_share'], True)} | {number(tail['positive_delta_share'], True)} | {number(tail['remaining_mean_ratio'])} |")
    lines += ["", "## 邻接已相同的范围", "",
              f"posterior_mean 中双方 GT 面邻接均全对的 {data['both_adjacency_exact_count']} 件使用既有冻结 ID 清单，并逐行核验 FP=FN=0。两方法的无向面 pair 集合均等于 GT，因此在此子集中可比较相同的面 pair 集合。全对集合中互反pair数量一致的对象数为 {data['exact_cohort_pair_counts_equal']}；双方均无有向重复/单向pair的对象数为 {data['exact_cohort_clean_reciprocal_count']}。", ""]
    posterior = data["modes"]["posterior_mean"]["cohorts"]
    for key in METRICS:
        total = posterior["all"]["metrics"][key]["positive_delta_total"]
        exact = posterior["both_adjacency_exact"]["metrics"][key]["positive_delta_total"]
        lines.append(f"{names[key]}：双方邻接全对集合贡献全部正增量的 {number(ratio(exact, total), True)}；其余对象贡献 {number(ratio(total - exact, total), True)}。")
    lines += ["", "CSV 未保存其余对象的完整 pair 集合，不能仅凭 pair 数相同断言集合相同。这些统计不隔离几何、训练曝光与模块的因果影响；不能据此归因于有序 conditioning。", "",
              "## posterior 平均互反差距：正增量前10 ID", "", "| ID | 双方邻接全对 | baseline | 本版 | 差值 | 互反pair数（旧/新） |", "|---|---|---:|---:|---:|---:|"]
    exact_ids = set(data["both_adjacency_exact_ids"])
    for row in posterior["all"]["metrics"][METRICS[0]]["tail"]["positive_delta_top1pct"]["top10"]:
        lines.append(f"| {row['model_id']} | {'是' if row['model_id'] in exact_ids else '否'} | {number(row['baseline'])} | {number(row['candidate'])} | {number(row['delta'])} | {number(row['reciprocal_pair_count_baseline'])}/{number(row['reciprocal_pair_count_candidate'])} |")
    lines += ["", "全部均值、中位数、P90/P95/P99、同ID倍数与差值分布、极端ID及来源核验见 [reciprocal_distribution.json](reciprocal_distribution.json)。", ""]
    return "\n".join(lines)


def main():
    outputs = [OUT / name for name in ("reciprocal_distribution.json", "RECIPROCAL.md")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Reciprocal diagnostics already exist")
    ids = sorted(line.strip() for line in SPLIT.read_text().splitlines() if line.strip())
    if digest(SPLIT) != SPLIT_SHA or len(ids) != 2417 or len(set(ids)) != len(ids):
        raise ValueError("Frozen split differs")
    adjacency_path = OUT / "adjacency_subset.json"
    adjacency = json.loads(adjacency_path.read_text())
    exact_ids = adjacency["cohorts"]["both_adjacency_exact"]["ids"]
    result = {"methods": METHODS, "modes": {}, "summary_checks": [],
              "both_adjacency_exact_ids": exact_ids, "both_adjacency_exact_count": len(exact_ids),
              "source_sha256": {str(path): digest(path) for path in (SPLIT, adjacency_path, OUT / "raw_distribution_analysis.py")},
              "script_sha256": digest(Path(__file__)), "spec_sha256": digest(OUT / "RECIPROCAL_SPEC.md")}
    for mode in ("posterior_mean", "archived_latents"):
        source = ROOT / "raw_metrics" / mode / "per_object.csv"
        summary_path = source.with_name("summary.json")
        result["source_sha256"].update({str(path): digest(path) for path in (source, summary_path)})
        summary = json.loads(summary_path.read_text())
        if (summary["mode"] != mode or summary["baseline_method"] != METHODS[0] or summary["candidate_method"] != METHODS[1]
                or summary["intended_count"] != 2417 or summary["split_sha256"] != SPLIT_SHA or not summary["protocol_valid"]):
            raise ValueError("Raw summary protocol differs")
        with source.open() as handle:
            rows = list(csv.DictReader(handle))
        lookup = {(row["method"], row["model_id"]): row for row in rows}
        if len(lookup) != len(rows) or set(lookup) != {(method, item) for method in METHODS for item in ids}:
            raise ValueError("Incomplete or duplicate CSV IDs")
        table = {}
        for item in ids:
            a, b = (lookup[m, item] for m in METHODS)
            if (a.get("shared_input_status") != b.get("shared_input_status") or a.get("shared_input_status") != "verified"
                    or not a.get("input_latent_sha256") or a["input_latent_sha256"] != b.get("input_latent_sha256")):
                raise ValueError("CSV latent provenance differs")
            for method in METHODS:
                table[method, item] = {key: value(lookup[method, item], key) for key in (*METRICS, "legacy_reciprocal_pair_count")}
        for key in METRICS:
            for method in METHODS:
                actual = stats([table[method, item][key] for item in ids if table[method, item][key] is not None])
                result["summary_checks"].append(verify_stats(actual, summary["methods"][method]["metrics"][key], f"{mode}/{method}/{key}"))
        cohorts = {"all": ids}
        if mode == "posterior_mean":
            derived_exact = [item for item in ids if all(lookup[m, item].get("topology_status") == "ok"
                             and value(lookup[m, item], "topology_fp") == value(lookup[m, item], "topology_fn") == 0 for m in METHODS)]
            if sorted(exact_ids) != derived_exact:
                raise ValueError("Adjacency subset IDs/conditions differ")
            cohorts.update(both_adjacency_exact=exact_ids, remaining=sorted(set(ids) - set(exact_ids)))
            result["exact_cohort_pair_counts_equal"] = sum(table[METHODS[0], item]["legacy_reciprocal_pair_count"] == table[METHODS[1], item]["legacy_reciprocal_pair_count"] for item in exact_ids)
            result["exact_cohort_clean_reciprocal_count"] = sum(all(value(lookup[m, item], key) == 0 for m in METHODS
                                                               for key in ("legacy_directed_duplicate_count", "legacy_unilateral_pair_count")) for item in exact_ids)
        result["modes"][mode] = {"cohorts": {name: {"count": len(samples), "metrics": {key: measure(table, samples, key) for key in METRICS}} for name, samples in cohorts.items()}}
    for path, expected in result["source_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError("Diagnostic input changed")
    document = report(result)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    for path, text in zip(outputs, (serialized, document)):
        with path.open("x") as handle:
            handle.write(text)
    print(json.dumps({"outputs": [str(path) for path in outputs], "summary_checks_passed": len(result["summary_checks"])}, indent=2))


if __name__ == "__main__":
    main()
