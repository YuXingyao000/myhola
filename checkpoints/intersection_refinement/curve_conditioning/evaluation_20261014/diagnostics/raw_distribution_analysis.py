"""Read-only paired raw diagnostic; see RAW_DISTRIBUTION_SPEC.md."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "diagnostics"
REPOSITORY = Path(__file__).resolve().parents[5]
SPLIT = REPOSITORY / "checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt"
SPLIT_SHA = "2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3"
METHODS = ("intersection", "conditioned")
MODES = ("posterior_mean", "archived_latents")
METRICS = ("face_mae", "face_cd_sq", "edge_cd_sq", "edge_pred_to_gt_sq", "edge_gt_to_pred_sq")
COMPLEXITY = ("gt_face_count", "gt_boundary_count", "pred_face_count", "raw_edge_count")
BINS = ((1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, None))
LABELS = {"intersection": "上一版 refinement", "conditioned": "本版曲线条件调制"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value(row, key):
    if row.get(key) in (None, ""):
        return None
    number = float(row[key])
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Invalid {key}: {row.get(key)}")
    return number


def stats(values):
    """Recompute without upstream NumPy mean/median/percentile implementations."""
    ordered = sorted(float(item) for item in values)
    fields = ("mean", "median", "p90", "p95", "p99", "max")
    if not ordered:
        return {"count": 0, **dict.fromkeys(fields)}
    result = {"count": len(ordered), "mean": math.fsum(ordered) / len(ordered),
              "median": statistics.median(ordered), "max": ordered[-1]}
    for q in (90, 95, 99):
        position = (len(ordered) - 1) * q / 100
        low, high = math.floor(position), math.ceil(position)
        result[f"p{q}"] = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return result


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator and numerator is not None else None


def verify_stats(actual, reference, context):
    # Historical raw summaries store P90 but not P95; P95 is newly reported here.
    fields = ("count", "mean", "median", "p90") + (("p95",) if "p95" in reference else ())
    for key in fields:
        a, b = actual[key], reference[key]
        matches = a == b if key == "count" or a is None or b is None else math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-14)
        if not matches:
            raise ValueError(f"CSV/summary mismatch {context}/{key}: {a} != {b}")
    return {"context": context, "count": actual["count"], "fields": fields, "passed": True}


def paired(table, ids, metric):
    common = [item for item in ids if all(table[m, item][metric] is not None for m in METHODS)]
    old, new = (np.array([table[m, item][metric] for item in common], dtype=np.float64) for m in METHODS)
    delta = new - old
    record = {"common_count": len(common), "missing_count": len(ids) - len(common),
              **{m: stats(values) for m, values in zip(METHODS, (old, new))},
              "mean_ratio": ratio(stats(new)["mean"], stats(old)["mean"]),
              "median_ratio": ratio(stats(new)["median"], stats(old)["median"]),
              "win_count": int(np.sum(delta < -1e-12)), "loss_count": int(np.sum(delta > 1e-12)),
              "tie_count": int(np.sum(np.abs(delta) <= 1e-12)),
              "win_fraction": float(np.mean(delta < -1e-12)) if len(common) else None,
              "paired_delta": stats(delta),
              "paired_ratio": stats([b / a for a, b in zip(old, new) if a > 0])}
    if record["win_count"] + record["loss_count"] + record["tie_count"] != len(common):
        raise ValueError("Paired changes do not partition the common cohort")
    return record, common, old, new


def read_mode(root, mode, ids):
    source = root / "raw_metrics" / mode / "per_object.csv"
    summary_path = source.with_name("summary.json")
    sources = {str(path): digest(path) for path in (source, summary_path)}
    summary = json.loads(summary_path.read_text())
    if (summary["mode"] != mode or summary["baseline_method"] != METHODS[0]
            or summary["candidate_method"] != METHODS[1] or not summary["protocol_valid"]
            or summary["split_sha256"] != SPLIT_SHA or summary["intended_count"] != len(ids)):
        raise ValueError(f"Unexpected raw summary contract: {mode}")
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    raw = {(row["method"], row["model_id"]): row for row in rows}
    if len(raw) != len(rows) or set(raw) != {(m, item) for m in METHODS for item in ids}:
        raise ValueError(f"CSV ID/method coverage differs from frozen split: {mode}")
    keys = (*METRICS, *COMPLEXITY, "topology_fp", "topology_fn", "topology_f1")
    table = {}
    for item in ids:
        a, b = (raw[m, item] for m in METHODS)
        if (a.get("shared_input_status") != "verified" or b.get("shared_input_status") != "verified"
                or not a.get("input_latent_sha256") or a["input_latent_sha256"] != b.get("input_latent_sha256")):
            raise ValueError(f"Unverified paired input: {mode}/{item}")
        for method in METHODS:
            table[method, item] = {key: value(raw[method, item], key) for key in keys}
            table[method, item]["topology_status"] = raw[method, item].get("topology_status")
    return table, summary, sources


def analyze_mode(table, summary, ids, mode):
    result = {"intended_count": len(ids), "metrics": {}, "complexity_groups": {}, "summary_checks": []}
    extremes, per_object = [], []
    for item in ids:
        record = {"mode": mode, "model_id": item}
        for method in METHODS:
            record.update({f"{method}_{key}": table[method, item][key] for key in (*METRICS, *COMPLEXITY)})
        for metric in METRICS:
            a, b = (table[m, item][metric] for m in METHODS)
            record[f"{metric}_delta"] = b - a if a is not None and b is not None else None
            record[f"{metric}_ratio"] = ratio(b, a)
        per_object.append(record)
    for metric in METRICS:
        record, common, old, new = paired(table, ids, metric)
        record["all_computable"] = {}
        for method in METHODS:
            own = stats([table[method, item][metric] for item in ids if table[method, item][metric] is not None])
            record["all_computable"][method] = own
            result["summary_checks"].append(verify_stats(own, summary["methods"][method]["metrics"][metric], f"{mode}/{metric}/{method}/all"))
            result["summary_checks"].append(verify_stats(record[method], summary["paired"][metric][method], f"{mode}/{metric}/{method}/paired"))
        if summary["paired"][metric]["common_count"] != len(common):
            raise ValueError(f"Paired common count mismatch: {mode}/{metric}")
        record["tail"], record["spearman"] = {}, {}
        delta = new - old
        ordered = sorted(range(len(common)), key=lambda i: (-new[i], common[i]))
        for fraction in (.01, .05, .10):
            count = math.ceil(fraction * len(common))
            top, remain = ordered[:count], ordered[count:]
            a, b = stats(old[remain]), stats(new[remain])
            record["tail"][f"top_{int(fraction * 100)}pct"] = {
                "count": count, "candidate_error_share": ratio(math.fsum(new[top]), math.fsum(new)),
                "same_ids_baseline_error_share": ratio(math.fsum(old[top]), math.fsum(old)),
                "same_ids_positive_delta_share": ratio(math.fsum(np.maximum(delta[top], 0)), math.fsum(np.maximum(delta, 0))),
                "retained_count": len(remain), "trimmed_baseline_mean": a["mean"], "trimmed_candidate_mean": b["mean"],
                "trimmed_mean_ratio": ratio(b["mean"], a["mean"]),
                "removed_ids": [common[i] for i in top] if fraction == .01 else None}
        top_delta = sorted(range(len(common)), key=lambda i: (-max(delta[i], 0), common[i]))[:math.ceil(.01 * len(common))]
        record["largest_positive_delta_1pct_share"] = ratio(math.fsum(np.maximum(delta[top_delta], 0)), math.fsum(np.maximum(delta, 0)))
        for ckey in COMPLEXITY:
            indices = [i for i, item in enumerate(common) if table[METHODS[1], item][ckey] is not None]
            counts = [table[METHODS[1], common[i]][ckey] for i in indices]
            record["spearman"][ckey] = {}
            for name, values in (("candidate_error", new), ("delta", delta)):
                target = values[indices]
                rho = float(spearmanr(counts, target).statistic) if len(set(counts)) > 1 and len(set(target)) > 1 else None
                record["spearman"][ckey][name] = {"count": len(indices), "rho": rho}
        result["metrics"][metric] = record
        for rank, index in enumerate(ordered[:20], 1):
            item = common[index]
            extremes.append({"mode": mode, "metric": metric, "candidate_error_rank": rank,
                             "model_id": item, "baseline_error": float(old[index]), "candidate_error": float(new[index]),
                             "delta": float(delta[index]), "ratio": ratio(new[index], old[index]),
                             **{key: table[METHODS[1], item][key] for key in COMPLEXITY}})
    for ckey in ("gt_face_count", "pred_face_count"):
        groups, covered = [], []
        for low, high in BINS:
            samples = [item for item in ids if table[METHODS[1], item][ckey] is not None
                       and table[METHODS[1], item][ckey] >= low
                       and (high is None or table[METHODS[1], item][ckey] <= high)]
            covered.extend(samples)
            groups.append({"range": f"{low}-{high}" if high else f"{low}+", "count": len(samples),
                           "metrics": {metric: paired(table, samples, metric)[0] for metric in ("face_cd_sq", "edge_cd_sq")}})
        unknown = sorted(set(ids) - set(covered))
        if len(set(covered)) != len(covered):
            raise ValueError("Overlapping complexity groups")
        result["complexity_groups"][ckey] = {"groups": groups, "unknown_count": len(unknown), "unknown_ids": unknown}
    return result, per_object, extremes


def adjacency_cohorts(table, ids):
    cohorts = {name: [] for name in ("both_adjacency_exact", "at_least_one_adjacency_inexact", "unknown")}
    per_method = {m: {"exact_count": 0, "computable_count": 0} for m in METHODS}
    for item in ids:
        states = []
        for method in METHODS:
            row = table[method, item]
            available = row["topology_status"] == "ok" and all(row[k] is not None for k in ("topology_fp", "topology_fn"))
            exact = row["topology_fp"] == row["topology_fn"] == 0 if available else None
            states.append(exact)
            per_method[method]["computable_count"] += int(available)
            per_method[method]["exact_count"] += int(exact is True)
        key = "unknown" if None in states else "both_adjacency_exact" if all(states) else "at_least_one_adjacency_inexact"
        cohorts[key].append(item)
    result = {"intended_count": len(ids), "methods": per_method, "cohorts": {}}
    for name, samples in cohorts.items():
        result["cohorts"][name] = {"count": len(samples), "ids": samples,
                                  "metrics": {metric: paired(table, samples, metric)[0] for metric in METRICS}}
    if sum(record["count"] for record in result["cohorts"].values()) != len(ids):
        raise ValueError("Adjacency cohorts do not cover intended IDs")
    return result


def number(value, percent=False):
    if value is None:
        return "不可计算"
    return f"{value:.2%}" if percent else f"{value:.7g}"


def build_report(data, adjacency, extremes):
    names = {"face_mae": "面 XYZ MAE", "face_cd_sq": "面平方 CD", "edge_cd_sq": "边平方 CD"}
    edge = data["modes"]["posterior_mean"]["metrics"]["edge_cd_sq"]
    lines = ["# 曲线条件调制：raw 分布与同 ID 变化", "",
             f"posterior mean 边平方 CD 的均值比（本版/上一版）为 {number(edge['mean_ratio'])}，中位数比为 {number(edge['median_ratio'])}；本版改善 {edge['win_count']}/{edge['common_count']} 件（{number(edge['win_fraction'], True)}）。这是当前权重的描述性结果。", "",
             "全部 2417 件保留在主分母。以下几何分布使用每指标共同可计算集合，缺测单列；各自可计算分布另存 JSON。CSV 独立重算的各自与配对 count/mean/median/P90 已与上游 summary 对照；上游未保存 P95，本诊断另行补算。候选存在额外训练曝光，不能把全部变化归因于单一模块。", "",
             "## 分布与逐件变化", "", "CD 为归一化 XYZ 中两个方向均方距离之和；这些值不是最大边面偏差，也不能直接证明误差均低于 1e-3 或 1e-4。分位数分别属于各方法；改善按同 ID 误差下降超过 1e-12 计。", "",
             "| 输入 | 指标 | 方法 | 共同数/缺测 | 均值 | 中位数 | P90 | P95 | 改善/退步/持平 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for mode, record in data["modes"].items():
        for metric, label in names.items():
            row = record["metrics"][metric]
            if mode == "archived_latents" and metric == "face_mae":
                continue
            for method in METHODS:
                s = row[method]
                changes = f"{row['win_count']}/{row['loss_count']}/{row['tie_count']}" if method == METHODS[1] else "—"
                lines.append(f"| {mode} | {label} | {LABELS[method]} | {row['common_count']}/{row['missing_count']} | {number(s['mean'])} | {number(s['median'])} | {number(s['p90'])} | {number(s['p95'])} | {changes} |")
    lines += ["", "## 固定去尾敏感性", "", "按候选误差最大 1%（ceil）选 ID，并从双方同时移除。此表不能替代全体主均值或用来重新选择 checkpoint；5%/10%与正增量贡献见 JSON。", "",
              "| 输入 | 指标 | 去除/剩余 | 候选总误差份额 | 同 ID baseline 份额 | 剩余 baseline 均值 | 剩余候选均值 | 剩余均值比 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for mode, record in data["modes"].items():
        for metric, label in names.items():
            row = record["metrics"][metric]
            if not row["common_count"]:
                continue
            tail = row["tail"]["top_1pct"]
            lines.append(f"| {mode} | {label} | {tail['count']}/{tail['retained_count']} | {number(tail['candidate_error_share'], True)} | {number(tail['same_ids_baseline_error_share'], True)} | {number(tail['trimmed_baseline_mean'])} | {number(tail['trimmed_candidate_mean'])} | {number(tail['trimmed_mean_ratio'])} |")
    lines += ["", "## 两方 GT 面邻接均正确的子集", "", "要求双方无向面 pair 集合均满足 FP=FN=0 且拓扑评分有效；不等于 coedge 方向、wire 顺序、几何边界或闭合 solid 正确，也不构成因果隔离。", "",
              "| 集合 | 件数 | 边 CD 共同数 | baseline 均值 | 本版均值 | 均值比 | 本版改善 |", "|---|---:|---:|---:|---:|---:|---:|"]
    cohort_labels = {"both_adjacency_exact": "双方邻接全对", "at_least_one_adjacency_inexact": "至少一方邻接不全对", "unknown": "至少一方不可计算"}
    for name, cohort in adjacency["cohorts"].items():
        row = cohort["metrics"]["edge_cd_sq"]
        lines.append(f"| {cohort_labels[name]} | {cohort['count']} | {row['common_count']} | {number(row[METHODS[0]]['mean'])} | {number(row[METHODS[1]]['mean'])} | {number(row['mean_ratio'])} | {row['win_count']} |")
    for method in METHODS:
        row = adjacency["methods"][method]
        lines.append(f"\n{LABELS[method]}：整件邻接全对 {row['exact_count']}/2417；邻接可计算 {row['computable_count']}/2417。")
    lines += ["", "## 复杂度关联与最大误差 ID", "",
              f"posterior 边 CD 与 GT 面数的 Spearman rho={number(edge['spearman']['gt_face_count']['candidate_error']['rho'])}；配对差值与 GT 面数 rho={number(edge['spearman']['gt_face_count']['delta']['rho'])}。只表示统计关联，不能据此识别唯一失败机制。", "",
              "| GT 面数 | 件数 | 边 CD 共同数 | 均值比 | 本版改善比例 |", "|---|---:|---:|---:|---:|"]
    grouping = data["modes"]["posterior_mean"]["complexity_groups"]["gt_face_count"]
    for group in grouping["groups"]:
        row = group["metrics"]["edge_cd_sq"]
        lines.append(f"| {group['range']} | {group['count']} | {row['common_count']} | {number(row['mean_ratio'])} | {number(row['win_fraction'], True)} |")
    lines += [f"\n另有 {grouping['unknown_count']} 件面数不可分组，仍在主分母。", "",
              "| 候选 posterior 边误差前 5 ID | GT 面数 | baseline 边 CD | 本版边 CD |", "|---|---:|---:|---:|"]
    for row in [r for r in extremes if r["mode"] == "posterior_mean" and r["metric"] == "edge_cd_sq"][:5]:
        lines.append(f"| {row['model_id']} | {number(row['gt_face_count'])} | {number(row['baseline_error'])} | {number(row['candidate_error'])} |")
    lines += ["", "归档 latent 仅为同一批旧 diffusion latent 的 decoder 重放，没有重新采样 diffusion，也没有 GT 面对应；不报告对应 MAE 或 GT 邻接准确率，不代表完整照片重建提升。", "",
              "[完整分布与来源核验](raw_distribution.json) · [邻接子集](adjacency_subset.json) · [逐件配对 CSV](raw_distribution_per_object.csv) · [每指标前 20 件](raw_distribution_extremes.csv)", ""]
    return "\n".join(lines)


def write_csv(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    paths = {name: OUT / name for name in ("raw_distribution.json", "adjacency_subset.json", "raw_distribution_per_object.csv", "raw_distribution_extremes.csv", "RAW_DISTRIBUTION.md")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Completed diagnostics already exist; refusing overwrite")
    ids = sorted(line.strip() for line in SPLIT.read_text().splitlines() if line.strip())
    if digest(SPLIT) != SPLIT_SHA or len(ids) != 2417 or len(set(ids)) != len(ids):
        raise ValueError("Frozen accepted split differs")
    sources = {str(SPLIT): digest(SPLIT)}
    result = {"methods": METHODS, "modes": {}, "sources_sha256": sources,
              "script_sha256": digest(Path(__file__)), "spec_sha256": digest(OUT / "RAW_DISTRIBUTION_SPEC.md"),
              "trimming_note": "Diagnostic only; candidate-ranked IDs removed from BOTH methods. Primary means remain unchanged.",
              "validation": {"independent_mean": "math.fsum / count", "independent_median": "statistics.median", "percentile": "explicit linear interpolation"}}
    all_rows, extremes = [], []
    for mode in MODES:
        table, summary, provenance = read_mode(ROOT, mode, ids)
        sources.update(provenance)
        record, rows, tails = analyze_mode(table, summary, ids, mode)
        result["modes"][mode] = record
        all_rows.extend(rows)
        extremes.extend(tails)
        if mode == "posterior_mean":
            adjacency = adjacency_cohorts(table, ids)
            adjacency.update(sources_sha256=provenance, script_sha256=result["script_sha256"], method_names=METHODS,
                             note="Descriptive undirected face adjacency only; not coedge/wire correctness or causal isolation. All 2417 IDs remain in primary results.")
    for path, expected in sources.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Input changed during analysis: {path}")
    report = build_report(result, adjacency, extremes)
    # Serialize before creating any files, so invalid numerics cannot leave partial results.
    serialized = {"raw_distribution.json": json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                  "adjacency_subset.json": json.dumps(adjacency, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                  "RAW_DISTRIBUTION.md": report}
    for name, content in serialized.items():
        with paths[name].open("x") as handle:
            handle.write(content)
    write_csv(paths["raw_distribution_per_object.csv"], all_rows)
    write_csv(paths["raw_distribution_extremes.csv"], extremes)
    print(json.dumps({"files": {key: str(path) for key, path in paths.items()},
                      "summary_checks_passed": sum(len(row["summary_checks"]) for row in result["modes"].values()),
                      "adjacency_cohorts": {key: row["count"] for key, row in adjacency["cohorts"].items()}}, indent=2))


if __name__ == "__main__":
    main()
