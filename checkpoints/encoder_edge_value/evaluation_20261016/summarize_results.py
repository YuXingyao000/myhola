"""Five fixed snapshots, with identical-cohort raw/post and owner comparisons.

The numerical primitives come from the audited 2026-10-15 summary. This task
only assembles the newly frozen snapshots; it never invokes model or CAD code.
"""
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "evaluation_20261015"
spec = importlib.util.spec_from_file_location("audited_summary", PREVIOUS / "summarize_results.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
CASES = {
    "conditioned": (PREVIOUS / "candidate", "conditioned", "Conditioning\n原基线", "4c7910b8169206c74fcf38c358cf171359149e0429c8debbc227d391aa56613f"),
    "candidate_1089": (PREVIOUS / "candidate", "encoder_edge_value", "全程验证最佳\n1,089 步", "d9e911e23a5dfc0e45fb90735a806706c9c89329898f0873df06927c8d2fb62d"),
    "candidate_14520": (PREVIOUS / "endpoint_14520/candidate", "encoder_edge_value", "续训前最后保存\n14,520 步", "e9120c78254d9b94aaaef88333c0f0656fd6a01346225d092fb8341297bff2bf"),
    "resume_best_27588": (ROOT / "resume_best", "encoder_edge_value", "续训窗口验证最佳\n27,588 步", "e1e579e7bc6bdf64b3c078d6b5414e33630222de989b180ca8f3eb22b59c6ccf"),
    "last_28677": (ROOT / "last", "encoder_edge_value", "本轮最后保存（主）\n28,677 步", "040c05a45dd7edae825f2e6dd8e2cb3489a418146f9a7b2122cc54d38070618c"),
}
NEW = ("resume_best_27588", "last_28677")
PAIRS = tuple((new, old) for new in NEW for old in ("conditioned", "candidate_1089", "candidate_14520")) + (("last_28677", "resume_best_27588"),)


def selected_rows(inputs, path, method, ids, key):
    rows = [r for r in inputs.csv(path) if r["method"] == method]
    result = {r[key]: r for r in rows}
    core.require(len(rows) == len(result) == len(ids) and set(result) == set(ids), f"Wrong fixed cohort: {path}/{method}")
    return result


def load_tables(inputs, ids):
    raw, post = {}, {mode: {} for mode in ("fit_only", "optimized")}
    raw_report = {"methods": {}, "pairs": {}, "top_one_percent": {}}
    post_report = {mode: {"methods": {}, "pairs": {}} for mode in post}
    for alias, (directory, method, _, _) in CASES.items():
        location = directory / "raw_metrics/posterior_mean"
        summary = inputs.json(location / "summary.json")
        core.require(summary["protocol_valid"] and summary["input_contract"] == "same_geometry_source", "Wrong raw contract")
        core.require(summary["intended_count"] == 2417 and summary["original_count"] == 2424, "Wrong raw denominator")
        raw[alias] = selected_rows(inputs, location / "per_object.csv", method, ids, "model_id")
        for row in raw[alias].values():
            core.require(row["shared_input_status"] == row["latent_digest_status"] == "verified", "Unverified raw provenance")
            core.require(row["input_source_sha256"] == row["actual_gt_source_sha256"], "Wrong canonical source")
        metrics = {key: core.stats([core.number(r, key) for r in raw[alias].values()], len(ids)) for key in core.RAW_METRICS}
        for key in metrics:
            core.verify_stats(metrics[key], summary["methods"][method]["metrics"][key])
        raw_report["methods"][alias] = {"metrics": metrics, "status_counts": dict(Counter(r["status"] for r in raw[alias].values())),
            "exact_adjacency_count": sum(core.topology_exact(r) for r in raw[alias].values())}
        if alias in NEW:
            baseline = selected_rows(inputs, location / "per_object.csv", "conditioned", ids, "model_id")
            core.same_baseline(raw["conditioned"], baseline, raw=True)
        for mode in post:
            location = directory / "post" / mode
            summary = inputs.json(location / "summary.json")
            core.require(summary["intended_count_per_method"] == 2417 and summary["original_test_count"] == 2424, "Wrong post denominator")
            group = selected_rows(inputs, location / "samples.csv", method, ids, "sample_id")
            post[mode][alias] = group
            core.require(all(r["status"] in core.TERMINAL for r in group.values()), "Pending post objects")
            counts = Counter("unknown" if core.boolean(r.get("native_valid_solid")) is None else "valid" if core.boolean(r.get("native_valid_solid")) else "failed" for r in group.values())
            core.require(summary["methods"][method]["counts"]["terminal_records"] == len(ids), "Incomplete post summary")
            core.require(counts["valid"] == summary["methods"][method]["counts"]["native_valid_solid"], "Wrong upstream valid count")
            metrics = {key: core.stats([core.number(r, key) for r in group.values() if r.get("geometry_status") == "ok"], len(ids)) for key in core.POST_METRICS}
            for key in metrics:
                core.verify_stats(metrics[key], summary["methods"][method]["geometry_all_computable"][key])
            post_report[mode]["methods"][alias] = {"native_solid_counts": {k: counts[k] for k in ("valid", "failed", "unknown")},
                "valid_fraction_intended": counts["valid"] / len(ids), "status_counts": dict(Counter(r["status"] for r in group.values())),
                "geometry_metrics": metrics}
            if alias in NEW:
                baseline = selected_rows(inputs, location / "samples.csv", "conditioned", ids, "sample_id")
                core.same_baseline(post[mode]["conditioned"], baseline, raw=False)
    for item in ids:
        core.require(len({raw[m][item]["input_source_sha256"] for m in CASES}) == 1, "Different CAD sources between snapshots")
    for new, old in PAIRS:
        key = new + "_vs_" + old
        exact = [item for item in ids if core.topology_exact(raw[new][item]) and core.topology_exact(raw[old][item])]
        raw_report["pairs"][key] = {"all_intended": core.paired(raw, ids, new, old, core.RAW_METRICS),
            "both_exact_adjacency": core.paired(raw, exact, new, old, core.RAW_METRICS), "both_exact_adjacency_ids": exact}
        for mode in post:
            common = [item for item in ids if all(post[mode][m][item].get("geometry_status") == "ok" for m in (new, old))]
            post_report[mode]["pairs"][key] = {"native_solid_transitions": core.solid_transitions(post[mode], ids, new, old),
                "common_geometry": core.paired(post[mode], common, new, old, core.POST_METRICS), "common_geometry_ids": common,
                "common_geometry_missing_count": len(ids) - len(common)}
    for metric in ("edge_cd_sq", "legacy_reciprocal_cd_sq_mean", "legacy_reciprocal_cd_sq_max"):
        raw_report["top_one_percent"][metric] = {}
        for alias in CASES:
            values = [(item, core.number(raw[alias][item], metric)) for item in ids]
            values = sorted([(item, v) for item, v in values if v is not None], key=lambda pair: (-pair[1], pair[0]))
            k = math.ceil(len(values) * .01)
            total, top = sum(v for _, v in values), sum(v for _, v in values[:k])
            raw_report["top_one_percent"][metric][alias] = {"finite_count": len(values), "missing_count": len(ids) - len(values),
                "top_count": k, "top_ids": [item for item, _ in values[:k]], "top_error_sum": top, "total_error_sum": total,
                "top_error_share": top / total if total else None}
    interesting = {"00335203", "00256640", "00772578"}
    interesting.update(item for record in raw_report["top_one_percent"]["edge_cd_sq"].values() for item in record["top_ids"])
    raw_report["tail_case_details"] = {}
    for item in sorted(interesting):
        record = {"metrics": {m: {k: core.number(raw[m][item], k) for k in ("edge_cd_sq", "edge_pred_to_gt_sq", "edge_gt_to_pred_sq", "topology_fp", "topology_fn")} for m in CASES}, "edge_delta_contributions": {}}
        for new, old in PAIRS:
            a, b = core.number(raw[new][item], "edge_cd_sq"), core.number(raw[old][item], "edge_cd_sq")
            delta = a - b if a is not None and b is not None else None
            net = sum(core.number(raw[new][sid], "edge_cd_sq") - core.number(raw[old][sid], "edge_cd_sq") for sid in ids if core.number(raw[new][sid], "edge_cd_sq") is not None and core.number(raw[old][sid], "edge_cd_sq") is not None)
            record["edge_delta_contributions"][new + "_vs_" + old] = {"object_delta": delta, "delta_divided_by_full_intended_count": delta / len(ids) if delta is not None else None,
                "fraction_of_net_error_change": delta / net if delta is not None and net else None}
        raw_report["tail_case_details"][item] = record
    return raw, post, raw_report, post_report, interesting


def owners(inputs, ids, raw, post, interesting):
    protocols, digests = {}, {}
    for alias, (directory, _, _, expected_sha) in CASES.items():
        location = core.BASE if alias == "conditioned" else directory
        protocols[alias] = inputs.json(location / "metadata/inference/posterior_mean/protocol.json")
        core.require(protocols[alias]["checkpoint_sha256"] == expected_sha, "Wrong snapshot checkpoint")
        digests[alias] = hashlib.sha256(json.dumps(protocols[alias], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    rows, details, common_records = [], {}, []
    for index, item in enumerate(ids):
        path = core.DATA / item / "data.npz"
        gt = inputs.arrays(path)
        gt_sha = inputs.records[str(path.resolve())]["sha256"]
        faces = gt["sample_points_faces"]
        core.require(faces.ndim == 4 and faces.shape[1:] == (16, 16, 6) and np.isfinite(faces).all(), "Invalid GT faces")
        truth = core.curve_map(gt["sample_points_lines"][..., :3], gt["edge_face_connectivity"], len(faces))
        maps, errors, probabilities = {}, {}, {}
        for alias in CASES:
            prediction = Path(raw[alias][item]["prediction_file"])
            if not prediction.is_file():
                errors[alias] = "missing_prediction"
                continue
            arrays = inputs.arrays(prediction)
            source_sha = inputs.records[str(prediction.resolve())]["sha256"]
            core.require(core.text(arrays["input_source_sha256"]) == gt_sha == raw[alias][item]["input_source_sha256"], "Wrong canonical geometry")
            core.require(core.text(arrays["protocol_sha256"]) == digests[alias], "Wrong raw protocol")
            core.require(core.text(arrays["comparison_mode"]) == "posterior_mean", "Wrong raw mode")
            latent = arrays["face_latents"]
            core.require(latent.shape == (len(faces), 32) and latent.dtype == np.float32 and np.isfinite(latent).all(), "Wrong posterior means")
            digest = hashlib.sha256(str(latent.shape).encode())
            digest.update(np.ascontiguousarray(latent).tobytes())
            core.require(digest.hexdigest() == core.text(arrays["input_latent_sha256"]) == raw[alias][item]["input_latent_sha256"], "Wrong mean digest")
            if alias != "conditioned":
                core.require(core.text(arrays["encoder_checkpoint_sha256"]) == CASES[alias][3] and core.text(arrays["input_source_kind"]) == "checkpoint_posterior_mean", "Wrong encoder source")
            for mode in post:
                core.require(post[mode][alias][item].get("raw_sha256") == source_sha, "Post used different raw prediction")
            core.require(arrays["pred_face"].shape == (len(faces), 16, 16, 3), "Wrong face output")
            try:
                maps[alias] = core.curve_map(arrays["pred_edge"], arrays["pred_edge_face_connectivity"], len(faces))
            except ValueError as exc:
                errors[alias] = str(exc)
            if alias in maps and raw[alias][item]["topology_status"] == "ok":
                actual, expected = {tuple(sorted(p)) for p in maps[alias]}, {tuple(sorted(p)) for p in truth}
                core.require(len(actual - expected) == core.number(raw[alias][item], "topology_fp") and len(expected - actual) == core.number(raw[alias][item], "topology_fn"), "Owner map contradicts adjacency metrics")
            if item in interesting:
                pairs, probs = arrays["all_candidate_pairs"], arrays["all_candidate_prob"]
                core.require(pairs.shape == (len(probs), 2) and np.isfinite(probs).all(), "Wrong candidate probabilities")
                probabilities[alias] = {tuple(int(x) for x in pair): float(p) for pair, p in zip(pairs, probs)}
                core.require(len(probabilities[alias]) == len(pairs), "Duplicate candidate pairs")
        common = sorted(set(truth).intersection(*(set(maps[m]) for m in CASES))) if len(maps) == len(CASES) else []
        target = np.stack([truth[p] for p in common]) if common else np.empty((0, 16, 3))
        common_records.append({"model_id": item, "gt_sha256": gt_sha, "common_directed_pair_count": len(common), "common_directed_pairs_sha256": hashlib.sha256(json.dumps(common, separators=(",", ":")).encode()).hexdigest()})
        if item in interesting:
            details[item] = {"common_directed_pair_count": len(common), "gt_directed_pair_count": len(truth), "methods": {}}
        for alias in CASES:
            points = np.stack([maps[alias][p] for p in common]) if common else np.empty((0, 16, 3))
            distances = core.owner_cd(points, target)
            row = {"model_id": item, "method": alias, "owner_status": "invalid_or_missing_input" if errors else "ok" if common else "empty_common_pairs",
                "error": json.dumps(errors, sort_keys=True) if errors else "", "gt_directed_pair_count": len(truth),
                "predicted_directed_pair_count": len(maps[alias]) if alias in maps else None,
                "missing_gt_directed_pairs": len(set(truth) - set(maps[alias])) if alias in maps else None,
                "extra_directed_pairs": len(set(maps[alias]) - set(truth)) if alias in maps else None,
                "common_directed_pair_count": len(common), "common_gt_pair_fraction": len(common) / len(truth) if truth else None,
                "owner_cd_sq_mean": float(distances.mean()) if len(distances) else None, "owner_cd_sq_max": float(distances.max()) if len(distances) else None}
            rows.append(row)
            if item in interesting:
                largest = sorted(range(len(distances)), key=lambda i: (-distances[i], common[i]))[:3]
                details[item]["methods"][alias] = {"owner_cd_sq_mean": row["owner_cd_sq_mean"], "owner_cd_sq_max": row["owner_cd_sq_max"],
                    "missing_directed_pairs": [{"pair": p, "existence_probability": probabilities.get(alias, {}).get(p)} for p in sorted(set(truth) - set(maps.get(alias, {})))],
                    "extra_directed_pairs": sorted(set(maps.get(alias, {})) - set(truth)),
                    "largest_common_owner_errors": [{"pair": common[i], "cd_sq": float(distances[i])} for i in largest]}
    frames = {m: {r["model_id"]: r for r in rows if r["method"] == m} for m in CASES}
    report = {"definition": "GT directed face-pairs present in all five fixed snapshots; 16-point bidirectional squared nearest-neighbor means summed. Macro summary of each object's mean/max. Missing/extra pairs excluded from this geometry subset and reported separately.",
        "intended_objects": len(ids), "objects_with_common_pairs": sum(frames["conditioned"][item]["common_directed_pair_count"] > 0 for item in ids),
        "all_gt_directed_pairs": sum(frames["conditioned"][item]["gt_directed_pair_count"] for item in ids),
        "common_directed_pairs": sum(frames["conditioned"][item]["common_directed_pair_count"] for item in ids),
        "methods": {}, "pairs": {}, "tail_case_details": details, "source_object_records": common_records}
    report["common_pair_fraction"] = report["common_directed_pairs"] / report["all_gt_directed_pairs"] if report["all_gt_directed_pairs"] else None
    for alias in CASES:
        report["methods"][alias] = {"status_counts": dict(Counter(r["owner_status"] for r in frames[alias].values())),
            "metrics": {key: core.stats([core.number(r, key) for r in frames[alias].values()], len(ids)) for key in ("owner_cd_sq_mean", "owner_cd_sq_max", "missing_gt_directed_pairs", "extra_directed_pairs", "common_gt_pair_fraction")},
            "missing_gt_directed_pairs_total": sum(r["missing_gt_directed_pairs"] or 0 for r in frames[alias].values()), "extra_directed_pairs_total": sum(r["extra_directed_pairs"] or 0 for r in frames[alias].values())}
    for new, old in PAIRS:
        report["pairs"][new + "_vs_" + old] = core.paired(frames, ids, new, old, ("owner_cd_sq_mean", "owner_cd_sq_max"))
    return rows, report


def figure(report, destination):
    font = Path("/root/.cache/hola-browser-libs/fonts/NotoSansCJKsc-Regular.otf")
    core.require(font.is_file(), "Chinese figure font is unavailable")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(), "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(2, 3, figsize=(19, 10), constrained_layout=True)
    labels = [record[2] for record in CASES.values()]
    colors = ["#89949c", "#3c78ad", "#6c9fa7", "#d99a3c", "#b85343"]
    for ax, metric, title, section in ((axes[0, 0], "face_mae", "原始面 XYZ MAE 均值", "raw"), (axes[0, 1], "edge_cd_sq", "原始全边合并 CD² 均值", "raw"), (axes[0, 2], "owner_cd_sq_mean", "按所属面对匹配的边 CD² 均值", "owner")):
        values = [report[section]["methods"][m]["metrics"][metric]["mean"] for m in CASES]
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, labels=[f"{v:.4g}" for v in values], padding=4, fontsize=9)
        ax.set_ylim(0, max(values) * 1.2)
        ax.set_title(title + "（越低越好）")
    for ax, mode in ((axes[1, 0], "fit_only"), (axes[1, 1], "optimized")):
        counts = [report["post"][mode]["methods"][m]["native_solid_counts"]["valid"] for m in CASES]
        values = [100 * n / report["intended_count"] for n in counts]
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, labels=[f"{n}/2417\n{v:.2f}%" for n, v in zip(counts, values)], padding=4, fontsize=9)
        ax.set_ylim(0, 103)
        ax.set_title(("只拟合" if mode == "fit_only" else "几何优化后") + "原生合法 solid（越高越好）")
    values = [100 * report["raw"]["top_one_percent"]["edge_cd_sq"][m]["top_error_share"] for m in CASES]
    bars = axes[1, 2].bar(labels, values, color=colors)
    axes[1, 2].bar_label(bars, fmt="%.1f%%", padding=4)
    axes[1, 2].set_ylim(0, 100)
    axes[1, 2].set_title("各快照最差 1% 样本的边误差份额\n各自 25 件，不从主结果中剔除")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
    owner = report["owner"]
    fig.suptitle("Encoder 边内容 GAT 续训：预先固定的五份权重\n同 2,417 件、各自 posterior mean、同后处理；owner 比较使用五方共同 GT 有向面对 "
                 f"{owner['common_directed_pairs']:,}/{owner['all_gt_directed_pairs']:,}（{owner['common_pair_fraction']:.2%}）", fontsize=15)
    for suffix in ("png", "svg"):
        fig.savefig(destination / ("five_checkpoint_summary." + suffix), dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), constrained_layout=True)
    for ax, statistic, title in zip(axes, ("mean", "median", "p90"), ("均值", "中位数", "P90")):
        values = [report["raw"]["methods"][m]["metrics"]["edge_cd_sq"][statistic] for m in CASES]
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, labels=[f"{v:.4g}" for v in values], padding=4, fontsize=9)
        ax.set_ylim(0, max(values) * 1.22)
        ax.set_title(f"原始全边合并 CD²：{title}（越低越好）")
        ax.tick_params(axis="x", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    fig.suptitle("边几何的典型样本与尾部分布\n27,588 为续训窗口验证最佳；28,677 为最后保存主结果；不按测试结果重新选择模型", fontsize=14)
    for suffix in ("png", "svg"):
        fig.savefig(destination / ("edge_distribution_summary." + suffix), dpi=170)
    plt.close(fig)


def main():
    core.self_check()
    targets = [ROOT / "diagnostics.json", ROOT / "owner_per_object.csv", ROOT / "input_hashes.jsonl"] + [ROOT / "visualizations" / (name + "." + suffix) for name in ("five_checkpoint_summary", "edge_distribution_summary") for suffix in ("png", "svg")]
    core.require(not any(p.exists() for p in targets), "Refuse to overwrite diagnostics")
    inputs = core.Inputs()
    data = inputs.read(core.SPLIT)
    core.require(hashlib.sha256(data).hexdigest() == core.SPLIT_SHA, "Wrong frozen split")
    ids = data.decode().splitlines()
    core.require(len(ids) == len(set(ids)) == 2417, "Wrong cohort")
    for path in (ROOT / "PLAN.md", Path(__file__), PREVIOUS / "summarize_results.py", ROOT / "validation/source_audit.json"):
        inputs.read(path)
    raw, post, raw_report, post_report, interesting = load_tables(inputs, ids)
    owner_rows, owner_report = owners(inputs, ids, raw, post, interesting)
    report = {"intended_count": 2417, "original_count": 2424, "unsupported_count": 7, "split_sha256": core.SPLIT_SHA,
        "methods": list(CASES), "paired_change_tolerance": core.TIE, "raw": raw_report, "post": post_report, "owner": owner_report,
        "selected_checkpoints": {alias: {"pair_root": str(row[0]), "method": row[1], "checkpoint_sha256": row[3]} for alias, row in CASES.items()},
        "checks": {"numeric_self_checks": "passed", "baseline_consistency": True, "upstream_summary_statistics_recomputed": True, "all_post_rows_terminal": True,
            "cad_latent_checkpoint_protocol_and_post_source_digests_verified": True, "source_files_checked_before_after_read": True},
        "limitations": ["Two new snapshots were fixed before test; results do not select a new official model.", "No same-budget original-GAT control at 28k steps; continuation changes are not an architecture-only causal comparison.",
            "Owner errors cover only common GT directed pairs in all five snapshots; missed/extra pairs and coverage are separate.", "Squared sampled distances are not linear CAD tolerance or maximum error.",
            "Post geometry includes all computable objects and is separately paired on common computable cohorts; it is not restricted to valid solids.", "Signed tail contribution fractions may exceed 100% because other objects offset the change. No outliers are excluded."]}
    for path, record in inputs.records.items():
        if not path.endswith(".npz"):
            core.require(core.sha256(path) == record["sha256"], "Summary input changed")
    with tempfile.TemporaryDirectory(prefix=".summary-", dir=ROOT) as temp:
        stage = Path(temp)
        with (stage / "owner_per_object.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(owner_rows[0])); writer.writeheader(); writer.writerows(owner_rows)
        with (stage / "input_hashes.jsonl").open("w") as stream:
            for name in sorted(inputs.records):
                stream.write(json.dumps(inputs.records[name], ensure_ascii=False) + "\n")
        report["input_hash_manifest"] = {"path": str(ROOT / "input_hashes.jsonl"), "sha256": core.sha256(stage / "input_hashes.jsonl"), "file_count": len(inputs.records)}
        report["owner_csv_sha256"] = core.sha256(stage / "owner_per_object.csv")
        (stage / "diagnostics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        figure(report, stage)
        core.require(not any(p.exists() for p in targets), "Outputs appeared during calculation")
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            (stage / target.name).replace(target)
    print(json.dumps({"output": str(ROOT / "diagnostics.json"), "owner_pair_coverage": owner_report["common_pair_fraction"], "optimized_solids": {m: post_report["optimized"]["methods"][m]["native_solid_counts"] for m in CASES}}))


if __name__ == "__main__":
    main()
