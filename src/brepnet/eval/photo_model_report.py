"""四组照片重建报告：固定分母、raw/post 分列、配对 bootstrap。"""

from collections import Counter
import csv
from pathlib import Path

import numpy as np

from src.brepnet.eval.photo_model_comparison import METHODS, checked_protocol, verify_generation
from src.brepnet.eval.photo_report import METRICS, paired_delta, setup_plot
from src.brepnet.eval.preference_compare import _verify_evaluation
from src.brepnet.eval.preference_report import finite_number, statistics
from tools.vae_compare_post import atomic_json, read_json, sha256_file

LABELS = ("照片 baseline\n310,000 步", "照片310k → DPO\n5,000 步",
          "空间条件 best\n20,930 步", "空间条件 last\n63,700 步")
KEYS = [f"{stage}_{key}" for stage in ("raw", "post") for key in METRICS] + ["raw_face_count_abs_error"]


def load_rows(run: Path):
    protocol = checked_protocol(run / "metadata/protocol.json")
    rows = {method: [] for method in METHODS}
    config = {"raw_data_root": protocol["config"]["gt_root"], "seed": 42}
    artifacts = read_json(run / "metadata/reused_artifacts.json")
    for name, digest in artifacts.items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"Reused baseline artifact changed: {name}")
    for method in METHODS:
        for mid in protocol["identities"]:
            infer = verify_generation(protocol, method, mid)
            directory = run / "evaluations" / method / mid
            if infer["status"] == "ok":
                _verify_evaluation(config, mid, directory, run / "raw_predictions" / method / mid, infer)
                result = read_json(directory / "metrics.json")
            else:
                result = {}
            raw, post = result.get("raw", {}), result.get("post", [])
            complete = (result.get("validity") == "valid" and len(post) == 3
                        and {p["repeat"] for p in post} == {0, 1, 2}
                        and all(finite_number(p.get(k)) for p in post for k in METRICS))
            row = {"image_id": mid, "method": method, "seed": infer["seed"],
                   "inference_status": infer["status"], "evaluation_status": result.get("status", "not_run"),
                   "validity": result.get("validity", "unknown"), "post_complete": complete,
                   "step_readable": bool(result.get("native", {}).get("step_readable")),
                   "step_exists": (directory / "post/c00/recon_brep.step").is_file(),
                   "inference_seconds": infer["seconds"], "evaluation_seconds": result.get("wall_seconds"),
                   "error": result.get("error") or infer.get("error", "")}
            for key in METRICS:
                row["raw_" + key] = raw.get(key) if finite_number(raw.get(key)) else None
                row["post_" + key] = float(np.mean([p[key] for p in post])) if complete else None
            counts = [raw.get(k) for k in ("pred_face_count", "gt_face_count")]
            row["raw_face_count_abs_error"] = abs(counts[0] - counts[1]) if all(map(finite_number, counts)) else None
            rows[method].append(row)
    return protocol, rows


def summarize(rows):
    ids = [r["image_id"] for r in rows[METHODS[0]]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Require nonempty unique CAD population")
    if any([r["image_id"] for r in rows[m]] != ids for m in METHODS):
        raise ValueError("Methods must cover the same ordered population")
    if any(len({rows[m][i]["seed"] for m in METHODS}) != 1 for i in range(len(ids))):
        raise ValueError("Unpaired random seeds")
    common = set.intersection(*[{r["image_id"] for r in rows[m] if r["post_complete"]} for m in METHODS])
    result = {"intended_per_method": len(ids), "methods": {}, "pairwise": {},
              "all_four_common_post_ids": sorted(common),
              "scope": "synthetic FLUX photos, development validation; not real capture or held-out test",
              "cd": "sum of directed mean squared nearest distances; no registration or square root"}
    for method in METHODS:
        records = rows[method]
        result["methods"][method] = {
            "generated": sum(r["inference_status"] == "ok" for r in records),
            "step_outputs": sum(r["step_exists"] for r in records),
            "readable_steps": sum(r["step_readable"] for r in records),
            "valid_solids": sum(r["validity"] == "valid" for r in records),
            "invalid": sum(r["validity"] == "invalid" for r in records),
            "unknown": sum(r["validity"] == "unknown" for r in records),
            "timeouts": sum(r["evaluation_status"] == "timeout" for r in records),
            "status_counts": dict(Counter(r["evaluation_status"] for r in records)),
            "generation_status_counts": dict(Counter(r["inference_status"] for r in records)),
            "metrics": {key: statistics([r[key] for r in records if finite_number(r[key])]) for key in KEYS},
            "post_on_all_four_common": {key: statistics([r["post_" + key] for r in records
                                                       if r["image_id"] in common]) for key in METRICS},
            **{key: statistics([r[key] for r in records if finite_number(r[key])])
               for key in ("inference_seconds", "evaluation_seconds")},
        }
    for a, b in [(METHODS[0], m) for m in METHODS[1:]] + [("spatial_best", "spatial_last")]:
        left = [{**r, "valid": int(r["validity"] == "valid")} for r in rows[a]]
        right = [{**r, "valid": int(r["validity"] == "valid")} for r in rows[b]]
        result["pairwise"][f"{b}_minus_{a}"] = {key: paired_delta(left, right, key) for key in ["valid", *KEYS]}
    return result


def result_table(run, summary):
    metrics, total = summary["methods"], summary["intended_per_method"]
    lines = []
    for title, key in (("生成成功", "generated"), ("有效 solid ↑", "valid_solids"),
                       ("产出 STEP", "step_outputs"), ("超时", "timeouts")):
        lines.append([title, *[f"{metrics[m][key]}/{total} ({metrics[m][key]/total:.1%})" for m in METHODS]])
    for title, key, scale in (("原始面 CD² × 1000 ↓", "raw_face_cd_sq", 1000),
                              ("原始边 CD² × 1000 ↓", "raw_edge_cd_sq", 1000),
                              ("原始面 F1 (%) ↑", "raw_face_f1", 100),
                              ("面数绝对误差 ↓", "raw_face_count_abs_error", 1),
                              ("有效 solid 面 CD² × 1000 ↓", "post_face_cd_sq", 1000)):
        values = [metrics[m]["metrics"][key] for m in METHODS]
        lines.append([title, *[f"{v['mean']*scale:.3f} (n={v['n']})" if v["n"] else "— (n=0)" for v in values]])
    values = [metrics[m]["post_on_all_four_common"]["face_cd_sq"] for m in METHODS]
    lines.append(["四组共同有效子集 面 CD² × 1000 ↓",
                  *[f"{v['mean']*1000:.3f} (n={v['n']})" if v["n"] else "— (n=0)" for v in values]])
    plt = setup_plot()
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.axis("off")
    fig.text(.045, .95, f"照片条件 B-Rep：完整 {total:,} 件 validation", fontsize=23, weight="bold")
    fig.text(.045, .90, "相同 FLUX 合成图 · K=1 · 同 seed · DDPM1000 FP32 · 同 VAE / OCC 后处理", fontsize=12)
    table = ax.table(cellText=lines, colLabels=["指标", *LABELS], cellLoc="center",
                     colWidths=[.30, .175, .175, .175, .175], bbox=[0, .12, 1, .78])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dbe3ec")
        if row == 0:
            cell.set_facecolor("#16324f")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#edf2f7")
            if col == 0:
                cell.set_text_props(ha="left")
    fig.text(.045, .075, "失败保留在固定分母；几何只统计可测子集并注明 n。CD² 为双向平方距离之和。", fontsize=11)
    fig.text(.045, .045, "best 按验证 MSE 选择；四组训练预算不同。此表不是独立 test、实拍或可见区域一致性评估。", fontsize=11)
    fig.savefig(run / "comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(run / "comparison.svg", bbox_inches="tight")
    plt.close(fig)
    return lines


def write_report(run: Path):
    protocol, rows = load_rows(run)
    summary = summarize(rows)
    atomic_json(run / "summary.json", summary)
    with (run / "per_object.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[METHODS[0]][0]), lineterminator="\n")
        writer.writeheader()
        for method in METHODS:
            writer.writerows(rows[method])
    lines = result_table(run, summary)
    text = ["# 照片模型完整验证比较", "", "固定3505件 validation；FLUX合成照片，不是实拍。",
            "", "| 指标 | 照片310k | DPO5000 | 空间best20930 | 空间last63700 |", "|---|---:|---:|---:|---:|"]
    text.extend("| " + " | ".join(row) + " |" for row in lines)
    text.extend(["", "DDPM1000 FP32、TF32关闭、每图一次生成、相同seed。原始VAE保持不变；recover_v1后处理、drop_num=0、200次优化、每件整体180秒、有效STEP几何采样重复3次。",
                 "", "CD²为双向均值平方距离之和，face F1阈值0.1；post只包含完整测量的有效solid，四组共同成功子集单列。失败不补零，合法率分母固定3505。",
                 "", "配对差值及95% bootstrap区间见summary.json。空间best按validation MSE选取；训练预算及架构同时不同，本比较不能隔离某个结构的因果贡献。",
                 "", "可见区域或重投影指标本轮未计算；完整GT几何距离不等于可见证据一致性。Baseline复用来源、输入/输出校验见metadata/protocol.json和REUSE_AUDIT.json。"])
    (run / "README.md").write_text("\n".join(text) + "\n")
    atomic_json(run / "VALIDATION.json", {"protocol_fingerprint": protocol["fingerprint"],
                "all_terminal_records_verified": True, "all_output_hashes_verified": True,
                "intended_per_method": len(protocol["identities"]), "methods": list(METHODS)})
