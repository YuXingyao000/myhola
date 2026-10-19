"""只汇总实测数据和图表，不选择checkpoint或修改评估规则。"""

import csv
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import GROUPS, feature_records, write_json
from checkpoints.diffusion.uncond_baseline_20261018.visualize import configure_fonts, draw_mesh, step_mesh
from src.brepnet.eval.metrics.lfd import summarize_lfd_pickle

LABELS = {"old": "旧 HoLa\n1,000,000 步", "new_last": "新版 last\n303,500 步", "new_best": "新版最低 val-loss\n276,185 步"}
COLORS = ["#658ba5", "#a784b3", "#ca9b61"]
OUT = ROOT / "report"


def save(fig, name):
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def finite_stats(values):
    values = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))])
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "p95": None}
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def summarize_lfd():
    result = {}
    for method in GROUPS:
        shards = [np.load(p) for p in sorted((ROOT / "distances/lfd" / method).glob("shard*.npz"))]
        names, refs = shards[0]["sample_ids"].tolist(), shards[0]["reference_ids"].tolist()
        matrix = np.empty((len(names), len(refs)), dtype=np.int64)
        covered = []
        for shard in shards:
            matrix[shard["indices"]] = shard["distances"]
            covered.extend(shard["indices"].tolist())
        assert sorted(covered) == list(range(len(names)))
        path = ROOT / "distances/lfd" / method / "result.pkl"
        nearest = [refs[index] for index in matrix.argmin(axis=1)]
        path.write_bytes(pickle.dumps([names, nearest, matrix]))
        existing = summarize_lfd_pickle(path)
        valid = np.array([json.loads((GROUPS[method] / "metadata/post/samples" / f"{name}.json").read_text()).get("native_valid_solid") is True for name in names])
        result[method] = {
            "n": len(names), "reference_n": len(refs), "mean_lfd": existing["mean_lfd"],
            "median_lfd": existing["median_lfd"], "p75_lfd": existing["p75_lfd"],
            "MMD-LFD": float(matrix.min(axis=0).mean()),
            "COV-LFD": len(np.unique(matrix.argmin(axis=1))) / len(refs),
            "native_valid_n": int(valid.sum()),
            "native_valid_mean_lfd": float(matrix[valid].min(axis=1).mean()) if valid.any() else None,
            "nearest_lfd": matrix.min(axis=1).tolist(), "sample_ids": names, "nearest_reference": nearest,
        }
    write_json(OUT / "lfd_metrics.json", result)
    return result


def collect():
    sets = json.loads((OUT / "set_metrics.json").read_text())
    graphs = json.loads((OUT / "graph_metrics.json").read_text())
    lfd = summarize_lfd()
    data, csv_rows = {}, []
    for method, folder in GROUPS.items():
        summary = json.loads((folder / "report/summary.json").read_text())
        assert summary["complete"] and summary["intended_count"] == 1000
        summary.pop("metrics_not_computed", None)  # 此字段仅描述原post runner，不适用于完整指标报告。
        with (folder / "report/per_sample.csv").open() as stream:
            records = list(csv.DictReader(stream))
        features = feature_records(method)
        complexities = [row["complexity"] for row in features if row["complexity"] is not None]
        stats = {key: finite_stats([row.get(key) for row in complexities]) for key in
                 ("num_faces", "num_edges", "num_vertices", "cyclomatic_complexity", "mean_curvature")}
        data[method] = {"summary": summary, "records": records, "sets": sets["methods"][method],
                        "graphs": graphs["methods"][method], "lfd": lfd[method], "complexity": stats}
        feature_map = {row["sample_id"]: row for row in features}
        graph_map = {row["sample_id"]: row for row in graphs["methods"][method]["rows"]}
        for row in records:
            feature = feature_map.get(row["sample_id"], {})
            complexity = feature.get("complexity") or {}
            graph = graph_map.get(row["sample_id"], {})
            csv_rows.append({"method": method, **row, "pointcloud_ok": feature.get("pointcloud_ok"),
                "pointcloud_error": feature.get("pointcloud_error"),
                "cyclomatic_complexity": complexity.get("cyclomatic_complexity"), "mean_curvature": complexity.get("mean_curvature"),
                "duplicate_of": graph.get("duplicate_of"), "identical_train": graph.get("identical_train")})
    with (OUT / "per_sample_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(csv_rows)
    write_json(OUT / "comparison.json", {method: {k: v for k, v in row.items() if k != "records"} for method, row in data.items()})
    return data


def tables(data):
    rows = []
    def add(name, unit, get):
        row = {"metric": name, "unit": unit}
        for method, entry in data.items():
            value, n = get(entry)
            row[method] = value; row[method + "_n"] = n
        rows.append(row)
    for key, name in [("inference_ok", "生成成功"), ("step_readable", "可读取STEP"),
                      ("native_valid_solid", "原生合法solid"), ("compat_valid_solid", "0.1兼容合法solid")]:
        add(name, "count / 1000", lambda d, k=key: (d["summary"]["counts"][k], 1000))
    for cohort, title in [("all_meshable", "全部可三角化"), ("native_valid_meshable", "合法且可三角化"), ("equal_valid_meshable", "等量合法子集"), ("equal_all_meshable", "等量可三角化子集")]:
        for key in ("MMD-CD", "COV-CD", "JSD"):
            add(f"{key} / {title}", "raw", lambda d, c=cohort, k=key: (d["sets"][c][k], d["sets"][c]["n"]))
    for key, title in [("Unique", "Unique / 全部raw"), ("Novel", "Novel / 当前训练集46627"),
                       ("native_valid_Unique", "Unique / 合法子集"), ("native_valid_Novel", "Novel / 合法子集")]:
        add(title, "ratio", lambda d, k=key: (d["graphs"][k], d["graphs"]["native_valid_n" if k.startswith("native") else "n"]))
    for key, title in [("mean_lfd", "LFD / 生成→参考最近邻"), ("MMD-LFD", "MMD-LFD / 参考→生成"),
                       ("COV-LFD", "COV-LFD"), ("native_valid_mean_lfd", "LFD / 合法子集")]:
        add(title, "raw", lambda d, k=key: (d["lfd"][k], d["lfd"]["native_valid_n" if k.startswith("native") else "n"]))
    for key in ("num_faces", "num_edges", "num_vertices", "cyclomatic_complexity", "mean_curvature"):
        add("Complexity mean / " + key, "raw", lambda d, k=key: (d["complexity"][k]["mean"], d["complexity"][k]["n"]))
    for key in ("raw_num_faces", "raw_num_directed_edges", "inference_seconds", "post_wall_seconds"):
        add("Mean / " + key, "seconds" if "seconds" in key else "count", lambda d, k=key: (d["summary"]["statistics"][k]["mean"], d["summary"]["statistics"][k]["n"]))
    with (OUT / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    return rows


def overview(data):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), facecolor="#f7f9fc")
    x = np.arange(3)
    labels = list(LABELS.values())
    specs = [("原生合法 solid（固定分母1000）", "%", lambda d: (100 * d["summary"]["rates"]["native_valid_solid"]["rate"], 1000)),
             ("MMD-CD · 全部可三角化输出", "CD² × 1000", lambda d: (1000 * d["sets"]["all_meshable"]["MMD-CD"], d["sets"]["all_meshable"]["n"])),
             ("COV-CD · 全部可三角化输出", "%", lambda d: (100 * d["sets"]["all_meshable"]["COV-CD"], d["sets"]["all_meshable"]["n"])),
             ("JSD · 全部可三角化输出", "JSD × 1000", lambda d: (1000 * d["sets"]["all_meshable"]["JSD"], d["sets"]["all_meshable"]["n"])),
             ("LFD · 生成→参考最近邻", "LFD", lambda d: (d["lfd"]["mean_lfd"], d["lfd"]["n"])),
             ("原始解码面数均值", "面", lambda d: (d["summary"]["statistics"]["raw_num_faces"]["mean"], 1000))]
    for ax, (title, unit, get) in zip(axes.flat, specs):
        values = [get(d)[0] for d in data.values()]
        ax.bar(x, values, color=COLORS, width=0.62)
        if title.startswith("原生合法"):
            bounds=np.asarray([d["summary"]["rates"]["native_valid_solid"]["wilson95"] for d in data.values()])*100
            ax.errorbar(x,values,yerr=[np.asarray(values)-bounds[:,0],bounds[:,1]-np.asarray(values)],fmt="none",color="#374151",capsize=4)
        for i, d in enumerate(data.values()):
            value, n = get(d)
            ax.annotate(f"{value:.3g}\nn={n}", (i, value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=10)
        if unit == "%": ax.set_ylim(0, 108)
        else: ax.set_ylim(0, max(values) * 1.28 if max(values) else 1)
        ax.set_xticks(x, labels, fontsize=9); ax.set_ylabel(unit)
        ax.set_title(title, fontsize=12, loc="left"); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("无条件生成对比 · 固定1000个seed · 相同VAE / DDPM / 后处理", fontsize=18)
    fig.text(0.02, 0.012, "参考：固定validation1000；表面点云2000点，沿用仓库归一化/轴对齐；图中集合指标包含可三角化的非solid输出。实际训练步数见各列。", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, "metrics_overview")


def balanced_sets(data):
    fig,axes=plt.subplots(1,3,figsize=(15,5),facecolor="#f7f9fc")
    for ax,key,scale in zip(axes,["MMD-CD","COV-CD","JSD"],[1000,100,1000]):
        values=[d["sets"]["equal_all_meshable"][key]*scale for d in data.values()]
        ax.bar(range(3),values,color=COLORS,width=.62)
        for i,value in enumerate(values):ax.annotate(f"{value:.4g}",(i,value),xytext=(0,6),textcoords="offset points",ha="center")
        ax.set_xticks(range(3),list(LABELS.values()),fontsize=9)
        ax.set_title(key+(" (%)" if key=="COV-CD" else " ×1000"),loc="left")
        ax.set_ylim(0,max(values)*1.3 if max(values) else 1)
        ax.spines[["top","right"]].set_visible(False)
    n=next(iter(data.values()))["sets"]["equal_all_meshable"]["n"]
    fig.suptitle(f"等量可三角化子集 · 各组n={n} · 固定validation参考1000",fontsize=16)
    fig.text(.02,.01,"使用seed20261019降采样，不按分数筛选；包含非solid输出。完整可三角化集合与合法子集的结果分别保留在CSV。",fontsize=10)
    fig.tight_layout(rect=(0,.06,1,.92));save(fig,"balanced_set_metrics")


def distributions(data):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), facecolor="#f7f9fc")
    specs = [("raw_num_faces", "原始面数", False), ("raw_num_directed_edges", "原始有向半边数", False),
             ("step_num_faces", "STEP面数（含非solid）", False), ("inference_seconds", "完整采样秒/件", True),
             ("post_wall_seconds", "后处理及检查秒/件", True)]
    for ax, (key, title, cdf) in zip(axes.flat, specs):
        for color, (method, row) in zip(COLORS, data.items()):
            values = np.asarray([float(r[key]) for r in row["records"] if r.get(key)])
            if cdf:
                values.sort(); ax.step(values, np.arange(1, len(values) + 1) / len(values), color=color, label=LABELS[method].replace("\n", " · "))
            else:
                bins = np.arange(0.5, max(31, values.max() + 2), 1) if key != "raw_num_directed_edges" else 30
                ax.hist(values, bins=bins, histtype="step", linewidth=1.8, density=True, color=color, label=LABELS[method].replace("\n", " · "))
        ax.set_title(title, loc="left"); ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("累计比例" if cdf else "频率密度")
        if key == "raw_num_faces":
            reference=[row["raw_num_faces"] for row in feature_records("reference")]
            ax.hist(reference,bins=np.arange(.5,32),histtype="step",linestyle="--",color="#515b63",label="validation参考 · n=1000",density=True)
    axes[0, 0].legend(fontsize=8)
    ax = axes[1, 2]
    for i, (method, row) in enumerate(data.items()):
        for j, key in enumerate(("Unique", "Novel")):
            ax.bar(i + (j - .5) * .3, row["graphs"][key] * 100, width=.3, color=COLORS[i], alpha=1 if j == 0 else .45)
    ax.set_xticks(range(3), list(LABELS.values()), fontsize=9); ax.set_ylim(0, 110)
    ax.set_title("Unique / Novel · raw图判据", loc="left")
    ax.set_ylabel("%（实色Unique / 浅色Novel）")
    fig.suptitle("数量、耗时与图重复性数据", fontsize=18)
    fig.text(.02, .012, "Unique/Novel：4bit面几何与邻接图同构；Novel对当前训练集46627件，不是旋转不变的训练记忆检验。", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .95)); save(fig, "distributions")


def complexity_plot(data):
    reference=feature_records("reference")
    source={"reference":[r["complexity"] for r in reference if r["complexity"] is not None]}
    for method in data:
        source[method]=[r["complexity"] for r in feature_records(method) if r["complexity"] is not None]
    keys=["num_faces","num_edges","num_vertices","cyclomatic_complexity","mean_curvature"]
    titles=["面数","边遍历计数（含重复）","坐标合并后的顶点数","现有环复杂度 E−V+2C","平均绝对曲率"]
    labels=[f"验证集参考\nn={len(source['reference'])}",
            *[LABELS[method]+f"\nn={len(source[method])}" for method in data]]
    fig,axes=plt.subplots(2,3,figsize=(17,10),facecolor="#f7f9fc")
    for ax,key,title in zip(axes.flat,keys,titles):
        for index,(method,rows) in enumerate(source.items()):
            values=[r[key] for r in rows]
            if values:
                ax.boxplot([values],positions=[index],widths=.5,whis=(5,95),showfliers=False,
                           medianprops={"color":"#25364a"},boxprops={"color":"#758ba0"})
            else:
                ax.text(index,.5,"N/A\nn=0",ha="center",transform=ax.get_xaxis_transform())
        ax.set_xticks(range(4),labels,fontsize=8);ax.set_title(title,loc="left")
        ax.spines[["top","right"]].set_visible(False)
        if key=="mean_curvature":
            ax.set_yscale("symlog",linthresh=.1)
            ax.set_ylim(bottom=0)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0,.7,"沿用现有Complexity实现\n0.1容差有效性检查\n箱线范围：P5–P95，数据未删除\n原始数值与各指标n见CSV",fontsize=13)
    fig.suptitle("已有复杂度指标 · 按现有0.1容差有效性检查计算",fontsize=17)
    fig.tight_layout(rect=(0,0,1,.95));save(fig,"complexity")
    write_json(OUT/"reference_complexity.json",{key:finite_stats([row[key] for row in source["reference"]]) for key in keys})


def training_curve():
    with (OUT / "training_history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#f7f9fc")
    for ax, key, title in zip(axes, ("train/loss_step_local", "val/loss"), ("训练loss（rank0逐步记录）", "验证loss（全局聚合）")):
        pairs = [(float(r["trainer/global_step"]), float(r[key])) for r in rows if r.get(key) and r.get("trainer/global_step")]
        points = np.asarray(pairs)
        ax.plot(points[:, 0], points[:, 1], color=COLORS[1], linewidth=.7)
        ax.set_title(title, loc="left"); ax.set_xlabel("日志global_step"); ax.set_yscale("log")
        ax.axvline(276185, color=COLORS[2], linestyle="--", label="最低val-loss checkpoint")
        ax.axvline(303500, color=COLORS[0], linestyle=":", label="last checkpoint")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=9)
    fig.suptitle("本次训练的已有loss记录", fontsize=16)
    fig.tight_layout(); save(fig, "training_curves")


def gallery(data):
    import trimesh
    fig = plt.figure(figsize=(12, 20), facecolor="#f7f9fc")
    for row in range(6):
        name = f"uncond_{row:06d}"
        for col, (method, folder) in enumerate(GROUPS.items()):
            ax = fig.add_subplot(6, 3, row * 3 + col + 1, projection="3d")
            status = json.loads((folder / "metadata/post/samples" / f"{name}.json").read_text())
            ax.set_axis_off()
            title = LABELS[method].replace("\n", " · ") + "\n" + name + (" · 合法solid" if status.get("native_valid_solid") else " · 非合法solid")
            ax.set_title(title, fontsize=9, color=COLORS[col])
            try:
                stl = folder / "post" / name / "recon_brep.stl"
                if stl.exists():
                    mesh = trimesh.load(stl, force="mesh", process=False)
                    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.faces)
                else:
                    vertices, triangles, _ = step_mesh(folder / "post" / name / "recon_brep.step")
                draw_mesh(ax, vertices, triangles, [])
            except Exception as error:
                ax.text2D(.5, .5, "预览失败\n" + str(error)[:45], ha="center", transform=ax.transAxes, fontsize=8)
    fig.suptitle("固定前6个seed · 各列独立生成 · 不是同一CAD的配对重建", fontsize=15)
    fig.subplots_adjust(top=.94, hspace=.28, wspace=.03)
    save(fig, "fixed_gallery")


def main():
    configure_fonts()
    OUT.mkdir(exist_ok=True)
    data = collect()
    tables(data); overview(data); balanced_sets(data); distributions(data); complexity_plot(data); training_curve(); gallery(data)


if __name__ == "__main__":
    main()
