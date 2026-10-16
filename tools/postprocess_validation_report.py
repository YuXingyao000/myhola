"""Fixed-input post-policy comparison and Chinese explanatory figures.

Reads isolated runner results; it never repairs geometry or selects by GT.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from tools.vae_compare_post import (
    REFERENCE_CONTRACT_FIELDS, TERMINAL, atomic_json, fingerprint, read_json,
    sha256_file, statistics,
)


POLICIES = ("legacy", "roundtrip_retry", "fit_score_v2", "recover_v1")
LABELS = ("原后处理", "检查边界＋继续尝试", "修正拟合评分", "原流程优先＋失败再修复")
COLORS = ("#68788e", "#3079ba", "#c99130", "#16877c")


def validate_policy_records(protocol: dict, rows: list[dict], policy: str,
                            split_hash: str, intended_count: int, original_count: int) -> None:
    expected = {"split_sha256": split_hash, "intended_count": intended_count,
                "original_test_count": original_count, "mode": "optimized"}
    for field, value in expected.items():
        if protocol.get(field) != value:
            raise ValueError(f"Protocol {field} differs from requested report: {policy}")
    if protocol.get("post_policy", "legacy") != policy:
        raise ValueError(f"Policy mismatch: {policy}")
    digest = fingerprint(protocol)
    for row in rows:
        if row.get("method") != "intersection" or row.get("mode") != "optimized":
            raise ValueError(f"Unexpected model/mode in {policy} record")
        if row.get("protocol_fingerprint") != digest:
            raise ValueError(f"Row protocol_fingerprint mismatch: {policy}/{row.get('sample_id')}")
        if row.get("post_policy", "legacy") != policy:
            raise ValueError(f"Row post_policy mismatch: {policy}/{row.get('sample_id')}")


def aggregate(records: dict[str, list[dict]], ids: list[str], original_count: int) -> dict:
    if len(ids) != len(set(ids)) or not ids or original_count < len(ids):
        raise ValueError("Invalid intended split/denominator")
    maps = {}
    for policy in POLICIES:
        rows = records[policy]
        if len(rows) != len(ids) or {r["sample_id"] for r in rows} != set(ids):
            raise ValueError(f"Record IDs differ: {policy}")
        if any(r.get("status") not in TERMINAL for r in rows):
            raise ValueError(f"Nonterminal record: {policy}")
        maps[policy] = {r["sample_id"]: r for r in rows}
    for sample_id in ids:
        if len({maps[p][sample_id].get("raw_sha256") for p in POLICIES}) != 1:
            raise ValueError(f"Raw predictions differ: {sample_id}")
    result = {"intended_count": len(ids), "original_count": original_count,
              "policies": {}, "paired": {}}
    for policy in POLICIES:
        rows = records[policy]
        count = sum(bool(r.get("native_valid_solid")) for r in rows)
        result["policies"][policy] = {
            "native_valid_count": count, "rate_accepted": count / len(ids),
            "rate_original": count / original_count,
            "compat_valid_count": sum(bool(r.get("compat_valid_solid")) for r in rows),
            "step_count": sum(bool(r.get("step_exists")) for r in rows),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "wall_seconds": statistics([r.get("wall_seconds") for r in rows]),
            "post_seconds": statistics([r.get("post_elapsed_seconds") for r in rows]),
            "geometry_count": sum(r.get("geometry_status") == "ok" for r in rows),
            "native_tolerances_on_valid": {
                name: statistics([r.get(f"native_{name}_tolerance_max") for r in rows
                                  if r.get("native_valid_solid")])
                for name in ("face", "edge", "vertex")},
            "valid_face_count_differs_from_raw": sum(
                r.get("native_valid_solid", False) and r.get("num_faces") != r.get("raw_num_faces")
                for r in rows),
        }
        if policy == "legacy":
            continue
        groups = {k: [] for k in ("both_valid", "rescued", "regressed", "both_failed")}
        for sample_id in ids:
            old, new = maps["legacy"][sample_id], maps[policy][sample_id]
            key = ("both_valid" if new.get("native_valid_solid") else "regressed") if old.get("native_valid_solid") else ("rescued" if new.get("native_valid_solid") else "both_failed")
            groups[key].append(sample_id)
        geometry = {}
        for group, chosen in (("common_computable", ids), ("common_valid", groups["both_valid"])):
            common = [i for i in chosen if all(maps[p][i].get("geometry_status") == "ok" for p in ("legacy", policy))]
            geometry[group] = {"count": len(common)}
            for metric in ("face_cd", "edge_cd"):
                geometry[group][metric] = {
                    p: statistics([maps[p][i][metric] for i in common]) for p in ("legacy", policy)}
                geometry[group][metric]["candidate_minus_baseline"] = statistics([
                    maps[policy][i][metric] - maps["legacy"][i][metric] for i in common])
        result["paired"][policy] = {"counts": {k: len(v) for k, v in groups.items()},
                                     "ids": groups, "geometry": geometry}
    return result


def configure_plot(font_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path).get_name(),
                         "font.size": 12, "axes.unicode_minus": False,
                         "savefig.facecolor": "#f6f8fc"})
    return plt


def draw_explanations(output: Path, font_path: Path) -> None:
    import numpy as np
    from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
    plt = configure_plot(font_path)
    output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 9), facecolor="#f6f8fc")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, 15), ylim=(0, 9)); ax.axis("off")
    ax.text(.5, 8.55, "从“形状差不多”到“可以闭合的实体”", fontsize=23, weight="bold", color="#17283c")
    ax.text(.5, 8.08, "每一步都可能引入误差；先修明确的程序行为，再处理更难的连续几何约束。", color="#52647a")
    boxes = [
        ("1  选择共享边", "两份半边 → 一条边", "风险：两边不一致、选错候选", "后续：同时看两张所属面"),
        ("2  调整几何", "移动面、缩放和移动边", "风险：点离得近，环仍不闭合", "后续：共用端点＋连续贴面"),
        ("3  拟合曲线 / 曲面", "采样点 → B-spline", "风险：评分单位错、漏看末端", "本轮：正确 RMS＋覆盖边界"),
        ("4  组环和裁剪", "边 → 环 → 裁剪面", "风险：失败的环被直接跳过", "本轮：已组好的环不能漏"),
        ("5  拼接、保存、重读", "面 → shell → solid", "风险：首个候选保存失败就停", "本轮：重读失败继续尝试"),
    ]
    for i, (title, flow, risk, action) in enumerate(boxes):
        x = .45 + i * 2.93
        ax.add_patch(FancyBboxPatch((x, 4.5), 2.65, 2.95, boxstyle="round,pad=.13", fc="white", ec="#d7e0eb"))
        color = "#16877c" if i >= 2 else "#68788e"
        ax.text(x+.12, 7.05, title, fontsize=13, weight="bold", color=color)
        ax.text(x+.12, 6.55, flow, fontsize=11)
        ax.text(x+.12, 5.95, risk.replace("：", "：\n"), fontsize=10.5, linespacing=1.7, color="#a24c3d")
        ax.text(x+.12, 5.05, action.replace("：", "：\n"), fontsize=10.5, linespacing=1.7, color=color)
        if i < 4:
            ax.annotate("", xy=(x+2.87, 6.5), xytext=(x+2.67, 6.5), arrowprops={"arrowstyle":"->", "color":"#8494a8", "lw":2})
    notes = [
        ("要保留：项目里的特殊圆柱", "它由采样网格拟合成 B-spline，可能出现完整的自然边界 loop。\n保留既有特殊处理；不能按解析圆柱的“两圆环＋seam”直接改。"),
        ("要分清：合法、完整、准确", "实体检查通过 ≠ 孔和边界都保留 ≠ 几何误差处处小于 1e-3。\n分别报告原生 STEP 合法率、边界检查、几何距离与耗时。"),
        ("另一个问题：旧 decoder 去重", "同平面上的网格漂移会漏检；共面本身也不等于同一张裁剪面。\n当前 refinement 评估未调用这段旧去重，本轮不修改模型。"),
    ]
    for i, (title, content) in enumerate(notes):
        y = 3.6 - i * 1.14
        ax.text(.6, y, title, color="#17283c", fontsize=13, weight="bold")
        ax.text(5.0, y+.05, content, fontsize=11, va="top", linespacing=1.7, color="#52647a")
    fig.savefig(output / "pipeline.png", dpi=145, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), facecolor="#f6f8fc")
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    fig.suptitle("这轮具体在修什么？", x=.05, ha="left", fontsize=23, weight="bold", color="#17283c")
    a = axes[0]
    a.text(0, .94, "① 拟合时，别漏掉最危险的边界", weight="bold", fontsize=13)
    x = np.linspace(.05, .92, 160)
    y = .66 + .11*np.sin(3*np.pi*x)*np.exp(2*(x-.8))
    a.plot(x, y, color="#df873b", lw=3)
    ix = np.array([4, 30, 56, 82, 108, 134, 159])
    a.scatter(x[ix], np.full(len(ix),.66), color="#16877c", s=28, zorder=3)
    a.scatter(x[-1], .66, facecolor="none", edgecolor="#c94848", s=180, lw=2)
    a.text(.03,.45,"绿色：输入采样点   橙色：拟合结果",fontsize=10)
    a.text(.03,.27,"旧检查会漏掉最后一个采样点。\n新检查覆盖末端、末行和末列，\n并用正确的距离单位评分。",fontsize=11,linespacing=1.8)
    a.text(.03,.06,"仍不是连续曲线处处精确的保证。",fontsize=10,color="#697b90")
    a=axes[1]; a.text(0,.94,"② 环失败，不能悄悄当它不存在",weight="bold",fontsize=13)
    for cx, hole in ((.24,True),(.77,False)):
        a.add_patch(Rectangle((cx-.18,.49),.36,.31,fc="#d9ece8",ec="#16877c",lw=2))
        if hole:
            a.add_patch(Circle((cx,.645),.09,fc="white",ec="#c94848",lw=2,ls="--"))
    a.annotate("",xy=(.56,.64),xytext=(.45,.64),arrowprops={"arrowstyle":"->","color":"#a24c3d","lw":2})
    a.text(.04,.4,"有孔的候选",fontsize=10); a.text(.6,.4,"孔被丢掉（示意）",fontsize=10)
    a.text(.03,.23,"原来有些失败环会被跳过。\n新修复要求已组好的普通环全保留；\n圆柱特殊 loop 转换单独记录。",fontsize=11,linespacing=1.8)
    a.text(.03,.02,"尚不能拦住：组环时已把两环接成一环。",fontsize=9.5,color="#697b90")
    a=axes[2];a.text(0,.94,"③ 保存失败，继续试下一候选",weight="bold",fontsize=13)
    for y,txt,col in ((.74,"候选 A：拼成了 → 重读失败","#b95b50"),(.55,"候选 B：拼成了 → 重读通过","#16877c")):
        a.add_patch(FancyBboxPatch((.03,y-.08),.91,.14,boxstyle="round,pad=.01",fc="white",ec=col))
        a.text(.07,y-.01,txt,fontsize=10,color=col)
    a.annotate("",xy=(.48,.64),xytext=(.48,.66),arrowprops={"arrowstyle":"->","color":"#68788e"})
    a.text(.03,.23,"原来拿到首个 solid 候选就停。\n现在以保存后原生重读为准，\n失败再继续，候选状态相互隔离。",fontsize=11,linespacing=1.8)
    a.text(.03,.02,"总时间预算固定；额外尝试成本照实报告。",fontsize=9.5,color="#697b90")
    fig.tight_layout(rect=[0,0,1,.90]);fig.savefig(output/"changes.png",dpi=145,bbox_inches="tight");plt.close(fig)


def draw_results(summary: dict, output: Path, font_path: Path) -> None:
    plt = configure_plot(font_path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#f6f8fc")
    values = [summary["policies"][p]["native_valid_count"] for p in POLICIES]
    n = summary["intended_count"]
    bars = axes[0].barh(LABELS, [100*v/n for v in values], color=COLORS, height=.55)
    axes[0].set(xlim=(0,100),xlabel=f"原生合法 solid / {n} 个预定 validation 样本（%）",title="固定 VAE、固定原始预测")
    axes[0].invert_yaxis()
    for bar,v in zip(bars,values):
        axes[0].text(1,bar.get_y()+bar.get_height()/2,f"{v}/{n} · {100*v/n:.2f}%",va="center",color="white",weight="bold")
    x=range(3)
    gains=[summary["paired"][p]["counts"]["rescued"] for p in POLICIES[1:]]
    losses=[summary["paired"][p]["counts"]["regressed"] for p in POLICIES[1:]]
    axes[1].bar(x,gains,color="#16877c",label="原失败 → 新成功")
    axes[1].bar(x,[-v for v in losses],color="#bb6255",label="原成功 → 新失败")
    axes[1].set(xticks=list(x),xticklabels=["边界＋重试","拟合评分","失败再修复"],ylabel="件数",title="和原后处理逐件配对",
                ylim=(-max(losses)-12, max(gains)+12))
    for i,(g,l) in enumerate(zip(gains,losses)):
        axes[1].annotate(f"+{g}",(i,g),xytext=(0,5),textcoords="offset points",ha="center")
        axes[1].annotate(f"−{l}" if l else "0",(i,-l),xytext=(0,-15),textcoords="offset points",ha="center")
    axes[1].axhline(0,color="#66768a",lw=.8);axes[1].legend(loc="best",fontsize=10)
    for a in axes:
        a.spines[["top","right"]].set_visible(False)
    fig.tight_layout();fig.savefig(output/"results.png",dpi=150,bbox_inches="tight");plt.close(fig)


def draw_case(run_root: Path, sample_id: str, output: Path, font_path: Path) -> dict:
    """Two common camera views; missing triangulations are explicitly labelled."""
    import numpy as np
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCC.Core.TopLoc import TopLoc_Location
    from shared.occ_utils import get_primitives
    from tools.vae_compare_post import native_validity, read_step_at_precision

    plt = configure_plot(font_path)
    protocol = read_json(run_root / "post_variants/legacy/metadata/post/optimized/protocol.json")
    paths = [Path(protocol["data_root"]) / sample_id / "normalized_shape.step"]
    paths.extend(run_root / "post_variants" / policy / "post/optimized/intersection" / sample_id / "recon_brep.step"
                 for policy in ("legacy", "recover_v1"))
    geometries = []
    all_points = []
    for path in paths:
        shape = read_step_at_precision(path)
        validity = native_validity(shape)
        polygons, curves, unmeshed = [], [], 0
        for face in get_primitives(shape, TopAbs_FACE, v_remove_half=True):
            BRepMesh_IncrementalMesh(face, .005, False, .1)
            loc = TopLoc_Location()
            mesh = BRep_Tool.Triangulation(face, loc)
            if mesh is None or not mesh.NbTriangles():
                unmeshed += 1
                continue
            vertices = np.array([mesh.Node(i).Transformed(loc.Transformation()).Coord()
                                 for i in range(1, mesh.NbNodes()+1)])
            triangles = np.array([mesh.Triangle(i).Get() for i in range(1, mesh.NbTriangles()+1)]) - 1
            polygons.extend(vertices[triangles])
            all_points.append(vertices)
        for edge in get_primitives(shape, TopAbs_EDGE, v_remove_half=True):
            if BRep_Tool.Degenerated(edge):
                continue
            curve = BRepAdaptor_Curve(edge)
            points = np.array([curve.Value(float(t)).Coord() for t in
                               np.linspace(curve.FirstParameter(), curve.LastParameter(), 128)])
            curves.append(points)
            all_points.append(points)
        geometries.append((polygons, curves, unmeshed, validity))
    points = np.concatenate(all_points)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = float(np.max(np.ptp(points, axis=0))) * .54
    fig = plt.figure(figsize=(14, 8.5), facecolor="#f6f8fc")
    titles = ("参考 CAD（只用于评估）", "原后处理", "失败后修复")
    colors = ("#abb8c9", "#dba58b", "#76b9ad")
    for column, (polygons, curves, unmeshed, validity) in enumerate(geometries):
        for row, angle in enumerate((35, 145)):
            ax = fig.add_subplot(2, 3, row*3+column+1, projection="3d", facecolor="#f6f8fc")
            if polygons:
                triangles = np.asarray(polygons)
                normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
                lengths = np.linalg.norm(normals, axis=1, keepdims=True)
                normals = normals / np.maximum(lengths, 1e-15)
                light = np.array([.3, -.5, 1.]); light /= np.linalg.norm(light)
                intensity = .48 + .52 * np.abs(normals @ light)
                facecolors = np.asarray(to_rgb(colors[column])) * intensity[:, None]
                ax.add_collection3d(Poly3DCollection(polygons, facecolors=facecolors, edgecolors="none", alpha=1.0))
            for curve in curves:
                ax.plot(*curve.T, color="#243c49", linewidth=.65)
            ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
                   zlim=(center[2]-radius, center[2]+radius))
            ax.set_box_aspect((1, 1, 1), zoom=1.2); ax.view_init(elev=23, azim=angle); ax.set_axis_off()
            if row == 0:
                text = f"{titles[column]}\n{'合法 solid' if validity['native_valid_solid'] else '未通过 solid 检查'} · {validity['num_faces']} 面"
                maximum_tolerance = max(validity[f"native_{name}_tolerance_max"] or 0.0
                                        for name in ("face", "edge", "vertex"))
                text += f"\n最大存储容差 {maximum_tolerance:.3g}"
                if unmeshed:
                    text += f" · {unmeshed} 面无法绘制"
                ax.set_title(text, fontsize=12)
    diagnostics = read_json(run_root / "post_variants/recover_v1/post/optimized/intersection" / sample_id / "post_diagnostics.json")
    path_label = {"roundtrip_fit_only": "跳过几何优化后救回", "roundtrip_optimized": "完整边界和重试后救回",
                  "fit_score_v2_roundtrip_optimized": "修正拟合后救回"}.get(diagnostics.get("chosen_path"), diagnostics.get("chosen_path", "unknown"))
    fig.suptitle(f"验证样本 {sample_id} · {path_label}", x=.04, ha="left", fontsize=18, weight="bold")
    fig.text(.05, .02, "两行是不同观察角度；三列使用相同坐标与比例。图像用于看形状，合法性由保存后的原生 STEP 检查确定。", color="#52647a", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .94))
    target = output / f"case_{sample_id}.png"
    fig.savefig(target, dpi=145, bbox_inches="tight"); plt.close(fig)
    return {"sample_id": sample_id, "figure": str(target), "paths": list(map(str, paths)),
            "views": [(23, 35), (23, 145)], "chosen_path": diagnostics.get("chosen_path"),
            "validity": [item[3] for item in geometries],
            "unmeshed_face_counts": [item[2] for item in geometries]}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--split-list",type=Path,required=True)
    parser.add_argument("--original-count",type=int,default=3505)
    parser.add_argument("--font-path",type=Path,default=Path("/root/.cache/hola-browser-libs/fonts/NotoSansCJKsc-Regular.otf"))
    parser.add_argument("--figures-only",action="store_true")
    parser.add_argument("--case-ids", nargs="*", default=[], help="Illustration only; does not select evaluation samples")
    args=parser.parse_args()
    output=args.run_root/"report";output.mkdir(parents=True,exist_ok=True)
    draw_explanations(output,args.font_path)
    if args.figures_only:return
    ids=[s.strip() for s in args.split_list.read_text().splitlines() if s.strip()]
    records={};protocols={}
    for policy in POLICIES:
        root=args.run_root/"post_variants"/policy
        meta=root/"metadata/post/optimized"
        protocols[policy]=read_json(meta/"protocol.json")
        if protocols[policy].get("post_policy","legacy") != policy:
            raise ValueError(f"Policy mismatch: {policy}")
        records[policy]=[read_json(meta/"samples/intersection"/(i+".json")) for i in ids]
        validate_policy_records(protocols[policy], records[policy], policy,
                                sha256_file(args.split_list), len(ids), args.original_count)
    fields = REFERENCE_CONTRACT_FIELDS
    for field in fields:
        if len({fingerprint(p.get(field)) for p in protocols.values()}) != 1:
            raise ValueError(f"Comparison contract differs: {field}")
    summary=aggregate(records,ids,args.original_count)
    summary["protocols"]=protocols
    atomic_json(output/"summary.json",summary)
    with (output/"cases.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["policy","sample_id","status","native_valid_solid","face_cd","edge_cd","wall_seconds"])
        writer.writeheader()
        for policy,rows in records.items():
            for row in rows:writer.writerow({"policy":policy,**{k:row.get(k) for k in writer.fieldnames if k != "policy"}})
    draw_results(summary,output,args.font_path)
    if args.case_ids:
        if set(args.case_ids) - set(ids):
            raise ValueError("Illustrated case is outside the frozen evaluation split")
        cases = [draw_case(args.run_root, sample_id, output, args.font_path) for sample_id in args.case_ids]
        atomic_json(output / "illustrated_cases.json", {"selection": "Explicit illustrative IDs; no effect on evaluation or repair selection", "cases": cases})
    print(json.dumps({p:v["native_valid_count"] for p,v in summary["policies"].items()}))


if __name__ == "__main__":
    main()
