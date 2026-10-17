"""Render a Chinese comparison table from completed report JSON/CSV only.

The selected methods define each common metric subset. No model, dataset or
STEP is opened, and missing values are never replaced with invented scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_METHODS = ["hola_1500k", "current_best_54639", "current_last_182130", "alignment_best_250k"]
LABELS = {
    "hola_1500k": ("原始 HoLa SVR", "用户指定的历史基线"),
    "current_best_54639": ("当前逐层条件 · best", "训练时验证 loss 最佳"),
    "current_last_182130": ("当前逐层条件 · last", "最后实际保存的 checkpoint"),
    "alignment_150k": ("历史 alignment · 150k", "对比对齐目标版本"),
    "alignment_best_250k": ("历史 CLIP 式对齐", "750 件小集 · 该 run callback best"),
    "alignment_last_352152": ("历史 alignment · last", "该次训练最后保存"),
}
TERMINAL_POST = {"ok", "missing_raw", "invalid_raw", "post_error", "no_step", "step_read_error",
                 "geometry_error", "timeout", "crash", "interrupted"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def indexed_rows(rows: list[dict], methods: list[str], intended: int) -> dict[str, dict[str, dict]]:
    result = {method: {} for method in methods}
    for row in rows:
        method = row["method"]
        if method not in result:
            continue
        sample_id = row["sample_id"]
        if sample_id in result[method]:
            raise ValueError(f"Repeated {method}/{sample_id} in report CSV")
        result[method][sample_id] = row
    reference = set(result[methods[0]])
    if len(reference) != intended:
        raise ValueError("CSV does not cover the fixed intended sample count")
    for method, records in result.items():
        if set(records) != reference:
            raise ValueError(f"CSV IDs differ for {method}")
    return result


def common_metric(rows: dict[str, dict[str, dict]], key: str) -> dict:
    """Recompute the selected methods' intersection, independent of summary scope."""
    values = {m: {i: number(r.get(key)) for i, r in records.items()} for m, records in rows.items()}
    valid = [{i for i, v in items.items() if v is not None} for items in values.values()]
    common = sorted(set.intersection(*valid))
    return {"metric": key, "n": len(common), "ids": common,
            "mean": {m: float(np.mean([items[i] for i in common])) if common else None
                     for m, items in values.items()}}


def own_metric(rows: dict[str, dict[str, dict]], key: str) -> dict:
    result = {}
    for method, records in rows.items():
        values = [number(r.get(key)) for r in records.values()]
        values = [v for v in values if v is not None]
        result[method] = {"n": len(values), "mean": float(np.mean(values)) if values else None}
    return result


def font_paths(explicit: str | None) -> tuple[Path, Path | None]:
    candidates = ([Path(explicit)] if explicit else []) + [
        Path("/root/.cache/hola-browser-libs/fonts/SourceHanSansSC-Regular.otf"),
        Path("/root/.cache/hola-browser-libs/fonts/NotoSansCJKsc-Regular.otf"),
        Path("/tmp/NotoSansCJKsc-Regular.otf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
    ]
    regular = next((p for p in candidates if p.is_file()), None)
    if regular is None:
        for name in ("Source Han Sans SC", "Noto Sans CJK SC", "Microsoft YaHei"):
            try:
                regular = Path(font_manager.findfont(name, fallback_to_default=False))
                break
            except ValueError:
                pass
    if regular is None:
        raise FileNotFoundError("A Chinese font is required; provide --font /path/to/font.otf")
    bold = regular.with_name(regular.name.replace("Regular", "Bold"))
    if not bold.is_file():
        bold = Path("/tmp/NotoSansCJKsc-Bold.otf")
    return regular, bold if bold.is_file() else None


def load_table(args) -> tuple[dict, dict]:
    summary_path = args.report / "summary.json"
    summary = json.loads(summary_path.read_text())
    methods = args.methods
    if len(set(methods)) != len(methods) or not methods:
        raise ValueError("Select unique nonempty methods")
    if any(m not in summary["methods"] for m in methods):
        raise ValueError("Selected method has no summary")
    intended = int(summary["intended_count"])
    raw = indexed_rows(read_csv(args.report / "raw_per_object.csv"), methods, intended)
    processed = indexed_rows(read_csv(args.report / "post_per_object.csv"), methods, intended)
    inference = indexed_rows(read_csv(args.report / "inference_per_object.csv"), methods, intended)
    reference = set(raw[methods[0]])
    if set(processed[methods[0]]) != reference or set(inference[methods[0]]) != reference:
        raise ValueError("Stages do not have the same intended IDs")
    for method in methods:
        if any(r.get("status") not in TERMINAL_POST for r in processed[method].values()):
            raise ValueError(f"Post-processing is incomplete for {method}; refusing a final table")
        if any(r.get("status") not in {"ok", "error"} for r in inference[method].values()):
            raise ValueError(f"Inference is incomplete for {method}; refusing a final table")
    supplied_steps = {}
    for item in args.steps:
        method, value = item.split("=", 1)
        if method in supplied_steps:
            raise ValueError(f"Duplicate step metadata for {method}")
        supplied_steps[method] = int(value)
    steps = {}
    for method in methods:
        recorded = summary["methods"][method].get("checkpoint_step")
        supplied = supplied_steps.get(method)
        if recorded is not None and supplied is not None and int(recorded) != supplied:
            raise ValueError(f"Conflicting checkpoint step for {method}")
        steps[method] = int(recorded) if recorded is not None else supplied
        if steps[method] is None or steps[method] < 0:
            raise ValueError(f"Provide audited checkpoint metadata: --steps {method}=ACTUAL_STEP")
    counts = {}
    for method in methods:
        counts[method] = {
            "inference_ok": sum(r["status"] == "ok" for r in inference[method].values()),
            "raw_complete": sum(str(r.get("raw_complete")).lower() == "true" for r in raw[method].values()),
            "native_valid_solids": sum(str(r.get("native_valid_solid")).lower() == "true"
                                       for r in processed[method].values()),
            "step_outputs": sum(str(r.get("step_exists")).lower() == "true" for r in processed[method].values()),
            "post_timeouts": sum(r["status"] == "timeout" for r in processed[method].values()),
        }
        expected = summary["methods"][method]
        for key in ("raw_complete", "native_valid_solids", "step_outputs", "post_timeouts"):
            if counts[method][key] != expected["counts"][key]:
                raise ValueError(f"CSV and summary disagree: {method}/{key}")
    table = {"methods": methods, "intended": intended, "steps": steps, "counts": counts,
             "raw_face_cd": common_metric(raw, "face_cd_sq"),
             "raw_face_f1": common_metric(raw, "face_f1"),
             "post_face_cd": common_metric(processed, "face_cd_sq"),
             "inference_seconds": own_metric(inference, "inference_seconds")}
    provenance = {"source_files": {p.name: sha256(p) for p in [summary_path, args.report / "raw_per_object.csv",
                  args.report / "post_per_object.csv", args.report / "inference_per_object.csv"]},
                  "methods": methods, "step_metadata_cli": supplied_steps,
                  "script_sha256": sha256(Path(__file__)), "table_values": table,
                  "common_subset_scope": "Only the explicitly selected methods; computed anew from CSV.",
                  "post_subset": "Common computable native STEP face geometry; not restricted to valid solids."}
    return table, provenance


def render(args, table: dict, provenance: dict) -> None:
    regular_path, bold_path = font_paths(args.font)
    regular = font_manager.FontProperties(fname=str(regular_path))
    bold = font_manager.FontProperties(fname=str(bold_path or regular_path), weight="bold")
    mono_path = Path("/tmp/hola-commitmono-sync/CommitMono/CommitMono-400-Regular.otf")
    mono = font_manager.FontProperties(fname=str(mono_path)) if mono_path.is_file() else regular
    plt.rcParams.update({"svg.fonttype": "path", "axes.unicode_minus": False})
    methods, intended = table["methods"], table["intended"]
    fig_height = 7.0 + max(0, len(methods) - 4) * 0.7
    fig, ax = plt.subplots(figsize=(18, fig_height), dpi=180)
    fig.patch.set_facecolor("#f5f7fb")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.025, top=0.975)

    def text(x, y, value, *, size=12, color="#203048", align="left", heavy=False, numeric=False):
        return ax.text(x, y, value, va="center", ha=align, fontsize=size, color=color,
                       fontproperties=mono if numeric else (bold if heavy else regular), linespacing=1.65)

    text(0.012, 0.946, "单视图灰模条件生成 · 定量对照", size=24, heavy=True)
    text(0.012, 0.886, f"DeepCAD validation 固定 {intended} 件  /  每图 1 次生成  /  原始 DDPM 1000 步", size=12, color="#52627b")
    text(0.988, 0.946, "SVR · 2026-10-17", size=12, align="right", color="#52627b", numeric=True)

    widths = [0.208, 0.105, 0.092, 0.117, 0.092, 0.133, 0.132, 0.121]
    edges = np.r_[0.0, np.cumsum(widths)]
    centers = (edges[:-1] + edges[1:]) / 2
    top, header_height = 0.82, 0.12
    row_height = 0.405 / len(methods)
    headers = ["模型 / checkpoint", "已训练步数", f"推理成功\n/ {intended}",
               f"原始面 CD² ↓\n×1000 · n={table['raw_face_cd']['n']}",
               f"原始面 F1 ↑\nn={table['raw_face_f1']['n']}",
               f"合法 solid ↑\n原生精度 / {intended}",
               f"后处理面 CD² ↓\n×1000 · n={table['post_face_cd']['n']}", "推理耗时 ↓\n秒 / 件"]
    ax.add_patch(Rectangle((0, top - header_height), 1, header_height, facecolor="#203048", edgecolor="none"))
    for index, header in enumerate(headers):
        text(0.015 if index == 0 else centers[index], top - header_height / 2,
             header, size=10.5, color="white", align="left" if index == 0 else "center", heavy=True)

    def winners(values: dict, larger: bool = False) -> set[str]:
        finite = {m: v for m, v in values.items() if v is not None}
        if not finite:
            return set()
        best = (max if larger else min)(finite.values())
        return {m for m, value in finite.items() if math.isclose(value, best, rel_tol=1e-10, abs_tol=1e-12)}

    best = {
        3: winners(table["raw_face_cd"]["mean"]),
        4: winners(table["raw_face_f1"]["mean"], larger=True),
        5: winners({m: table["counts"][m]["native_valid_solids"] for m in methods}, larger=True),
        6: winners(table["post_face_cd"]["mean"]),
    }
    for index, method in enumerate(methods):
        y1 = top - header_height - index * row_height
        y0 = y1 - row_height
        ax.add_patch(Rectangle((0, y0), 1, row_height, facecolor="white" if index % 2 == 0 else "#edf1f7", edgecolor="none"))
        label, note = LABELS.get(method, (method, ""))
        text(0.015, (y0 + y1) / 2 + 0.014, label, size=12, heavy=True)
        text(0.015, (y0 + y1) / 2 - 0.022, note, size=8.5, color="#62718a")
        counts = table["counts"][method]

        def scaled(metric: str, scale: float = 1.0, suffix: str = "", decimals: int = 3) -> str:
            value = table[metric]["mean"][method]
            return "—" if value is None else f"{value * scale:.{decimals}f}{suffix}"

        timing = table["inference_seconds"][method]
        values = ["", f"{table['steps'][method]:,}", f"{counts['inference_ok']}/{intended}",
                  scaled("raw_face_cd", 1000), scaled("raw_face_f1", 100, "%", 1),
                  f"{counts['native_valid_solids']}/{intended}", scaled("post_face_cd", 1000),
                  "—" if timing["mean"] is None else f"{timing['mean']:.2f}"]
        for column in range(1, len(values)):
            is_best = method in best.get(column, set())
            if is_best:
                ax.add_patch(Rectangle((edges[column] + 0.004, y0 + 0.012), widths[column] - 0.008,
                                       row_height - 0.024, facecolor="#dceee6", edgecolor="none"))
            text(centers[column], (y0 + y1) / 2 + (0.009 if column in (2, 5, 7) else 0),
                 values[column], size=13.5, align="center", numeric=True,
                 color="#13704e" if is_best else "#203048")
            if column == 2:
                text(centers[column], (y0 + y1) / 2 - 0.026, f"可后处理 {counts['raw_complete']}/{intended}",
                     size=7.8, align="center", color="#62718a")
            elif column == 5:
                text(centers[column], (y0 + y1) / 2 - 0.026, f"STEP {counts['step_outputs']} · 超时 {counts['post_timeouts']}",
                     size=7.8, align="center", color="#62718a")
            elif column == 7:
                text(centers[column], (y0 + y1) / 2 - 0.026, f"n={timing['n']} · 含首件", size=7.8,
                     align="center", color="#62718a")

    bottom = top - header_height - row_height * len(methods)
    notes = [
        f"几何均值：各列 n 为图中 {len(methods)} 个方法共同可计算的同一批 ID；后处理集合包含可测量的非 solid STEP。",
        "CD² 为双向最近邻平方距离均值之和；原始网格与后处理裁剪面分列，均不做旋转搜索、缩放或 ICP。",
        "Face F1：以面 CD² 做 Hungarian 匹配，cost < 0.1 记为命中；绿色仅表示该列本次最优数值。",
        "后处理：所有模型同一 legacy 策略，drop=0、名义优化 200 步、单件总预算 180 秒；失败仍计入 100 件分母。",
        "不同 checkpoint 的训练历史与预算不同；本表属于 validation 开发对照，不能归因于单一架构改动。",
        "历史导出的整数值浮点索引仅无损转为 int64 供后处理读取；原始几何、连接数值与原始预测文件均保留。",
        "CLIP 式对齐行是已恢复的历史候选，尚未确认是用户所指最佳；其训练和验证曾共用 750 件小集。",
    ]
    for index, line in enumerate(notes):
        text(0.012, bottom - 0.045 - index * 0.036, line, size=9.3, color="#52627b")
    args.output.mkdir(parents=True, exist_ok=True)
    basename = args.output / args.name
    for suffix in ("png", "svg"):
        fig.savefig(basename.with_suffix("." + suffix), dpi=180, facecolor=fig.get_facecolor())
    # Matplotlib leaves trailing spaces in SVG paths; normalize only whitespace.
    svg = basename.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    provenance.update(font_regular=str(regular_path), font_bold=str(bold_path) if bold_path else None,
                      output_files={suffix: str(basename.with_suffix("." + suffix)) for suffix in ("png", "svg")})
    basename.with_suffix(".json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(provenance["output_files"], indent=2))


def self_check() -> None:
    # A method omitted from the chart must not narrow the shown intersection.
    rows = {"a": {"1": {"metric": "1"}, "2": {"metric": "3"}},
            "b": {"1": {"metric": "2"}, "2": {"metric": "5"}},
            "hidden": {"1": {"metric": "4"}, "2": {"metric": ""}}}
    assert common_metric(rows, "metric")["n"] == 1
    visible = common_metric({k: rows[k] for k in ("a", "b")}, "metric")
    assert visible["n"] == 2 and visible["mean"] == {"a": 2.0, "b": 3.5}
    assert number("nan") is None and number("") is None
    print("Visual metric self-check passed: selected-method intersection and missing values.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "report")
    parser.add_argument("--output", type=Path, default=ROOT / "report")
    parser.add_argument("--name", default="quantitative_comparison")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--steps", nargs="*", default=[], help="Audited actual checkpoint steps: METHOD=INTEGER")
    parser.add_argument("--font", help="Optional explicit Chinese OTF/TTF/TTC font")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    table, provenance = load_table(args)
    render(args, table, provenance)


if __name__ == "__main__":
    main()
