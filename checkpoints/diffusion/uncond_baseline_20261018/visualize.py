"""绘制旧无条件架构与冻结协议下的结果；不调用模型或改变生成样本。"""

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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parent
INK = "#26384c"
MUTED = "#66778a"
BLUE = "#e8eff8"
TEAL = "#e4f2ec"
ORANGE = "#fff0dc"
PURPLE = "#eee9f7"


def configure_fonts() -> dict[str, font_manager.FontProperties]:
    """嵌入本机可用字体；中文缺字时明确失败，不生成乱码图。"""
    regular = Path("/root/.cache/hola-browser-libs/fonts/NotoSansCJKsc-Regular.otf")
    bold = regular.with_name("NotoSansCJKsc-Bold.otf")
    if not regular.is_file():
        raise FileNotFoundError(f"Chinese font is required: {regular}")
    fonts = {"body": font_manager.FontProperties(fname=str(regular)),
             "bold": font_manager.FontProperties(fname=str(bold if bold.is_file() else regular))}
    mono = Path("/tmp/hola-commitmono-sync/CommitMono/CommitMono-400-Regular.otf")
    if mono.is_file():
        fonts["mono"] = font_manager.FontProperties(fname=str(mono))
    else:
        fonts["mono"] = font_manager.FontProperties(family="DejaVu Sans Mono")
    font_manager.fontManager.addfont(str(regular))
    plt.rcParams.update({"font.family": fonts["body"].get_name(), "svg.fonttype": "path",
                         "axes.unicode_minus": False, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    return fonts


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(path.with_suffix("." + suffix), dpi=175, facecolor=fig.get_facecolor())
    # Matplotlib 的 SVG path 可能带行尾空格，仅规范空白，不改变图形。
    svg_path = path.with_suffix(".svg")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def architecture(output: Path, fonts: dict) -> None:
    """展开一个 encoder block，另外标出完整 DDPM 的反馈回路。"""
    fig, ax = plt.subplots(figsize=(24, 15), dpi=175)
    fig.patch.set_facecolor("#f7f9fc")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    ax.set(xlim=(0, 24), ylim=(0, 15))
    ax.axis("off")

    def label(x, y, text, size=12, *, bold=False, mono=False, color=INK,
              ha="center", va="center"):
        return ax.text(x, y, text, fontsize=size, color=color, ha=ha, va=va,
                       fontproperties=fonts["mono" if mono else "bold" if bold else "body"],
                       linespacing=1.55)

    def box(x, y, w, h, title, lines="", color=BLUE, size=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.035,rounding_size=0.09",
                                   facecolor=color, edgecolor="#a9b9ca", linewidth=1.2))
        label(x+w/2, y+h-(0.28 if lines else h/2), title, size+1, bold=True)
        if lines:
            label(x+w/2, y+(h-0.45)/2, lines, size)

    def arrow(start, end, color=MUTED, *, dashed=False, rad=0):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=17,
                                    linewidth=1.6, color=color,
                                    linestyle="--" if dashed else "-",
                                    connectionstyle=f"arc3,rad={rad}"))

    def add(x, y):
        ax.add_patch(Circle((x, y), 0.19, facecolor="white", edgecolor=INK, linewidth=1.5))
        label(x, y, "+", 19, mono=True)

    label(0.3, 14.45, "历史 HoLa 无条件生成 · 实际恢复架构", 25, bold=True, ha="left")
    label(0.3, 13.87, "Diffusion_uncond_1100k.ckpt  ·  实际 global_step = 1,000,000  ·  FP32 / eval / 无 CFG", 13,
          color=MUTED, ha="left")
    label(0.3, 13.24, "① 每个去噪时间步：32 维 face latent → 768 维特征 + 256 维零条件 → 1024 维主干", 15,
          bold=True, ha="left")

    box(0.3, 10.9, 2.2, 1.6, "Noisy face latent", "Z_t\n[B,30,32]", TEAL)
    box(3.05, 10.9, 3.15, 1.6, "输入 MLP", "Linear 32 → 768\nLayerNorm · SiLU\nLinear 768 → 768")
    arrow((2.5, 11.7), (3.05, 11.7))
    box(6.8, 10.9, 2.65, 1.6, "Concat（末维）", "[B,30,768]\n＋ [B,30,256]\n= [B,30,1024]", ORANGE)
    arrow((6.2, 11.7), (6.8, 11.7))
    box(6.8, 9.3, 2.65, 1.0, "零条件", "zeros [B,1,256] → repeat 30", ORANGE, size=9)
    arrow((8.125, 10.3), (8.125, 10.9))
    add(10.1, 11.7)
    arrow((9.45, 11.7), (9.91, 11.7))
    box(10.8, 10.9, 3.55, 1.6, "TransformerEncoder × 24", "24 个完整 pre-LN block\nhidden 1024 · 16 heads\n层内结构见下图")
    arrow((10.29, 11.7), (10.8, 11.7))
    box(15.0, 10.9, 3.45, 1.6, "最终 LN + 输出 MLP", "LayerNorm 1024\nLinear 1024 → 1024 · LN · SiLU\nLinear 1024 → 32")
    arrow((14.35, 11.7), (15.0, 11.7))
    box(19.1, 10.9, 2.3, 1.6, "预测噪声", "εθ(Z_t,t)\n[B,30,32]", TEAL)
    arrow((18.45, 11.7), (19.1, 11.7))

    box(0.3, 9.3, 2.2, 1.0, "时间步", "t: [B]", PURPLE)
    box(3.05, 9.05, 3.15, 1.5, "时间嵌入", "SinCos 1024\nLinear 1024 · LN · SiLU\nLinear 1024 → [B,1,1024]", PURPLE, size=10)
    arrow((2.5, 9.8), (3.05, 9.8))
    ax.plot([6.2, 6.5, 6.5, 10.1, 10.1], [9.8, 9.8, 9.0, 9.0, 11.5], color="#8b73ac", lw=1.6)
    arrow((10.1, 10.95), (10.1, 11.5), "#8b73ac")
    label(10.95, 9.5, "沿 30 个 slots 广播相加\n不在层间重复注入时间", 10, color="#786195", ha="left")

    box(19.1, 9.15, 2.3, 1.1, "DDPM 更新", "Z_t → Z_(t-1)", TEAL)
    arrow((20.25, 10.9), (20.25, 10.25))
    ax.plot([21.4, 22.4, 22.4, 0.08, 0.08], [9.7, 9.7, 8.4, 8.4, 11.7],
            color="#3d8a6b", lw=1.5, linestyle="--")
    arrow((0.08, 11.7), (0.3, 11.7), "#3d8a6b", dashed=True)
    label(16.2, 8.7, "1000 步 linear DDPM；同一网络反复运行", 11, color="#287052")

    label(0.3, 7.75, "② 单个 TransformerEncoderLayer（24 层各自独立参数）", 15, bold=True, ha="left")
    ax.add_patch(FancyBboxPatch((0.3, 4.55), 23.1, 2.65, boxstyle="round,pad=0.03,rounding_size=0.1",
                               facecolor="white", edgecolor="#c4cfdb", linestyle="--", linewidth=1.5))
    label(1.4, 5.85, "h_l\n[B,30,1024]", 11)
    box(3.0, 5.15, 1.35, 1.4, "LN", "1024", size=11)
    box(4.9, 5.15, 3.35, 1.4, "Self-attention", "Q,K,V: 1024 → 1024\n16 heads × 64 dim\nattention dropout 0.1", size=10)
    box(8.8, 5.15, 1.45, 1.4, "Dropout", "p = 0.1", size=10)
    add(10.9, 5.85)
    arrow((2.25, 5.85), (3.0, 5.85)); arrow((4.35, 5.85), (4.9, 5.85))
    arrow((8.25, 5.85), (8.8, 5.85)); arrow((10.25, 5.85), (10.71, 5.85))
    ax.plot([2.55, 2.55, 10.9, 10.9], [5.85, 6.85, 6.85, 6.05], color=MUTED, lw=1.3)
    arrow((10.9, 6.4), (10.9, 6.04))
    label(6.6, 6.93, "残差连接", 9, color=MUTED)
    box(11.65, 5.15, 1.35, 1.4, "LN", "1024", size=11)
    box(13.55, 5.15, 3.9, 1.4, "FFN", "Linear 1024 → 2048\nReLU → Dropout 0.1\nLinear 2048 → 1024", size=10)
    box(18.0, 5.15, 1.45, 1.4, "Dropout", "p = 0.1", size=10)
    add(20.1, 5.85)
    arrow((11.09, 5.85), (11.65, 5.85)); arrow((13.0, 5.85), (13.55, 5.85))
    arrow((17.45, 5.85), (18.0, 5.85)); arrow((19.45, 5.85), (19.91, 5.85))
    ax.plot([11.35, 11.35, 20.1, 20.1], [5.85, 6.85, 6.85, 6.05], color=MUTED, lw=1.3)
    arrow((20.1, 6.4), (20.1, 6.04))
    label(15.7, 6.93, "残差连接", 9, color=MUTED)
    arrow((20.29, 5.85), (21.0, 5.85))
    label(22.0, 5.85, "h_(l+1)\n[B,30,1024]", 11)
    label(11.85, 4.77, "序列长度始终为 30；这里没有图像编码器或 image cross-attention。eval 模式关闭 dropout。", 10, color=MUTED)

    label(0.3, 3.95, "③ 完成采样后的几何解码与实体构造（B = 1，逐样本执行）", 15, bold=True, ha="left")
    box(0.3, 1.9, 2.7, 1.45, "最终 latent", "Z_0 [30,32]\nCPU 随机噪声起步", TEAL)
    box(3.65, 1.9, 3.1, 1.45, "Latent 去重", "L1 均值阈值 0.01\n[30,32] → [N,32]\n面数由采样间接产生", ORANGE, size=10)
    box(7.4, 1.9, 3.8, 1.45, "冻结 VAE Decoder", "AutoEncoder_1119_light\n使用 diffusion checkpoint 内 AE\n面解码 / 几何去重 / 成对求交", PURPLE, size=10)
    box(11.85, 1.9, 4.45, 1.45, "原始几何与连接", "faces: [Nf,16,16,3]\nhalf-edges: [Nh,16,3]\nconnectivity: [Nh,3]", TEAL, size=10)
    box(16.95, 1.9, 3.15, 1.45, "OCC 后处理", "拟合 / 优化 / 裁剪 / 缝合\n每件 180 秒预算\n不读取 GT", ORANGE, size=10)
    box(20.75, 1.9, 2.65, 1.45, "STEP + 检查", "原生精度读取\nSolid + BRepCheck\n失败保留在分母", TEAL, size=10)
    for x, end in [(3.0, 3.65), (6.75, 7.4), (11.2, 11.85), (16.3, 16.95), (20.1, 20.75)]:
        arrow((x, 2.625), (end, 2.625))
    label(0.3, 1.2, "张量约定：B 为 batch；30 为 padding slots；32 为每面 latent；1024 为去噪主干 hidden size。N 与 Nf 可因去重不同。", 11,
          color=MUTED, ha="left")
    label(0.3, 0.67, "固定源码 c2660c00 · 本图描述已恢复的旧无条件模型；本轮每样本一次采样，不提供图片、GT 面数或 GT 拓扑。", 11,
          color=MUTED, ha="left")
    save_figure(fig, output / "architecture")


def finite_number(value) -> float | None:
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def flag(value) -> bool:
    return str(value).strip().lower() in {"true", "1"}


def read_report(root: Path, allow_partial: bool) -> tuple[dict, list[dict]]:
    """固定编号与种子同时核对，防止图库误读另一批采样。"""
    summary = json.loads((root / "report/summary.json").read_text())
    with (root / "report/per_sample.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    intended = int(summary["intended_count"])
    expected_ids = [f"uncond_{index:06d}" for index in range(1000)]
    if intended != 1000 or len(rows) != intended or [r["sample_id"] for r in rows] != expected_ids:
        raise ValueError("Report must contain all 1000 frozen IDs in order")
    for row in rows:
        expected_seed = int.from_bytes(hashlib.sha256(
            ("hola-uncond-20261018-v1:" + row["sample_id"]).encode()).digest()[:4], "little")
        if int(row["seed"]) != expected_seed:
            raise ValueError(f"Seed mismatch: {row['sample_id']}")
    if not summary["complete"] and not allow_partial:
        raise ValueError("The run is incomplete; use --allow-partial for a visibly marked progress figure")
    return summary, rows


def results(output: Path, summary: dict, rows: list[dict], fonts: dict) -> None:
    """计数分母固定1000；数值分布仅显示可计算值并列出 n。"""
    fig = plt.figure(figsize=(18, 12), dpi=175, facecolor="#f7f9fc")
    grid = fig.add_gridspec(3, 2, left=0.07, right=0.96, top=0.84, bottom=0.12,
                           height_ratios=[0.8, 1, 1], hspace=0.5, wspace=0.3)
    progress = "" if summary["complete"] else " · 运行中，非最终结果"
    fig.text(0.055, 0.94, "旧无条件生成 · 固定 1000 次采样" + progress,
             fontproperties=fonts["bold"], fontsize=22)
    fig.text(0.055, 0.895, "checkpoint 实际 1,000,000 步  |  FP32 · linear DDPM 1000 步 · 一次提议 / seed",
             fontproperties=fonts["body"], fontsize=12, color=MUTED)
    counts = summary["counts"]
    stages = [("推理成功", "inference_ok"), ("原始可后处理", "raw_usable"),
              ("STEP 产出", "step_exists"), ("原生合法 solid", "native_valid_solid")]
    ax = fig.add_subplot(grid[0, :])
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(0, 1); ax.axis("off")
    for index, (title, key) in enumerate(stages):
        count = int(counts[key])
        rate = summary["rates"][key]
        if count != int(rate["count"]) or int(rate["denominator"]) != 1000:
            raise ValueError(f"Summary count and rate mismatch: {key}")
        ax.text(index, 0.86, title, ha="center", fontsize=13, fontproperties=fonts["bold"])
        ax.text(index, 0.49, f"{count} / 1000", ha="center", fontsize=25,
                fontproperties=fonts["mono"], color="#217256" if key == "native_valid_solid" else INK)
        low, high = rate["wilson95"]
        ax.text(index, 0.13, f"{count / 10:.1f}%   Wilson 95% [{100*low:.1f}, {100*high:.1f}]%",
                ha="center", fontsize=10, fontproperties=fonts["body"], color=MUTED)

    def histogram(panel, key, title, xlabel, color, integer=False):
        ax = fig.add_subplot(panel)
        values = [finite_number(row.get(key)) for row in rows]
        values = np.asarray([value for value in values if value is not None])
        if len(values):
            bins = np.arange(values.min()-0.5, values.max()+1.5) if integer else 24
            ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.6)
            ax.text(0.98, 0.96, f"中位数 {np.median(values):.2f} · P95 {np.percentile(values,95):.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=10,
                    fontproperties=fonts["body"], color=MUTED)
        else:
            ax.text(0.5, 0.5, "无可计算记录", ha="center", transform=ax.transAxes)
        ax.set_title(f"{title}  ·  n={len(values)}", fontproperties=fonts["bold"], fontsize=13, loc="left")
        ax.set_xlabel(xlabel); ax.set_ylabel("样本数")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("white")

    histogram(grid[1, 0], "raw_num_faces", "原始解码面数", "面数（无面数过滤）", "#6d9db8", integer=True)
    histogram(grid[1, 1], "raw_num_directed_edges", "原始有向半边数", "有向半边数", "#9d8bb8")
    histogram(grid[2, 0], "inference_seconds", "完整 DDPM + AE 推理耗时", "秒 / 件（包含首件）", "#7eaf98")
    ax = fig.add_subplot(grid[2, 1])
    labels = ["合法实体", "检查未通过", "尚未判定"]
    values = [0, 0, 0]
    for row in rows:
        value = str(row.get("native_valid_solid", "")).strip().lower()
        # 未完成的检查保留 unknown，不能合并进检查未通过。
        group = 0 if value in {"true", "1"} else (1 if value in {"false", "0"} else 2)
        values[group] += 1
    ax.barh(labels, values, color=["#7eaf98", "#d2a36b", "#b3beca"])
    for y, value in enumerate(values):
        ax.text(value + 8, y, str(value), va="center", fontsize=10, fontproperties=fonts["mono"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1000)
    ax.set_title("原生精度实体判定 · 全部 1000 件", fontproperties=fonts["bold"], fontsize=13, loc="left")
    ax.set_xlabel("样本数"); ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.055, 0.057, "面边数量与图库描述生成复杂程度，不代表形状覆盖率或新颖性；本轮未计算 COV / MMD / LFD。",
             fontsize=10.5, fontproperties=fonts["body"], color=MUTED)
    fig.text(0.055, 0.028, "后处理：同一 legacy 策略 · drop=0 · 名义优化 200 次 · 单件 180 秒；合法 solid 使用 STEP 原生精度检查。",
             fontsize=10.5, fontproperties=fonts["body"], color=MUTED)
    save_figure(fig, output / "quantitative_summary")


def raw_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """仅供展示的规则网格三角化；不把未裁剪 face patch 当作 STEP。"""
    with np.load(path, allow_pickle=False) as data:
        faces = np.asarray(data["pred_face"], dtype=np.float64)
        curves = np.asarray(data["pred_edge"], dtype=np.float64)
    if faces.ndim != 4 or faces.shape[-1] != 3 or not np.isfinite(faces).all():
        raise ValueError("Invalid raw surface grid")
    vertices, triangles = [], []
    for face in faces:
        u = np.unique(np.r_[np.arange(0, face.shape[0], 2), face.shape[0]-1])
        v = np.unique(np.r_[np.arange(0, face.shape[1], 2), face.shape[1]-1])
        grid = face[np.ix_(u, v)]
        offset = len(vertices)
        vertices.extend(grid.reshape(-1, 3))
        for row in range(len(u)-1):
            for col in range(len(v)-1):
                a = offset + row*len(v) + col
                b, c, d = a+1, a+len(v), a+len(v)+1
                triangles.extend(((a, b, c), (b, d, c)))
    return np.asarray(vertices), np.asarray(triangles, dtype=np.int64), list(curves)


def step_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """按原生 STEP 读取；可视化网格变换使用每面的 TopLoc_Location。"""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone or reader.TransferRoots() == 0:
        raise ValueError("STEP read failed")
    shape = reader.OneShape()
    mesher = BRepMesh_IncrementalMesh(shape, 0.004, False, 0.25, False)
    if not mesher.IsDone():
        raise ValueError("STEP triangulation failed")
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    vertices, triangles = [], []
    while explorer.More():
        face = topods.Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is None:
            raise ValueError("Missing STEP face triangulation")
        offset = len(vertices)
        for index in range(1, triangulation.NbNodes()+1):
            p = triangulation.Node(index).Transformed(location.Transformation())
            vertices.append((p.X(), p.Y(), p.Z()))
        for index in range(1, triangulation.NbTriangles()+1):
            nodes = triangulation.Triangle(index).Get()
            triangles.append(tuple(offset+node-1 for node in nodes))
        explorer.Next()
    return np.asarray(vertices), np.asarray(triangles, dtype=np.int64), []


def draw_mesh(ax, vertices: np.ndarray, triangles: np.ndarray, curves: list[np.ndarray]) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if not len(vertices) or not len(triangles) or not np.isfinite(vertices).all():
        raise ValueError("Empty or nonfinite mesh")
    polygons = vertices[triangles]
    normals = np.cross(polygons[:, 1]-polygons[:, 0], polygons[:, 2]-polygons[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-12)
    light = np.array([0.3, -0.5, 0.8]); light /= np.linalg.norm(light)
    intensity = 0.52 + 0.48*np.abs(normals @ light)
    colors = intensity[:, None] * np.array([0.69, 0.80, 0.87])[None]
    collection = Poly3DCollection(polygons, facecolors=colors, edgecolors="none", linewidth=0,
                                  rasterized=True)
    ax.add_collection3d(collection)
    for curve in curves:
        if curve.ndim == 2 and curve.shape[1] == 3 and np.isfinite(curve).all():
            ax.plot(*curve.T, color="#344658", lw=0.45, alpha=0.65)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = (low+high)/2
    half = max(float((high-low).max())*0.54, 0.01)
    ax.set_xlim(center[0]-half, center[0]+half)
    ax.set_ylim(center[1]-half, center[1]+half)
    ax.set_zlim(center[2]-half, center[2]+half)
    ax.set_box_aspect((1, 1, 1)); ax.set_proj_type("ortho")
    ax.view_init(elev=24, azim=-55)


def gallery(root: Path, output: Path, rows: list[dict], fonts: dict, *, stage: str, complete: bool) -> None:
    """图库只取冻结清单前24项；缺失或失败保留原位置。"""
    first_rows = rows[:24]
    title = "原始面网格（未裁剪）与曲线" if stage == "raw" else "OCC 后处理 STEP"
    fig = plt.figure(figsize=(21, 14), facecolor="#f7f9fc")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.86, bottom=0.075, wspace=0.025, hspace=0.20)
    fig.text(0.025, 0.953, f"无条件生成 · {title} · 预定前 24 件", fontsize=23, fontproperties=fonts["bold"])
    fig.text(0.025, 0.915, "按固定 ID 顺序展示，不按成功或外观筛选；每件一次提议；展示角度固定、单件等比例居中。",
             fontsize=12, fontproperties=fonts["body"], color=MUTED)
    if not complete:
        fig.text(0.975, 0.915, "运行中 · 非最终结果", ha="right", fontsize=12,
                 fontproperties=fonts["bold"], color="#a26030")
    for index, row in enumerate(first_rows):
        ax = fig.add_subplot(4, 6, index+1, projection="3d")
        ax.set_facecolor("#f7f9fc"); ax.set_axis_off()
        sample_id = row["sample_id"]
        valid_value = str(row.get("native_valid_solid", "")).strip().lower()
        valid = flag(valid_value)
        if stage == "raw":
            subtitle = f"Nf={row.get('raw_num_faces') or '—'} · Nh={row.get('raw_num_directed_edges') or '—'}"
        elif valid:
            subtitle = "原生合法 solid"
        elif valid_value in {"false", "0"}:
            subtitle = "非合法 solid（已检查）"
        else:
            subtitle = f"有效性未判定 · {row.get('post_status', 'missing')}"
        ax.set_title(sample_id + "\n" + subtitle, fontsize=10.5, y=1.01,
                     fontproperties=fonts["body"], color="#237154" if stage == "post" and valid else INK)
        path = (root / "raw_predictions" / sample_id / "data.npz" if stage == "raw"
                else root / "post" / sample_id / "recon_brep.step")
        try:
            if not path.is_file():
                raise FileNotFoundError("未产出原始几何" if stage == "raw" else "未产出 STEP")
            vertices, triangles, curves = raw_mesh(path) if stage == "raw" else step_mesh(path)
            draw_mesh(ax, vertices, triangles, curves)
        except Exception as exc:
            # 绘图异常也不替换样本，并保留可审查的简短失败原因。
            if stage == "post":
                validity_note = "实体检查合法" if valid else ("实体检查：非合法 solid" if valid_value in {"false", "0"} else "实体检查未判定")
                failure_note = "预览三角化失败" if "triangulation" in str(exc) else "STEP 预览失败"
                message = f"{validity_note}；\n{failure_note}\n{str(exc)[:50]}"
            else:
                message = f"原始网格未能展示\n{type(exc).__name__}\n{str(exc)[:50]}"
            ax.text2D(0.5, 0.5, message,
                      transform=ax.transAxes, ha="center", va="center", fontsize=9,
                      fontproperties=fonts["body"], color="#a15b47")
    note = ("原始规则 face patch 尚未按边界裁剪，可能覆盖孔洞；深色线是预测半边，不能据此判定 STEP 有效性。"
            if stage == "raw" else "仅展示实际输出 STEP 的裁剪表面；是否有效取自独立 OCC 检查，显示成功不等于合法 solid。")
    fig.text(0.025, 0.035, note, fontsize=11, fontproperties=fonts["body"], color=MUTED)
    save_figure(fig, output / f"gallery_{stage}_first24")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--architecture", action="store_true", help="只生成架构图，不读取结果")
    parser.add_argument("--allow-partial", action="store_true", help="允许明确标为进行中的汇总图")
    parser.add_argument("--no-gallery", action="store_true", help="仅架构与汇总，不读取几何")
    args = parser.parse_args()
    output = args.output or args.root / "report"
    fonts = configure_fonts()
    architecture(output, fonts)
    if not args.architecture:
        summary, rows = read_report(args.root, args.allow_partial)
        results(output, summary, rows, fonts)
        if not args.no_gallery:
            for stage in ("raw", "post"):
                gallery(args.root, output, rows, fonts, stage=stage, complete=summary["complete"])
    print(output)


if __name__ == "__main__":
    main()
