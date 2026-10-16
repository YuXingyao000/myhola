"""Render four presentation figures from the audited architecture, without inference."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "story_figures"
FONT = Path("/root/.cache/hola-browser-libs/fonts/NotoSansCJKsc-Regular.otf")
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({
    "font.family": [font_manager.FontProperties(fname=str(FONT)).get_name(), "DejaVu Sans"],
    "svg.fonttype": "path", "pdf.fonttype": 42, "axes.unicode_minus": False,
})

INK = "#23394B"
MUTED = "#64798B"
BLUE = "#286694"
BLUE_BG = "#EEF5FC"
ORANGE = "#BA6B1B"
ORANGE_BG = "#FFF4E5"
GREEN = "#23796A"
GREEN_BG = "#EAF7F2"
RED = "#C55850"
RED_BG = "#FDF0EE"
GRAY_BG = "#F4F6F8"


def text(ax, x, y, label, size=19, color=INK, ha="center", weight="normal"):
    return ax.text(x, y, label, fontsize=size, color=color, ha=ha, va="center",
                   fontweight=weight, linespacing=1.45, zorder=8)


def box(ax, x, y, w, h, label="", color=BLUE, fill=BLUE_BG, size=19):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.02,rounding_size=0.13",
                              facecolor=fill, edgecolor=color, linewidth=1.5, zorder=3))
    if label:
        text(ax, x+w/2, y+h/2, label, size)


def arrow(ax, points, color=BLUE, dashed=False, lw=2):
    if len(points) > 2:
        a = np.asarray(points[:-1])
        ax.plot(a[:, 0], a[:, 1], color=color, lw=lw,
                linestyle="--" if dashed else "-", zorder=5)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                mutation_scale=17, linewidth=lw, color=color,
                                linestyle="--" if dashed else "-", zorder=5))


def canvas(number, title, subtitle):
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 16), ylim=(0, 9))
    ax.axis("off")
    text(ax, .65, 8.35, f"{number:02d}  {title}", 28, ha="left", weight="bold")
    text(ax, .67, 7.75, subtitle, 16, MUTED, ha="left")
    return fig, ax


def footer(ax, label, fill=GRAY_BG, color=INK, y=.42, size=18):
    box(ax, .65, y, 14.7, .66, color=fill, fill=fill)
    text(ax, 8, y+.33, label, size, color)


def geometry(ax, origin, scale=1, corrected=False, show_gap=True):
    """Illustrative projected planes and a curve; no experimental CAD is depicted."""
    origin = np.asarray(origin)

    def p(x, y, z):
        return origin + scale*np.array([x-.38*y+.12*z, .10*x+.38*y+z])

    face_a = [p(x, y, 0) for x, y in [(0, -.95), (3.5, -.95), (3.5, .95), (0, .95)]]
    face_b = [p(x, 0, z) for x, z in [(0, 0), (3.5, 0), (3.5, 1.55), (0, 1.55)]]
    ax.add_patch(Polygon(face_b, facecolor="#D8EEE7", edgecolor=GREEN, lw=1.5, zorder=4))
    ax.add_patch(Polygon(face_a, facecolor="#DAE9F6", edgecolor=BLUE, lw=1.5, alpha=.92, zorder=4))
    shared = np.array([p(0, 0, 0), p(3.5, 0, 0)])
    ax.plot(shared[:, 0], shared[:, 1], "--", color=MUTED, lw=1.8, zorder=5)
    xs = np.linspace(.08, 3.42, 80)
    ys = (.025 if corrected else .30) + (.012 if corrected else .10)*np.sin(xs*2)
    zs = (.055 if corrected else .48) + (.018 if corrected else .12)*np.sin(xs*1.5+.4)
    curve = np.array([p(x, y, z) for x, y, z in zip(xs, ys, zs)])
    ax.plot(curve[:, 0], curve[:, 1], color=GREEN if corrected else RED, lw=3.3, zorder=6)
    for i in [4, 19, 34, 49, 64, 77]:
        ax.scatter(*curve[i], s=20, color=GREEN if corrected else RED, zorder=7)
    if show_gap:
        for i in [18, 45, 68]:
            q = p(xs[i], ys[i], 0)
            arrow(ax, [tuple(curve[i]), tuple(q)], RED, dashed=True, lw=1.3)
    text(ax, *p(.50, -.60, 0), "面 A", 15, BLUE)
    text(ax, *p(2.65, 0, 1.18), "面 B", 16, GREEN)
    return p


def motivation():
    fig, ax = canvas(1, "为什么从 Intersection 入手？", "从高频失败现象，追到边的生成路径。")
    box(ax, .65, 1.55, 6.65, 5.5, color="#D7E1EA", fill="white")
    text(ax, 3.95, 6.52, "形状接近，边却贴不住面", 22, weight="bold")
    text(ax, 3.95, 6.03, "红线：预测边 · 虚线：两面交线（示意）", 13, MUTED)
    geometry(ax, (1.65, 3.55), scale=1.22)
    text(ax, 4.0, 2.60, "预测边存在偏差 → 裁剪 / 缝合失败", 18, RED)
    text(ax, 4.0, 2.05, "此前失败统计与观察：边几何精度是主要问题", 14.5, MUTED)
    box(ax, 8.3, 5.7, 6.15, 1.0, "两个面的特征", size=22)
    box(ax, 8.3, 3.8, 6.15, 1.15, "Intersection\n成对注意力 → 边特征", size=21)
    box(ax, 8.3, 1.85, 6.15, 1.15, "曲线 Decoder → 边的几何", size=21)
    arrow(ax, [(11.38, 5.68), (11.38, 4.99)])
    arrow(ax, [(11.38, 3.78), (11.38, 3.04)])
    footer(ax, "改进切入点：让边的生成过程，更直接地知道它与所属面的几何关系。")
    return fig


def original():
    fig, ax = canvas(2, "原始结构：先交换面特征，再预测边", "关键路径示意；展示 i → j，反方向同样计算。")
    box(ax, .7, 5.65, 2.35, .95, "面特征 Fᵢ", size=22)
    box(ax, .7, 3.95, 2.35, .95, "面特征 Fⱼ", size=22)
    box(ax, 4.15, 4.45, 3.65, 1.65, "Pair-wise\ncross-attention\n原 Transformer block", size=19)
    arrow(ax, [(3.08, 6.12), (3.55, 6.12), (3.55, 5.65), (4.10, 5.65)])
    arrow(ax, [(3.08, 4.42), (3.55, 4.42), (3.55, 4.88), (4.10, 4.88)])
    text(ax, 3.60, 6.48, "Q", 16, BLUE)
    text(ax, 3.58, 4.02, "K / V", 16, BLUE)
    box(ax, 8.5, 4.8, 2.25, .95, "边特征 Eᵢⱼ", size=21)
    arrow(ax, [(7.83, 5.27), (8.46, 5.27)])
    box(ax, 11.65, 5.95, 3.6, 1.05, "Classifier\n这条边是否存在？", size=20)
    box(ax, 11.65, 3.55, 3.6, 1.05, "曲线 Decoder\n这条边长什么样？", size=20)
    arrow(ax, [(10.78, 5.3), (11.15, 5.3), (11.15, 6.47), (11.61, 6.47)])
    arrow(ax, [(10.78, 5.3), (11.15, 5.3), (11.15, 4.07), (11.61, 4.07)])
    arrow(ax, [(13.45, 5.92), (13.45, 4.64)])
    text(ax, 14.05, 5.27, "筛选", 15, MUTED)
    box(ax, 4.15, 1.85, 3.65, .95, "面 Decoder", size=21)
    box(ax, 8.5, 1.85, 2.25, .95, "面几何", size=21)
    arrow(ax, [(.67, 6.12), (.35, 6.12), (.35, 2.32), (4.11, 2.32)])
    ax.plot([.67, .35], [4.42, 4.42], color=BLUE, lw=2, zorder=5)
    text(ax, 2.40, 2.66, "各面特征", 14.5, MUTED)
    arrow(ax, [(7.83, 2.32), (8.46, 2.32)])
    text(ax, 13.45, 2.20, "边几何", 21, BLUE)
    arrow(ax, [(13.45, 3.52), (13.45, 2.63)])
    footer(ax, "原结构的缺口：已经生成的面和边，不会再反馈给彼此，检查是否对得上。")
    return fig


def addition():
    fig, ax = canvas(3, "在原结构上，增加两处信息传递", "蓝色：沿用原结构    橙色：新增几何反馈    绿色：新增最终曲线条件调制")
    box(ax, .65, 2.1, 4.0, 4.95, color=BLUE, fill=BLUE_BG)
    text(ax, 2.65, 6.55, "原网络保留", 23, BLUE, weight="bold")
    box(ax, 1.05, 4.75, 3.2, 1.0, "面特征 F\n原 Intersection → E", fill="white", size=19)
    box(ax, 1.05, 2.7, 3.2, 1.25, "原几何头 + Classifier\n初稿面、边、存在概率", fill="white", size=18)
    arrow(ax, [(2.65, 4.72), (2.65, 3.99)])
    box(ax, 5.5, 2.1, 4.65, 4.95, color=ORANGE, fill=ORANGE_BG)
    text(ax, 7.83, 6.55, "新增①  Refinement", 23, ORANGE, weight="bold")
    box(ax, 5.93, 4.75, 3.8, 1.0, "读取：边到两个面的偏差", ORANGE, "white", size=18)
    box(ax, 5.93, 2.7, 3.8, 1.25, "小网络联合更新\n面特征 F + 边特征 E", ORANGE, "white", size=20)
    arrow(ax, [(7.83, 4.72), (7.83, 3.99)], ORANGE)
    arrow(ax, [(4.68, 4.2), (5.46, 4.2)], ORANGE)
    text(ax, 5.05, 3.75, "几何\n+ 特征", 13.5, MUTED)
    box(ax, 11.0, 5.05, 4.35, 1.6, "新增②  两面条件调制\n读取边特征与两所属面特征", GREEN, GREEN_BG, size=19)
    box(ax, 11.0, 2.7, 4.35, 1.25, "最终解码\n复用原几何头 + Classifier", size=19)
    arrow(ax, [(10.18, 3.32), (10.58, 3.32), (10.96, 3.32)])
    arrow(ax, [(10.58, 3.32), (10.58, 5.85), (10.96, 5.85)], GREEN)
    arrow(ax, [(13.18, 5.02), (13.18, 3.99)], GREEN)
    text(ax, 14.05, 4.48, "只调制\n最终曲线头", 15, GREEN)
    text(ax, 13.18, 1.70, "最终面、边、存在性", 21, BLUE)
    arrow(ax, [(13.18, 2.67), (13.18, 2.08)])
    footer(ax, "先生成初稿，再反馈一次；原 Intersection 不重跑，几何头重复使用。")
    return fig


def mechanism():
    fig, ax = canvas(4, "为什么这样做可能有效？", "补上两种信息：哪里没对齐，以及这条边应该同时服务哪两张面。")
    box(ax, .65, 3.60, 6.45, 3.52, color="#D7E1EA", fill="white")
    box(ax, 8.9, 3.60, 6.45, 3.52, color="#D7E1EA", fill="white")
    text(ax, 3.88, 6.67, "初次预测", 22, RED, weight="bold")
    text(ax, 12.13, 6.67, "希望学到的结果", 22, GREEN, weight="bold")
    geometry(ax, (1.7, 4.36), scale=1.02, show_gap=True)
    geometry(ax, (9.95, 4.36), scale=1.02, corrected=True, show_gap=False)
    arrow(ax, [(7.35, 5.22), (8.65, 5.22)], ORANGE, lw=2.7)
    text(ax, 8, 5.97, "反馈\n再解码", 17, ORANGE)
    text(ax, 3.9, 3.89, "看起来相邻，实际仍有间隙", 16, RED)
    text(ax, 12.15, 3.89, "边与两面的几何更一致", 16, GREEN)
    box(ax, .65, 1.76, 7.05, 1.38, color=ORANGE, fill=ORANGE_BG)
    text(ax, 4.18, 2.63, "① 几何反馈：把偏差交给网络", 20, ORANGE, weight="bold")
    text(ax, 4.18, 2.08, "投影只读出位置差，再学习更新面 / 边特征。", 16)
    box(ax, 8.3, 1.76, 7.05, 1.38, color=GREEN, fill=GREEN_BG)
    text(ax, 11.83, 2.63, "② 条件调制：输出前再参考两张面", 20, GREEN, weight="bold")
    text(ax, 11.83, 2.08, "曲线头的形状与位置输出，都接收所属面信息。", 16)
    footer(ax, "同一原后处理下：Validation  85.67% → 87.73%     Test  84.90% → 87.46%", size=18, y=.69)
    text(ax, 8, .29, "几何图是机制示意，并非真实样本或直接投影修复；数字评价两项模型改进整体，不拆分单项贡献。", 12.5, MUTED)
    return fig


def main():
    OUT.mkdir(exist_ok=True)
    entries = [("01_motivation", motivation), ("02_original", original),
               ("03_additions", addition), ("04_mechanism", mechanism)]
    with PdfPages(OUT / "intersection_story.pdf") as pdf:
        for name, render in entries:
            fig = render()
            fig.savefig(OUT / f"{name}.png", dpi=160, facecolor="white")
            fig.savefig(OUT / f"{name}.svg", facecolor="white")
            svg = OUT / f"{name}.svg"
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            print(OUT / f"{name}.png")
    print(OUT / "intersection_story.pdf")


if __name__ == "__main__":
    main()
