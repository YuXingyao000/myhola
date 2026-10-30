"""生成讨论稿的静态流程图；不调用 LLM、Blender 或位置搜索工具。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
from matplotlib.path import Path as PlotPath


OUTPUT = Path(__file__).parent / "figures"
FONT = Path("/root/.cache/cad_scene_figures/SourceHanSansSC-Regular.otf")
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=str(FONT)).get_name(),
    "font.size": 14,
    "pdf.fonttype": 42,
    "svg.fonttype": "path",
    "axes.unicode_minus": False,
})

INK = "#202B34"
MUTED = "#53616C"
LINE = "#677782"
COLORS = {"F": ("#F1F3F5", "#47535E"),
          "A": ("#EAF1FB", "#2865A6"),
          "T": ("#E9F4EF", "#27735B")}


def text(ax, x, y, label, size=14, ha="center", color=INK, **kwargs):
    return ax.text(x, y, label, fontsize=size, ha=ha, va="center",
                   color=color, linespacing=1.4, **kwargs)


def node(ax, x, y, w, h, role, title, subtitle=""):
    fill, accent = COLORS[role]
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill,
                           edgecolor="#B8C2C9", linewidth=0.85, zorder=2))
    ax.add_patch(Rectangle((x, y), 0.34, 0.34, facecolor=accent,
                           edgecolor="none", zorder=3))
    text(ax, x + 0.17, y + 0.17, role, 11, color="white", zorder=4)
    title_y = y + h * (0.27 if "\n" in subtitle else 0.34)
    text(ax, x + w / 2, title_y, title, 14, zorder=4)
    if subtitle:
        text(ax, x + w / 2, y + h * 0.72, subtitle, 12, color=MUTED, zorder=4)


def arrow(ax, points, label=None, label_at=None, dashed=False, color=LINE):
    path = PlotPath(points, [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=13,
                                linewidth=1.2, color=color,
                                linestyle=(0, (4, 3)) if dashed else "-", zorder=1))
    if label:
        text(ax, *label_at, label, 12, color=MUTED,
             bbox={"facecolor": "white", "edgecolor": "none", "pad": 2}, zorder=5)


def save(fig, stem):
    OUTPUT.mkdir(exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"{stem}.{extension}", dpi=220,
                    facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def workflow():
    fig, ax = plt.subplots(figsize=(12.4, 12.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(14.6, 0)
    ax.axis("off")
    text(ax, 0.2, 0.35, "从一个场景寻找多个可拍摄位置", 22, ha="left")
    text(ax, 13.7, 0.35, "拟议工作流 · 尚未实现", 11, ha="right", color=MUTED)
    for x, role, label in [(0.25, "F", "外围程序：调度与判定"),
                           (4.8, "A", "Agent：提示词驱动的判断"),
                           (9.7, "T", "工具：测量与渲染")]:
        ax.add_patch(Rectangle((x, 0.91), 0.29, 0.29,
                               facecolor=COLORS[role][1], edgecolor="none"))
        text(ax, x + 0.145, 1.055, role, 11, color="white")
        text(ax, x + 0.42, 1.055, label, 12, ha="left")

    text(ax, 0.2, 1.62, "(a)   区域发现与空间筛选", 17, ha="left")
    w, h, y = 2.8, 1.13, 2.08
    xs = [0.3, 3.65, 7.0, 10.35]
    node(ax, xs[0], y, w, h, "F", "准备场景证据", "固定场景与规格\n全景及对象编号图")
    node(ax, xs[1], y, w, h, "A", "选择候选区域", "优先拍摄价值\n不要求已有空位")
    node(ax, xs[2], y, w, h, "T", "测量表面与占用", "当前空位、潜在落点\n邻近安置空间")
    node(ax, xs[3], y, w, h, "F", "建立候选集", "直接可试 / 需挪物\n不适用")
    for start, end in zip(xs, xs[1:]):
        arrow(ax, [(start + w, y + h / 2), (end, y + h / 2)])
    node(ax, 10.35, 3.87, 2.8, 0.97, "F", "记录不适用区域", "保留状态及原因")
    arrow(ax, [(12.55, 3.21), (12.55, 3.87)])
    arrow(ax, [(11.2, 3.21), (11.2, 3.52), (0.04, 3.52), (0.04, 6.66), (0.48, 6.66)],
          "直接可试或可局部整理 → 活跃候选集", (6.1, 3.52))

    ax.add_patch(Rectangle((0.2, 5.1), 13.1, 6.7, fill=False,
                           edgecolor="#AAB7C0", linewidth=0.95))
    text(ax, 0.5, 5.43, "(b)   有限整理与构图循环", 17, ha="left")
    text(ax, 12.95, 5.43, "A 按轮合批调用；T 逐候选执行", 12, ha="right", color=MUTED)
    row_y, row_h, row_w = 6.04, 1.24, 3.35
    x1, x2, x3 = 0.48, 5.05, 9.62
    node(ax, x1, row_y, row_w, row_h, "A", "提议整理与构图", "落点、机位、完整道具组移动")
    node(ax, x2, row_y, row_w, row_h, "T", "在副本执行并检查", "通过后渲染前后对照图")
    node(ax, x3, row_y, row_w, row_h, "A", "审阅前后图与报告", "主体重心、周边保留、自然感")
    arrow(ax, [(x1 + row_w, 6.66), (x2, 6.66)], "方案", (4.43, 6.42))
    arrow(ax, [(x2 + row_w, 6.66), (x3, 6.66)], "证据", (9.0, 6.42))

    node(ax, x2, 8.65, row_w, 1.32, "F", "外围判定", "数值 ∧ 视觉；核对剩余预算")
    arrow(ax, [(11.3, 7.28), (11.3, 9.31), (8.40, 9.31)])
    node(ax, x1, 8.65, row_w, 1.32, "F", "保留拍摄变体", "proxy_ready · 待人工审阅")
    arrow(ax, [(5.05, 9.31), (3.83, 9.31)], "均通过", (4.43, 9.06))
    arrow(ax, [(6.72, 8.65), (6.72, 7.87), (2.15, 7.87), (2.15, 7.28)],
          "需要调整且有预算：仅重做失败项", (4.4, 7.87), dashed=True)
    node(ax, x2, 10.68, row_w, 0.85, "F", "记录失败或未完成", "rejected / incomplete")
    arrow(ax, [(6.72, 9.97), (6.72, 10.68)], "不可修正 / 预算耗尽", (8.28, 10.32))

    node(ax, 3.4, 12.65, 6.5, 1.13, "F", "活跃集为空 → 汇总全部结果", "落点 · 相机 · 移动清单 · 前后图 · 全部尝试与终态")
    arrow(ax, [(2.15, 9.97), (2.15, 13.21), (3.4, 13.21)])
    arrow(ax, [(6.72, 11.53), (6.72, 12.65)])
    arrow(ax, [(13.15, 4.36), (13.65, 4.36), (13.65, 13.21), (9.9, 13.21)])
    text(ax, 0.25, 14.27,
         "最多 6 个区域、初稿 + 2 次调整；副本内少量挪物，基础场景保留。故障另记 incomplete。",
         12, ha="left", color=MUTED)
    save(fig, "placement-workflow")


def spatial_contract():
    fig, ax = plt.subplots(figsize=(12.4, 6.1))
    ax.set_xlim(0, 14)
    ax.set_ylim(6.75, 0)
    ax.axis("off")
    text(ax, 0.2, 0.35, "正方体代理检查什么", 21, ha="left")
    text(ax, 13.7, 0.35, "几何示意 · 非实际搜索结果", 11, ha="right", color=MUTED)

    text(ax, 0.2, 1.05, "(a)   俯视：先选区域，再确定落点", 16, ha="left")
    ax.add_patch(Rectangle((0.5, 1.75), 5.7, 3.55, facecolor="#F7F7F6",
                           edgecolor=LINE, linewidth=1.1))
    text(ax, 0.58, 1.52, "有限支撑面", 12, ha="left", color=MUTED)
    for x, y, w, h in [(0.85, 2.08, 1.1, 0.65), (5.05, 2.1, 0.8, 1.0),
                        (1.1, 4.3, 0.75, 0.57)]:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#B7BDC1", edgecolor="#849198", linewidth=0.8))
    text(ax, 5.45, 2.6, "道具", 12)
    ax.add_patch(Rectangle((2.33, 2.4), 2.52, 2.52, fill=False, edgecolor="#27735B",
                           linewidth=1.2, linestyle=(0, (4, 2))))
    ax.add_patch(Rectangle((2.54, 2.61), 2.1, 2.1, facecolor="#DCEBE5",
                           edgecolor="#27735B", linewidth=1.2))
    ax.plot(3.59, 3.66, marker="+", markersize=13, markeredgewidth=1.4, color=INK)
    text(ax, 3.59, 3.21, "20 × 20 cm", 14)
    text(ax, 3.59, 4.2, "代理占地", 12)
    text(ax, 3.57, 5.62, "每侧余量 2 cm → 至少 24 × 24 cm", 13)
    text(ax, 0.5, 6.23, "落点：支撑点 (x, y, z)，不是正方体中心", 12, ha="left", color=MUTED)

    text(ax, 7.1, 1.05, "(b)   侧视：检查相机到目标的有限空间", 16, ha="left")
    ax.plot([7.2, 13.3], [5.3, 5.3], color=LINE, linewidth=1.2)
    ax.add_patch(Rectangle((10.95, 3.65), 1.65, 1.65, facecolor="#DCEBE5",
                           edgecolor="#27735B", linewidth=1.2, zorder=4))
    camera = (7.65, 2.4)
    ax.add_patch(Polygon([camera, (12.6, 3.65), (12.6, 5.3), (10.95, 5.3)],
                         closed=True, facecolor="#EDF2F9", edgecolor="#668DB2",
                         linewidth=1.0, linestyle=(0, (4, 2)), zorder=1))
    ax.plot(*camera, marker="o", color="#2865A6", markersize=7, zorder=5)
    text(ax, 7.65, 1.98, "相机", 13)
    text(ax, 9.0, 3.12, "保护区域", 12)
    ax.add_patch(Rectangle((9.15, 3.55), 0.38, 1.75, facecolor="#B7BDC1",
                           edgecolor="#849198", linewidth=0.8, zorder=3))
    text(ax, 9.22, 5.72, "前景阻挡 → 拒绝", 12)
    ax.add_patch(Rectangle((13.1, 2.25), 0.16, 3.05, facecolor="#B7BDC1",
                           edgecolor="#849198", linewidth=0.8))
    text(ax, 13.18, 1.98, "背景墙", 12)
    text(ax, 11.77, 4.3, "代理", 13, zorder=5)
    ax.plot([11.77, 11.77], [5.3, 4.6], color=INK, linewidth=1.0, zorder=5)
    ax.plot(11.77, 5.3, marker="+", color=INK, markersize=10, zorder=5)
    text(ax, 11.77, 5.72, "底面接触", 12)
    text(ax, 7.1, 6.23, "另查：完整入镜、目标占比、相机不在实体内", 12, ha="left", color=MUTED)
    save(fig, "placement-spatial-contract")


def rearrangement():
    """用相同的六组物件示意局部腾位；不是场景渲染或检测结果。"""
    fig, ax = plt.subplots(figsize=(12.4, 6.1))
    ax.set_xlim(0, 14)
    ax.set_ylim(6.5, 0)
    ax.axis("off")
    text(ax, 0.2, 0.35, "腾出拍摄位置，同时保留周围物件", 21, ha="left")
    text(ax, 13.7, 0.35, "俯视示意 · 非实际运行结果", 11, ha="right", color=MUTED)
    text(ax, 0.3, 1.15, "(a)   好位置被道具占用", 16, ha="left")
    text(ax, 7.5, 1.15, "(b)   少量挪动后，主体留在中央", 16, ha="left")
    warm, warm_edge = "#F4DEBC", "#A4763B"

    for left, moved in [(0.3, False), (7.5, True)]:
        top = 1.7
        ax.add_patch(Rectangle((left, top), 5.8, 3.8, facecolor="#F7F7F6",
                               edgecolor=LINE, linewidth=1.0))
        # The same four surrounding groups remain in both panels.
        ax.add_patch(Rectangle((left + 0.45, top + 0.5), 1.0, 0.65,
                               facecolor="#E1E5E7", edgecolor="#849198", linewidth=0.8))
        for i in range(3):
            ax.add_patch(Circle((left + 0.68 + i * 0.26, top + 0.83), 0.075,
                                facecolor="#9BA7AF", edgecolor="none"))
        ax.add_patch(Rectangle((left + 4.5, top + 0.35), 0.7, 0.6,
                               facecolor="#CBD2D6", edgecolor="#849198", linewidth=0.8))
        ax.add_patch(Rectangle((left + 0.5, top + 2.8), 1.1, 0.28, angle=7,
                               facecolor="#CBD2D6", edgecolor="#849198", linewidth=0.8))
        ax.add_patch(Circle((left + 4.8, top + 2.95), 0.22,
                            facecolor="#CBD2D6", edgecolor="#849198", linewidth=0.8))
        ax.add_patch(Rectangle((left + 2.3, top + 1.15), 1.6, 1.6,
                               fill=False, edgecolor="#27735B", linewidth=1.1,
                               linestyle=(0, (4, 3))))

        positions = [(1.6, 0.42, 1.0, 0.5, -10), (4.25, 2.05, 0.7, 0.5, 16)] if moved else [
            (2.1, 1.2, 1.0, 0.5, 0), (2.9, 2.15, 0.7, 0.5, 0)]
        for index, (x, y, width, height, yaw) in enumerate(positions, 1):
            ax.add_patch(Rectangle((left + x, top + y), width, height, angle=yaw,
                                   rotation_point="center", facecolor=warm,
                                   edgecolor=warm_edge, linewidth=1.1, zorder=4))
            text(ax, left + x + width / 2, top + y + height / 2, str(index), 12, zorder=5)
        if moved:
            arrow(ax, [(left + 2.6, top + 1.3), (left + 2.1, top + 0.96)],
                  dashed=True, color=warm_edge)
            arrow(ax, [(left + 3.65, top + 2.4), (left + 4.16, top + 2.34)],
                  dashed=True, color=warm_edge)
            ax.add_patch(Rectangle((left + 2.6, top + 1.45), 1.0, 1.0,
                                   facecolor="#DCEBE5", edgecolor="#27735B", linewidth=1.2))
            text(ax, left + 3.1, top + 1.95, "零件\n占位体", 12)
            text(ax, left + 2.9, 5.87, "只移动组 1、2；其余保持，周围仍可见", 12, color=MUTED)
        else:
            text(ax, left + 2.9, 5.87, "虚线：目标所需空间，不是整张桌面", 12, color=MUTED)
    arrow(ax, [(6.4, 3.45), (7.16, 3.45)])
    save(fig, "placement-rearrangement")


if __name__ == "__main__":
    workflow()
    spatial_contract()
    rearrangement()
    print(OUTPUT)
