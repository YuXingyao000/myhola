"""Export source-grounded post-processing explanations for the weekly report."""
from pathlib import Path
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
FONT = Path("/root/.cache/hola-browser-libs/fonts/NotoSansCJKsc-Regular.otf")
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT).get_name(),
                     "font.size": 12, "axes.unicode_minus": False, "svg.fonttype": "path"})
INK, MUTED = "#17324d", "#60758a"
GREEN, RED, AMBER, BLUE = "#118477", "#b9584e", "#b98025", "#327bb4"


def box(ax, x, y, w, h, text, *, color=BLUE, fill="#f2f6fa", size=12, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=.10",
                              fc=fill, ec=color, lw=1.3))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=size,
            color=INK, linespacing=1.5, weight=weight)


def arrow(ax, start, end, *, color=MUTED, style="-", lw=1.5):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw, "linestyle": style})


def save(fig, name):
    for ext in ("png", "svg"):
        path = ROOT / f"{name}.{ext}"
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
        if ext == "svg":
            svg = re.sub(r'\bd="([^"]*)"',
                         lambda m: 'd="' + " ".join(m.group(1).split()) + '"', path.read_text())
            path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    plt.close(fig)


def recovery_flow():
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, 16), ylim=(0, 10)); ax.axis("off")
    ax.text(.5, 9.55, "保留原有成功结果，失败时再从原始预测恢复", fontsize=25, weight="bold", color=INK)
    ax.text(.5, 9.10, "固定 VAE；路径按 0 → 3 顺序执行，第一条通过就停止。每条新路径重新读取原始预测。", fontsize=13, color=MUTED)
    centers = [8.15, 6.20, 4.25, 2.30]
    box(ax, .45, 7.58, 2.35, 1.14, "原始 data.npz\n面点 · 半边点 · 连接", color=MUTED, fill="#f5f7fa", size=12)
    ax.plot([1.6, 1.6], [1.72, 7.55], color=MUTED, lw=1.3, ls="--")
    ax.text(.48, 5.8, "重新读取\n不接着修改\n上条路径的结果", fontsize=11, color=MUTED, linespacing=1.7)
    recipes = [
        ("路径 0   原后处理", "原几何优化 → 原拟合 → 原裁剪与缝合", "原生重读\n合法 solid？", "V：3076    T：2111"),
        ("路径 1   优化后使用新构造", "原几何优化 → 原拟合 → 新构造", "原生合法＋\n面／环计数检查", "V：26       T：15"),
        ("路径 2   跳过几何优化", "直接拟合原预测 → 新构造", "原生合法＋\n面／环计数检查", "V：44       T：28"),
        ("路径 3   换用修正评分的拟合", "原几何优化 → 修正拟合评分 → 新构造", "原生合法＋\n面／环计数检查", "V：1         T：1"),
    ]
    for idx, (y, recipe) in enumerate(zip(centers, recipes)):
        title, body, gate, counts = recipe
        ax.add_patch(FancyBboxPatch((3.5, y-.65), 8.25, 1.30, boxstyle="round,pad=.025,rounding_size=.10", fc="white", ec="#d5dfe8", lw=1.2))
        ax.text(3.75, y+.27, title, fontsize=13, weight="bold", color=INK)
        ax.text(3.75, y-.21, body, fontsize=11, color=MUTED)
        box(ax, 9.03, y-.49, 2.5, .98, gate, color=GREEN, fill="#edf8f5", size=11)
        arrow(ax, (8.55, y), (9.0, y), color=GREEN)
        if idx == 0:
            arrow(ax, (2.82, y), (3.47, y), style="--")
        else:
            arrow(ax, (1.6, y), (3.47, y), style="--")
        box(ax, 12.45, y-.54, 3.05, 1.08, f"通过：接受并结束\n{counts}", color=GREEN, fill="#edf8f5", size=12)
        arrow(ax, (11.77, y), (12.41, y), color=GREEN)
        if idx < 3:
            arrow(ax, (7.1, y-.68), (7.1, centers[idx+1]+.68), color=RED)
            ax.text(7.33, (y+centers[idx+1])/2, "未通过且预算未耗尽，进入下一条", fontsize=10.5, color=RED, va="center")
    arrow(ax, (7.1, 1.61), (7.1, 1.05), color=RED)
    ax.text(7.35, 1.30, "全部未通过：V 348 件／T 262 件，保留失败记录", fontsize=11, color=RED, va="center")
    ax.text(.5, .70, "新构造＝独立候选＋不跳过已组好的普通环＋所有面均须裁出＋保存重读后才接受。", fontsize=11.5, color=INK)
    ax.text(.5, .31, "V＝validation，T＝test；共用每件 180 秒总预算，耗尽即停止。原成功路径不追加完整性要求；计数通过也不保证拓扑正确。", fontsize=10.5, color=MUTED)
    save(fig, "recovery_flow")


def repair_motivation():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.3))
    fig.suptitle("四处代码行为，为什么值得修", x=.04, ha="left", fontsize=25, color=INK, weight="bold")
    for ax in axes.ravel():
        ax.set(xlim=(0, 10), ylim=(0, 6)); ax.axis("off")
    a = axes[0, 0]
    a.text(.1, 5.65, "① 首个 solid 保存／重读未通过，旧循环已结束", fontsize=14, color=INK, weight="bold")
    a.text(.1, 5.08, "旧：保存检查在候选循环外", fontsize=12, color=RED)
    box(a, .1, 3.85, 2.45, .83, "内存 solid 通过\n立即 break", color=RED, size=11)
    box(a, 3.05, 3.85, 2.5, .83, "保存 STEP\n读回检查失败", color=RED, size=11)
    box(a, 6.10, 3.85, 3.4, .83, "直接返回失败\n后续候选失去机会", color=RED, size=11)
    arrow(a, (2.6, 4.26), (3.0, 4.26), color=RED); arrow(a, (5.6, 4.26), (6.05, 4.26), color=RED)
    a.text(.1, 3.16, "新：保存和原生重读都放进候选循环", fontsize=12, color=GREEN)
    box(a, .1, 1.95, 2.65, .83, "候选 A\n保存／重读未通过", color=GREEN, size=11)
    box(a, 3.3, 1.95, 2.8, .83, "继续候选 B\n保存／重读通过", color=GREEN, size=11)
    box(a, 6.65, 1.95, 2.85, .83, "正式接受\n停止尝试", color=GREEN, size=11)
    arrow(a, (2.8, 2.36), (3.25, 2.36), color=GREEN); arrow(a, (6.15, 2.36), (6.6, 2.36), color=GREEN)
    a.text(.1, .94, "旧代码本来就会重试“没有形成 solid”的情况。\n新增的是：保存、重读或计数验收失败后仍可继续。", fontsize=11, linespacing=1.7, color=MUTED)
    a.text(.1, .14, "construct_brep.py  ·  _try_save_complete_solid", fontsize=10, color=MUTED)

    a = axes[0, 1]
    a.text(.1, 5.65, "② 前一次修复，不应污染下一候选", fontsize=15, color=INK, weight="bold")
    a.text(.1, 5.08, "旧：循环复用可被修改的 OCC 对象", fontsize=12, color=RED)
    box(a, .1, 3.85, 2.3, .83, "同一份对象", color=RED, size=11)
    box(a, 2.95, 3.85, 2.5, .83, "尝试 A\n修几何／改拓扑", color=RED, size=11)
    box(a, 6.05, 3.85, 3.45, .83, "尝试 B\n可能继承前次修改", color=RED, size=11)
    arrow(a, (2.45, 4.26), (2.9, 4.26), color=RED); arrow(a, (5.5, 4.26), (6.0, 4.26), color=RED)
    a.text(.1, 3.16, "新：每次从未修改的输入复制候选", fontsize=12, color=GREEN)
    box(a, .1, 1.60, 2.3, 1.22, "保留输入", color=GREEN, size=11)
    box(a, 3.3, 2.25, 2.75, .67, "OCC 副本 A", color=GREEN, size=11)
    box(a, 3.3, 1.28, 2.75, .67, "OCC 副本 B", color=GREEN, size=11)
    arrow(a, (2.45, 2.5), (3.25, 2.58), color=GREEN); arrow(a, (2.45, 1.9), (3.25, 1.61), color=GREEN)
    a.text(6.4, 2.28, "各自修复\n互不继承修改", fontsize=12, color=GREEN, va="center", linespacing=1.7)
    a.text(.1, .59, "保留单次复制内已有的共享关系；\n本轮没有建立新的全局共享 edge／pcurve。", fontsize=10.5, linespacing=1.6, color=MUTED)
    a.text(.1, -.02, "occ_topology.py  ·  copy_occ_shapes", fontsize=10, color=MUTED)

    a = axes[1, 0]
    a.text(.1, 5.65, "③ 失败的环不能直接当作不存在", fontsize=15, color=INK, weight="bold")
    a.text(.1, 5.08, "旧：未闭合 wire 被 continue 跳过", fontsize=12, color=RED)
    for x, hole in ((.3, True), (3.8, False)):
        a.add_patch(Rectangle((x, 3.07), 2.35, 1.48, fc="#e5f2ef", ec=GREEN, lw=1.8))
        if hole:
            a.add_patch(Circle((x+1.17, 3.81), .43, fc="white", ec=RED, lw=1.8, ls="--"))
    arrow(a, (2.8, 3.81), (3.65, 3.81), color=RED)
    a.text(6.55, 3.85, "仍可能做出面\n但边界变少", fontsize=12, color=RED, va="center", linespacing=1.7)
    a.text(.3, 2.68, "有失败内环的候选（示意）", fontsize=10.5, color=MUTED)
    a.text(.1, 2.10, "新：已组好的普通环必须全接纳，所有面必须裁出", fontsize=12, color=GREEN)
    a.text(.1, 1.15, "环被跳过或边界数量减少 → 拒绝候选，继续尝试。\n圆柱完整自然 loop 的既有特殊替换保留并记录。", fontsize=11, color=INK, linespacing=1.7)
    a.text(.1, .49, "若组环时两环已经变成一环，数量检查仍可能放过。", fontsize=10.5, color=MUTED)
    a.text(.1, .07, "occ_topology.py  ·  require_all_wires", fontsize=10, color=MUTED)

    a = axes[1, 1]
    a.text(.1, 5.65, "④ 旧 ori_shape 已经经过几何优化", fontsize=15, color=INK, weight="bold")
    a.text(.1, 5.08, "旧：备份在优化之后，失败不等于撤销优化", fontsize=12, color=RED)
    box(a, .1, 3.85, 2.1, .83, "原始预测", color=RED, size=11)
    box(a, 2.75, 3.85, 2.3, .83, "几何优化", color=RED, size=11)
    box(a, 5.65, 3.85, 3.85, .83, "ori_shape = deepcopy(shape)\n此时已经是优化后的点", color=RED, size=10.5)
    arrow(a, (2.25, 4.26), (2.7, 4.26), color=RED); arrow(a, (5.1, 4.26), (5.6, 4.26), color=RED)
    a.text(.1, 3.16, "新：必要时重新读 raw，允许完全跳过优化", fontsize=12, color=GREEN)
    box(a, .1, 1.95, 2.45, .83, "重新读 data.npz", color=GREEN, size=11)
    box(a, 3.15, 1.95, 2.7, .83, "直接拟合原预测", color=GREEN, size=11)
    box(a, 6.45, 1.95, 3.05, .83, "新构造＋保存验收", color=GREEN, size=11)
    arrow(a, (2.6, 2.36), (3.1, 2.36), color=GREEN); arrow(a, (5.9, 2.36), (6.4, 2.36), color=GREEN)
    a.text(.1, .95, "这条组合路径在 validation 救回 44 件。\n支持保留备用路径，不代表几何优化总体有害。", fontsize=11, color=MUTED, linespacing=1.7)
    a.text(.1, .14, "construct_brep.py  ·  _run_complete_post_path", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(.01, .02, .99, .94), h_pad=2, w_pad=2)
    save(fig, "repair_motivation")


def fitting_checks():
    fig = plt.figure(figsize=(15, 8.8))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, 15), ylim=(0, 8.8)); ax.axis("off")
    ax.text(.5, 8.3, "拟合评分修正：检查哪些点，误差如何计算", fontsize=24, color=INK, weight="bold")
    ax.text(.5, 7.82, "面始终用全部 16×16 点拟合；下面只画评分时被检查的位置，不是 VAE 采样方式的改变。", fontsize=12, color=MUTED)
    for x0, label, inds, color in ((.7, "旧：16 个检查点", [0, 4, 8, 12], RED),
                                   (4.8, "新：25 个检查点", [0, 4, 8, 12, 15], GREEN)):
        ax.text(x0, 7.17, label, fontsize=14, color=color, weight="bold")
        for i in range(16):
            for j in range(16):
                checked = i in inds and j in inds
                ax.scatter(x0+i*.19, 6.7-j*.19, s=24 if checked else 6,
                           color=color if checked else "#d6dfe7", zorder=3)
        ax.add_patch(Rectangle((x0-.10, 3.74), 3.05, 3.05, fc="none", ec="#c3cfda", lw=.8))
        ax.text(x0, 3.32, "索引：" + ", ".join(map(str, inds)), fontsize=10.5, color=MUTED)
    ax.text(9.0, 7.12, r"$d_i$：输入点到候选形状的最短距离", fontsize=12, color=INK)
    box(ax, 9.0, 5.68, 5.35, .95, r"旧评分：$\sqrt{\mathrm{mean}(d_i)} + \max_i d_i$", color=RED, fill="#fdf3f0", size=14)
    box(ax, 9.0, 4.22, 5.35, .95, r"新评分：$\sqrt{\mathrm{mean}(d_i^2)} + \max_i d_i$", color=GREEN, fill="#edf8f5", size=14)
    ax.text(9.0, 3.35, "旧公式不是真正 RMS；\n新公式两项都使用输入 XYZ 的长度单位。", fontsize=12, color=MUTED, linespacing=1.7)
    ax.text(.7, 2.59, "曲线评分：旧检查索引 0, 2, …, 14；新检查追加索引 15（第 16 个点），检查点 8 → 9 个。", fontsize=12, color=INK)
    ax.text(.7, 2.05, r"提前停止：当前候选 $\max_i d_i \leq 10^{-3}$ 时停止继续尝试；最后仍在已评过分的候选中选最低分。", fontsize=12, color=INK)
    ax.text(.7, 1.35, "1e−3 是停止条件，不是最终输出的精度承诺；评分后还有周期转换。\n边与面仍各自拟合，未检查整条曲线是否同时贴住两个所属面。", fontsize=11.5, color=MUTED, linespacing=1.6)
    ax.text(.7, .40, "Validation：单独用新评分／检查策略，合法数 3076 → 3073；在固定恢复流程的末路径，与新构造组合多救回 1 件。", fontsize=11, color=AMBER)
    save(fig, "fitting_checks")


if __name__ == "__main__":
    recovery_flow()
    repair_motivation()
    fitting_checks()
