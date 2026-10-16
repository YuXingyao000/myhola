"""Render the checkpoint-audited decoder comparison as PNG and vector SVG."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
FONT = Path("/root/.cache/hola-browser-libs/fonts/NotoSansCJKsc-Regular.otf")
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({
    "font.family": [font_manager.FontProperties(fname=str(FONT)).get_name(), "DejaVu Sans"],
    "font.size": 12,
    "svg.fonttype": "path",
    "axes.unicode_minus": False,
})

BLUE = "#27618C"
BLUE_BG = "#EEF5FB"
ORANGE = "#A96117"
ORANGE_BG = "#FFF5E6"
GREEN = "#227364"
GREEN_BG = "#EAF7F2"
INK = "#203448"
GRAY = "#6D7A87"
GRAY_BG = "#F4F6F8"

fig, ax = plt.subplots(figsize=(24, 25))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 24)
ax.set_ylim(0, 25)
ax.axis("off")
fig.subplots_adjust(left=0.012, right=0.988, bottom=0.012, top=0.988)


def text(x, y, value, size=12, color=INK, weight="normal", ha="center", va="center"):
    ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, linespacing=1.45, zorder=6)


def box(x, y, w, h, value="", color=BLUE, fill=BLUE_BG, size=12, lw=1.3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.10",
        facecolor=fill, edgecolor=color, linewidth=lw, zorder=3,
    ))
    if value:
        text(x + w / 2, y + h / 2, value, size=size, color=INK)


def arrow(points, color=BLUE, lw=1.6, dashed=False):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw,
                linestyle="--" if dashed else "-", zorder=4)
    ax.add_patch(FancyArrowPatch(
        points[-2], points[-1], arrowstyle="-|>", mutation_scale=12,
        color=color, linewidth=lw, linestyle="--" if dashed else "-", zorder=4,
    ))


def panel(x, y, w, h, title, color=GRAY, fill="white"):
    box(x, y, w, h, color=color, fill=fill, lw=1.1)
    text(x + .23, y + h - .32, title, size=15, color=color, weight="bold", ha="left")


text(12, 24.55, "Intersection 改进前后：保留原注意力内核，增加几何反馈与最终曲线条件调制", 22, weight="bold")
text(12, 24.06, "本图对应原始 AE 与已采纳的 refinement + curve conditioning；不包含 local attention、point-feedback 或新 GAT", 12.5, color=GRAY)

# A. Side-by-side overview. Reused structure is not reused old checkpoint values.
panel(.25, 17.35, 10.9, 6.25, "A1  原始 AE：直接解码", BLUE)
panel(11.45, 17.35, 12.3, 6.25, "A2  两项改进整体：先解码，再反馈一次", ORANGE)

box(3.5, 22.42, 4.4, .52, "Holistic face latent  Z [N, 32]")
box(2.2, 21.38, 7, .72, "Face Transformer：32 → 768 → 32\n8 层 · 16 heads · FFN 2048", size=12)
arrow([(5.7, 22.42), (5.7, 22.10)])
text(5.7, 21.08, "F₀ [N, 32]", 12, BLUE)
arrow([(5.7, 21.38), (5.7, 21.22)])
box(.75, 19.55, 3.1, .82, "面几何头\n2D CNN + center/scale")
box(4.5, 19.55, 6.0, .82, "原 Intersection（下方 B 展开）\n有向候选 N² → E₀ [N², 64]")
arrow([(5.05, 21.07), (2.3, 21.07), (2.3, 20.40)])
arrow([(6.35, 21.07), (7.5, 21.07), (7.5, 20.40)])
box(4.5, 18.35, 3.0, .78, "保留的 E → 曲线几何头\n1D CNN + center/scale", size=11)
box(7.9, 18.35, 2.6, .78, "Linear 64 → 1\nsigmoid > 0.5", size=11.5)
arrow([(7.5, 19.55), (7.5, 19.32), (9.2, 19.32), (9.2, 19.16)])
arrow([(7.87, 18.74), (7.53, 18.74)])
text(2.3, 18.58, "面 [N, 16, 16, 3]", 11.5)
arrow([(2.3, 19.55), (2.3, 18.90)])
text(7.45, 17.88, "存在的有向曲线 [M, 16, 3]", 12)
arrow([(6, 18.35), (6, 18.08)])

box(14.8, 22.42, 5.6, .52, "同一 encoder / posterior，Z [N, 32]")
box(13.1, 21.38, 9, .72, "同样的 Face Transformer（8 层 / 768）\n得到 F₀；原 Intersection（8 层 / 512）得到 E₀", size=12)
arrow([(17.6, 22.42), (17.6, 22.10)])
box(12.2, 20.15, 10.8, .82, "初稿：面 / 曲线几何头 + 原 Linear classifier\nS₀ [N,16,16,3]，C₀ [P,16,3]，p₀ [P]；P = N(N−1)", size=12)
arrow([(17.6, 21.38), (17.6, 20.99)])
box(12.2, 18.94, 10.8, .80, "新增①  单轮 Refinement（下方 C 展开）\n读取初稿几何与 F₀ / E₀ / p₀ → ΔF、ΔE → F₁、E₁", ORANGE, ORANGE_BG)
arrow([(17.6, 20.15), (17.6, 19.77)], ORANGE)
box(12.2, 17.69, 10.8, .82, "最终：复用面 / 曲线几何头与 classifier\n新增② 仅最终曲线 / bbox 头读取 [E₁, F₁ᵢ, F₁ⱼ]（D）", GREEN, GREEN_BG)
arrow([(17.6, 18.94), (17.6, 18.54)], GREEN)
text(12, 16.98, "蓝色 = 沿用结构，权重随 decoder 微调；新模型的初稿 / 最终阶段共享同一套几何头与 classifier。Face Transformer 和原 inter 均只运行一次。", 11.8, color=GRAY)

# B. The original attention core, retained at exactly the same capacity.
panel(.25, 13.25, 23.5, 3.30, "B  两版共有的原 Intersection 内核：方向 i→j 单独输出一个 64 维特征", BLUE)
box(.75, 14.50, 3.15, 1.05, "[Fᵢ, Fⱼ]\n[P, 2, 32]\nLinear 32 → 512 + 位置编码", size=11.5)
box(4.40, 14.50, 2.5, 1.05, "第 1 个 token → tgt\n第 2 个 token → memory\n各 [P, 1, 512]", size=11.5)
arrow([(3.93, 15.03), (4.37, 15.03)])
box(7.40, 14.25, 11.50, 1.52, color=BLUE, fill="white")
text(13.15, 15.53, "TransformerDecoderLayer × 8    |    512 维 · 8 heads · FFN 2048 · dropout 0.1", 11.5, BLUE, weight="bold")
box(7.7, 14.50, 3.0, .62, "tgt self-attention\nQ / K / V 来自 tgt", size=11)
box(11.25, 14.50, 3.60, .62, "cross-attention\nQ 来自 tgt；K / V 来自 memory", size=10.5)
box(15.4, 14.50, 3.18, .62, "FFN 512 → 2048 → 512\n各子层：pre-LN + residual", size=10.5)
arrow([(6.93, 15.03), (7.67, 15.03)])
arrow([(6.93, 14.68), (7.15, 14.68), (7.15, 14.10), (13.05, 14.10), (13.05, 14.47)])
text(10.0, 13.94, "memory 保持固定，供每层 cross-attention 读取", 9.5, color=BLUE)
arrow([(10.73, 14.81), (11.22, 14.81)])
arrow([(14.88, 14.81), (15.37, 14.81)])
box(19.45, 14.50, 3.75, 1.05, "Linear 512 → 64\nEᵢⱼ [P, 64]\n反方向 j→i 另算", size=11.5)
arrow([(18.93, 15.03), (19.42, 15.03)])
text(12, 13.70, "两侧是整张面的 feature，不是局部 UV tokens。这里不生成显式顶点、wire、next/prev，也不平均 ij / ji 的几何。", 12, color=GRAY)

# C. Nested refinement blocks, each with explicit input/output shapes.
panel(.25, 5.99, 23.5, 6.84, "C  新增①：由初稿几何读出偏差，再用候选边消息更新面和边（只做一轮）", ORANGE)
panel(.65, 6.44, 6.80, 5.72, "C1  GeometryRead", ORANGE, ORANGE_BG)
panel(7.83, 6.44, 7.62, 5.72, "C2  边 → 面汇聚", ORANGE, ORANGE_BG)
panel(15.83, 6.44, 7.52, 5.72, "C3  面 / 边关系更新", ORANGE, ORANGE_BG)

box(1.05, 10.83, 6.0, .65, "曲线 C₀ [P,16,3] + 两个所属面 S₀\n每面 16×16 网格 → 450 个三角片", ORANGE, "white", 11.5)
box(1.05, 9.65, 6.0, .80, "最近点投影：曲线点 → 两个面（非 attention）\nXYZ(3) + 最近点(6) + 偏移(6) + 距离(2)\n→ observations [P,16,17]", ORANGE, "white", 11.4)
arrow([(4.05, 10.83), (4.05, 10.48)], ORANGE)
box(1.05, 8.48, 6.0, .78, "逐点 MLP：17 → 128 → 64\nLinear → LN → SiLU → Linear → SiLU\n→ [P,16,64]", ORANGE, "white", 11.3)
arrow([(4.05, 9.65), (4.05, 9.29)], ORANGE)
box(1.05, 7.25, 6.0, .82, "16 点 mean / max pooling → concat [P,128]\nLinear 128 → 64\n几何读数 G [P,64]", ORANGE, "white", 11.5)
arrow([(4.05, 8.48), (4.05, 8.10)], ORANGE)
text(4.05, 6.78, "投影用于读取，不直接替换曲线坐标。", 11, color=ORANGE)

box(8.25, 10.83, 6.78, .65, "concat [E₀(64), G(64)]\nLinear 128 → 128 → LN → SiLU", ORANGE, "white", 11.5)
box(8.25, 9.68, 6.78, .76, "用预测 p₀ 加权，汇聚到两个所属面\n加权求和 / 权重和\n得到每面上下文 C [N,128]", ORANGE, "white", 11.5)
arrow([(11.64, 10.83), (11.64, 10.47)], ORANGE)
box(8.25, 8.30, 6.78, .85, "关系向量 R 再按同一 p₀ 汇聚到面\nconcat [F₀(32), pooled R(128)]\nLinear 160 → 128 → LN → SiLU", ORANGE, "white", 11.5)
box(8.25, 7.05, 6.78, .80, "零初始化 Linear 128 → 32\nΔF [N,32]\nF₁ = F₀ + ΔF", ORANGE, "white", 11.5)
arrow([(11.64, 8.30), (11.64, 7.88)], ORANGE)

box(16.25, 10.83, 6.68, .65, "concat [E₀, G, F₀ᵢ, F₀ⱼ, Cᵢ, Cⱼ]\n64 + 64 + 32 + 32 + 128 + 128 = 448", ORANGE, "white", 11.5)
box(16.25, 9.68, 6.68, .76, "Linear 448 → 128 → LN → SiLU\nLinear 128 → 128 → SiLU\n关系向量 R [P,128]", ORANGE, "white", 11.5)
arrow([(19.59, 10.83), (19.59, 10.47)], ORANGE)
box(16.25, 8.30, 6.68, .85, "零初始化 Linear 128 → 64\nΔE [P,64]\nE₁ = E₀ + ΔE", ORANGE, "white", 11.5)
arrow([(19.59, 9.68), (19.59, 9.18)], ORANGE)
text(19.59, 7.43, "F₁ / E₁ 再进入最终几何头\n没有再次运行原 inter\n新增 143,328 个参数", 12, color=ORANGE)

# The gutters route features without crossing block interiors.
arrow([(7.08, 7.66), (7.64, 7.66), (7.64, 11.16), (8.22, 11.16)], ORANGE)
text(7.65, 8.95, "G", 11, color=ORANGE)
arrow([(15.06, 10.06), (15.64, 10.06), (15.64, 11.16), (16.22, 11.16)], ORANGE)
text(15.64, 10.66, "C", 11, color=ORANGE)
arrow([(16.22, 9.90), (15.64, 9.90), (15.64, 8.73), (15.06, 8.73)], ORANGE)
text(15.64, 9.32, "R", 11, color=ORANGE)
text(12, 6.18, "p₀ = sigmoid((logitᵢⱼ + logitⱼᵢ) / 2)；只控制软汇聚，不用 GT 筛边。最终几何由共享解码头输出，非 OCC 优化。", 11.5, color=GRAY)

# D. Exact insertion point and dimensions for the retained curve head.
panel(.25, .80, 23.5, 4.78, "D  新增②：只在最终曲线头中，用更新后的两张面调制隐层通道", GREEN)
box(.75, 3.85, 4.3, .86, "[E₁(64), F₁ᵢ(32), F₁ⱼ(32)]\nconcat 128 → Linear 128 → SiLU\n条件 hidden [P,128]", GREEN, GREEN_BG, 11.5)
box(5.65, 4.15, 4.55, .56, "零初始化 Linear 128 → 192\nγpoint / βpoint 各 96 维", GREEN, GREEN_BG, 11)
box(5.65, 3.15, 4.55, .56, "零初始化 Linear 128 → 1536\nγbbox / βbbox 各 768 维", GREEN, GREEN_BG, 11)
arrow([(5.08, 4.28), (5.35, 4.28), (5.35, 4.43), (5.62, 4.43)], GREEN)
arrow([(5.08, 4.28), (5.35, 4.28), (5.35, 3.43), (5.62, 3.43)], GREEN)

box(10.90, 4.00, 6.10, .82, "E₁ → 原 1D CNN → h [P,96,16]\nh′ = (1 + γpoint) × h + βpoint\n沿 16 个点广播", GREEN, GREEN_BG, 11.5)
box(17.60, 4.00, 5.55, .82, "原最终 Conv1d 96 → 6\nlocal curve [P,16,6]\nXYZ + normal", BLUE, BLUE_BG, 11.5)
arrow([(10.23, 4.43), (10.87, 4.43)], GREEN)
arrow([(17.03, 4.43), (17.57, 4.43)], GREEN)

box(10.90, 2.68, 6.10, .82, "E₁ → 原 7 个 ResBlock → h [P,768]\nh′ = (1 + γbbox) × h + βbbox", GREEN, GREEN_BG, 11.5)
box(17.60, 2.68, 5.55, .82, "原最终 Linear 768 → 4\ncenter (3) + scale (1)", BLUE, BLUE_BG, 11.5)
arrow([(10.23, 3.43), (10.56, 3.43), (10.56, 3.09), (10.87, 3.09)], GREEN)
arrow([(17.03, 3.09), (17.57, 3.09)], GREEN)
box(.75, 1.72, 9.45, 1.25, "沿用的 CNN 上采样（每阶段 1 个 ResBlock）\n曲线：32×2 → 768×2 → 384×4 → 192×8 → 96×16\n面：8×2² → 768×2² → 384×4² → 192×8² → 96×16²\n面末层 Conv2d 96→6；面头没有新增条件调制", BLUE, BLUE_BG, 11.1)
text(17.02, 2.20, "全局 XYZ = local XYZ × scale + center", 13, color=GREEN, weight="bold")
arrow([(20.38, 2.68), (20.38, 2.43)], GREEN)
text(12, 1.32, "conditioning 新增 239,424 参数；初稿与 encoder-edge teacher 不启用。两项总新增 382,752 参数；原 inter 8 层 / 512 维保持不变。", 11.5, color=GRAY)
text(12, .35, "候选规则：原模型 N²、逐方向判定；新模型 N(N−1)，仅 logit 对称化。最终仍输出两份有向曲线，并未增加显式 vertex / wire / coedge 拓扑头。", 11.7, color=GRAY)

fig.savefig(ROOT / "architecture.png", dpi=180, facecolor="white")
fig.savefig(ROOT / "architecture.svg", facecolor="white")
svg_path = ROOT / "architecture.svg"
svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
plt.close(fig)
print(ROOT / "architecture.png")
print(ROOT / "architecture.svg")
