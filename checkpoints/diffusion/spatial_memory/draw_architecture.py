"""Draw the implemented spatial_photo_v1 network and its module boundaries."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
INK, GRAY, BLUE, TEAL, ORANGE = "#23374a", "#edf1f5", "#226ac1", "#167f7b", "#ad6b14"


def box(ax, x, y, w, h, title, detail, color=INK, fill=GRAY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.02,rounding_size=.07",
                              facecolor=fill, edgecolor=color, linewidth=1.3, zorder=3))
    ax.text(x + w / 2, y + .25, title, color=color, ha="center", va="center",
            fontsize=11, weight="bold", zorder=4)
    ax.text(x + w / 2, y + .51, detail, color=INK, ha="center", va="top",
            fontsize=9.8, linespacing=1.55, zorder=4)


def arrow(ax, points, color=INK, dashed=False):
    for p, q in zip(points[:-2], points[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=1.4, zorder=1,
                linestyle="--" if dashed else "-")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=1.4, zorder=2, linestyle="--" if dashed else "-"))


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})
    fig, ax = plt.subplots(figsize=(16, 12.4), dpi=160)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 16), ylim=(12.4, 0))
    ax.axis("off")
    ax.text(.45, .45, "Photo → spatial memory → 30 B-Rep latent slots", fontsize=22, weight="bold", color=INK)
    ax.text(.45, .85, "Implemented: spatial_photo_v1  |  B = batch size  |  Frozen original DINO + VAE  |  No camera / GT topology at inference",
            fontsize=10.4, color="#586b7d")

    box(ax, .5, 1.3, 2.7, 1.35, "Photo input", "flux RGB [B,512,512,3]\nResize224 + ImageNet normalize\nI [B,3,224,224]")
    box(ax, 3.8, 1.3, 3.6, 1.35, "Frozen DINOv2 ViT-L/14-reg", "Patch14; 24 blocks; width1024\n16 heads; FP32 / eval / no grad\nCLS+patch: [B,257,1024]")
    box(ax, 8., 1.3, 3.2, 1.35, "Image token projection", "LayerNorm1024 → Linear1024→512\nP [B,257,512]\nCLS used by spatial lifting", BLUE, "#edf5ff")
    box(ax, 11.8, 1.3, 3.7, 1.35, "RGB detail memory", "Drop CLS → 256 image patches\n+ learned RGB type embedding\nM_rgb [B,256,512]", BLUE, "#edf5ff")
    arrow(ax, [(3.2, 1.98), (3.8, 1.98)])
    arrow(ax, [(7.4, 1.98), (8., 1.98)], BLUE)
    arrow(ax, [(11.2, 1.98), (11.8, 1.98)], BLUE)

    box(ax, .5, 3.2, 2.7, 1.65, "8 × 8 × 8 spatial anchors", "Fixed grid-cell centers [512,3]\nExtent [-1,1] in canonical space\nXYZ + sin/cos, 6 bands → [512,39]\nMLP 39→512→512 (SiLU)", TEAL, "#e9f7f5")
    box(ax, 3.8, 3.2, 3.6, 1.65, "Image-to-3D query encoder × 4", "Q0 [B,512,512] from anchor MLP\nSelf-attn → image cross-attn → FFN\n8 heads; FFN512→2048→512 GELU\nPre-LN + residual at each sublayer", TEAL, "#e9f7f5")
    box(ax, 8., 3.2, 3.2, 1.65, "Spatial memory", "Final LN + spatial type embedding\nM_xyz [B,512,512]\n512 positions × 512 channels\nPosition ≠ guaranteed 3D evidence", TEAL, "#e9f7f5")
    box(ax, 11.8, 3.2, 3.7, 1.65, "Shared condition memory M", "Concat spatial + RGB on token axis\n[B,768,512] = 512 + 256 tokens\nComputed ONCE per input photo\nRead-only across denoising steps", TEAL, "#e9f7f5")
    arrow(ax, [(3.2, 4.02), (3.8, 4.02)], TEAL)
    arrow(ax, [(7.4, 4.02), (8., 4.02)], TEAL)
    arrow(ax, [(11.2, 4.02), (11.8, 4.02)], TEAL)
    arrow(ax, [(13.65, 2.65), (13.65, 3.2)], BLUE)
    arrow(ax, [(9.6, 2.65), (9.6, 2.94), (5.6, 2.94), (5.6, 3.2)], BLUE)

    box(ax, .5, 5.5, 2.7, 1.35, "Noisy face latent z_t", "[B,30,32]\nRandom duplicate padding to 30\nNo face-slot positional encoding", BLUE, "#edf5ff")
    box(ax, 3.8, 5.5, 3.6, 1.35, "Face input / timestep", "Linear32→1024; LN; SiLU; Linear\n+ sin/cos(t)→MLP1024→1024→1024\nH0 [B,30,1024]", BLUE, "#edf5ff")
    box(ax, 8., 5.5, 3.2, 1.35, "24 denoising blocks", "Face self-attn → read M → FFN\n16 heads × 64; time adaLN\nH24 [B,30,1024]", BLUE, "#edf5ff")
    box(ax, 11.8, 5.5, 3.7, 1.35, "Noise prediction", "Final LN1024 → Linear1024→32\nepsilon [B,30,32]\nTraining: uniform t + epsilon MSE", BLUE, "#edf5ff")
    arrow(ax, [(3.2, 6.17), (3.8, 6.17)], BLUE)
    arrow(ax, [(7.4, 6.17), (8., 6.17)], BLUE)
    arrow(ax, [(11.2, 6.17), (11.8, 6.17)], BLUE)
    arrow(ax, [(13.65, 4.85), (13.65, 5.17), (9.6, 5.17), (9.6, 5.5)], TEAL)

    box(ax, .5, 7.5, 6.9, 1.2, "One denoising block: attention shapes", "Self: Q/K/V [B,16,30,64] → attention [B,16,30,30]\nCross: Q [B,16,30,64]; K/V [B,16,768,64] → attention [B,16,30,768]\nCross output returns [B,30,1024]; memory never becomes extra face slots", BLUE, "#edf5ff")
    box(ax, 8., 7.5, 3.2, 1.2, "DDPM reverse update", "1000 steps, cosine noise schedule\nz_t → z_(t-1) [B,30,32]\nEach step runs all 24 blocks")
    box(ax, 11.8, 7.5, 3.7, 1.2, "Fixed original VAE → B-Rep", "z0 → latent dedup → [N_i,32]\nVAE raw surfaces / intersections\nExisting OCC postprocess → solid")
    arrow(ax, [(13.65, 6.85), (13.65, 7.12), (9.6, 7.12), (9.6, 7.5)])
    arrow(ax, [(11.2, 8.1), (11.8, 8.1)])
    arrow(ax, [(8., 8.1), (7.7, 8.1), (7.7, 7.12), (.2, 7.12), (.2, 6.17), (.5, 6.17)], BLUE)

    ax.plot([.5, 15.5], [9.1, 9.1], color="#cbd5df")
    ax.text(.5, 9.48, "Module boundaries and per-layer operations", fontsize=15, weight="bold", color=INK)
    box(ax, .5, 9.85, 6.9, 1.9, "spatial_blocks.py · SpatialDenoisingBlock", "a = adaLN(H, t); U = H + gate_a(t) × SelfAttention(a,a,a)\nV = U + CrossAttention(LN(U), LN(M), LN(M))\nH_next = V + gate_f(t) × FFN(adaLN(V,t))\nFFN: Linear1024→2048 → GELU → dropout → Linear2048→1024\nShift/scale initially 0; gates initially 1; each of 24 blocks has its own weights", BLUE, "#edf5ff")
    box(ax, 8., 9.85, 7.5, 1.9, "Composition in spatial_model.py", "SpatialPhotoDataset → RGB + original VAE mean for training\nspatial_conditioning.py: DINO tokens → SpatialImageMemory\nspatial_blocks.py: SpatialMemoryDenoiser → epsilon\nspatial_model.py: noise / MSE / sampling / fixed original VAE decode\nmodels registry + train.py: unified Lightning entry and strict checkpoint restore", TEAL, "#e9f7f5")
    ax.text(.5, 12.07, "New trainable encoder + denoiser; no old denoiser resume. Implementation verified; geometric correspondence and reconstruction gains are untested.",
            fontsize=10.4, color="#586b7d")
    output = ROOT / "visualizations"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "spatial_photo_v1.png", dpi=160)
    fig.savefig(output / "spatial_photo_v1.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
