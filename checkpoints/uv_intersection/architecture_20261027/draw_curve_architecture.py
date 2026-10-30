"""Paper exports of the current UVIntersectionDecoder, not proposed changes.

Run with matplotlib. Coordinates are pixels on a top-left-origin drawing board.
SVG retains text; PDF embeds TrueType fonts. No model loading is required.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, PathPatch
from matplotlib.path import Path as MplPath

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
INK = '#183042'; MUTED = '#506477'; LINE = '#64788a'
BLUE = '#e6f0fa'; PURPLE = '#eee8f8'; GREEN = '#e5f2ed'; ORANGE = '#fff0dc'; GREY = '#f1f4f7'


def canvas(w, h):
    fig = plt.figure(figsize=(w / 100, h / 100), facecolor='white')
    ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')
    return fig, ax


def text(ax, x, y, s, size=16, weight='normal', color=INK, align='center', mono=False):
    ax.text(x, y, s, fontsize=size, ha=align, va='center', color=color, weight=weight,
            family='DejaVu Sans Mono' if mono else 'DejaVu Sans', linespacing=1.45)


def box(ax, x, y, w, h, title, body='', color=BLUE, title_size=18, body_size=14):
    if isinstance(body, str) and body.startswith('#'):
        title_size = color if isinstance(color, (int, float)) else title_size
        color, body = body, ''
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=12',
                              linewidth=1.5, edgecolor=LINE, facecolor=color, zorder=2))
    if body:
        text(ax, x+w/2, y+h*.29, title, title_size, 'bold')
        text(ax, x+w/2, y+h*.68, body, body_size)
    else:
        text(ax, x+w/2, y+h/2, title, title_size, 'bold')


def arrow(ax, points, label='', lx=None, ly=None, size=13, color=LINE):
    path=MplPath(points, [MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=16,
                               linewidth=1.8, color=color, zorder=1))
    if label:
        if lx is None: lx=(points[0][0]+points[-1][0])/2
        if ly is None: ly=(points[0][1]+points[-1][1])/2-22
        ax.text(lx, ly, label, fontsize=size, ha='center', va='center', color=MUTED,
                family='DejaVu Sans Mono', linespacing=1.4,
                bbox=dict(facecolor='white', edgecolor='none', pad=2), zorder=4)


def add(ax, x, y, r=23):
    ax.add_patch(Circle((x,y),r,facecolor='white',edgecolor=LINE,linewidth=1.6,zorder=3))
    text(ax,x,y,'+',23,'bold')


def panel(ax, x, y, w, h, label, subtitle):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=18',
                              facecolor='white',edgecolor='#d9e1e8',linewidth=1.7,zorder=0))
    text(ax,x+25,y+35,label,22,'bold',align='left')
    text(ax,x+25,y+70,subtitle,14,color=MUTED,align='left')


def save(fig, name):
    for ext in ('svg','pdf','png'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=150 if ext=='png' else 100,facecolor='white')
    plt.close(fig)


def main_figure():
    fig,ax=canvas(2600,1940)
    text(ax,65,55,'Current curve decoder: shared queries, two UV trajectories',30,'bold',align='left')
    text(ax,65,102,'Source: src/brepnet/models/vae/uv_intersection.py  |  Current 64-D configuration  |  No proposed modules',16,color=MUTED,align='left')
    panel(ax,40,140,2520,790,'A  End-to-end data flow','N = face tokens; P = within-object unordered face pairs (i < j). Batch axes are shown here.')
    box(ax,80,320,140,90,'Latents','[N, 64]',GREY)
    box(ax,285,285,280,160,'GlobalFaceAttention','Linear 64 → 768\n8 × TransformerEncoder\nLinear 768 → 64',BLUE,17,14)
    arrow(ax,[(220,365),(285,365)],'[N,64]',252,329,11)
    box(ax,630,320,150,90,'Face features\nF','[N, 64]',GREY,13,14)
    arrow(ax,[(565,365),(630,365)],'[N,64]',597,329,11)
    box(ax,860,285,250,160,'CurveQueries','Gather Fᵢ, Fⱼ\nPair seed + t embedding\nExpanded in panel C',PURPLE,18,13)
    arrow(ax,[(780,365),(860,365)],'[N,64]',820,329,11)
    box(ax,1280,285,290,160,'CurveAttentionStack','8 × CurveAttentionBlock\nQ changes; M stays fixed\nSee the attention detail figure',BLUE,17,13)
    arrow(ax,[(1110,365),(1280,365)],'Q₀ [P,16,512]',1195,329,12)
    box(ax,1650,320,170,90,'Curve features\nE','[P, 16, 512]',GREY,13,13)
    arrow(ax,[(1570,365),(1650,365)],'[P,16,512]',1610,329,10)
    box(ax,1900,285,270,160,'DualUVHead','LayerNorm → Linear 512 → 4\nSigmoid (FP32)\nReshape + permute',PURPLE,18,13)
    arrow(ax,[(1820,365),(1900,365)],'[P,16,512]',1860,329,10)
    box(ax,2250,320,230,90,'Two UV paths','[P, 2, 16, 2]',PURPLE,18,14)
    arrow(ax,[(2170,365),(2250,365)],'[P,2,16,2]',2210,395,9)
    box(ax,1900,165,270,80,'EdgeExistenceHead','Mean(t) → LN → Linear 512 → 1',ORANGE,16,11)
    box(ax,2250,165,230,80,'Existence logits','[P]',ORANGE,17,14)
    arrow(ax,[(1735,320),(1735,205),(1900,205)],'[P,16,512]',1790,177,11)
    arrow(ax,[(2170,205),(2250,205)],'[P]',2210,182,11)
    box(ax,540,550,300,125,'FaceDecoder','Face CNN + placement head\nInternals folded',GREEN,18,14)
    arrow(ax,[(705,410),(705,550)],'[N,64]',760,480,13)
    box(ax,930,550,300,125,'FaceSpatialTokens','Spatial + XYZ feature concat\nPer-face grid UV embedding\nExpanded in panel B',GREEN,18,13)
    arrow(ax,[(840,612),(930,612)],'spatial [N,96,16,16]\nXYZ [N,16,16,3]',885,510,11)
    box(ax,1320,550,300,125,'PairMemory','Owner side embeddings\nConcatenate 256 + 256 tokens',GREEN,18,13)
    arrow(ax,[(1230,612),(1320,612)],'[N,256,512]',1275,581,10)
    arrow(ax,[(1470,550),(1470,445)],'K,V = M\n[P,512,512]',1553,497,12)
    box(ax,540,770,300,100,'FixedBSplineSurface','Fixed cubic, nonperiodic fit',GREEN,16,13)
    arrow(ax,[(690,675),(690,770)],'XYZ [N,16,16,3]',802,720,12)
    box(ax,1900,750,270,140,'Surface evaluation','Cᵢ(t) = Sᵢ(UVᵢ(t))\nCⱼ(t) = Sⱼ(UVⱼ(t))',GREEN,18,16)
    arrow(ax,[(840,820),(1900,820)],'Predicted poles [N,16,16,3] → gather owner pairs [P,2,16,16,3]',1370,790,13)
    arrow(ax,[(2365,410),(2365,692),(2035,692),(2035,750)],'[P,2,16,2]',2200,665,13)
    box(ax,2250,770,230,100,'Two 3D\ntrajectories','[P, 2, 16, 3]',GREEN,15,14)
    arrow(ax,[(2170,820),(2250,820)],'[P,2,16,3]',2210,784,10)
    text(ax,2365,902,'Not one guaranteed shared 3D curve',12,color=MUTED)
    text(ax,95,890,'Global attention: d = 768, 16 heads, FFN = 2048, pre-LN; attention is masked across objects.',13,color=MUTED,align='left')

    panel(ax,40,970,1240,900,'B  Memory construction','Per-face / per-pair views below omit the N or P batch axis. ⊕ means elementwise addition.')
    text(ax,80,1080,'B1  FaceSpatialTokens — concatenate features, then add grid position',16,'bold',align='left')
    box(ax,85,1120,210,100,'Spatial CNN\ngrid','96 × 16 × 16',GREEN,15,13)
    box(ax,85,1270,210,85,'Predicted XYZ','16 × 16 × 3',GREEN,16,13)
    box(ax,355,1195,200,95,'Feature\nconcat','96 + 3 = 99',GREEN,16,14)
    arrow(ax,[(295,1172),(325,1172),(325,1220),(355,1220)],'flatten → [256,96]',445,1158,11)
    arrow(ax,[(295,1312),(325,1312),(325,1265),(355,1265)],'flatten\n[256,3]',326,1365,11)
    box(ax,625,1195,225,95,'Linear + LN','99 → 512',GREEN,18,15)
    arrow(ax,[(555,1242),(625,1242)],'[256,99]',590,1167,10)
    add(ax,940,1242)
    arrow(ax,[(850,1242),(917,1242)],'[256,512]',885,1208,10)
    box(ax,1030,1195,200,95,'Face tokens T','[256, 512]',GREEN,17,14)
    arrow(ax,[(963,1242),(1030,1242)],'[256,512]',996,1208,10)
    box(ax,710,1325,400,95,'Grid UV position MLP','[256,2] → Linear 2→128 → SiLU → Linear 128→512',GREY,16,10)
    arrow(ax,[(940,1325),(940,1265)],'[256,512]',1033,1300,11)
    text(ax,80,1475,'B2  PairMemory — add owner identity, then concatenate tokens',16,'bold',align='left')
    box(ax,85,1555,210,75,'Tᵢ  [256,512]',GREEN,16)
    box(ax,85,1740,210,75,'Tⱼ  [256,512]',GREEN,16)
    add(ax,410,1592);add(ax,410,1777)
    arrow(ax,[(295,1592),(387,1592)],'[256,512]',341,1558,10)
    arrow(ax,[(295,1777),(387,1777)],'[256,512]',341,1743,10)
    box(ax,325,1505,170,52,'Side 0 [1,512]',GREY,13)
    box(ax,325,1690,170,52,'Side 1 [1,512]',GREY,13)
    arrow(ax,[(410,1557),(410,1569)],'broadcast',529,1547,10)
    arrow(ax,[(410,1742),(410,1754)],'broadcast',529,1732,10)
    box(ax,640,1610,270,120,'Token concat','256 + 256 = 512 tokens\nFeature width stays 512',GREEN,19,13)
    arrow(ax,[(433,1592),(535,1592),(535,1635),(640,1635)],'[256,512]',550,1563,11)
    arrow(ax,[(433,1777),(535,1777),(535,1700),(640,1700)],'[256,512]',550,1810,11)
    box(ax,1000,1610,230,120,'Memory M','[512, 512]\nK,V for all 8 blocks',GREEN,18,12)
    arrow(ax,[(910,1670),(1000,1670)],'[512,512]',955,1635,10)

    panel(ax,1320,970,1240,900,'C  Query construction','One shared 16-position query set per unordered pair. These are features, not geometric curve points.')
    box(ax,1400,1110,200,90,'Fᵢ','[64]',GREY,20,15)
    box(ax,1770,1110,200,90,'Fⱼ','[64]',GREY,20,15)
    box(ax,1450,1270,500,100,'Symmetric pair features','sum: Fᵢ + Fⱼ     |     difference: |Fᵢ − Fⱼ|',PURPLE,20,15)
    arrow(ax,[(1500,1200),(1500,1270)],'[64]',1547,1235,13)
    arrow(ax,[(1870,1200),(1870,1270)],'[64]',1917,1235,13)
    box(ax,2070,1270,350,100,'Feature concat','[sum, absolute difference]  →  [128]',PURPLE,20,14)
    arrow(ax,[(1950,1320),(2070,1320)],'2 × [64]',2010,1287,12)
    box(ax,2070,1440,350,110,'Seed MLP','Linear 128→512 → SiLU\nLinear 512→512',PURPLE,20,15)
    arrow(ax,[(2245,1370),(2245,1440)],'[128]',2300,1405,13)
    box(ax,2070,1610,350,90,'Broadcast seed\nover 16 positions',PURPLE,16)
    arrow(ax,[(2245,1550),(2245,1610)],'[512]',2300,1580,13)
    box(ax,1400,1450,200,90,'Uniform t','[16,1] in [0,1]',GREY,20,14)
    box(ax,1670,1450,330,90,'Position MLP','Linear 1→32 → SiLU → Linear 32→512',PURPLE,18,11)
    arrow(ax,[(1600,1495),(1670,1495)],'[16,1]',1635,1460,11)
    add(ax,1935,1748)
    arrow(ax,[(1835,1540),(1835,1748),(1912,1748)],'[16,512]',1773,1655,12)
    arrow(ax,[(2245,1700),(2245,1748),(1958,1748)],'[16,512]',2145,1725,12)
    box(ax,1745,1790,380,70,'Initial queries Q₀  [16,512]',PURPLE,16)
    arrow(ax,[(1935,1771),(1935,1790)],'[16,512]',2055,1775,11)
    text(ax,65,1900,'All three concatenations are explicit: (1) 96+3 features, (2) 256+256 tokens, (3) 64+64 pair features. Query and memory are never concatenated.',14,color=MUTED,align='left')
    save(fig,'curve_architecture')


def attention_figure():
    fig,ax=canvas(2920,1000)
    text(ax,65,55,'CurveAttentionBlock: the exact pre-LN update',30,'bold',align='left')
    text(ax,65,105,'Repeated 8 times. Only Q changes between blocks; the same M is supplied to every cross-attention layer. Shapes omit the pair batch axis P.',16,color=MUTED,align='left')
    panel(ax,40,150,2840,770,'One block','Every attention module has 8 heads, model width 512, and attention dropout 0.1. The circles are residual additions.')
    box(ax,80,440,165,90,'Qᶩ','[16,512]',PURPLE,22,14)
    box(ax,315,440,125,90,'LN',BLUE,22)
    box(ax,520,390,270,190,'Self-attention','Q = K = V = LN(Qᶩ)\nMHA, 8 heads\nOutput dropout 0.1',BLUE,19,15)
    add(ax,880,485)
    arrow(ax,[(245,485),(315,485)],'[16,512]',280,451,10)
    arrow(ax,[(440,485),(520,485)],'Q,K,V\n[16,512]',480,432,10)
    arrow(ax,[(790,485),(857,485)],'[16,512]',823,451,10)
    arrow(ax,[(275,485),(275,290),(880,290),(880,462)],'Residual [16,512]',580,262,13)
    box(ax,985,440,125,90,'LN',BLUE,22)
    arrow(ax,[(903,485),(985,485)],'[16,512]',944,451,10)
    box(ax,1190,390,300,190,'Cross-attention','Q = LN(updated query)\nK = V = memory M\nMHA → output dropout 0.1',GREEN,19,14)
    arrow(ax,[(1110,485),(1190,485)],'Q\n[16,512]',1150,432,10)
    add(ax,1580,485)
    arrow(ax,[(1490,485),(1557,485)],'[16,512]',1523,451,10)
    arrow(ax,[(945,485),(945,290),(1580,290),(1580,462)],'Residual [16,512]',1255,262,13)
    box(ax,1190,710,300,110,'Memory M','[512,512]\nRead only; no write-back',GREEN,19,14)
    arrow(ax,[(1340,710),(1340,580)],'K,V [512,512]',1455,650,13)
    box(ax,1695,440,125,90,'LN',BLUE,22)
    arrow(ax,[(1603,485),(1695,485)],'[16,512]',1649,451,10)
    box(ax,1900,365,410,240,'FFN + output dropout','Linear 512 → 2048\nReLU → Dropout 0.1\nLinear 2048 → 512\nDropout 0.1',BLUE,21,17)
    arrow(ax,[(1820,485),(1900,485)],'[16,512]',1860,451,10)
    add(ax,2400,485)
    arrow(ax,[(2310,485),(2377,485)],'[16,512]',2343,451,10)
    arrow(ax,[(1655,485),(1655,290),(2400,290),(2400,462)],'Residual [16,512]',2030,262,13)
    box(ax,2515,430,300,110,'Qᶩ⁺¹','[16,512]',PURPLE,23,17)
    arrow(ax,[(2423,485),(2515,485)],'[16,512]',2469,451,10)
    text(ax,90,875,'The final block output is E = Q⁸. DualUVHead consumes E directly; no face feature or memory concatenation is inserted before that head.',16,color=MUTED,align='left')
    save(fig,'curve_attention_detail')


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    main_figure();attention_figure()
