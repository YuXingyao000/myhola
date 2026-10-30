"""Draw the audited workflow, without running scene/data generation.

Solid nodes describe current source. Dashed orange nodes are the agreed,
not-yet-implemented dataset changes. Each page is also exported as vector art.
"""

from pathlib import Path
import ast
import hashlib
import json
import math
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path as MPath
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'figures'
REPO = ROOT.parents[3]
FONT = Path('/root/.cache/cad_scene_figures/SourceHanSansSC-Regular.otf')
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(FONT)).get_name(),
                     'font.size': 12, 'pdf.fonttype': 42, 'svg.fonttype': 'path',
                     'axes.unicode_minus': False})

INK, MUTED, LINE = '#172B3A', '#52616D', '#70808A'
ROLE = {
    'host': ('#F0F3F5', '#586B79', '程序'),
    'agent': ('#E9F3FC', '#0072B2', 'Agent'),
    'tool': ('#E8F5EE', '#00845C', '工具'),
    'model': ('#F1ECF8', '#7953A6', '模型'),
    'pending': ('#FFF5E9', '#B86916', '待改'),
    'stop': ('#FBEFED', '#B24D3D', '终态'),
}
PAGES, NODES, EDGES = [], [], []


def label(ax, x, y, value, size=12, color=INK, ha='center', **kwargs):
    return ax.text(x, y, value, fontsize=size, color=color, ha=ha, va='center',
                   linespacing=1.45, zorder=6, **kwargs)


def snowflake(ax, x, y, radius=.12, color=INK):
    for angle in range(0, 360, 60):
        a = math.radians(angle)
        u = (math.cos(a), math.sin(a))
        ax.plot([x, x + radius*u[0]], [y, y + radius*u[1]], color=color, lw=1.1, zorder=6)
        for sign in (-1, 1):
            b = a + sign*math.pi/3
            bx, by = x + .65*radius*u[0], y + .65*radius*u[1]
            ax.plot([bx, bx + .35*radius*math.cos(b)],
                    [by, by + .35*radius*math.sin(b)], color=color, lw=.9, zorder=6)


def page(stem, title, subtitle, height):
    fig, ax = plt.subplots(figsize=(12.6, height*.9))
    fig.subplots_adjust(left=.025, right=.985, top=.985, bottom=.015)
    ax.set(xlim=(0, 14), ylim=(height, 0))
    ax.axis('off')
    ax._stem = stem
    label(ax, .25, .35, title, 21, ha='left')
    label(ax, .25, .83, subtitle, 11.5, MUTED, ha='left')
    for x, role in [(0.3, 'host'), (2.35, 'agent'), (4.4, 'tool'), (6.45, 'model'), (8.5, 'pending')]:
        fill, accent, name = ROLE[role]
        ax.add_patch(Rectangle((x, 1.18), .2, .2, fc=fill, ec=accent,
                               lw=1, linestyle='--' if role == 'pending' else '-'))
        label(ax, x+.3, 1.28, name, 11, ha='left')
    snowflake(ax, 10.65, 1.28, .13)
    label(ax, 10.92, 1.28, '只推理，不训练', 11, ha='left')
    PAGES.append((stem, fig))
    return ax


def node(ax, key, x, y, w, h, role, title, algorithm, output, frozen=False):
    assert len(title) <= 10, title
    fill, accent, _ = ROLE[role]
    ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=accent, lw=1.1,
                           linestyle=(0, (4, 3)) if role == 'pending' else '-', zorder=3))
    label(ax, x+w/2, y+(.24 if h <= 1.2 else .29), title, 15)
    label(ax, x+w/2, y+(.59 if h <= 1.2 else .68), algorithm, 11.5, MUTED)
    label(ax, x+w/2, y+h-.23, output, 11.2)
    if frozen:
        snowflake(ax, x+w-.2, y+.23, .12, accent)
    NODES.append(dict(figure=ax._stem, id=key, title=title, role=role,
                      algorithm=algorithm, output=output, frozen_in_this_pipeline=frozen))


def arrow(ax, key, points, condition, at, color=LINE, style='-'):
    path = MPath(points, [MPath.MOVETO] + [MPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=13,
                                color=color, linewidth=1.1, linestyle=style, zorder=2))
    if condition:
        label(ax, *at, condition, 10.8, color,
              bbox={'facecolor':'white', 'edgecolor':'none', 'pad':1.2})
    EDGES.append(dict(figure=ax._stem, id=key, condition=condition))


def panel(ax, x, y, w, h, title):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec='#C0C9CF', lw=.85, zorder=0))
    label(ax, x+.2, y+.25, title, 13, ha='left')


def footer(ax, y, lines):
    for index, line in enumerate(lines):
        label(ax, .3, y+index*.31, line, 10.5, MUTED, ha='left')


def picture(ax, path, x, y, w, h):
    image = Image.open(path).convert('RGB')
    iw, ih = image.size
    ratio = min(w/iw, h/ih)
    dx, dy = (w-iw*ratio)/2, (h-ih*ratio)/2
    ax.imshow(image, extent=(x+dx, x+dx+iw*ratio, y+dy+ih*ratio, y+dy), zorder=1)


def overview():
    ax = page('00-overview', '代码总览：四个阶段，文件连接',
              '按 2026-10-29 源码绘制；橙色虚线为新约定，尚未接入现有入口。', 8.8)
    xs = [.3, 3.8, 7.3, 10.8]
    definitions = [
        ('s1','agent','生成场景','Codex + Blender','模板包 / room.blend',True),
        ('s2','agent','选择点位','Agent + 几何检查','准备场景 / ready.json',True),
        ('s3','tool','拍摄零件','Cycles + 程序化 PBR','RGB / mask / depth',False),
        ('s4','model','整图调色','Image-Adaptive-3DLUT','调色 PNG / 配比记录',True),
    ]
    for x, d in zip(xs, definitions):
        key, role, title, algo, out, frozen = d
        node(ax,key,x,1.95,2.9,1.45,role,title,algo,out,frozen)
    for i, condition in enumerate(['场景产物齐备','有效点位非空','图片清单导出']):
        arrow(ax,f's{i+1}_s{i+2}',[(xs[i]+2.9,2.66),(xs[i+1],2.66)],condition,
              (xs[i]+3.25,3.65))
    pictures = [
        '/mnt/d/data/cad_scene_templates/office_20261023_001/workflow/round_00/preview/renders/room_overview.png',
        '/mnt/d/data/cad_scene_placements/office_step2_support_fix_20261026_001/candidate_00/after.png',
        '/mnt/d/data/cad_scene_renders/office_step3_support_fix_20261026_001/00269315/rgb.png',
        '/mnt/d/data/cad_scene_enhancements/office_phone_lut_20261027_001/00269315/paired_mix_0.5.png',
    ]
    captions = ['已有办公室','20 cm 方体试拍','STL 实际拍摄','paired · mix 0.5']
    for x, path, caption in zip(xs,pictures,captions):
        picture(ax,path,x,4.0,2.9,1.75)
        label(ax,x+1.45,5.96,caption,11.5)
    node(ax,'dataset_pending',.3,6.55,8.15,1.35,'pending','全点位数据集',
         '52,556 CAD × 5 点位；每 CAD 固定尺寸；仅 paired / 0.5',
         '262,780 拍摄样本 + 同数调色图；按 CAD / 点位组织')
    node(ax,'history',8.8,6.55,4.9,1.35,'host','历史支线',
         'FLUX 重绘 / 五层合成 / 旧特写模板', '代码仍保留；不在本轮拟用路线')
    footer(ax,8.26,[
        '四段目前分别启动，没有自动串联的总入口；Step 3→LUT 仍需准备输入清单。缩略图为历史产物示意。',
        '前置：prepare_stl.py 准备网格（PLY 优先，STEP 为回退）。雪花仅表示本流程不训练参数。',
    ])


def scene():
    ax = page('01-scene', 'Step 1 · 生成场景',
              '外围主循环：room_workflow._execute_locked()；制作、审阅、返修续接同一个 Codex 会话。', 15.1)
    w,h=3.6,1.35
    for args in [
        ('inputs',.4,1.9,'host','输入固化','参数 / 指令 / SHA256','request.json',False),
        ('assets',5.15,1.9,'host','素材准备','adapt 复制 / quality_only 空包','assets/ + assets.json',False),
        ('create',9.9,1.9,'agent','场景制作','Astra / ultra · 新 exec [1]','脚本 / 视角 / 素材',True),
        ('build',.4,4.25,'tool','独立重建','执行脚本 / 依赖与摘要检查','room.blend / build_report',False),
        ('render',5.15,4.25,'tool','六图渲染','Blender Cycles [2]','6 PNG / preview_report',False),
        ('inspect',9.9,4.25,'agent','看图审阅','原会话 resume / 只读多图','ready / revise / blocked',True),
        ('repair',.4,6.8,'agent','场景返修','反馈 + 原会话 resume','修改脚本 / 素材 / 视角',True),
        ('budget',5.15,6.8,'host','检查预算','repairs_used < max_repairs','返修次数 / 待处理反馈',False),
        ('ready',9.9,6.8,'host','待人审','ready 且无遗留问题','ready_for_review',False),
    ]:
        key,x,y,role,title,algo,out,frozen=args
        node(ax,key,x,y,w,h,role,title,algo,out,frozen)
    arrow(ax,'input_assets',[(4,2.575),(5.15,2.575)],'输入有效',(4.58,2.29))
    arrow(ax,'assets_create',[(8.75,2.575),(9.9,2.575)],'准备完成',(9.33,2.29))
    arrow(ax,'create_build',[(11.7,3.25),(11.7,3.76),(2.2,3.76),(2.2,4.25)],'制作回合完成',(6.95,3.76))
    arrow(ax,'build_render',[(4,4.925),(5.15,4.925)],'构建通过',(4.58,4.64))
    arrow(ax,'render_inspect',[(8.75,4.925),(9.9,4.925)],'六图齐备',(9.33,4.64))
    arrow(ax,'inspect_ready',[(11.7,5.6),(11.7,6.8)],'ready',(12.2,6.19))
    arrow(ax,'technical_failure',[(3.5,5.6),(3.5,6.18),(6.4,6.18),(6.4,6.8)],'技术失败',(4.5,6.18),ROLE['stop'][1])
    arrow(ax,'render_failure',[(6.95,5.6),(6.95,6.8)],'渲染失败',(7.58,6.16),ROLE['stop'][1])
    arrow(ax,'revise',[(10.4,5.6),(10.4,6.18),(7.8,6.18),(7.8,6.8)],'revise',(9.1,6.18))
    arrow(ax,'budget_repair',[(5.15,7.475),(4,7.475)],'有预算',(4.58,7.18))
    arrow(ax,'repair_build',[(2.2,6.8),(2.2,5.6)],'返修完成',(.91,6.43))
    node(ax,'incomplete',5.15,8.95,3.6,1.1,'stop','未完成','保留错误与各轮证据','incomplete')
    arrow(ax,'exhausted',[(6.95,8.15),(6.95,8.95)],'预算耗尽',(7.68,8.54),ROLE['stop'][1])
    arrow(ax,'blocked',[(13.5,4.925),(13.87,4.925),(13.87,9.5),(8.75,9.5)],'blocked',(11.6,9.5),ROLE['stop'][1])
    panel(ax,.4,10.55,13.1,3.14,'Codex 内部工具循环：具体工具顺序与迭代次数由 Codex 决定')
    label(ax,.65,11.12,'制作 / 返修可读写与渲染；审阅仅读取与看图（指令 + 摘要检查，非 OS 隔离）。',10.8,MUTED,ha='left')
    node(ax,'inner_model',.65,11.48,3.25,1.1,'agent','模型判断','Astra 推理 [1]','工具请求 / 最终回复',True)
    node(ax,'inner_tool',5.4,11.48,3.25,1.1,'tool','工具执行','按阶段许可选择工具','文件 / 图像 / 日志')
    node(ax,'inner_result',10.0,11.48,3.25,1.1,'host','回传结果','Codex CLI 管理上下文','下一轮可用证据')
    arrow(ax,'tool_call',[(3.9,12.03),(5.4,12.03)],'需要工具',(4.65,11.75))
    arrow(ax,'tool_result',[(8.65,12.03),(10,12.03)],'工具结束',(9.32,11.75))
    arrow(ax,'inner_loop',[(11.62,12.58),(11.62,13.0),(2.27,13.0),(2.27,12.58)],'带工具结果继续判断',(6.95,13.0))
    arrow(ax,'inner_return',[(1.1,12.58),(1.1,13.42),(3.45,13.42)],'最终回复',(2.15,13.42))
    label(ax,4.25,13.42,'返回外围',12)
    footer(ax,14.0,[
        '默认最多 3 次外围返修；构建或六图失败直接形成 revise。宿主 / 调用异常保留 incomplete。',
        '待人审即程序返回，不弹出人工 QA。外部反馈经显式 resume 再入返修预算；云端 checkpoint 不由本仓库冻结。',
        '素材：Poly Haven / ambientCG；Sketchfab 需凭据；CGBookcase 通过本地目录。均不是模型权重。',
        '[1] Codex exec / exec resume；[2] Blender Cycles。每框从上到下为：名称、算法 / 方法、产物。',
    ])


def placement():
    ax=page('02-placement','Step 2 · 选择点位',
            '外围主循环：step2.workflow.run_workflow()；Agent 返回 JSON，Blender 动作由外围固定调度。',14.0)
    xs=[.35,3.85,7.35,10.85]
    entries=[('describe','tool','建立索引','对象 / AABB / 图像编号','scene_context.json'),
             ('regions','agent','选取区域','多图判断 · 最多 6 区','区域 / 支撑 / 可动组'),
             ('support','tool','准备支撑','材质位移 → BUMP 副本','prepared_scene.blend'),
             ('measure','tool','测量空间','BVH 射线 / 1 cm 高度场','落点 / 占用分类')]
    for x,(key,role,title,algo,out) in zip(xs,entries):
        node(ax,key,x,1.92,2.8,1.4,role,title,algo,out,role=='agent')
    for index,condition in enumerate(['索引就绪','候选合法','副本就绪']):
        arrow(ax,f'prep_{index}',[(xs[index]+2.8,2.62),(xs[index+1],2.62)],condition,(xs[index]+3.15,3.58))
    node(ax,'propose',.45,4.5,3.6,1.4,'agent','构图提议','Agent + 测量 + 上轮反馈','落点 / 相机 / 完整移动方案',True)
    node(ax,'preview',5.2,4.5,3.6,1.4,'tool','执行检查','preview_placement()','数值报告 / 可用图片')
    node(ax,'inspect',9.95,4.5,3.6,1.4,'agent','看图评议','图片 + 报告；原会话续接','keep / revise / drop',True)
    arrow(ax,'measured',[(12.25,3.32),(12.25,4.01),(2.25,4.01),(2.25,4.5)],'measured 的候选进入循环',(7.0,4.01))
    arrow(ax,'proposed',[(4.05,5.2),(5.2,5.2)],'方案齐全',(4.62,4.92))
    arrow(ax,'previewed',[(8.8,5.2),(9.95,5.2)],'检查结束',(9.37,4.92))
    node(ax,'reject',.45,7.2,3.6,1.2,'stop','记录淘汰','drop / 末轮未过 / 无表面','rejected + 原因')
    node(ax,'decide',5.2,7.2,3.6,1.2,'host','合并判定','数值 ∧ 图片声明 ∧ keep','候选状态 / 剩余轮次')
    node(ax,'keep',9.95,7.2,3.6,1.2,'host','保留点位','三项同时满足','proxy_ready')
    arrow(ax,'review_done',[(11.75,5.9),(11.75,6.5),(7.0,6.5),(7.0,7.2)],'结构化意见返回',(9.35,6.5))
    arrow(ax,'keep_result',[(8.8,7.8),(9.95,7.8)],'三项满足',(9.37,7.5))
    arrow(ax,'reject_result',[(5.2,7.8),(4.05,7.8)],'drop /\n末轮未过',(4.62,7.35),ROLE['stop'][1])
    arrow(ax,'retry',[(5.65,7.2),(5.65,6.5),(2.25,6.5),(2.25,5.9)],'其余 + 有轮次：反馈后重试',(3.7,6.5))
    arrow(ax,'unmeasured',[(13.65,2.62),(13.9,2.62),(13.9,9.0),(2.25,9.0),(2.25,8.4)],'无有效支撑表面',(9.7,9.0),ROLE['stop'][1])
    node(ax,'export',5.2,9.15,3.6,1.2,'host','导出点位','活跃候选为空 / 轮次结束','ready.json / placements.json')
    arrow(ax,'reject_export',[(2.25,8.4),(2.25,9.75),(5.2,9.75)],'全部候选结束',(3.7,9.75))
    arrow(ax,'keep_export',[(11.75,8.4),(11.75,9.75),(8.8,9.75)],'全部候选结束',(10.25,9.75))
    ax.plot([11.75,11.75],[8.88,9.12],color='white',lw=5,zorder=2.5)
    ax.plot([11.75,11.75],[8.88,9.12],color=LINE,lw=1.1,zorder=2.6)
    panel(ax,.45,10.85,13.1,1.9,'执行检查内部：每次重开准备场景，移动不累积；数值失败也要回传报告供 Agent 评议')
    node(ax,'move',.7,11.4,3.6,1.1,'tool','执行移动','整组 XY 平移 / Z 轴旋转','终态矩阵 / 移动检查')
    node(ax,'geometry',5.25,11.4,3.6,1.1,'tool','检查构图','AABB / 保护体 / 支撑 / 投影','碰撞 / 遮挡 / 入镜报告')
    node(ax,'photo',9.8,11.4,3.5,1.1,'tool','前后试拍','Cycles / IDMask / Z 反投影','before / after / 支撑检查')
    arrow(ax,'move_geometry',[(4.3,11.95),(5.25,11.95)],'移动合法',(4.77,12.69))
    arrow(ax,'geometry_photo',[(8.85,11.95),(9.8,11.95)],'数值通过',(9.32,12.69))
    footer(ax,13.13,[
        '默认初版 + 2 次修订；每轮只处理未结束候选。代理 20 cm，边余量 2 cm，支撑高度容差 2 mm。',
        '遮挡：场景 AABB 与相机→目标盒有限保护体相交；不做物理推移模拟。completed 允许 0 个通过，人工仍 pending。',
        '无活跃候选直接导出；prepare_only 在索引后结束。工具 / JSON 异常停止并保留 incomplete；原场景不回写。',
    ])


def capture():
    ax=page('03-capture','Step 3 · 拍摄零件',
            '当前代码：workflow.py 生成任务；capture_batch() 先循环机位，再循环分配给该机位的 CAD。',12.9)
    w,h=3.6,1.35
    defs=[
        ('sample',.45,1.9,'host','抽取零件','split 筛选 / STL 摘要去重','seed 固定的 count 个 CAD',False),
        ('assign',5.2,1.9,'host','分配机位','打乱点位 / 轮转分配','每 CAD 仅 1 点；15–20 cm',False),
        ('request',9.95,1.9,'host','保存请求','输入 / 配置 / 源码快照','request.json / source/',False),
        ('open',.45,4.15,'tool','打开机位','加载准备场景 / 应用变换','相机 / 材质 / 输出节点',False),
        ('place',5.2,4.15,'tool','导入摆放','STL / 平滑法线 / 等比缩放','XY 居中 / 最低 Z 贴表面',False),
        ('check',9.95,4.15,'host','检查摆放','代理包络 / 高度差 / 投影','接触与完整入镜报告',False),
        ('next',.45,6.7,'host','推进拍摄','遍历机位和该机位任务','下一件 / 下一机位 / 结束',False),
        ('cleanup',5.2,6.7,'host','记录清理','写记录 / 删除本件 CAD','metadata / renders.jsonl',False),
        ('render',9.95,6.7,'tool','渲染照片','Cycles PBR + OIDN [2,3]','RGB / mask / 全场景深度',False),
    ]
    for key,x,y,role,title,algo,out,freeze in defs:
        node(ax,key,x,y,w,h,role,title,algo,out,freeze)
    snowflake(ax,13.32,7.38,.105,ROLE['model'][1])
    arrow(ax,'sample_assign',[(4.05,2.575),(5.2,2.575)],'数量足够',(4.62,2.29))
    arrow(ax,'assign_request',[(8.8,2.575),(9.95,2.575)],'任务已定',(9.37,2.29))
    arrow(ax,'request_open',[(11.75,3.25),(11.75,3.73),(2.25,3.73),(2.25,4.15)],'启动 Blender 子进程',(7,3.73))
    arrow(ax,'open_place',[(4.05,4.825),(5.2,4.825)],'机位就绪',(4.62,4.54))
    arrow(ax,'place_check',[(8.8,4.825),(9.95,4.825)],'模型已放',(9.37,4.54))
    arrow(ax,'check_render',[(11.75,5.5),(11.75,6.7)],'检查通过',(12.4,6.1))
    arrow(ax,'render_record',[(9.95,7.375),(8.8,7.375)],'mask 非空',(9.37,7.08))
    arrow(ax,'record_next',[(5.2,7.375),(4.05,7.375)],'本件结束',(4.62,7.08))
    arrow(ax,'next_cad',[(2.25,6.7),(2.25,6.07),(7,6.07),(7,5.5)],'本机位还有 CAD',(6.6,6.07))
    arrow(ax,'next_view',[(.45,7.375),(.15,7.375),(.15,4.825),(.45,4.825)],'还有机位',(1.13,5.95))
    node(ax,'failed',9.95,9.05,3.6,1.1,'stop','记录失败','逐件异常捕获；不重试','错误原因 / 保留部分产物')
    arrow(ax,'failure',[(13.55,4.825),(13.85,4.825),(13.85,9.6),(13.55,9.6)],'导入 / 摆放 / 渲染异常',(12.1,8.78),ROLE['stop'][1])
    arrow(ax,'failure_cleanup',[(11.75,9.05),(11.75,8.42),(7,8.42),(7,8.05)],'失败也清理',(8.4,8.42),ROLE['stop'][1])
    node(ax,'summary',.45,9.05,3.6,1.1,'host','汇总结果','退出码 + 完成数 / 计划数','completed / incomplete')
    arrow(ax,'finished',[(2.25,8.05),(2.25,9.05)],'没有剩余任务',(3.21,8.6))
    node(ax,'pending',.45,10.7,13.1,1.15,'pending','全点位拍摄',
         '待改：52,556 ID × 5 点位；按 CAD 固定尺寸；任务键改为 (CAD, 点位)',
         '待改：按点位命名文件；取消按摘要丢编号；256 samples 尚未设为默认')
    footer(ax,12.17,[
        '现默认 64 samples；RGB、mask、depth 同次渲染。OIDN 为冻结去噪器；Cycles / 几何算法本身无学习参数。',
        '打开机位失败会中止 Blender，其余计 not_run；无自动补拍。原三角形不改，着色法线变化不等于精确 CAD 法线。',
    ])


def retouch():
    ax=page('04-retouch','Step 4 · 整图调色',
            '选用 step4/retouch 分支；主循环在 workflow.py，模型在 model.py。不是物理重光照，也不调用 FLUX。',12.9)
    defs=[
        ('input',.45,1.9,'host','读取清单','512² sRGB / SHA256 / 快照','输入图 I / request.json',False),
        ('load',5.2,1.9,'host','加载权重','官方 sRGB 权重 [4] / CPU','CNN 参数 + 3 张基础 LUT',False),
        ('predict',9.95,1.9,'model','预测配比','256² 缩略图 → 小 CNN [4]','每图 3 个配比 w(I)',True),
        ('basis',.45,4.3,'model','基础色表','3 张 33³ RGB 色表 [4]','固定基础 LUT 参数',True),
        ('fuse',5.2,4.3,'host','合成色表','按 w(I) 线性组合','每图自适应色表 T(I)',False),
        ('lookup',9.95,4.3,'host','逐像素调色','RGB 空间三线性查表','整图调色结果 T(I)[I]',False),
        ('next',.45,6.7,'host','切换输入','先遍历图片，再切权重组','下一图 / 下一组 / 结束',False),
        ('save',5.2,6.7,'host','保存结果','PNG / 配比 / 截断率 / 用时','图片 / results.jsonl',False),
        ('mix',9.95,6.7,'host','原图混合','sRGB 数值混合 / 裁剪量化','J = (1−m) I + m T(I)[I]',False),
    ]
    for key,x,y,role,title,algo,out,frozen in defs:
        node(ax,key,x,y,3.6,1.35,role,title,algo,out,frozen)
    arrow(ax,'load_inputs',[(4.05,2.575),(5.2,2.575)],'清单有效',(4.62,2.29))
    arrow(ax,'infer',[(8.8,2.575),(9.95,2.575)],'本组已加载',(9.37,2.29))
    arrow(ax,'parameters',[(5.8,3.25),(5.8,3.75),(2.25,3.75),(2.25,4.3)],'加载所得参数',(3.95,3.75))
    arrow(ax,'weights_fuse',[(11.75,3.25),(11.75,3.75),(7,3.75),(7,4.3)],'配比已预测',(9.5,3.75))
    arrow(ax,'basis_fuse',[(4.05,4.975),(5.2,4.975)],'基础表齐备',(4.62,4.68))
    arrow(ax,'fuse_lookup',[(8.8,4.975),(9.95,4.975)],'组合完成',(9.37,4.68))
    arrow(ax,'lookup_mix',[(11.75,5.65),(11.75,6.7)],'输出有限',(12.4,5.9))
    arrow(ax,'mix_save',[(9.95,7.375),(8.8,7.375)],'每个 mix',(9.37,7.08))
    arrow(ax,'save_next',[(5.2,7.375),(4.05,7.375)],'两档均保存',(4.62,7.08))
    arrow(ax,'next_image',[(2.25,6.7),(2.25,6.15),(13.85,6.15),(13.85,2.575),(13.55,2.575)],'本组仍有输入图',(7,6.15))
    # White clearance marks a crossing, not a junction between the two loops.
    ax.plot([11.75,11.75],[6.03,6.27],color='white',lw=5,zorder=2.5)
    ax.plot([11.75,11.75],[6.03,6.27],color=LINE,lw=1.1,zorder=2.6)
    arrow(ax,'next_variant',[(.45,7.375),(.15,7.375),(.15,3.53),(7,3.53),(7,3.25)],'本组结束，仍有权重组',(2.6,3.53))
    node(ax,'summary',.45,9.05,3.6,1.1,'host','汇总对照','所有图和权重组结束','对照图 / summary.json')
    arrow(ax,'all_done',[(2.25,8.05),(2.25,9.05)],'全部循环结束',(3.26,8.6))
    node(ax,'abort',9.95,9.05,3.6,1.1,'stop','中止运行','摘要 / 尺寸 / 非有限异常','保留已有文件；无自动重试')
    label(ax,8.5,9.6,'任一步校验异常 →',10.8,ROLE['stop'][1])
    node(ax,'selected',.45,10.7,13.1,1.15,'pending','仅留选定版',
         '当前入口：paired / unpaired × mix 0.5 / 1.0，共四版；用户选定：paired × 0.5',
         'CNN 与基础 LUT 冻结；每张图预测的配比、组合色表会变化。全图变色，XY 像素坐标不变。')
    footer(ax,12.17,[
        '[4] Zeng et al., Image-Adaptive-3DLUT；HuiZeng/Image-Adaptive-3DLUT；来源修订 b491f6df64a5。',
        '本地适配：PyTorch grid_sample 实现 RGB 三线性插值。无逐样本异常跳过；中止时最终汇总可能尚未生成。',
    ])


def save_source_map():
    """Bind diagram states to existing symbols without importing Blender code."""
    groups = {
        '01-scene': [
            ('inputs', 'room_workflow.py', ['generate_room']),
            ('assets', 'room_workflow.py', ['_stage_assets']),
            ('create repair budget ready incomplete', 'room_workflow.py', ['_execute_locked', '_finish']),
            ('build render', 'room_workflow.py', ['preview_room', '_check_template_assets']),
            ('build', 'room_tools.py', ['build', 'dependencies']),
            ('render', 'room_tools.py', ['render']),
            ('inspect', 'room_workflow.py', ['_inspect']),
            ('inner_model inner_tool inner_result', 'call_llm.py', ['_codex_command', 'run_codex_turn']),
        ],
        '02-placement': [
            ('describe', 'step2/scene.py', ['describe_scene']),
            ('regions propose inspect decide keep reject', 'step2/workflow.py', ['run_workflow', 'ask_agent']),
            ('support', 'step2/scene.py', ['prepare_support_scene']),
            ('measure', 'step2/scene.py', ['measure_region']),
            ('preview geometry photo', 'step2/preview.py', ['preview_placement']),
            ('move', 'step2/scene.py', ['validate_moves', 'apply_moves']),
            ('geometry', 'step2/spatial.py', ['SceneGeometry', 'check_framing']),
            ('photo', 'step2/support.py', ['check_rendered_support']),
            ('export', 'step2/workflow.py', ['export_placements']),
        ],
        '03-capture': [
            ('sample', 'step3/workflow.py', ['read_candidates', 'prepare_request']),
            ('assign', 'step3/workflow.py', ['prepare_request']),
            ('request', 'step3/workflow.py', ['save_request', 'launch_blender']),
            ('open', 'step3/capture.py', ['prepare_placement']),
            ('place', 'step3/part.py', ['load_part', 'place_part', 'make_material']),
            ('check', 'step3/part.py', ['check_support_contact']),
            ('check', 'step3/capture.py', ['check_part_framing']),
            ('render', 'step3/capture.py', ['add_geometry_outputs', 'take_photo']),
            ('next cleanup failed', 'step3/capture.py', ['capture_batch']),
            ('summary', 'step3/workflow.py', ['summarize_results']),
        ],
        '04-retouch': [
            ('input load mix save next abort', 'step4/retouch/workflow.py', ['run_workflow']),
            ('load predict basis fuse', 'step4/retouch/model.py', ['Classifier', 'AdaptiveLUT']),
            ('lookup', 'step4/retouch/model.py', ['apply_lut']),
            ('summary', 'step4/retouch/workflow.py', ['run_workflow', 'make_comparisons']),
        ],
    }
    files, states = {}, []
    for figure, entries in groups.items():
        for keys, relative, names in entries:
            path = REPO / 'packages/cad_scene_factory' / relative
            raw = path.read_bytes()
            definitions = {item.name: item for item in ast.parse(raw).body
                           if isinstance(item, (ast.FunctionDef, ast.ClassDef))}
            files[relative] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
            symbols = [{'name': name, 'line': definitions[name].lineno,
                        'end_line': definitions[name].end_lineno} for name in names]
            for key in keys.split():
                assert any(n['figure'] == figure and n['id'] == key for n in NODES)
                states.append({'figure': figure, 'state': key, 'file': relative, 'symbols': symbols})
    payload = {
        'date': '2026-10-29', 'worktree': str(REPO),
        'git_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'provenance': 'Audited working-tree files, including uncommitted changes; SHA256 identifies the read version.',
        'scope': 'Four current selected stages; pending nodes have no implemented state. Historical branches are only inventoried in the overview.',
        'agent_boundary': 'call_llm.py starts the Codex CLI. Its internal tool loop is managed outside this repository; the inset is conceptual, not a local Python implementation.',
        'files': files, 'states': states,
    }
    (ROOT/'source_map.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')


def save():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with PdfPages(OUTPUT/'workflow-atlas.pdf') as pdf:
        for stem,fig in PAGES:
            for suffix in ('png','svg','pdf'):
                fig.savefig(OUTPUT/f'{stem}.{suffix}',dpi=190,facecolor='white')
            pdf.savefig(fig,facecolor='white')
            plt.close(fig)
    (ROOT/'figure_manifest.json').write_text(json.dumps({
        'date':'2026-10-29','scope':'current source with explicitly marked pending contract',
        'nodes':NODES,'transitions':EDGES,
        'freeze_semantics':'No training in this pipeline; managed LLM checkpoint is not pinned locally.',
        'references':[
            {'id':1,'title':'Codex non-interactive mode','url':'https://learn.chatgpt.com/docs/non-interactive-mode'},
            {'id':2,'title':'Blender Cycles','url':'https://www.blender.org/features/rendering/'},
            {'id':3,'title':'Intel Open Image Denoise','url':'https://www.openimagedenoise.org/'},
            {'id':4,'title':'Image-Adaptive-3DLUT','url':'https://github.com/HuiZeng/Image-Adaptive-3DLUT',
             'paper':'https://arxiv.org/abs/2009.14468','source_revision':'b491f6df64a588864739a157db271e5c848e1805'},
        ],
    },ensure_ascii=False,indent=2)+'\n')
    save_source_map()


if __name__=='__main__':
    overview()
    scene()
    placement()
    capture()
    retouch()
    save()
    print(OUTPUT)
