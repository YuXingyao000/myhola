"""Render the completed paired evaluation; do not rerun any CAD processing."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


ROOT = Path('/mnt/d/python/checkpoints/diffusion/preference_eval/geometry_v1_evaluation')
OUT = Path(__file__).resolve().parent
FONT = Path('/root/.cache/hola-browser-libs/fonts')
for name in ('NotoSansCJKsc-Regular.otf', 'NotoSansCJKsc-Bold.otf'):
    font_manager.fontManager.addfont(FONT / name)
plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'axes.unicode_minus': False,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})


def read_json(path):
    return json.loads(path.read_text())


def scientific(value):
    """Show the real order of magnitude rather than scaled column headings."""
    mantissa, exponent = f'{value:.3e}'.split('e')
    return rf'${mantissa}\times10^{{{int(exponent)}}}$'


def main():
    for sub in ('', 'comparison', 'observation'):
        assert read_json(ROOT / sub / 'status.json')['status'] == 'complete'
    source_paths = {s: ROOT / s / 'report/summary.json' for s in ('comparison', 'observation')}
    c, o = (read_json(source_paths[s]) for s in ('comparison', 'observation'))
    assert c['intended_count'] == 3505
    assert 'continued DPO' in c['protocol']['comparison']
    assert c['method_labels'] == {'baseline': 'DPO control', 'dpo_5000': 'DPO + geometry'}
    assert all(v['intended_count'] == 3505 for v in o['methods'].values())

    # The main table covers whole-shape quality and input-view agreement.
    specs = [
        ('生成结果', 'validity', '合法实体率 ↑', 'percent', c['validity_paired']),
        ('后处理前 · 全部 3,505 件', 'raw_face_cd_sq', '面平方 Chamfer ↓', 'distance', c['metrics']['raw_face_cd_sq']['paired_common']),
        ('', 'raw_edge_cd_sq', '边平方 Chamfer ↓', 'distance', c['metrics']['raw_edge_cd_sq']['paired_common']),
        ('', 'raw_face_f1', '面匹配 F1 ↑', 'percent', c['metrics']['raw_face_f1']['paired_common']),
        ('', 'raw_face_count_abs_error', '面数平均绝对误差 ↓', 'count', c['metrics']['raw_face_count_abs_error']['paired_common']),
        ('后处理后 · 双方合法且测量完整的 2,885 件', 'post_face_cd_sq', '面平方 Chamfer ↓', 'distance', c['metrics']['post_face_cd_sq']['paired_common']),
        ('', 'post_edge_cd_sq', '边平方 Chamfer ↓', 'distance', c['metrics']['post_edge_cd_sq']['paired_common']),
        ('', 'post_face_f1', '面匹配 F1 ↑', 'percent', c['metrics']['post_face_f1']['paired_common']),
        ('输入视角 · 共同可测的 3,353 件', 'silhouette_iou', '输入轮廓 IoU ↑', 'percent', o['observation_metrics']['silhouette_iou']['paired_common']),
        ('', 'visible_face_cd_sq', '可见面平方 Chamfer ↓', 'distance', o['observation_metrics']['visible_face_cd_sq']['paired_common']),
        ('', 'depth_mae', '可见深度平均绝对误差 ↓', 'distance', o['observation_metrics']['depth_mae']['paired_common']),
        ('分项诊断 · 同一可见面指标的一个方向', 'visible_face_pred_to_gt_cd_sq', '可见面：预测 → GT 平方距离 ↓', 'distance', o['observation_metrics']['visible_face_pred_to_gt_cd_sq']['paired_common']),
    ]
    # Keep all performance metrics in the export, including those outside the figure.
    all_metrics = {'validity': c['validity_paired']}
    all_metrics.update({k: v['paired_common'] for k, v in c['metrics'].items()})
    all_metrics.update({k: v['paired_common'] for k, v in o['observation_metrics'].items() if not k.startswith('gt_input_')})
    rows = []
    for key, p in all_metrics.items():
        assert p['n'] > 0 and p['bootstrap_repeats'] == 10000
        assert abs(p['dpo_5000_mean'] - p['baseline_mean'] - p['dpo_minus_baseline']) < 1e-12
        rows.append({'metric': key, 'paired_n': p['n'], 'without_geometry': p['baseline_mean'],
                     'with_geometry': p['dpo_5000_mean'], 'difference': p['dpo_minus_baseline'],
                     'ci95_low': p['ci95'][0], 'ci95_high': p['ci95'][1]})
    with (OUT / 'all_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    evidence = {'status': 'complete', 'intended_per_method': 3505,
                'comparison': c['protocol']['comparison'],
                'source_sha256': {s: hashlib.sha256(p.read_bytes()).hexdigest() for s, p in source_paths.items()},
                'checkpoint_sha256': c['checkpoint_sha256'], 'metrics': rows,
                'coverage': {m: v['counts'] for m, v in c['methods'].items()},
                'observation_coverage': {m: v['observation_status_counts'] for m, v in o['methods'].items()}}
    (OUT / 'data.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')

    fig = plt.figure(figsize=(15.2, 13.4), dpi=160, facecolor='#f5f7fa')
    ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    dark, muted, blue, orange = '#192c40', '#617181', '#326da6', '#b96431'
    def txt(x, y, text, size=14, color=dark, weight='normal', ha='left'):
        ax.text(x, y, text, fontsize=size, color=color, weight=weight, ha=ha, va='center')

    txt(.045, .958, '加入几何 loss，是否改善 DPO？', 25, weight='bold')
    txt(.045, .923, '完整 validation · 每组 3,505 件 · 同一旧 DPO5000 起点，各额外训练 5,000 步', 13, muted)
    ax.add_patch(FancyBboxPatch((.04, .844), .92, .054, boxstyle='round,pad=0.008,rounding_size=0.008',
                              facecolor='#e7eef5', edgecolor='none'))
    txt(.055, .872, '主要汇总指标尚无明确收益；合法实体少 18 件，差值区间仍跨零。', 16, weight='bold')
    txt(.055, .812, '指标与方向', 14, muted)
    txt(.423, .812, '不加几何 loss', 15, blue, 'bold', 'center')
    txt(.620, .812, '加入几何 loss', 15, orange, 'bold', 'center')
    txt(.835, .812, '变化  [95% 配对区间]', 14, muted, ha='center')

    y = .777
    for index, (group, key, label, kind, p) in enumerate(specs):
        if group:
            ax.add_patch(plt.Rectangle((.04, y-.013), .92, .027, color='#e8edf2', ec='none'))
            txt(.055, y, group, 11, muted, 'bold'); y -= .033
        diagnostic = key == 'visible_face_pred_to_gt_cd_sq'
        ax.add_patch(plt.Rectangle((.04, y-.023), .92, .045,
                                  color=('#fff0e8' if diagnostic else '#ffffff'), ec='none'))
        txt(.055, y, label, 13)
        for x, field, color in ((.423, 'baseline_mean', blue), (.620, 'dpo_5000_mean', orange)):
            value = p[field]
            display = f'{value*100:.2f}%' if kind == 'percent' else f'{value:.3f}' if kind == 'count' else scientific(value)
            txt(x, y, display, 15, color, ha='center')
        scale = 100 if kind == 'percent' else 100/p['baseline_mean']
        unit = '百分点' if kind == 'percent' else '%'
        delta = p['dpo_minus_baseline'] * scale
        low, high = [v*scale for v in p['ci95']]
        change = f'{delta:+.2f} {unit}  [{low:+.2f}, {high:+.2f}]'
        txt(.835, y, change, 11, '#a34f25' if diagnostic else muted, ha='center')
        ax.plot([.05, .95], [y-.024, y-.024], color='#e4e9ee', lw=.6)
        y -= .043

    txt(.045, .088, '读图：↑ 越高越好；↓ 越低越好。差值＝加入−不加；误差变化以不加组均值为基准。', 10.5, muted)
    txt(.045, .063, '所有均值按同一批 CAD 配对；后处理与可见指标各自缺测保留记录，未填零。配对 bootstrap：10,000 次。', 10.5, muted)
    txt(.045, .038, '浅橙行：单向可见距离出现变差信号，区间未做多重比较校正；不等于全部几何都显著变差。', 10.5, muted)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'comparison.{extension}', dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(json.dumps({'png': str(OUT/'comparison.png'), 'metrics_exported': len(rows),
                      'source_status': 'complete'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
