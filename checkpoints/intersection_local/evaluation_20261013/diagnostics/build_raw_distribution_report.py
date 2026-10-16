"""Render this run's independently computed raw distributions."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT / 'raw_distribution.json').read_text())
    subsets = json.loads((ROOT / 'adjacency_subset.json').read_text())
    edge = data['modes']['posterior_mean']['metrics']['edge_cd_sq']
    tail = edge['tail']['top_1pct']
    exact = subsets['cohorts']['both_adjacency_exact']
    x = exact['metrics']['edge_cd_sq']
    lines = ['# 完整容量版 raw 分布与配对变化', '',
             f"本轮 posterior mean 的边平方 CD 均值为上一版 refinement 的 {edge['mean_ratio']:.3f} 倍，中位数为 {edge['median_ratio']:.3f} 倍；{edge['win_count']}/{edge['common_count']} 件改善（{edge['win_fraction']:.2%}）。尾部放大了均值差距，但不是仅有少量异常件：排除本版误差最大的 {tail['count']} 件后，对同一批剩余 ID 配对比较仍为 {tail['trimmed_mean_ratio']:.3f} 倍。", '',
             '该结论描述当前 checkpoint 的几何表现，不证明架构无效、过拟合或某一唯一根因。全部 2417 件始终是主结果；下列去尾和子集统计仅用于解释。两种 raw 均为2417个唯一、输入latent hash相同的配对ID，独立重算均值与既有summary一致。', '',
             '## 分布与逐件改善', '',
             'CD 是归一化 XYZ 中双向平方距离之和。表中分位数分别属于各方法分布，“均值倍数”是两均值相除，“改善”按同ID数值下降且超过1e-12计算。', '',
             '| 输入 | 指标 | 方法 | 均值 | 中位数 | P90 | P95 | 本版改善 |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    names = {'face_mae': '面XYZ MAE', 'face_cd_sq': '面平方CD', 'edge_cd_sq': '边平方CD'}
    labels = {'intersection': '上一版refinement', 'local': '完整容量local'}
    for mode, record in data['modes'].items():
        for metric, label in names.items():
            row = record['metrics'][metric]
            if not row['common_count']:
                continue
            for method in ('intersection', 'local'):
                s = row[method]
                win = f"{row['win_count']}/{row['common_count']} ({row['win_fraction']:.2%})" if method == 'local' else '—'
                lines.append(f"| {mode} | {label} | {labels[method]} | {s['mean']:.8g} | {s['median']:.8g} | {s['p90']:.8g} | {s['p95']:.8g} | {win} |")
    lines.extend(['', '## 尾部敏感性', '',
                  '| 输入 | 指标 | 去除件数 | 本版总误差份额 | 同ID baseline份额 | 剩余baseline均值 | 剩余local均值 | 剩余均值倍数 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|'])
    for mode, record in data['modes'].items():
        for metric, label in names.items():
            row = record['metrics'][metric]
            if not row['common_count']:
                continue
            t = row['tail']['top_1pct']
            lines.append(f"| {mode} | {label} | {t['count']} | {t['local_error_share']:.2%} | {t['same_ids_baseline_error_share']:.2%} | {t['trimmed_intersection_mean']:.8g} | {t['trimmed_local_mean']:.8g} | {t['trimmed_mean_ratio']:.3f} |")
    lines.extend(['', '始终按local误差排序取ceil(1%×2417)=25件，并从两方法同时移除同一组ID；不是各自去不同尾部，也不用于重新选模型。5%/10%敏感性及被移除ID保存在JSON。', '',
                  '## 两方邻接全对的同ID集合', '',
                  f"共有 {exact['count']} 件满足两方无向面pair集合与GT均相等（FP=FN=0）。其中边平方CD均值 {x['intersection']['mean']:.8g} → {x['local']['mean']:.8g}（{x['local_mean_over_intersection']:.3f}倍），中位数 {x['intersection']['median']:.8g} → {x['local']['median']:.8g}；本版改善 {x['local_improved_count']} 件。当前几何差距不能只由错连的少量对象解释。", '',
                  '该集合只固定无向面邻接的预测正确性，不保证半边方向、环顺序、曲线形状或边面一致性，不能当作拓扑已完全正确的因果消融。另一方至少有邻接错误的集合、全部ID和各几何指标见 [adjacency_subset.json](adjacency_subset.json)。', '',
                  '## 复杂度关联与最大误差ID', '',
                  f"posterior边误差与GT面数的Spearman rho={edge['spearman']['gt_face_count']['local_error']['rho']:.3f}；配对退化量与面数rho={edge['spearman']['gt_face_count']['delta']['rho']:.3f}，关联较弱。不能把面数解释成唯一失败机制。", '',
                  '| GT面数 | 件数 | 边CD均值倍数 | 本版改善比例 |', '|---|---:|---:|---:|'])
    for group in data['modes']['posterior_mean']['complexity_groups']['gt_face_count']:
        if group['count']:
            v = group['metrics']['edge_cd_sq']
            lines.append(f"| {group['range']} | {group['count']} | {v['mean_ratio']:.3f} | {v['win_fraction']:.2%} |")
    lines.extend(['', '| posterior边误差前5 ID | GT面数 | GT边界数 | baseline边CD | local边CD |', '|---|---:|---:|---:|---:|'])
    with (ROOT / 'raw_distribution_extremes.csv').open() as handle:
        extremes = [row for row in csv.DictReader(handle) if row['mode'] == 'posterior_mean' and row['metric'] == 'edge_cd_sq'][:5]
    for row in extremes:
        lines.append(f"| {row['model_id']} | {int(float(row['gt_face_count']))} | {int(float(row['gt_boundary_count']))} | {float(row['intersection_error']):.8g} | {float(row['local_error']):.8g} |")
    a = data['modes']['archived_latents']['metrics']
    lines.extend(['', '## 归档latent的范围', '',
                  f"固定归档diffusion latent下，面CD均值比为{a['face_cd_sq']['mean_ratio']:.4f}，边CD均值比为{a['edge_cd_sq']['mean_ratio']:.4f}；两版总体接近，面/边改善件数分别为{a['face_cd_sq']['win_count']}与{a['edge_cd_sq']['win_count']}。这与posterior重建的相对差异不同，不能把两种输入混为同一质量结论。", '',
                  '这里只检查同一批旧latent的decoder重放，没有重新采样diffusion，没有GT面对应，不报告对应MAE/连接F1，也不证明完整照片条件重建提升。', '',
                  '来源、完整分位数、方向CD与相关统计：[raw_distribution.json](raw_distribution.json)；逐件配对：[CSV](raw_distribution_per_object.csv)；每指标前20件：[extremes CSV](raw_distribution_extremes.csv)。', ''])
    with (ROOT / 'RAW_DISTRIBUTION.md').open('x') as handle:
        handle.write('\n'.join(lines))


if __name__ == '__main__':
    main()
