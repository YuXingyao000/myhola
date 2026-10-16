"""Read-only pair evidence for the two predeclared distribution extremes."""
from pathlib import Path
import csv
import importlib.util
import json
import math
import sys

import numpy as np
import torch

from raw_distribution_analysis import ROOT, OUT, SOURCE, digest

spec = importlib.util.spec_from_file_location('historical_pair_helpers', SOURCE / 'outlier_pair_probe.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
from src.brepnet.data.shared_boundaries import build_shared_boundary_targets
from tools.vae_compare_metrics import squared_chamfer

METHODS = ('conditioned', 'point_feedback')
IDS = ('00335203', '00326127')


def main():
    torch.set_num_threads(1)
    output = OUT / 'outlier_pair_probe.json'
    if output.exists() or (OUT / 'OUTLIER_PAIR_PROBE.md').exists():
        raise FileExistsError('Extreme probe already exists')
    csv_path = ROOT / 'raw_metrics/posterior_mean/per_object.csv'
    with csv_path.open() as handle:
        rows = {(r['method'], r['model_id']): r for r in csv.DictReader(handle)}
    sources = {str(p): digest(p) for p in (csv_path, Path(__file__), OUT / 'OUTLIER_PAIR_SPEC.md', SOURCE / 'outlier_pair_probe.py')}
    result = {'sample_ids': IDS, 'methods': METHODS, 'sources_sha256': sources, 'samples': {}}
    lines = ['# 两个极端对象的面对核验', '',
             '预定选择：候选边CD最高对象00335203；平均互反CD正增量最高对象00326127。只读原始预测和GT，不重跑模型、不换权重。', '',
             '下面的距离针对16点曲线与16×16预测面三角网格，使用既有FP64 CPU几何helper；不是光滑CAD最大容差。', '']
    for sid in IDS:
        gt_path = Path('/mnt/d/data/deepcad_v7') / sid / 'data.npz'
        sources[str(gt_path)] = digest(gt_path)
        with np.load(gt_path, allow_pickle=False) as gt:
            targets = build_shared_boundary_targets(gt['sample_points_faces'], gt['sample_points_lines'],
                gt['edge_face_connectivity'], sample_id=sid, reciprocal_tolerance=.01,
                closed_tolerance=1e-5, allow_closed_phase=True)
        lookup = {tuple(map(int, p)): i for i, p in enumerate(targets['face_pairs'])}
        loaded, summaries, pairs = {}, {}, set()
        for method in METHODS:
            path = ROOT / f'raw_predictions/posterior_mean/{method}/{sid}/data.npz'
            sources[str(path)] = digest(path)
            raw, groups, twins = helper.read_raw(path)
            loaded[method] = (raw, groups, twins)
            missing, extra = sorted(set(lookup) - set(groups)), sorted(set(groups) - set(lookup))
            pairs.update(missing)
            pairs.add(max(twins, key=twins.get))
            r = rows[method, sid]
            for metric, value in [('legacy_reciprocal_cd_sq_mean', float(np.mean(list(twins.values())))),
                                  ('legacy_reciprocal_cd_sq_max', max(twins.values()))]:
                if not math.isclose(float(r[metric]), value, rel_tol=1e-10, abs_tol=1e-14):
                    raise ValueError(f'CSV/helper mismatch {sid}/{method}/{metric}')
            summaries[method] = {'missing_gt_pairs': missing, 'extra_gt_pairs': extra,
                **{k: float(r[k]) for k in ('edge_cd_sq', 'edge_pred_to_gt_sq', 'edge_gt_to_pred_sq',
                    'legacy_reciprocal_cd_sq_mean', 'legacy_reciprocal_cd_sq_max')}}
        if str(loaded[METHODS[0]][0]['input_latent_sha256']) != str(loaded[METHODS[1]][0]['input_latent_sha256']):
            raise ValueError('Latent inputs differ')
        evidence = []
        lines += [f'## {sid}', '']
        for method, s in summaries.items():
            lines += [f"{method}：缺失GT面对{s['missing_gt_pairs']}；额外面对{s['extra_gt_pairs']}；边CD²={s['edge_cd_sq']:.7g}，其中预测→GT={s['edge_pred_to_gt_sq']:.7g}，GT→预测={s['edge_gt_to_pred_sq']:.7g}。", '']
        lines += ['| 面对 | 方法 | 半边ID | 到自身GT边CD² | 到任意GT边平均平方距离 | 到两个所属预测面平均距离 |', '|---|---|---:|---:|---:|---|']
        for pair in sorted(pairs):
            target = lookup.get(pair)
            for method, (raw, groups, twins) in loaded.items():
                if pair not in groups:
                    evidence.append({'pair': pair, 'method': method, 'status': 'pair_not_predicted'})
                    lines.append(f'| {pair} | {method} | 缺失 | — | — | — |')
                    continue
                for edge, a, b in groups[pair]:
                    curve = raw['pred_edge'][edge]
                    own = squared_chamfer(curve, targets['curves'][target])[2] if target is not None else None
                    any_gt = squared_chamfer(curve, targets['curves'].reshape(-1, 3))[0]
                    distances = helper.surface_distances(curve, raw['pred_face'][list(pair)])
                    evidence.append({'pair': pair, 'method': method, 'edge_id': edge, 'status': 'ok',
                        'directed_pair': [a, b], 'own_pair_gt_cd_sq': own, 'curve_to_any_gt_mean_sq': any_gt,
                        'twin_cd_sq': twins[pair], 'owner_predicted_surface_distance': distances})
                    text = ', '.join(f"{p}: {v['mean']:.7g}" for p, v in zip(pair, distances))
                    lines.append(f"| {pair} | {method} | {edge} | {format(own, '.7g') if own is not None else '无GT邻接'} | {any_gt:.7g} | {text} |")
        result['samples'][sid] = {'summary': summaries, 'pair_evidence': evidence}
        lines.append('')
    lines += ['整体边CD合并全部边点，不能保证每条边位于自身面对的正确位置；互反一致性又不检查是否遗漏整条边。本诊断不把这两件推广为全集唯一失败机制，也不隔离新增模块的因果作用。', '']
    for p, expected in sources.items():
        if digest(Path(p)) != expected:
            raise ValueError(f'Source changed: {p}')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (OUT / 'OUTLIER_PAIR_PROBE.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
