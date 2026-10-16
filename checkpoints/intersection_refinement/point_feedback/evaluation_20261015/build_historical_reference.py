"""Compare three frozen checkpoints; retain IDs, missing values and provenance."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
ORIGINAL = Path('/mnt/d/python/checkpoints/intersection_refinement/evaluation_20261011')
SPLIT = REPO / 'checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt'
METHODS = ('intersection', 'conditioned', 'point_feedback')
LABELS = {'intersection': '原 Intersection＋refinement', 'conditioned': '曲线条件调制',
          'point_feedback': '本轮逐点几何反馈'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path, method, id_key, ids, sources):
    sources[str(path)] = digest(path)
    with path.open(newline='') as handle:
        selected = [r for r in csv.DictReader(handle) if r['method'] == method]
    result = {r[id_key]: r for r in selected}
    if len(result) != len(selected) or set(result) != set(ids):
        raise ValueError(f'Incomplete/duplicate historical IDs: {method}/{path}')
    return result


def numeric(value):
    if value in (None, ''):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'Invalid geometry metric: {value}')
    return result


def boolean(value):
    if value not in ('True', 'False', 'true', 'false', '1', '0'):
        raise ValueError(f'Unknown validity: {value}')
    return value in ('True', 'true', '1')


def main():
    outputs = [ROOT / 'historical_reference.json', ROOT / 'HISTORICAL_REFERENCE.md']
    if any(p.exists() for p in outputs):
        raise FileExistsError('Historical reference already exists')
    ids = sorted(SPLIT.read_text().split())
    if len(ids) != 2417 or len(set(ids)) != 2417:
        raise ValueError('Frozen split size differs')
    sources = {str(SPLIT): digest(SPLIT), str(Path(__file__)): digest(Path(__file__))}
    if sources[str(SPLIT)] != '2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3':
        raise ValueError('Frozen split SHA differs')
    result = {'methods': METHODS, 'intended_count': 2417, 'original_count': 2424,
              'raw': {}, 'post': {}, 'sources_sha256': sources}
    for mode in ('posterior_mean', 'archived_latents'):
        tables = {m: rows((ORIGINAL if m == 'intersection' else ROOT) / f'raw_metrics/{mode}/per_object.csv',
                          m, 'model_id', ids, sources) for m in METHODS}
        for sid in ids:
            hashes = {tables[m][sid].get('input_latent_sha256') for m in METHODS}
            if len(hashes) != 1 or None in hashes or '' in hashes:
                raise ValueError(f'Historical input differs: {mode}/{sid}')
        metrics = ('face_mae', 'face_cd_sq', 'edge_cd_sq') if mode == 'posterior_mean' else ('face_cd_sq', 'edge_cd_sq')
        result['raw'][mode] = {}
        for metric in metrics:
            values = {m: {sid: numeric(tables[m][sid].get(metric)) for sid in ids} for m in METHODS}
            common = [sid for sid in ids if all(values[m][sid] is not None for m in METHODS)]
            record = {'common_count': len(common), 'missing_count': len(ids) - len(common)}
            for method in METHODS:
                v = [values[method][sid] for sid in common]
                record[method] = {'mean': math.fsum(v) / len(v) if v else None,
                                  'median': statistics.median(v) if v else None}
            result['raw'][mode][metric] = record
    for mode in ('fit_only', 'optimized'):
        protocols = [json.loads((root / f'metadata/post/{mode}/protocol.json').read_text()) for root in (ORIGINAL, ROOT)]
        for key in ('split_sha256', 'timeout_seconds', 'max_optimize_iter', 'adaptive_max_optimize_iter',
                    'drop_num', 'use_cuda', 'threads_per_process', 'post_algorithm_files'):
            if protocols[0].get(key) != protocols[1].get(key):
                raise ValueError(f'Historical post protocol differs: {mode}/{key}')
        tables = {m: rows((ORIGINAL if m == 'intersection' else ROOT) / f'post/{mode}/samples.csv',
                          m, 'sample_id', ids, sources) for m in METHODS}
        valid = {m: {sid: boolean(tables[m][sid]['native_valid_solid']) for sid in ids} for m in METHODS}
        record = {m: {'native_valid_solid': sum(valid[m].values()),
                      'rate_over_accepted': sum(valid[m].values()) / len(ids)} for m in METHODS}
        record['point_feedback_vs_original'] = {
            'rescued_ids': [sid for sid in ids if valid['point_feedback'][sid] and not valid['intersection'][sid]],
            'regressed_ids': [sid for sid in ids if valid['intersection'][sid] and not valid['point_feedback'][sid]],
        }
        result['post'][mode] = record
        for root in (ORIGINAL, ROOT):
            path = root / f'metadata/post/{mode}/protocol.json'
            sources[str(path)] = digest(path)
    for path, expected in sources.items():
        if digest(Path(path)) != expected:
            raise ValueError(f'Historical source changed: {path}')
    text = ['# 三版固定 checkpoint 的历史参考', '',
            '三版输入latent逐ID哈希一致，固定2417件；生产后处理及预算相同。训练曝光不同，不是等预算结构消融。直接配对失败分析仍以conditioned→point_feedback为准。', '',
            '| 方法 | 仅拟合有效solid | 优化后有效solid | 优化后比例 |', '|---|---:|---:|---:|']
    for m in METHODS:
        a, b = result['post']['fit_only'][m], result['post']['optimized'][m]
        text.append(f"| {LABELS[m]} | {a['native_valid_solid']}/2417 | {b['native_valid_solid']}/2417 | {b['rate_over_accepted']:.2%} |")
    text += ['', '| 输入 | 指标 | 三方共同数 | 方法 | 均值 | 中位数 |', '|---|---|---:|---|---:|---:|']
    for mode, metrics in result['raw'].items():
        for metric, record in metrics.items():
            for method in METHODS:
                v = record[method]
                text.append(f"| {mode} | {metric} | {record['common_count']} | {LABELS[method]} | {format(v['mean'], '.8g') if v['mean'] is not None else '不可计算'} | {format(v['median'], '.8g') if v['median'] is not None else '不可计算'} |")
    changes = result['post']['optimized']['point_feedback_vs_original']
    text += ['', f"对最早refinement的optimized直接配对：救回{len(changes['rescued_ids'])}件，退步{len(changes['regressed_ids'])}件。", '',
             'raw CD为归一化坐标双向平方距离；不代表最大边面间隙。归档latent没有重新采样diffusion。7件合同不支持样本仍在原2424总体分母中单列。', '']
    outputs[0].write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    outputs[1].write_text('\n'.join(text))


if __name__ == '__main__':
    main()
