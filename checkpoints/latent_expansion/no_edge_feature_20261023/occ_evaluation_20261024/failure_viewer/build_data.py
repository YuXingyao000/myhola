"""Pack the fixed validation failure census without changing the OCC evaluation."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
REPO = HERE.parents[4]
GT = Path('/mnt/d/data/deepcad_v7')
SPLIT = REPO / 'checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt'
RAW = EVAL / 'raw_predictions/posterior_mean/no_edge_feature64'
CACHE = EVAL / 'failure_analysis/raw/no_edge_feature64'
SIGNATURE = '704227ad75de74bd531ae8641c4060818dcfac02089bf6f43053eb95b9c4c77a'


def read_json(path: Path):
    return json.loads(path.read_text())


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def encode_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def maximum(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(max(values)) if values else None


def describe(values):
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return {'intended': len(values), 'measured': len(finite), 'missing': len(values) - len(finite),
            'mean': mean(finite), 'median': float(np.median(finite)) if finite else None,
            'p90': float(np.quantile(finite, .9)) if finite else None,
            'p95': float(np.quantile(finite, .95)) if finite else None,
            'max': maximum(finite)}


def raw_metrics(record):
    owner = record.get('owner', {})
    endpoints = record.get('endpoints', {})
    return {'raw_mean': mean(r.get('mean') for r in owner.get('records', [])),
            'raw_p95': owner.get('p95_max'), 'raw_max': owner.get('max'),
            'endpoint_max': endpoints.get('max'),
            'endpoint_mean': mean(r.get('distance') for r in endpoints.get('records', [])),
            'endpoint_unknown': sum(r.get('distance') is None for r in endpoints.get('records', [])),
            'endpoint_empty_owner_faces': endpoints.get('unknown_count', 0) - sum(
                r.get('basis') == 'GT_pair_unknown' for r in endpoints.get('records', []))}


def stage_of(diagnostics):
    last = diagnostics['paths'][-1]
    if last['status'] == 'incomplete_faces':
        return 'trimming'
    if last['status'] == 'no_accepted_solid':
        return ('roundtrip_boundary' if any(a.get('status') in ('native_rejected', 'boundary_rejected')
                for a in last.get('solid_attempts', [])) else 'solid_construction')
    raise ValueError(f"Unexpected terminal path: {last['status']}")


def failure_reason(face):
    if not face['input_edge_ids']:
        return 'no_input_edges'
    if not face['attempts']:
        return 'no_attempt_recorded'
    last = face['attempts'][-1]
    return last.get('rejection') or last['status']


def display_array(value):
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError('Non-finite display coordinates')
    return np.round(value.astype(np.float64), 6).tolist()


def geometry_view(path, label, truth=False):
    with np.load(path, allow_pickle=False) as data:
        faces, edges, conn = (data[k] for k in (
            ('sample_points_faces', 'sample_points_lines', 'edge_face_connectivity') if truth else
            ('pred_face', 'pred_edge', 'pred_edge_face_connectivity')))
        if faces.ndim != 4 or edges.ndim != 3 or conn.ndim != 2 or conn.shape[1] != 3:
            raise ValueError(f'Invalid geometry shapes: {path}')
        if len(conn) and (np.min(conn) < 0 or np.max(conn[:, 0]) >= len(edges) or np.max(conn[:, 1:]) >= len(faces)):
            raise ValueError(f'Invalid connectivity: {path}')
        return {'label': label, 'faces': display_array(faces[..., :3]),
                'edges': display_array(edges[..., :3]), 'connectivity': conn.astype(int).tolist()}, edges[..., :3].astype(np.float64)


def add_endpoint_neighbors(records, edges, connectivity):
    # Preserve the existing GT-assisted raw diagnostic, including unknowns.
    entries = {}
    pairs = {}
    for edge, first, second in connectivity:
        pair = tuple(sorted((first, second)))
        pairs[edge] = pair
        if first != second:
            entries.setdefault(first, []).append((edge, pair))
    output = []
    for record in records:
        item = dict(record, neighbor_edge=None, neighbor_endpoint=None)
        edge, endpoint = record['edge'], record['endpoint']
        if record['basis'] == 'GT_closed_own_gap':
            item.update(neighbor_edge=edge, neighbor_endpoint=-1 if endpoint == 0 else 0)
        elif record['basis'] == 'other_pair_endpoint':
            choices = [(other, side) for other, pair in entries.get(record['owner'], [])
                       if pair != pairs[edge] for side in (0, -1)]
            if choices:
                distances = [np.linalg.norm(edges[edge, endpoint] - edges[other, side]) for other, side in choices]
                index = int(np.argmin(distances))
                if not np.isclose(distances[index], record['distance'], rtol=1e-10, atol=1e-12):
                    raise ValueError('Endpoint cache differs from raw geometry')
                item.update(neighbor_edge=choices[index][0], neighbor_endpoint=choices[index][1])
        output.append(item)
    return output


def prepare_base():
    source_summary = read_json(EVAL / 'failure_analysis/summary.json')
    groups = source_summary['paired']['ids']
    failures = sorted(groups['regressed'] + groups['both_failed'])
    population = SPLIT.read_text().split()
    if len(failures) != len(set(failures)) or len(failures) != 316 or len(population) != 3495:
        raise ValueError('Fixed population does not match')
    metrics_all, reason_counts, stages, source_hashes = {}, Counter(), Counter(), {}
    failure_set = set(failures)
    for index, sid in enumerate(population):
        cache_path = CACHE / f'{sid}.json'
        record = read_json(cache_path)
        if record.get('protocol_fingerprint') != SIGNATURE or record.get('status') != 'ok':
            raise ValueError(f'Unexpected raw cache: {sid}')
        metrics_all[sid] = raw_metrics(record)
        if sid not in failure_set:
            continue
        raw_path, gt_path = RAW / sid / 'data.npz', GT / sid / 'data.npz'
        for key, path in [('raw', raw_path), ('gt', gt_path)]:
            if digest(path) != record['sources'][key]['sha256']:
                raise ValueError(f'Cached source changed: {sid}/{key}')
        diag_path = EVAL / f'post/optimized/no_edge_feature64/{sid}/post_diagnostics.json'
        diagnostics = read_json(diag_path)
        meta = read_json(EVAL / f'metadata/post/optimized/samples/no_edge_feature64/{sid}.json')
        if meta['native_valid_solid']:
            raise ValueError(f'Failure list contains valid sample: {sid}')
        if digest(diag_path) != meta['post_diagnostics_sha256']:
            raise ValueError(f'Historical diagnostics changed: {sid}')
        source_hashes[sid] = {**record['sources'], 'diagnostics_sha256': digest(diag_path), 'cache_sha256': digest(cache_path)}
        stage = stage_of(diagnostics)
        stages[stage] += 1
        failed_faces = [face for face in diagnostics['paths'][-1]['faces'] if face['accepted_attempt'] is None]
        reasons = [failure_reason(face) for face in failed_faces]
        reason_counts.update(reasons)
        raw, edges = geometry_view(raw_path, '原始预测 · 优化前16点/16×16网格')
        raw['owner_records'] = record['owner']['records']
        raw['endpoint_records'] = add_endpoint_neighbors(record['endpoints']['records'], edges, raw['connectivity'])
        raw['endpoint_unknown'] = metrics_all[sid]['endpoint_unknown']
        raw['endpoint_empty_owner_faces'] = metrics_all[sid]['endpoint_empty_owner_faces']
        raw['metric_note'] = '边到16×16三角网格距离；端点使用GT闭合类型作诊断，不是生产wire配对。'
        gt, _ = geometry_view(gt_path, 'GT采样参照 · 仅诊断，不参与修复', truth=True)
        gt['owner_records'] = record['gt_owner']['records']
        gt['endpoint_records'] = []
        gt['metric_note'] = 'GT曲面/曲线的离散采样网格；非零边面距离也可能来自采样近似。'
        case = {'id': sid, 'stage': stage, 'cohort': 'regressed' if sid in groups['regressed'] else 'both_failed',
                'failed_faces': [face['face_index'] for face in failed_faces], 'reasons': reasons,
                'diagnostics': diagnostics, 'views': {'raw': raw, 'gt': gt},
                'metrics': metrics_all[sid], 'replay': {'status': 'not_loaded', 'matches_history': False},
                'structure': record['structure']}
        out = HERE / f'data/base/{sid}.json.gz'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(gzip.compress(encode_json(case), compresslevel=6, mtime=0))
    if dict(stages) != {'trimming': 307, 'solid_construction': 6, 'roundtrip_boundary': 3}:
        raise ValueError(f'Unexpected stage census: {stages}')
    if sum(reason_counts.values()) != 767:
        raise ValueError('Failure face census differs')
    write_json(HERE / 'data/base_manifest.json', {
        'ids': failures, 'population': population, 'metrics_all': metrics_all, 'stages': dict(stages),
        'reasons': dict(reason_counts), 'source_hashes': source_hashes,
        'summary_source_sha256': digest(EVAL / 'failure_analysis/summary.json'), 'split_sha256': digest(SPLIT),
        'checkpoint_sha256': source_summary['checkpoint_sha256']})
    print(json.dumps({'base_cases': len(failures), 'stages': dict(stages), 'reasons': dict(reason_counts)}, ensure_ascii=False))


def pack(require_replay=True):
    base = read_json(HERE / 'data/base_manifest.json')
    cases, encoded, replay_counts, failed_face_count = [], {}, Counter(), 0
    projection_unknown = projection_total = 0
    projection_recomputed_error = 0.0
    for sid in base['ids']:
        case = json.loads(gzip.decompress((HERE / f'data/base/{sid}.json.gz').read_bytes()))
        replay_path = HERE / f'replay/{sid}.json.gz'
        if not replay_path.exists():
            if require_replay:
                raise FileNotFoundError(replay_path)
            replay = {'status': 'not_loaded', 'matches_history': False}
        else:
            replay = json.loads(gzip.decompress(replay_path.read_bytes()))
        case['replay'] = {k: v for k, v in replay.items() if k not in ('view', 'report', 'optimized_geometry')}
        replay_counts[replay.get('status', 'unknown')] += 1
        fitted = replay.get('view')
        if fitted:
            fitted = {k: v for k, v in fitted.items() if k not in ('optimized_face_points', 'optimized_edge_points')}
            fitted['endpoint_other_edge_records'] = fitted['endpoint_records']
            candidates = []
            for record in fitted['endpoint_records']:
                edge = fitted['edges'][record['edge']]
                own_gap = float(np.linalg.norm(np.asarray(edge[0]) - edge[-1]))
                candidate = dict(record, own_seam_distance=own_gap,
                                 original_neighbor_distance=record['distance'],
                                 original_neighbor_edge=record['neighbor_edge'],
                                 original_neighbor_endpoint=record['neighbor_endpoint'])
                if record['distance'] is None or own_gap < record['distance']:
                    candidate.update(distance=own_gap, neighbor_edge=record['edge'],
                                     neighbor_endpoint=-1 if record['endpoint'] == 0 else 0,
                                     basis='own_opposite_endpoint_candidate; nearest_local_connection_not_wire_proof')
                candidates.append(candidate)
            fitted['endpoint_records'] = candidates
            fitted['endpoint_metric_note'] = '最近连接候选包含同曲线首尾；仅局部下界，可能是退化自闭合，不证明正确wire配对。原不同边近邻距离同时保留。'
            case['views']['fitted'] = fitted
            owners = fitted.get('owner_records', [])
            measured = replay['metrics']
            case['metrics'].update(replay_mean=measured['owner_mean'],
                                   replay_p95=measured['owner_p95_max'],
                                   replay_max=measured['owner_max'],
                                   replay_endpoint_mean=mean(r['distance'] for r in candidates),
                                   replay_endpoint_max=maximum(r['distance'] for r in candidates),
                                   replay_endpoint_other_mean=measured['endpoint_mean'],
                                   replay_endpoint_other_max=measured['endpoint_max'],
                                   replay_endpoint_other_unknown=measured['unknown_endpoint_count'])
            for rec in owners:
                projection_unknown += rec.get('unknown_points', 0)
                projection_total += rec.get('valid_points', 0) + rec.get('unknown_points', 0)
                for point, closest, distance in zip(fitted['edges'][rec['edge']], rec.get('closest', []), rec.get('distances', [])):
                    if closest is not None and distance is not None:
                        error = abs(np.linalg.norm(np.asarray(point) - closest) - distance)
                        projection_recomputed_error = max(projection_recomputed_error, float(error))
            # Preserve measurement precision; round only display positions.
            fitted['faces'] = display_array(fitted['faces'])
            fitted['edges'] = [display_array(edge) for edge in fitted['edges']]
            for rec in owners:
                if 'closest' in rec:
                    rec['closest'] = [display_array(point) if point is not None else None for point in rec['closest']]
        case['metrics'].setdefault('replay_mean', None)
        case['metrics'].setdefault('replay_p95', None)
        case['metrics'].setdefault('replay_max', None)
        case['metrics'].setdefault('replay_endpoint_mean', None)
        case['metrics'].setdefault('replay_endpoint_max', None)
        case['metrics'].setdefault('replay_endpoint_other_mean', None)
        case['metrics'].setdefault('replay_endpoint_other_max', None)
        case['metrics'].setdefault('replay_endpoint_other_unknown', None)
        row = {k: case[k] for k in ('id', 'stage', 'cohort', 'failed_faces', 'reasons')}
        row.update(case['metrics'], replay_match=replay.get('matches_history', False))
        cases.append(row)
        failed_face_count += len(row['failed_faces'])
        blob = gzip.compress(encode_json(case), compresslevel=6, mtime=0)
        out = HERE / f'data/cases/{sid}.json.gz'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blob)
        encoded[sid] = base64.b64encode(blob).decode()
    distributions = {}
    failed_ids = set(base['ids'])
    for metric in ('raw_mean', 'raw_p95', 'raw_max', 'endpoint_mean', 'endpoint_max', 'replay_mean', 'replay_p95', 'replay_max', 'replay_endpoint_mean', 'replay_endpoint_max', 'replay_endpoint_other_mean', 'replay_endpoint_other_max'):
        if metric.startswith('replay_'):
            values = {'valid': [None] * 3179, 'failed': [row[metric] for row in cases]}
        else:
            values = {label: [row[metric] for sid, row in base['metrics_all'].items() if (sid in failed_ids) == is_failed]
                      for label, is_failed in [('valid', False), ('failed', True)]}
        distributions[metric] = {}
        if metric.startswith('replay_'):
            distributions[metric]['not_measured_valid'] = 3179
        for label, array in values.items():
            distributions[metric][label] = [v for v in array if v is not None]
            distributions[metric]['unknown_' + label] = sum(v is None for v in array)
    summary = {'population': 3495, 'valid': 3179, 'failed': 316, 'stages': base['stages'],
               'failed_faces': failed_face_count, 'reasons': base['reasons'], 'regressed': 29, 'both_failed': 287,
               'replay_status_counts': dict(replay_counts), 'replay_match': sum(row['replay_match'] for row in cases),
               'projection_sample_count': projection_total, 'projection_unknown_count': projection_unknown,
               'projection_connector_max_numeric_error_before_display_rounding': projection_recomputed_error,
               'distributions': {metric: {label: describe(distributions[metric][label] + [None] * distributions[metric]['unknown_' + label])
                                         for label in ('valid', 'failed')} for metric in distributions},
               'raw_endpoint_partial_unknown_objects': {label: sum(row['endpoint_unknown'] > 0 for sid, row in base['metrics_all'].items()
                    if (sid in failed_ids) == failed) for label, failed in [('valid', False), ('failed', True)]},
               'provenance': {key: base[key] for key in ('summary_source_sha256', 'split_sha256', 'checkpoint_sha256')}}
    manifest = {'version': 1, 'title': 'VAE · 裁剪失败观察室', 'summary': summary, 'cases': cases,
                'distributions': distributions, 'definitions': {
                    'units': '归一化坐标中的L2长度；不是毫米，不是平方Chamfer。',
                    'raw_p95': '每条边-owner的16点到网格距离P95，再取对象内最大值。',
                    'replay_p95': '每条拟合边-owner的64参数点到有界解析面的距离P95，再取对象内最大值。',
                    'mean': '对象内边-owner均值等权平均；跨对象统计按CAD等权。',
                    'endpoint': '局部最近端点代理，不证明全局wire可闭合；raw闭合类型使用GT诊断。',
                    'replay_endpoint_mean': '含同曲线首尾的最近连接候选距离均值；局部下界，不能证明wire闭合。不同边原代理另保留。',
                    'replay_endpoint_max': '含同曲线首尾的最近连接候选距离最大值；局部下界，不能证明wire闭合。不同边原代理另保留。',
                    'replay_endpoint_other_mean': '原不同边最近端点代理均值，严格闭合曲线使用自身首尾；近闭合整圈可能误连远处另一个环。未知端点保留，数值仅为已测部分。',
                    'replay_endpoint_other_max': '原不同边最近端点代理最大值，严格闭合曲线使用自身首尾；近闭合整圈可能误连远处另一个环。未知端点保留，数值仅为已测部分。',
                    'replay': '独立重放最终恢复路径；仅日志严格一致时可作为历史行为的可复现观察。',
                    'tolerance': '原OCC已尝试8档容差0.002–0.08；分布查看阈值不是合法性门槛。'}}
    write_json(HERE / 'data/manifest.json', manifest)
    write_json(HERE / 'summary.json', summary)
    with (HERE / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()} for row in cases)
    template = (HERE / 'template.html').read_text()
    app = (HERE / 'app.js').read_text()
    import plotly
    js_path = Path(plotly.__file__).parent / 'package_data/plotly.min.js'
    font_style = ''
    font_path = Path('/root/.cache/hola-report-fonts/SourceHanSansSC-Regular.otf')
    if font_path.exists():
        from fontTools import subset
        options = subset.Options()
        options.flavor = 'woff2'
        font = subset.load_font(str(font_path), options)
        subsetter = subset.Subsetter(options=options)
        subsetter.populate(text=template + app + json.dumps(manifest, ensure_ascii=False))
        subsetter.subset(font)
        buffer = io.BytesIO()
        font.flavor = 'woff2'
        font.save(buffer)
        font_style = '@font-face{font-family:SourceHanSans;src:url(data:font/woff2;base64,' + base64.b64encode(buffer.getvalue()).decode() + ') format("woff2");font-display:swap;}'
    substitutions = {'__PLOTLY_JS__': js_path.read_text().replace('</script', '<\\/script'),
                     '__FONT_STYLE__': font_style,
                     '__MANIFEST_JSON__': encode_json(manifest).decode().replace('<', '\\u003c'),
                     '__CASE_DATA_JSON__': json.dumps(encoded, separators=(',', ':')),
                     '__APP_JS__': app}
    # One-pass substitution prevents generated content from being interpreted as a placeholder.
    html = re.sub('|'.join(map(re.escape, substitutions)), lambda m: substitutions[m.group()], template)
    output = HERE / 'failure_viewer.html'
    output.write_text(html)
    write_json(HERE / 'validation/build.json', {'html_bytes': output.stat().st_size, 'html_sha256': digest(output),
               'case_count': len(cases), 'replay_match': summary['replay_match'], 'plotly_version': plotly.__version__,
               'plotly_sha256': digest(js_path), 'sources': {name: digest(HERE / name) for name in ('SPEC.md', 'SCHEMA.md', 'build_data.py', 'app.js', 'template.html')},
               'require_replay': require_replay, 'argv': sys.argv})
    print(json.dumps({'html': str(output), 'bytes': output.stat().st_size, 'replay_match': summary['replay_match'], 'projection_unknown': projection_unknown}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['base', 'pack'])
    parser.add_argument('--allow-missing-replay', action='store_true', help='UI development only; not a complete deliverable.')
    args = parser.parse_args()
    prepare_base() if args.command == 'base' else pack(not args.allow_missing_replay)
