"""Read one interim evaluation snapshot without changing the active run."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from src.brepnet.eval.preference_report import (
    EVALUATION_TERMINAL, METHODS, POST_METRICS, VALIDITIES,
    finite_number, paired_statistics, statistics,
)


ROOT = Path('/mnt/d/python/checkpoints/diffusion/preference_eval/geometry_v1_evaluation/comparison')
OUTPUT = Path(__file__).resolve().parent


def read_record(path: Path) -> tuple[dict, str | None]:
    if not path.exists():
        return {}, None
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    try:
        return json.loads(data), digest
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return {'read_error': str(error)}, digest


def main() -> None:
    # Preserve the first snapshot rather than silently replacing its numbers.
    if (OUTPUT / 'snapshot.json').exists():
        raise FileExistsError('This snapshot already exists; use a new output directory.')
    started = datetime.now(timezone.utc).isoformat()
    ids = ROOT.joinpath('validation.txt').read_text().split()
    protocol, protocol_sha = read_record(ROOT / 'metadata/protocol.json')
    assert len(ids) == len(set(ids)) == 3505
    assert protocol['identities'] == ids
    rows = {method: {} for method in METHODS}
    for image_id in ids:
        for method in METHODS:
            inference, inference_sha = read_record(ROOT / 'raw_predictions' / method / image_id / 'inference.json')
            evaluation, evaluation_sha = read_record(ROOT / 'evaluations' / method / image_id / 'metrics.json')
            terminal = evaluation.get('status') in EVALUATION_TERMINAL
            validity = evaluation.get('validity', 'unknown') if terminal else 'pending'
            assert validity in (*VALIDITIES, 'pending')
            raw = evaluation.get('raw', {}) if terminal else {}
            row = {
                'inference_status': inference.get('status', 'missing'),
                'seed': inference.get('seed'),
                'evaluation_status': evaluation.get('status', 'missing'),
                'terminal': terminal, 'validity': validity,
                'inference_sha256': inference_sha, 'evaluation_sha256': evaluation_sha,
                'error': evaluation.get('read_error') or evaluation.get('error'),
            }
            for metric in POST_METRICS:
                value = raw.get(metric)
                row[f'raw_{metric}'] = value if finite_number(value) else None
            counts = [raw.get(name) for name in ('pred_face_count', 'gt_face_count')]
            row['raw_face_count_abs_error'] = abs(counts[0] - counts[1]) if all(map(finite_number, counts)) else None
            repeats = evaluation.get('post', [])
            complete = (
                validity == 'valid' and len(repeats) == 3
                and {r.get('repeat') for r in repeats} == {0, 1, 2}
                and all(finite_number(r.get(m)) for r in repeats for m in POST_METRICS)
            )
            for metric in POST_METRICS:
                row[f'post_{metric}'] = float(np.mean([r[metric] for r in repeats])) if complete else None
            rows[method][image_id] = row
        left, right = (rows[m][image_id] for m in METHODS)
        assert left['seed'] is not None and left['seed'] == right['seed'], image_id

    # Pending items stay in full-list accounting, outside the interim paired table.
    common = [i for i in ids if all(rows[m][i]['terminal'] for m in METHODS)]
    summary = {
        'status': 'interim_not_final', 'started_utc': started,
        'finished_read_utc': datetime.now(timezone.utc).isoformat(),
        'source': str(ROOT), 'protocol_sha256': protocol_sha,
        'methods': {'baseline': 'continued DPO', 'dpo_5000': 'DPO + geometry'},
        'intended_count_per_method': len(ids), 'paired_terminal_count': len(common),
        'limitations': [
            'Completion order may select easier/faster CADs; this is not full validation.',
            'Bootstrap intervals describe the observed subset, not completion-order bias.',
            'Observation metrics have not completed; no checkpoint selection is made.',
        ],
        'full_list_status': {}, 'paired_validity_counts': {}, 'metrics': {},
    }
    for method in METHODS:
        records = list(rows[method].values())
        counts = dict(Counter(r['validity'] for r in records))
        assert sum(counts.values()) == len(ids)
        summary['full_list_status'][method] = {
            'inference': dict(Counter(r['inference_status'] for r in records)),
            'evaluation': dict(Counter(r['evaluation_status'] for r in records)),
            'validity': counts,
        }
        summary['paired_validity_counts'][method] = dict(Counter(rows[method][i]['validity'] for i in common))

    keys = ['validity', 'raw_face_cd_sq', 'raw_edge_cd_sq', 'raw_face_f1',
            'raw_face_count_abs_error', 'post_face_cd_sq', 'post_edge_cd_sq', 'post_face_f1']
    for key in keys:
        values = {}
        for method in METHODS:
            values[method] = {}
            for image_id in common:
                row = rows[method][image_id]
                value = float(row['validity'] == 'valid') if key == 'validity' else row[key]
                if value is not None:
                    values[method][image_id] = value
        paired = paired_statistics(values[METHODS[0]], values[METHODS[1]])
        summary['metrics'][key] = {
            'paired_common': paired,
            'own_measurable_within_paired_terminal': {
                m: statistics(list(values[m].values())) for m in METHODS
            },
        }
    summary['validity_transitions'] = dict(Counter(
        rows['baseline'][i]['validity'] + '_to_' + rows['dpo_5000'][i]['validity'] for i in common
    ))
    snapshot = {'summary': summary, 'identities': ids, 'paired_terminal_ids': common, 'rows': rows}
    (OUTPUT / 'snapshot.json').write_text(json.dumps(snapshot, ensure_ascii=False, separators=(',', ':')) + '\n')
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

    labels = ['有效 solid ↑ (%)', 'Raw 面 CD² ↓ (×1000)', 'Raw 边 CD² ↓ (×1000)',
              'Raw 面 F1 ↑ (%)', 'Raw 面数 MAE ↓', 'Post 面 CD² ↓ (×1000)',
              'Post 边 CD² ↓ (×1000)', 'Post 面 F1 ↑ (%)']
    scales = [100, 1000, 1000, 100, 1, 1000, 1000, 100]
    lines = [
        '# 几何辅助 loss：中途快照，非最终结果', '',
        f'读取区间（UTC）：{started} 至 {summary["finished_read_utc"]}。', '',
        f'全量每组3505件；双方均达到终态的共同集合为{len(common)}件。完成顺序可能有选择偏差。', '',
        '两组由同一DPO5000初始化，各额外训练5000步。差值为加几何项减控制组；区间为CAD配对bootstrap的95%区间。', '',
        '| 指标 | 配对件数 | 继续DPO，不加几何项 | DPO＋几何项 | 差值 [95%区间] |',
        '|---|---:|---:|---:|---:|',
    ]
    for key, label, scale in zip(keys, labels, scales):
        p = summary['metrics'][key]['paired_common']
        if not p['n']:
            continue
        lo, hi = [v * scale for v in p['ci95']]
        lines.append(f'| {label} | {p["n"]} | {p["baseline_mean"]*scale:.4f} | {p["dpo_5000_mean"]*scale:.4f} | {p["dpo_minus_baseline"]*scale:+.4f} [{lo:+.4f}, {hi:+.4f}] |')
    lines += ['', 'Post只取双方合法且三次几何测量完整的共同集合；不能当作全体输出的误差。未完成项目保留pending，不计为失败。可见一致性尚未完成。', '',
              '全量状态：', '', '```json', json.dumps(summary['full_list_status'], ensure_ascii=False, indent=2), '```', '',
              '源文件摘要及全部3505件状态见snapshot.json；统计口径见SPEC.md。']
    (OUTPUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:20]))
    print(json.dumps(summary['full_list_status'], ensure_ascii=False))
    print(json.dumps(summary['validity_transitions'], ensure_ascii=False))


if __name__ == '__main__':
    main()
