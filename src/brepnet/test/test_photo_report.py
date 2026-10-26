"""Evaluation summaries retain failures and compare the same CAD populations."""

from src.brepnet.eval.photo_report import METHODS, paired_delta, summarize


def test_geometry_missing_is_not_zero_and_validity_keeps_full_population():
    rows = {}
    for method, valid_count in zip(METHODS, (60, 50, 70)):
        records = []
        for index in range(100):
            valid = index < valid_count
            row = {"image_id": str(index), "inference_status": "ok", "step_exists": valid,
                   "step_readable": valid, "validity": "valid" if valid else "unknown",
                   "evaluation_status": "ok" if valid else "timeout", "post_complete": valid,
                   "inference_seconds": 5., "raw_face_count_abs_error": 2.}
            for metric in ("face_cd_sq", "edge_cd_sq", "face_f1"):
                row['raw_'+metric] = 1. if index < 99 else None
                row['post_'+metric] = 1. if valid else None
            records.append(row)
        rows[method] = records
    result = summarize(rows)
    assert result['intended_per_method'] == 100
    assert result['methods'][METHODS[0]]['valid_solids'] == 60
    assert result['methods'][METHODS[0]]['unknown'] == 40
    assert result['methods'][METHODS[0]]['metrics']['raw_face_cd_sq'] == {'n': 99, 'mean': 1., 'median': 1.}
    assert len(result['all_three_common_post_ids']) == 50
    key = f'{METHODS[2]}_minus_{METHODS[0]}'
    assert result['pairwise'][key]['valid']['n'] == 100
    assert result['pairwise'][key]['raw_face_cd_sq']['n'] == 99


def test_empty_paired_geometry_stays_missing():
    assert paired_delta([{'cd': None}], [{'cd': 1.}], 'cd') == {'n': 0, 'mean_delta': None, 'ci95': None}
