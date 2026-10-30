"""Evaluation summaries retain failures and compare the same CAD populations."""

from src.brepnet.eval.photo_report import METHODS, paired_delta, summarize
import numpy as np
import pytest


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


def test_paired_bootstrap_preserves_draws_and_rejects_misalignment():
    left = [{'image_id': str(i), 'cd': 0.} for i in range(7)]
    right = [{'image_id': str(i), 'cd': float(i)} for i in range(7)]
    actual = paired_delta(left, right, 'cd')
    draws = np.random.default_rng(42).integers(7, size=(10000, 7)).mean(1)
    assert actual['ci95'] == np.quantile(draws, [.025, .975]).tolist()
    with pytest.raises(ValueError, match='order'):
        paired_delta(left, right[::-1], 'cd')
    with pytest.raises(ValueError, match='size'):
        paired_delta(left, right[:-1], 'cd')


def test_full_population_denominator_is_inferred():
    rows = {method: [] for method in METHODS}
    for method in METHODS:
        for i in range(3):
            row = {'image_id': str(i), 'inference_status': 'ok', 'evaluation_status': 'timeout',
                   'step_exists': False, 'step_readable': False, 'validity': 'unknown',
                   'post_complete': False, 'inference_seconds': 5., 'raw_face_count_abs_error': None}
            row.update({f'{stage}_{metric}': None for stage in ('raw', 'post')
                        for metric in ('face_cd_sq', 'edge_cd_sq', 'face_f1')})
            rows[method].append(row)
    result = summarize(rows)
    assert result['intended_per_method'] == 3
    assert result['methods'][METHODS[0]]['unknown'] == 3
    assert result['methods'][METHODS[0]]['metrics']['post_face_cd_sq']['n'] == 0
    rows[METHODS[-1]].reverse()
    with pytest.raises(ValueError, match='ordered CAD'):
        summarize(rows)
