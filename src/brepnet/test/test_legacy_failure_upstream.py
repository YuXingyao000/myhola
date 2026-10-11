"""False-positive checks for upstream geometric screening, without OCC repair."""
import numpy as np

from tools.legacy_failure_upstream import (
    cross_pair_curves, curve_distance, endpoint_proxy, face_proximity,
    mechanism_flags, reciprocal_diagnostics, stage_deltas, structure,
)


def line(first=(0, 0, 0), second=(1, 0, 0)):
    return np.linspace(first, second, 16)


def planar(z=0.):
    x, y = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16))
    return np.stack([x, y, x * 0 + z], -1)


def test_reciprocal_is_not_duplicate_and_reversal_has_zero_distance():
    curves = np.stack([line(), line()[::-1]])
    conn = np.array([[0, 0, 1], [1, 1, 0]])
    report = structure(conn, 2, 2, {(0, 1)})
    assert report['normal_reciprocal_pair_count'] == 1
    assert report['exact_directed_duplicate_count'] == 0
    assert not report['fp_pairs'] and not report['fn_pairs']
    reciprocal = reciprocal_diagnostics(curves, conn, 2, curves, conn)
    assert reciprocal['distance_p95']['max'] < 1e-12
    near = cross_pair_curves(curves, conn, 2, {(0, 1): {'curve': curves[0], 'closed': False}})
    assert not near['records'] and near['exact_coordinate_duplicate_count'] == 0


def test_closed_phase_shift_does_not_create_curve_disagreement():
    vertices = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    first = np.r_[vertices, vertices[:1]]
    shifted = np.roll(vertices, 2, axis=0)[::-1]
    second = np.r_[shifted, shifted[:1]]
    assert curve_distance(first, second)['max'] < 1e-12


def test_directed_duplicates_and_missing_pairs_are_separate():
    report = structure(np.array([[0, 0, 1], [1, 0, 1], [2, 1, 2]]), 3, 3, {(0, 1), (0, 2)})
    assert report['exact_directed_duplicate_count'] == 1
    assert report['fp_pairs'] == [(1, 2)] and report['fn_pairs'] == [(0, 2)]


def test_gt_close_parallel_boundaries_are_not_new_duplicates():
    curves = np.stack([line(), line((0, 0, .001), (1, 0, .001))])
    conn = np.array([[0, 0, 1], [1, 0, 2]])
    gt = {(0, 1): {'curve': curves[0], 'closed': False}, (0, 2): {'curve': curves[1], 'closed': False}}
    report = cross_pair_curves(curves, conn, 3, gt)
    assert report['thresholds']['0.002']['gt_baseline_close_count'] == 1
    assert report['thresholds']['0.002']['new_nearness_count'] == 0
    gt[(0, 2)]['curve'] = line((0, 0, .03), (1, 0, .03))
    report = cross_pair_curves(curves, conn, 3, gt)
    assert report['thresholds']['0.002']['new_nearness_count'] == 1


def test_touching_only_at_corner_is_not_curve_coincidence():
    curves = np.stack([line(), line((1, 0, 0), (1, 1, 0))])
    report = cross_pair_curves(curves, np.array([[0, 0, 1], [1, 0, 2]]), 3, {})
    assert not report['records']


def test_thin_gt_wall_is_not_new_face_collapse():
    truth = np.stack([planar(), planar(.001)])
    report = face_proximity(truth.copy(), truth)
    assert report['thresholds']['0.002']['new_nearness_count'] == 0
    separated_gt = np.stack([planar(), planar(.02)])
    report = face_proximity(truth, separated_gt)
    assert report['thresholds']['0.002']['new_nearness_count'] == 1


def test_short_open_edge_is_not_exempted_as_closed():
    curve = line((0, 0, 0), (.001, 0, 0))
    report = endpoint_proxy(curve[None], np.array([[0, 0, 1]]), 2,
                            {(0, 1): {'curve': line(), 'closed': False}})
    assert report['thresholds']['0.002']['endpoints_with_residual'] == 4
    assert report['thresholds']['0.002']['new_open_endpoint_collapse_edges'] == 1
    assert all(row['unmatchable_endpoint_count'] == 2 for row in report['faces'])


def test_known_closed_single_edge_can_use_own_endpoint_gap():
    vertices = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]])
    report = endpoint_proxy(vertices[None], np.array([[0, 0, 1]]), 2,
                            {(0, 1): {'curve': vertices, 'closed': True}})
    assert report['thresholds']['0.002']['endpoints_with_residual'] == 0
    assert report['thresholds']['0.002']['new_open_endpoint_collapse_edges'] == 0


def test_coincident_many_endpoints_are_ambiguous_not_validated_loops():
    curves = np.stack([line(), line(), line()])
    conn = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 3]])
    gt = {tuple(row[1:]): {'curve': curves[row[0]], 'closed': False} for row in conn}
    report = endpoint_proxy(curves, conn, 4, gt)
    face = report['faces'][0]
    assert face['distances']['max'] == 0
    assert report['thresholds']['0.002']['endpoints_with_multiple_near_partners'] == 6
    assert 'no ordering, perfect matching' in report['description']
    assert 'loop_valid' not in report


def test_missing_stage_stays_unknown_not_zero():
    unknown = {'status': 'unknown', 'reason': 'no snapshot'}
    assert stage_deltas(unknown, unknown)['status'] == 'unknown'
    flags, threshold = mechanism_flags({'stages': {'selected': unknown, 'optimized': unknown}})
    assert flags['selection_removed_gt_pair'] is None
    assert threshold['0.01']['selected_endpoint_gap'] is None


def test_unknown_gt_baseline_or_empty_edges_does_not_become_false():
    proximity = {'status': 'ok', 'thresholds': {str(t): {'new_nearness_count': 0, 'gt_baseline_unknown_count': 1}
                                              for t in (0.002, 0.006, 0.01, 0.08)}}
    endpoint = endpoint_proxy(np.zeros((0, 16, 3)), np.zeros((0, 3), dtype=int), 1, {})
    _, flags = mechanism_flags({'stages': {'selected': {'cross_pair_near_curves': proximity, 'endpoint_proxy': endpoint}}})
    assert flags['0.01']['selected_new_cross_pair_curve_nearness'] is None
    assert flags['0.01']['selected_endpoint_gap'] is None
