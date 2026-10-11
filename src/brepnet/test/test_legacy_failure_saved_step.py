import numpy as np
import pytest

from tools.legacy_failure_saved_step import nearby_free_edges, proximity


def line(a, b):
    return np.linspace(a, b, 128)


def test_reversed_curve_is_same_locus():
    edge = line([0, 0, 0], [1, 0, 0])
    assert proximity(edge, edge[::-1])['max'] < 1e-12


def test_shared_vertex_does_not_mean_duplicate_curve():
    first = line([0, 0, 0], [1, 0, 0])
    second = line([1, 0, 0], [1, 1, 0])
    assert proximity(first, second)['p95'] > 0.9


def test_different_face_free_edges_and_thresholds():
    first = line([0, 0, 0], [1, 0, 0])
    second = first+[0, 0.004, 0]
    result = nearby_free_edges([{'id': 0, 'unique_face_ids': [0]}, {'id': 1, 'unique_face_ids': [1]}],
                               {0: first, 1: second})
    assert result['thresholds']['0.002']['pair_count'] == 0
    assert result['thresholds']['0.006']['pair_count'] == 1
    assert result['thresholds']['0.006']['involved_edge_count'] == 2


def test_same_face_edges_are_not_called_unsewn_pair():
    first = line([0, 0, 0], [1, 0, 0])
    result = nearby_free_edges([{'id': 0, 'unique_face_ids': [0]}, {'id': 1, 'unique_face_ids': [0]}],
                               {0: first, 1: first.copy()})
    assert result['pairs'] == []


def test_nonfinite_geometry_is_not_a_zero_distance():
    with pytest.raises(ValueError, match='finite'):
        proximity([[float('nan'), 0, 0]], [[0, 0, 0], [1, 0, 0]])
