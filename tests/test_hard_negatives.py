"""Hard negative labelling and negative-set composition for the edge classifier."""
import numpy as np

from src.preprocess.hard_negatives import compose_negatives, hard_negative_pairs


def ring_of_four():
    """Faces 0-1-2-3 form a ring of shared edges; 0 and 2 share only a vertex; 1 and 3 come close."""
    face_adj = np.zeros((4, 4), dtype=bool)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        face_adj[a, b] = face_adj[b, a] = True
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16), indexing="ij"), -1)
    flat = np.concatenate((grid, np.zeros((16, 16, 1))), -1)
    grids = np.stack((flat, flat + [0, 0, 0.03], flat + [5, 0, 0], flat + [0, 0, 0.06]))
    face_pairs = np.array([[0, 1], [1, 2], [2, 3], [0, 3]])
    # Vertex 7 sits on edges 0-1 and 1-2, so faces 0 and 2 share it without an edge; faces 1 and 3 share nothing.
    endpoint_vertex_ids = np.array([[7, 8], [7, 13], [9, 10], [11, 12]])
    return face_adj, grids, face_pairs, endpoint_vertex_ids


def test_vertex_and_near_pairs_are_labelled_both_ways():
    face_adj, grids, face_pairs, endpoints = ring_of_four()
    labels = hard_negative_pairs(face_adj, grids, face_pairs, endpoints, near_threshold=0.05)
    assert sorted(map(tuple, labels["vertex_pairs"])) == [(0, 2), (2, 0)]
    assert sorted(map(tuple, labels["near_pairs"])) == [(1, 3), (3, 1)]
    assert np.allclose(labels["near_distance"], 0.03, atol=1e-6)


def test_compose_keeps_count_and_repeats_hard_pairs():
    rng = np.random.RandomState(0)
    pool = np.array([[i, j] for i in range(5) for j in range(5) if (i, j) not in {(0, 1), (1, 0), (2, 3), (3, 2)}])
    hard = np.array([[0, 2], [2, 0]])
    out = compose_negatives(pool, hard, count=10, repeat=3, rng=rng)
    assert out.shape == (10, 2)
    assert sum(tuple(row) == (0, 2) for row in out) == 3 and sum(tuple(row) == (2, 0) for row in out) == 3
    pool_keys = {tuple(row) for row in pool}
    assert all(tuple(row) in pool_keys for row in out)
    assert not any(tuple(row) in {(0, 1), (1, 0), (2, 3), (3, 2)} for row in out)
    short = compose_negatives(pool, hard, count=4, repeat=3, rng=rng)
    assert short.shape == (4, 2) and all(tuple(row) in {(0, 2), (2, 0)} for row in short)
    plain = compose_negatives(pool, np.zeros((0, 2), dtype=np.int64), count=6, repeat=3, rng=rng)
    assert plain.shape == (6, 2) and len({tuple(row) for row in plain}) == 6
