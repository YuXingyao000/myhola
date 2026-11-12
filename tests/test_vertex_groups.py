"""Native endpoint labels: direction, group balancing and supervision boundary."""
import numpy as np
import pytest
import torch

from src.common.io import sha256_file
from src.datasets.vertex_groups import collate_corner_targets
from src.models.vae.losses.corner import corner_consistency
from src.preprocess.vertex_groups import canonical_edges, corner_pairs, load_vertex_groups


def test_native_ids_keep_high_valence_short_edges_and_exclude_single_closed_curve():
    source_ids = np.array([6, 1, 8, 3, 9])
    vertices = np.array([[10, 11], [12, 10], [10, 13], [20, 20], [10, 11]])
    targets = corner_pairs(source_ids, vertices)
    assert targets['corner_group_vertices'].tolist() == [10, 11]
    assert np.bincount(targets['corner_pair_groups']).tolist() == [6, 1]
    assert not (targets['corner_pairs'][..., 0] == 3).any()  # closed curve excluded
    assert np.all(targets['corner_pairs'][:, 0, 0] != targets['corner_pairs'][:, 1, 0])
    assert [((6, 1), (9, 1))] == [tuple(map(tuple, x)) for x in targets['corner_pairs'][-1:]]
    assert corner_pairs(source_ids, np.full((5, 2), -1))['corner_pairs'].shape == (0, 2, 2)


def test_sidecar_source_digest_direction_and_corruption(tmp_path):
    conn = np.array([[3, 2, 1], [0, 1, 0], [1, 0, 1], [2, 1, 2]])
    ids, pairs = canonical_edges(conn)
    assert ids.tolist() == [1, 2]  # not smallest raw edge ID; actual i<j direction
    vertices = np.array([[4, 5], [5, 6]])
    source = tmp_path / 'data.npz'; source.write_bytes(b'original')
    data = dict(schema_version=np.array(1), source_sha256=np.array(sha256_file(source)),
                source_edge_ids=ids, face_pairs=pairs, endpoint_vertex_ids=vertices,
                **corner_pairs(ids, vertices))
    path = tmp_path / 'vertex_groups.npz'; np.savez(path, **data)
    actual = load_vertex_groups(path, source_sha256=sha256_file(source), connectivity=conn[::-1])
    assert actual['corner_pairs'].tolist() == [[[1, 1], [2, 0]]]
    with pytest.raises(ValueError, match='digest mismatch'):
        load_vertex_groups(path, source_sha256='wrong', connectivity=conn)
    data['corner_pairs'][0, 0, 1] = 0
    np.savez(path, **data)
    with pytest.raises(ValueError, match='corner_pairs'):
        load_vertex_groups(path, source_sha256=sha256_file(source), connectivity=conn)


def labels(ids, vertices):
    return {k: torch.from_numpy(v) for k, v in corner_pairs(np.array(ids), np.array(vertices)).items()}


def test_collate_offsets_raw_edges_groups_and_reuse_without_cross_object_pairs():
    a = labels([1, 4], [[0, 1], [1, 2]])
    b = labels([0, 2], [[0, 1], [1, 2]])
    empty = labels([0], [[8, 8]])
    packed = collate_corner_targets([a, empty, b], [6, 2, 4])
    assert packed['pairs'].tolist() == [[[1, 1], [4, 0]], [[8, 1], [10, 0]]]
    assert packed['pair_groups'].tolist() == [0, 1]
    assert packed['group_objects'].tolist() == [0, 2]
    torch.testing.assert_close(collate_corner_targets([a, empty, b], [6, 2, 4])['pairs'], packed['pairs'])
    assert a['corner_pairs'].tolist() == [[[1, 1], [4, 0]]]


def test_loss_only_selected_endpoint_gradients_and_translation_invariance():
    curves = torch.zeros(5, 16, 3, requires_grad=True)
    with torch.no_grad():
        curves[3, -1] = torch.tensor([.3, .6, .9])
    target = collate_corner_targets([labels([1, 4], [[0, 1], [1, 2]])], [6])
    edge_to_candidate = torch.tensor([4, 3, 1, 0, 2, 4])
    result = corner_consistency(curves, edge_to_candidate, target, 1)
    torch.testing.assert_close(result['corner'], torch.tensor(.6))
    result['corner'].backward()
    assert torch.count_nonzero(curves.grad) == 6
    torch.testing.assert_close(curves.grad[3, -1], -curves.grad[2, 0])
    shifted = corner_consistency(curves.detach()+torch.tensor([3., 4., 5.]), edge_to_candidate, target, 1)
    torch.testing.assert_close(shifted['corner'], result['corner'])
    assert not result['corner_gap'].requires_grad


def test_group_then_object_balancing_and_empty_objects():
    # Object0: one 3-member vertex (mean pair L1=4/3), one 2-member vertex (=4).
    # Object1: no corners. Batch loss = ((4/3+4)/2 + 0)/2 = 4/3.
    t = labels([0, 1, 2, 3, 4], [[9,-1],[9,-1],[9,-1],[10,-1],[10,-1]])
    e = labels([0], [[1,1]])
    target = collate_corner_targets([t,e], [5,1])
    curves = torch.zeros(6, 16, 3, dtype=torch.float16, requires_grad=True)
    with torch.no_grad():
        curves[:5,0] = torch.tensor([0.,1.,2.,0.,4.])[:,None]
    result = corner_consistency(curves, torch.arange(6), target, 2)
    assert result['corner'].dtype == torch.float32
    torch.testing.assert_close(result['corner'], torch.tensor(4/3))
    empty = collate_corner_targets([e], [1])
    zero = corner_consistency(curves, torch.arange(6), empty, 1)['corner']
    zero.backward()
    assert zero.item() == 0 and torch.isfinite(curves.grad).all()


def test_final_loss_mapping_ignores_candidate_order_and_existence_probability():
    from tests.test_intersection_losses import make_case
    from src.models.vae.losses.intersection import IntersectionLoss
    from src.models.vae.state import PrimitiveGeometry
    output, batch, teacher, teacher_geo = make_case(face_counts=(3,))
    # Distinct canonical pairs (0,1) and (1,2); row order need not equal edge IDs.
    for key in ('edge_points', 'edge_norm', 'edge_bbox'):
        batch[key] = torch.cat((batch[key], batch[key]))
    teacher = torch.cat((teacher, teacher)).detach().requires_grad_(True)
    teacher_geo = PrimitiveGeometry(*[
        torch.cat((value, value)).detach().requires_grad_(True)
        for value in (teacher_geo.local_points, teacher_geo.center_scale, teacher_geo.points)
    ])
    batch['edge_face_connectivity'] = torch.tensor([[3,2,1], [0,0,1], [1,1,0], [2,1,2]])
    batch['corner_targets'] = collate_corner_targets([labels([0,2], [[4,5],[5,6]])], [4])
    criterion = IntersectionLoss({'corner_weight':1.0})
    losses = criterion(output, batch, teacher, teacher_geo, torch.tensor(0.), False)
    assert losses['corner'] > 0
    losses['corner'].backward()
    assert output.refined.edge_geometry.points.grad is not None
    assert output.coarse.edge_geometry.points.grad is None
    assert teacher_geo.points.grad is None
    assert output.refined.face_geometry.points.grad is None
    assert output.refined.edge_logits.grad is None
    # Only the Final tensor receives the direct derivative; its upstream network
    # still trains through the normal computational graph.
    before = losses['corner'].detach()
    with torch.no_grad(): output.refined.edge_logits.fill_(-100)
    changed = criterion(output, batch, teacher, teacher_geo, torch.tensor(0.), False)
    torch.testing.assert_close(changed['corner'], before)
    torch.testing.assert_close(changed['weighted_corner'], changed['corner'])
    torch.testing.assert_close(changed['total_loss'], sum(v for k,v in changed.items() if k.startswith('weighted_')))
    # Parameter-free supervision: old checkpoints load without new state keys.
    assert criterion.state_dict() == IntersectionLoss({}).state_dict()
    baseline = IntersectionLoss({})(output, batch, teacher, teacher_geo, torch.tensor(0.), False)
    assert 'corner' not in baseline
