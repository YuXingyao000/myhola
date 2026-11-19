"""Directed distribution, null fallback and native coedge indexing."""
import numpy as np
import pytest
import torch

from src.datasets.hola_corner import HolaCornerDataset, load_corner_topology
from src.models.vae.hola_corner import CornerMessageExchange, distribute_corner_messages


def test_shared_pair_reaches_the_correct_two_endpoints():
    z = torch.zeros(1, 3, 3, 2, requires_grad=True)
    with torch.no_grad():
        z[0, 0, 1] = torch.tensor([10., 20.])
    scores = torch.full((1, 3, 3), -20., requires_grad=True)
    with torch.no_grad():
        scores[0, 0, 1] = 8.
    logits = torch.zeros(1, 3, requires_grad=True)
    start, end, nw, pw = distribute_corner_messages(z, scores, logits, torch.ones(1, 3, dtype=torch.bool))
    torch.testing.assert_close(end[0, 0], nw[0, 0, 1] * z[0, 0, 1])
    torch.testing.assert_close(start[0, 1], pw[0, 0, 1] * z[0, 0, 1])
    assert start[0, [0, 2]].count_nonzero() == 0
    assert end[0, [1, 2]].count_nonzero() == 0
    (start.sum() + end.sum()).backward()
    assert logits.grad is None
    assert scores.grad.abs().sum() > 0 and z.grad.abs().sum() > 0


def test_self_connection_and_null_do_not_force_renormalization():
    z = torch.tensor([[[[2., 4.]]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    start, end, nw, pw = distribute_corner_messages(z, torch.zeros(1, 1, 1), torch.zeros(1, 1), valid)
    torch.testing.assert_close(start, torch.tensor([[[2 / 3, 4 / 3]]]))
    torch.testing.assert_close(start, end)
    torch.testing.assert_close(nw.sum(-1), torch.ones(1, 1))
    torch.testing.assert_close(pw.sum(-2), torch.ones(1, 1))
    for logits, mask in ((torch.full((1, 1), -torch.inf), valid), (torch.zeros(1, 1), ~valid)):
        start, end, nw, pw = distribute_corner_messages(z, torch.zeros(1, 1, 1), logits, mask)
        assert start.count_nonzero() == end.count_nonzero() == 0
        assert nw[0, 0, -1] == pw[0, -1, 0] == 1


def test_identity_same_owner_isolation_and_closed_edge_self_messages():
    torch.manual_seed(42)
    module = CornerMessageExchange(4, 8, 1)
    b = torch.randn(3, 4, requires_grad=True)
    xyz = torch.randn(3, 16, 3, requires_grad=True)
    logits = torch.ones(3, requires_grad=True)
    owners, slots = torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0])
    output, scores = module(b, xyz, logits, owners, slots, 2, 2)
    torch.testing.assert_close(output, b, atol=0, rtol=0)
    assert torch.isneginf(scores[1, 1]).all()
    with torch.no_grad():
        module.readout.weight.fill_(0.1)
    output, _ = module(b, xyz, logits, owners, slots, 2, 2)
    output[0].sum().backward()
    assert b.grad[1].abs().sum() > 0 and b.grad[2].count_nonzero() == 0
    assert xyz.grad[1].abs().sum() > 0 and xyz.grad[2].count_nonzero() == 0
    assert logits.grad is None
    zero, _ = module(b, xyz, torch.full_like(logits, -torch.inf), owners, slots, 2, 2)
    torch.testing.assert_close(zero, b, atol=0, rtol=0)


def test_sorted_cache_axis_is_mapped_to_connection_rows(tmp_path):
    # Raw edge IDs are deliberately non-contiguous, and rows are shuffled.
    conn = np.array([[9, 0, 3], [2, 0, 1], [8, 1, 2], [5, 0, 2]])
    path = tmp_path / 'coedge_topology.npz'
    kwargs = dict(schema_version=1, source_sha256='hash', source_edge_ids=[2, 5, 8, 9],
                  owner_faces=[0, 0, 1, 0], orientation=np.ones(4, dtype=np.int8),
                  previous=[3, 0, 2, 1], next=[1, 3, 2, 0])
    np.savez(path, **kwargs)
    labels = load_corner_topology(path, 'hash', conn)
    assert labels['corner_next'].tolist() == [1, 3, 2, 0]
    assert labels['corner_previous'].tolist() == [3, 0, 2, 1]
    for k, v in labels.items():
        assert torch.equal(labels['corner_next'][labels['corner_previous']], torch.arange(4))
        assert np.array_equal(conn[v, 1], conn[:, 1])
    with pytest.raises(ValueError, match='source mismatch'):
        load_corner_topology(path, 'wrong', conn)
    kwargs['next'] = [2, 3, 1, 0]
    np.savez(path, **kwargs)
    with pytest.raises(ValueError, match='cycle'):
        load_corner_topology(path, 'hash', conn)


def test_collate_offsets_connection_rows_not_raw_edges():
    def sample(prefix):
        conn = torch.tensor([[4, 0, 1], [1, 1, 0]])
        item = (prefix, torch.zeros(2, 16, 16, 6), torch.zeros(7, 16, 6),
                torch.zeros(2, 16, 16, 6), torch.zeros(7, 16, 6),
                torch.zeros(2, 6), torch.zeros(7, 6), conn,
                torch.tensor([[0, 0]]), torch.ones(2, 2, dtype=torch.bool), {})
        return item, {'corner_previous': torch.tensor([0, 1]), 'corner_next': torch.tensor([0, 1])}
    data = HolaCornerDataset.collate_fn([sample('a'), sample('b')])
    assert data['corner_next'].tolist() == [0, 1, 2, 3]
    assert data['edge_face_connectivity'][:, 0].tolist() == [4, 1, 11, 8]
