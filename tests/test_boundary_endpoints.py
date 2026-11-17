"""Information routing, native target indexing, and explicit head ownership."""
import numpy as np
import pytest
import torch

from src.models.vae.decoders.boundary import EndpointExchange, OwnerAttention, owner_layout, source_weights
from src.datasets.boundary import load_endpoint_topology, BoundaryDataset
from src.models.vae.decoders.boundary import EndpointOutput
from src.models.vae.losses.boundary import endpoint_losses
from types import SimpleNamespace


def test_confidence_gate_is_detached_and_null_can_suppress_all_sources():
    scores = torch.randn(2, 4, 3, 3, requires_grad=True)
    confidence = torch.zeros(2, 3, requires_grad=True)
    weights = source_weights(scores, confidence)
    assert torch.equal(weights, torch.zeros_like(weights))
    weights.sum().backward()
    assert confidence.grad is None
    assert torch.isfinite(scores.grad).all()


def test_owner_attention_is_equivariant_and_does_not_mix_objects():
    torch.manual_seed(3)
    block = OwnerAttention()
    messages = torch.randn(7, 128)
    owners = torch.tensor([0, 0, 1, 1, 1, 3, 3])
    probability = torch.full((7,), 0.9)
    result, _, _ = block(messages, probability, owners, 4)
    altered = messages.clone()
    altered[owners == 3] += 20
    changed, _, _ = block(altered, probability, owners, 4)
    torch.testing.assert_close(changed[owners != 3], result[owners != 3])
    permutation = torch.tensor([6, 2, 0, 5, 1, 4, 3])
    permuted, _, _ = block(messages[permutation], probability[permutation], owners[permutation], 4)
    torch.testing.assert_close(permuted, result[permutation])


def test_endpoint_scores_are_one_relation_table_with_distinct_values():
    torch.manual_seed(5)
    block = EndpointExchange()
    boundary = torch.randn(5, 128, requires_grad=True)
    owners = torch.tensor([0, 0, 2, 2, 2])
    rows, slots = owner_layout(owners, 3)
    updated, output = block(boundary, torch.ones(5), owners, rows, slots)
    q = block.endpoint(boundary).reshape(5, 2, 128)
    first = torch.tensor([0, 1])
    e = block.end_key(q[first, 1]).reshape(2, 4, 32).transpose(0, 1)
    s = block.start_key(q[first, 0]).reshape(2, 4, 32).transpose(0, 1)
    expected = e @ s.transpose(-1, -2) / 32 ** 0.5
    torch.testing.assert_close(output.scores[0, :, :2, :2], expected)
    assert updated.shape == (5, 2, 128)
    assert output.vertices.shape == (5, 2, 3)
    (updated.square().mean() + output.vertices.square().mean() + output.scores.square().mean()).backward()
    for module in (block.endpoint, block.start_key, block.end_key, block.start_value, block.end_value, block.vertex):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    assert boundary.grad.abs().sum() > 0


def test_native_topology_allows_self_closed_loops_and_rejects_wrong_direction(tmp_path):
    connectivity = np.array([[0, 0, 1], [1, 1, 0]])
    data = dict(schema_version=np.array(1), source_sha256=np.array('digest'),
                source_edge_ids=np.arange(2), owner_faces=np.array([0, 1]),
                orientation=np.ones(2), previous=np.arange(2), next=np.arange(2))
    path = tmp_path / 'coedge.npz'
    np.savez(path, **data)
    labels = load_endpoint_topology(path, 'digest', connectivity)
    assert labels['endpoint_next'].tolist() == [0, 1]
    with pytest.raises(ValueError, match='source'):
        load_endpoint_topology(path, 'wrong', connectivity)
    data['orientation'][0] = -1
    np.savez(path, **data)
    with pytest.raises(ValueError, match='loop-oriented'):
        load_endpoint_topology(path, 'digest', connectivity)


def test_endpoint_collation_uses_raw_edge_offsets(monkeypatch):
    from src.datasets.brep import AutoEncoder_joint_dataset
    monkeypatch.setattr(AutoEncoder_joint_dataset, 'collate_fn', lambda batch: {})
    def item(neighbors):
        return ('id', None, torch.empty(len(neighbors), 16, 6),
                {'endpoint_previous': torch.tensor(neighbors), 'endpoint_next': torch.tensor(neighbors)})
    result = BoundaryDataset.collate_fn([item([1, 0]), item([2, 1, 0])])
    assert result['endpoint_targets']['next'].tolist() == [1, 0, 4, 3, 2]


def test_connection_targets_follow_raw_ids_and_transposed_previous():
    # One three-edge cycle and one closed edge on a different face. Raw GT IDs
    # deliberately differ from candidate order; forward cycle is not symmetric.
    owners = torch.tensor([0, 0, 0, 1])
    rows, _ = owner_layout(owners, 2)
    scores = torch.full((2, 4, 3, 3), -8.0, requires_grad=True)
    with torch.no_grad():
        scores[0, :, 0, 1] = scores[0, :, 1, 2] = scores[0, :, 2, 0] = 8
        scores[1, :, 0, 0] = 8
    vertices = torch.randn(4, 2, 3, requires_grad=True)
    candidates = torch.tensor([2, 0, 3, 1])
    edge_ids = torch.arange(4)
    curves = torch.zeros(4, 16, 6)
    curves[:, 0, :3] = vertices.detach()[candidates, 0]
    curves[:, -1, :3] = vertices.detach()[candidates, 1]
    batch = {'edge_points': curves, 'endpoint_targets': {
        'next': torch.tensor([1, 3, 2, 0]), 'previous': torch.tensor([3, 0, 2, 1]),
    }}
    output = SimpleNamespace(endpoints=EndpointOutput(vertices, scores, rows),
                             pairs=SimpleNamespace(indices=torch.stack((owners, torch.zeros_like(owners)), -1)))
    targets = SimpleNamespace(candidate_rows=candidates, edge_ids=edge_ids)
    losses = endpoint_losses(output, batch, targets)
    assert losses['endpoint_connection'] < 0.001
    assert losses['endpoint_vertex'] == 0
    assert losses['endpoint_next_accuracy'] == 1
    losses['endpoint_connection'].backward()
    assert torch.isfinite(scores.grad).all()
