"""Vertex identity is native topology, independent of numbering and CAD packing."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.common.io import sha256_file
from src.common.prediction_export import save_item_predictions, vae_prediction_items
from src.datasets.brep import AutoEncoder_dataset3
from src.datasets.vertex_identity import VertexIdentityDataset
from src.models.vae.vertex_identity import VertexGroupEncoder, VertexIdentityDecoder
from src.models.vae.losses.vertex_identity import vertex_identity_loss, vertex_geometry_metrics
from src.preprocess.endpoint_vertices import load_endpoint_vertices


def test_encoder_identity_unknown_and_relabeling():
    torch.manual_seed(42)
    model = VertexGroupEncoder(4, 8, 16)
    edges = torch.randn(5, 4)
    ids = torch.tensor([[0, 1], [1, 2], [2, 0], [-1, 1], [-1, -1]])
    torch.testing.assert_close(model(edges, ids), edges, atol=0, rtol=0)
    # Once the zero-initialized writeback opens, group membership must matter.
    with torch.no_grad():
        model.writeback[-1].weight.normal_()
        model.writeback[-1].bias.fill_(0.2)
    result = model(edges, ids)
    relabeled = torch.where(ids >= 0, (ids + 1) * 19, ids)
    torch.testing.assert_close(model(edges, relabeled), result)
    order = torch.tensor([3, 2, 4, 0, 1])
    torch.testing.assert_close(model(edges[order], ids[order]), result[order])
    torch.testing.assert_close(result[-1], edges[-1], atol=0, rtol=0)
    changed = edges.clone()
    changed[1] += 3
    assert not torch.allclose(model(changed, ids)[0], result[0])
    isolated = ids.clone()
    isolated[1] = torch.tensor([80, 81])
    torch.testing.assert_close(model(changed, isolated)[0], model(edges, isolated)[0])
    assert model(edges[:0], ids[:0]).shape == (0, 4)


def test_collate_offsets_native_ids_without_changing_geometry(monkeypatch):
    original = {'original_geometry': object()}
    monkeypatch.setattr(AutoEncoder_dataset3, 'collate_fn', lambda samples: dict(original))
    a = torch.tensor([[0, 5], [-1, 0]])
    b = torch.tensor([[0, 5], [5, 5]])
    result = VertexIdentityDataset.collate_fn([(None, a), (None, b)])
    assert result['original_geometry'] is original['original_geometry']
    assert result['vertex_edge_counts'] == [2, 2]
    expected = torch.tensor([[0, 5], [-1, 0], [6, 11], [11, 11]])
    torch.testing.assert_close(result['endpoint_vertex_ids'], expected)
    torch.testing.assert_close(a, torch.tensor([[0, 5], [-1, 0]]))


def dense_loss(head, embeddings, ids, counts):
    losses, offset = [], 0
    for count in counts:
        u = embeddings[offset:offset + count].flatten(0, 1)
        labels = ids[offset:offset + count].flatten()
        u, labels = u[labels >= 0], labels[labels >= 0]
        score = head.logits(u, u)
        mask = torch.ones_like(score, dtype=torch.bool).triu(1)
        equal = labels[:, None] == labels[None, :]
        terms = []
        for target in (True, False):
            chosen = score[mask & (equal == target)]
            if chosen.numel():
                terms.append(F.binary_cross_entropy_with_logits(chosen, torch.full_like(chosen, float(target))))
        losses.append(torch.stack(terms).mean() if terms else score.sum() * 0)
        offset += count
    return torch.stack(losses).mean()


@pytest.mark.parametrize('chunk_size', [1, 3, 100])
def test_pair_loss_dense_parity_gradients_and_cad_isolation(chunk_size):
    torch.manual_seed(1)
    head = VertexIdentityDecoder(4, 8, 16)
    embeddings = head(torch.randn(5, 4))
    ids = torch.tensor([[0, 1], [1, 2], [2, -1], [0, 0], [-1, -1]])
    # Second CAD intentionally reuses IDs. Its closed edge forms a positive pair.
    counts = [3, 1, 1]
    pairs = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 1], [0, 1]])
    actual, metrics = vertex_identity_loss(head, embeddings, ids, counts, pairs, chunk_size)
    expected = dense_loss(head, embeddings, ids, counts)
    torch.testing.assert_close(actual, expected)
    params = [embeddings, head.log_scale, head.bias]
    ag = torch.autograd.grad(actual, params, retain_graph=True)
    eg = torch.autograd.grad(expected, params)
    for first, second in zip(ag, eg):
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
    assert all(not x.requires_grad for x in metrics.values())
    assert ag[0][-1].count_nonzero() == 0


@pytest.mark.parametrize('ids', [torch.tensor([[-1, -1]]), torch.tensor([[1, 2]]), torch.empty(0, 2, dtype=torch.long)])
def test_missing_pairs_or_positive_class(ids):
    head = VertexIdentityDecoder(4, 8, 16)
    embedding = head(torch.randn(len(ids), 4))
    loss, metrics = vertex_identity_loss(head, embedding, ids, [len(ids)], torch.zeros(len(ids), 2, dtype=torch.long), 2)
    assert torch.isfinite(loss)
    loss.backward()
    assert head.bias.grad is not None
    assert all(torch.isfinite(v) for v in metrics.values())
    if not (ids >= 0).any():
        assert loss.item() == 0


def test_decoder_symmetry_and_original_geometry_metric_units():
    head = VertexIdentityDecoder(4, 8, 16)
    embedding = head(torch.randn(3, 4))
    probability = head.probabilities(embedding)
    torch.testing.assert_close(probability, probability.T)
    assert probability.shape == (6, 6)
    local = torch.zeros(2, 16, 6)
    bbox = torch.tensor([[1., 0, 0, 2], [1., 0, 0, 2]])
    gt = torch.zeros(2, 16, 3)
    metrics = vertex_geometry_metrics(local, bbox, gt, torch.tensor([[0, 1], [1, 0]]), [2])
    assert metrics['endpoint_xyz_mae'].item() == pytest.approx(1 / 3)
    assert metrics['interior_xyz_mae'].item() == pytest.approx(1 / 3)
    assert metrics['vertex_spread'].item() == 0


def test_sidecar_binds_actual_directed_row_order_and_source(tmp_path):
    source = tmp_path / 'data.npz'
    np.savez(source, placeholder=np.array(1))
    conn = np.array([[3, 0, 1], [9, 1, 0]])
    ids = np.array([[4, 8], [8, -1]])
    labels = tmp_path / 'endpoint_vertices.npz'
    np.savez(labels, schema_version=np.array(1), source_sha256=np.array(sha256_file(source)),
             source_edge_ids=conn[:, 0], face_pairs=conn[:, 1:], endpoint_vertex_ids=ids,
             endpoint_valid=ids >= 0)
    np.testing.assert_array_equal(load_endpoint_vertices(labels, source_path=source, connectivity=conn), ids)
    with pytest.raises(ValueError, match='connection order'):
        load_endpoint_vertices(labels, source_path=source, connectivity=conn[::-1])
    np.savez(source, placeholder=np.array(2))
    with pytest.raises(ValueError, match='digest'):
        load_endpoint_vertices(labels, source_path=source, connectivity=conn)


def test_identity_export_survives_npz_without_changing_geometry(tmp_path):
    relations = np.eye(4, dtype=np.float32)
    curves = np.zeros((2, 16, 3), dtype=np.float32)
    items = vae_prediction_items(['cad'], {'pred_edge': curves, 'pred_endpoint_same_vertex': relations}, {})
    save_item_predictions(['cad'], items, tmp_path)
    with np.load(tmp_path / 'cad/data.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['pred_endpoint_same_vertex'], relations)
        np.testing.assert_array_equal(saved['pred_edge'], curves)
