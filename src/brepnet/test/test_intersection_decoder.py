"""Decoder data-flow contracts; these tests do not measure CAD quality."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from src.brepnet.models.intersection_decoder import (
    IntersectionDecoder,
    enumerate_candidate_pairs,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class TinyLegacy(nn.Module):
    """Cheap heads with the same input/output contracts as HoLa's modules."""

    def __init__(self):
        super().__init__()
        self.face_feature_dim = 4
        self.edge_feature_dim = 8
        self.in_channels = 6
        self.face_attn_proj_in2 = nn.Linear(4, 8)
        layer = nn.TransformerEncoderLayer(
            8, 2, dim_feedforward=16, batch_first=True, dropout=0.0, norm_first=True
        )
        self.face_attn2 = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.face_attn_proj_out2 = nn.Linear(8, 4)
        self.face_points_decoder = nn.Sequential(
            nn.Linear(4, 16 * 16 * 6), nn.Unflatten(-1, (16, 16, 6))
        )
        self.edge_points_decoder = nn.Sequential(
            nn.Linear(8, 16 * 6), nn.Unflatten(-1, (16, 6))
        )
        self.face_center_scale_decoder = nn.Linear(4, 4)
        self.edge_center_scale_decoder = nn.Linear(8, 4)
        self.inter = nn.Sequential(nn.Flatten(1), nn.Linear(8, 8), nn.SiLU())
        self.classifier = nn.Linear(8, 1)


def make_decoder(**overrides):
    config = dict(
        hidden_dim=12, geometry_dim=8, geometry_chunk_size=2,
        candidate_chunk_size=2, checkpoint_candidates=False, enabled=True,
    )
    config.update(overrides)
    return IntersectionDecoder(TinyLegacy(), config)


def activate_residuals(model):
    with torch.no_grad():
        for head in (model.refiner.face_delta, model.refiner.edge_delta):
            head.weight.normal_(std=0.03)
            head.bias.normal_(std=0.01)


def test_all_ordered_pairs_reciprocal_and_object_partition():
    pairs = enumerate_candidate_pairs(torch.tensor([1, 3, 2]))
    assert pairs.indices.shape == (8, 2)
    assert (pairs.indices[:, 0] != pairs.indices[:, 1]).all()
    torch.testing.assert_close(pairs.indices[pairs.reverse], pairs.indices.flip(-1))
    torch.testing.assert_close(pairs.reverse[pairs.reverse], torch.arange(8))
    face_objects = torch.tensor([0, 1, 1, 1, 2, 2])
    assert (face_objects[pairs.indices[:, 0]] == pairs.object_ids).all()
    assert (face_objects[pairs.indices[:, 1]] == pairs.object_ids).all()
    with pytest.raises(ValueError, match="integers"):
        enumerate_candidate_pairs(torch.tensor([2.0]))
    with pytest.raises(ValueError, match="at least one"):
        enumerate_candidate_pairs(torch.tensor([0]))


def test_zero_residual_preserves_legacy_geometry_and_directional_features():
    torch.manual_seed(10)
    model = make_decoder().eval()
    z = torch.randn(3, 4)
    output = model(z, torch.tensor([3]))
    assert torch.count_nonzero(output.delta_face) == 0
    assert torch.count_nonzero(output.delta_edge) == 0
    torch.testing.assert_close(output.coarse.face_geometry.points, output.refined.face_geometry.points)
    torch.testing.assert_close(output.coarse.edge_geometry.points, output.refined.edge_geometry.points)
    expected_edges = model.inter(output.coarse.face_features[output.pairs.indices])
    torch.testing.assert_close(output.coarse.edge_features, expected_edges)
    assert not torch.allclose(expected_edges, expected_edges[output.pairs.reverse])
    torch.testing.assert_close(
        output.refined.edge_logits, output.refined.edge_logits[output.pairs.reverse]
    )
    local = model.face_geometry.points_decoder(output.coarse.face_features)
    bbox = model.face_geometry.bbox_decoder(output.coarse.face_features)
    expected_xyz = local[..., :3] * bbox[:, None, None, 3:] + bbox[:, None, None, :3]
    torch.testing.assert_close(output.coarse.face_geometry.points, expected_xyz)


def test_nonzero_refinement_is_object_isolated_and_face_permutation_equivariant():
    torch.manual_seed(11)
    model = make_decoder().eval()
    activate_residuals(model)
    z = torch.randn(5, 4)
    packed = model(z, torch.tensor([3, 2]))
    alone = model(z[:3], torch.tensor([3]))
    torch.testing.assert_close(
        packed.refined.face_geometry.points[:3], alone.refined.face_geometry.points,
        atol=1e-6, rtol=1e-5,
    )
    torch.testing.assert_close(
        packed.refined.edge_geometry.points[:6], alone.refined.edge_geometry.points,
        atol=1e-6, rtol=1e-5,
    )
    permutation = torch.tensor([2, 0, 1])
    reordered = model(z[:3][permutation], torch.tensor([3]))
    torch.testing.assert_close(
        reordered.refined.face_geometry.points,
        alone.refined.face_geometry.points[permutation], atol=1e-6, rtol=1e-5,
    )
    original_pair_rows = {tuple(pair): row for row, pair in enumerate(alone.pairs.indices.tolist())}
    rows = torch.tensor([
        original_pair_rows[tuple(pair)] for pair in permutation[reordered.pairs.indices].tolist()
    ])
    torch.testing.assert_close(
        reordered.refined.edge_geometry.points,
        alone.refined.edge_geometry.points[rows], atol=1e-6, rtol=1e-5,
    )
    torch.testing.assert_close(
        reordered.refined.edge_logits, alone.refined.edge_logits[rows], atol=1e-6, rtol=1e-5
    )
    assert not torch.allclose(alone.coarse.face_geometry.points, alone.refined.face_geometry.points)


def test_final_classifier_supervision_reaches_edge_residual_head():
    torch.manual_seed(12)
    model = make_decoder()
    output = model(torch.randn(3, 4), torch.tensor([3]))
    target = torch.ones_like(output.refined.edge_logits)
    nn.functional.binary_cross_entropy_with_logits(output.refined.edge_logits, target).backward()
    assert model.refiner.edge_delta.weight.grad.abs().sum() > 0
    assert model.classifier.weight.grad.abs().sum() > 0


def test_checkpointed_candidate_gradients_match_direct_path():
    torch.manual_seed(13)
    direct = make_decoder()
    activate_residuals(direct)
    checkpointed = deepcopy(direct)
    checkpointed.checkpoint_candidates = True
    z = torch.randn(3, 4)
    outputs = []
    for model in (direct, checkpointed):
        output = model(z, torch.tensor([3]))
        loss = (
            output.refined.face_geometry.points.square().mean()
            + output.refined.edge_geometry.points.square().mean()
            + output.refined.edge_logits.square().mean()
        )
        loss.backward()
        outputs.append(output)
    torch.testing.assert_close(outputs[0].refined.edge_geometry.points, outputs[1].refined.edge_geometry.points)
    for (name, param), (_, other) in zip(direct.named_parameters(), checkpointed.named_parameters()):
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        torch.testing.assert_close(param.grad, other.grad, atol=1e-6, rtol=1e-4, msg=name)


def test_single_faces_and_loss_only_ablation():
    model = make_decoder().eval()
    activate_residuals(model)
    output = model(torch.randn(2, 4), torch.tensor([1, 1]))
    assert output.pairs.indices.shape == (0, 2)
    assert output.refined.edge_geometry.points.shape == (0, 16, 3)
    assert output.refined.edge_geometry.local_points.shape == (0, 16, 6)
    assert torch.count_nonzero(output.delta_face) == 0
    mixed = model(torch.randn(3, 4), torch.tensor([1, 2]))
    assert torch.count_nonzero(mixed.delta_face[0]) == 0
    baseline = make_decoder(enabled=False)
    assert baseline.refiner is None
    assert not any(name.startswith("refiner.") for name, _ in baseline.named_parameters())
    output = baseline(torch.randn(2, 4), torch.tensor([2]))
    assert output.refined is output.coarse


def test_mixed_precision_keeps_geometric_operations_float32():
    torch.manual_seed(18)
    model = make_decoder(checkpoint_candidates=True)
    activate_residuals(model)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(torch.randn(3, 4), torch.tensor([3]))
        loss = (
            output.refined.face_geometry.points.square().mean()
            + output.refined.edge_geometry.points.square().mean()
            + output.refined.edge_logits.square().mean()
        )
    loss.backward()
    assert output.refined.face_geometry.points.dtype == torch.float32
    assert output.refined.edge_geometry.points.dtype == torch.float32
    assert torch.isfinite(loss)
    assert all(torch.isfinite(param.grad).all() for param in model.parameters() if param.grad is not None)


def test_bad_latent_contracts_rejected():
    model = make_decoder()
    with pytest.raises(ValueError, match="packed"):
        model(torch.randn(2, 5), torch.tensor([2]))
    with pytest.raises(ValueError, match="Sum"):
        model(torch.randn(2, 4), torch.tensor([3]))
    with pytest.raises(ValueError, match="finite"):
        model(torch.full((2, 4), torch.nan), torch.tensor([2]))
