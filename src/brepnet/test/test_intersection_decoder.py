"""Decoder data-flow contracts; these tests do not measure CAD quality."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from src.brepnet.models.blocks import (
    CurvePointFeedback,
    IntersectionDecoder,
    PrimitiveGeometry,
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


def make_conditioned_decoder(*, enabled=True, checkpoint_candidates=False, point_feedback=None):
    """Use the actual CNN/MLP head layout at a small, inexpensive width."""
    from src.brepnet.models.vae import AutoEncoder_light
    from src.brepnet.test.test_intersection_vae import tiny_config

    config = tiny_config()
    config["refinement"]["checkpoint_candidates"] = checkpoint_candidates
    model = IntersectionDecoder(
        AutoEncoder_light(config), config["refinement"],
        curve_conditioning_cfg={"enabled": enabled, "hidden_dim": 128},
        curve_point_feedback_cfg=point_feedback,
    )
    # Deterministic direct/checkpoint comparisons still exercise training paths.
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        elif isinstance(module, nn.MultiheadAttention):
            module.dropout = 0.0
    return model


def activate_conditioning(model):
    with torch.no_grad():
        for head in (
            model.edge_geometry.conditioning.point_affine,
            model.edge_geometry.conditioning.bbox_affine,
        ):
            head.weight.normal_(std=.02)
            head.bias.normal_(std=.01)


def test_zero_conditioning_preserves_all_old_outputs_and_registration_order():
    torch.manual_seed(81)
    original = make_conditioned_decoder(enabled=False).eval()
    activate_residuals(original)
    conditioned = make_conditioned_decoder().eval()
    mismatch = conditioned.load_state_dict(original.state_dict(), strict=False)
    assert not mismatch.unexpected_keys
    assert mismatch.missing_keys
    assert all(key.startswith("edge_geometry.conditioning.") for key in mismatch.missing_keys)
    old_names = list(dict(original.named_parameters()))
    retained_names = [
        key for key, _ in conditioned.named_parameters()
        if not key.startswith("edge_geometry.conditioning.")
    ]
    assert retained_names == old_names
    z, counts = torch.randn(5, 8), torch.tensor([3, 2])
    with torch.no_grad():
        before, after = original(z, counts), conditioned(z, counts)
    for stage in ("coarse", "refined"):
        a, b = getattr(before, stage), getattr(after, stage)
        for feature in ("face_features", "edge_features", "edge_logits"):
            torch.testing.assert_close(getattr(a, feature), getattr(b, feature), rtol=0, atol=0)
        for primitive in ("face_geometry", "edge_geometry"):
            for component in ("local_points", "center_scale", "points"):
                torch.testing.assert_close(
                    getattr(getattr(a, primitive), component),
                    getattr(getattr(b, primitive), component), rtol=0, atol=0,
                )


def test_conditioning_only_changes_final_edges_and_reads_refined_owner_pair():
    torch.manual_seed(82)
    model = make_conditioned_decoder().eval()
    activate_residuals(model)
    z, counts = torch.randn(3, 8), torch.tensor([3])
    with torch.no_grad():
        before = model(z, counts)
        teacher_before = model.edge_geometry(before.coarse.edge_features)
    activate_conditioning(model)
    seen = []
    handle = model.edge_geometry.conditioning.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    with torch.no_grad():
        after = model(z, counts)
        teacher_after = model.edge_geometry(before.coarse.edge_features)
    handle.remove()
    expected = torch.cat((
        after.refined.edge_features,
        after.refined.face_features[after.pairs.indices[:, 0]],
        after.refined.face_features[after.pairs.indices[:, 1]],
    ), dim=-1)
    torch.testing.assert_close(torch.cat(seen), expected, rtol=0, atol=0)
    for stage in ("coarse", "refined"):
        a, b = getattr(before, stage), getattr(after, stage)
        for field in ("face_features", "edge_features", "edge_logits"):
            torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=0, atol=0)
        torch.testing.assert_close(a.face_geometry.points, b.face_geometry.points, rtol=0, atol=0)
    torch.testing.assert_close(before.coarse.edge_geometry.points, after.coarse.edge_geometry.points, rtol=0, atol=0)
    torch.testing.assert_close(teacher_before.points, teacher_after.points, rtol=0, atol=0)
    assert not torch.allclose(before.refined.edge_geometry.local_points, after.refined.edge_geometry.local_points)
    assert not torch.allclose(before.refined.edge_geometry.center_scale, after.refined.edge_geometry.center_scale)


def test_active_conditioning_has_gradients_to_both_owner_features():
    torch.manual_seed(83)
    model = make_conditioned_decoder().eval()
    activate_conditioning(model)
    edge = torch.randn(2, model.edge_dim, requires_grad=True)
    first = torch.randn(2, model.face_dim, requires_grad=True)
    second = torch.randn(2, model.face_dim, requires_grad=True)
    geometry = model.edge_geometry(edge, torch.cat((edge, first, second), -1))
    geometry.points.square().mean().backward()
    for value in (edge, first, second):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert (value.grad.abs().sum(-1) > 0).all()
    projection_grad = model.edge_geometry.conditioning.projection[0].weight.grad
    assert projection_grad is not None and projection_grad.abs().sum() > 0


@pytest.mark.parametrize("active", [False, True])
def test_conditioned_chunk_checkpoint_gradients_match_direct(active):
    torch.manual_seed(84)
    direct = make_conditioned_decoder()
    activate_residuals(direct)
    if active:
        activate_conditioning(direct)
    checkpointed = deepcopy(direct)
    checkpointed.checkpoint_candidates = True
    z = torch.randn(3, 8)
    outputs = []
    for model in (direct, checkpointed):
        output = model(z, torch.tensor([3]))
        loss = (
            output.refined.edge_geometry.points.square().mean()
            + output.refined.face_geometry.points.square().mean()
            + output.refined.edge_logits.square().mean()
        )
        loss.backward()
        outputs.append(output)
    torch.testing.assert_close(outputs[0].refined.edge_geometry.points, outputs[1].refined.edge_geometry.points)
    for (name, parameter), (other_name, other) in zip(direct.named_parameters(), checkpointed.named_parameters()):
        assert name == other_name
        assert parameter.grad is not None and other.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, other.grad, atol=1e-6, rtol=1e-4, msg=name)
    for head in (direct.edge_geometry.conditioning.point_affine, direct.edge_geometry.conditioning.bbox_affine):
        assert head.weight.grad.abs().sum() > 0


def test_conditioned_single_faces_keep_empty_edge_contract():
    model = make_conditioned_decoder().eval()
    activate_conditioning(model)
    output = model(torch.randn(2, 8), torch.tensor([1, 1]))
    assert output.refined.edge_geometry.local_points.shape == (0, 16, 6)
    assert output.refined.edge_geometry.center_scale.shape == (0, 4)
    assert output.refined.edge_geometry.points.shape == (0, 16, 3)
    assert output.refined.edge_logits.shape == (0,)


def make_point_feedback_decoder(*, enabled=True, checkpoint_candidates=False):
    return make_conditioned_decoder(
        checkpoint_candidates=checkpoint_candidates,
        point_feedback={"enabled": enabled, "hidden_dim": 16, "feature_dim": 8},
    )


def activate_point_feedback(model):
    """Emulate a learned nonzero output head while retaining the hidden mapping."""
    head = [module for module in model.curve_point_feedback.modules() if isinstance(module, nn.Linear)][-1]
    with torch.no_grad():
        head.weight.normal_(std=.02)
        head.bias.normal_(std=.01)
    return head


def test_point_feedback_zero_start_preserves_conditioned_checkpoint_outputs():
    torch.manual_seed(91)
    original = make_conditioned_decoder().eval()
    activate_residuals(original)
    activate_conditioning(original)
    disabled = make_point_feedback_decoder(enabled=False).eval()
    disabled.load_state_dict(original.state_dict(), strict=True)
    assert list(disabled.state_dict()) == list(original.state_dict())

    feedback = make_point_feedback_decoder().eval()
    mismatch = feedback.load_state_dict(original.state_dict(), strict=False)
    assert mismatch.missing_keys and not mismatch.unexpected_keys
    assert all(name.startswith("curve_point_feedback.") for name in mismatch.missing_keys)
    retained_names = [name for name, _ in feedback.named_parameters() if not name.startswith("curve_point_feedback.")]
    assert retained_names == list(dict(original.named_parameters()))
    z, counts = torch.randn(5, 8), torch.tensor([3, 2])
    with torch.no_grad():
        before = original(z, counts)
        after = feedback(z, counts)
    for stage_name in ("coarse", "refined"):
        first, second = getattr(before, stage_name), getattr(after, stage_name)
        for field in ("face_features", "edge_features", "edge_logits"):
            torch.testing.assert_close(getattr(first, field), getattr(second, field), rtol=0, atol=0)
        for primitive in ("face_geometry", "edge_geometry"):
            for field in ("local_points", "center_scale", "points"):
                torch.testing.assert_close(
                    getattr(getattr(first, primitive), field),
                    getattr(getattr(second, primitive), field), rtol=0, atol=0,
                )


def test_point_feedback_changes_only_final_xyz_and_preserves_export_consistency():
    torch.manual_seed(92)
    model = make_point_feedback_decoder().eval()
    activate_residuals(model)
    activate_conditioning(model)
    z, counts = torch.randn(3, 8), torch.tensor([3])
    with torch.no_grad():
        before = model(z, counts)
        teacher_before = model.edge_geometry(before.coarse.edge_features)
    activate_point_feedback(model)
    with torch.no_grad():
        after = model(z, counts)
        teacher_after = model.edge_geometry(before.coarse.edge_features)
    for stage_name in ("coarse", "refined"):
        first, second = getattr(before, stage_name), getattr(after, stage_name)
        for field in ("face_features", "edge_features", "edge_logits"):
            torch.testing.assert_close(getattr(first, field), getattr(second, field), rtol=0, atol=0)
        torch.testing.assert_close(first.face_geometry.points, second.face_geometry.points, rtol=0, atol=0)
    torch.testing.assert_close(before.coarse.edge_geometry.points, after.coarse.edge_geometry.points, rtol=0, atol=0)
    torch.testing.assert_close(teacher_before.points, teacher_after.points, rtol=0, atol=0)
    old_edge, new_edge = before.refined.edge_geometry, after.refined.edge_geometry
    torch.testing.assert_close(old_edge.center_scale, new_edge.center_scale, rtol=0, atol=0)
    torch.testing.assert_close(old_edge.local_points[..., 3:], new_edge.local_points[..., 3:], rtol=0, atol=0)
    assert not torch.allclose(old_edge.points, new_edge.points)
    reconstructed = new_edge.local_points[..., :3] * new_edge.center_scale[:, None, 3:] + new_edge.center_scale[:, None, :3]
    torch.testing.assert_close(new_edge.points, reconstructed, rtol=2e-6, atol=2e-7)


def test_point_feedback_checkpointed_and_mixed_precision_gradients():
    torch.manual_seed(93)
    direct = make_point_feedback_decoder()
    activate_residuals(direct)
    activate_conditioning(direct)
    activate_point_feedback(direct)
    checkpointed = deepcopy(direct)
    checkpointed.checkpoint_candidates = True
    z = torch.randn(3, 8)
    predictions = []
    for model in (direct, checkpointed):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(z, torch.tensor([3]))
            loss = output.refined.edge_geometry.points.square().mean() + output.refined.face_geometry.points.square().mean()
        loss.backward()
        assert output.refined.edge_geometry.points.dtype == torch.float32
        assert torch.isfinite(loss)
        predictions.append(output.refined.edge_geometry.points)
    torch.testing.assert_close(predictions[0], predictions[1], rtol=0, atol=0)
    for (name, parameter), (other_name, other) in zip(direct.named_parameters(), checkpointed.named_parameters()):
        assert name == other_name
        if parameter.grad is None:
            assert other.grad is None, name
            continue
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, other.grad, atol=1e-6, rtol=1e-4, msg=name)
    for name, parameter in direct.curve_point_feedback.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_point_feedback_empty_candidates_are_valid():
    model = make_point_feedback_decoder().eval()
    activate_point_feedback(model)
    output = model(torch.randn(2, 8), torch.tensor([1, 1]))
    assert output.refined.edge_geometry.local_points.shape == (0, 16, 6)
    assert output.refined.edge_geometry.center_scale.shape == (0, 4)
    assert output.refined.edge_geometry.points.shape == (0, 16, 3)
    assert output.refined.edge_logits.shape == (0,)


def point_feedback_geometry(points):
    bbox = points.new_tensor([.15, -.2, .3, .7]).expand(points.shape[0], -1).clone()
    spatial_dims = (None,) * (points.ndim - 2)
    expanded = bbox[(slice(None), *spatial_dims)]
    local = (points - expanded[..., :3]) / expanded[..., 3:]
    directions = torch.nn.functional.normalize(torch.ones_like(points), dim=-1)
    return PrimitiveGeometry(torch.cat((local, directions), -1), bbox, points)


def test_point_feedback_is_local_permutation_equivariant_and_keeps_closed_seam():
    """No point-index encoding should split identical samples at a closed seam."""
    torch.manual_seed(94)
    block = CurvePointFeedback(4, 8, hidden_dim=16, feature_dim=8, geometry_chunk_size=2)
    with torch.no_grad():
        block.delta.weight.normal_(std=.1)
        block.delta.bias.normal_(std=.01)
    t = torch.linspace(-.8, .8, 16)
    curve = torch.stack((t, .2 + t.square(), .1 + .1 * t), dim=-1)
    curve[-1] = curve[0]
    edge = point_feedback_geometry(curve[None].repeat(2, 1, 1))
    a, b = torch.meshgrid(torch.linspace(-1., 1., 3), torch.linspace(-1., 1., 3), indexing="ij")
    face = point_feedback_geometry(torch.stack((
        torch.stack((a, b, torch.zeros_like(a)), -1),
        torch.stack((a, torch.zeros_like(a), b), -1),
    )))
    features = torch.randn(2, 8, requires_grad=True)
    owners = torch.randn(2, 4, requires_grad=True)
    pairs = torch.tensor([[0, 1], [1, 0]])
    result = block(edge, face, features, owners, pairs)
    delta = result.points - edge.points
    assert delta.std(dim=1).amax() > 1e-4
    torch.testing.assert_close(result.points[:, 0], result.points[:, -1], rtol=0, atol=0)
    permutation = torch.randperm(16)
    shuffled = PrimitiveGeometry(edge.local_points[:, permutation], edge.center_scale, edge.points[:, permutation])
    reordered = block(shuffled, face, features, owners, pairs)
    torch.testing.assert_close(reordered.points, result.points[:, permutation], rtol=0, atol=0)
    result.points.square().mean().backward()
    for value in (features, owners):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert (value.grad.abs().sum(-1) > 0).all()


def test_point_feedback_exact_zero_scale_preserves_geometry_without_clipping_other_scales():
    torch.manual_seed(98)
    block = CurvePointFeedback(4, 8, hidden_dim=16, feature_dim=8, geometry_chunk_size=2)
    displacement = torch.tensor([.01, -.02, .03])
    with torch.no_grad():
        block.delta.bias.copy_(displacement)
    local = torch.randn(4, 16, 6).requires_grad_()
    bbox = torch.tensor([
        [.15, -.2, .3, 0.],
        [.15, -.2, .3, .7],
        [.15, -.2, .3, -.7],
        [.15, -.2, .3, 1e-8],
    ], requires_grad=True)
    points = local[..., :3] * bbox[:, None, 3:] + bbox[:, None, :3]
    edge = PrimitiveGeometry(local, bbox, points)
    face = point_feedback_geometry(torch.randn(2, 3, 3, 3))
    result = block(edge, face, torch.randn(4, 8), torch.randn(2, 4), torch.tensor([[0, 1]] * 4))
    torch.testing.assert_close(result.local_points[0], local[0], rtol=0, atol=0)
    torch.testing.assert_close(result.points[0], points[0], rtol=0, atol=0)
    torch.testing.assert_close(result.center_scale, bbox, rtol=0, atol=0)
    torch.testing.assert_close(result.local_points[..., 3:], local[..., 3:], rtol=0, atol=0)
    # Ordinary, negative, and very small nonzero scales all retain the full delta.
    torch.testing.assert_close(result.points[1:], points[1:] + displacement, rtol=0, atol=0)
    torch.testing.assert_close(
        result.local_points[1:, :, :3],
        local[1:, :, :3] + displacement / bbox[1:, None, 3:], rtol=0, atol=0,
    )
    reconstructed = result.local_points[..., :3] * bbox[:, None, 3:] + bbox[:, None, :3]
    torch.testing.assert_close(result.points, reconstructed, rtol=2e-6, atol=2e-7)
    collapsed_delta_grad = torch.autograd.grad(result.points[0].sum(), block.delta.bias, retain_graph=True)[0]
    assert torch.count_nonzero(collapsed_delta_grad) == 0
    loss = result.local_points.abs().mean() + result.points.square().mean()
    loss.backward()
    for value in (local, bbox, *block.parameters()):
        assert value.grad is not None and torch.isfinite(value.grad).all()
    assert local.grad[0].abs().sum() > 0
    assert bbox.grad[0, 3].abs() > 0


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), -float("inf")])
def test_point_feedback_rejects_nonfinite_coordinate_scale(scale):
    block = CurvePointFeedback(4, 8, hidden_dim=16, feature_dim=8, geometry_chunk_size=2)
    edge = point_feedback_geometry(torch.randn(2, 16, 3))
    face = point_feedback_geometry(torch.randn(2, 3, 3, 3))
    edge.center_scale[0, 3] = scale
    with pytest.raises(ValueError, match="scale"):
        block(edge, face, torch.randn(2, 8), torch.randn(2, 4), torch.tensor([[0, 1], [1, 0]]))
