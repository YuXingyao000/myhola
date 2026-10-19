"""Shared XYZ / owner UV semantics, checkpoint migration and gradient paths."""
from copy import deepcopy

import pytest
import torch

from src.brepnet.models.vae import AutoEncoder_intersection, _map_directed_targets
from src.brepnet.models.vae.decoders.primitives import SharedCurveFusion, CurveConditionedUVPathHead
from src.brepnet.models.vae.losses.intersection import IntersectionLoss
from src.brepnet.test.test_intersection_vae import encoder_batch
from src.brepnet.test.test_uv_boundary_model import uv_config, add_targets, uv_case


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def config(**changes):
    return uv_config(shared_curve={"enabled": True, "hidden_dim": 16}, **changes)


def test_shared_fusion_retains_canonical_initialization_and_learns_from_both_sides():
    fusion = SharedCurveFusion(8, 16)
    a, b = (torch.randn(5, 8, requires_grad=True) for _ in range(2))
    torch.testing.assert_close(fusion(a, b), a, rtol=0, atol=0)
    # The residual output learns first; after it opens, both inputs contribute.
    torch.nn.init.normal_(fusion.residual[-1].weight)
    fusion(a, b).square().sum().backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0


def test_single_curve_decode_and_reciprocal_uv_use_one_parameterization():
    model = AutoEncoder_intersection(config()).eval()
    batches = []
    handle = model.decoder.edge_geometry.register_forward_hook(lambda module, args, out: batches.append(len(args[0])))
    counts = torch.tensor([2, 3])
    output = model.decode_state(torch.randn(5, model.face_feature_dim), counts)
    handle.remove()
    p = len(output.pairs.indices)
    assert sum(batches) == p + p // 2  # P coarse predictions, K final curves
    reverse = output.pairs.reverse
    geometry = output.refined.edge_geometry
    torch.testing.assert_close(geometry.points, geometry.points[reverse].flip(1), rtol=0, atol=0)
    torch.testing.assert_close(geometry.local_points, geometry.local_points[reverse].flip(1), rtol=0, atol=0)
    torch.testing.assert_close(geometry.center_scale, geometry.center_scale[reverse], rtol=0, atol=0)
    torch.testing.assert_close(output.edge_uv, output.edge_uv[reverse].flip((1, 2)), rtol=0, atol=0)
    assert output.edge_uv.shape == (8, 2, 16, 2)
    exported = model.export_items(output, torch.randn(5, model.face_feature_dim))
    assert len(exported) == 2
    assert all(item["representation"] == "legacy_half_edges" for item in exported)


def test_initial_curve_matches_canonical_old_head_for_each_face_order():
    model = AutoEncoder_intersection(config()).eval()
    latent = torch.randn(4, model.face_feature_dim)
    order = torch.tensor([3, 0, 2, 1])
    for values in (latent, latent[order]):
        with torch.no_grad():
            result = model.decode_state(values)
            pairs = result.pairs.indices
            canonical = pairs[:, 0] < pairs[:, 1]
            original = model.decoder._decode_stage(
                result.refined.face_features, result.refined.edge_features,
                result.pairs, condition_curves=True,
            )
        torch.testing.assert_close(result.refined.edge_geometry.points[canonical],
                                   original.edge_geometry.points[canonical], atol=1e-6, rtol=1e-5)
        # Canonical selection uses face indices. Arbitrary face reindexing may
        # select the other old half-edge; exact permutation equivariance is not claimed.


def test_uv_reads_actual_curve_xyz_with_no_detach_and_chunking_agrees():
    head = CurveConditionedUVPathHead(4, 8, hidden_dim=16, num_layers=2, nhead=4,
                                     dim_feedforward=32, candidate_chunk_size=2)
    edges, shared = (torch.randn(4, 8, requires_grad=True) for _ in range(2))
    faces = torch.randn(3, 4, requires_grad=True)
    owners = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]])
    curves = torch.randn(4, 16, 3, requires_grad=True)
    result = head(edges, shared, faces, owners, curves)
    whole = deepcopy(head)
    whole.chunk_size = 100
    torch.testing.assert_close(whole(edges, shared, faces, owners, curves), result, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(head(edges, shared, faces, owners, curves + .25), result)
    result.square().mean().backward()
    for tensor in (edges, shared, faces, curves):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0


def test_empty_owner_uv_connects_every_parameter():
    head = CurveConditionedUVPathHead(4, 8, hidden_dim=16, num_layers=1, nhead=4, dim_feedforward=32)
    output = head(torch.empty(0, 8), torch.empty(0, 8), torch.randn(1, 4),
                  torch.empty(0, 2, dtype=torch.long), torch.empty(0, 16, 3))
    assert output.shape == (0, 16, 2)
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_canonical_targets_follow_connection_rows_and_preserve_closed_uv_branches():
    case = uv_case()
    output, batch = case[:2]
    # Closed representative, reciprocal sampled from a different start phase.
    t = torch.linspace(0, 2 * torch.pi, 16)
    circle = torch.stack((t.cos(), t.sin(), torch.zeros_like(t)), -1)
    batch["edge_points"][0, :, :3] = circle
    batch["edge_points"][1, :, :3] = circle.roll(3, 0)
    batch["edge_uv_targets"]["uv"][0, :, :, 0] = torch.linspace(0, 1, 16)
    batch["edge_uv_targets"]["valid"][0, 0, 3] = False
    # Shuffle connectivity rows independently of edge array IDs.
    batch["edge_face_connectivity"] = batch["edge_face_connectivity"].flip(0)
    batch["edge_uv_targets"] = {key: values.flip(0) for key, values in batch["edge_uv_targets"].items()}
    original = deepcopy(batch)
    targets = _map_directed_targets(output, batch)
    aligned = IntersectionLoss._shared_curve_batch(output, batch, targets)
    torch.testing.assert_close(aligned["edge_points"][0, :, :3], circle)
    torch.testing.assert_close(aligned["edge_points"][1], aligned["edge_points"][0].flip(0))
    uv = aligned["edge_uv_targets"]["uv"]
    assert uv[1, 0, 0, 0] == 0 and uv[1, 0, -1, 0] == 1  # no modulo/closed resampling
    torch.testing.assert_close(uv[0], uv[1].flip((0, 1)))
    assert not aligned["edge_uv_targets"]["valid"][0, 1, 12]
    for key in ("edge_points", "edge_norm", "edge_bbox"):
        torch.testing.assert_close(batch[key], original[key], rtol=0, atol=0)


@pytest.mark.parametrize("connected,face_counts", [(True, (2,)), (False, (1,)), (False, (3,))])
def test_shared_loss_handles_empty_edges_and_invalid_uv(connected, face_counts):
    case = uv_case(connected=connected, face_counts=face_counts)
    case[1]["edge_uv_targets"]["valid"].zero_()
    losses = IntersectionLoss({}, True, True)(*case, torch.tensor(0.), True)
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses["total_loss"].backward()


def test_disabled_shared_curve_is_bitwise_compatible_with_previous_uv_model():
    torch.manual_seed(81)
    old = AutoEncoder_intersection(uv_config()).eval()
    old_rng = torch.rand(3)
    torch.manual_seed(81)
    disabled = AutoEncoder_intersection(uv_config(shared_curve={"enabled": False})).eval()
    torch.testing.assert_close(torch.rand(3), old_rng, rtol=0, atol=0)
    assert old.state_dict().keys() == disabled.state_dict().keys()
    for key, value in old.state_dict().items():
        torch.testing.assert_close(disabled.state_dict()[key], value, rtol=0, atol=0)
    z = torch.randn(3, old.face_feature_dim)
    with torch.no_grad():
        a, b = old.decode_state(z), disabled.decode_state(z)
    torch.testing.assert_close(a.refined.edge_geometry.points, b.refined.edge_geometry.points, rtol=0, atol=0)
    torch.testing.assert_close(a.edge_uv, b.edge_uv, rtol=0, atol=0)


def test_trained_uv_migration_retains_backbone_and_full_reload_is_strict(tmp_path):
    old = AutoEncoder_intersection(uv_config()).eval()
    source = tmp_path / "uv.ckpt"
    torch.save({"state_dict": {"model." + k: v for k, v in old.state_dict().items()},
                "epoch": 176, "global_step": 64251}, source)
    model = AutoEncoder_intersection(config(init_shared_curve_checkpoint=str(source), freeze_encoder=True)).eval()
    changed = set(model.initialization_report["mapped_uv_keys"])
    for key, value in old.state_dict().items():
        if key not in changed:
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    e, f = model.decoder.edge_dim, model.decoder.face_dim
    context = model.decoder.uv_boundary.context[0].weight
    torch.testing.assert_close(context[:, :e], old.decoder.uv_boundary.context[0].weight[:, :e])
    torch.testing.assert_close(context[:, 2*e:], old.decoder.uv_boundary.context[0].weight[:, e:e+f])
    assert context[:, e:2*e].count_nonzero() == 0
    batch = add_targets(encoder_batch())
    model.train()
    losses, _ = model(batch)
    losses["total_loss"].backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    for parameter in (model.decoder.shared_curve.residual[-1].weight,
                      model.decoder.uv_boundary.curve_embedding[0].weight,
                      model.decoder.refiner.edge_delta.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    torch.testing.assert_close(losses["weighted_uv"], losses["uv"] * .01)
    torch.testing.assert_close(losses["weighted_uv_anchor"], losses["uv_anchor"] * .1)
    saved = tmp_path / "shared.ckpt"
    torch.save(model.state_dict(), saved)
    loaded = AutoEncoder_intersection(config(checkpoint=str(saved))).eval()
    resumed = AutoEncoder_intersection(config(restoring_checkpoint=True)).eval()
    resumed.load_state_dict(torch.load(saved, weights_only=False), strict=True)
    z = torch.randn(3, model.face_feature_dim)
    with torch.no_grad():
        a, b, c = [m.eval().decode_state(z) for m in (model, loaded, resumed)]
    torch.testing.assert_close(a.edge_uv, b.edge_uv, rtol=0, atol=0)
    torch.testing.assert_close(a.edge_uv, c.edge_uv, rtol=0, atol=0)
    with pytest.raises(ValueError, match="complete previous"):
        AutoEncoder_intersection(config(init_shared_curve_checkpoint=str(saved)))
    broken = model.state_dict()
    broken.pop("decoder.uv_boundary.curve_embedding.0.weight")
    torch.save(broken, saved)
    with pytest.raises(RuntimeError, match="Missing key"):
        AutoEncoder_intersection(config(checkpoint=str(saved)))
