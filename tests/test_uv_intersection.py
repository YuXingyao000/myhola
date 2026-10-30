"""UV decoder contracts: geometry, ownership, gradients, and mixed precision."""

import numpy as np
import pytest
import torch

from src.models.vae.uv_intersection import (
    BSplineBasis,
    CurveGeometryLoss,
    DualUVHead,
    EdgeSupervision,
    FixedBSplineSurface,
    UVDecoderConfig,
    UVDecoderLoss,
    UVIntersectionDecoder,
)


def small_decoder():
    torch.manual_seed(19)
    config = UVDecoderConfig(
        latent_dim=64, face_width=64, face_layers=1, face_heads=4, face_ffn=96,
        curve_width=32, curve_layers=1, curve_heads=4, curve_ffn=64,
        dropout=0.0, pair_chunk_size=2,
    )
    return UVIntersectionDecoder(config)


def reconstruction_targets(prediction, edge_pairs, edge_xyz):
    target = {
        name: prediction[name].detach().clone()
        for name in ("face_local", "face_center_scale", "face_xyz")
    }
    return dict(target, edge_pairs=edge_pairs, edge_xyz=edge_xyz)



@pytest.mark.parametrize("target_pairs", [
    [[-2, 3]], [[2, 4]], [[1, 1]], [[1, 0]], [[0, 1], [0, 1]], [[0, 2]],
], ids=["negative-owner", "out-of-range", "self-edge", "reversed", "multiple-edge", "cross-object"])
def test_edge_supervision_rejects_unsupported_target_ownership(target_pairs):
    candidates = torch.tensor([[0, 1], [2, 3]])  # Two independent two-face objects.
    with pytest.raises(ValueError):
        EdgeSupervision()(candidates, torch.tensor(target_pairs), num_faces=4)


def test_surface_v_u_grid_recovers_asymmetric_plane_and_endpoint_derivatives():
    surface = FixedBSplineSurface()
    grid = torch.linspace(0, 1, 16, dtype=torch.double)
    v, u = torch.meshgrid(grid, grid, indexing="ij")
    samples = torch.stack((2 * u + v, -u + 3 * v, 2 + 0 * u), -1)[None]
    uv = torch.tensor(
        [[[0., 0.], [0., 1.], [1., 0.], [1., 1.], [.23, .71]]],
        dtype=torch.double, requires_grad=True,
    )
    xyz = surface.evaluate(surface.fit(samples), uv)
    expected = torch.stack((2 * uv[..., 0] + uv[..., 1],
                            -uv[..., 0] + 3 * uv[..., 1], 2 + 0 * uv[..., 0]), -1)
    torch.testing.assert_close(xyz, expected, rtol=1e-11, atol=1e-11)
    derivative = torch.autograd.grad(xyz[..., 0].sum(), uv)[0]
    torch.testing.assert_close(derivative, uv.new_tensor([2., 1.]).expand_as(uv))
    sample_uv = torch.stack((u, v), -1).reshape(1, 256, 2)
    torch.testing.assert_close(surface.evaluate(surface.fit(samples), sample_uv).reshape_as(samples), samples)


def test_basis_matches_scipy_values_and_derivatives_including_endpoints():
    scipy_interpolate = pytest.importorskip("scipy.interpolate")
    basis = BSplineBasis(16, 3)
    uv = torch.linspace(0, 1, 37, dtype=torch.double, requires_grad=True)
    result = basis(uv)
    derivative = torch.stack([
        torch.autograd.grad(result[:, index].sum(), uv, retain_graph=True)[0]
        for index in range(16)
    ], -1)
    oracle = scipy_interpolate.BSpline(basis.knots.numpy(), np.eye(16), 3)
    np.testing.assert_allclose(result.detach().numpy(), oracle(uv.detach().numpy()), atol=1e-13)
    np.testing.assert_allclose(derivative.detach().numpy(), oracle.derivative()(uv.detach().numpy()), atol=1e-12)
    torch.testing.assert_close(result.sum(-1), torch.ones_like(uv))


def test_surface_gradcheck_samples_controls_and_uv():
    torch.manual_seed(3)
    surface = FixedBSplineSurface(5, 3)
    samples = torch.randn(1, 5, 5, 3, dtype=torch.double, requires_grad=True)
    poles = torch.randn(1, 2, 5, 5, 3, dtype=torch.double, requires_grad=True)
    uv = (torch.rand(1, 2, 3, 2, dtype=torch.double) * .8 + .1).requires_grad_()
    assert torch.autograd.gradcheck(surface.fit, (samples,), atol=1e-5)
    assert torch.autograd.gradcheck(surface.evaluate, (poles, uv), atol=1e-5)
    query = uv[:, 0].detach().requires_grad_()
    assert torch.autograd.gradcheck(lambda points, coords: surface.evaluate(surface.fit(points), coords),
                                    (samples, query), atol=1e-5)


def test_packed_objects_cannot_read_or_form_edges_across_object_boundaries():
    model = small_decoder().eval()
    latent = torch.randn(4, 64)
    counts = torch.tensor([2, 2])
    with torch.no_grad():
        original = model(latent, counts)
        changed_latent = latent.clone()
        changed_latent[2:] = torch.randn_like(latent[2:]) * 5
        changed = model(changed_latent, counts)
    assert original["edge_pairs"].tolist() == [[0, 1], [2, 3]]
    torch.testing.assert_close(original["face_xyz"][:2], changed["face_xyz"][:2])
    torch.testing.assert_close(original["edge_uv"][:1], changed["edge_uv"][:1])
    torch.testing.assert_close(original["edge_logits"][:1], changed["edge_logits"][:1])
    assert not torch.allclose(original["face_xyz"][2:], changed["face_xyz"][2:])


def test_geometry_reaches_uv_and_surfaces_only_for_positive_edges_and_bce_covers_all_pairs():
    model = small_decoder()
    prediction = model(torch.randn(3, 64), torch.tensor([3]))
    target_curve = prediction["edge_xyz"][:1].mean(1).detach() + .2
    target = reconstruction_targets(prediction, torch.tensor([[0, 1]]), target_curve)
    losses = UVDecoderLoss()(prediction, target)
    geometry = losses["curve_xyz"] + losses["curve_incidence"]
    uv_grad, face_grad, curve_grad = torch.autograd.grad(
        geometry, (prediction["edge_uv"], prediction["face_xyz"], prediction["edge_xyz"]),
        retain_graph=True,
    )
    assert uv_grad[0].abs().sum() > 0 and face_grad[:2].abs().sum() > 0
    assert torch.count_nonzero(uv_grad[1:]) == 0
    assert torch.count_nonzero(curve_grad[1:]) == 0
    assert torch.count_nonzero(face_grad[2:]) == 0
    logits_grad = torch.autograd.grad(losses["edge_existence"], prediction["edge_logits"], retain_graph=True)[0]
    assert logits_grad[0] < 0 and torch.all(logits_grad[1:] > 0)
    losses["total"].backward()
    assert model.intersection.uv_head.output[-1].weight.grad.abs().sum() > 0
    assert model.face_decoder.points.output.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_geometry_loss_uses_common_curve_orientation_for_both_sides():
    t = torch.linspace(0, 1, 16)
    target = torch.stack((t, t.square(), t.pow(3)), -1)[None]
    loss = CurveGeometryLoss()
    shared_reverse = target.flip(1)[:, None].expand(-1, 2, -1, -1)
    torch.testing.assert_close(loss(shared_reverse, target)["curve_xyz"], torch.tensor(0.))
    independent_reverse = torch.stack((target, target.flip(1)), dim=1)
    assert loss(independent_reverse, target)["curve_xyz"] > 0
    assert loss(independent_reverse, target)["curve_incidence"] > 0


def test_two_opposite_curve_errors_cannot_cancel_in_geometry_supervision():
    target = torch.zeros(1, 16, 3)
    offset = torch.tensor([.3, -.2, .4])
    predicted = torch.stack((target + offset, target - offset), dim=1)
    torch.testing.assert_close(predicted.mean(1), target)
    losses = CurveGeometryLoss()(predicted, target)
    assert losses["curve_xyz"] > 0 and losses["curve_incidence"] > 0


def test_single_face_empty_edge_set_has_finite_loss_and_backward():
    model = small_decoder()
    prediction = model(torch.randn(1, 64), torch.tensor([1]))
    assert prediction["edge_pairs"].shape == (0, 2)
    assert prediction["edge_xyz"].shape == (0, 2, 16, 3)
    target = reconstruction_targets(prediction, torch.empty(0, 2, dtype=torch.long), torch.empty(0, 16, 3))
    target["face_xyz"] = target["face_xyz"] + .1
    losses = UVDecoderLoss()(prediction, target)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert model.face_decoder.points.output.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_uv_head_keeps_float32_coordinates_inside_cuda_bfloat16_autocast():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support BF16")
    head = DualUVHead(32).cuda()
    queries = torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        uv = head(queries)
    assert uv.dtype == torch.float32 and uv.shape == (2, 2, 16, 2)
    assert torch.all((uv > 0) & (uv < 1))
    uv.square().sum().backward()
    assert torch.isfinite(queries.grad).all() and queries.grad.abs().sum() > 0
