"""Projection/gradient contracts; these do not establish CAD validity."""

import pytest
import torch

from src.models.blocks import closest_polyline_points, closest_surface_points


def plane_grid(dtype=torch.float32, device="cpu"):
    return torch.tensor(
        [[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]],
        dtype=dtype, device=device,
    )


def test_sparse_plane_interior_projects_to_surface_not_grid_vertices():
    points = torch.tensor([[[0.2, 0.4, 0.0], [0.8, 0.3, 0.0], [0.3, 0.7, 0.5]]])
    closest, distances = closest_surface_points(points, plane_grid())
    expected = points.clone()
    expected[..., 2] = 0.0
    torch.testing.assert_close(closest, expected)
    torch.testing.assert_close(distances, torch.tensor([[0.0, 0.0, 0.5]]), atol=1e-7, rtol=0.0)


def test_outside_grid_uses_edges_and_vertices():
    points = torch.tensor([[[-0.2, 0.4, 0.3], [1.4, 1.3, 0.0]]])
    closest, distances = closest_surface_points(points, plane_grid())
    torch.testing.assert_close(closest, torch.tensor([[[0.0, 0.4, 0.0], [1.0, 1.0, 0.0]]]))
    torch.testing.assert_close(distances, torch.tensor([[0.13 ** 0.5, 0.5]]))


@pytest.mark.parametrize("collapsed", [False, True])
def test_degenerate_surface_is_finite_and_differentiable(collapsed):
    surfaces = torch.zeros(1, 2, 2, 3)
    if not collapsed:
        surfaces[0, :, :, 0] = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    surfaces.requires_grad_()
    points = torch.tensor([[[0.5, 1.0, 0.0], [4.0, 0.0, 0.0]]], requires_grad=True)
    closest, distances = closest_surface_points(points, surfaces)
    expected = torch.zeros_like(points) if collapsed else torch.tensor([[[0.5, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    torch.testing.assert_close(closest, expected)
    distances.sum().backward()
    assert torch.isfinite(points.grad).all()
    assert torch.isfinite(surfaces.grad).all()


def test_surface_distance_gradients_reach_points_and_surface():
    points = torch.tensor([[[0.2, 0.4, 0.3]]], requires_grad=True)
    surfaces = plane_grid().requires_grad_()
    _, distances = closest_surface_points(points, surfaces)
    distances.sum().backward()
    for gradient in (points.grad, surfaces.grad):
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    torch.testing.assert_close(points.grad.sum((0, 1)), -surfaces.grad.sum((0, 1, 2)))


def test_zero_surface_distance_has_finite_gradient():
    points = torch.tensor([[[0.25, 0.5, 0.0]]], requires_grad=True)
    surfaces = plane_grid().requires_grad_()
    closest, distances = closest_surface_points(points, surfaces)
    assert distances.item() == 0.0
    (closest.sum() + distances.sum()).backward()
    assert torch.isfinite(points.grad).all()
    assert torch.isfinite(surfaces.grad).all()


def test_surface_selected_projection_passes_gradcheck():
    points = torch.tensor([[[0.2, 0.4, 0.3]]], dtype=torch.float64, requires_grad=True)
    surfaces = plane_grid(torch.float64).requires_grad_()
    assert torch.autograd.gradcheck(closest_surface_points, (points, surfaces), atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_unused_skinny_triangle_plane_has_finite_backward(dtype):
    # The nearest point is an edge endpoint. The unused plane's barycentric
    # weights are enormous: its unmasked division backward previously made
    # surface gradients NaN although the forward projection was finite.
    surfaces = torch.tensor(
        [[[[0.0, 0.0, 0.0], [5e-5, 1e-11, 0.0]],
          [[1e-4, 0.0, 0.0], [1e-4, 2e-11, 0.0]]]],
        dtype=dtype, requires_grad=True,
    )
    points = torch.tensor([[[0.5, -0.2, 0.1]]], dtype=dtype, requires_grad=True)
    closest, distances = closest_surface_points(points, surfaces)
    torch.testing.assert_close(closest, surfaces.detach()[:, 1, :1].float())
    distances.sum().backward()
    assert points.grad.isfinite().all()
    assert surfaces.grad.isfinite().all()
    assert surfaces.grad.abs().sum() > 0


@pytest.mark.parametrize("scale, aspect", [(1e-8, 1.0), (1e-7, 1e-3)])
def test_tiny_selected_plane_matches_double_precision_gradients(scale, aspect):
    # Both cases have area_squared**2 below float32's representable range;
    # the selected plane must remain differentiable, not detached or clamped.
    points = torch.tensor([[[0.4, 0.2 * aspect, 0.3 * aspect]]]) * scale
    surfaces = plane_grid() * torch.tensor([scale, scale * aspect, scale])
    results = []
    for dtype in (torch.float32, torch.float64):
        typed_points = points.to(dtype).detach().requires_grad_()
        typed_surfaces = surfaces.to(dtype).detach().requires_grad_()
        closest, distances = closest_surface_points(typed_points, typed_surfaces)
        gradients = torch.autograd.grad(closest.sum() + distances.sum(), (typed_points, typed_surfaces))
        results.append((closest, distances, *gradients))
    for actual, reference in zip(*results):
        assert actual.isfinite().all()
        # Barycentric derivatives amplify roundoff with the aspect ratio.
        gradient_atol = max(2e-6, 2 * torch.finfo(torch.float32).eps / aspect)
        torch.testing.assert_close(actual, reference.float(), atol=gradient_atol, rtol=2e-4)
    torch.testing.assert_close(results[0][0][..., :2], points[..., :2], atol=scale * 1e-6, rtol=1e-5)
    torch.testing.assert_close(results[0][0][..., 2], torch.zeros_like(points[..., 2]))


def test_scaled_skinny_selected_plane_passes_gradcheck():
    points = torch.tensor([[[0.4, 0.2, 0.3]]], dtype=torch.float64, requires_grad=True)
    surfaces = plane_grid(torch.float64).requires_grad_()
    scale = torch.tensor([1e-7, 1e-10, 1e-7], dtype=torch.float64)

    def scaled_projection(unit_points, unit_surfaces):
        closest, distances = closest_surface_points(unit_points * scale, unit_surfaces * scale)
        return closest / scale, distances / scale[0]

    assert torch.autograd.gradcheck(scaled_projection, (points, surfaces), eps=1e-7, atol=1e-5)


def test_surface_batch_chunks_preserve_values_and_gradients():
    points = torch.tensor([[[0.2, 0.4, 0.3]], [[1.2, 0.4, 0.3]], [[0.1, 0.8, -0.3]]], requires_grad=True)
    surfaces = plane_grid().expand(3, -1, -1, -1).clone().requires_grad_()
    one_at_a_time = closest_surface_points(points, surfaces, chunk_size=1)
    together = closest_surface_points(points, surfaces, chunk_size=8)
    for first, second in zip(one_at_a_time, together):
        torch.testing.assert_close(first, second)
    gradients_one = torch.autograd.grad(one_at_a_time[1].sum(), (points, surfaces))
    gradients_together = torch.autograd.grad(together[1].sum(), (points, surfaces))
    for first, second in zip(gradients_one, gradients_together):
        torch.testing.assert_close(first, second)


@pytest.mark.parametrize("batch, samples", [(0, 4), (2, 0)])
def test_empty_inputs_preserve_shapes_and_gradient_connections(batch, samples):
    points = torch.empty(batch, samples, 3, requires_grad=True)
    surfaces = torch.empty(batch, 2, 2, 3, requires_grad=True)
    closest, distances = closest_surface_points(points, surfaces)
    assert closest.shape == (batch, samples, 3)
    assert distances.shape == (batch, samples)
    (closest.sum() + distances.sum()).backward()
    assert points.grad is not None and surfaces.grad is not None
    polylines = torch.empty(batch, 3, 3)
    closest, indices, fractions = closest_polyline_points(points.detach(), polylines)
    assert closest.shape == (batch, samples, 3)
    assert indices.shape == fractions.shape == (batch, samples)
    assert indices.dtype == torch.long


def test_polyline_mapping_reconstructs_selected_points():
    points = torch.tensor([[[0.25, 0.2, 0.0], [1.3, 0.75, 0.0], [-1.0, 0.0, 0.0]]])
    polylines = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    closest, indices, fractions = closest_polyline_points(points, polylines)
    torch.testing.assert_close(indices, torch.tensor([[0, 1, 0]]))
    torch.testing.assert_close(fractions, torch.tensor([[0.25, 0.75, 0.0]]))
    gather_indices = indices[..., None].expand(-1, -1, 3)
    starts = polylines.gather(1, gather_indices)
    ends = polylines.gather(1, gather_indices + 1)
    torch.testing.assert_close(closest, starts + fractions[..., None] * (ends - starts))


def test_degenerate_polyline_and_projection_gradients():
    points = torch.tensor([[[0.25, 0.2, 0.0], [0.0, 0.0, 0.0]]], requires_grad=True)
    line = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], requires_grad=True)
    closest, indices, fractions = closest_polyline_points(points, line)
    torch.testing.assert_close(closest, torch.tensor([[[0.25, 0.0, 0.0], [0.0, 0.0, 0.0]]]))
    assert indices[0, 0] == 1
    assert fractions.isfinite().all()
    torch.linalg.vector_norm(points - closest, dim=-1).sum().backward()
    for gradient in (points.grad, line.grad):
        assert gradient.isfinite().all()
        assert gradient.abs().sum() > 0


@pytest.mark.parametrize("length", [1e-13, 1e-18])
def test_tiny_segment_endpoint_clamps_have_finite_backward(length):
    points = torch.tensor([[[-0.5, 0.2, 0.1], [0.5, 0.2, 0.1]]], requires_grad=True)
    line = torch.tensor([[[0.0, 0.0, 0.0], [length, 0.0, 0.0]]], requires_grad=True)
    closest, _, fractions = closest_polyline_points(points, line)
    torch.testing.assert_close(closest, line.detach(), atol=0, rtol=0)
    torch.testing.assert_close(fractions, torch.tensor([[0.0, 1.0]]))
    torch.linalg.vector_norm(points - closest, dim=-1).sum().backward()
    assert points.grad.isfinite().all()
    assert line.grad.isfinite().all()
    assert (line.grad.abs().sum(dim=-1) > 0).all()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_amp_geometry_is_float32_and_has_finite_gradients(device):
    input_dtype = torch.bfloat16 if device == "cpu" else torch.float16
    points = torch.tensor([[[0.25, 0.5, 0.25]]], device=device, dtype=input_dtype, requires_grad=True)
    surfaces = plane_grid(dtype=input_dtype, device=device).requires_grad_()
    with torch.autocast(device_type=device, dtype=input_dtype):
        closest, distances = closest_surface_points(points, surfaces)
        line_closest, _, fractions = closest_polyline_points(points, surfaces[:, 0])
    for result in (closest, distances, line_closest, fractions):
        assert result.dtype == torch.float32
        assert result.isfinite().all()
    (distances.sum() + line_closest.sum()).backward()
    assert points.grad.isfinite().all()
    assert surfaces.grad.isfinite().all()
