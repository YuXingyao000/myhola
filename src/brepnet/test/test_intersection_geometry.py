"""Projection/gradient contracts; these do not establish CAD validity."""

import pytest
import torch

from src.brepnet.models.blocks import closest_polyline_points, closest_surface_points


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


def test_invalid_geometry_contracts_are_explicit():
    with pytest.raises(ValueError, match="U,V >= 2"):
        closest_surface_points(torch.zeros(1, 2, 3), torch.zeros(1, 1, 2, 3))
    with pytest.raises(ValueError, match="batch size"):
        closest_surface_points(torch.zeros(2, 2, 3), plane_grid())
    with pytest.raises(ValueError, match="positive integer"):
        closest_surface_points(torch.zeros(1, 2, 3), plane_grid(), chunk_size=0)
    with pytest.raises(ValueError, match="K >= 2"):
        closest_polyline_points(torch.zeros(1, 2, 3), torch.zeros(1, 1, 3))
