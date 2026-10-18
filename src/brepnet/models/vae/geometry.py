"""Differentiable curve and sampled-surface projection used by VAE supervision."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_geometry_pair(points: Tensor, geometry: Tensor) -> torch.dtype:
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B, L, 3].")
    if points.shape[0] != geometry.shape[0]:
        raise ValueError("points and geometry must have the same batch size.")
    if points.device != geometry.device:
        raise ValueError("points and geometry must be on the same device.")
    if not points.is_floating_point() or not geometry.is_floating_point():
        raise TypeError("Geometry coordinates must be floating point tensors.")
    return torch.float64 if torch.float64 in (points.dtype, geometry.dtype) else torch.float32


def _project_to_segments(points: Tensor, starts: Tensor, ends: Tensor) -> tuple[Tensor, Tensor]:
    """Broadcast over leading dimensions; collapsed segments project to starts."""
    directions = ends - starts
    squared_lengths = directions.square().sum(dim=-1)
    valid = squared_lengths > torch.finfo(points.dtype).tiny
    denominators = torch.where(valid, squared_lengths, torch.ones_like(squared_lengths))
    numerators = ((points - starts) * directions).sum(dim=-1)
    with torch.no_grad():
        fractions = numerators / denominators
        interior = valid & (fractions >= 0.0) & (fractions <= 1.0)
        fractions = torch.where(valid, fractions.clamp(0.0, 1.0), torch.zeros_like(fractions))
    if torch.is_grad_enabled():
        # A clamped endpoint has no derivative through its fraction. Mask before
        # division: masking afterwards can still produce 0 * inf in backward
        # for a tiny segment and a distant point. Keep endpoint clamp semantics.
        interior_fractions = torch.where(interior, numerators, 0.0) / torch.where(
            interior, denominators, 1.0
        )
        fractions = torch.where(interior, interior_fractions, fractions)
    return starts + fractions[..., None] * directions, fractions


def _project_to_triangles(points: Tensor, triangles: Tensor) -> Tensor:
    """Choose a plane projection inside the triangle or the nearest edge point.

    Degenerate triangles use their three segments. Relative area guards prevent
    divisions by zero without imposing an absolute minimum triangle size.
    """
    vertex_a, vertex_b, vertex_c = triangles.unbind(dim=-2)
    side_b = vertex_b - vertex_a
    side_c = vertex_c - vertex_a
    normal = torch.linalg.cross(side_b, side_c, dim=-1)
    area_squared = normal.square().sum(dim=-1)
    side_product = side_b.square().sum(dim=-1) * side_c.square().sum(dim=-1)
    dtype_info = torch.finfo(points.dtype)
    valid = area_squared > (dtype_info.eps ** 2 * side_product).clamp_min(dtype_info.tiny)
    denominator = torch.where(valid, area_squared, torch.ones_like(area_squared))

    relative = points - vertex_a
    numerator_b = (torch.linalg.cross(relative, side_c, dim=-1) * normal).sum(dim=-1)
    numerator_c = (torch.linalg.cross(side_b, relative, dim=-1) * normal).sum(dim=-1)
    with torch.no_grad():
        weight_b = numerator_b / denominator
        weight_c = numerator_c / denominator
        inside = valid & (weight_b >= 0.0) & (weight_c >= 0.0) & (weight_b + weight_c <= 1.0)
    if torch.is_grad_enabled():
        # The chosen plane remains differentiable. Unused plane projections can
        # have huge barycentric weights on thin triangles; exclude their unsafe
        # divisions from backward instead of relying on the final torch.where.
        selected_denominator = torch.where(inside, denominator, 1.0)
        weight_b = torch.where(inside, numerator_b, 0.0) / selected_denominator
        weight_c = torch.where(inside, numerator_c, 0.0) / selected_denominator
    plane_point = vertex_a + weight_b[..., None] * side_b + weight_c[..., None] * side_c

    edge_ab, _ = _project_to_segments(points, vertex_a, vertex_b)
    edge_bc, _ = _project_to_segments(points, vertex_b, vertex_c)
    edge_ca, _ = _project_to_segments(points, vertex_c, vertex_a)
    edge_points = torch.stack((edge_ab, edge_bc, edge_ca), dim=-2)
    nearest_edge = (points[..., None, :] - edge_points).square().sum(dim=-1).argmin(dim=-1)
    nearest_edge_point = edge_points.gather(
        -2, nearest_edge[..., None, None].expand(*nearest_edge.shape, 1, 3)
    ).squeeze(-2)
    return torch.where(inside[..., None], plane_point, nearest_edge_point)


def _surface_triangle_indices(num_u: int, num_v: int, device: torch.device) -> Tensor:
    """Split each UV cell along its (u,v) to (u+1,v+1) diagonal."""
    grid = torch.arange(num_u * num_v, device=device).reshape(num_u, num_v)
    corner_a = grid[:-1, :-1].reshape(-1)
    corner_b = grid[1:, :-1].reshape(-1)
    corner_c = grid[:-1, 1:].reshape(-1)
    corner_d = grid[1:, 1:].reshape(-1)
    first = torch.stack((corner_a, corner_b, corner_d), dim=-1)
    second = torch.stack((corner_a, corner_d, corner_c), dim=-1)
    return torch.cat((first, second), dim=0)


def closest_surface_points(
    points: Tensor,
    surfaces: Tensor,
    chunk_size: int = 32,
) -> tuple[Tensor, Tensor]:
    """Project [B,L,3] points onto triangulated [B,U,V,3] surface grids.

    Returns closest points [B,L,3] and Euclidean distances [B,L]. These are exact
    for the piecewise linear grid, not for the unknown smooth CAD surface.
    ``chunk_size`` bounds the number of batch items in each nearest-triangle
    search; its largest intermediate is proportional to chunk_size * L * U * V.
    Half/bfloat16 inputs are evaluated and returned in float32, even under AMP.
    Gradients propagate through the selected triangle to both inputs; discrete
    triangle changes are not differentiated. Zero distances have finite gradients.
    """
    if surfaces.ndim != 4 or surfaces.shape[-1] != 3 or min(surfaces.shape[1:3]) < 2:
        raise ValueError("surfaces must have shape [B, U, V, 3] with U,V >= 2.")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    dtype = _validate_geometry_pair(points, surfaces)
    with torch.autocast(device_type=points.device.type, enabled=False):
        points = points.to(dtype=dtype)
        surfaces = surfaces.to(dtype=dtype)
        if points.shape[0] == 0 or points.shape[1] == 0:
            closest = points + surfaces.sum() * 0.0
            return closest, torch.linalg.vector_norm(points - closest, dim=-1)

        triangle_indices = _surface_triangle_indices(surfaces.shape[1], surfaces.shape[2], points.device)
        surface_vertices = surfaces.flatten(1, 2)
        closest_chunks = []
        for start in range(0, points.shape[0], chunk_size):
            point_chunk = points[start:start + chunk_size]
            vertex_chunk = surface_vertices[start:start + chunk_size]
            with torch.no_grad():
                triangles = vertex_chunk[:, triangle_indices]
                projections = _project_to_triangles(point_chunk[:, :, None], triangles[:, None])
                squared_distances = (point_chunk[:, :, None] - projections).square().sum(dim=-1)
                nearest_triangle = squared_distances.argmin(dim=-1)
                selected_indices = triangle_indices[nearest_triangle]

            selected_vertices = vertex_chunk.gather(
                1, selected_indices.flatten(1, 2)[..., None].expand(-1, -1, 3)
            ).reshape(*point_chunk.shape[:2], 3, 3)
            closest_chunks.append(_project_to_triangles(point_chunk, selected_vertices))

        closest = torch.cat(closest_chunks, dim=0)
        distances = torch.linalg.vector_norm(points - closest, dim=-1)
        return closest, distances


def closest_polyline_points(points: Tensor, polylines: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Project [B,L,3] points onto [B,K,3] polylines, K >= 2.

    Returns closest points, segment start indices [B,L] (long), and fractions
    [B,L] in [0,1]. This correspondence can be computed on GT under no_grad and
    applied to predicted vertices by linear interpolation. Polylines are not
    automatically closed: a closed polyline must repeat its first endpoint.
    Precision and gradients follow ``closest_surface_points``.
    """
    if polylines.ndim != 3 or polylines.shape[-1] != 3 or polylines.shape[1] < 2:
        raise ValueError("polylines must have shape [B, K, 3] with K >= 2.")
    dtype = _validate_geometry_pair(points, polylines)
    with torch.autocast(device_type=points.device.type, enabled=False):
        points = points.to(dtype=dtype)
        polylines = polylines.to(dtype=dtype)
        if points.shape[0] == 0 or points.shape[1] == 0:
            closest = points + polylines.sum() * 0.0
            return closest, torch.empty(points.shape[:2], dtype=torch.long, device=points.device), points[..., 0]

        closest_chunks, index_chunks, fraction_chunks = [], [], []
        for start in range(0, points.shape[0], 32):
            point_chunk = points[start:start + 32]
            line_chunk = polylines[start:start + 32]
            with torch.no_grad():
                projections, _ = _project_to_segments(
                    point_chunk[:, :, None], line_chunk[:, None, :-1], line_chunk[:, None, 1:]
                )
                squared_distances = (point_chunk[:, :, None] - projections).square().sum(dim=-1)
                nearest_segment = squared_distances.argmin(dim=-1)

            selected_indices = nearest_segment[..., None].expand(-1, -1, 3)
            starts = line_chunk.gather(1, selected_indices)
            ends = line_chunk.gather(1, selected_indices + 1)
            closest, fractions = _project_to_segments(point_chunk, starts, ends)
            closest_chunks.append(closest)
            index_chunks.append(nearest_segment)
            fraction_chunks.append(fractions)

        return torch.cat(closest_chunks), torch.cat(index_chunks), torch.cat(fraction_chunks)
