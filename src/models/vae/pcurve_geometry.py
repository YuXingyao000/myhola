"""Differentiable fixed-basis curves and the ordered-grid surface bridge."""

import torch
from torch import nn


def clamped_knots(count, degree=3):
    interior = torch.linspace(0, 1, count - degree + 1, dtype=torch.float64)[1:-1]
    return torch.cat(
        [torch.zeros(degree + 1, dtype=torch.float64), interior, torch.ones(degree + 1, dtype=torch.float64)]
    )


def basis(t, knots, degree=3):
    x = t[..., None]
    b = ((x >= knots[:-1]) & (x < knots[1:])).to(t.dtype)
    first, last = degree, knots.numel() - degree - 2
    b[..., first] = torch.where(t < knots[degree], torch.ones_like(t), b[..., first])
    b[..., last] = torch.where(t >= knots[-degree - 1], torch.ones_like(t), b[..., last])
    for p in range(1, degree + 1):
        size = knots.numel() - p - 1
        dl = knots[p : p + size] - knots[:size]
        dr = knots[p + 1 : p + 1 + size] - knots[1 : 1 + size]
        left = (x - knots[:size]) / torch.where(dl != 0, dl, 1)
        right = (knots[p + 1 : p + 1 + size] - x) / torch.where(dr != 0, dr, 1)
        b = left * (dl != 0) * b[..., :size] + right * (dr != 0) * b[..., 1 : size + 1]
    return b


class SplineGeometry(nn.Module):
    def __init__(self, controls=16, samples=16):
        super().__init__()
        surface_knots = clamped_knots(16)
        matrix = basis(torch.linspace(0, 1, 16, dtype=torch.float64), surface_knots)
        self.register_buffer("surface_knots", surface_knots.float())
        self.register_buffer("grid_inverse", torch.linalg.pinv(matrix).float())
        self.register_buffer(
            "curve_basis", basis(torch.linspace(0, 1, samples, dtype=torch.float64), clamped_knots(controls)).float()
        )

    def surface_controls(self, grid):
        return torch.einsum("av,fvuc,bu->fabc", self.grid_inverse, grid.float(), self.grid_inverse)

    def forward(self, grid, owner, curve_controls):
        # Stable FP32 polynomial evaluation; never replace out-of-chart UV by zero.
        with torch.autocast(grid.device.type, enabled=False):
            uv = torch.einsum("mk,hkc->hmc", self.curve_basis, curve_controls.float())
            poles = self.surface_controls(grid)[owner]
            bu = basis(uv[..., 0], self.surface_knots)
            bv = basis(uv[..., 1], self.surface_knots)
            xyz = torch.einsum("hma,hmb,habc->hmc", bv, bu, poles)
        return uv, xyz
