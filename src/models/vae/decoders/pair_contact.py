"""Contact geometry between the two decoded faces of a candidate pair."""

from __future__ import annotations

import torch
from torch import nn, Tensor


class PairContactReader(nn.Module):
    """Summarise how two decoded face grids touch, as an additive term for the edge classifier.

    For each face of the pair, every grid point gets its distance to the other face's grid and
    the normal agreement at the closest point. A shared edge shows as a line of near-zero
    distances, a vertex contact as a spot, a close parallel neighbour as a uniform offset. The
    output projection starts at zero, so a model with this reader equals the model without it
    until training moves the projection.
    """

    def __init__(self, out_dim: int, width: int = 32, chunk_size: int = 512) -> None:
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.out = nn.Linear(width, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def contact_maps(faces_a: Tensor, faces_b: Tensor) -> Tensor:
        """[P, 4, H, W]: for each face, distance to the other face and normal dot at the closest point."""
        count, height, width = faces_a.shape[:3]
        pa, na = faces_a[..., :3].reshape(count, -1, 3), faces_a[..., 3:6].reshape(count, -1, 3)
        pb, nb = faces_b[..., :3].reshape(count, -1, 3), faces_b[..., 3:6].reshape(count, -1, 3)
        distance = torch.cdist(pa, pb)
        normal_dot = torch.einsum("pic,pjc->pij", na, nb)
        da, ja = distance.min(dim=2)
        db, jb = distance.min(dim=1)
        dot_a = normal_dot.gather(2, ja[..., None])[..., 0]
        dot_b = normal_dot.gather(1, jb[:, None, :])[:, 0]
        return torch.stack((da, dot_a, db, dot_b), dim=1).reshape(count, 4, height, width)

    def forward(self, faces: Tensor, pairs: Tensor) -> Tensor:
        """faces [F, H, W, 6] object-space points and unit normals; pairs [P, 2] face indices."""
        if len(pairs) == 0:
            return faces.new_zeros((0, self.out.out_features), dtype=torch.float32)
        outputs = []
        with torch.autocast(device_type=faces.device.type, enabled=False):
            faces = faces.float()
            for start in range(0, len(pairs), self.chunk_size):
                index = pairs[start:start + self.chunk_size]
                maps = self.contact_maps(faces[index[:, 0]], faces[index[:, 1]])
                outputs.append(self.out(self.cnn(maps)))
        return torch.cat(outputs, dim=0)
