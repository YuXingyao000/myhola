"""One residual exchange between the retained, directed curve records."""
from collections.abc import Mapping

import torch
from torch import Tensor, nn


class CoedgeFeatureFusion(nn.Module):
    """Read self/previous/next/mate in fixed roles before the existing face GAT.

    ``neighbors`` indexes raw curve records, not face IDs or native STEP edges.
    The offline topology quotient removes seams and contracts merged chains.
    Orientation is +1/-1 for NPZ sample order relative to the owning loop.
    The original curve features remain available to the geometry teacher.
    """

    def __init__(self, edge_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(4 * (edge_dim + 1), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_dim),
        )
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self, edges: Tensor, batch: Mapping[str, Tensor]) -> Tensor:
        # [R,D] + direction -> [R,D+1]; fixed roles -> [R,4*(D+1)].
        oriented = torch.cat((edges, batch["coedge_orientation"].to(edges)), dim=-1)
        neighbors = oriented[batch["coedge_neighbors"]]  # [R,3,D+1]
        context = torch.cat((oriented, neighbors.flatten(1)), dim=-1)
        return edges + self.update(context)
