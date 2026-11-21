"""Ordered sample coordinates as octree paths and normalized leaf offsets."""
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def encode_local_xyz(local_xyz: Tensor, depth: int):
    """Continuous GT stays unchanged; encode after augmentation/normalization."""
    with torch.autocast(local_xyz.device.type, enabled=False):
        xyz = local_xyz.float()
        resolution = 1 << depth
        delta = 2.0 / resolution
        cell = torch.floor((xyz + 1.0) / delta).long().clamp(0, resolution - 1)
        shifts = torch.arange(depth - 1, -1, -1, device=xyz.device)
        bits = (cell[..., None, :] >> shifts[:, None]) & 1
        path = 4 * bits[..., 0] + 2 * bits[..., 1] + bits[..., 2]
        center = -1.0 + (cell.float() + 0.5) * delta
        return path, (xyz - center) / delta


def descend(center, half_width, child):
    bits = ((child[..., None] >> child.new_tensor([2, 1, 0])) & 1).float()
    half_width = half_width * 0.5
    return center + (2.0 * bits - 1.0) * half_width, half_width


def decode_local_xyz(path, offset):
    with torch.autocast(offset.device.type, enabled=False):
        center = torch.zeros_like(offset, dtype=torch.float32)
        half_width = torch.ones_like(center[..., :1])
        for child in path.unbind(-1):
            center, half_width = descend(center, half_width, child)
        return center + 2.0 * half_width * offset.float()


class PointMLPBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, 2 * width)
        self.down = nn.Linear(2 * width, width)

    def forward(self, state):
        return state + self.down(F.silu(self.up(self.norm(state))))


class OctreeCoordinateHead(nn.Module):
    """One shared scorer, conditioned on the current cell and spatial depth."""
    def __init__(self, feature_dim=96, width=768, blocks=4, depth=12,
                 checkpoint_levels=True):
        super().__init__()
        self.depth = depth
        self.checkpoint_levels = checkpoint_levels
        self.feature_projection = nn.Linear(feature_dim, width)
        self.cell_projection = nn.Linear(4, width)
        self.level_embedding = nn.Embedding(depth + 1, width)
        self.blocks = nn.Sequential(*(PointMLPBlock(width) for _ in range(blocks)))
        self.norm = nn.LayerNorm(width)
        self.child_classifier = nn.Linear(width, 8)
        self.offset_regressor = nn.Linear(width, 3)

    def _state(self, features, center, half_width, level):
        cell = torch.cat((center, half_width), dim=-1)
        state = features + self.cell_projection(cell) + self.level_embedding.weight[level]
        return self.norm(self.blocks(state))

    def _logits(self, features, center, half_width, level):
        return self.child_classifier(self._state(features, center, half_width, level))

    def teacher_forced(self, point_features, gt_path):
        """Predict each child before consuming that label; GT leaf only trains r."""
        with torch.autocast(point_features.device.type, enabled=False):
            features = self.feature_projection(point_features.float())
            center = features.new_zeros(*features.shape[:-1], 3)
            half_width = torch.ones_like(center[..., :1])
            logits = []
            for level in range(self.depth):
                if self.checkpoint_levels and self.training and torch.is_grad_enabled():
                    scores = checkpoint(self._logits, features, center, half_width, level,
                                        use_reentrant=False)
                else:
                    scores = self._logits(features, center, half_width, level)
                logits.append(scores)
                center, half_width = descend(center, half_width, gt_path[..., level])
            leaf = self._state(features, center, half_width, self.depth)
            offset = 0.5 * torch.tanh(self.offset_regressor(leaf))
        return {'logits': torch.stack(logits, dim=-2), 'offset': offset}

    def generate(self, point_features):
        """No GT. Hard child choices have zero gradient; the leaf offset has gradients."""
        with torch.autocast(point_features.device.type, enabled=False):
            features = self.feature_projection(point_features.float())
            center = features.new_zeros(*features.shape[:-1], 3)
            half_width = torch.ones_like(center[..., :1])
            path = []
            # Dropping these graphs is the exact derivative of hard argmax,
            # not a straight-through estimator. This matters for all-pair coarse reads.
            with torch.no_grad():
                for level in range(self.depth):
                    child = self._logits(features, center, half_width, level).argmax(-1)
                    path.append(child)
                    center, half_width = descend(center, half_width, child)
            leaf = self._state(features, center, half_width, self.depth)
            offset = 0.5 * torch.tanh(self.offset_regressor(leaf))
        return {'path': torch.stack(path, dim=-1), 'offset': offset,
                'local_xyz': center + 2.0 * half_width * offset}
