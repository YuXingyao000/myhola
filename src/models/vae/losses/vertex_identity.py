"""Balanced identity reconstruction and separate, non-optimizing diagnostics."""
import torch
from torch.nn import functional as F
from torch_scatter import scatter_mean


def vertex_identity_loss(head, embeddings, vertex_ids, edge_counts, face_pairs, chunk_size):
    """Average valid positive/negative pairs per CAD, then average CADs.

    Compare unordered endpoint pairs within one object. Unknown identities and
    the diagonal are excluded. Row chunks change memory use, not the objective.
    """
    zero = embeddings.sum() * 0 + head.log_scale * 0 + head.bias * 0
    losses, diagnostics = [], []
    offset = 0
    for count in edge_counts:
        embedding = embeddings[offset:offset + count].flatten(0, 1)
        ids = vertex_ids[offset:offset + count].flatten()
        physical_pairs = face_pairs[offset:offset + count].sort(dim=-1).values
        _, physical_ids = torch.unique(physical_pairs, dim=0, return_inverse=True)
        valid = ids >= 0
        physical_ids = physical_ids.repeat_interleave(2)[valid]
        embedding, ids = embedding[valid], ids[valid]
        indices = torch.arange(len(ids), device=ids.device)
        sums = [zero, zero]
        counts = torch.zeros(2, device=ids.device)
        tp, predicted, cross_tp, cross_true = (zero.detach().clone() for _ in range(4))
        for start in range(0, len(ids), chunk_size):
            end = start + chunk_size
            scores = head.logits(embedding[start:end], embedding)
            unordered = indices[start:end, None] < indices[None, :]
            equal = ids[start:end, None] == ids[None, :]
            positive, negative = unordered & equal, unordered & ~equal
            sums[0] = sums[0] + (F.softplus(-scores) * positive).sum()
            sums[1] = sums[1] + (F.softplus(scores) * negative).sum()
            counts = counts + torch.stack((positive.sum(), negative.sum()))
            with torch.no_grad():
                active = scores > 0
                cross = positive & (physical_ids[start:end, None] != physical_ids[None, :])
                tp += (active & positive).sum()
                predicted += (active & unordered).sum()
                cross_tp += (active & cross).sum()
                cross_true += cross.sum()
        loss = (torch.stack(sums) / counts.clamp_min(1)).sum() / (counts > 0).sum().clamp_min(1)
        losses.append(loss)
        diagnostics.append(torch.stack((
            tp / predicted.clamp_min(1), tp / counts[0].clamp_min(1),
            cross_tp / cross_true.clamp_min(1), valid.float().sum() / max(1, valid.numel()),
        )))
        offset += count
    metrics = torch.stack(diagnostics).mean(0)
    return torch.stack(losses).mean(), dict(zip((
        'vertex_relation_precision', 'vertex_relation_recall',
        'vertex_relation_cross_edge_recall', 'vertex_label_coverage',
    ), metrics.detach().unbind()))


@torch.no_grad()
def vertex_geometry_metrics(local, bbox, target_xyz, vertex_ids, edge_counts):
    """Object-mean XYZ MAE and mean vertex-wise endpoint distance to centroid."""
    xyz = local[..., :3].float() * bbox[:, None, 3:].float() + bbox[:, None, :3].float()
    error = (xyz - target_xyz.float()).abs()
    metrics, offset = [], 0
    for count in edge_counts:
        endpoints = xyz[offset:offset + count, [0, -1]].flatten(0, 1)
        ids = vertex_ids[offset:offset + count].flatten()
        valid = ids >= 0
        _, group, sizes = torch.unique(ids[valid], return_inverse=True, return_counts=True)
        centers = scatter_mean(endpoints[valid], group, dim=0)
        distance = torch.linalg.vector_norm(endpoints[valid] - centers[group], dim=-1)
        spreads = scatter_mean(distance, group, dim=0)
        spread = (spreads * (sizes > 1)).sum() / (sizes > 1).sum().clamp_min(1)
        errors = error[offset:offset + count]
        endpoint = errors[:, [0, -1]].sum() / max(1, count * 2 * 3)
        interior = errors[:, 1:-1].sum() / max(1, count * (error.shape[1] - 2) * 3)
        metrics.append(torch.stack((endpoint, interior, spread)))
        offset += count
    return dict(zip(('endpoint_xyz_mae', 'interior_xyz_mae', 'vertex_spread'),
                    torch.stack(metrics).mean(0).unbind()))
