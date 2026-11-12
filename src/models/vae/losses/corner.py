"""Cross-curve corner coincidence on final XYZ; no new decoder parameters."""
import torch


def _group_mean(values, group_ids, count):
    totals = values.new_zeros(count).index_add(0, group_ids, values)
    sizes = values.new_zeros(count).index_add(0, group_ids, torch.ones_like(values))
    return totals / sizes.clamp_min(1)


def corner_consistency(curves, edge_to_candidate, labels, object_count):
    """Pair mean -> vertex mean -> object mean (empty objects contribute zero).

    curves[P,L,3]; pairs[N,2,2] hold packed GT edge IDs and endpoint slots.
    The mapping uses GT connectivity, never a predicted existence threshold.
    """
    pairs = labels["pairs"]
    if not pairs.numel():
        zero = curves.float().sum() * 0.0
        return {"corner": zero, "corner_gap": zero.detach()}
    rows = edge_to_candidate[pairs[..., 0]]
    samples = pairs[..., 1] * (curves.shape[1] - 1)
    endpoints = curves[rows, samples].float()  # [N,2,3], both members get gradients
    residual = endpoints[:, 0] - endpoints[:, 1]
    group_objects = labels["group_objects"]

    def reduce(values):
        groups = _group_mean(values, labels["pair_groups"], len(group_objects))
        return _group_mean(groups, group_objects, object_count).mean()

    return {
        "corner": reduce(residual.abs().mean(-1)),
        "corner_gap": reduce(torch.linalg.vector_norm(residual.detach(), dim=-1)),
    }
