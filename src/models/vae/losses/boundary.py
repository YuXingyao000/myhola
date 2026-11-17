"""Output supervision for the two face32 boundary experiments."""
import torch
from torch import nn
import torch.nn.functional as F

from .corner import corner_consistency
from .intersection import _map_directed_targets, balanced_binary_cross_entropy


def mean_l1(prediction, target):
    return F.l1_loss(prediction.float(), target.float()) if prediction.numel() else prediction.sum() * 0


def endpoint_losses(output, batch, targets):
    """Native next/previous supervise rows/columns of the SAME score matrix.

    GT is used only here. Decoder receives neither neighbor indices nor vertices.
    Closed one-edge loops may target their own candidate; padding is never a key.
    """
    endpoints = output.endpoints
    rows = endpoints.candidate_rows
    candidates = targets.candidate_rows
    edge_to_candidate = candidates.new_full((len(batch['edge_points']),), -1)
    edge_to_candidate[targets.edge_ids] = candidates
    face_ids = output.pairs.indices[candidates, 0]
    slot = candidates.new_empty(len(output.pairs.indices))
    valid = rows >= 0
    slot[rows[valid]] = torch.arange(rows.shape[1], device=rows.device).expand_as(rows)[valid]
    source_slots = slot[candidates]
    native = batch['endpoint_targets']
    next_slots = slot[edge_to_candidate[native['next'][targets.edge_ids]]]
    prev_slots = slot[edge_to_candidate[native['previous'][targets.edge_ids]]]
    heads = endpoints.scores.shape[1]

    def classify(scores, target_slots):
        logits = scores[face_ids, :, source_slots, :].float()  # [GT,H,L]
        logits = logits.masked_fill(~valid[face_ids, None, :], -torch.inf)
        logits = torch.cat((logits, logits.new_zeros((*logits.shape[:-1], 1))), -1)
        labels = target_slots[:, None].expand(-1, heads)
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten())

    connection = (classify(endpoints.scores, next_slots)
                  + classify(endpoints.scores.transpose(-1, -2), prev_slots)) * 0.5
    gt_vertices = batch['edge_points'][targets.edge_ids][:, [0, -1], :3]
    predicted_vertices = endpoints.vertices[candidates]
    with torch.no_grad():
        score = endpoints.scores.mean(1).masked_fill(~valid[:, None, :], -torch.inf)
        next_guess = score[face_ids, source_slots].argmax(-1)
        next_accuracy = (next_guess == next_slots).float().mean()
    return {'endpoint_connection': connection,
            'endpoint_vertex': mean_l1(predicted_vertices, gt_vertices),
            'endpoint_next_accuracy': next_accuracy}


class BoundaryLoss(nn.Module):
    """No feature matching, UV, incidence or twin objectives in this experiment."""

    def __init__(self, cfg, endpoints):
        super().__init__()
        self.weights = dict(cfg)
        self.endpoints = endpoints

    def forward(self, output, batch, teacher, kl):
        targets = _map_directed_targets(output, batch)
        rows, edge_ids = targets.candidate_rows, targets.edge_ids
        final, coarse = output.refined, output.coarse
        values = {
            'face_local': mean_l1(final.face_geometry.local_points, batch['face_norm']),
            'face_bbox': mean_l1(final.face_geometry.center_scale, batch['face_bbox']),
            'edge_local': mean_l1(final.edge_geometry.local_points[rows], batch['edge_norm'][edge_ids]),
            'edge_bbox': mean_l1(final.edge_geometry.center_scale[rows], batch['edge_bbox'][edge_ids]),
            'classification': balanced_binary_cross_entropy(final.edge_logits, targets.labels),
            'coarse_edge_local': mean_l1(coarse.edge_geometry.local_points[rows], batch['edge_norm'][edge_ids]),
            'coarse_edge_bbox': mean_l1(coarse.edge_geometry.center_scale[rows], batch['edge_bbox'][edge_ids]),
            'coarse_classification': balanced_binary_cross_entropy(coarse.edge_logits, targets.labels),
            'teacher_local': mean_l1(teacher.local_points, batch['edge_norm']),
            'teacher_bbox': mean_l1(teacher.center_scale, batch['edge_bbox']),
            'face_xyz': mean_l1(final.face_geometry.points, batch['face_points'][..., :3]),
            'edge_xyz': mean_l1(final.edge_geometry.points[rows], batch['edge_points'][edge_ids, :, :3]),
            'kl_loss': kl,
        }
        lookup = rows.new_full((len(batch['edge_points']),), -1)
        lookup[edge_ids] = rows
        values.update(corner_consistency(final.edge_geometry.points, lookup, batch['corner_targets'], len(batch['face_counts'])))
        if self.endpoints:
            if len(rows) and output.endpoints is not None:
                values.update(endpoint_losses(output, batch, targets))
            else:
                zero = final.edge_geometry.points.sum() * 0
                values.update(endpoint_connection=zero, endpoint_vertex=zero, endpoint_next_accuracy=zero.detach())
        weighted = {f'weighted_{key}': float(weight) * values[key]
                    for key, weight in self.weights.items() if key in values}
        values.update(weighted)
        values['total_loss'] = torch.stack(list(weighted.values())).sum()
        with torch.no_grad():
            active, positive = final.edge_logits > 0, targets.labels.bool()
            tp = (active & positive).sum().float()
            values['precision'] = tp / active.sum().clamp_min(1)
            values['recall'] = tp / positive.sum().clamp_min(1)
            # Comparable model selection excludes candidate-only auxiliary terms.
            values['geometry_selection'] = values['total_loss'].detach()
            for name in ('endpoint_connection', 'endpoint_vertex'):
                if f'weighted_{name}' in values:
                    values['geometry_selection'] = values['geometry_selection'] - values[f'weighted_{name}'].detach()
        return values
