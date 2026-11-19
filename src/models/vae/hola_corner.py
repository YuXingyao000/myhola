"""HoLa owner-exchange followed by shared end-to-start messages on each face."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .hola_owner import AutoEncoder_hola_owner


def distribute_corner_messages(messages, scores, logits, valid):
    """z[...,h,k,:] goes both to h.end and k.start, with a zero-valued null."""
    with torch.autocast(messages.device.type, enabled=False):
        source = F.logsigmoid(logits.detach().float()).masked_fill(~valid, -torch.inf)
        next_scores = F.pad(scores.float() + source[..., None, :], (0, 1), value=0)
        previous_scores = F.pad(scores.float() + source[..., :, None], (0, 0, 0, 1), value=0)
        next_weights = next_scores.softmax(-1)
        previous_weights = previous_scores.softmax(-2)
        end = (next_weights[..., :-1, None] * messages.float()).sum(-2)
        start = (previous_weights[..., :-1, :, None] * messages.float()).sum(-3)
    return start, end, next_weights, previous_weights


class CornerMessageExchange(nn.Module):
    def __init__(self, edge_dim=64, width=128, face_chunk_size=16):
        super().__init__()
        self.face_chunk_size = face_chunk_size
        self.start = nn.Linear(edge_dim, width)
        self.end = nn.Linear(edge_dim, width)
        self.pair_message = nn.Sequential(nn.Linear(2 * width + 3, width), nn.SiLU(), nn.Linear(width, width))
        self.score = nn.Linear(width, 1)
        self.readout = nn.Linear(2 * width, edge_dim, bias=False)
        nn.init.zeros_(self.readout.weight)

    def _face_messages(self, boundary, endpoints, logits, valid):
        start, end = self.start(boundary), self.end(boundary)       # [F,L,128]
        length = boundary.shape[1]
        start = start[:, None].expand(-1, length, -1, -1)
        end = end[:, :, None].expand(-1, -1, length, -1)
        relative = endpoints[:, None, :, 0] - endpoints[:, :, None, 1]
        z = self.pair_message(torch.cat((end, start, relative), dim=-1))
        scores = self.score(z).squeeze(-1)                         # [F,L,L]
        previous, following, _, _ = distribute_corner_messages(z, scores, logits, valid)
        delta = self.readout(torch.cat((previous, following), dim=-1))
        # Padding is masked for supervision as well as message distribution.
        scores = scores.float().masked_fill(~(valid[:, :, None] & valid[:, None, :]), -torch.inf)
        return boundary + delta, scores

    def forward(self, boundary, coarse_xyz, logits, owners, slots, face_count, max_length):
        padded = boundary.new_zeros(face_count, max_length, boundary.shape[-1])
        points = coarse_xyz.new_zeros(face_count, max_length, 2, 3)
        probabilities = logits.new_zeros(face_count, max_length)
        valid = torch.zeros((face_count, max_length), dtype=torch.bool, device=boundary.device)
        padded[owners, slots] = boundary
        points[owners, slots] = coarse_xyz[:, [0, -1]]
        probabilities[owners, slots] = logits
        valid[owners, slots] = True
        parts = []
        for args in zip(*(x.split(self.face_chunk_size) for x in (padded, points, probabilities, valid))):
            if self.training and torch.is_grad_enabled():
                parts.append(checkpoint(self._face_messages, *args, use_reentrant=False))
            else:
                parts.append(self._face_messages(*args))
        final, scores = (torch.cat(x) for x in zip(*parts))
        return final[owners, slots], scores


class AutoEncoder_hola_corner(AutoEncoder_hola_owner):
    def __init__(self, cfg):
        if cfg.get('init_hola_checkpoint') and cfg.get('checkpoint'):
            raise ValueError('Choose original HoLa initialization or a full corner-exchange checkpoint')
        super().__init__({**cfg, 'checkpoint': None})
        self.corner_exchange = CornerMessageExchange(self.edge_feature_dim, **cfg['corner_exchange'])
        self.corner_loss_weights = cfg['corner_loss_weights']
        self._load_checkpoint(cfg.get('checkpoint'))
        self._apply_runtime_mode()

    def _exchange_candidates(self, h, geometry, logits, coarse_xyz, pairs, slots,
                             face_count, max_length, result):
        boundary = super()._exchange_candidates(
            h, geometry, logits, coarse_xyz, pairs, slots, face_count, max_length, result,
        )
        final, scores = self.corner_exchange(
            boundary, coarse_xyz, logits, pairs[:, 0], slots, face_count, max_length,
        )
        result['corner_scores'] = scores
        return final

    def loss(self, result, batch):
        loss = super().loss(result, batch)
        conn = batch['edge_face_connectivity']
        slots, scores = result['candidate_slots'], result['corner_scores']
        previous, following = batch['corner_previous'], batch['corner_next']
        local, bbox = result['edge_points_local'], result['edge_center_scale']
        with torch.autocast(local.device.type, enabled=False):
            xyz = local[..., :3].float() * bbox[:, None, 3:].float() + bbox[:, None, :3].float()
            if len(conn):
                row = F.pad(scores[conn[:, 1], slots, :], (0, 1), value=0)
                col = F.pad(scores.transpose(1, 2)[conn[:, 1], slots, :], (0, 1), value=0)
                connection = (F.cross_entropy(row, slots[following]) + F.cross_entropy(col, slots[previous])) / 2
                endpoint = F.l1_loss(xyz[:, [0, -1]], batch['edge_points'][conn[:, 0]][:, [0, -1], :3].float())
                join = F.l1_loss(xyz[:, -1], xyz[following, 0])
            else:
                connection = scores.masked_fill(~torch.isfinite(scores), 0).sum() * 0
                endpoint = join = xyz.sum() * 0
        for name, value in (('connection', connection), ('endpoint', endpoint), ('join', join)):
            loss['corner_' + name] = value * self.corner_loss_weights[name]
        return loss

    def forward(self, batch, v_test=False):
        loss, data = super().forward(batch, v_test=v_test)
        # Diagnostics are appended after legacy forward has formed total_loss.
        loss['legacy_loss'] = (loss['total_loss'] - sum(loss['corner_' + k] for k in self.corner_loss_weights)).detach()
        return loss, data
