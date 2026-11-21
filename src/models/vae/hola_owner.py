"""Original HoLa with same-owner geometric messages before the curve head."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.datasets.brep import denormalize_coord1112
from .decoders.refinement import GeometryRead
from .legacy import AutoEncoder_light


class OwnerBoundaryAttention(nn.Module):
    """Selected directed (i,j) boundaries with owner i share one sequence."""

    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, messages, logits, owners, slots, face_count, max_length):
        width = messages.shape[-1]
        padded = messages.new_zeros(face_count, max_length, width)
        padded[owners, slots] = messages
        source_bias = logits.new_full((face_count, max_length), -torch.inf, dtype=torch.float32)
        source_bias[owners, slots] = F.logsigmoid(logits.detach().float())
        q, k, v = self.qkv(self.norm(padded)).reshape(
            face_count, max_length, 3, self.heads, width // self.heads,
        ).permute(2, 0, 3, 1, 4).unbind(0)
        with torch.autocast(messages.device.type, enabled=False):
            scores = q.float() @ k.float().transpose(-1, -2) / q.shape[-1] ** 0.5
            scores = scores + source_bias[:, None, None, :]
            # The final key has zero logit and zero value, including on padded rows.
            weights = F.pad(scores, (0, 1), value=0).softmax(-1)[..., :-1]
        context = (weights.to(v.dtype) @ v).transpose(1, 2).reshape_as(padded)
        return self.output(context[owners, slots])


class OwnerBoundaryExchange(nn.Module):
    def __init__(self, edge_dim, width, geometry_dim, heads, geometry_chunk_size):
        super().__init__()
        self.geometry = GeometryRead(geometry_dim, width, geometry_chunk_size)
        self.message = nn.Sequential(nn.Linear(edge_dim + geometry_dim, width), nn.LayerNorm(width), nn.SiLU())
        self.attention = OwnerBoundaryAttention(width, heads)
        self.merge = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU())
        self.readout = nn.Linear(edge_dim + width, edge_dim)
        with torch.no_grad():
            self.readout.weight.zero_()
            self.readout.weight[:, :edge_dim].copy_(torch.eye(edge_dim))
            self.readout.bias.zero_()

    def forward(self, h, geometry, logits, pairs, slots, face_count, max_length):
        own = self.message(torch.cat((h, geometry), dim=-1))
        adjacent = self.attention(own, logits, pairs[:, 0], slots, face_count, max_length)
        message = self.merge(torch.cat((own, adjacent), dim=-1))
        return self.readout(torch.cat((h, message), dim=-1))


class AutoEncoder_hola_owner(AutoEncoder_light):
    """The legacy curve/Teacher parameters, sampling protocol and loss are shared."""

    def __init__(self, cfg):
        if cfg.get('init_hola_checkpoint') and cfg.get('checkpoint'):
            raise ValueError('Choose original HoLa initialization or a full owner-exchange checkpoint')
        # Strictly load ALL legacy parameters before registering the new branch.
        super().__init__({**cfg, 'checkpoint': cfg.get('init_hola_checkpoint')})
        self.owner_exchange = OwnerBoundaryExchange(self.edge_feature_dim, **cfg['owner_exchange'])
        self.pair_chunk_size = int(cfg['pair_chunk_size'])
        self._load_checkpoint(cfg.get('checkpoint'))
        self._apply_runtime_mode()

    def sample(self, v_fused_face_features, v_is_test=False):
        return super().sample(v_fused_face_features, v_is_test or not self.training)

    def forward(self, v_data, v_test=False):
        if v_test:
            # The legacy evaluation entry exports NumPy arrays of a single CAD.
            with torch.autocast(v_data['face_points'].device.type, enabled=False):
                return super().forward(v_data, v_test=True)
        return super().forward(v_data)

    def _candidate_features(self, faces, pairs):
        pair_tokens = self._add_intersection_training_noise(faces[pairs])
        h = self.inter(pair_tokens)                                 # [P,64], ordered
        logits = self.classifier(h).squeeze(-1)                     # one classifier
        return h, logits

    def _selected_geometry(self, h, surfaces, pairs):
        local = self.edge_points_decoder(h)
        bbox = self.edge_center_scale_decoder(h)
        coarse_xyz = local[..., :3].float() * bbox[:, None, 3:].float() + bbox[:, None, :3].float()
        return self.owner_exchange.geometry(coarse_xyz, surfaces, pairs)

    def _intersect_faces(self, faces, encoding, batch, result, kept_faces):
        if kept_faces is not None:
            for key in ('face_points_local', 'face_center_scale'):
                result[key] = result[key][kept_faces]
        local, bbox = result['face_points_local'], result['face_center_scale']
        surfaces = local[..., :3].float() * bbox[:, None, None, 3:].float() + bbox[:, None, None, :3].float()
        counts = batch['face_counts'].tolist() if batch is not None else [len(faces)]
        ids = torch.arange(len(faces), device=faces.device)
        # Reproduce the original inference meshgrid, including the diagonal.
        pairs = torch.cat([torch.cartesian_prod(group, group) for group in ids.split(counts)])
        chunks = []
        for group in pairs.split(self.pair_chunk_size):
            if self.training and torch.is_grad_enabled():
                values = checkpoint(self._candidate_features, faces, group, use_reentrant=False)
            else:
                values = self._candidate_features(faces, group)
            chunks.append(values)
        h, logits = (torch.cat(parts) for parts in zip(*chunks))

        if batch is None:
            probabilities = logits.float().sigmoid()
            selected = probabilities > 0.5
            edge_ids = torch.arange(int(selected.sum()), device=faces.device)
            result['pred_face_adj'] = selected
            result['pred_face_adj_prob'] = probabilities
            result['pred_edge_face_connectivity'] = torch.cat((edge_ids[:, None], pairs[selected]), dim=-1)
        else:
            # As in HoLa, GT rows supervise geometry even if the classifier misses them.
            lookup = pairs.new_full((len(faces), len(faces)), -1)
            lookup[pairs[:, 0], pairs[:, 1]] = torch.arange(len(pairs), device=faces.device)
            gt_edge, owner, neighbor = batch['edge_face_connectivity'].unbind(1)
            selected = lookup[owner, neighbor]
            neg_owner, neg_neighbor = batch['zero_positions'].unbind(1)
            negative = lookup[neg_owner, neg_neighbor]
            predictions = torch.cat((logits[selected], logits[negative]))
            targets = torch.cat((torch.ones_like(logits[selected]), torch.zeros_like(logits[negative])))
            result['loss_edge'] = F.binary_cross_entropy_with_logits(predictions, targets)
            result['loss_edge_feature'] = self.loss_fn(h[selected], encoding['edge_features'][gt_edge].detach())

        # Rejected rows never reach the coarse curve head or the exchange.
        h, logits, pairs = h[selected], logits[selected], pairs[selected]
        if len(h) == 0:
            return h
        geometry_chunks = []
        for features, group in zip(h.split(self.pair_chunk_size), pairs.split(self.pair_chunk_size)):
            if self.training and torch.is_grad_enabled():
                geometry = checkpoint(self._selected_geometry, features, surfaces, group, use_reentrant=False)
            else:
                geometry = self._selected_geometry(features, surfaces, group)
            geometry_chunks.append(geometry)
        geometry = torch.cat(geometry_chunks)
        offsets = [part[0].expand(len(part)) for part in ids.split(counts)]
        slots = pairs[:, 1] - torch.cat(offsets)[pairs[:, 0]]
        return self.owner_exchange(h, geometry, logits, pairs, slots, len(faces), max(counts))

    def decode_latents(self, face_latents):
        pred = self.decode({'face_latents': face_latents}, v_deduplicated=True)
        return {
            'face_latents': face_latents.float().cpu().numpy(),
            'pred_face_adj': pred['pred_face_adj'],
            'pred_face_adj_prob': pred['pred_face_adj_prob'].float().cpu().numpy(),
            'pred_edge_face_connectivity': pred['pred_edge_face_connectivity'].cpu().numpy(),
            'pred_face': denormalize_coord1112(pred['face_points_local'], pred['face_center_scale'])[..., :3].float().cpu().numpy(),
            'pred_edge': denormalize_coord1112(pred['edge_points_local'], pred['edge_center_scale'])[..., :3].float().cpu().numpy(),
        }
