"""Owner-local boundary sequences replace all-pairs Intersection candidates.

Only face latents enter free decoding. Teacher-forced events and GT mate are
supervision conditions in forward(), never inputs to decode_latents().
"""
import torch
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter_mean

from src.preprocess.loop_sequences import EDGE, relations_from_events
from .legacy import AutoEncoder_light
from .pcurve_geometry import SplineGeometry
from .pcurve_relations import PcurveDecoder
from .loop_blocks import OrderedLoopEncoder, OwnerBoundaryAR, MateDecoder, mate_objective, match_mates


class AutoEncoder_loop_ar(AutoEncoder_light):
    def __init__(self, cfg):
        if cfg.get('init_hola_checkpoint') and cfg.get('checkpoint'):
            raise ValueError('Choose published HoLa initialization or a complete loop-AR checkpoint')
        super().__init__({**cfg, 'checkpoint': cfg.get('init_hola_checkpoint')})
        settings = cfg['loop_ar']
        shape = dict(width=int(settings['width']), depth=int(settings['depth']))
        self.loop_encoder = OrderedLoopEncoder(**shape, loop_layers=int(settings['encoder_layers']))
        self.boundary_decoder = OwnerBoundaryAR(self.inter, self.hidden_channels // 8,
                                                int(settings['max_events']), int(settings['face_chunk_size']))
        self.mate_decoder = MateDecoder(**shape)
        self.pcurve_decoder = PcurveDecoder(**shape, controls=int(settings['controls']))
        self.splines = SplineGeometry(controls=int(settings['controls']))
        self.loss_weights = dict(settings['loss_weights'])
        self.drop_probability = float(settings['mate_drop_probability'])
        # The Transformer weights now belong to boundary_decoder; no old candidate path remains.
        del self.inter, self.classifier, self.edge_points_decoder, self.edge_center_scale_decoder
        self._load_checkpoint(cfg.get('checkpoint'))
        self._apply_runtime_mode()

    def encode(self, data, v_test=False):
        face = self.face_coords(data['face_points'].permute(0, 3, 1, 2).contiguous())
        edge = self.edge_coords(data['edge_points'].permute(0, 2, 1).contiguous())
        face = self.loop_encoder(face, edge, data)
        enriched = face + self.face_attn_proj_out(self.face_attn(self.face_attn_proj_in(face), data['attn_mask']))
        x = enriched
        adjacency = data['edge_face_connectivity'][:, 1:].T
        for layer in self.graph_face_edge:
            x = layer(x, adjacency, edge) + x if isinstance(layer, GATv2Conv) else layer(x)
        face = x + enriched
        group = torch.arange(len(data['face_counts']), device=face.device).repeat_interleave(data['face_counts'])
        global_feature = self.global_feature1(scatter_mean(face, group, dim=0)).repeat_interleave(data['face_counts'], dim=0)
        return {'face_features': face + self.global_feature2(torch.cat([face, global_feature], 1))}

    def _faces(self, z, mask=None):
        face = self.face_attn_proj_out2(self.face_attn2(self.face_attn_proj_in2(z), mask))
        with torch.autocast(face.device.type, enabled=False):
            phi = self.face_points_decoder[:-2](face.float())
            local = self.face_points_decoder[-2:](phi)
            bbox = self.face_center_scale_decoder(face.float())
        grid = local[..., :3] * bbox[:, None, None, 3:] + bbox[:, None, None, :3]
        memory = self.boundary_decoder.surface_memory(phi, grid)
        return face, local, bbox, grid, memory

    def forward(self, data, v_test=False):
        encoded = self.encode(data)
        z, kl, stats = self.sample(encoded['face_features'], v_is_test=not self.training)
        face, local, bbox, grid, memory = self._faces(z, data['attn_mask'])
        targets = data['event_targets']
        states, event_logits = self.boundary_decoder.teacher_forcing(targets, face, memory)
        # The loader serializes coedges in this same face/event order.
        h = states[targets == EDGE]
        objects = torch.arange(len(data['edge_counts']), device=h.device).repeat_interleave(data['edge_counts'])
        scores, null = self.mate_decoder(h, face, data['owner'], objects)
        poles = self.pcurve_decoder(h, face, data['owner'], data['mate'], data['next'])
        uv, xyz = self.splines(grid, data['owner'], poles)
        reference = data['edge_points'][..., :3]
        losses = dict(
            face_local=F.l1_loss(local, data['face_norm']),
            face_bbox=F.l1_loss(bbox, data['face_bbox']),
            boundary_event=F.cross_entropy(event_logits.flatten(0, 1), targets.flatten()),
            mate_relation=mate_objective(scores, null, data['mate'], self.drop_probability, self.training),
            pcurve_uv=F.l1_loss(uv, data['pcurve_uv']),
            edge_xyz=F.l1_loss(xyz, reference),
            mate_compatibility=(xyz - xyz[data['mate']].flip(1)).abs().mean(),
            kl_loss=kl,
        )
        if self.loss_weights.get('pcurve_gt_surface', 0) > 0:
            gt_trace = self.splines.evaluate(data['face_points'][..., :3].detach(), data['owner'], uv)
            losses['pcurve_gt_surface'] = F.l1_loss(gt_trace, reference)
        weighted = {f'weighted_{k}': value * self.loss_weights.get(k, 1) for k, value in losses.items()}
        with torch.no_grad():
            valid = targets >= 0
            event_correct = event_logits.argmax(-1) == targets
            decisions = valid & (torch.isfinite(event_logits).sum(-1) > 1)
            delta = xyz - reference
            start = xyz[:, 0]
            centers = scatter_mean(start, data['vertex'], dim=0)
            metrics = dict(
                event_accuracy_teacher=event_correct[valid].float().mean(),
                event_accuracy_decisions_teacher=event_correct[decisions].float().mean(),
                event_face_exact_teacher=(event_correct | ~valid).all(1).float().mean(),
                mate_accuracy_gt_halfedges=(torch.cat([scores, null[:, None]], -1).argmax(-1) == data['mate']).float().mean(),
                endpoint_mae=delta[:, [0, -1]].abs().mean(),
                middle_mae=delta[:, 1:-1].abs().mean(),
                endpoint_l2=torch.linalg.vector_norm(delta[:, [0, -1]], dim=-1).mean(),
                vertex_rms=(start - centers[data['vertex']]).square().sum(-1).mean().sqrt(),
                mate_endpoint_gap=torch.linalg.vector_norm(xyz[:, [0, -1]] - xyz[data['mate']][:, [-1, 0]], dim=-1).mean(),
                same_face_uv_gap=torch.linalg.vector_norm(uv[:, -1] - uv[data['next'], 0], dim=-1).max(),
            )
        output = {**{k: v.detach() for k, v in losses.items()}, **weighted, **metrics,
                  'total_loss': sum(weighted.values())}
        recon = {}
        if v_test:
            items = [self.decode_latents(latent) for latent in z.split(data['face_counts'].tolist())]
            for item, latent_stats in zip(items, stats.split(data['face_counts'].tolist())):
                item['cached_latent_stats'] = latent_stats.detach()
            recon = {'items': items}
        return output, recon

    @torch.no_grad()
    def decode_latents(self, z):
        face, _, _, grid, memory = self._faces(z)
        events, states, confidence = self.boundary_decoder.generate(face, memory)
        owners, next_ids, ended = relations_from_events(events.cpu().numpy())
        owner = torch.as_tensor(owners, device=z.device)
        nxt = torch.as_tensor(next_ids, device=z.device)
        h = states[events == EDGE]
        result = dict(pred_face=grid, pred_surface_controls=self.splines.surface_controls(grid),
                      pred_owner=owner, pred_next=nxt, pred_events=events,
                      pred_face_sequence_finished=torch.as_tensor(ended, device=z.device),
                      representation='native_pcurve_v1')
        if len(h) and bool(ended.all()):
            scores, null = self.mate_decoder(h, face, owner, torch.zeros_like(owner))
            mate = match_mates(scores, null)
            probability = torch.cat([scores, null[:, None]], -1).softmax(-1)
            mate_confidence = probability[torch.arange(len(h), device=z.device), mate.clamp_min(0)]
            mate_confidence = torch.where(mate >= 0, mate_confidence, 0)
        else:
            mate = torch.full_like(owner, -1)
            mate_confidence = grid.new_zeros(len(owner))
        result['pred_mate'] = mate
        result['pred_edge_confidence'] = confidence[events == EDGE] * mate_confidence
        adjacency = torch.zeros(len(face), len(face), device=z.device, dtype=torch.bool)
        paired = mate >= 0
        adjacency[owner[paired], owner[mate[paired]]] = True
        result['pred_face_adj'] = adjacency
        complete = bool(ended.all() and len(h) and (mate >= 0).all() and (nxt >= 0).all())
        result['topology_complete'] = complete
        if not complete:
            result['pred_edge'] = grid.new_empty((0, 16, 3))
            result['pred_edge_face_connectivity'] = owner.new_empty((0, 3))
            return result
        poles = self.pcurve_decoder(h, face, owner, mate, nxt)
        uv, xyz = self.splines(grid, owner, poles)
        result.update(pred_pcurve_controls=poles, pred_pcurve_uv=uv, pred_edge=xyz,
                      pred_edge_face_connectivity=torch.stack([torch.arange(len(h), device=z.device), owner, owner[mate]], 1))
        return result
