"""Small HoLa owner exchange with a shared local octree coordinate readout."""
import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange

from .decoders.octree import OctreeCoordinateHead, encode_local_xyz
from .hola_owner import AutoEncoder_hola_owner


class AutoEncoder_hola_octree(AutoEncoder_hola_owner):
    def __init__(self, cfg):
        if cfg.get('init_hola_checkpoint') and cfg.get('checkpoint'):
            raise ValueError('Choose released HoLa initialization or a full octree checkpoint')
        # Strictly reuse the released model before changing its coordinate readout.
        super().__init__({**cfg, 'checkpoint': None})
        old = self.edge_points_decoder[-2]
        direction = nn.Conv1d(old.in_channels, old.out_channels - 3, 1)
        with torch.no_grad():
            direction.weight.copy_(old.weight[3:])
            direction.bias.copy_(old.bias[3:])
        self.edge_points_decoder[-2] = direction
        self.edge_points_decoder[-1] = Rearrange('... c w -> ... w c')
        self.coordinate_head = OctreeCoordinateHead(feature_dim=old.in_channels, **cfg['octree'])
        self.coordinate_loss_weights = cfg['coordinate_loss_weights']
        self._load_checkpoint(cfg.get('checkpoint'))
        self._apply_runtime_mode()

    def _curve_features(self, h):
        grid = self.edge_points_decoder[:-2](h)              # [P,96,16]
        direction = self.edge_points_decoder[-2:](grid)     # [P,16,3]
        with torch.autocast(h.device.type, enabled=False):
            bbox = self.edge_center_scale_decoder(h.float())
        return grid.transpose(1, 2), direction, bbox

    def _candidate_geometry(self, faces, surfaces, pairs):
        h = self.inter(self._add_intersection_training_noise(faces[pairs]))
        logits = self.classifier(h).squeeze(-1)
        features, _, bbox = self._curve_features(h)
        # Coarse geometry always runs freely, even during training: no GT leaf leak.
        xyz = self.coordinate_head.generate(features)['local_xyz']
        coarse_xyz = xyz * bbox[:, None, 3:] + bbox[:, None, :3]
        geometry = self.owner_exchange.geometry(coarse_xyz, surfaces, pairs)
        return h, logits, geometry

    def _edge_readout(self, h, target, result, suffix):
        features, direction, bbox = self._curve_features(h)
        result['edge_direction' + suffix] = direction
        result['edge_center_scale' + suffix] = bbox
        if target is not None:
            path, offset = encode_local_xyz(target[..., :3], self.coordinate_head.depth)
            result['octree' + suffix] = self.coordinate_head.teacher_forced(features, path)
            result['octree_target' + suffix] = (path, offset)
        # Training only needs categorical/normalized-offset losses. Do not invent
        # an XYZ prediction by filling in the GT leaf. Validation/export is free-run.
        if not self.training or target is None:
            prediction = self.coordinate_head.generate(features)
            result['edge_points_local' + suffix] = torch.cat((prediction['local_xyz'], direction.float()), -1)
            result['edge_path' + suffix] = prediction['path']

    def _decode_edges(self, features, encoding, batch, result):
        final_target = None if batch is None else batch['edge_norm'][batch['edge_face_connectivity'][:, 0]]
        self._edge_readout(features, final_target, result, '')
        if 'edge_features' in encoding:
            teacher_target = None if batch is None else batch['edge_norm']
            self._edge_readout(encoding['edge_features'], teacher_target, result, '1')

    def loss(self, result, batch):
        losses = {
            'face_norm': self.loss_fn(result['face_points_local'], batch['face_norm']),
            'face_bbox': self.loss_fn(result['face_center_scale'], batch['face_bbox']),
            'edge_feature': result['loss_edge_feature'],
            'edge_classification': 0.1 * result['loss_edge'],
        }
        ids = batch['edge_face_connectivity'][:, 0]
        for suffix, target, bbox in [('', batch['edge_norm'][ids], batch['edge_bbox'][ids]),
                                      ('1', batch['edge_norm'], batch['edge_bbox'])]:
            out = result['octree' + suffix]
            path, offset = result['octree_target' + suffix]
            losses['edge_path_ce' + suffix] = self.coordinate_loss_weights['path'] * F.cross_entropy(
                out['logits'].reshape(-1, 8), path.reshape(-1))
            losses['edge_leaf_offset' + suffix] = self.coordinate_loss_weights['offset'] * F.l1_loss(out['offset'], offset)
            # Preserve the direction half of the original six-channel mean L1.
            losses['edge_direction' + suffix] = 0.5 * self.loss_fn(result['edge_direction' + suffix], target[..., 3:])
            losses['edge_bbox' + suffix] = self.loss_fn(result['edge_center_scale' + suffix], bbox)
        if self.gaussian_weights > 0:
            losses['kl_loss'] = result['kl_loss']
        return losses

    @torch.no_grad()
    def _loss_metrics(self, result, batch):
        metrics = {}
        ids = batch['edge_face_connectivity'][:, 0]
        for suffix, target, xyz in [('', batch['edge_norm'][ids], batch['edge_points'][ids, ..., :3]),
                                     ('1', batch['edge_norm'], batch['edge_points'][..., :3])]:
            out = result['octree' + suffix]
            path, offset = result['octree_target' + suffix]
            # Conditional accuracy is named separately from autonomous path accuracy.
            correct = out['logits'].argmax(-1) == path
            metrics['path_teacher_accuracy' + suffix] = correct.float().mean()
            metrics['leaf_offset_mae' + suffix] = F.l1_loss(out['offset'], offset)
            if 'edge_path' + suffix not in result:
                continue
            free_correct = result['edge_path' + suffix] == path
            prefix_correct = free_correct.long().cumprod(-1)
            for depth in (1, 4, 8, self.coordinate_head.depth):
                if depth <= self.coordinate_head.depth:
                    metrics[f'path_prefix_{depth}_accuracy' + suffix] = prefix_correct[..., depth - 1].float().mean()
            local = result['edge_points_local' + suffix][..., :3]
            bbox = result['edge_center_scale' + suffix]
            global_xyz = local * bbox[:, None, 3:] + bbox[:, None, :3]
            metrics['edge_local_xyz' + suffix] = F.l1_loss(local, target[..., :3])
            metrics['edge_xyz' + suffix] = F.l1_loss(global_xyz, xyz)
            metrics['endpoint_xyz' + suffix] = F.l1_loss(global_xyz[:, [0, -1]], xyz[:, [0, -1]])
            metrics['interior_xyz' + suffix] = F.l1_loss(global_xyz[:, 1:-1], xyz[:, 1:-1])
        return metrics
