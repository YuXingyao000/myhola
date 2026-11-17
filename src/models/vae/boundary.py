"""Minimal face32 VAE assembly; reuse the established encode/export interface."""
import torch
from torch import nn

from .legacy import AutoEncoder_light
from .intersection import AutoEncoder_intersection
from .encoders import HolisticEncoder, FacePosterior
from .decoders.boundary import BoundaryDecoder
from .losses.boundary import BoundaryLoss


class AutoEncoder_boundary(AutoEncoder_intersection):
    """Two explicit configs: owner exchange, or owner + endpoint exchange.

    Inherit only encode/sample/train and prediction export operations. Construct
    no IntersectionDecoder, legacy Refiner, UV head or feature-matching loss.
    """

    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.freeze_encoder = bool(cfg['freeze_encoder'])
        self.runtime_mode = cfg['mode']
        self.edge_threshold = float(cfg['edge_threshold'])
        legacy = AutoEncoder_light({**cfg, 'checkpoint': None, 'mode': 'train'})
        self.face_feature_dim = legacy.face_feature_dim
        self.encoder = HolisticEncoder(legacy)
        self.posterior = FacePosterior(legacy)
        self.decoder = BoundaryDecoder(legacy, cfg['boundary'])
        self.criterion = BoundaryLoss(cfg['loss_weights'], cfg['boundary']['endpoints'])
        if cfg.get('checkpoint'):
            saved = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
            self.load_state_dict({k.removeprefix('model.'): v for k, v in saved['state_dict'].items()}, strict=True)
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.posterior.requires_grad_(False)
        if self.runtime_mode == 'frozen_inference':
            self.requires_grad_(False)
        self.train(self.runtime_mode == 'train')

    def forward(self, batch, v_test=False):
        encoding = self.encode(batch, v_test)
        latents, kl, statistics = self.sample(encoding['face_features'], v_is_test=v_test)
        prediction = self.decode(latents, batch['face_counts'])
        teacher = self.decoder.decode_teacher(encoding['edge_features'])
        losses = self.criterion(prediction, batch, teacher, kl)
        items = self.export_training_items(prediction, latents, statistics, batch) if v_test else {}
        return losses, items
