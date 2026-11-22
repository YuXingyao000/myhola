"""Encode native vertex groups; reconstruct their identities from the latent.

The original large-owner geometry, existence classifier and Teacher remain intact.
Only the GAT edge attributes consume the new Encoder messages.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean

from src.datasets.brep import denormalize_coord1112
from .hola_owner import AutoEncoder_hola_owner
from .losses.vertex_identity import vertex_identity_loss, vertex_geometry_metrics


class EndpointProjection(nn.Module):
    """Two learned output slots, each reading the complete directed edge feature."""

    def __init__(self, edge_dim, width):
        super().__init__()
        self.width = width
        self.layers = nn.Sequential(nn.Linear(edge_dim, width), nn.GELU(), nn.Linear(width, 2 * width))

    def forward(self, edge_features):
        return self.layers(edge_features).reshape(len(edge_features), 2, self.width)


class VertexGroupEncoder(nn.Module):
    def __init__(self, edge_dim=64, width=512, hidden_dim=1024):
        super().__init__()
        self.width = width
        self.endpoint = EndpointProjection(edge_dim, width)
        self.message = nn.Sequential(
            nn.Linear(2 * width, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, width),
        )
        self.writeback = nn.Sequential(
            nn.Linear(2 * width, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, edge_dim),
        )
        nn.init.zeros_(self.writeback[-1].weight)
        nn.init.zeros_(self.writeback[-1].bias)

    def forward(self, edge_features, vertex_ids):
        q = self.endpoint(edge_features)                          # [H,2,D]
        valid = vertex_ids >= 0
        _, group = torch.unique(vertex_ids[valid], return_inverse=True)
        context = torch.zeros_like(q)
        # Reductions use FP32 even when the projection uses BF16.
        summary = scatter_mean(q[valid].float(), group, dim=0)
        context[valid] = summary[group].to(q.dtype)
        message = self.message(torch.cat((q, context), dim=-1))
        message = message * valid[..., None]
        delta = self.writeback(message.reshape(len(q), 2 * self.width))
        delta = delta * valid.any(dim=1, keepdim=True)
        return edge_features + delta


class VertexIdentityDecoder(nn.Module):
    def __init__(self, edge_dim=64, width=512, hidden_dim=1024):
        super().__init__()
        self.endpoint = EndpointProjection(edge_dim, width)
        self.embedding = nn.Sequential(
            nn.Linear(width, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, width),
        )
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.tensor(-5.0))

    def forward(self, halfedge_features):
        q = self.endpoint(halfedge_features)                      # [K,2,D]
        embedding = self.embedding(q)
        with torch.autocast(embedding.device.type, enabled=False):
            return F.normalize(embedding.float(), dim=-1, eps=1e-6)

    def logits(self, first, second):
        with torch.autocast(first.device.type, enabled=False):
            return self.log_scale.float().exp() * (first.float() @ second.float().T) + self.bias.float()

    def probabilities(self, embedding):
        endpoints = embedding.flatten(0, 1)
        return self.logits(endpoints, endpoints).sigmoid()


class AutoEncoder_vertex_identity(AutoEncoder_hola_owner):
    def __init__(self, cfg):
        if cfg.get('init_hola_checkpoint') and cfg.get('checkpoint'):
            raise ValueError('Choose original HoLa initialization or a complete vertex-identity checkpoint')
        super().__init__({**cfg, 'checkpoint': None})
        settings = cfg['vertex_identity']
        self.vertex_loss_weight = settings['loss_weight']
        self.vertex_pair_chunk_size = settings['pair_chunk_size']
        # Additional initialization must not change the retained owner's weights
        # or the subsequent random sequence of the baseline training path.
        with torch.random.fork_rng(devices=[]):
            dimensions = dict(edge_dim=self.edge_feature_dim, width=settings['width'],
                              hidden_dim=settings['hidden_dim'])
            self.vertex_encoder = VertexGroupEncoder(**dimensions)
            self.vertex_decoder = VertexIdentityDecoder(**dimensions)
        self._load_checkpoint(cfg.get('checkpoint'))
        self._apply_runtime_mode()

    def _encoder_edge_attributes(self, edge_features, batch):
        directed = edge_features[batch['edge_face_connectivity'][:, 0]]
        return self.vertex_encoder(directed, batch['endpoint_vertex_ids'])

    def _intersect_faces(self, faces, encoding, batch, result, kept_faces):
        # The retained owner processes ALL candidates before the original
        # supervised/predicted row selection. Vertex IDs never enter this path.
        final = super()._intersect_faces(faces, encoding, batch, result, kept_faces)
        result['endpoint_embeddings'] = self.vertex_decoder(final)
        if batch is None:
            result['pred_endpoint_same_vertex'] = self.vertex_decoder.probabilities(result['endpoint_embeddings'])
        return final

    def loss(self, result, batch):
        losses = super().loss(result, batch)
        identity, diagnostics = vertex_identity_loss(
            self.vertex_decoder, result['endpoint_embeddings'], batch['endpoint_vertex_ids'],
            batch['vertex_edge_counts'], batch['edge_face_connectivity'][:, 1:],
            self.vertex_pair_chunk_size,
        )
        losses['weighted_vertex_identity'] = self.vertex_loss_weight * identity
        diagnostics['vertex_identity'] = identity.detach()
        diagnostics.update(vertex_geometry_metrics(
            result['edge_points_local'], result['edge_center_scale'],
            batch['edge_points'][batch['edge_face_connectivity'][:, 0], :, :3],
            batch['endpoint_vertex_ids'], batch['vertex_edge_counts'],
        ))
        result['loss_diagnostics'] = diagnostics
        return losses

    def decode_latents(self, face_latents):
        pred = self.decode({'face_latents': face_latents}, v_deduplicated=True)
        return {
            'face_latents': face_latents.float().cpu().numpy(),
            'pred_face_adj': pred['pred_face_adj'],
            'pred_face_adj_prob': pred['pred_face_adj_prob'].float().cpu().numpy(),
            'pred_edge_face_connectivity': pred['pred_edge_face_connectivity'].cpu().numpy(),
            'pred_face': denormalize_coord1112(pred['face_points_local'], pred['face_center_scale'])[..., :3].float().cpu().numpy(),
            'pred_edge': denormalize_coord1112(pred['edge_points_local'], pred['edge_center_scale'])[..., :3].float().cpu().numpy(),
            'pred_endpoint_same_vertex': pred['pred_endpoint_same_vertex'].detach().cpu().numpy(),
        }
