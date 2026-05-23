from __future__ import annotations

import torch
from torch import nn

from src.brepnet.models.diffusion_padding import FacePadder, LatentSequence


class LatentProvider(nn.Module):
    def __init__(self, latent_cfg: dict, face_padder: FacePadder, autoencoder: nn.Module):
        super().__init__()
        self.latent_dim = latent_cfg["dim"]
        self.use_cached_latents = latent_cfg["use_cached_latents"]
        self.use_mean = latent_cfg["use_mean"]
        self.face_padder = face_padder
        self.autoencoder = autoencoder

    def forward(self, batch: dict) -> LatentSequence:
        if self.use_cached_latents:
            cached_latent_stats = batch["cached_latent_stats"]
            mean = cached_latent_stats[..., :self.latent_dim]
            std = cached_latent_stats[..., self.latent_dim:self.latent_dim * 2]
            if self.use_mean:
                latent_sequence = mean
            else:
                latent_sequence = mean + std * torch.randn_like(mean)
            return LatentSequence(values=latent_sequence, face_mask=batch["face_mask"])

        with torch.no_grad():
            encoding_result = self.autoencoder.encode(batch, True)
            face_latents, _, _ = self.autoencoder.sample(
                encoding_result["face_features"],
                v_is_test=self.use_mean,
            )
        return self.face_padder.pack(face_latents, batch["face_counts"])
