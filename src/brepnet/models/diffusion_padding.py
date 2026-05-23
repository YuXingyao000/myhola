from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class LatentSequence:
    values: Tensor
    face_mask: Tensor | None


class FacePadder(nn.Module):
    def pack(self, face_latents: Tensor, face_counts: Tensor) -> LatentSequence:
        raise NotImplementedError

    def inference_mask(self, latent_sequence: Tensor) -> Tensor:
        return torch.ones_like(latent_sequence[..., 0], dtype=torch.bool)

    def validity_loss(self, prediction: Tensor, face_mask: Tensor | None, weight: float) -> Tensor | None:
        return None

    def postprocess(self, face_latents: Tensor) -> Tensor:
        return face_latents


class ZeroFacePadder(FacePadder):
    def __init__(self, latent_dim: int, max_faces: int):
        super().__init__()
        self.max_faces = max_faces
        self.validity_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 1),
        )

    def pack(self, face_latents: Tensor, face_counts: Tensor) -> LatentSequence:
        latent_sequence = torch.zeros(
            (face_counts.shape[0], self.max_faces, face_latents.shape[-1]),
            device=face_latents.device,
            dtype=face_latents.dtype,
        )
        face_mask = face_counts[:, None] > torch.arange(self.max_faces, device=face_counts.device)
        latent_sequence[face_mask] = face_latents
        return LatentSequence(values=latent_sequence, face_mask=face_mask)

    def inference_mask(self, latent_sequence: Tensor) -> Tensor:
        logits = self.validity_head(latent_sequence)[..., 0]
        return torch.sigmoid(logits) > 0.5

    def validity_loss(self, prediction: Tensor, face_mask: Tensor | None, weight: float) -> Tensor:
        logits = self.validity_head(prediction)[..., 0]
        return F.binary_cross_entropy_with_logits(logits, face_mask.float()) * weight


class RandomFacePadder(FacePadder):
    def __init__(self, max_faces: int):
        super().__init__()
        self.max_faces = max_faces

    def pack(self, face_latents: Tensor, face_counts: Tensor) -> LatentSequence:
        positions = torch.arange(self.max_faces, device=face_latents.device).unsqueeze(0).repeat(face_counts.shape[0], 1)
        mandatory_mask = positions < face_counts[:, None]
        random_indices = (torch.rand((face_counts.shape[0], self.max_faces), device=face_latents.device) * face_counts[:, None]).long()
        indices = torch.where(mandatory_mask, positions, random_indices)
        count_offsets = face_counts.cumsum(dim=0).roll(1)
        count_offsets[0] = 0
        indices += count_offsets[:, None]
        random_order = torch.argsort(torch.rand((face_counts.shape[0], self.max_faces), device=face_latents.device), dim=1)
        indices = indices.gather(1, random_order)
        face_mask = torch.ones((face_counts.shape[0], self.max_faces), device=face_latents.device, dtype=torch.bool)
        return LatentSequence(values=face_latents[indices], face_mask=face_mask)

    def postprocess(self, face_latents: Tensor) -> Tensor:
        threshold = 1e-2
        num_faces = face_latents.shape[0]
        if num_faces == 0:
            return face_latents
        index = torch.stack(
            torch.meshgrid(
                torch.arange(num_faces, device=face_latents.device),
                torch.arange(num_faces, device=face_latents.device),
                indexing="ij",
            ),
            dim=2,
        )
        pair_features = face_latents[index]
        distance = (pair_features[:, :, 0] - pair_features[:, :, 1]).abs().mean(dim=-1)
        keep = []
        for face_idx in range(num_faces):
            is_unique = True
            for kept_idx in keep:
                if distance[face_idx, kept_idx] < threshold:
                    is_unique = False
                    break
            if is_unique:
                keep.append(face_idx)
        return face_latents[keep]


def build_face_padder(padding_cfg: dict, latent_dim: int) -> FacePadder:
    if padding_cfg["type"] == "zero":
        return ZeroFacePadder(latent_dim=latent_dim, max_faces=padding_cfg["max_faces"])
    if padding_cfg["type"] == "random":
        return RandomFacePadder(max_faces=padding_cfg["max_faces"])
    raise ValueError(f"Unknown padding type '{padding_cfg['type']}'.")
