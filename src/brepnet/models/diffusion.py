from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch import Tensor, nn
from tqdm import tqdm

from src.brepnet.models.diffusion_condition import build_condition_encoder
from src.brepnet.models.diffusion_denoiser import (
    BRepDenoiser,
    TopologyBias,
    build_condition_fuser,
)
from src.brepnet.models.diffusion_latents import LatentProvider
from src.brepnet.models.diffusion_padding import build_face_padder
from src.brepnet.models.vae import build_autoencoder


class Diffusion(nn.Module):
    def __init__(self, cfg: dict, condition_cfg: dict):
        super().__init__()
        self.latent_dim         =   cfg["latent"]["dim"]
        self.max_faces          =   cfg["padding"]["max_faces"]
        self.prediction_type    =   cfg["noise"]["prediction_type"]
        self.valid_loss_weight  =   cfg["padding"]["valid_loss_weight"]
        self.loss_fn            =   F.l1_loss if cfg["loss"] == "l1" else F.mse_loss

        noise_cfg = cfg["noise"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps =   noise_cfg["num_train_timesteps"],
            beta_schedule       =   noise_cfg["beta_schedule"],
            prediction_type     =   self.prediction_type,
            beta_start          =   noise_cfg["beta_start"],
            beta_end            =   noise_cfg["beta_end"],
            variance_type       =   noise_cfg["variance_type"],
            clip_sample         =   False,
        )

        self.autoencoder = build_autoencoder(cfg["autoencoder"])
        self.face_padder = build_face_padder(cfg["padding"], self.latent_dim)
        self.latent_provider = LatentProvider(
            latent_cfg  =   cfg["latent"],
            face_padder =   self.face_padder,
            autoencoder =   self.autoencoder,
        )

        self.condition_encoder = build_condition_encoder(condition_cfg)
        condition_fuser = build_condition_fuser(
            fuser_cfg       =   cfg["condition_fuser"],
            condition_cfg   =   condition_cfg,
            hidden_dim      =   cfg["denoiser"]["hidden_dim"],
        )
        self.denoiser = BRepDenoiser(
            latent_dim      =   self.latent_dim,
            cfg             =   cfg["denoiser"],
            condition_fuser =   condition_fuser,
        )
        self.topology_bias = TopologyBias(
            cfg                 =   cfg["topology_bias"],
            num_train_timesteps =   noise_cfg["num_train_timesteps"],
        )

    def extract_condition(self, batch: dict) -> Tensor | None:
        if self.condition_encoder is None:
            return None
        return self.condition_encoder(batch)

    def diffuse(
        self,
        noisy_latent_sequence: Tensor,
        timesteps: Tensor,
        condition: Tensor | None,
        attention_bias: Tensor | None = None,
    ) -> Tensor:
        return self.denoiser(noisy_latent_sequence, timesteps, condition, attention_bias)

    def forward(self, batch: dict, v_test: bool = False) -> dict[str, Tensor]:
        latent_batch = self.latent_provider(batch)
        latent_sequence = latent_batch.values
        batch_size = latent_sequence.size(0)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=latent_sequence.device,
        ).long()
        condition = self.extract_condition(batch)
        noise = torch.randn(latent_sequence.shape, device=latent_sequence.device)
        noisy_latent_sequence = self.noise_scheduler.add_noise(latent_sequence, noise, timesteps)
        face_adjacency = batch["face_adj"] if self.topology_bias.enabled else None
        attention_bias = self.topology_bias(face_adjacency, timesteps)

        prediction = self.diffuse(noisy_latent_sequence, timesteps, condition, attention_bias)
        target = latent_sequence if self.prediction_type == "sample" else noise
        loss_item = self.loss_fn(prediction, target, reduction="none")
        loss = {
            "diffusion_loss": loss_item.mean(),
            "t": torch.stack((timesteps, loss_item.mean(dim=1).mean(dim=1)), dim=1),
        }
        valid_loss = self.face_padder.validity_loss(
            prediction,
            latent_batch.face_mask,
            self.valid_loss_weight,
        )
        if valid_loss is not None:
            loss["valid_loss"] = valid_loss
        loss["total_loss"] = sum(value for key, value in loss.items() if key != "t")
        return loss

    def inference(
        self,
        num_samples: int,
        device: torch.device,
        v_data: dict | None = None,
        v_log: bool = True,
        **kwargs,
    ) -> list[dict]:
        latent_sequence = torch.randn((num_samples, self.max_faces, self.latent_dim), device=device)
        condition = self.extract_condition(v_data) if v_data is not None else None
        if condition is not None:
            condition = condition[:num_samples]

        for timestep in tqdm(self.noise_scheduler.timesteps, disable=not v_log):
            timesteps = timestep.reshape(-1).to(device)
            prediction = self.diffuse(latent_sequence, timesteps, condition)
            latent_sequence = self.noise_scheduler.step(prediction, timestep, latent_sequence).prev_sample

        face_mask = self.face_padder.inference_mask(latent_sequence)
        recon_data = []
        for item_idx in range(num_samples):
            face_latents = latent_sequence[item_idx:item_idx + 1][face_mask[item_idx:item_idx + 1]]
            face_latents = self.face_padder.postprocess(face_latents)
            recon_data.append(self.autoencoder.decode_latents(face_latents))
        return recon_data


__all__ = ["Diffusion"]
