"""Diffusion 装配与训练入口；保留原 checkpoint 的 nn.Module 属性树。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch import Tensor, nn

from src.brepnet.models.vae import build_autoencoder

from .conditioning import build_condition_encoder
from .denoiser import BRepDenoiser
from .fusion import build_condition_fuser
from .latents import LatentProvider
from .losses import compute_diffusion_losses
from .padding import build_face_padder
from .sampling import sample_latent_sequence
from .topology import LearnedTopologySelfAttentionMask, build_topology_self_attention


class Diffusion(nn.Module):
    """图像条件 → 面 latent DDPM → 冻结 VAE 解码。"""

    def __init__(self, cfg: dict, condition_cfg: dict):
        super().__init__()
        self.latent_dim = cfg["latent"]["dim"]
        self.max_faces = cfg["padding"]["max_faces"]
        self.prediction_type = cfg["noise"]["prediction_type"]
        self.valid_loss_weight = cfg["padding"]["valid_loss_weight"]
        self.loss_fn = F.l1_loss if cfg["loss"] == "l1" else F.mse_loss

        # 1. 噪声过程：默认 epsilon prediction，沿用原 DDPM 配置。
        noise_cfg = cfg["noise"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_cfg["num_train_timesteps"],
            beta_schedule=noise_cfg["beta_schedule"],
            prediction_type=self.prediction_type,
            beta_start=noise_cfg["beta_start"],
            beta_end=noise_cfg["beta_end"],
            variance_type=noise_cfg["variance_type"],
            clip_sample=False,
        )

        # 2. 表示接口：缓存或冻结 VAE → 固定数量的面 slots。
        self.autoencoder = build_autoencoder(cfg["autoencoder"])
        self.face_padder = build_face_padder(cfg["padding"], self.latent_dim)
        self.latent_provider = LatentProvider(
            latent_cfg=cfg["latent"],
            face_padder=self.face_padder,
            autoencoder=self.autoencoder,
        )

        # 3. 条件与去噪网络：保持注册顺序，兼容已有优化器参数顺序。
        self.condition_encoder = build_condition_encoder(condition_cfg)
        condition_fuser = build_condition_fuser(
            fuser_cfg=cfg["condition_fuser"],
            condition_cfg=condition_cfg,
            hidden_dim=cfg["denoiser"]["hidden_dim"],
        )
        self.denoiser = BRepDenoiser(
            latent_dim=self.latent_dim,
            cfg=cfg["denoiser"],
            condition_fuser=condition_fuser,
        )
        self.oracle_topology_self_attention = build_topology_self_attention(
            cfg=cfg["topology_bias"],
            num_train_timesteps=noise_cfg["num_train_timesteps"],
        )

    def extract_condition(self, batch: dict) -> Tensor | None:
        """单图默认返回 [B, 1, 257, 1024]；无条件分支返回 None。"""
        if self.condition_encoder is None:
            return None
        return self.condition_encoder(batch)

    def diffuse(
        self,
        noisy_latent_sequence: Tensor,
        timesteps: Tensor,
        condition: Tensor | None,
        clean_latent_sequence: Tensor | None = None,
        self_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        return self.denoiser(
            noisy_latent_sequence,
            timesteps,
            condition,
            clean_latent_sequence,
            self_attention_mask,
        )

    def _topology_input(self, batch: dict) -> Tensor | None:
        """Pick the right input for the topology self-attention module.

        - OracleTopologySelfAttentionMask: needs GT face_adj from batch
        - LearnedTopologySelfAttentionMask: needs raw images from batch
        """
        if not self.oracle_topology_self_attention.enabled:
            return None
        if isinstance(self.oracle_topology_self_attention, LearnedTopologySelfAttentionMask):
            return batch["conditions"]["imgs"]
        return batch["face_adj"]

    def forward(self, batch: dict, v_test: bool = False) -> dict[str, Tensor]:
        """训练：clean latent → 加噪 → 预测 epsilon/x0 → 汇总损失。"""
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
        topology_input = self._topology_input(batch)
        self_attention_mask = self.oracle_topology_self_attention(topology_input, timesteps)

        prediction, align_loss = self.diffuse(
            noisy_latent_sequence,
            timesteps,
            condition,
            latent_sequence,
            self_attention_mask,
        )
        target = latent_sequence if self.prediction_type == "sample" else noise
        valid_loss = self.face_padder.validity_loss(
            prediction,
            latent_batch.face_mask,
            self.valid_loss_weight,
        )
        return compute_diffusion_losses(
            prediction, target, timesteps, self.loss_fn, align_loss, valid_loss
        )

    def inference(
        self,
        num_samples: int,
        device: torch.device,
        v_data: dict | None = None,
        v_log: bool = True,
        **kwargs,
    ) -> list[dict]:
        """推理：高斯噪声 [B, F, C] → DDPM → 面筛选/去重 → VAE 几何。"""
        latent_sequence = torch.randn((num_samples, self.max_faces, self.latent_dim), device=device)
        condition = self.extract_condition(v_data) if v_data is not None else None
        if condition is not None:
            condition = condition[:num_samples]
        if not self.oracle_topology_self_attention.enabled or v_data is None:
            topology_input = None
        elif isinstance(self.oracle_topology_self_attention, LearnedTopologySelfAttentionMask):
            topology_input = v_data["conditions"]["imgs"][:num_samples]
        else:
            topology_input = v_data["face_adj"][:num_samples]

        latent_sequence = sample_latent_sequence(
            self.diffuse,
            self.noise_scheduler,
            latent_sequence,
            condition,
            self.oracle_topology_self_attention,
            topology_input,
            show_progress=v_log,
        )

        face_mask = self.face_padder.inference_mask(latent_sequence)
        recon_data = []
        for item_idx in range(num_samples):
            face_latents = latent_sequence[item_idx:item_idx + 1][face_mask[item_idx:item_idx + 1]]
            face_latents = self.face_padder.postprocess(face_latents)
            recon_data.append(self.autoencoder.decode_latents(face_latents))
        return recon_data


__all__ = ["Diffusion"]
