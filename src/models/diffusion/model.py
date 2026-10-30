"""Diffusion 装配、训练和采样；逐层条件版本需使用新架构 checkpoint。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch import Tensor, nn

from src.models.vae import build_autoencoder

from .conditioning import build_condition_encoder
from .denoiser import BRepDenoiser
from .latents import LatentProvider
from .losses import compute_diffusion_losses
from .padding import build_face_padder
from .sampling import sample_latent_sequence
from .topology import LearnedTopologySelfAttentionMask, build_topology_self_attention


class Diffusion(nn.Module):
    """图像条件 → 面 latent DDPM → 冻结 VAE 解码。"""

    def __init__(self, cfg: dict, condition_cfg: dict):
        super().__init__()
        if "condition_fuser" in cfg:
            raise ValueError(
                "Legacy condition_fuser config is incompatible with layerwise conditioning. "
                "Use the new model/diffusion.yaml; reproduce old runs on codex/diffusion-modular-baseline."
            )
        if cfg["denoiser"].get("conditioning") != "layerwise_cross_attention":
            raise ValueError("Set denoiser.conditioning='layerwise_cross_attention' explicitly.")
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

        # 3. 图像 memory 在每一层独立注入，不再经过前端 fuser。
        self.condition_encoder = build_condition_encoder(condition_cfg)
        self.denoiser = BRepDenoiser(
            latent_dim=self.latent_dim,
            cfg=cfg["denoiser"],
            condition_dim=condition_cfg["output_dim"] if self.condition_encoder is not None else None,
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
        """保留旧调用返回格式；clean_latent_sequence 已弃用且不会被读取。"""
        prediction = self.denoiser(
            noisy_latent_sequence,
            timesteps,
            condition,
            self_attention_mask,
        )
        return prediction, None

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """禁止把旧架构或残缺权重静默加载成部分随机的新模型。"""
        if not strict:
            raise ValueError(
                "Diffusion requires strict=True. For inference with a separately loaded VAE, "
                "use load_diffusion_state_dict(); legacy fuser checkpoints require the baseline branch."
            )
        return super().load_state_dict(state_dict, strict=True, assign=assign)

    def load_diffusion_state_dict(self, state_dict: dict[str, Tensor]):
        """推理时保留已加载的 VAE，其余全部参数必须严格匹配新架构。"""
        ae_prefixes = (
            "autoencoder.", "latent_provider.autoencoder.",
            "latent_codec.autoencoder.", "ae_model.",
        )
        current = self.state_dict()
        weights = {key: value for key, value in state_dict.items() if not key.startswith(ae_prefixes)}
        expected = {key for key in current if not key.startswith(ae_prefixes)}
        missing = sorted(expected - weights.keys())
        unexpected = sorted(weights.keys() - expected)
        if missing or unexpected:
            raise RuntimeError(
                "Diffusion checkpoint does not match the layerwise architecture: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}. "
                "Old fuser checkpoints require codex/diffusion-modular-baseline."
            )
        ae_weights = {key: value for key, value in current.items() if key.startswith(ae_prefixes)}
        return self.load_state_dict({**ae_weights, **weights}, strict=True)

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

        prediction, _ = self.diffuse(
            noisy_latent_sequence,
            timesteps,
            condition,
            self_attention_mask=self_attention_mask,
        )
        target = latent_sequence if self.prediction_type == "sample" else noise
        valid_loss = self.face_padder.validity_loss(
            prediction,
            latent_batch.face_mask,
            self.valid_loss_weight,
        )
        return compute_diffusion_losses(
            prediction, target, timesteps, self.loss_fn, validity_loss=valid_loss
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
