"""DDPM 反向迭代；网络构建、条件提取和 VAE 解码由主模型负责。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from torch import Tensor, nn
from tqdm import tqdm

if TYPE_CHECKING:
    from diffusers import DDPMScheduler


def sample_latent_sequence(
    denoise: Callable[..., tuple[Tensor, Tensor | None]],
    noise_scheduler: DDPMScheduler,
    latent_sequence: Tensor,
    condition: Tensor | None,
    topology_attention: nn.Module,
    topology_input: Tensor | None,
    *,
    show_progress: bool = True,
) -> Tensor:
    """将初始高斯噪声 [B, F, C] 逐步去噪为面 latent，沿用 scheduler 的步序。"""
    for timestep in tqdm(noise_scheduler.timesteps, disable=not show_progress):
        timesteps = timestep.reshape(-1).to(latent_sequence.device)
        attention_mask = topology_attention(topology_input, timesteps)
        prediction, _ = denoise(
            latent_sequence,
            timesteps,
            condition,
            self_attention_mask=attention_mask,
        )
        latent_sequence = noise_scheduler.step(
            prediction, timestep, latent_sequence
        ).prev_sample
    return latent_sequence
