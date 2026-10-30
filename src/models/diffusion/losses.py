"""Diffusion 逐元素训练目标与可选 padding validity loss。"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def compute_diffusion_losses(
    prediction: Tensor,
    target: Tensor,
    timesteps: Tensor,
    loss_fn: Callable[..., Tensor],
    validity_loss: Tensor | None = None,
) -> dict[str, Tensor]:
    """汇总 [B, F, C] 逐元素损失，保留 trainer 消费的原字段与归约方式。"""
    loss_item = loss_fn(prediction, target, reduction="none")
    losses = {
        "diffusion_loss": loss_item.mean(),
        "t": torch.stack((timesteps, loss_item.mean(dim=1).mean(dim=1)), dim=1),
    }
    if validity_loss is not None:
        losses["valid_loss"] = validity_loss
    losses["total_loss"] = sum(value for key, value in losses.items() if key != "t")
    return losses
