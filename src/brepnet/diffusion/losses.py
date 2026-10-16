"""Diffusion 与全局图像/CAD alignment 的训练目标。"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


def clip_style_symmetric_infonce(cad_embedding: Tensor, condition_embedding: Tensor, temperature: float) -> Tensor:
    """对齐同一对象的全局 embedding [B, D]，批内其他对象作为负样本。"""
    logits = cad_embedding @ condition_embedding.T / temperature
    labels = torch.arange(cad_embedding.shape[0], device=cad_embedding.device)
    cad_to_condition = F.cross_entropy(logits, labels)
    condition_to_cad = F.cross_entropy(logits.T, labels)
    return 0.5 * (cad_to_condition + condition_to_cad)


def compute_diffusion_losses(
    prediction: Tensor,
    target: Tensor,
    timesteps: Tensor,
    loss_fn: Callable[..., Tensor],
    alignment_loss: Tensor | None = None,
    validity_loss: Tensor | None = None,
) -> dict[str, Tensor]:
    """汇总 [B, F, C] 逐元素损失，保留 trainer 消费的原字段与归约方式。"""
    loss_item = loss_fn(prediction, target, reduction="none")
    losses = {
        "diffusion_loss": loss_item.mean(),
        "t": torch.stack((timesteps, loss_item.mean(dim=1).mean(dim=1)), dim=1),
    }
    if alignment_loss is not None:
        losses["align_loss"] = alignment_loss
    if validity_loss is not None:
        losses["valid_loss"] = validity_loss
    losses["total_loss"] = sum(value for key, value in losses.items() if key != "t")
    return losses
