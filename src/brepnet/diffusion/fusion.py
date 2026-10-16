"""每个去噪层独立读取图像 memory；只更新 face tokens。"""

from __future__ import annotations

from torch import Tensor, nn


class ImageCrossAttention(nn.Module):
    """Pre-LN cross-attention residual: [B, F, H] × [B, T, D] → [B, F, H]。"""

    def __init__(self, hidden_dim: int, condition_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        # 默认 Q: 768→768，K/V: 1024→768，12 heads × 64；包含输出投影。
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            kdim=condition_dim,
            vdim=condition_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, memory: Tensor) -> Tensor:
        update, _ = self.attention(
            self.query_norm(hidden), memory, memory, need_weights=False
        )
        return hidden + self.dropout(update)


__all__ = ["ImageCrossAttention"]
