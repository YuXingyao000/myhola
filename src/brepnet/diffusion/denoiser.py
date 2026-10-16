"""时间条件 face tokens → 逐层 self-/image cross-attention → 噪声预测。"""

from __future__ import annotations

from torch import Tensor, nn

from src.brepnet.models.blocks import sincos_embedding

from .fusion import ImageCrossAttention


class BRepDenoiser(nn.Module):
    """默认 [B, 30, 32] → [B, 30, 768] → 24 层 → [B, 30, 32]。"""

    def __init__(self, latent_dim: int, cfg: dict, condition_dim: int | None):
        super().__init__()
        if cfg.get("conditioning") != "layerwise_cross_attention":
            raise ValueError("denoiser.conditioning must be 'layerwise_cross_attention'.")
        self.hidden_dim = cfg["hidden_dim"]
        self.condition_dim = condition_dim
        if (
            cfg["num_layers"] < 1
            or cfg["nhead_divisor"] < 1
            or self.hidden_dim < 1
            or self.hidden_dim % cfg["nhead_divisor"]
        ):
            raise ValueError("Use at least one layer and a positive hidden_dim divisible by nhead_divisor.")
        num_heads = self.hidden_dim // cfg["nhead_divisor"]

        # 输入/时间/输出仍使用原来的两层 MLP。
        self.input_projection = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # 保留完整原 EncoderLayer：pre-LN self-attention + residual；pre-LN FFN + residual。
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            norm_first=True,
            dim_feedforward=cfg["feedforward_dim"],
            dropout=cfg["dropout"],
            activation="relu",
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(
            layer, cfg["num_layers"], nn.LayerNorm(self.hidden_dim), enable_nested_tensor=False
        )
        # 同一份图像 memory 供各层读取；各层投影参数独立。
        self.cross_attention_layers = nn.ModuleList()
        if condition_dim is not None:
            self.cross_attention_layers.extend(
                ImageCrossAttention(self.hidden_dim, condition_dim, num_heads, cfg["dropout"])
                for _ in range(cfg["num_layers"])
            )
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, latent_dim),
        )

    def _condition_memory(self, condition: Tensor | None, batch_size: int) -> Tensor | None:
        """单视图旧接口 [B, 1, T, D] 只移除 view 轴，不合并多个视图。"""
        if self.condition_dim is None:
            if condition is not None:
                raise ValueError("An unconditional denoiser does not accept condition tokens.")
            return None
        if condition is None:
            raise ValueError("Image/condition tokens are required by this denoiser.")
        if condition.ndim == 4 and condition.shape[1] == 1:
            condition = condition[:, 0]
        if (
            condition.ndim != 3
            or condition.shape[0] != batch_size
            or condition.shape[1] == 0
            or condition.shape[2] != self.condition_dim
        ):
            raise ValueError(
                f"Expected condition [B, T, {self.condition_dim}] or [B, 1, T, {self.condition_dim}] "
                f"with B={batch_size}, T>0; got {tuple(condition.shape)}."
            )
        return condition

    def forward(
        self,
        noisy_latent_sequence: Tensor,
        timesteps: Tensor,
        condition: Tensor | None,
        self_attention_mask: Tensor | None = None,
    ) -> Tensor:
        memory = self._condition_memory(condition, noisy_latent_sequence.shape[0])
        time_embedding = self.time_embedding(sincos_embedding(timesteps, self.hidden_dim)).unsqueeze(1)
        hidden = self.input_projection(noisy_latent_sequence) + time_embedding
        hidden = self.run_backbone(hidden, memory, self_attention_mask)
        return self.output_projection(hidden)

    def run_backbone(
        self, hidden: Tensor, memory: Tensor | None, self_attention_mask: Tensor | None
    ) -> Tensor:
        """每层完整 EncoderLayer 后接图像 cross-attention；最终只做一次 LayerNorm。"""
        src_mask = None
        if self_attention_mask is not None:
            batch_size, seq_len, _ = hidden.shape
            if self_attention_mask.shape != (batch_size, seq_len, seq_len):
                raise ValueError("self_attention_mask must have shape [B, F, F].")
            num_heads = self.backbone.layers[0].self_attn.num_heads
            src_mask = self_attention_mask[:, None].expand(-1, num_heads, -1, -1)
            src_mask = src_mask.reshape(batch_size * num_heads, seq_len, seq_len)

        for index, layer in enumerate(self.backbone.layers):
            if src_mask is not None and src_mask.is_floating_point():
                hidden = self._encoder_with_bias(layer, hidden, src_mask)
            else:
                hidden = layer(hidden, src_mask=src_mask)
            if memory is not None:
                hidden = self.cross_attention_layers[index](hidden, memory)
        return self.backbone.norm(hidden)

    @staticmethod
    def _encoder_with_bias(layer: nn.TransformerEncoderLayer, hidden: Tensor, bias: Tensor) -> Tensor:
        """执行相同的完整 pre-LN EncoderLayer，保留浮点 additive bias 语义。"""
        # PyTorch 2.4 fused encoder 在 eval/no_grad 下会将 float mask 错当屏蔽。
        # 直接调用其已有子层；MHA 的 float-mask 路径会保留 additive logits。
        query = layer.norm1(hidden)
        attention, _ = layer.self_attn(query, query, query, attn_mask=bias, need_weights=False)
        hidden = hidden + layer.dropout1(attention)
        feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(hidden)))))
        return hidden + layer.dropout2(feedforward)


__all__ = ["BRepDenoiser"]
