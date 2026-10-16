"""历史条件组件：保留公开接口与 checkpoint 中的参数名称。

当前单视图路线不使用相机 embedding，也不启用文本条件。
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class CameraEmbedding(nn.Module):
    """Legacy learnable camera view embedding, inactive in single-view forward.

    Maps discrete camera view indices (0..7) to dense embeddings.

    Parameters
    ----------
    num_views : int
        Number of discrete camera views (default 8).
    output_dim : int
        Output embedding dimension (default 1024).
    """

    def __init__(self, num_views: int = 8, output_dim: int = 1024):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(num_views, 256),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, view_ids: torch.Tensor) -> torch.Tensor:
        """Compute camera embeddings.

        Parameters
        ----------
        view_ids : LongTensor of shape [B] or [B, num_views]
            Camera view indices.

        Returns
        -------
        Tensor of shape matching input with last dim = output_dim
        """
        return self.embedding(view_ids)


class TextEncoder(nn.Module):
    """Legacy frozen text encoder with learnable projection.

    Uses 'Alibaba-NLP/gte-large-en-v1.5' as the backbone.
    The current condition factory rejects text conditioning; direct construction
    remains available for callers using the historical public class.

    Parameters
    ----------
    output_dim : int
        Output feature dimension (default 1024).
    model_path : str
        HuggingFace model path for sentence-transformers.
    """

    def __init__(
        self,
        output_dim: int = 1024,
        model_path: str = 'Alibaba-NLP/gte-large-en-v1.5',
    ):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.txt_model = SentenceTransformer(model_path, trust_remote_code=True)
        for param in self.txt_model.parameters():
            param.requires_grad = False
        self.txt_model.eval()

        self.projection = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, output_dim),
        )

    def train(self, mode: bool = True):
        """Keep the frozen text model in eval mode regardless of parent state."""
        super().train(mode)
        self.txt_model.eval()
        return self

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of text strings.

        Parameters
        ----------
        texts : List[str]
            Batch of text descriptions.

        Returns
        -------
        Tensor of shape [B, output_dim]
        """
        with torch.no_grad():
            txt_feat = self.txt_model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=False,
                device=self.txt_model.device,
            )
            txt_feat = torch.stack(txt_feat, dim=0)

        return self.projection(txt_feat)
