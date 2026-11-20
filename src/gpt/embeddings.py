from __future__ import annotations

import torch
import torch.nn as nn

from gpt.config import GPTConfig
from gpt.ops import Dropout, Embedding


class GPTEmbeddings(nn.Module):
    """Input Embedding + Learned Positional Encoding + Dropout (Notes §1.1–1.3).

    Maps discrete token IDs of shape (B, T) -> contextualizable residual-stream
    vectors of shape (B, T, d_model):
        x = Dropout(W_e[idx] + W_p[0..T-1])
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.block_size = config.block_size
        # Token lookup table W_e in R^(|V| x d_model)
        self.token_embedding = Embedding(config.vocab_size, config.d_model)
        # Learned positional lookup table W_p in R^(block_size x d_model)
        self.pos_embedding = Embedding(config.block_size, config.d_model)
        # Regularize combined embeddings before Layer 1 (Notes §1.3)
        self.dropout = Dropout(config.dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError(f"Expected 2D token tensor (B, T), got shape {tuple(idx.shape)}")
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(
                f"Sequence length T={T} exceeds maximum block_size={self.block_size}"
            )

        # (B, T) -> (B, T, d_model)
        tok_emb = self.token_embedding(idx)

        # Position indices [0, 1, ..., T-1] -> (T, d_model), broadcast over batch B
        positions = torch.arange(T, device=idx.device)
        pos_emb = self.pos_embedding(positions)  # (T, d_model)

        # Element-wise addition (+) followed by Dropout -> (B, T, d_model)
        return self.dropout(tok_emb + pos_emb)
