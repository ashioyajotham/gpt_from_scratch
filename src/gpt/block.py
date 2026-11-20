from __future__ import annotations

import torch
import torch.nn as nn

from gpt.attention import CausalMultiHeadAttention
from gpt.config import GPTConfig
from gpt.ops import GELU, Dropout, LayerNorm, Linear


class FeedForward(nn.Module):
    """Position-Wise Feed-Forward Network (Notes §2, Sub-Layer B).

    Processes each token vector independently (zero cross-token interaction):
        d_model -> 4 * d_model -> GELU -> d_model -> Dropout
    """

    def __init__(self, config: GPTConfig, expansion_factor: int = 4) -> None:
        super().__init__()
        hidden_dim = expansion_factor * config.d_model
        self.fc1 = Linear(config.d_model, hidden_dim, bias=config.bias)
        self.act = GELU(approximate="none")
        self.fc2 = Linear(hidden_dim, config.d_model, bias=config.bias)
        self.dropout = Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d_model) -> (B, T, 4*d_model) -> (B, T, d_model)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Single Pre-LayerNorm GPT Transformer Block (Notes §2, Right Diagram).

    Combines:
      - Sub-Layer A (Token-to-token communication): x = x + MHA(LN_1(x))
      - Sub-Layer B (Per-token computation)       : x = x + FFN(LN_2(x))

    Placing LayerNorm *before* each sub-layer keeps the right-hand skip path
    completely unaltered, forming an unimpeded gradient highway across L blocks.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalMultiHeadAttention(config)
        self.ln2 = LayerNorm(config.d_model, bias=config.bias)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sub-Layer A: Pre-LayerNorm -> Masked Multi-Head Self-Attention -> Residual (+)
        x = x + self.attn(self.ln1(x))
        # Sub-Layer B: Pre-LayerNorm -> Feed-Forward Network -> Residual (+)
        x = x + self.ffn(self.ln2(x))
        return x
