from __future__ import annotations

import math

import torch
import torch.nn as nn

from gpt.config import GPTConfig
from gpt.ops import Dropout, Linear, softmax


class SingleHeadCausalAttention(nn.Module):
    """Single-head masked self-attention (Notes §2, Sub-Layer A — 1 head).

    Explicitly separates W_q, W_k, W_v to make the 4 steps inside a head transparent:
      1. First Matmul : scores = (Q @ K^T) / sqrt(d_k)          -> (B, T, T)
      2. Causal Mask  : overwrite future entries (j > i) with -inf
      3. Softmax+Drop : weights = Dropout(softmax(scores))      -> (B, T, T)
      4. Second Matmul: out = weights @ V                       -> (B, T, d_k)
    """

    def __init__(
        self,
        d_model: int,
        d_head: int,
        block_size: int,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_head = d_head
        self.w_q = Linear(d_model, d_head, bias=bias)
        self.w_k = Linear(d_model, d_head, bias=bias)
        self.w_v = Linear(d_model, d_head, bias=bias)
        self.attn_dropout = Dropout(dropout)

        # Lower-triangular boolean mask: True where token i is allowed to attend to j (j <= i)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        if T > self.causal_mask.size(0):
            raise ValueError(f"Sequence length T={T} exceeds block_size={self.causal_mask.size(0)}")

        q = self.w_q(x)  # (B, T, d_head)
        k = self.w_k(x)  # (B, T, d_head)
        v = self.w_v(x)  # (B, T, d_head)

        # 1. Scaled dot-product affinity scores: (B, T, d_head) @ (B, d_head, T) -> (B, T, T)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))

        # 2. Causal look-ahead mask: future positions become -inf
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(~mask, float("-inf"))

        # 3. Softmax (-inf -> exact 0.0) + attention dropout
        weights = softmax(scores, dim=-1)
        weights_dropped = self.attn_dropout(weights)

        # 4. Weighted blend of Value vectors: (B, T, T) @ (B, T, d_head) -> (B, T, d_head)
        out = weights_dropped @ v
        if return_weights:
            return out, weights
        return out


class CausalMultiHeadAttention(nn.Module):
    """Masked Multi-Head Self-Attention (Notes §2, Sub-Layer A).

    Fuses all H heads' Q, K, V projections into a single Linear(d_model -> 3*d_model),
    computes scaled causal attention across all H heads in parallel, concatenates
    head outputs back to d_model, and mixes them via an output Linear + Dropout.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_head = config.d_head
        self.block_size = config.block_size

        # Single fused projection for Q, K, V across all H heads
        self.qkv_proj = Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # Output projection mixes information across concatenated heads
        self.out_proj = Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = Dropout(config.dropout)
        self.resid_dropout = Dropout(config.dropout)

        # Shape (1, 1, block_size, block_size) broadcasts cleanly over (B, H, T, T)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        if C != self.d_model:
            raise ValueError(f"Expected last dim d_model={self.d_model}, got {C}")
        if T > self.block_size:
            raise ValueError(f"Sequence length T={T} exceeds block_size={self.block_size}")

        # Project (B, T, d_model) -> (B, T, 3 * d_model) and split into Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)  # each: (B, T, d_model)

        # Split d_model into (H, d_head) and move H before T for batched matmul:
        # (B, T, H * d_head) -> (B, T, H, d_head) -> (B, H, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # 1. First Matmul: Q @ K^T / sqrt(d_k) -> (B, H, T, T)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))

        # 2. Causal Mask: disallow attending to future tokens j > i
        mask = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(~mask, float("-inf"))

        # 3. Softmax & Attention Dropout -> (B, H, T, T)
        weights = softmax(scores, dim=-1)
        weights_dropped = self.attn_dropout(weights)

        # 4. Second Matmul: Weights @ V -> (B, H, T, d_head)
        head_out = weights_dropped @ v

        # Concatenate all H heads back to (B, T, d_model):
        # (B, H, T, d_head) -> (B, T, H, d_head) -> (B, T, d_model)
        concat = head_out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # Output Linear + Residual Dropout -> (B, T, d_model)
        out = self.resid_dropout(self.out_proj(concat))
        if return_weights:
            return out, weights
        return out
