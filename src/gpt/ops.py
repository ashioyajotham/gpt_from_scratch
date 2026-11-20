from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


class Linear(nn.Module):
    """First-principles affine transformation: y = x @ W^T + b.

    Used throughout GPT for Q/K/V projections, output projections, FFN expansion/
    contraction, and the unembedding head (Notes §1.6 & §2).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Shape (out_features, in_features) matches PyTorch convention & weight tying
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, in_features) -> out: (*, out_features)
        out = x @ self.weight.transpose(-2, -1)
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(nn.Module):
    """First-principles lookup table W_e in R^(|V| x d_model) (Notes §1.1 & §1.2).

    Maps integer token IDs of shape (*) -> dense vectors of shape (*, embedding_dim).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # Pure row indexing into the embedding matrix W_e
        return self.weight[indices]


class LayerNorm(nn.Module):
    """First-principles Layer Normalization across the feature dimension (Notes §1.5 & §2).

    For each token vector x in R^(d_model):
        mean = mean(x)
        var  = mean((x - mean)^2)          # population variance (unbiased=False)
        x_hat = (x - mean) / sqrt(var + eps)
        out   = weight * x_hat + bias      # learnable scale (gamma) and shift (beta)
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_hat = (x - mean) * torch.rsqrt(var + self.eps)
        out = x_hat * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class GELU(nn.Module):
    """Gaussian Error Linear Unit: GELU(x) = x * Phi(x) (Notes §2, Sub-Layer B).

    Supports both:
      - approximate="none": exact x * 0.5 * (1 + erf(x / sqrt(2)))
      - approximate="tanh": GPT-2 tanh approximation
    """

    def __init__(self, approximate: Literal["none", "tanh"] = "none") -> None:
        super().__init__()
        if approximate not in ("none", "tanh"):
            raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.approximate == "none":
            return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
        coeff = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(coeff * (x + 0.044715 * (x**3))))


class Dropout(nn.Module):
    """Inverted Dropout (Notes §1.3 & §2).

    During training, zeroes elements with probability p and scales survivors by
    1 / (1 - p) so E[Dropout(x)] == x. During eval(), acts as the identity.
    """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability p must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = (torch.rand_like(x) < keep_prob).to(x.dtype)
        return (x * mask) / keep_prob


def softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax via max-subtraction (Notes §1.7 & §2).

    Subtracting m = max(z) leaves probabilities mathematically identical while
    ensuring the largest exponent is exp(0) = 1.0 (preventing float overflow).
    """
    max_val = logits.max(dim=dim, keepdim=True).values
    shifted = logits - max_val
    exp_x = torch.exp(shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable log-softmax using the log-sum-exp identity.

    Computes (z - m) - log(sum(exp(z - m))) directly, avoiding log(0) = -inf
    when softmax underflows tiny probabilities to 0.
    """
    max_val = logits.max(dim=dim, keepdim=True).values
    shifted = logits - max_val
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True))
    return shifted - log_sum_exp


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood across all token predictions (Notes §3).

    Args:
        logits: Raw unnormalized scores of shape (N, V) or (B, T, V).
        targets: Ground-truth integer token IDs of shape (N,) or (B, T).

    Returns:
        Scalar mean cross-entropy loss.
    """
    if logits.ndim == 3:
        B, T, V = logits.shape
        logits = logits.reshape(B * T, V)
        targets = targets.reshape(B * T)
    elif logits.ndim != 2:
        raise ValueError(f"Expected logits with 2 or 3 dims, got shape {tuple(logits.shape)}")

    log_probs = log_softmax(logits, dim=-1)  # (N, V)
    row_indices = torch.arange(logits.size(0), device=logits.device)
    target_log_probs = log_probs[row_indices, targets]  # (N,)
    return -target_log_probs.mean()
