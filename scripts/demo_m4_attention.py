"""Milestone 4 inspection script: causal mask, 1/sqrt(d_k) scaling, and contextualization."""

import math

import torch

from gpt.attention import CausalMultiHeadAttention
from gpt.config import GPTConfig
from gpt.embeddings import GPTEmbeddings
from gpt.ops import softmax


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()


def main() -> None:
    torch.manual_seed(42)

    # 1. Inspect the causal attention weight matrix for a 5-token sequence
    cfg = GPTConfig(vocab_size=10, block_size=16, d_model=32, n_heads=4, dropout=0.0)
    emb = GPTEmbeddings(cfg)
    attn = CausalMultiHeadAttention(cfg)
    emb.eval()
    attn.eval()

    tokens = torch.tensor([[0, 1, 2, 3, 4]])
    x = emb(tokens)
    _, weights = attn(x, return_weights=True)  # (1, H, 5, 5)
    head0 = weights[0, 0]

    print("1) Causal attention weights (Batch 0, Head 0) for T=5 tokens:")
    print("   Rows = query position i, Cols = key position j (future j > i is exact 0.000)")
    for i in range(5):
        row = "  ".join(f"{head0[i, j].item():.3f}" for j in range(5))
        print(f"   pos {i}: [{row}]  sum={head0[i].sum().item():.3f}")

    # 2. Why scale by 1 / sqrt(d_k)? Compare score variance & softmax peaking at d_k = 64
    d_k = 64
    q = torch.randn(1, 20, d_k)  # unit-variance entries
    k = torch.randn(1, 20, d_k)
    unscaled = q @ k.transpose(-2, -1)
    scaled = unscaled / math.sqrt(d_k)
    p_unscaled = softmax(unscaled, dim=-1)
    p_scaled = softmax(scaled, dim=-1)

    print(f"\n2) Why scale QK^T by 1/sqrt(d_k) (d_k={d_k}):")
    print(f"   Unscaled QK^T var = {unscaled.var().item():6.2f} | max softmax prob = {p_unscaled.max(dim=-1).values.mean().item():.3f} (nearly one-hot -> vanishing grads!)")
    print(f"   Scaled   QK^T var = {scaled.var().item():6.2f} | max softmax prob = {p_scaled.max(dim=-1).values.mean().item():.3f} (well-behaved distribution)")

    # 3. Milestone 3 payoff: 'bites' at pos 1 in 'dog bites man' vs 'man bites dog'
    s1 = torch.tensor([[0, 1, 2]])  # "dog bites man"
    s2 = torch.tensor([[2, 1, 0]])  # "man bites dog"
    e1, e2 = emb(s1), emb(s2)
    a1, a2 = attn(e1), attn(e2)

    print("\n3) Contextualizing 'bites' (pos 1) in 'dog bites man' vs 'man bites dog':")
    print(f"   Cosine sim BEFORE self-attention : {cosine_sim(e1[0, 1], e2[0, 1]):.4f} (blind to neighbors)")
    print(f"   Cosine sim AFTER  self-attention : {cosine_sim(a1[0, 1], a2[0, 1]):.4f} (now reflects 'dog' vs 'man' at pos 0!)")


if __name__ == "__main__":
    main()
