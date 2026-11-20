"""Milestone 2 inspection script: verify numerical stability & LayerNorm statistics."""

import math

import torch

from gpt.ops import GELU, LayerNorm, cross_entropy_loss, softmax


def main() -> None:
    torch.manual_seed(0)

    # 1. LayerNorm: zero mean, unit variance across feature dimension per token
    x = torch.randn(2, 4, 16) * 5.0 + 12.0  # badly shifted & scaled activations
    ln = LayerNorm(16)
    y = ln(x)
    print("1) LayerNorm per-token stats (before -> after):")
    print(f"   Before: mean={x[0, 0].mean():+.3f}, std={x[0, 0].std(unbiased=False):.3f}")
    print(f"   After : mean={y[0, 0].mean():+.3f}, std={y[0, 0].std(unbiased=False):.3f}")

    # 2. Softmax numerical stability demonstration
    extreme = torch.tensor([1000.0, 1002.0, 998.0])
    naive_softmax = torch.exp(extreme) / torch.exp(extreme).sum()
    stable_softmax = softmax(extreme)
    print("\n2) Softmax on extreme logits [1000, 1002, 998]:")
    print(f"   Naive  exp(z)/sum(exp(z)): {naive_softmax.tolist()}")
    print(f"   Stable max-shifted       : {[round(v, 4) for v in stable_softmax.tolist()]}")

    # 3. Exact GELU vs GPT-2 tanh approximation
    xs = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    exact = GELU("none")(xs)
    approx = GELU("tanh")(xs)
    max_diff = (exact - approx).abs().max().item()
    print(f"\n3) Exact GELU vs tanh approx max diff on [-2..2]: {max_diff:.6f}")

    # 4. Initial cross-entropy loss sanity check: uniform logits -> ln(|V|)
    V = 65
    zero_logits = torch.zeros(2, 8, V)
    targets = torch.randint(0, V, (2, 8))
    loss = cross_entropy_loss(zero_logits, targets)
    print(f"\n4) Cross-entropy on uniform logits (|V|={V}):")
    print(f"   Computed loss = {loss.item():.4f} | Theoretical ln({V}) = {math.log(V):.4f}")


if __name__ == "__main__":
    main()
