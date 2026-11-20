"""Milestone 3 inspection script: 'dog bites man' vs 'man bites dog' (Notes §1.2)."""

import torch

from gpt.config import GPTConfig
from gpt.embeddings import GPTEmbeddings


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()


def main() -> None:
    torch.manual_seed(42)
    cfg = GPTConfig(vocab_size=10, block_size=8, d_model=32, n_heads=4, dropout=0.0)
    emb = GPTEmbeddings(cfg)
    emb.eval()

    # Toy word-level IDs: dog=0, bites=1, man=2
    s1 = torch.tensor([[0, 1, 2]])  # "dog bites man"
    s2 = torch.tensor([[2, 1, 0]])  # "man bites dog"

    # Raw token embeddings (before adding positional encoding)
    tok1 = emb.token_embedding(s1)  # (1, 3, 32)
    tok2 = emb.token_embedding(s2)  # (1, 3, 32)

    # Full embeddings (token + learned positional encoding)
    full1 = emb(s1)  # (1, 3, 32)
    full2 = emb(s2)  # (1, 3, 32)

    print("Shape trace: (B=1, T=3) -> (B=1, T=3, d_model=32)")
    print(f"  tok_emb shape : {tuple(tok1.shape)}")
    print(f"  pos_emb shape : ({s1.size(1)}, {cfg.d_model})  [broadcast across batch B]")
    print(f"  output  shape : {tuple(full1.shape)}\n")

    print("Comparing the word 'dog' in 'dog bites man' (pos 0) vs 'man bites dog' (pos 2):")
    print(f"  Cosine sim BEFORE positional encoding : {cosine_sim(tok1[0, 0], tok2[0, 2]):.4f} (identical!)")
    print(f"  Cosine sim AFTER  positional encoding : {cosine_sim(full1[0, 0], full2[0, 2]):.4f} (now order-aware!)")

    print("\nComparing the word 'bites' at pos 1 in both sentences:")
    print(f"  Cosine sim AFTER  positional encoding : {cosine_sim(full1[0, 1], full2[0, 1]):.4f} (same word + same pos -> identical)")


if __name__ == "__main__":
    main()
