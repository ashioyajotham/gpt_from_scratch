"""Milestone 5 inspection script: residual stream accumulation & Pre-LayerNorm gradient highway."""

import torch

from gpt.block import TransformerBlock
from gpt.config import GPTConfig
from gpt.ops import LayerNorm


def main() -> None:
    torch.manual_seed(42)
    L = 8
    cfg = GPTConfig(vocab_size=65, block_size=16, d_model=64, n_heads=4, n_layers=L, dropout=0.0)
    blocks = [TransformerBlock(cfg) for _ in range(L)]
    for b in blocks:
        b.eval()
    final_ln = LayerNorm(cfg.d_model, bias=cfg.bias)

    # Start with unit-variance embeddings (just like after GPTEmbeddings)
    x0 = torch.randn(4, 12, cfg.d_model, requires_grad=True)
    x = x0

    print("1) Residual stream std across L=8 stacked Pre-LayerNorm blocks (init_std=0.02):")
    print(f"   Input (before Layer 1)  : std = {x.std().item():.4f}")
    for i, block in enumerate(blocks, start=1):
        x = block(x)
        print(f"   After TransformerBlock {i}: std = {x.std().item():.4f}")

    # Apply Final LayerNorm (Notes §1.5)
    x_normed = final_ln(x)
    print(f"   After Final LayerNorm   : std = {x_normed.std().item():.4f}  (stabilized before unembedding!)")

    # 2. Check unimpeded gradient flow from Layer L all the way back to x0
    #    Project onto a random target direction (like an unembedding head)
    target_dir = torch.randn_like(x_normed)
    loss = (x_normed * target_dir).mean()
    loss.backward()
    print(f"\n2) Gradient norm reaching Input x0 through {L} blocks: {x0.grad.norm().item():.4f} (healthy, non-vanishing!)")

    # 3. Parameter breakdown inside a single TransformerBlock
    attn_params = sum(p.numel() for p in blocks[0].attn.parameters())
    ffn_params = sum(p.numel() for p in blocks[0].ffn.parameters())
    ln_params = sum(p.numel() for p in blocks[0].ln1.parameters()) + sum(
        p.numel() for p in blocks[0].ln2.parameters()
    )
    total = attn_params + ffn_params + ln_params
    print(f"\n3) Single TransformerBlock parameter share (d_model={cfg.d_model}):")
    print(f"   Attention (4 * d^2) : {attn_params:6,d} ({100 * attn_params / total:4.1f}%)")
    print(f"   FFN       (8 * d^2) : {ffn_params:6,d} ({100 * ffn_params / total:4.1f}%)")
    print(f"   LayerNorms          : {ln_params:6,d} ({100 * ln_params / total:4.1f}%)")
    print(f"   Total per block     : {total:6,d}")


if __name__ == "__main__":
    main()
