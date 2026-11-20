"""Milestone 6 inspection script: parameter audit, init loss check, and single-batch overfit."""

import math
from pathlib import Path

import torch

from gpt.config import GPTConfig
from gpt.data import get_batch, load_text, train_val_split
from gpt.model import GPT
from gpt.tokenizer import CharTokenizer


def main() -> None:
    torch.manual_seed(42)
    text = load_text(Path("data/tiny_shakespeare.txt"))
    tok = CharTokenizer(text)
    train_data, _ = train_val_split(tok.encode(text))

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=32,
        d_model=64,
        n_heads=4,
        n_layers=3,
        dropout=0.0,
    )
    model = GPT(cfg, tie_weights=True)
    untied_params = GPT(cfg, tie_weights=False).count_parameters()

    print("1) GPT Parameter Audit:")
    print(f"   Config        : L={cfg.n_layers}, H={cfg.n_heads}, d_model={cfg.d_model}, |V|={cfg.vocab_size}, T={cfg.block_size}")
    print(f"   Tied params   : {model.count_parameters():,}")
    print(f"   Untied params : {untied_params:,} (weight tying saves {untied_params - model.count_parameters():,} = |V|*d_model)")

    # Grab ONE fixed batch of real Shakespeare text
    gen = torch.Generator().manual_seed(7)
    x, y = get_batch(train_data, batch_size=2, block_size=cfg.block_size, generator=gen)

    # 2. Initial loss sanity check (should be ~ ln(65) = 4.1744)
    model.eval()
    _, init_loss = model(x, y)
    print(f"\n2) Initial Loss Sanity Check:")
    print(f"   Untrained loss = {init_loss.item():.4f} | Expected ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.4f}")

    # 3. Single-batch overfit test: prove end-to-end gradient flow
    print("\n3) Single-Batch Overfit Sanity Check (memorizing 1 Shakespeare batch):")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for step in range(1, 151):
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if step in (1, 25, 50, 100, 150):
            acc = (logits.argmax(dim=-1) == y).float().mean().item()
            print(f"   step {step:3d} | loss = {loss.item():.4f} | next-char accuracy = {100 * acc:5.1f}%")

    model.eval()
    logits, _ = model(x, y)
    pred_text = tok.decode(logits[0].argmax(dim=-1).tolist())
    true_text = tok.decode(y[0].tolist())
    print(f"\n   True target [0]: {true_text!r}")
    print(f"   Predicted   [0]: {pred_text!r}")


if __name__ == "__main__":
    main()
