"""Milestone 1 inspection script: tokenizer + parallel next-token targets."""

from pathlib import Path

import torch

from gpt.data import get_batch, load_text, train_val_split
from gpt.tokenizer import CharTokenizer


def main() -> None:
    corpus_path = Path("data/tiny_shakespeare.txt")
    text = load_text(corpus_path)
    tok = CharTokenizer(text)

    print(f"Corpus length : {len(text):,} characters")
    print(f"Vocab size |V|: {tok.vocab_size}")
    print(f"Vocab preview : {''.join(tok.itos[i] for i in range(tok.vocab_size))!r}")

    ids = tok.encode(text)
    train_data, val_data = train_val_split(ids, train_fraction=0.9)
    print(f"Train tokens  : {len(train_data):,} | Val tokens: {len(val_data):,}")

    gen = torch.Generator().manual_seed(0)
    B, T = 2, 12
    x, y = get_batch(train_data, batch_size=B, block_size=T, generator=gen)
    print(f"\nSampled batch shapes: x={tuple(x.shape)}, y={tuple(y.shape)}")
    print(f"Parallel next-token targets in this single batch: B * T = {B * T}\n")

    seq_x = x[0].tolist()
    seq_y = y[0].tolist()
    print(f"Sequence 0 input  x[0]: {tok.decode(seq_x)!r}")
    print(f"Sequence 0 target y[0]: {tok.decode(seq_y)!r}\n")
    print("How position t predicts token t+1 in parallel (Notes §3):")
    for t in range(T):
        context = tok.decode(seq_x[: t + 1])
        target = tok.decode([seq_y[t]])
        print(f"  pos {t:2d}: context {context!r:18s} -> target {target!r} (id={seq_y[t]})")


if __name__ == "__main__":
    main()
