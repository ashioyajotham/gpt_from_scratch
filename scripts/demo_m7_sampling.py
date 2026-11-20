"""Milestone 7 inspection script: compare greedy, temperature, top-k, and top-p decoding."""

from pathlib import Path

import torch

from gpt.data import load_text
from gpt.model import GPT
from gpt.tokenizer import CharTokenizer


def main() -> None:
    ckpt_path = Path("checkpoints/shakespeare_char.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found. Run `python scripts/train_shakespeare.py` first."
        )

    text = load_text(Path("data/tiny_shakespeare.txt"))
    tok = CharTokenizer(text)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = GPT(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    prompt = "KING:\n"
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long)

    configs = [
        ("Greedy (deterministic)", dict(greedy=True)),
        ("Low temp (T=0.5, top_k=20)", dict(temperature=0.5, top_k=20)),
        ("Balanced (T=0.8, top_p=0.9)", dict(temperature=0.8, top_p=0.9)),
        ("High temp (T=1.4, unfiltered)", dict(temperature=1.4)),
    ]

    for label, kwargs in configs:
        gen = torch.Generator().manual_seed(7)
        out = model.generate(idx, max_new_tokens=140, generator=gen, **kwargs)
        print(f"=== {label} ===")
        print(tok.decode(out[0].tolist()))
        print()


if __name__ == "__main__":
    main()
