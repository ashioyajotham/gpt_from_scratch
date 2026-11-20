"""Milestone 7: Train our first-principles GPT on Tiny Shakespeare and generate text!"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from gpt.config import GPTConfig
from gpt.data import load_text, train_val_split
from gpt.model import GPT
from gpt.tokenizer import CharTokenizer
from gpt.train import TrainConfig, resolve_device, train_gpt


def sample_text(
    model: GPT,
    tok: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 0.8,
    top_k: int | None = 20,
) -> str:
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    gen = torch.Generator(device=device).manual_seed(42)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=gen,
    )
    return tok.decode(out[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train from-scratch GPT on Tiny Shakespeare")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save", type=str, default="checkpoints/shakespeare_char.pt")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = resolve_device(args.device)

    text = load_text(Path("data/tiny_shakespeare.txt"))
    tok = CharTokenizer(text)
    train_data, val_data = train_val_split(tok.encode(text), train_fraction=0.9)

    model_cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    model = GPT(model_cfg).to(device)
    print(f"Device: {device} | Parameters: {model.count_parameters():,}")
    print(f"Config: L={model_cfg.n_layers}, H={model_cfg.n_heads}, d_model={model_cfg.d_model}, T={model_cfg.block_size}\n")

    prompt = "ROMEO:\n"
    print("--- Sample BEFORE training (step 0) ---")
    print(sample_text(model, tok, prompt, max_new_tokens=140, device=device))
    print("---------------------------------------\n")

    t0 = time.perf_counter()

    def log_eval(step: int, metrics: dict[str, float], lr: float) -> None:
        elapsed = time.perf_counter() - t0
        print(
            f"step {step:4d}/{args.steps} | "
            f"train loss {metrics['train']:.4f} | "
            f"val loss {metrics['val']:.4f} | "
            f"lr {lr:.2e} | {elapsed:5.1f}s"
        )

    train_cfg = TrainConfig(
        max_steps=args.steps,
        batch_size=args.batch_size,
        max_lr=args.lr,
        min_lr=args.lr * 0.1,
        warmup_steps=min(60, args.steps // 10),
        eval_interval=max(1, args.steps // 4),
        eval_iters=20,
        device=str(device),
    )
    train_gpt(model, train_data, val_data, train_cfg, on_eval=log_eval)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": model_cfg, "state_dict": model.state_dict()}, save_path)
        print(f"\nSaved checkpoint to {save_path}")

    print("\n--- Sample AFTER training (temperature=0.8, top_k=20) ---")
    print(sample_text(model, tok, prompt, max_new_tokens=280, device=device, temperature=0.8, top_k=20))
    print("---------------------------------------------------------")


if __name__ == "__main__":
    main()
