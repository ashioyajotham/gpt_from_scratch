from __future__ import annotations

from pathlib import Path

import torch


def load_text(path: str | Path) -> str:
    """Read a UTF-8 text corpus from disk."""
    return Path(path).read_text(encoding="utf-8")


def train_val_split(
    token_ids: list[int], train_fraction: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a flat token stream into contiguous train and validation 1D int64 tensors.

    We split contiguously (not by random token shuffling) so validation windows
    do not leak substrings from the training set.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    data = torch.tensor(token_ids, dtype=torch.long)
    split_idx = int(len(data) * train_fraction)
    return data[:split_idx], data[split_idx:]


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) token sequences of shape (B, T).

    For every random start index i:
        x = data[i     : i + T    ]   (context tokens)
        y = data[i + 1 : i + T + 1]   (next-token targets shifted by +1)

    Because of causal masking (Notes §3), a single (B, T) forward pass trains on
    all B * T prefix -> next-token pairs simultaneously.
    """
    if data.ndim != 1:
        raise ValueError(f"Expected 1D token tensor, got shape {tuple(data.shape)}")
    max_start = len(data) - block_size
    if max_start <= 0:
        raise ValueError(
            f"Corpus length ({len(data)}) must be strictly greater than block_size ({block_size})"
        )

    starts = torch.randint(low=0, high=max_start, size=(batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
