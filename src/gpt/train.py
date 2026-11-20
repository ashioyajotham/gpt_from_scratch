from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from gpt.data import get_batch
from gpt.model import GPT


@dataclass
class TrainConfig:
    """Hyperparameters for GPT training (Notes §3)."""

    max_steps: int = 800
    batch_size: int = 32
    max_lr: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 80
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_iters: int = 20
    device: str = "auto"


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_optimizer(
    model: GPT,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    """Create AdamW with decoupled weight decay ONLY on 2D+ weight tensors.

    Biases and LayerNorm scale/shift vectors (1D parameters) are never decayed,
    matching GPT-2 / GPT-3 training best practices.
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas)


def get_lr(
    step: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    """Linear warmup followed by cosine learning-rate decay."""
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_steps = max(1, max_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    """Estimate mean train and validation cross-entropy loss in eval mode."""
    was_training = model.training
    model.eval()
    results: dict[str, float] = {}
    try:
        for split_name, split_data in (("train", train_data), ("val", val_data)):
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = get_batch(split_data, batch_size, block_size, device=device)
                _, loss = model(x, targets=y)
                assert loss is not None
                losses[k] = loss.item()
            results[split_name] = losses.mean().item()
    finally:
        model.train(was_training)
    return results


def train_gpt(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    cfg: TrainConfig,
    on_eval: Callable[[int, dict[str, float], float], None] | None = None,
) -> list[dict[str, float]]:
    """Run the complete GPT training loop and return evaluation history."""
    device = resolve_device(cfg.device)
    model.to(device)
    optimizer = configure_optimizer(
        model,
        lr=cfg.max_lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )

    history: list[dict[str, float]] = []
    block_size = model.config.block_size

    model.train()
    for step in range(cfg.max_steps):
        lr = get_lr(step, cfg.max_steps, cfg.max_lr, cfg.min_lr, cfg.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step == 0 or (step % cfg.eval_interval == 0):
            metrics = estimate_loss(
                model, train_data, val_data, cfg.batch_size, block_size, cfg.eval_iters, device
            )
            history.append({"step": float(step), "lr": lr, **metrics})
            if on_eval is not None:
                on_eval(step, metrics, lr)

        x, y = get_batch(train_data, cfg.batch_size, block_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)
        assert loss is not None
        loss.backward()
        if cfg.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    # Final evaluation after last step
    final_lr = get_lr(cfg.max_steps, cfg.max_steps, cfg.max_lr, cfg.min_lr, cfg.warmup_steps)
    final_metrics = estimate_loss(
        model, train_data, val_data, cfg.batch_size, block_size, cfg.eval_iters, device
    )
    history.append({"step": float(cfg.max_steps), "lr": final_lr, **final_metrics})
    if on_eval is not None:
        on_eval(cfg.max_steps, final_metrics, final_lr)

    return history
