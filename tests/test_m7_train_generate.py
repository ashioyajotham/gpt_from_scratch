import pytest
import torch

from gpt.config import GPTConfig
from gpt.model import GPT
from gpt.train import TrainConfig, configure_optimizer, get_lr, train_gpt


def test_generate_appends_tokens_and_crops_long_prompts() -> None:
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=20, block_size=8, d_model=16, n_heads=2, n_layers=1, dropout=0.0)
    model = GPT(cfg)

    # Prompt intentionally LONGER than block_size (12 > 8) to verify context cropping
    prompt = torch.randint(0, 20, (2, 12))
    out = model.generate(prompt, max_new_tokens=5, greedy=True)

    assert out.shape == (2, 17)
    # Original prompt must remain completely untouched
    assert torch.equal(out[:, :12], prompt)


def test_generate_greedy_is_deterministic_and_restores_train_mode() -> None:
    torch.manual_seed(1)
    cfg = GPTConfig(vocab_size=20, block_size=8, d_model=16, n_heads=2, n_layers=1, dropout=0.2)
    model = GPT(cfg)
    model.train()

    prompt = torch.randint(0, 20, (1, 4))
    out1 = model.generate(prompt, max_new_tokens=6, greedy=True)
    out2 = model.generate(prompt, max_new_tokens=6, greedy=True)

    assert torch.equal(out1, out2)
    assert model.training  # train() state was preserved


def test_generate_top_k_restricts_samples_to_top_k_tokens() -> None:
    torch.manual_seed(2)
    cfg = GPTConfig(vocab_size=30, block_size=8, d_model=16, n_heads=2, n_layers=1, dropout=0.0)
    model = GPT(cfg)
    model.eval()

    prompt = torch.tensor([[1, 2, 3]])
    logits, _ = model(prompt)
    allowed = set(torch.topk(logits[0, -1], k=3).indices.tolist())

    # Sample 40 independent 1-step continuations with top_k=3
    batch_prompt = prompt.repeat(40, 1)
    gen = torch.Generator().manual_seed(99)
    sampled = model.generate(batch_prompt, max_new_tokens=1, temperature=1.0, top_k=3, generator=gen)
    next_tokens = set(sampled[:, -1].tolist())
    assert next_tokens.issubset(allowed)


def test_generate_top_p_and_temperature_valid_and_guard_inputs() -> None:
    torch.manual_seed(3)
    cfg = GPTConfig(vocab_size=20, block_size=8, d_model=16, n_heads=2, n_layers=1, dropout=0.0)
    model = GPT(cfg)

    prompt = torch.tensor([[0, 1, 2]])
    out = model.generate(prompt, max_new_tokens=4, temperature=0.8, top_p=0.9)
    assert out.shape == (1, 7)

    with pytest.raises(ValueError, match="temperature"):
        model.generate(prompt, max_new_tokens=1, temperature=0.0, greedy=False)
    with pytest.raises(ValueError, match="top_p"):
        model.generate(prompt, max_new_tokens=1, top_p=1.5)


def test_configure_optimizer_separates_2d_and_1d_params() -> None:
    cfg = GPTConfig(vocab_size=20, block_size=8, d_model=16, n_heads=2, n_layers=2, bias=True)
    model = GPT(cfg, tie_weights=True)
    opt = configure_optimizer(model, lr=1e-3, weight_decay=0.1)

    decay_group, no_decay_group = opt.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay_group["params"])
    assert all(p.ndim < 2 for p in no_decay_group["params"])

    total_opt_params = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert total_opt_params == model.count_parameters()


def test_cosine_lr_schedule_warmup_peak_and_decay() -> None:
    max_lr, min_lr = 1e-3, 1e-4
    assert get_lr(0, 100, max_lr, min_lr, warmup_steps=10) == pytest.approx(1e-4)
    assert get_lr(9, 100, max_lr, min_lr, warmup_steps=10) == pytest.approx(max_lr)
    assert get_lr(10, 100, max_lr, min_lr, warmup_steps=10) == pytest.approx(max_lr)
    mid = get_lr(55, 100, max_lr, min_lr, warmup_steps=10)
    assert mid == pytest.approx(0.5 * (max_lr + min_lr))
    assert get_lr(100, 100, max_lr, min_lr, warmup_steps=10) == pytest.approx(min_lr)


def test_train_gpt_reduces_loss_on_repeating_pattern() -> None:
    torch.manual_seed(42)
    # Simple repeating pattern 0,1,2,3,4,5,6,7... so tiny GPT should learn it rapidly
    pattern = torch.arange(8, dtype=torch.long).repeat(60)
    train_data, val_data = pattern[:400], pattern[400:]

    cfg = GPTConfig(vocab_size=8, block_size=16, d_model=32, n_heads=2, n_layers=1, dropout=0.0)
    model = GPT(cfg)
    tcfg = TrainConfig(
        max_steps=40,
        batch_size=8,
        max_lr=5e-3,
        min_lr=1e-3,
        warmup_steps=5,
        eval_interval=20,
        eval_iters=5,
        device="cpu",
    )
    history = train_gpt(model, train_data, val_data, tcfg)
    assert history[-1]["val"] < history[0]["val"] * 0.3
