import math

import torch

from gpt.config import GPTConfig
from gpt.model import GPT


def test_gpt_forward_shapes_and_init_loss_near_log_vocab() -> None:
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=65, block_size=32, d_model=64, n_heads=4, n_layers=2, dropout=0.0)
    model = GPT(cfg)
    model.eval()

    x = torch.randint(0, cfg.vocab_size, (4, 24))
    y = torch.randint(0, cfg.vocab_size, (4, 24))

    logits, loss = model(x, targets=y)
    assert logits.shape == (4, 24, 65)
    assert loss is not None and loss.ndim == 0

    # Untrained model with small init should start very close to ln(|V|) = ln(65) ~= 4.174
    expected_init_loss = math.log(cfg.vocab_size)
    assert abs(loss.item() - expected_init_loss) < 0.25


def test_weight_tying_shares_tensor_and_reduces_parameter_count() -> None:
    cfg = GPTConfig(vocab_size=65, block_size=32, d_model=64, n_heads=4, n_layers=2)
    tied = GPT(cfg, tie_weights=True)
    untied = GPT(cfg, tie_weights=False)

    assert tied.lm_head.weight is tied.embeddings.token_embedding.weight
    assert untied.lm_head.weight is not untied.embeddings.token_embedding.weight

    saved_params = untied.count_parameters() - tied.count_parameters()
    assert saved_params == cfg.vocab_size * cfg.d_model


def test_gpt_end_to_end_causality() -> None:
    """Changing future token IDs must not affect earlier positions' output logits."""
    torch.manual_seed(1)
    cfg = GPTConfig(vocab_size=65, block_size=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model = GPT(cfg)
    model.eval()

    x1 = torch.randint(0, cfg.vocab_size, (2, 12))
    x2 = x1.clone()
    t_cut = 5
    # Change all tokens after position t_cut
    x2[:, t_cut + 1 :] = (x2[:, t_cut + 1 :] + 1) % cfg.vocab_size

    logits1, _ = model(x1)
    logits2, _ = model(x2)

    assert torch.allclose(logits1[:, : t_cut + 1, :], logits2[:, : t_cut + 1, :], atol=1e-5)
    assert not torch.allclose(logits1[:, t_cut + 1 :, :], logits2[:, t_cut + 1 :, :])


def test_gpt_overfits_single_batch_to_near_zero_loss() -> None:
    """Gold-standard sanity check: a working GPT must easily memorize one batch."""
    torch.manual_seed(42)
    cfg = GPTConfig(vocab_size=32, block_size=16, d_model=48, n_heads=4, n_layers=2, dropout=0.0)
    model = GPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # One fixed (x, y) batch
    seq = torch.randint(0, cfg.vocab_size, (2, 17))
    x, y = seq[:, :-1], seq[:, 1:]

    model.train()
    _, init_loss = model(x, y)
    for _ in range(150):
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

    model.eval()
    logits, final_loss = model(x, y)
    preds = logits.argmax(dim=-1)
    acc = (preds == y).float().mean().item()

    assert init_loss.item() > 3.0
    assert final_loss.item() < 0.05
    assert acc == 1.0
