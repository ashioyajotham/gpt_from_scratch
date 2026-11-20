import pytest
import torch

from gpt.attention import CausalMultiHeadAttention, SingleHeadCausalAttention
from gpt.config import GPTConfig


def test_multihead_attention_output_shape_and_weight_properties() -> None:
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=65, block_size=32, d_model=48, n_heads=4, dropout=0.0)
    attn = CausalMultiHeadAttention(cfg)
    attn.eval()

    x = torch.randn(3, 10, 48)
    out, weights = attn(x, return_weights=True)

    assert out.shape == (3, 10, 48)
    assert weights.shape == (3, 4, 10, 10)
    # Every attention row must sum to 1.0
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 4, 10), atol=1e-6)
    # Strictly upper-triangular (future) weights must be EXACTLY 0.0
    upper_mask = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
    assert torch.equal(weights[:, :, upper_mask], torch.zeros_like(weights[:, :, upper_mask]))


def test_causal_no_future_leakage_forward_and_backward() -> None:
    """Mutating tokens after position t_cut must NOT affect outputs or grads at <= t_cut."""
    torch.manual_seed(1)
    cfg = GPTConfig(vocab_size=65, block_size=16, d_model=32, n_heads=4, dropout=0.0)
    attn = CausalMultiHeadAttention(cfg)
    attn.eval()

    B, T = 2, 10
    t_cut = 4  # positions 0..4 are past/present; 5..9 are future

    x_orig = torch.randn(B, T, cfg.d_model, requires_grad=True)
    x_perturbed = x_orig.detach().clone()
    x_perturbed[:, t_cut + 1 :, :] += 1000.0  # huge perturbation in the future

    out_orig = attn(x_orig)
    out_perturbed = attn(x_perturbed)

    # Forward causality: positions 0..t_cut must be unaffected by future perturbation
    assert torch.allclose(out_orig[:, : t_cut + 1, :], out_perturbed[:, : t_cut + 1, :], atol=1e-6)
    # Sanity: future positions themselves MUST change
    assert not torch.allclose(out_orig[:, t_cut + 1 :, :], out_perturbed[:, t_cut + 1 :, :])

    # Backward causality: loss on positions 0..t_cut must give EXACT zero gradient to future inputs
    loss = out_orig[:, : t_cut + 1, :].sum()
    loss.backward()
    assert torch.equal(
        x_orig.grad[:, t_cut + 1 :, :],
        torch.zeros_like(x_orig.grad[:, t_cut + 1 :, :]),
    )
    assert x_orig.grad[:, : t_cut + 1, :].abs().sum().item() > 0.0


def test_multihead_equals_concatenated_single_heads_plus_out_proj() -> None:
    """Prove mathematically that CausalMultiHeadAttention == H parallel single heads + out_proj."""
    torch.manual_seed(2)
    cfg = GPTConfig(vocab_size=32, block_size=16, d_model=24, n_heads=3, dropout=0.0, bias=True)
    mha = CausalMultiHeadAttention(cfg)
    mha.eval()

    # Slice mha's fused qkv_proj weights/biases into 3 separate SingleHeadCausalAttention modules
    w_q, w_k, w_v = mha.qkv_proj.weight.chunk(3, dim=0)
    b_q, b_k, b_v = mha.qkv_proj.bias.chunk(3, dim=0)

    heads = []
    for h in range(cfg.n_heads):
        sl = slice(h * cfg.d_head, (h + 1) * cfg.d_head)
        head = SingleHeadCausalAttention(
            d_model=cfg.d_model,
            d_head=cfg.d_head,
            block_size=cfg.block_size,
            dropout=0.0,
            bias=True,
        )
        head.eval()
        with torch.no_grad():
            head.w_q.weight.copy_(w_q[sl])
            head.w_q.bias.copy_(b_q[sl])
            head.w_k.weight.copy_(w_k[sl])
            head.w_k.bias.copy_(b_k[sl])
            head.w_v.weight.copy_(w_v[sl])
            head.w_v.bias.copy_(b_v[sl])
        heads.append(head)

    x = torch.randn(2, 7, cfg.d_model)
    mha_out = mha(x)
    manual_concat = torch.cat([head(x) for head in heads], dim=-1)
    manual_out = mha.out_proj(manual_concat)

    assert torch.allclose(mha_out, manual_out, atol=1e-6)


def test_attention_rejects_sequence_longer_than_block_size() -> None:
    cfg = GPTConfig(vocab_size=20, block_size=8, d_model=16, n_heads=2)
    attn = CausalMultiHeadAttention(cfg)
    with pytest.raises(ValueError, match="exceeds block_size"):
        attn(torch.randn(1, 9, 16))
