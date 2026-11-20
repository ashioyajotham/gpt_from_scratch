import torch

from gpt.block import FeedForward, TransformerBlock
from gpt.config import GPTConfig


def test_feedforward_is_strictly_position_wise() -> None:
    """FFN must process every token independently with zero cross-token leakage."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=65, block_size=16, d_model=32, n_heads=4, dropout=0.0)
    ffn = FeedForward(cfg)
    ffn.eval()

    x = torch.randn(2, 8, 32)
    x_mod = x.clone()
    # Wildly perturb every token EXCEPT position 3
    x_mod[:, :3, :] += 500.0
    x_mod[:, 4:, :] -= 500.0

    y = ffn(x)
    y_mod = ffn(x_mod)

    assert y.shape == (2, 8, 32)
    # Position 3 must be bit-for-bit identical since FFN is purely position-wise
    assert torch.allclose(y[:, 3, :], y_mod[:, 3, :], atol=1e-6)
    assert not torch.allclose(y[:, 0, :], y_mod[:, 0, :])


def test_transformer_block_preserves_shape_and_causality() -> None:
    """Stacking two TransformerBlocks must preserve (B, T, d_model) and strict causality."""
    torch.manual_seed(1)
    cfg = GPTConfig(vocab_size=65, block_size=16, d_model=32, n_heads=4, dropout=0.0)
    b1 = TransformerBlock(cfg)
    b2 = TransformerBlock(cfg)
    b1.eval()
    b2.eval()

    x = torch.randn(2, 10, 32, requires_grad=True)
    x_future_mod = x.detach().clone()
    t_cut = 5
    x_future_mod[:, t_cut + 1 :, :] += 1000.0

    out = b2(b1(x))
    out_mod = b2(b1(x_future_mod))

    assert out.shape == (2, 10, 32)
    # Forward causality across stacked blocks
    assert torch.allclose(out[:, : t_cut + 1, :], out_mod[:, : t_cut + 1, :], atol=1e-5)
    assert not torch.allclose(out[:, t_cut + 1 :, :], out_mod[:, t_cut + 1 :, :])

    # Backward causality across stacked blocks
    out[:, : t_cut + 1, :].sum().backward()
    assert torch.equal(
        x.grad[:, t_cut + 1 :, :],
        torch.zeros_like(x.grad[:, t_cut + 1 :, :]),
    )


def test_pre_layernorm_residual_highway_preserves_identity_gradient() -> None:
    """Even if both sub-layers output zero, the residual highway passes dL/dx = dL/dy 1:1."""
    torch.manual_seed(2)
    cfg = GPTConfig(vocab_size=32, block_size=8, d_model=16, n_heads=2, dropout=0.0, bias=False)
    block = TransformerBlock(cfg)
    block.eval()

    # Zero out the output projections of both sub-layers
    with torch.no_grad():
        block.attn.out_proj.weight.zero_()
        block.ffn.fc2.weight.zero_()

    x = torch.randn(2, 6, 16, requires_grad=True)
    y = block(x)
    assert torch.allclose(y, x, atol=1e-6)

    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    assert torch.allclose(x.grad, grad_out, atol=1e-6)
