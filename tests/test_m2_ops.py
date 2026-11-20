import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt.ops import (
    GELU,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    cross_entropy_loss,
    log_softmax,
    softmax,
)


def test_linear_matches_pytorch_forward_and_backward() -> None:
    torch.manual_seed(0)
    ours = Linear(16, 32, bias=True)
    ref = nn.Linear(16, 32, bias=True)
    with torch.no_grad():
        ref.weight.copy_(ours.weight)
        ref.bias.copy_(ours.bias)

    x1 = torch.randn(2, 5, 16, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    y1 = ours(x1)
    y2 = ref(x2)
    assert torch.allclose(y1, y2, atol=1e-6)

    grad_out = torch.randn_like(y1)
    y1.backward(grad_out)
    y2.backward(grad_out)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-6)
    assert torch.allclose(ours.weight.grad, ref.weight.grad, atol=1e-6)
    assert torch.allclose(ours.bias.grad, ref.bias.grad, atol=1e-6)


def test_embedding_matches_pytorch_forward_and_backward() -> None:
    torch.manual_seed(1)
    ours = Embedding(65, 24)
    ref = nn.Embedding(65, 24)
    with torch.no_grad():
        ref.weight.copy_(ours.weight)

    idx = torch.randint(0, 65, (3, 10))
    y1 = ours(idx)
    y2 = ref(idx)
    assert y1.shape == (3, 10, 24)
    assert torch.allclose(y1, y2, atol=1e-7)

    grad_out = torch.randn_like(y1)
    y1.backward(grad_out)
    y2.backward(grad_out)
    assert torch.allclose(ours.weight.grad, ref.weight.grad, atol=1e-6)


def test_layernorm_matches_pytorch_forward_and_backward() -> None:
    torch.manual_seed(2)
    ours = LayerNorm(32, eps=1e-5, bias=True)
    ref = nn.LayerNorm(32, eps=1e-5, elementwise_affine=True)
    with torch.no_grad():
        ours.weight.copy_(torch.randn(32))
        ours.bias.copy_(torch.randn(32))
        ref.weight.copy_(ours.weight)
        ref.bias.copy_(ours.bias)

    x1 = (torch.randn(4, 7, 32) * 10.0 + 5.0).requires_grad_(True)
    x2 = x1.detach().clone().requires_grad_(True)

    y1 = ours(x1)
    y2 = ref(x2)
    assert torch.allclose(y1, y2, atol=1e-5)

    grad_out = torch.randn_like(y1)
    y1.backward(grad_out)
    y2.backward(grad_out)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert torch.allclose(ours.weight.grad, ref.weight.grad, atol=1e-5)
    assert torch.allclose(ours.bias.grad, ref.bias.grad, atol=1e-5)


@pytest.mark.parametrize("approx", ["none", "tanh"])
def test_gelu_matches_pytorch_forward_and_backward(approx: str) -> None:
    torch.manual_seed(3)
    ours = GELU(approximate=approx)
    ref = nn.GELU(approximate=approx)

    x1 = torch.linspace(-5.0, 5.0, 200, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    y1 = ours(x1)
    y2 = ref(x2)
    assert torch.allclose(y1, y2, atol=1e-6)

    y1.sum().backward()
    y2.sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)


def test_dropout_preserves_expectation_in_train_and_is_identity_in_eval() -> None:
    torch.manual_seed(4)
    p = 0.25
    drop = Dropout(p=p)
    x = torch.ones(100_000)

    drop.train()
    y_train = drop(x)
    zero_frac = (y_train == 0).float().mean().item()
    # ~25% dropped, survivors scaled by 1 / 0.75 = 1.3333 -> mean stays ~1.0
    assert abs(zero_frac - p) < 0.01
    assert abs(y_train.mean().item() - 1.0) < 0.02

    drop.eval()
    y_eval = drop(x)
    assert torch.equal(y_eval, x)


def test_softmax_and_log_softmax_numerical_stability() -> None:
    # Extreme logits would overflow naive exp(1000) -> inf/nan
    logits = torch.tensor([[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -1001.0]])

    probs = softmax(logits, dim=-1)
    ref_probs = F.softmax(logits, dim=-1)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(probs, ref_probs, atol=1e-6)

    lprobs = log_softmax(logits, dim=-1)
    ref_lprobs = F.log_softmax(logits, dim=-1)
    assert torch.isfinite(lprobs).all()
    assert torch.allclose(lprobs, ref_lprobs, atol=1e-6)


def test_cross_entropy_loss_matches_pytorch_3d_and_2d() -> None:
    torch.manual_seed(5)
    B, T, V = 4, 12, 65
    logits1 = torch.randn(B, T, V, requires_grad=True)
    logits2 = logits1.detach().clone().requires_grad_(True)
    targets = torch.randint(0, V, (B, T))

    loss1 = cross_entropy_loss(logits1, targets)
    loss2 = F.cross_entropy(logits2.reshape(B * T, V), targets.reshape(B * T))
    assert torch.allclose(loss1, loss2, atol=1e-6)

    loss1.backward()
    loss2.backward()
    assert torch.allclose(logits1.grad, logits2.grad, atol=1e-6)
