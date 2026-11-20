import pytest
import torch

from gpt.data import get_batch, train_val_split
from gpt.tokenizer import CharTokenizer


def test_char_tokenizer_roundtrip() -> None:
    corpus = "First Citizen:\nBefore we proceed any further, hear me speak."
    tok = CharTokenizer(corpus)

    assert tok.vocab_size == len(set(corpus))
    ids = tok.encode(corpus)
    assert all(isinstance(i, int) and 0 <= i < tok.vocab_size for i in ids)
    assert tok.decode(ids) == corpus


def test_char_tokenizer_unknown_char_raises() -> None:
    tok = CharTokenizer("abc")
    with pytest.raises(ValueError, match="Unknown character"):
        tok.encode("abd")


def test_train_val_split_preserves_order_and_length() -> None:
    ids = list(range(100))
    train, val = train_val_split(ids, train_fraction=0.9)

    assert train.dtype == torch.long
    assert val.dtype == torch.long
    assert len(train) == 90
    assert len(val) == 10
    assert torch.equal(torch.cat([train, val]), torch.tensor(ids, dtype=torch.long))


def test_get_batch_shapes_and_shift_alignment() -> None:
    # Use a monotonic sequence 0, 1, 2, ... so target[b, t] MUST equal input[b, t] + 1
    data = torch.arange(200, dtype=torch.long)
    gen = torch.Generator().manual_seed(42)

    B, T = 4, 16
    x, y = get_batch(data, batch_size=B, block_size=T, generator=gen)

    assert x.shape == (B, T)
    assert y.shape == (B, T)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    # Exact +1 shift at every single position
    assert torch.equal(y, x + 1)
    # Consecutive tokens within each sequence are contiguous
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_get_batch_rejects_too_short_data() -> None:
    data = torch.arange(8, dtype=torch.long)
    with pytest.raises(ValueError, match="strictly greater"):
        get_batch(data, batch_size=2, block_size=8)
