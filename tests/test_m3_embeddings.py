import pytest
import torch

from gpt.config import GPTConfig
from gpt.embeddings import GPTEmbeddings


def test_embeddings_output_shape_and_dtypes() -> None:
    cfg = GPTConfig(vocab_size=65, block_size=32, d_model=48, n_heads=4, dropout=0.1)
    emb = GPTEmbeddings(cfg)
    emb.eval()

    idx = torch.randint(0, cfg.vocab_size, (3, 12))
    out = emb(idx)
    assert out.shape == (3, 12, 48)
    assert out.dtype == torch.float32


def test_embeddings_reject_sequence_longer_than_block_size() -> None:
    cfg = GPTConfig(vocab_size=20, block_size=8, d_model=16, n_heads=2)
    emb = GPTEmbeddings(cfg)

    idx = torch.randint(0, 20, (1, 9))
    with pytest.raises(ValueError, match="exceeds maximum block_size"):
        emb(idx)


def test_positional_encoding_distinguishes_word_order() -> None:
    """Directly verifies Notes §1.2: 'dog bites man' vs 'man bites dog'."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=10, block_size=8, d_model=16, n_heads=2, dropout=0.0)
    emb = GPTEmbeddings(cfg)
    emb.eval()

    # Token IDs: dog=0, bites=1, man=2
    dog_bites_man = torch.tensor([[0, 1, 2]])
    man_bites_dog = torch.tensor([[2, 1, 0]])

    # Without positional encoding, the unordered bag (sum) of token vectors is identical
    tok_a = emb.token_embedding(dog_bites_man)
    tok_b = emb.token_embedding(man_bites_dog)
    assert torch.allclose(tok_a.sum(dim=1), tok_b.sum(dim=1), atol=1e-6)
    # Same token 'dog' at pos 0 vs pos 2 has identical raw token vector
    assert torch.equal(tok_a[0, 0], tok_b[0, 2])

    # With positional encoding added (+), 'dog' at pos 0 != 'dog' at pos 2
    out_a = emb(dog_bites_man)
    out_b = emb(man_bites_dog)
    assert not torch.allclose(out_a[0, 0], out_b[0, 2])
    # And the exact difference equals W_p[0] - W_p[2]
    expected_diff = emb.pos_embedding.weight[0] - emb.pos_embedding.weight[2]
    assert torch.allclose(out_a[0, 0] - out_b[0, 2], expected_diff, atol=1e-6)


def test_embeddings_dropout_active_only_in_train() -> None:
    torch.manual_seed(7)
    cfg = GPTConfig(vocab_size=20, block_size=16, d_model=32, n_heads=4, dropout=0.5)
    emb = GPTEmbeddings(cfg)
    idx = torch.randint(0, 20, (2, 10))

    emb.eval()
    e1 = emb(idx)
    e2 = emb(idx)
    assert torch.equal(e1, e2)

    emb.train()
    t1 = emb(idx)
    t2 = emb(idx)
    assert not torch.equal(t1, t2)
