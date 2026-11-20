from __future__ import annotations


class CharTokenizer:
    """Character-level tokenizer mapping unique characters <-> integer token IDs.

    This is our Milestone 1 drop-in tokenizer (Notes §1.1) before we build
    Byte Pair Encoding (BPE) from scratch in Milestone 8.
    """

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.itos: dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        """Size of vocabulary |V|."""
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        """Map raw string -> list of discrete integer token IDs."""
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:
            raise ValueError(f"Unknown character {exc.args[0]!r} not in vocabulary") from exc

    def decode(self, ids: list[int]) -> str:
        """Map list of integer token IDs -> reconstructed string."""
        try:
            return "".join(self.itos[i] for i in ids)
        except KeyError as exc:
            raise ValueError(f"Unknown token ID {exc.args[0]!r} not in vocabulary") from exc
