from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Hyperparameters defining our Pre-LayerNorm GPT architecture."""

    vocab_size: int
    block_size: int = 128  # Maximum context length T
    d_model: int = 128     # Residual stream / embedding dimension
    n_heads: int = 4       # Number of attention heads H (d_model % n_heads == 0)
    n_layers: int = 4      # Number of stacked Transformer blocks L
    dropout: float = 0.1   # Dropout probability p
    bias: bool = True      # Use bias terms in Linear and LayerNorm

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

    @property
    def d_head(self) -> int:
        """Per-head dimension d_k = d_model // H."""
        return self.d_model // self.n_heads
