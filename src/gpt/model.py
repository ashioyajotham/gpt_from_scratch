from __future__ import annotations

import math

import torch
import torch.nn as nn

from gpt.block import TransformerBlock
from gpt.config import GPTConfig
from gpt.embeddings import GPTEmbeddings
from gpt.ops import LayerNorm, Linear, cross_entropy_loss, softmax


class GPT(nn.Module):
    """Full Pre-LayerNorm GPT-2 Decoder-Only Architecture (Notes §1.1–1.7).

    Data flows bottom-to-top:
      1. Token + Learned Positional Embeddings + Dropout : (B, T) -> (B, T, d_model)
      2. Stack of L Pre-LayerNorm Transformer Blocks     : (B, T, d_model) -> (B, T, d_model)
      3. Final LayerNorm                                 : (B, T, d_model) -> (B, T, d_model)
      4. Linear Unembedding Head (weight-tied to W_e)    : (B, T, d_model) -> (B, T, |V|)
      5. Cross-Entropy Loss (when targets are provided)  : scalar
    """

    def __init__(self, config: GPTConfig, tie_weights: bool = True) -> None:
        super().__init__()
        self.config = config
        self.tie_weights = tie_weights

        # Bottom of macro pipeline (Notes §1.1–1.3)
        self.embeddings = GPTEmbeddings(config)

        # Stack of L Transformer Blocks (Notes §1.4)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final LayerNorm stabilizing accumulated residual stream (Notes §1.5)
        self.final_ln = LayerNorm(config.d_model, bias=config.bias)

        # Unembedding Linear Head projecting d_model -> |V| logits (Notes §1.6)
        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share W_e between Input Embedding and Unembedding Head
        if tie_weights:
            self.lm_head.weight = self.embeddings.token_embedding.weight

        # GPT-2 residual projection scaling: 0.02 / sqrt(2 * L)
        # Prevents residual stream variance from growing with depth L at initialization.
        self._apply_gpt2_residual_init()

    def _apply_gpt2_residual_init(self) -> None:
        resid_std = 0.02 / math.sqrt(2.0 * self.config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=resid_std)
            nn.init.normal_(block.ffn.fc2.weight, mean=0.0, std=resid_std)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the end-to-end GPT forward pass.

        Args:
            idx: Input token IDs of shape (B, T), where T <= config.block_size.
            targets: Optional next-token target IDs of shape (B, T).

        Returns:
            (logits, loss) where logits has shape (B, T, |V|) and loss is a scalar
            cross-entropy tensor if targets is given, else None.
        """
        # 1. Input + Positional Embeddings + Dropout -> (B, T, d_model)
        x = self.embeddings(idx)

        # 2. Pass sequentially through Layer 1 -> Layer L -> (B, T, d_model)
        for block in self.blocks:
            x = block(x)

        # 3. Final LayerNorm -> (B, T, d_model)
        x = self.final_ln(x)

        # 4. Unembedding Head -> raw unnormalized logits (B, T, |V|)
        logits = self.lm_head(x)

        # 5. Parallel next-token cross-entropy loss across all B * T positions
        loss = cross_entropy_loss(logits, targets) if targets is not None else None
        return logits, loss

    def count_parameters(self, exclude_pos_emb: bool = False) -> int:
        """Count unique trainable parameters (automatically respects weight tying)."""
        total = sum(p.numel() for p in self.parameters())
        if exclude_pos_emb:
            total -= self.embeddings.pos_embedding.weight.numel()
        return total

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample new tokens one at a time (Notes §1.7 & §3).

        At each step:
          1. Crop context to the most recent `block_size` tokens.
          2. Run forward pass and take logits at the final position: logits[:, -1, :].
          3. Apply optional temperature, top-k, and/or top-p (nucleus) filtering.
          4. Convert logits -> probabilities via softmax and sample the next token.
          5. Append the new token to the sequence and repeat.
        """
        if not greedy and temperature <= 0.0:
            raise ValueError(f"temperature must be > 0 when greedy=False, got {temperature}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if top_p is not None and not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")

        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                # 1. Crop sequence to fit within learned positional embedding table
                idx_cond = idx[:, -self.config.block_size :]

                # 2. Forward pass -> extract logits for the final position only: (B, |V|)
                logits, _ = self(idx_cond)
                next_logits = logits[:, -1, :]

                if greedy:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    # 3a. Temperature scaling (T < 1 sharper, T > 1 flatter)
                    scaled = next_logits / temperature

                    # 3b. Optional Top-k filtering
                    if top_k is not None:
                        k = min(top_k, scaled.size(-1))
                        kth_vals = torch.topk(scaled, k=k, dim=-1).values[:, -1:]
                        scaled = scaled.masked_fill(scaled < kth_vals, float("-inf"))

                    # 3c. Optional Top-p (nucleus) filtering
                    if top_p is not None and top_p < 1.0:
                        sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
                        sorted_probs = softmax(sorted_logits, dim=-1)
                        cum_probs = torch.cumsum(sorted_probs, dim=-1)
                        # Exclude tokens once cumulative mass BEFORE this token exceeds top_p
                        # (guarantees the top-1 token is always retained)
                        remove_sorted = (cum_probs - sorted_probs) >= top_p
                        sorted_logits = sorted_logits.masked_fill(remove_sorted, float("-inf"))
                        # Scatter filtered logits back to original vocabulary indices
                        scaled = torch.empty_like(scaled).scatter(dim=-1, index=sorted_idx, src=sorted_logits)

                    # 4. Softmax -> sample 1 token per batch item
                    probs = softmax(scaled, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1, generator=generator)

                # 5. Append sampled token to running sequence
                idx = torch.cat([idx, next_token], dim=1)
        finally:
            self.train(was_training)

        return idx
