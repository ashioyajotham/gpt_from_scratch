# GPT From Scratch

A minimal, readable **GPT-2** implementation built from first principles using raw PyTorch tensor operations and `nn.Parameter` — without using `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.GELU`, `nn.Dropout`, `nn.MultiheadAttention`, or `F.softmax` in the model.

For a complete top-to-bottom explanation of every file, tensor shape, and design choice, see **[WALKTHROUGH.md](WALKTHROUGH.md)**.

## Architecture Overview

- **`src/gpt/ops.py`** — Hand-built primitives (`Linear`, `Embedding`, `LayerNorm`, `GELU`, `Dropout`, numerically stable `softmax`, `log_softmax`, `cross_entropy_loss`), verified forward and backward against PyTorch references.
- **`src/gpt/embeddings.py`** — Token + learned positional embeddings + dropout.
- **`src/gpt/attention.py`** — `SingleHeadCausalAttention` (educational reference) and fused `CausalMultiHeadAttention` with scaled dot-product attention and `-inf` causal look-ahead masking.
- **`src/gpt/block.py`** — Position-wise 4× `FeedForward` network and Pre-LayerNorm `TransformerBlock` with unaltered residual highways.
- **`src/gpt/model.py`** — Full `GPT` macro pipeline with weight tying ($W_e$), GPT-2 $0.02/\sqrt{2L}$ residual init scaling, and autoregressive `generate()` (greedy, temperature, top-$k$, top-$p$).
- **`src/gpt/train.py`** — Decoupled AdamW parameter groups (2D weights decayed, 1D biases/LayerNorms un-decayed), linear warmup + cosine LR decay, gradient clipping, and train/val loss estimation.
- **`src/gpt/tokenizer.py` & `src/gpt/data.py`** — Character-level tokenizer, contiguous 90/10 train/val split, and `+1`-shifted `(B, T)` next-token batching.

## Quickstart

```bash
# Install dependencies (Python 3.12 + uv)
uv sync --python 3.12

# Run all 35 unit & property tests (equivalence, causal leak, single-batch overfit)
uv run pytest -v

# Train on Tiny Shakespeare and sample text
uv run python scripts/train_shakespeare.py --steps 600

# Longer run (automatically uses Apple Silicon MPS / CUDA when available)
uv run python scripts/train_shakespeare.py --steps 2500 --d-model 128 --n-layers 4

# Compare greedy, temperature, top-k, and top-p decoding on the saved checkpoint
uv run python scripts/demo_m7_sampling.py
```

## Milestone Demos

Each milestone has a standalone inspection script in `scripts/`:
- `scripts/demo_m1_data.py` — Parallel next-token target alignment
- `scripts/demo_m2_ops.py` — LayerNorm stats, softmax numerical stability, GELU exact vs. tanh, initial $\ln(|V|)$ loss
- `scripts/demo_m3_embeddings.py` — Order sensitivity (`"dog bites man"` vs. `"man bites dog"`)
- `scripts/demo_m4_attention.py` — Lower-triangular causal weights, $1/\sqrt{d_k}$ variance scaling, contextualization
- `scripts/demo_m5_block.py` — Residual stream norm across depth, gradient highway, parameter share (2/3 FFN)
- `scripts/demo_m6_model.py` — Weight-tying parameter audit, init loss check, single-batch Shakespeare overfit
- `scripts/train_shakespeare.py` & `scripts/demo_m7_sampling.py` — Full training & decoding comparison
