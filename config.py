"""
config.py — Transformer Hyperparameters & Configuration

All default values match the "base model" from:
    "Attention Is All You Need" (Vaswani et al., 2017)
"""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for the Transformer model and training."""

    # ── Model Architecture ──────────────────────────────────────────────
    d_model: int = 512          # Dimensionality of embeddings & hidden states
    n_heads: int = 8            # Number of parallel attention heads
    n_layers: int = 6           # Number of encoder / decoder layers
    d_ff: int = 2048            # Inner dimensionality of the feed-forward network
    dropout: float = 0.1        # Dropout probability applied throughout
    max_seq_len: int = 512      # Maximum sequence length for positional encoding

    # ── Training ────────────────────────────────────────────────────────
    batch_size: int = 128       # Number of sentences per training batch
    num_epochs: int = 30        # Total training epochs
    learning_rate: float = 1e-4 # Peak learning rate (used by Noam scheduler)
    warmup_steps: int = 4000    # Linear warmup steps for Noam LR schedule
    label_smoothing: float = 0.1  # Label smoothing ε for cross-entropy loss
    clip_grad_norm: float = 1.0 # Max gradient norm for clipping

    # ── Data ────────────────────────────────────────────────────────────
    src_language: str = "de"    # Source language (German)
    tgt_language: str = "en"    # Target language (English)

    # ── Special Token Indices (set during vocab construction) ─────────
    pad_idx: int = 0
    sos_idx: int = 1
    eos_idx: int = 2
    unk_idx: int = 3

    # ── Checkpointing ──────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5         # Save a checkpoint every N epochs
