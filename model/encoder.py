"""
encoder.py — Transformer Encoder

Implements the encoder side of the Transformer (Section 3.1):
    • EncoderLayer: one block of self-attention + FFN with residual connections & layer norm
    • Encoder:      N stacked EncoderLayers preceded by embedding + positional encoding
"""

import torch.nn as nn
from torch import Tensor

from model.attention import MultiHeadAttention
from model.feed_forward import PositionWiseFeedForward
from model.embedding import TokenEmbedding, PositionalEncoding


class EncoderLayer(nn.Module):
    """
    Single Transformer encoder layer.

    Processing pipeline:
        x → MultiHeadAttention(x, x, x) → Add & LayerNorm
          → FeedForward                  → Add & LayerNorm

    Args:
        d_model: Model dimensionality.
        n_heads: Number of attention heads.
        d_ff:    Feed-forward inner dimensionality.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        # Sub-layers
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        # Layer normalization (applied *after* the residual connection — Post-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout on each sub-layer output before residual addition
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            src:      Encoder input of shape (batch, src_len, d_model).
            src_mask: Padding mask of shape (batch, 1, 1, src_len).

        Returns:
            Encoder layer output of shape (batch, src_len, d_model).
        """
        # ── 1. Self-attention sub-layer ────────────────────────────────
        attn_output = self.self_attention(src, src, src, mask=src_mask)
        src = self.norm1(src + self.dropout1(attn_output))

        # ── 2. Feed-forward sub-layer ─────────────────────────────────
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(ff_output))

        return src


class Encoder(nn.Module):
    """
    Full Transformer encoder: embedding → positional encoding → N × EncoderLayer.

    Args:
        vocab_size:  Source vocabulary size.
        d_model:     Model dimensionality.
        n_heads:     Number of attention heads.
        n_layers:    Number of stacked encoder layers.
        d_ff:        Feed-forward inner dimensionality.
        max_seq_len: Maximum sequence length for positional encoding.
        dropout:     Dropout probability.
        pad_idx:     Padding token index (used to ignore padding in embeddings).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            src:      Source token indices of shape (batch, src_len).
            src_mask: Padding mask of shape (batch, 1, 1, src_len).

        Returns:
            Encoder output of shape (batch, src_len, d_model).
        """
        # Embed tokens + add positional encoding
        x = self.positional_encoding(self.token_embedding(src))

        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x, src_mask)

        return x
