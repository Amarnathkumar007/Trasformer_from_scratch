"""
decoder.py — Transformer Decoder

Implements the decoder side of the Transformer (Section 3.1):
    • DecoderLayer: masked self-attention → cross-attention → FFN
    • Decoder:      N stacked DecoderLayers preceded by embedding + positional encoding
"""

import torch.nn as nn
from torch import Tensor

from model.attention import MultiHeadAttention
from model.feed_forward import PositionWiseFeedForward
from model.embedding import TokenEmbedding, PositionalEncoding


class DecoderLayer(nn.Module):
    """
    Single Transformer decoder layer.

    Processing pipeline:
        x   → Masked Self-Attention(x, x, x)  → Add & LayerNorm
            → Cross-Attention(x, enc, enc)     → Add & LayerNorm
            → FeedForward                      → Add & LayerNorm

    The masked self-attention prevents the decoder from looking at future
    positions.  The cross-attention attends over the encoder output.

    Args:
        d_model: Model dimensionality.
        n_heads: Number of attention heads.
        d_ff:    Feed-forward inner dimensionality.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        # Sub-layers
        self.masked_self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        # Layer normalizations (Post-LN, as in the original paper)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout on each sub-layer output
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(
        self,
        tgt: Tensor,
        encoder_output: Tensor,
        tgt_mask: Tensor | None = None,
        src_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            tgt:            Decoder input of shape (batch, tgt_len, d_model).
            encoder_output: Encoder output of shape (batch, src_len, d_model).
            tgt_mask:       Combined causal + padding mask (batch, 1, tgt_len, tgt_len).
            src_mask:       Encoder padding mask       (batch, 1, 1, src_len).

        Returns:
            Decoder layer output of shape (batch, tgt_len, d_model).
        """
        # ── 1. Masked self-attention ──────────────────────────────────
        attn_output = self.masked_self_attention(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_output))

        # ── 2. Cross-attention (queries from decoder, keys/values from encoder)
        attn_output = self.cross_attention(tgt, encoder_output, encoder_output, mask=src_mask)
        tgt = self.norm2(tgt + self.dropout2(attn_output))

        # ── 3. Feed-forward ───────────────────────────────────────────
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout3(ff_output))

        return tgt


class Decoder(nn.Module):
    """
    Full Transformer decoder: embedding → positional encoding → N × DecoderLayer.

    Args:
        vocab_size:  Target vocabulary size.
        d_model:     Model dimensionality.
        n_heads:     Number of attention heads.
        n_layers:    Number of stacked decoder layers.
        d_ff:        Feed-forward inner dimensionality.
        max_seq_len: Maximum sequence length for positional encoding.
        dropout:     Dropout probability.
        pad_idx:     Padding token index.
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
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        tgt: Tensor,
        encoder_output: Tensor,
        tgt_mask: Tensor | None = None,
        src_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            tgt:            Target token indices of shape (batch, tgt_len).
            encoder_output: Encoder output of shape (batch, src_len, d_model).
            tgt_mask:       Combined causal + padding mask.
            src_mask:       Encoder padding mask.

        Returns:
            Decoder output of shape (batch, tgt_len, d_model).
        """
        # Embed tokens + add positional encoding
        x = self.positional_encoding(self.token_embedding(tgt))

        # Pass through each decoder layer
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)

        return x
