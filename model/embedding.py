"""
embedding.py — Token Embedding & Sinusoidal Positional Encoding

Implements Sections 3.4 and 3.5 of "Attention Is All You Need":
    • Token embeddings scaled by √d_model
    • Fixed sinusoidal positional encodings added to the embeddings
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    """
    Learnable token embedding layer.

    Converts token indices → dense vectors and scales them by √d_model
    so that the embedding magnitudes are comparable to the positional
    encodings that will be added afterwards.

    Args:
        vocab_size: Size of the vocabulary.
        d_model:    Dimensionality of the embedding vectors.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Token indices of shape (batch_size, seq_len).

        Returns:
            Scaled embeddings of shape (batch_size, seq_len, d_model).
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Section 3.5).

    Adds position-dependent signals to the token embeddings so that the
    model can attend to relative and absolute positions.  The encoding is
    computed once during __init__ and registered as a non-learnable buffer.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model:     Dimensionality of the model.
        max_seq_len: Maximum sequence length to pre-compute.
        dropout:     Dropout probability applied after adding the encoding.
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # ── Pre-compute the positional encoding matrix ──────────────────
        pe = torch.zeros(max_seq_len, d_model)                      # (max_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)   # even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # odd  indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model) — broadcastable over batch
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Embeddings of shape (batch_size, seq_len, d_model).

        Returns:
            Embeddings + positional encoding, with dropout applied.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
