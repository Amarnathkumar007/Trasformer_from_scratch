"""
feed_forward.py — Position-wise Feed-Forward Network

Implements Section 3.3 of "Attention Is All You Need":
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

Each position is processed independently with the same two-layer MLP.
The inner layer expands the dimensionality to d_ff (typically 4× d_model),
applies ReLU, then projects back down to d_model.
"""

import torch.nn as nn
from torch import Tensor


class PositionWiseFeedForward(nn.Module):
    """
    Two-layer feed-forward network applied to each position independently.

    Args:
        d_model: Input and output dimensionality.
        d_ff:    Inner (hidden) layer dimensionality.
        dropout: Dropout probability between the two layers.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
