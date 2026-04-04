"""
attention.py — Scaled Dot-Product Attention & Multi-Head Attention

Implements Section 3.2 of "Attention Is All You Need":
    Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

Multi-head attention linearly projects Q, K, V into h sub-spaces,
applies attention in parallel, concatenates, and projects back.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Compute Scaled Dot-Product Attention.

    Args:
        query:   (batch, heads, seq_q, d_k)
        key:     (batch, heads, seq_k, d_k)
        value:   (batch, heads, seq_k, d_v)
        mask:    Optional broadcastable mask. Positions with True / 1 are
                 *allowed*; positions with False / 0 are masked (set to -inf).
        dropout: Optional dropout applied to attention weights.

    Returns:
        output:  Weighted values  (batch, heads, seq_q, d_v)
        weights: Attention weights (batch, heads, seq_q, seq_k)
    """
    d_k = query.size(-1)

    # ── QKᵀ / √d_k ─────────────────────────────────────────────────────
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # ── Apply mask (set masked positions to -inf before softmax) ────────
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # ── Softmax → attention weights ─────────────────────────────────────
    attention_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # ── Weighted sum of values ──────────────────────────────────────────
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Section 3.2.2).

    Instead of performing a single attention pass with d_model-dimensional
    keys, values, and queries, we linearly project them h times to d_k, d_k,
    and d_v dimensions respectively, perform attention in parallel, concatenate,
    and project once more.

    Args:
        d_model: Total model dimensionality.
        n_heads: Number of parallel attention heads.
        dropout: Dropout probability on attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads   # dimensionality per head

        # ── Linear projections for Q, K, V, and the output ─────────────
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.attention_weights: Tensor | None = None   # stored for visualization

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            query: (batch, seq_q, d_model)
            key:   (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask:  Optional mask of shape broadcastable to
                   (batch, 1, seq_q, seq_k).

        Returns:
            Output tensor of shape (batch, seq_q, d_model).
        """
        batch_size = query.size(0)

        # ── 1. Linear projections: (batch, seq, d_model) ──────────────
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # ── 2. Reshape → (batch, n_heads, seq, d_k) ───────────────────
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # ── 3. Scaled dot-product attention ────────────────────────────
        attn_output, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        self.attention_weights = attn_weights  # save for visualization

        # ── 4. Concatenate heads: (batch, seq_q, d_model) ─────────────
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        # ── 5. Final linear projection ────────────────────────────────
        return self.W_o(attn_output)
