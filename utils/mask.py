"""
mask.py — Mask Utilities for the Transformer

Provides two types of masks:
    1. **Padding mask**: prevents attention to <pad> tokens.
    2. **Causal (look-ahead) mask**: prevents the decoder from attending
       to future positions during training.

Both masks use the convention:
    1 = allowed position,  0 = masked position
"""

import torch
from torch import Tensor


def create_padding_mask(seq: Tensor, pad_idx: int) -> Tensor:
    """
    Create a padding mask that marks non-<pad> positions as 1.

    Args:
        seq:     Token indices of shape (batch, seq_len).
        pad_idx: Index of the padding token.

    Returns:
        Mask of shape (batch, 1, 1, seq_len), broadcastable over heads
        and query positions.
    """
    # (batch, seq_len) → (batch, 1, 1, seq_len)
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2).int()


def create_causal_mask(size: int, device: torch.device | None = None) -> Tensor:
    """
    Create a causal (lower-triangular) mask of shape (1, 1, size, size).

    Position (i, j) is 1 if j ≤ i (allowed), else 0 (masked).
    This prevents the decoder from looking ahead at future tokens.

    Args:
        size:   Sequence length of the target.
        device: Torch device.

    Returns:
        Causal mask of shape (1, 1, size, size).
    """
    mask = torch.tril(torch.ones(size, size, device=device)).unsqueeze(0).unsqueeze(0)
    return mask.int()


def create_masks(
    src: Tensor,
    tgt: Tensor,
    pad_idx: int,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Create both source and target masks in one call.

    Args:
        src:     Source token indices (batch, src_len).
        tgt:     Target token indices (batch, tgt_len).
        pad_idx: Padding token index.
        device:  Torch device.

    Returns:
        src_mask: (batch, 1, 1, src_len)
        tgt_mask: (batch, 1, tgt_len, tgt_len) — combines causal + padding
    """
    src_mask = create_padding_mask(src, pad_idx)

    # Target padding mask: (batch, 1, 1, tgt_len)
    tgt_padding_mask = create_padding_mask(tgt, pad_idx)

    # Causal mask: (1, 1, tgt_len, tgt_len)
    tgt_len = tgt.size(1)
    causal_mask = create_causal_mask(tgt_len, device=device)

    # Combine: both conditions must be satisfied (element-wise AND)
    tgt_mask = tgt_padding_mask & causal_mask

    return src_mask, tgt_mask
