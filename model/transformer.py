"""
transformer.py — Complete Transformer Model

Ties the encoder, decoder, and output projection together into the
full sequence-to-sequence Transformer described in:
    "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch.nn as nn
from torch import Tensor

from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    """
    Full Transformer: Encoder → Decoder → Linear projection.

    Args:
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        d_model:        Model dimensionality.
        n_heads:        Number of attention heads.
        n_layers:       Number of encoder & decoder layers.
        d_ff:           Feed-forward inner dimensionality.
        max_seq_len:    Maximum sequence length for positional encoding.
        dropout:        Dropout probability.
        pad_idx:        Padding token index.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_idx=pad_idx,
        )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_idx=pad_idx,
        )

        # Final linear layer projects decoder output to target vocabulary logits
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Initialize weights using Xavier uniform (as recommended)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for all linear layers and embeddings."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Public API ──────────────────────────────────────────────────────

    def encode(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """
        Run the encoder.

        Args:
            src:      Source token indices (batch, src_len).
            src_mask: Padding mask       (batch, 1, 1, src_len).

        Returns:
            Encoder output (batch, src_len, d_model).
        """
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: Tensor,
        encoder_output: Tensor,
        tgt_mask: Tensor | None = None,
        src_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Run the decoder and project to vocabulary logits.

        Args:
            tgt:            Target token indices (batch, tgt_len).
            encoder_output: Encoder output      (batch, src_len, d_model).
            tgt_mask:       Causal + padding mask.
            src_mask:       Encoder padding mask.

        Returns:
            Logits over target vocabulary (batch, tgt_len, tgt_vocab_size).
        """
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        return self.output_projection(decoder_output)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Full forward pass: encode source → decode target → logits.

        Args:
            src:      Source token indices (batch, src_len).
            tgt:      Target token indices (batch, tgt_len).
            src_mask: Source padding mask  (batch, 1, 1, src_len).
            tgt_mask: Target causal + padding mask (batch, 1, tgt_len, tgt_len).

        Returns:
            Logits over target vocabulary (batch, tgt_len, tgt_vocab_size).
        """
        encoder_output = self.encode(src, src_mask)
        logits = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        return logits
