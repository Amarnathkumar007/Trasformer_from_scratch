"""
model — Transformer architecture modules.

Submodules:
    embedding       Token + positional encoding
    attention       Scaled dot-product & multi-head attention
    feed_forward    Position-wise feed-forward network
    encoder         Encoder layer & full encoder stack
    decoder         Decoder layer & full decoder stack
    transformer     Complete Transformer (encoder + decoder)
"""

from model.transformer import Transformer

__all__ = ["Transformer"]
