"""
inference.py — Translation with Greedy & Beam Search Decoding

Load a trained Transformer checkpoint and translate German → English.

Usage:
    python inference.py --checkpoint checkpoints/best_model.pt \\
                        --text "Zwei Frauen sitzen auf einer Bank."

    python inference.py --checkpoint checkpoints/best_model.pt \\
                        --text "Ein Mann geht auf der Straße." \\
                        --beam_size 5
"""

import argparse

import torch
import torch.nn.functional as F
from torch import Tensor

from model.transformer import Transformer
from utils.mask import create_padding_mask, create_causal_mask


# ────────────────────────────────────────────────────────────────────────
#  Greedy Decoding
# ────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: Tensor,
    src_mask: Tensor,
    max_len: int,
    sos_idx: int,
    eos_idx: int,
    device: torch.device,
) -> list[int]:
    """
    Autoregressively decode using greedy selection (argmax).

    Args:
        model:    Trained Transformer.
        src:      Source token indices  (1, src_len).
        src_mask: Source padding mask   (1, 1, 1, src_len).
        max_len:  Maximum number of decode steps.
        sos_idx:  Start-of-sequence token index.
        eos_idx:  End-of-sequence token index.
        device:   Torch device.

    Returns:
        List of predicted token indices (excluding <sos>).
    """
    model.eval()

    # Encode the source once
    encoder_output = model.encode(src, src_mask)

    # Start with <sos>
    tgt_indices = [sos_idx]
    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_indices], dtype=torch.long, device=device)

        # Build causal mask for current target length
        tgt_len = tgt_tensor.size(1)
        tgt_mask = create_causal_mask(tgt_len, device=device)

        # Decode
        logits = model.decode(tgt_tensor, encoder_output, tgt_mask, src_mask)

        # Greedy pick: take the argmax of the last time-step
        next_token = logits[:, -1, :].argmax(dim=-1).item()
        tgt_indices.append(next_token)

        if next_token == eos_idx:
            break

    return tgt_indices[1:]  # exclude <sos>


# ────────────────────────────────────────────────────────────────────────
#  Beam Search Decoding
# ────────────────────────────────────────────────────────────────────────

def sequence_length_penalty(length: int, alpha: float = 0.6) -> float:
    """Length normalization penalty: ((5 + length) / 6) ^ alpha."""
    return ((5 + length) / 6) ** alpha


@torch.no_grad()
def beam_search_decode(
    model: Transformer,
    src: Tensor,
    src_mask: Tensor,
    max_len: int,
    sos_idx: int,
    eos_idx: int,
    device: torch.device,
    beam_size: int = 3,
    alpha: float = 0.6,
) -> list[int]:
    """
    Beam search decoding with length normalization.

    Args:
        model:     Trained Transformer.
        src:       Source token indices  (1, src_len).
        src_mask:  Source padding mask   (1, 1, 1, src_len).
        max_len:   Maximum number of decode steps.
        sos_idx:   Start-of-sequence token index.
        eos_idx:   End-of-sequence token index.
        device:    Torch device.
        beam_size: Number of beams to keep.
        alpha:     Length penalty exponent.

    Returns:
        List of predicted token indices (excluding <sos>) from the best beam.
    """
    model.eval()

    # Encode the source once
    encoder_output = model.encode(src, src_mask)

    # Initialize beams: (beam_size, seq_len)
    decoder_input = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
    scores = torch.tensor([0.0], device=device)

    vocab_size = model.output_projection.out_features

    for step in range(max_len):
        tgt_len = decoder_input.size(1)
        tgt_mask = create_causal_mask(tgt_len, device=device)

        logits = model.decode(decoder_input, encoder_output, tgt_mask, src_mask)
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

        # Length normalization
        log_probs = log_probs / sequence_length_penalty(step + 1, alpha)

        # Zero-out scores for beams that have already finished
        log_probs[decoder_input[:, -1] == eos_idx, :] = 0

        # Expand scores: (beam, 1) + (beam, vocab) → (beam, vocab)
        scores = scores.unsqueeze(1) + log_probs

        # Flatten and take top-k
        scores, indices = torch.topk(scores.reshape(-1), beam_size)
        beam_indices = torch.div(indices, vocab_size, rounding_mode="floor")
        token_indices = torch.remainder(indices, vocab_size)

        # Construct next decoder inputs
        next_inputs = []
        for beam_idx, token_idx in zip(beam_indices, token_indices):
            prev = decoder_input[beam_idx]
            if prev[-1] == eos_idx:
                token_idx = torch.tensor(eos_idx, device=device)
            next_inputs.append(torch.cat([prev, token_idx.unsqueeze(0)]))
        decoder_input = torch.stack(next_inputs)

        # All beams finished?
        if (decoder_input[:, -1] == eos_idx).sum() == beam_size:
            break

        # Expand encoder output on first step to match beam size
        if step == 0:
            encoder_output = encoder_output.expand(beam_size, -1, -1)
            src_mask = src_mask.expand(beam_size, -1, -1, -1)

    # Pick the best beam
    best_beam = decoder_input[scores.argmax()]
    result = best_beam[1:].tolist()  # remove <sos>
    if eos_idx in result:
        result = result[: result.index(eos_idx)]
    return result


# ────────────────────────────────────────────────────────────────────────
#  Translate a sentence
# ────────────────────────────────────────────────────────────────────────

def translate(
    text: str,
    model: Transformer,
    src_tokenizer,
    src_vocab,
    tgt_vocab,
    device: torch.device,
    max_len: int = 100,
    beam_size: int | None = None,
) -> str:
    """
    High-level translation function.

    Args:
        text:          Input German sentence.
        model:         Trained Transformer.
        src_tokenizer: Source language tokenizer.
        src_vocab:     Source Vocab object.
        tgt_vocab:     Target Vocab object.
        device:        Torch device.
        max_len:       Maximum output length.
        beam_size:     If specified, use beam search; otherwise greedy.

    Returns:
        Translated English string.
    """
    sos_idx = src_vocab["<sos>"]
    eos_idx = src_vocab["<eos>"]
    pad_idx = src_vocab["<pad>"]

    # Tokenize & numericalize source
    src_tokens = src_tokenizer(text.strip())
    src_indices = [sos_idx] + src_vocab(src_tokens) + [eos_idx]
    src_tensor = torch.tensor([src_indices], dtype=torch.long, device=device)

    src_mask = create_padding_mask(src_tensor, pad_idx).to(device)

    if beam_size and beam_size > 1:
        out_indices = beam_search_decode(
            model, src_tensor, src_mask, max_len, sos_idx, eos_idx, device, beam_size
        )
    else:
        out_indices = greedy_decode(
            model, src_tensor, src_mask, max_len, sos_idx, eos_idx, device
        )

    # Convert indices → words
    tgt_tokens = tgt_vocab.get_itos()
    output_words = [tgt_tokens[i] for i in out_indices if i not in (sos_idx, eos_idx, pad_idx)]
    return " ".join(output_words)


# ────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Translate DE→EN with a trained Transformer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--text", type=str, required=True, help="German sentence to translate")
    parser.add_argument("--beam_size", type=int, default=None, help="Beam size (omit for greedy)")
    parser.add_argument("--max_len", type=int, default=100, help="Max output length")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda / cpu)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ── Load checkpoint ────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    cfg = ckpt["config"]
    src_vocab = ckpt["src_vocab"]
    tgt_vocab = ckpt["tgt_vocab"]

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        pad_idx=cfg.pad_idx,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded model from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    # ── Tokenizer ──────────────────────────────────────────────────────
    from torchtext.data.utils import get_tokenizer
    src_tokenizer = get_tokenizer("spacy", language="de_core_news_sm")

    # ── Translate ──────────────────────────────────────────────────────
    mode = f"beam search (β={args.beam_size})" if args.beam_size else "greedy"
    print(f"\nDecoding mode: {mode}")
    print(f"  Source (DE): {args.text}")

    translation = translate(
        text=args.text,
        model=model,
        src_tokenizer=src_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        max_len=args.max_len,
        beam_size=args.beam_size,
    )
    print(f"  Target (EN): {translation}")


if __name__ == "__main__":
    main()
