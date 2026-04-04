"""
train.py — Training Script for the Transformer

Trains a Transformer model on the Multi30k German→English translation
dataset using:
    • Adam optimizer with Noam LR schedule
    • Cross-entropy loss with label smoothing
    • Gradient clipping

Usage:
    python train.py [--epochs 30] [--batch_size 128] [--d_model 512]
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TransformerConfig
from model.transformer import Transformer
from utils.mask import create_masks
from utils.lr_scheduler import NoamScheduler

# ── torchtext / spacy tokenization ─────────────────────────────────────
try:
    from torchtext.datasets import Multi30k
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
except ImportError:
    raise ImportError(
        "Please install torchtext: pip install torchtext"
    )


# ────────────────────────────────────────────────────────────────────────
#  Tokenizer & Vocabulary helpers
# ────────────────────────────────────────────────────────────────────────

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]


def build_tokenizers():
    """Return spaCy-based tokenizers for German and English."""
    src_tokenizer = get_tokenizer("spacy", language="de_core_news_sm")
    tgt_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    return src_tokenizer, tgt_tokenizer


def yield_tokens(data_iter, tokenizer, language_index):
    """Yield tokenized sentences from a dataset iterator."""
    for data_sample in data_iter:
        yield tokenizer(data_sample[language_index])


def build_vocab(dataset_iter, tokenizer, language_index, min_freq=2):
    """Build a Vocab object from a dataset iterator."""
    vocab = build_vocab_from_iterator(
        yield_tokens(dataset_iter, tokenizer, language_index),
        min_freq=min_freq,
        specials=SPECIAL_TOKENS,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


# ────────────────────────────────────────────────────────────────────────
#  Collation — convert raw text pairs to padded tensor batches
# ────────────────────────────────────────────────────────────────────────

def collate_fn(batch, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab, cfg):
    """
    Collate a list of (src_text, tgt_text) pairs into padded tensors.

    Each target sequence is split into:
        tgt_input:  <sos> tok1 tok2 ... tokN        (decoder input)
        tgt_output: tok1 tok2 ... tokN <eos>        (expected output / labels)
    """
    src_batch, tgt_input_batch, tgt_output_batch = [], [], []

    for src_text, tgt_text in batch:
        # Tokenize
        src_tokens = src_tokenizer(src_text)
        tgt_tokens = tgt_tokenizer(tgt_text)

        # Numericalize
        src_indices = [cfg.sos_idx] + src_vocab(src_tokens) + [cfg.eos_idx]
        tgt_indices = [cfg.sos_idx] + tgt_vocab(tgt_tokens) + [cfg.eos_idx]

        src_batch.append(torch.tensor(src_indices, dtype=torch.long))
        tgt_input_batch.append(torch.tensor(tgt_indices[:-1], dtype=torch.long))   # exclude <eos>
        tgt_output_batch.append(torch.tensor(tgt_indices[1:], dtype=torch.long))    # exclude <sos>

    # Pad to the same length within the batch
    src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=cfg.pad_idx)
    tgt_input_batch = nn.utils.rnn.pad_sequence(tgt_input_batch, batch_first=True, padding_value=cfg.pad_idx)
    tgt_output_batch = nn.utils.rnn.pad_sequence(tgt_output_batch, batch_first=True, padding_value=cfg.pad_idx)

    return src_batch, tgt_input_batch, tgt_output_batch


# ────────────────────────────────────────────────────────────────────────
#  Training loop
# ────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, cfg, device):
    """Run a single training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for src, tgt_input, tgt_output in dataloader:
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)

        # Build masks
        src_mask, tgt_mask = create_masks(src, tgt_input, cfg.pad_idx, device)

        # Forward pass
        logits = model(src, tgt_input, src_mask, tgt_mask)

        # Reshape for cross-entropy: (batch * tgt_len, vocab_size) vs (batch * tgt_len,)
        logits = logits.reshape(-1, logits.size(-1))
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(logits, tgt_output)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, cfg, device):
    """Evaluate model on a validation set. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for src, tgt_input, tgt_output in dataloader:
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)

        src_mask, tgt_mask = create_masks(src, tgt_input, cfg.pad_idx, device)

        logits = model(src, tgt_input, src_mask, tgt_mask)
        logits = logits.reshape(-1, logits.size(-1))
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(logits, tgt_output)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ────────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Transformer on Multi30k DE→EN")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimensionality")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of encoder/decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward inner dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--warmup_steps", type=int, default=4000, help="LR warmup steps")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda / cpu)")
    args = parser.parse_args()

    # ── Configuration ──────────────────────────────────────────────────
    cfg = TransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
    )
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # ── Tokenizers ─────────────────────────────────────────────────────
    print("Loading tokenizers ...")
    src_tokenizer, tgt_tokenizer = build_tokenizers()

    # ── Dataset & vocabularies ──────────────────────────────────────────
    print("Building vocabularies from Multi30k training data ...")

    train_iter = Multi30k(split="train", language_pair=("de", "en"))
    src_vocab = build_vocab(train_iter, src_tokenizer, 0)

    train_iter = Multi30k(split="train", language_pair=("de", "en"))
    tgt_vocab = build_vocab(train_iter, tgt_tokenizer, 1)

    print(f"  Source vocab size: {len(src_vocab)}")
    print(f"  Target vocab size: {len(tgt_vocab)}")

    # ── Data loaders ───────────────────────────────────────────────────
    def make_loader(split):
        dataset = list(Multi30k(split=split, language_pair=("de", "en")))
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            collate_fn=lambda b: collate_fn(b, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab, cfg),
        )

    train_loader = make_loader("train")
    val_loader = make_loader("valid")

    # ── Model ──────────────────────────────────────────────────────────
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Optimizer & Scheduler ──────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)

    # Cross-entropy with label smoothing, ignoring padding
    criterion = nn.CrossEntropyLoss(
        ignore_index=cfg.pad_idx, label_smoothing=cfg.label_smoothing
    )

    # ── Checkpointing directory ────────────────────────────────────────
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ── Training ───────────────────────────────────────────────────────
    best_val_loss = float("inf")
    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60)

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, cfg, device)
        val_loss = evaluate(model, val_loader, criterion, cfg, device)

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:3d}/{cfg.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"PPL: {math.exp(val_loss):.2f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "src_vocab": src_vocab,
                    "tgt_vocab": tgt_vocab,
                    "config": cfg,
                },
                os.path.join(cfg.checkpoint_dir, "best_model.pt"),
            )
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

        # Periodic checkpoint
        if epoch % cfg.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "src_vocab": src_vocab,
                    "tgt_vocab": tgt_vocab,
                    "config": cfg,
                },
                os.path.join(cfg.checkpoint_dir, f"checkpoint_epoch{epoch}.pt"),
            )

    print("\n" + "=" * 60)
    print(f"Training complete.  Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
