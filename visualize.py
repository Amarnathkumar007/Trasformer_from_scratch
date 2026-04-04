"""
visualize.py — Attention Map Visualization

Generates heatmap plots of the multi-head attention weights from a
trained Transformer model.

Usage:
    python visualize.py --checkpoint checkpoints/best_model.pt \\
                        --text "Zwei Frauen sitzen auf einer Bank." \\
                        --layer 0 --attn_type encoder
"""

import argparse
import math

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from model.transformer import Transformer
from utils.mask import create_padding_mask, create_causal_mask

# ────────────────────────────────────────────────────────────────────────
#  Hook-based attention extraction
# ────────────────────────────────────────────────────────────────────────

def extract_attention_weights(model, src, tgt, src_mask, tgt_mask, device):
    """
    Perform a forward pass and collect attention weights from every layer.

    Returns:
        encoder_attns: list of Tensors (n_layers), each (1, n_heads, src_len, src_len)
        decoder_self_attns: list of Tensors, each (1, n_heads, tgt_len, tgt_len)
        decoder_cross_attns: list of Tensors, each (1, n_heads, tgt_len, src_len)
    """
    model.eval()
    with torch.no_grad():
        model(src, tgt, src_mask, tgt_mask)

    encoder_attns = [
        layer.self_attention.attention_weights.cpu()
        for layer in model.encoder.layers
    ]
    decoder_self_attns = [
        layer.masked_self_attention.attention_weights.cpu()
        for layer in model.decoder.layers
    ]
    decoder_cross_attns = [
        layer.cross_attention.attention_weights.cpu()
        for layer in model.decoder.layers
    ]

    return encoder_attns, decoder_self_attns, decoder_cross_attns


# ────────────────────────────────────────────────────────────────────────
#  Plotting
# ────────────────────────────────────────────────────────────────────────

def plot_attention_heads(
    attention: torch.Tensor,
    x_labels: list[str],
    y_labels: list[str],
    title: str = "Attention Weights",
    save_path: str | None = None,
):
    """
    Plot all attention heads for a single layer as a grid of heatmaps.

    Args:
        attention: (1, n_heads, seq_q, seq_k)
        x_labels:  Tokens along the key axis.
        y_labels:  Tokens along the query axis.
        title:     Figure title.
        save_path: If provided, save the figure to this path.
    """
    n_heads = attention.size(1)
    ncols = 4
    nrows = math.ceil(n_heads / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 4 * nrows),
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")

    if nrows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] for ax in axes]

    for head in range(n_heads):
        row, col = divmod(head, ncols)
        ax = axes[row][col] if isinstance(axes[row], (list, np.ndarray)) else axes[row]
        weights = attention[0, head].numpy()

        im = ax.imshow(weights, cmap="viridis", aspect="auto", vmin=0, vmax=weights.max())
        ax.set_title(f"Head {head + 1}", fontsize=11)

        # X-axis (keys)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)

        # Y-axis (queries)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for idx in range(n_heads, nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row][col] if isinstance(axes[row], (list, np.ndarray)) else axes[row]
        ax.set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved figure → {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_positional_encoding(d_model: int = 512, max_len: int = 100, save_path: str | None = None):
    """
    Visualize the sinusoidal positional encoding matrix.

    Args:
        d_model:   Model dimensionality.
        max_len:   Number of positions to show.
        save_path: If provided, save the figure.
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pe.numpy(), aspect="auto", cmap="RdBu_r")
    ax.set_title("Sinusoidal Positional Encoding", fontsize=14, fontweight="bold")
    ax.set_xlabel("Encoding Dimension")
    ax.set_ylabel("Position Index")
    fig.colorbar(im, ax=ax)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved figure → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize Transformer attention weights")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint .pt file")
    parser.add_argument("--text", type=str, required=True, help="Source (DE) sentence")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to visualize (0-indexed)")
    parser.add_argument(
        "--attn_type",
        type=str,
        default="encoder",
        choices=["encoder", "decoder_self", "decoder_cross"],
        help="Which attention to visualize",
    )
    parser.add_argument("--save", type=str, default=None, help="Save figure to this path")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda / cpu)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ── Load checkpoint ────────────────────────────────────────────────
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

    # ── Tokenize source ───────────────────────────────────────────────
    from torchtext.data.utils import get_tokenizer
    src_tokenizer = get_tokenizer("spacy", language="de_core_news_sm")
    tgt_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

    sos_idx = src_vocab["<sos>"]
    eos_idx = src_vocab["<eos>"]
    pad_idx = src_vocab["<pad>"]

    src_tokens = src_tokenizer(args.text.strip())
    src_indices = [sos_idx] + src_vocab(src_tokens) + [eos_idx]
    src_tensor = torch.tensor([src_indices], dtype=torch.long, device=device)
    src_mask = create_padding_mask(src_tensor, pad_idx).to(device)

    # For visualization, do a simple greedy decode to get target tokens
    from inference import greedy_decode
    tgt_indices_list = greedy_decode(model, src_tensor, src_mask, 50, sos_idx, eos_idx, device)
    tgt_indices = [sos_idx] + tgt_indices_list
    tgt_tensor = torch.tensor([tgt_indices], dtype=torch.long, device=device)

    tgt_len = tgt_tensor.size(1)
    tgt_mask = create_causal_mask(tgt_len, device=device)

    # ── Extract attention weights ─────────────────────────────────────
    enc_attns, dec_self_attns, dec_cross_attns = extract_attention_weights(
        model, src_tensor, tgt_tensor, src_mask, tgt_mask, device
    )

    # Build token labels
    src_itos = src_vocab.get_itos()
    tgt_itos = tgt_vocab.get_itos()
    src_labels = ["<sos>"] + src_tokens + ["<eos>"]
    tgt_labels = ["<sos>"] + [tgt_itos[i] for i in tgt_indices_list]

    # ── Plot ──────────────────────────────────────────────────────────
    layer = args.layer
    if args.attn_type == "encoder":
        attn = enc_attns[layer]
        plot_attention_heads(attn, src_labels, src_labels,
                             title=f"Encoder Self-Attention — Layer {layer + 1}",
                             save_path=args.save)
    elif args.attn_type == "decoder_self":
        attn = dec_self_attns[layer]
        plot_attention_heads(attn, tgt_labels, tgt_labels,
                             title=f"Decoder Self-Attention — Layer {layer + 1}",
                             save_path=args.save)
    elif args.attn_type == "decoder_cross":
        attn = dec_cross_attns[layer]
        plot_attention_heads(attn, src_labels, tgt_labels,
                             title=f"Decoder Cross-Attention — Layer {layer + 1}",
                             save_path=args.save)


if __name__ == "__main__":
    main()
