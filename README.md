# 🤖 Transformer from Scratch

A clean, well-documented PyTorch implementation of the **Transformer** architecture from the landmark paper:

> **"Attention Is All You Need"**  
> Vaswani et al., NeurIPS 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

This project implements every component from scratch — no `nn.Transformer` shortcuts — so you can read the code alongside the paper and understand exactly how each piece works.

---

## 📁 Project Structure

```
transformer_from_scratch/
│
├── model/                      # Core Transformer modules
│   ├── __init__.py
│   ├── embedding.py            # Token embedding + sinusoidal positional encoding
│   ├── attention.py            # Scaled dot-product & multi-head attention
│   ├── feed_forward.py         # Position-wise feed-forward network
│   ├── encoder.py              # Encoder layer + full encoder stack
│   ├── decoder.py              # Decoder layer + full decoder stack
│   └── transformer.py          # Complete Transformer model
│
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── mask.py                 # Padding & causal mask creation
│   └── lr_scheduler.py         # Noam learning rate scheduler
│
├── train.py                    # Training script (Multi30k DE→EN)
├── inference.py                # Greedy & beam search translation
├── visualize.py                # Attention heatmap visualization
├── config.py                   # Hyperparameters & configuration
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🏗️ Architecture Overview

The Transformer is a sequence-to-sequence model built entirely on **attention mechanisms**, eliminating the need for recurrence or convolutions. It consists of an **Encoder** and a **Decoder**, each composed of stacked identical layers.

```
Input (source)                              Input (target, shifted right)
      │                                              │
      ▼                                              ▼
┌───────────────┐                              ┌───────────────┐
│   Token       │                              │   Token       │
│   Embedding   │                              │   Embedding   │
│   + Positional│                              │   + Positional│
│   Encoding    │                              │   Encoding    │
└──────┬────────┘                              └──────┬────────┘
       │                                              │
       ▼                                              ▼
┌──────────────────┐                   ┌───────────────────────────┐
│                  │                   │  Masked                   │
│  Multi-Head      │                   │  Multi-Head               │
│  Self-Attention  │                   │  Self-Attention           │
│                  │                   │                           │
├──────────────────┤                   ├───────────────────────────┤
│  Add & LayerNorm │                   │  Add & LayerNorm          │
├──────────────────┤                   ├───────────────────────────┤
│                  │          ┌───────►│  Multi-Head               │
│  Feed-Forward    │          │        │  Cross-Attention          │
│  Network         │          │        │  (Q from decoder,         │
│                  │          │        │   K,V from encoder)       │
├──────────────────┤          │        ├───────────────────────────┤
│  Add & LayerNorm │          │        │  Add & LayerNorm          │
└──────┬───────────┘          │        ├───────────────────────────┤
       │                      │        │  Feed-Forward Network     │
       │    ×N layers         │        ├───────────────────────────┤
       │                      │        │  Add & LayerNorm          │
       ▼                      │        └──────────┬────────────────┘
  Encoder Output ─────────────┘                   │     ×N layers
                                                  ▼
                                           ┌───────────────┐
                                           │    Linear     │
                                           │   + Softmax   │
                                           └──────┬────────┘
                                                  ▼
                                           Output Probabilities
```

### Key Components

| Component | Paper Section | File | Description |
|:---|:---:|:---|:---|
| Token Embedding | 3.4 | `model/embedding.py` | Learnable embeddings scaled by √d_model |
| Positional Encoding | 3.5 | `model/embedding.py` | Sinusoidal position signals (sin/cos) |
| Scaled Dot-Product Attention | 3.2.1 | `model/attention.py` | `softmax(QKᵀ/√d_k) V` |
| Multi-Head Attention | 3.2.2 | `model/attention.py` | h parallel attention heads, concatenated |
| Position-wise FFN | 3.3 | `model/feed_forward.py` | Two linear layers with ReLU: `max(0, xW₁+b₁)W₂+b₂` |
| Encoder Layer | 3.1 | `model/encoder.py` | Self-attention → Add&Norm → FFN → Add&Norm |
| Decoder Layer | 3.1 | `model/decoder.py` | Masked self-attn → Cross-attn → FFN (each with Add&Norm) |
| Noam LR Schedule | 5.3 | `utils/lr_scheduler.py` | Warmup + inverse-sqrt decay |

### Default Hyperparameters (Base Model)

| Parameter | Value |
|:---|:---:|
| d_model | 512 |
| n_heads | 8 |
| n_layers | 6 |
| d_ff | 2048 |
| dropout | 0.1 |
| warmup_steps | 4000 |
| label_smoothing | 0.1 |

---

## 🔍 Component Deep Dive

### Scaled Dot-Product Attention

The core attention mechanism computes a weighted sum of values, where the weight for each value is determined by the compatibility of the corresponding key with the query:

```
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V
```

The scaling factor `√d_k` prevents the dot products from growing too large, which would push softmax into regions with vanishingly small gradients.

### Multi-Head Attention

Instead of computing a single attention function, the model projects Q, K, V into `h` different subspaces, runs attention in parallel, and concatenates:

```
MultiHead(Q,K,V) = Concat(head₁, ..., headₕ) Wᴼ
where headᵢ = Attention(Q Wᵢᵠ, K Wᵢᴷ, V Wᵢⱽ)
```

This allows the model to jointly attend to information from different representation subspaces at different positions.

### Positional Encoding

Since the Transformer has no recurrence, it uses sinusoidal functions to inject position information:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Each dimension of the encoding corresponds to a sinusoid with a different wavelength, forming a geometric progression from 2π to 10000·2π.

### Noam Learning Rate Schedule

The learning rate warms up linearly for `warmup_steps`, then decays proportionally to the inverse square root of the step number:

```
lr = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Installation

```bash
# Clone the repository

cd transformer_from_scratch

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy language models (required for tokenization)
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

### Training

Train the Transformer on the **Multi30k** German→English translation dataset:

```bash
# Train with default settings (base model)
python train.py

# Customize training
python train.py --epochs 50 --batch_size 64 --d_model 256 --n_layers 3

# All available options
python train.py --help
```

**Training arguments:**

| Argument | Default | Description |
|:---|:---:|:---|
| `--epochs` | 30 | Number of training epochs |
| `--batch_size` | 128 | Training batch size |
| `--d_model` | 512 | Model dimensionality |
| `--n_heads` | 8 | Number of attention heads |
| `--n_layers` | 6 | Number of encoder/decoder layers |
| `--d_ff` | 2048 | Feed-forward inner dimension |
| `--dropout` | 0.1 | Dropout probability |
| `--warmup_steps` | 4000 | LR warmup steps |
| `--device` | auto | `cuda` or `cpu` |

The script will:
- Download Multi30k automatically on first run
- Print training/validation loss and perplexity each epoch
- Save the best model to `checkpoints/best_model.pt`
- Save periodic checkpoints to `checkpoints/checkpoint_epochN.pt`

### Inference (Translating)

Translate German sentences to English using a trained model:

```bash
# Greedy decoding
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --text "Zwei Frauen sitzen auf einer Bank."

# Beam search decoding (beam_size=5)
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --text "Ein Mann geht auf der Straße." \
    --beam_size 5
```

### Visualizing Attention

Visualize what the model is "paying attention to":

```bash
# Encoder self-attention (layer 1)
python visualize.py \
    --checkpoint checkpoints/best_model.pt \
    --text "Zwei Frauen in pinkfarbenen T-Shirts." \
    --layer 0 --attn_type encoder \
    --save encoder_attention.png

# Decoder cross-attention (layer 3)
python visualize.py \
    --checkpoint checkpoints/best_model.pt \
    --text "Ein Hund läuft durch den Park." \
    --layer 2 --attn_type decoder_cross \
    --save cross_attention.png

# Decoder self-attention (layer 6)
python visualize.py \
    --checkpoint checkpoints/best_model.pt \
    --text "Ein Hund läuft durch den Park." \
    --layer 5 --attn_type decoder_self
```

**Attention types:**
- `encoder` — Encoder self-attention (how source tokens attend to each other)
- `decoder_self` — Decoder self-attention (how target tokens attend to previous target tokens, masked)
- `decoder_cross` — Decoder cross-attention (how target tokens attend to source tokens)

---

## 📖 Reading the Code

The codebase is designed to be read alongside the paper. Each module maps directly to a section:

1. **Start with** [`model/attention.py`](model/attention.py) — the heart of the Transformer. Read `scaled_dot_product_attention()` first, then `MultiHeadAttention`.

2. **Then read** [`model/embedding.py`](model/embedding.py) — token embeddings and positional encoding (Sections 3.4–3.5).

3. **Next** [`model/feed_forward.py`](model/feed_forward.py) — the simple two-layer FFN (Section 3.3).

4. **Assemble layers** in [`model/encoder.py`](model/encoder.py) and [`model/decoder.py`](model/decoder.py) — see how attention, FFN, residual connections, and layer normalization come together.

5. **Full model** in [`model/transformer.py`](model/transformer.py) — the encoder-decoder composition with weight initialization.

6. **Utilities** in [`utils/mask.py`](utils/mask.py) (padding & causal masks) and [`utils/lr_scheduler.py`](utils/lr_scheduler.py) (Noam schedule).

---

## 🔬 Technical Notes

### Masking

Two types of masks are used:

- **Padding mask**: Prevents the model from attending to `<pad>` tokens in variable-length batches.
- **Causal mask**: A lower-triangular matrix that prevents the decoder from looking at future positions during autoregressive generation.

Both use the convention: `1 = attend`, `0 = mask out`.

### Weight Initialization

All linear layers and embedding weights are initialized with **Xavier uniform** initialization, which helps maintain stable gradients at the beginning of training.

### Label Smoothing

Following the paper (Section 5.4), we use label smoothing with ε = 0.1 during training. This hurts perplexity slightly but improves accuracy and BLEU score by making the model less confident.

---

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) — Harvard NLP
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar
- [PyTorch nn.Transformer source](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py) — Official PyTorch

---

