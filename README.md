# DA6401 Assignment 3 — Attention Is All You Need

PyTorch implementation of the Transformer architecture from scratch for Neural Machine Translation (German → English) on the Multi30k dataset.

> **Paper:** [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Vaswani et al., NeurIPS 2017

---

## Project Structure

```
assignment3/
├── model.py          # Transformer architecture (MHA, PE, Encoder, Decoder)
├── lr_scheduler.py   # Noam learning rate scheduler
├── dataset.py        # Multi30k loading, spaCy tokenisation, vocabulary
├── train.py          # Training loop, greedy decoding, BLEU, checkpointing
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download spaCy language models

```bash
python -m spacy download de_core_news_sm   # German tokeniser
python -m spacy download en_core_web_sm    # English tokeniser
```

### 3. (Optional) Log in to W&B

```bash
wandb login
```

---

## Running

### Main training experiment

```bash
python train.py
```

This runs `run_training_experiment()` with default hyperparameters, logs everything to W&B, and saves checkpoints to `checkpoints/`.

### W&B ablation experiments

Each experiment from the report can be launched individually by editing the `__main__` block in `train.py`, or by importing and calling the launchers directly:

```python
from train import (
    run_noam_vs_fixed_lr,        # §2.1 — Noam vs fixed LR
    run_scale_ablation,          # §2.2 — with / without √dₖ scaling
    run_pe_ablation,             # §2.4 — sinusoidal vs learned PE
    run_label_smoothing_ablation # §2.5 — ε=0.1 vs ε=0.0
)

run_noam_vs_fixed_lr()
```

### Quick sanity check (no GPU, no dataset needed)

```bash
python -c "
import torch
from model import Transformer, make_src_mask, make_tgt_mask
m = Transformer(100, 80, d_model=64, N=2, num_heads=4, d_ff=128, dropout=0.0)
src = torch.randint(2, 100, (2, 7))
tgt = torch.randint(2, 80,  (2, 5))
out = m(src, tgt, make_src_mask(src, 1), make_tgt_mask(tgt, 1))
print('Output shape:', out.shape)   # → torch.Size([2, 5, 80])
"
```

```bash
python lr_scheduler.py    # plots the Noam LR curve and saves noam_lr_schedule.png
python dataset.py         # loads a batch and prints sample sentences
```

---

## Hyperparameters

| Parameter | Default | Paper (base) |
|---|---|---|
| `d_model` | 256 | 512 |
| `N` (layers) | 3 | 6 |
| `num_heads` | 8 | 8 |
| `d_ff` | 512 | 2048 |
| `dropout` | 0.1 | 0.1 |
| `warmup_steps` | 4000 | 4000 |
| `label_smoothing` | 0.1 | 0.1 |
| `batch_size` | 128 | — |
| `num_epochs` | 15 | — |
| Optimizer | Adam β₁=0.9 β₂=0.98 ε=1e-9 | same |

The defaults are scaled down from the paper to suit Multi30k's 29k training pairs.

---

## Architecture Details

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V
```

Implemented in `scaled_dot_product_attention()`. Accepts an optional boolean mask where `True` = masked out (set to −∞ before softmax). `nan_to_num` handles fully-masked rows.

### Multi-Head Attention

```
MultiHead(Q,K,V) = Concat(head₁,…,headₕ) · W_O
        headᵢ   = Attention(Q·W_Qᵢ, K·W_Kᵢ, V·W_Vᵢ)
```

Implemented from scratch with `nn.Linear` — **`torch.nn.MultiheadAttention` is not used**. Attention weights are stored in `self.attn_weights` after each forward pass for visualisation.

### Positional Encoding

```
PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )
```

Registered as a buffer (not a trainable parameter) so it is saved with the model but excluded from gradient updates.

### Layer Normalisation — Pre-LN

This implementation uses **Pre-LayerNorm** (`x = x + sublayer(norm(x))`) rather than the Post-LN of the original paper. Pre-LN places the normalisation *before* each sub-layer, which distributes gradients more evenly through the residual stream and avoids the early-training instability that Post-LN is known for on smaller datasets.

### Noam Learning Rate Schedule

```
lrate = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))
```

- Linearly increases LR during warm-up (`step ≤ warmup_steps`).
- Decays proportionally to `step^(-0.5)` afterwards.
- Peak occurs exactly at `step = warmup_steps`.

### Label Smoothing

```
y_smooth[c] = 1 − ε        if c == gold class
              ε / (V − 2)  otherwise  (excluding pad and gold)
              0             if c == pad_idx
```

`<pad>` positions contribute zero to both the smoothed distribution and the loss.

---

## Mask Convention

Both mask functions use **`True` = masked out** (position is excluded from attention):

| Function | Output shape | Meaning of `True` |
|---|---|---|
| `make_src_mask(src, pad_idx)` | `[B, 1, 1, S]` | `<pad>` token |
| `make_tgt_mask(tgt, pad_idx)` | `[B, 1, T, T]` | `<pad>` token OR future position |

---

## Autograder Contract

The following signatures are fixed and must not be changed:

```python
# model.py
scaled_dot_product_attention(Q, K, V, mask)  →  (output, attn_weights)
MultiHeadAttention.forward(query, key, value, mask)  →  Tensor
PositionalEncoding.forward(x)  →  Tensor
make_src_mask(src, pad_idx)  →  BoolTensor  [B, 1, 1, S]
make_tgt_mask(tgt, pad_idx)  →  BoolTensor  [B, 1, T, T]
Transformer.encode(src, src_mask)  →  Tensor  [B, S, d_model]
Transformer.decode(memory, src_mask, tgt, tgt_mask)  →  Tensor  [B, T, vocab]

# train.py
greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device)
    →  Tensor  [1, out_len]

evaluate_bleu(model, test_dataloader, tgt_vocab, device)
    →  float   (corpus BLEU, 0–100)

save_checkpoint(model, optimizer, scheduler, epoch, path)  →  None
load_checkpoint(path, model, optimizer, scheduler)  →  int  (epoch)
```

---

## Checkpoints

Checkpoints are saved every epoch to `checkpoints/epoch_XX.pt`. The best validation-loss checkpoint is always kept at `checkpoints/best_checkpoint.pt`.

Each checkpoint stores:

```python
{
    "epoch":                int,
    "model_state_dict":     ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "model_config": {
        "src_vocab_size": int,
        "tgt_vocab_size": int,
        "d_model": int,
        "N": int,
        "num_heads": int,
        "d_ff": int,
        "dropout": float,
        "use_scale": bool,
    }
}
```

To resume training or run inference from a saved checkpoint:

```python
from model import Transformer
from train import load_checkpoint

model = Transformer(**model_config)
epoch = load_checkpoint("checkpoints/best_checkpoint.pt", model)
```

---

## Inference

After attaching vocabularies, the model can translate raw German text:

```python
model.set_vocab(src_vocab, tgt_vocab, src_tokenizer=de_nlp)
print(model.infer("Ein Mann sitzt auf einer Bank."))
# → "a man is sitting on a bench ."
```

---

## W&B Report Experiments

| Section | What is compared | Launcher |
|---|---|---|
| §2.1 | Noam scheduler vs fixed LR 1e-4 | `run_noam_vs_fixed_lr()` |
| §2.2 | With vs without √dₖ scaling | `run_scale_ablation()` |
| §2.3 | Attention head visualisation | auto-logged after training |
| §2.4 | Sinusoidal PE vs learned `nn.Embedding` | `run_pe_ablation()` |
| §2.5 | Label smoothing ε=0.1 vs ε=0.0 | `run_label_smoothing_ablation()` |

Attention maps for all encoder heads of the last layer are automatically logged to W&B as images at the end of the main experiment.

---

## Dependencies

| Library | Purpose |
|---|---|
| `torch` | Model, training, tensors |
| `datasets` | HuggingFace Multi30k loader |
| `spacy` | German & English tokenisation |
| `sacrebleu` | Corpus-level BLEU evaluation |
| `wandb` | Experiment tracking & visualisation |
| `matplotlib` | Attention map plots |
| `tqdm` | (available for progress bars) |
| `numpy` | Numerical utilities |
