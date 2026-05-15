"""
utils.py — Utility Belt for DA6401 Assignment 3
================================================
Re-exports the core building blocks from their canonical modules so the
W&B notebook can do a single import:

    from utils import LabelSmoothingLoss, NoamScheduler, make_src_mask, ...

Also provides helpers for:
  • gradient-norm tracking        (§2.2)
  • attention map extraction      (§2.3)
  • prediction-confidence logging (§2.5)
  • translation sample W&B table  (qualitative)
  • model summary / param count
"""

from __future__ import annotations

# ── re-exports (autograder imports these from the canonical modules) ──
from model      import make_src_mask, make_tgt_mask          # noqa: F401
from lr_scheduler import NoamScheduler, get_lr_history       # noqa: F401
from train      import LabelSmoothingLoss                    # noqa: F401

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════
#  MODEL SUMMARY
# ══════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> str:
    lines = ["Model parameter summary", "=" * 46]
    total = 0
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(False) if p.requires_grad)
        if params > 0:
            lines.append(f"  {name:<38s}  {params:>10,}")
            total += params
    lines += ["=" * 46, f"  {'TOTAL':<38s}  {total:>10,}"]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
#  GRADIENT-NORM TRACKER  (§2.2)
# ══════════════════════════════════════════════════════════════════════

class GradNormTracker:
    """
    Record per-step gradient L2 norms for selected parameter names.

    Usage:
        tracker = GradNormTracker(model, names=["W_q", "W_k"])
        loss.backward()
        tracker.record()
        tracker.plot("grad_norms.png")
    """

    def __init__(self, model: nn.Module, names: Optional[List[str]] = None) -> None:
        self.model   = model
        self.filter  = names
        self.history: dict[str, List[float]] = {}

    def record(self) -> dict[str, float]:
        """Call after loss.backward(), before optimizer.step()."""
        step: dict[str, float] = {}
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if self.filter and not any(f in name for f in self.filter):
                continue
            n = param.grad.data.norm(2).item()
            self.history.setdefault(name, []).append(n)
            step[name] = n
        return step

    def total_norm(self) -> float:
        total = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.model.parameters()
            if p.grad is not None
        )
        return math.sqrt(total)

    def plot(self, save_path: str = "grad_norms.png", max_params: int = 8) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, (name, vals) in enumerate(self.history.items()):
            if i >= max_params:
                break
            ax.plot(vals, label=f"{name.split('.')[-1]}[{i}]", alpha=0.8)
        ax.set_xlabel("Step"); ax.set_ylabel("Gradient L2 norm")
        ax.set_title("Gradient norms (Q & K weights)")
        ax.legend(fontsize=7, ncol=2)
        plt.tight_layout(); fig.savefig(save_path, dpi=150)
        return fig


# ══════════════════════════════════════════════════════════════════════
#  ATTENTION MAP EXTRACTION & PLOTTING  (§2.3)
# ══════════════════════════════════════════════════════════════════════

def extract_encoder_attention(
    model,
    src:       torch.Tensor,
    pad_idx:   int = 1,
    layer_idx: int = -1,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Run encoder and return attention weights from one layer.

    Returns:
        attn_w : [num_heads, S, S]
        tokens : list of decoded token strings
    """
    model.eval()
    with torch.no_grad():
        src_mask = make_src_mask(src, pad_idx)
        x = model.src_pos_enc(model.src_tok_embed(src))
        for layer in model.encoder.layers:
            x = layer(x, src_mask)
        attn_w = model.encoder.layers[layer_idx].self_attn.attn_weights

    if attn_w is None:
        raise RuntimeError("No attn_weights found — run encode first.")

    tokens = (
        [model.src_vocab.lookup_token(i) for i in src[0].tolist()]
        if model.src_vocab else [str(i) for i in src[0].tolist()]
    )
    return attn_w[0].cpu(), tokens


def plot_attention_heads(
    attn_w:    torch.Tensor,
    tokens:    List[str],
    title:     str = "Encoder Self-Attention",
    save_path: Optional[str] = None,
) -> plt.Figure:
    h    = attn_w.size(0)
    cols = min(h, 4)
    rows = math.ceil(h / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if h > 1 else [axes]
    S    = len(tokens)
    for i in range(h):
        ax = axes[i]
        im = ax.imshow(attn_w[i, :S, :S].numpy(), cmap="viridis",
                       aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"Head {i+1}", fontsize=9)
        ax.set_xticks(range(S)); ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticks(range(S)); ax.set_yticklabels(tokens, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(h, len(axes)):
        axes[i].set_visible(False)
    plt.suptitle(title, fontsize=12); plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def log_attention_maps_wandb(model, dataloader, src_vocab, device,
                              wandb_run, n_sentences=3, layer_idx=-1) -> None:
    import wandb
    pad_idx = getattr(src_vocab, "PAD", 1)
    count   = 0
    for src, _ in dataloader:
        if count >= n_sentences:
            break
        src = src[:1].to(device)
        try:
            attn_w, tokens = extract_encoder_attention(model, src, pad_idx, layer_idx)
        except RuntimeError:
            continue
        fig = plot_attention_heads(attn_w, tokens,
                                   title=f"Encoder attn — sentence {count+1}")
        wandb_run.log({f"attention/sentence_{count+1}": wandb.Image(fig)})
        plt.close(fig)
        count += 1


# ══════════════════════════════════════════════════════════════════════
#  PREDICTION CONFIDENCE  (§2.5)
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_prediction_confidence(
    model,
    dataloader,
    device:    str = "cpu",
    pad_idx:   int = 1,
    n_batches: int = 20,
) -> float:
    """
    Average softmax probability assigned to the correct token.
    Lower with label smoothing, higher with ε=0.
    """
    model.eval()
    confs: List[float] = []
    for i, (src, tgt) in enumerate(dataloader):
        if i >= n_batches:
            break
        src = src.to(device); tgt = tgt.to(device)
        tgt_inp = tgt[:, :-1]; tgt_out = tgt[:, 1:]
        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_inp, pad_idx)
        probs    = torch.softmax(model(src, tgt_inp, src_mask, tgt_mask), dim=-1)
        gold     = probs.gather(2, tgt_out.unsqueeze(-1)).squeeze(-1)
        mask     = tgt_out != pad_idx
        if mask.any():
            confs.append(gold[mask].mean().item())
    return float(sum(confs) / max(len(confs), 1))


# ══════════════════════════════════════════════════════════════════════
#  TRANSLATION SAMPLE TABLE  (W&B qualitative logging)
# ══════════════════════════════════════════════════════════════════════

def log_translation_samples(model, dataloader, src_vocab, tgt_vocab,
                             device, wandb_run, n_samples=10, max_len=60) -> None:
    import wandb
    from train import greedy_decode

    pad_idx = tgt_vocab.stoi.get("<pad>", 1) if hasattr(tgt_vocab, "stoi") else 1
    sos_idx = tgt_vocab.stoi.get("<sos>", 2) if hasattr(tgt_vocab, "stoi") else 2
    eos_idx = tgt_vocab.stoi.get("<eos>", 3) if hasattr(tgt_vocab, "stoi") else 3

    model.eval()
    table = wandb.Table(columns=["source", "reference", "hypothesis"])
    count = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            if count >= n_samples:
                break
            for i in range(src.size(0)):
                if count >= n_samples:
                    break
                src_i    = src[i:i+1].to(device)
                src_mask = make_src_mask(src_i, pad_idx)
                out      = greedy_decode(model, src_i, src_mask,
                                         max_len, sos_idx, eos_idx, device)

                def ids2str(ids, vocab, stop):
                    w = []
                    for idx in ids:
                        if idx == stop:
                            break
                        tok = vocab.lookup_token(idx)
                        if tok not in ("<sos>","<eos>","<pad>","<unk>"):
                            w.append(tok)
                    return " ".join(w)

                table.add_data(
                    ids2str(src[i].tolist()[1:],  src_vocab, eos_idx),
                    ids2str(tgt[i].tolist()[1:],  tgt_vocab, eos_idx),
                    ids2str(out[0].tolist()[1:],  tgt_vocab, eos_idx),
                )
                count += 1

    wandb_run.log({"translation_samples": table})


# ══════════════════════════════════════════════════════════════════════
#  LR SCHEDULE PLOT
# ══════════════════════════════════════════════════════════════════════

def plot_lr_schedule(d_model=512, warmup_steps=4000,
                     total_steps=20_000, save_path="lr_schedule.png") -> plt.Figure:
    lrs = get_lr_history(d_model, warmup_steps, total_steps)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(lrs, color="steelblue", linewidth=1.5)
    ax.axvline(warmup_steps, color="red", linestyle="--",
               label=f"warmup={warmup_steps}")
    ax.set_xlabel("Step"); ax.set_ylabel("LR (base=1)")
    ax.set_title(f"Noam LR  (d_model={d_model})"); ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ══════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from utils import (LabelSmoothingLoss, NoamScheduler, get_lr_history,
                       make_src_mask, make_tgt_mask, count_parameters, model_summary)
    import torch
    from model import Transformer

    print("✓ All re-exports OK")

    m = Transformer(50, 40, d_model=32, N=1, num_heads=4, d_ff=64, dropout=0.0)
    print(f"✓ count_parameters: {count_parameters(m):,}")
    assert "TOTAL" in model_summary(m)
    print("✓ model_summary OK")

    tracker = GradNormTracker(m, names=["W_q", "W_k"])
    src = torch.randint(2, 50, (2, 6))
    tgt = torch.randint(2, 40, (2, 4))
    m(src, tgt, make_src_mask(src, 1), make_tgt_mask(tgt, 1)).sum().backward()
    norms = tracker.record()
    assert len(norms) > 0
    print(f"✓ GradNormTracker: {len(norms)} norms recorded")

    print("\nAll utils self-tests passed.")
