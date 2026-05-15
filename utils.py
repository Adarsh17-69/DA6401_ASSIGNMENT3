"""
utils.py — Utility Belt for DA6401 Assignment 3
================================================
Re-exports the core building blocks from their canonical modules so that
the rest of the codebase (and the W&B report notebooks) can do a single
convenient import:

    from utils import (
        LabelSmoothingLoss, NoamScheduler,
        make_src_mask, make_tgt_mask,
        get_lr_history, ...
    )

Also provides standalone helpers for:
  • gradient-norm tracking          (§2.2 ablation)
  • attention map extraction        (§2.3 head visualisation)
  • prediction-confidence logging   (§2.5 label-smoothing ablation)
  • translation sample logging      (qualitative W&B table)
  • parameter counting & model summary
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════
#  RE-EXPORTS  — canonical implementations live in their own modules
#               so autograder import paths are never broken.
# ══════════════════════════════════════════════════════════════════════

# Masking utilities (autograder imports these from model.py)
from model import make_src_mask, make_tgt_mask          # noqa: F401

# Noam scheduler (autograder imports from lr_scheduler.py)
from lr_scheduler import NoamScheduler, get_lr_history  # noqa: F401

# Label smoothing loss (autograder imports from train.py)
from train import LabelSmoothingLoss                    # noqa: F401


# ══════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════

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
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> str:
    """
    Return a concise string summary: per-module parameter counts
    and the grand total.
    """
    lines = ["Model parameter summary", "=" * 42]
    total = 0
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(False)
                     if p.requires_grad)
        if params > 0:
            lines.append(f"  {name:<35s}  {params:>10,}")
            total += params
    lines += ["=" * 42, f"  {'TOTAL':<35s}  {total:>10,}"]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
#  GRADIENT-NORM TRACKER  (§2.2 — ablation: with / without √dₖ)
# ══════════════════════════════════════════════════════════════════════

class GradNormTracker:
    """
    Accumulate per-step gradient norms for selected parameter groups.

    Usage:
        tracker = GradNormTracker(model, names=["W_q", "W_k"])
        # after loss.backward():
        tracker.record()
        # at end of run:
        tracker.plot("grad_norms.png")
    """

    def __init__(
        self,
        model: nn.Module,
        names: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            model : Transformer (or any nn.Module).
            names : Substring filter — only track params whose name
                    contains at least one of these strings.
                    Pass None to track all parameters.
        """
        self.model  = model
        self.filter = names
        self.history: dict[str, List[float]] = {}

    def record(self) -> dict[str, float]:
        """
        Call after `loss.backward()` but before `optimizer.step()`.
        Returns the norms recorded this step.
        """
        step_norms: dict[str, float] = {}
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if self.filter and not any(f in name for f in self.filter):
                continue
            norm = param.grad.data.norm(2).item()
            self.history.setdefault(name, []).append(norm)
            step_norms[name] = norm
        return step_norms

    def total_norm(self) -> float:
        """L2 norm across all tracked gradients at the latest step."""
        total = 0.0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                total += param.grad.data.norm(2).item() ** 2
        return math.sqrt(total)

    def plot(
        self,
        save_path: str = "grad_norms.png",
        max_params: int = 8,
    ) -> plt.Figure:
        """
        Plot gradient-norm curves for the first `max_params` tracked tensors.
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, (name, values) in enumerate(self.history.items()):
            if i >= max_params:
                break
            ax.plot(values, label=name.split(".")[-1] + f" [{i}]", alpha=0.8)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Gradient L2 norm")
        ax.set_title("Gradient norms (Query & Key weights)")
        ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        return fig


# ══════════════════════════════════════════════════════════════════════
#  ATTENTION MAP EXTRACTION & PLOTTING  (§2.3)
# ══════════════════════════════════════════════════════════════════════

def extract_encoder_attention(
    model,
    src: torch.Tensor,
    pad_idx: int = 1,
    layer_idx: int = -1,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Run encoder on a single sentence and return the attention weights
    from a chosen layer.

    Args:
        model     : Trained Transformer (in eval mode).
        src       : [1, S] token-index tensor.
        pad_idx   : <pad> index for masking.
        layer_idx : Which encoder layer to extract from (-1 = last).

    Returns:
        attn_w : [num_heads, S, S]  — attention weight matrix.
        tokens : list[str]          — decoded token strings.
    """
    model.eval()
    device = src.device

    with torch.no_grad():
        src_mask = make_src_mask(src, pad_idx)
        x = model.src_pos_enc(model.src_tok_embed(src))
        for layer in model.encoder.layers:
            x = layer(x, src_mask)

        target_layer = model.encoder.layers[layer_idx]
        attn_w = target_layer.self_attn.attn_weights  # [1, h, S, S]

    if attn_w is None:
        raise RuntimeError("No attention weights found — did you run encode first?")

    # Decode token strings (skip <sos>/<eos>/<pad> in labels only)
    tokens: List[str] = []
    if model.src_vocab is not None:
        for idx in src[0].tolist():
            tokens.append(model.src_vocab.lookup_token(idx))
    else:
        tokens = [str(i) for i in src[0].tolist()]

    return attn_w[0].cpu(), tokens           # [h, S, S], list[str]


def plot_attention_heads(
    attn_w: torch.Tensor,
    tokens: List[str],
    title: str = "Encoder Self-Attention",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot one heat map per attention head in a grid.

    Args:
        attn_w    : [num_heads, S, S]
        tokens    : List of token strings (length S).
        title     : Figure super-title.
        save_path : If given, save the figure to this path.

    Returns:
        matplotlib Figure.
    """
    h = attn_w.size(0)
    cols = min(h, 4)
    rows = math.ceil(h / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if h > 1 else [axes]

    S = len(tokens)
    for head_idx in range(h):
        ax  = axes[head_idx]
        mat = attn_w[head_idx, :S, :S].numpy()
        im  = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"Head {head_idx + 1}", fontsize=9)
        ax.set_xticks(range(S))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticks(range(S))
        ax.set_yticklabels(tokens, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused subplot axes
    for idx in range(h, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def log_attention_maps_wandb(
    model,
    dataloader,
    src_vocab,
    device: str,
    wandb_run,
    n_sentences: int = 3,
    layer_idx: int = -1,
) -> None:
    """
    Extract encoder attention maps for `n_sentences` test samples and
    upload them as W&B Image objects.
    """
    import wandb

    pad_idx = getattr(src_vocab, "PAD", 1)
    count   = 0

    for src, _ in dataloader:
        if count >= n_sentences:
            break
        src = src[:1].to(device)
        try:
            attn_w, tokens = extract_encoder_attention(
                model, src, pad_idx, layer_idx
            )
        except RuntimeError:
            continue

        fig = plot_attention_heads(
            attn_w, tokens,
            title=f"Encoder attention — sentence {count + 1} "
                  f"(layer {'last' if layer_idx == -1 else layer_idx + 1})",
        )
        wandb_run.log({f"attention/sentence_{count+1}": wandb.Image(fig)})
        plt.close(fig)
        count += 1


# ══════════════════════════════════════════════════════════════════════
#  PREDICTION CONFIDENCE  (§2.5 — label-smoothing ablation)
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_prediction_confidence(
    model,
    dataloader,
    device: str,
    pad_idx: int = 1,
    n_batches: int = 20,
) -> float:
    """
    Compute the average softmax probability assigned to the *correct* token
    across non-pad positions.

    A well-calibrated model trained with ε=0.1 should show lower confidence
    than one trained with ε=0.0 (standard cross-entropy), even if its BLEU
    score is higher — the core insight of §2.5.

    Returns:
        float in (0, 1).
    """
    model.eval()
    confidences: List[float] = []

    for i, (src, tgt) in enumerate(dataloader):
        if i >= n_batches:
            break
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_inp = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_inp, pad_idx)

        logits = model(src, tgt_inp, src_mask, tgt_mask)       # [B, T, V]
        probs  = torch.softmax(logits, dim=-1)                 # [B, T, V]

        gold_probs = probs.gather(2, tgt_out.unsqueeze(-1)).squeeze(-1)  # [B, T]

        mask = tgt_out != pad_idx
        if mask.any():
            confidences.append(gold_probs[mask].mean().item())

    return float(sum(confidences) / max(len(confidences), 1))


# ══════════════════════════════════════════════════════════════════════
#  TRANSLATION SAMPLE TABLE  (qualitative W&B logging)
# ══════════════════════════════════════════════════════════════════════

def log_translation_samples(
    model,
    dataloader,
    src_vocab,
    tgt_vocab,
    device: str,
    wandb_run,
    n_samples: int = 10,
    max_len: int = 60,
) -> None:
    """
    Decode a handful of test sentences and upload them as a W&B Table
    with columns: source | reference | hypothesis.

    Useful for qualitative inspection in the W&B report.
    """
    import wandb
    from train import greedy_decode

    pad_idx = getattr(tgt_vocab, "PAD", 1)
    sos_idx = getattr(tgt_vocab, "SOS", 2)
    eos_idx = getattr(tgt_vocab, "EOS", 3)
    if hasattr(tgt_vocab, "stoi"):
        pad_idx = tgt_vocab.stoi.get("<pad>", pad_idx)
        sos_idx = tgt_vocab.stoi.get("<sos>", sos_idx)
        eos_idx = tgt_vocab.stoi.get("<eos>", eos_idx)

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

                output = greedy_decode(
                    model, src_i, src_mask,
                    max_len, sos_idx, eos_idx, device,
                )

                def ids_to_str(ids, vocab, stop_id):
                    words = []
                    for idx in ids:
                        if idx == stop_id:
                            break
                        tok = vocab.lookup_token(idx)
                        if tok not in ("<sos>", "<eos>", "<pad>", "<unk>"):
                            words.append(tok)
                    return " ".join(words)

                src_str = ids_to_str(src[i].tolist()[1:], src_vocab, eos_idx)
                ref_str = ids_to_str(tgt[i].tolist()[1:], tgt_vocab, eos_idx)
                hyp_str = ids_to_str(output[0].tolist()[1:], tgt_vocab, eos_idx)

                table.add_data(src_str, ref_str, hyp_str)
                count += 1

    wandb_run.log({"translation_samples": table})


# ══════════════════════════════════════════════════════════════════════
#  NOAM LR VISUALISATION
# ══════════════════════════════════════════════════════════════════════

def plot_lr_schedule(
    d_model: int = 512,
    warmup_steps: int = 4000,
    total_steps: int = 20_000,
    save_path: str = "lr_schedule.png",
) -> plt.Figure:
    """
    Plot and (optionally) save the Noam LR curve.
    """
    lrs = get_lr_history(d_model, warmup_steps, total_steps)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(lrs, color="steelblue", linewidth=1.5)
    ax.axvline(warmup_steps, color="red", linestyle="--",
               label=f"warmup = {warmup_steps}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning rate  (base_lr = 1)")
    ax.set_title(f"Noam LR Schedule  (d_model = {d_model})")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig


# ══════════════════════════════════════════════════════════════════════
#  QUICK SELF-TEST  —  python utils.py
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import torch

    # 1. Re-exports accessible
    from utils import (
        LabelSmoothingLoss, NoamScheduler, get_lr_history,
        make_src_mask, make_tgt_mask,
    )
    print("✓ Re-exports OK")

    # 2. LR schedule plot
    fig = plot_lr_schedule(d_model=256, warmup_steps=4000,
                           total_steps=15_000, save_path="lr_schedule.png")
    plt.close(fig)
    print("✓ LR schedule plot saved → lr_schedule.png")

    # 3. GradNormTracker
    from model import Transformer
    model = Transformer(50, 40, d_model=32, N=1, num_heads=4,
                        d_ff=64, dropout=0.0)
    tracker = GradNormTracker(model, names=["W_q", "W_k"])

    src = torch.randint(2, 50, (2, 6))
    tgt = torch.randint(2, 40, (2, 4))
    sm  = make_src_mask(src, 1)
    tm  = make_tgt_mask(tgt, 1)
    out = model(src, tgt, sm, tm)
    out.sum().backward()
    norms = tracker.record()
    assert len(norms) > 0, "GradNormTracker recorded nothing"
    print(f"✓ GradNormTracker: recorded {len(norms)} param norms")

    # 4. count_parameters
    n = count_parameters(model)
    assert n > 0
    print(f"✓ count_parameters: {n:,} trainable params")

    # 5. model_summary (smoke test)
    summary = model_summary(model)
    assert "TOTAL" in summary
    print("✓ model_summary OK")

    print("\nAll utils self-tests passed.")
