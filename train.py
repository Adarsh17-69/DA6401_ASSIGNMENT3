"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol,        │
  │                end_symbol, device)                                  │
  │      → torch.Tensor  shape [1, out_len]  (token indices)           │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)          │
  │      → float  (corpus-level BLEU score, 0–100)                     │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None  │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int   │
  └─────────────────────────────────────────────────────────────────────┘
"""

import os
import time
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".

        y_smooth[c] = 1 − ε        if c == gold class
                      ε / (V − 2)  otherwise  (excluding pad and gold)
                      0            if c == pad_idx

    Args:
        vocab_size : V — number of output classes.
        pad_idx    : <pad> index — always receives 0 probability.
        smoothing  : ε (default 0.1).
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx:    int,
        smoothing:  float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits : [N, vocab_size]  raw pre-softmax scores
            target : [N]              gold token indices
        Returns:
            Scalar mean loss over non-pad positions.
        """
        V  = self.vocab_size
        N  = logits.size(0)

        smooth_val   = self.smoothing / max(V - 2, 1)
        smooth_labels = torch.full(
            (N, V), smooth_val, dtype=torch.float, device=logits.device
        )
        smooth_labels.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_labels[:, self.pad_idx] = 0.0

        pad_mask = target.eq(self.pad_idx)
        smooth_labels[pad_mask] = 0.0

        log_probs       = F.log_softmax(logits, dim=-1)
        loss_per_token  = -(smooth_labels * log_probs).sum(dim=-1)
        num_tokens      = (~pad_mask).sum().clamp(min=1)
        return loss_per_token.sum() / num_tokens


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model:     Transformer,
    loss_fn:   nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int  = 0,
    is_train:  bool = True,
    device:    str  = "cpu",
    wandb_run=None,
    log_grad_norm: bool = False,
) -> float:
    """
    Run one epoch of training or evaluation.

    Returns:
        avg_loss : Average per-token loss over the epoch.
    """
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_tokens = 0
    pad_idx      = 1                                         # Vocabulary.PAD

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for src, tgt in data_iter:
            src = src.to(device)                            # [B, S]
            tgt = tgt.to(device)                            # [B, T]

            tgt_inp = tgt[:, :-1]                           # [B, T-1]  fed to decoder
            tgt_out = tgt[:, 1:]                            # [B, T-1]  prediction target

            src_mask = make_src_mask(src, pad_idx)
            tgt_mask = make_tgt_mask(tgt_inp, pad_idx)

            logits = model(src, tgt_inp, src_mask, tgt_mask)  # [B, T-1, V]

            B, T, V = logits.shape
            loss = loss_fn(
                logits.contiguous().view(B * T, V),
                tgt_out.contiguous().view(B * T),
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                if log_grad_norm and wandb_run is not None:
                    g_norm = sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in model.parameters()
                        if p.grad is not None
                    ) ** 0.5
                    wandb_run.log({
                        "grad_norm": g_norm,
                        "lr": optimizer.param_groups[0]["lr"],
                    })

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            n_tok        = (tgt_out != pad_idx).sum().item()
            total_loss  += loss.item() * n_tok
            total_tokens += n_tok

    return total_loss / max(total_tokens, 1)


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model:        Transformer,
    src:          torch.Tensor,
    src_mask:     torch.Tensor,
    max_len:      int,
    start_symbol: int,
    end_symbol:   int,
    device:       str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : [1, src_len] source token indices.
        src_mask     : [1, 1, 1, src_len].
        max_len      : Maximum tokens to generate.
        start_symbol : <sos> vocabulary index.
        end_symbol   : <eos> vocabulary index.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : [1, out_len] — includes <sos>; stops at (and includes) <eos>
             or when max_len is reached.
    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys     = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            T = ys.size(1)
            # Pure causal mask — no padding inside ys during autoregressive decode
            tgt_mask = ~torch.tril(
                torch.ones(1, 1, T, T, dtype=torch.bool, device=device)
            )
            logits     = model.decode(memory, src_mask, ys, tgt_mask)  # [1, T, V]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True) # [1, 1]
            ys         = torch.cat([ys, next_token], dim=1)
            if next_token.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model:           Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device:  str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU.

    Returns:
        bleu_score : float in [0, 100].
    """
    import sacrebleu

    # Resolve special-token indices from vocab
    pad_idx = getattr(tgt_vocab, "PAD", 1)
    sos_idx = getattr(tgt_vocab, "SOS", 2)
    eos_idx = getattr(tgt_vocab, "EOS", 3)
    if hasattr(tgt_vocab, "stoi"):
        pad_idx = tgt_vocab.stoi.get("<pad>", pad_idx)
        sos_idx = tgt_vocab.stoi.get("<sos>", sos_idx)
        eos_idx = tgt_vocab.stoi.get("<eos>", eos_idx)

    hypotheses: list = []
    references:  list = []

    model.eval()
    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                src_i    = src[i:i+1]
                src_mask = make_src_mask(src_i, pad_idx)
                output   = greedy_decode(
                    model, src_i, src_mask,
                    max_len, sos_idx, eos_idx, device,
                )

                def ids_to_str(ids, stop):
                    toks = []
                    for idx in ids:
                        if idx == stop:
                            break
                        tok = tgt_vocab.lookup_token(idx)
                        if tok not in ("<sos>", "<eos>", "<pad>", "<unk>"):
                            toks.append(tok)
                    return " ".join(toks)

                hyp = ids_to_str(output[0].tolist()[1:], eos_idx)
                ref = ids_to_str(tgt[i].tolist()[1:],    eos_idx)
                hypotheses.append(hyp)
                references.append(ref)

    return sacrebleu.corpus_bleu(hypotheses, [references]).score


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model:     Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch:     int,
    path:      str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state.

    Saved keys:
        epoch, model_state_dict, optimizer_state_dict,
        scheduler_state_dict, model_config
    """
    model_config = {
        "src_vocab_size": model.src_tok_embed.embedding.num_embeddings,
        "tgt_vocab_size": model.tgt_tok_embed.embedding.num_embeddings,
        "d_model":        model.d_model,
        "N":              len(model.encoder.layers),
        "num_heads":      model.encoder.layers[0].self_attn.num_heads,
        "d_ff":           model.encoder.layers[0].ff.linear1.out_features,
        "dropout":        model.encoder.layers[0].dropout.p,
        "use_scale":      model.use_scale,
    }
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
                                    if scheduler is not None else None,
            "model_config":         model_config,
        },
        path,
    )
    print(f"[ckpt] Saved epoch {epoch} → {path}")


def load_checkpoint(
    path:      str,
    model:     Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) from checkpoint.

    Returns:
        epoch : int
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    epoch = ckpt.get("epoch", 0)
    print(f"[ckpt] Loaded epoch {epoch} from {path}")
    return epoch


# ══════════════════════════════════════════════════════════════════════
#  FULL TRAINING EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment(
    # model
    d_model:         int   = 256,
    N:               int   = 3,
    num_heads:       int   = 8,
    d_ff:            int   = 512,
    dropout:         float = 0.1,
    # training
    num_epochs:      int   = 15,
    batch_size:      int   = 128,
    warmup_steps:    int   = 4000,
    label_smoothing: float = 0.1,
    min_freq:        int   = 2,
    # ablations
    use_noam:        bool  = True,
    fixed_lr:        float = 1e-4,
    use_scale:       bool  = True,
    learned_pe:      bool  = False,
    # infra
    checkpoint_dir:  str   = "checkpoints",
    device_str:      str   = "auto",
    wandb_project:   str   = "da6401-a3",
    run_name:        str   = None,
) -> None:
    """Full training experiment with W&B logging."""
    import wandb
    from dataset import build_dataloaders, Vocabulary
    from lr_scheduler import NoamScheduler

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if device_str == "auto" else device_str
    print(f"Device: {device}")

    config = dict(d_model=d_model, N=N, num_heads=num_heads, d_ff=d_ff,
                  dropout=dropout, num_epochs=num_epochs, batch_size=batch_size,
                  warmup_steps=warmup_steps, label_smoothing=label_smoothing,
                  use_noam=use_noam, use_scale=use_scale, learned_pe=learned_pe)
    run = wandb.init(
        project=wandb_project,
        name=run_name or f"d{d_model}_N{N}_h{num_heads}_ls{label_smoothing}",
        config=config,
    )

    # ── data ──────────────────────────────────────────────────────────
    train_dl, val_dl, test_dl, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=batch_size, min_freq=min_freq
    )
    pad_idx = Vocabulary.PAD

    # ── model ─────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model, N=N, num_heads=num_heads,
        d_ff=d_ff, dropout=dropout,
        use_scale=use_scale, learned_pe=learned_pe,
    ).to(device)

    model.set_vocab(
        src_vocab, tgt_vocab,
        pad_idx=pad_idx, sos_idx=Vocabulary.SOS, eos_idx=Vocabulary.EOS,
    )
    wandb.log({"num_params": sum(p.numel() for p in model.parameters()
                                  if p.requires_grad)})

    # ── optimiser ─────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )

    # ── scheduler ─────────────────────────────────────────────────────
    if use_noam:
        scheduler = NoamScheduler(optimizer, d_model=d_model,
                                  warmup_steps=warmup_steps)
    else:
        for pg in optimizer.param_groups:
            pg["lr"] = fixed_lr
        scheduler = None

    # ── loss ──────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(len(tgt_vocab), pad_idx, label_smoothing)

    # ── training loop ─────────────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val  = float("inf")
    best_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")

    for epoch in range(num_epochs):
        t0 = time.time()

        train_loss = run_epoch(
            train_dl, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
            wandb_run=run, log_grad_norm=(epoch == 0),
        )
        val_loss = run_epoch(
            val_dl, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )

        # Val BLEU every 2 epochs and on the last epoch
        val_bleu = 0.0
        if epoch % 2 == 0 or epoch == num_epochs - 1:
            val_bleu = evaluate_bleu(model, val_dl, tgt_vocab, device)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"bleu={val_bleu:.2f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"t={elapsed:.1f}s"
        )
        wandb.log({
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "train_ppl":  math.exp(min(train_loss, 20)),
            "val_loss":   val_loss,
            "val_ppl":    math.exp(min(val_loss,   20)),
            "val_bleu":   val_bleu,
            "lr":         optimizer.param_groups[0]["lr"],
        })

        epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1:02d}.pt")
        save_checkpoint(model, optimizer, scheduler, epoch + 1, epoch_path)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best_path)
            print(f"  ↑ best checkpoint → {best_path}")

        # Also save a root-level checkpoint.pt for the autograder
        save_checkpoint(model, optimizer, scheduler, epoch + 1, "checkpoint.pt")

    # ── final test BLEU ───────────────────────────────────────────────
    load_checkpoint(best_path, model)
    model.to(device)
    test_bleu = evaluate_bleu(model, test_dl, tgt_vocab, device)
    print(f"Test BLEU: {test_bleu:.2f}")
    wandb.log({"test_bleu": test_bleu})

    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  ABLATION LAUNCHERS
# ══════════════════════════════════════════════════════════════════════

def run_noam_vs_fixed_lr():
    """§2.1 — Noam scheduler vs fixed LR."""
    run_training_experiment(use_noam=True,  run_name="noam_scheduler")
    run_training_experiment(use_noam=False, fixed_lr=1e-4, run_name="fixed_lr_1e4")


def run_scale_ablation():
    """§2.2 — with vs without √dk scaling."""
    run_training_experiment(use_scale=True,  run_name="with_scale")
    run_training_experiment(use_scale=False, run_name="no_scale")


def run_pe_ablation():
    """§2.4 — sinusoidal PE vs learned embeddings."""
    run_training_experiment(learned_pe=False, run_name="sinusoidal_pe")
    run_training_experiment(learned_pe=True,  run_name="learned_pe")


def run_label_smoothing_ablation():
    """§2.5 — label smoothing ε=0.1 vs ε=0.0."""
    run_training_experiment(label_smoothing=0.1, run_name="smooth_0.1")
    run_training_experiment(label_smoothing=0.0, run_name="smooth_0.0")


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_training_experiment()
