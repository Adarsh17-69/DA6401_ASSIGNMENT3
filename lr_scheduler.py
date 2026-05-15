"""
lr_scheduler.py — Noam Learning Rate Scheduler
Reference: "Attention Is All You Need" (Vaswani et al., 2017)

Formula:
    lrate = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))

Peak occurs exactly at step == warmup_steps where both arms are equal.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


class NoamScheduler(LRScheduler):
    """
    Noam LR schedule: linear warm-up then inverse-sqrt decay.

    PyTorch's LRScheduler calls step() once inside __init__, setting
    last_epoch to 0.  We therefore define:
        step = self.last_epoch + 1
    so step 1 is the very first training update.

    Args:
        optimizer    : Wrapped optimizer.
        d_model      : Model dimensionality.
        warmup_steps : Number of linear warm-up steps.
        last_epoch   : Starting index (default -1).
    """

    def __init__(
        self,
        optimizer:    optim.Optimizer,
        d_model:      int,
        warmup_steps: int,
        last_epoch:   int = -1,
    ) -> None:
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def _get_lr_scale(self) -> float:
        """
        Noam scaling factor for the current step.

            step  = last_epoch + 1
            scale = d_model^(-0.5) · min(step^(-0.5), step · warmup^(-1.5))
        """
        step  = self.last_epoch + 1
        scale = (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5),
        )
        return scale

    def get_lr(self) -> list:
        """One LR per param group = base_lr × Noam scale."""
        scale = self._get_lr_scale()
        return [base_lr * scale for base_lr in self.base_lrs]


# ─────────────────────────────────────────────────────────────────────
# Helper (do NOT modify — used by the autograder)
# ─────────────────────────────────────────────────────────────────────

def get_lr_history(d_model: int, warmup_steps: int, total_steps: int) -> list:
    """Simulate the LR trajectory for total_steps steps."""
    dummy     = torch.nn.Linear(1, 1)
    optimizer = optim.Adam(dummy.parameters(), lr=1.0)
    scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    history = []
    for _ in range(total_steps):
        history.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return history


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    D_MODEL, WARMUP, TOTAL = 512, 4000, 20_000
    lrs  = get_lr_history(D_MODEL, WARMUP, TOTAL)
    peak = lrs.index(max(lrs))
    print(f"Peak LR = {max(lrs):.6f}  at step {peak + 1}  (expected {WARMUP})")

    plt.figure(figsize=(9, 4))
    plt.plot(lrs)
    plt.axvline(WARMUP, color="red", linestyle="--", label=f"warmup={WARMUP}")
    plt.xlabel("Step"); plt.ylabel("LR"); plt.title(f"Noam (d_model={D_MODEL})")
    plt.legend(); plt.tight_layout()
    plt.savefig("noam_lr_schedule.png", dpi=150)
    plt.show()
