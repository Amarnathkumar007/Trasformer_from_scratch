"""
lr_scheduler.py — Noam Learning Rate Scheduler

Implements the learning rate schedule from Section 5.3 of
"Attention Is All You Need":

    lr = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))

The learning rate linearly increases during the warmup phase, then
decays proportionally to the inverse square root of the step number.
"""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class NoamScheduler(_LRScheduler):
    """
    Noam learning rate scheduler (Vaswani et al., 2017).

    Args:
        optimizer:     Wrapped optimizer.
        d_model:       Model dimensionality (affects the scale).
        warmup_steps:  Number of linear warmup steps.
        last_epoch:    Index of last epoch (for resuming).
        verbose:       Print LR updates.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for the current step."""
        step = max(self._step_count, 1)  # avoid division by zero
        scale = self.d_model ** (-0.5) * min(
            step ** (-0.5),
            step * self.warmup_steps ** (-1.5),
        )
        return [scale for _ in self.base_lrs]
