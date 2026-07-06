import torch
import torchmetrics


class MaskedMAE(torchmetrics.Metric):
    """torchmetrics.Metric wrapper around masked_mae_loss for NaN-masked targets."""

    higher_is_better = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        mask = ~targets.isnan()
        if mask.sum() == 0:
            return
        self.sum_abs_error += (preds[mask] - targets[mask]).abs().sum()
        self.total += mask.sum()

    def compute(self):
        if self.total == 0:
            return torch.tensor(float("nan"), device=self.sum_abs_error.device)
        return self.sum_abs_error / self.total


def masked_mae_loss(preds, targets):
    mask = ~targets.isnan()
    if mask.sum() == 0:
        # Return a zero that stays in the computation graph (connected to preds)
        # so DDP doesn't see unused probe parameters. Using a NaN leaf tensor
        # would detach the probe from the backward graph entirely.
        return (preds * 0.0).sum()
    
    return torchmetrics.functional.mean_absolute_error(preds[mask], targets[mask])
