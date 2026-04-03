import torch
from lightning.pytorch.callbacks import Callback


class GradientNormLogger(Callback):
    def __init__(self, norm_type=2, log_on_step=True):
        """
        Args:
            norm_type (int/float): The order of the norm (2 = L2 norm).
            log_on_step (bool):
                - True → log after every training step
                - False → log once per training epoch
        """
        super().__init__()
        self.norm_type = norm_type
        self.log_on_step = log_on_step

    def _compute_grad_norm(self, pl_module):
        total_norm = 0.0
        for p in pl_module.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(self.norm_type)
                total_norm += param_norm.item() ** self.norm_type
        total_norm = total_norm ** (1.0 / self.norm_type)
        return total_norm

    def on_after_backward(self, trainer, pl_module):
        """Called after every backward pass."""
        if self.log_on_step:
            grad_norm = self._compute_grad_norm(pl_module)
            trainer.logger.log_metrics(
                {f"grad_norm_{self.norm_type}": grad_norm}, step=trainer.global_step
            )

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of the train epoch."""
        if not self.log_on_step:
            grad_norm = self._compute_grad_norm(pl_module)
            trainer.logger.log_metrics(
                {f"grad_norm_epoch_{self.norm_type}": grad_norm},
                step=trainer.current_epoch,
            )
