import torch
from lightning.pytorch.callbacks import Callback


class ThresholdStopping(Callback):
    def __init__(
        self, metric_name="val_loss", threshold=0.5, epoch_checkpoint=10, mode="min"
    ):
        """
        Args:
            metric_name: The metric to monitor.
            threshold: The value the metric must reach (or be better than).
            epoch_checkpoint: The epoch at which to perform the check (0-indexed).
            mode: "min" if lower is better (e.g. loss), "max" if higher is better (e.g. accuracy).
        """
        super().__init__()
        self.metric_name = metric_name
        self.threshold = threshold
        self.epoch_checkpoint = epoch_checkpoint
        self.mode = mode
        self.is_within_threshold = None

    def setup(self, trainer, pl_module, stage=None):
        if stage == "fit":
            if self.mode == "min":
                self.is_within_threshold = lambda current: current <= self.threshold
            else:
                self.is_within_threshold = lambda current: current >= self.threshold

            print(
                f"ThresholdStopping setup: Monitoring {self.metric_name} at epoch {self.epoch_checkpoint}"
            )

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        # Lightning's current_epoch is 0-indexed
        if trainer.current_epoch == self.epoch_checkpoint:
            logs = trainer.callback_metrics

            if self.metric_name not in logs:
                print(f"Warning: {self.metric_name} not found in logs at checkpoint.")
                return

            current_score = logs[self.metric_name].item()

            if not self.is_within_threshold(current_score):
                print(
                    f"\n[Threshold Met] Epoch {trainer.current_epoch}: "
                    f"Metric {self.metric_name} is {current_score:.4f}, "
                    f"which does not meet the threshold of {self.threshold}. Stopping training."
                )
                trainer.should_stop = True
            else:
                print(
                    f"\n[Threshold Met] Metric {self.metric_name} is {current_score:.4f}. Continuing..."
                )
