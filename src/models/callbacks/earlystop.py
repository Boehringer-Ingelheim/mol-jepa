import torch
from lightning.pytorch.callbacks import Callback


class EarlyStopping(Callback):
    def __init__(self, metric_name="validate_loss", patience=10, mode="min"):
        super().__init__()
        self.metric_name = metric_name
        self.patience = patience
        self.mode = mode

        # Initialized in setup
        self.best_score = None
        self.counter = 0
        self.is_better = None

    def setup(self, trainer, pl_module, stage=None):
        """
        Called at the beginning of fit, validate, test, or predict.
        We use this to reset state or define logic based on the mode.
        """
        if stage == "fit":
            self.counter = 0
            self.best_score = float("inf") if self.mode == "min" else float("-inf")

            if self.mode == "min":
                self.is_better = lambda current, best: current < best
            else:
                self.is_better = lambda current, best: current > best

            print(f"EarlyStopping setup complete: Monitoring {self.metric_name}")

    def on_validation_end(self, trainer, pl_module):
        # Skip if we are in 'sanity check' (Lightning runs 2 steps of val at start)
        if trainer.sanity_checking:
            return

        # 1. Grab the metric
        logs = trainer.callback_metrics
        if self.metric_name not in logs:
            # Optional: Warn the user if they misspelled the metric name
            return

        current_score = logs[self.metric_name].item()

        # 2. Comparison Logic
        if self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        # 3. Stop the trainer
        if self.counter >= self.patience:
            print(
                f"[{self.metric_name}] has not improved for {self.patience} epochs. Stopping."
            )
            trainer.should_stop = True
