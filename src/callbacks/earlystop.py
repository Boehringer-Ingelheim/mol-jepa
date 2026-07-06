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
        if trainer.sanity_checking:
            return

        logs = trainer.callback_metrics
        if self.metric_name not in logs:
            print(
                f"EarlyStopping: Metric '{self.metric_name}' not found in logs. Available metrics: {list(logs.keys())}"
            )
            return

        current_score = logs[self.metric_name].item()
        if self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(
                f"[{self.metric_name}] has not improved for {self.patience} epochs. Stopping."
            )
            trainer.should_stop = True
