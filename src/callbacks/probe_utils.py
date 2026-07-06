import math
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from torch import distributed as dist
from lightning.pytorch import Callback, LightningModule, Trainer
import matplotlib.pyplot as plt


class ProbeScatterPlot(Callback):
    """Log an nxn grid of scatter plots (pred vs true) for a group of probes."""

    def __init__(
        self, probe_names: List[str], target_keys: List[str], name: str
    ) -> None:
        super().__init__()
        self.probe_names = probe_names
        self.target_keys = target_keys
        self.name = name
        self._preds: Dict[str, List[torch.Tensor]] = {}
        self._targets: Dict[str, List[torch.Tensor]] = {}

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not isinstance(outputs, dict):
            return
        for pname, tkey in zip(self.probe_names, self.target_keys):
            pred_key = f"{pname}_preds"
            if pred_key in outputs and tkey in batch:
                p = outputs[pred_key].detach().cpu().float().squeeze(-1)
                t = batch[tkey].detach().cpu().float().squeeze(-1)
                mask = ~torch.isnan(t)
                if mask.sum() > 0:
                    self._preds.setdefault(pname, []).append(p[mask])
                    self._targets.setdefault(pname, []).append(t[mask])

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # Gather predictions from all ranks onto rank 0
        for pname in self.probe_names:
            if pname in self._preds:
                local_p = torch.cat(self._preds[pname])
                local_t = torch.cat(self._targets[pname])
            else:
                local_p = torch.zeros(0)
                local_t = torch.zeros(0)
            self._preds[pname] = [_gather_tensors(local_p)]
            self._targets[pname] = [_gather_tensors(local_t)]

        if trainer.global_rank != 0:
            self._preds.clear()
            self._targets.clear()
            return
        if not self._preds:
            return

        names_with_data = [n for n in self.probe_names if n in self._preds]
        if not names_with_data:
            self._preds.clear()
            self._targets.clear()
            return

        n = len(names_with_data)
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(
            rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False
        )

        for idx, pname in enumerate(names_with_data):
            ax = axes[idx // cols][idx % cols]
            p = torch.cat(self._preds[pname]).numpy()
            t = torch.cat(self._targets[pname]).numpy()
            if t.size == 0 or p.size == 0:
                ax.set_title(f"{pname}\n(no data)", fontsize=8)
                continue
            ax.scatter(t, p, alpha=0.4, s=10)
            lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            ss_res = ((t - p) ** 2).sum()
            ss_tot = ((t - t.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            short_name = pname.replace(
                f"probe_{self.name.split('_', 1)[0]}_"
                if "_" in self.name
                else "probe_",
                "",
            )
            ax.set_title(f"{short_name}\n$R^2$={r2:.3f}", fontsize=8)
            ax.set_xlabel("true")
            ax.set_ylabel("pred")

        for idx in range(n, rows * cols):
            axes[idx // cols][idx % cols].axis("off")

        fig.suptitle(self.name, fontsize=12)
        fig.tight_layout()

        if trainer.logger and hasattr(trainer.logger, "experiment"):
            exp = trainer.logger.experiment
            if hasattr(exp, "add_figure"):
                exp.add_figure(
                    f"eval/{self.name}_scatter", fig, global_step=trainer.global_step
                )
        plt.close(fig)

        self._preds.clear()
        self._targets.clear()


class ModalityLossBarPlot(Callback):
    """Bar plot of per-modality sigreg and prediction losses (dual y-axes)."""

    def __init__(
        self,
        modality_names: List[str],
        stage: str = "validate",
        name: str = "modality_losses",
    ) -> None:
        super().__init__()
        self.modality_names = modality_names
        self.stage = stage
        self.name = name

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.global_rank != 0:
            return

        metrics = trainer.callback_metrics
        n = len(self.modality_names)

        sigreg_vals = []
        pred_vals = []
        for i in range(n):
            s = metrics.get(f"{self.stage}_sigreg_loss_m_{i}")
            p = metrics.get(f"{self.stage}_pred_loss_m_{i}")
            sigreg_vals.append(
                s.item() if s is not None and hasattr(s, "item") else 0.0
            )
            pred_vals.append(p.item() if p is not None and hasattr(p, "item") else 0.0)

        x = np.arange(n)
        width = 0.35

        fig, ax_left = plt.subplots(figsize=(max(6, n * 1.5), 4), tight_layout=True)
        ax_right = ax_left.twinx()

        bars_sig = ax_left.bar(
            x - width / 2, sigreg_vals, width, label="sigreg loss", color="steelblue"
        )
        bars_pred = ax_right.bar(
            x + width / 2, pred_vals, width, label="pred loss", color="coral"
        )

        for ax, bars in ((ax_left, bars_sig), (ax_right, bars_pred)):
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        ax_left.set_xlabel("Modality")
        ax_left.set_ylabel("Sigreg Loss", color="steelblue")
        ax_right.set_ylabel("Prediction Loss", color="coral")
        ax_left.tick_params(axis="y", labelcolor="steelblue")
        ax_right.tick_params(axis="y", labelcolor="coral")

        ax_left.set_xticks(x)
        ax_left.set_xticklabels(self.modality_names, rotation=45, ha="right")

        # Combined legend
        lines = [bars_sig, bars_pred]
        labels = [l.get_label() for l in lines]
        ax_left.legend(lines, labels, loc="upper right")

        ax_left.set_title(f"Per-modality losses (epoch {trainer.current_epoch})")

        if trainer.logger and hasattr(trainer.logger, "experiment"):
            exp = trainer.logger.experiment
            if hasattr(exp, "add_figure"):
                exp.add_figure(
                    f"eval/{self.name}", fig, global_step=trainer.global_step
                )
        plt.close(fig)


class MultimodalWeightLogger(Callback):
    """Log aggregated modality importance weights across weighted probes as a bar plot."""

    def __init__(
        self,
        probe_names: Optional[List[str]] = None,
        modality_names: Optional[List[str]] = None,
        name: str = "modality_importance",
    ) -> None:
        self.probe_names = probe_names
        self.modality_names = modality_names
        self.name = name

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.global_rank != 0:
            return

        all_weights = []
        for cb in trainer.callbacks:
            if hasattr(cb, "name") and cb.name in self.probe_names:
                if hasattr(cb, "_probe_config") and hasattr(
                    cb._probe_config, "weights"
                ):
                    all_weights.append(cb._probe_config.weights.cpu().numpy())

        if not all_weights:
            return

        all_weights = np.nan_to_num(np.stack(all_weights), 0)
        means = all_weights.mean(axis=0)
        stds = all_weights.std(axis=0)
        labels = self.modality_names or [f"mod_{i}" for i in range(means.shape[0])]

        fig, ax = plt.subplots(figsize=(6, 3), tight_layout=True)
        bars = ax.bar(labels, means, yerr=stds, capsize=4)
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
        ax.set_ylabel("Importance weight")
        ax.set_title(f"Modality importance (epoch {trainer.current_epoch})")

        trainer.logger.experiment.add_figure(
            f"eval/{self.name}", fig, global_step=trainer.global_step
        )
        plt.close(fig)


class BestPredictionSaver(Callback):
    """Save probe predictions to CSV when validate_loss improves."""

    def __init__(
        self,
        probe_names: List[str],
        target_keys: List[str],
        monitor: str = "validate_loss",
    ) -> None:
        super().__init__()
        self.probe_names = probe_names
        self.target_keys = target_keys
        self.monitor = monitor
        self.best_loss = float("inf")
        self._preds: Dict[str, List[torch.Tensor]] = {}
        self._targets: Dict[str, List[torch.Tensor]] = {}
        self._smiles: Dict[str, List] = {}

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if not isinstance(outputs, dict):
            return
        for pname, tkey in zip(self.probe_names, self.target_keys):
            pred_key = f"{pname}_preds"
            if pred_key in outputs and tkey in batch:
                p = outputs[pred_key].detach().cpu().float().squeeze(-1)
                t = batch[tkey].detach().cpu().float().squeeze(-1)
                mask = ~torch.isnan(t)
                if mask.sum() > 0:
                    self._preds.setdefault(pname, []).append(p[mask])
                    self._targets.setdefault(pname, []).append(t[mask])
                    if "smiles" in batch:
                        smiles = batch["smiles"]
                        if isinstance(smiles, torch.Tensor):
                            smiles = smiles.tolist()
                        masked_smiles = [s for s, m in zip(smiles, mask.tolist()) if m]
                        self._smiles.setdefault(pname, []).extend(masked_smiles)

    def _get_log_dir(self, trainer) -> Optional[Path]:
        logger = trainer.logger
        if logger is None:
            return None
        if hasattr(logger, "log_dir"):
            return Path(logger.log_dir)
        return None

    def on_validation_epoch_end(self, trainer, pl_module):
        # Gather predictions from all ranks onto rank 0
        for pname in self.probe_names:
            if pname in self._preds:
                local_p = torch.cat(self._preds[pname])
                local_t = torch.cat(self._targets[pname])
            else:
                local_p = torch.zeros(0)
                local_t = torch.zeros(0)
            self._preds[pname] = [_gather_tensors(local_p)]
            self._targets[pname] = [_gather_tensors(local_t)]
            if pname in self._smiles:
                self._smiles[pname] = _gather_strings(self._smiles[pname])
            else:
                self._smiles[pname] = _gather_strings([])

        if trainer.global_rank != 0:
            self._preds.clear()
            self._targets.clear()
            self._smiles.clear()
            return

        current_loss = trainer.callback_metrics.get(self.monitor)
        if current_loss is None:
            self._preds.clear()
            self._targets.clear()
            self._smiles.clear()
            return

        current_loss = (
            current_loss.item()
            if hasattr(current_loss, "item")
            else float(current_loss)
        )
        if current_loss >= self.best_loss:
            self._preds.clear()
            self._targets.clear()
            self._smiles.clear()
            return

        self.best_loss = current_loss
        log_dir = self._get_log_dir(trainer)
        if log_dir is None:
            self._preds.clear()
            self._targets.clear()
            self._smiles.clear()
            return

        pred_dir = log_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)

        for pname, tkey in zip(self.probe_names, self.target_keys):
            if pname not in self._preds:
                continue
            p = torch.cat(self._preds[pname]).numpy()
            t = torch.cat(self._targets[pname]).numpy()
            if len(t) == 0:
                continue
            smiles_col = self._smiles.get(pname, [None] * len(t))
            df = pd.DataFrame({"smiles": smiles_col, "target": t, "prediction": p})
            df.to_csv(pred_dir / f"val_preds_{pname}.csv", index=False)

        self._preds.clear()
        self._targets.clear()
        self._smiles.clear()


def _gather_tensors(local_tensor: torch.Tensor) -> torch.Tensor:
    """Gather variable-length 1-D tensors from all ranks onto rank 0."""
    if not dist.is_initialized():
        return local_tensor
    local_size = torch.tensor([local_tensor.numel()], device="cuda")
    world_size = dist.get_world_size()
    all_sizes = [
        torch.zeros(1, device="cuda", dtype=torch.long) for _ in range(world_size)
    ]
    dist.all_gather(all_sizes, local_size)
    max_size = int(max(s.item() for s in all_sizes))
    padded = torch.zeros(max_size, device="cuda", dtype=local_tensor.dtype)
    padded[: local_tensor.numel()] = local_tensor.to("cuda")
    gathered = [
        torch.zeros(max_size, device="cuda", dtype=local_tensor.dtype)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered, padded)
    parts = [g[: int(s.item())].cpu() for g, s in zip(gathered, all_sizes)]
    return torch.cat(parts)


def _gather_strings(local_list: List[str]) -> List[str]:
    """Gather variable-length string lists from all ranks onto rank 0."""
    if not dist.is_initialized():
        return local_list
    data = pickle.dumps(local_list)
    local_size = torch.tensor([len(data)], device="cuda")
    world_size = dist.get_world_size()
    all_sizes = [
        torch.zeros(1, device="cuda", dtype=torch.long) for _ in range(world_size)
    ]
    dist.all_gather(all_sizes, local_size)
    max_size = int(max(s.item() for s in all_sizes))
    padded = torch.zeros(max_size, device="cuda", dtype=torch.uint8)
    data_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).to("cuda")
    padded[: len(data)] = data_tensor
    gathered = [
        torch.zeros(max_size, device="cuda", dtype=torch.uint8)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered, padded)
    result = []
    for g, s in zip(gathered, all_sizes):
        raw = g[: int(s.item())].cpu().numpy().tobytes()
        result.extend(pickle.loads(raw))
    return result
