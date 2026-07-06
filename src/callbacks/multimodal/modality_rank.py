"""Per-modality effective rank (RankMe) callback.

Computes the effective rank of each modality's embedding matrix within a
batch using the RankMe metric (exponential of the entropy of the normalised
singular values).  Displays results as a grouped bar plot for train/validate.

A high effective rank means the modality's encoder uses many dimensions of
the embedding space.  A low effective rank indicates collapse — the encoder
maps all molecules to a low-dimensional subspace.
"""

from __future__ import annotations

import random
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer


def rankme(embeddings: torch.Tensor, eps: float = 1e-7) -> float:
    """Compute the effective rank (RankMe) of an (N, D) embedding matrix.

    RankMe = exp(H(p)) where p_i = sigma_i / sum(sigma) and H is Shannon
    entropy.  Returns a scalar >= 1.
    """
    if embeddings.shape[0] < 2:
        return float("nan")
    # Centre the embeddings
    embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    # SVD (only need singular values)
    s = torch.linalg.svdvals(embeddings.float())
    # Normalise to a probability distribution
    s = s / (s.sum() + eps)
    s = s[s > eps]
    # Shannon entropy
    entropy = -(s * s.log()).sum().item()
    return float(np.exp(entropy))


class ModalityRankLogger(Callback):
    """Log per-modality effective rank as a grouped bar plot.

    Uses the ``embeddings`` tensor (MultiModalPredictor output for the
    full pass where all active modalities are visible).  Per-modality
    embeddings at index *i* are valid when ``active_mask[:, i]`` is True.

    Args:
        name:               TensorBoard tag prefix.
        modality_names:     Ordered list of human-readable modality names.
        log_every_n_epochs: Only produce the plot every N epochs.
    """

    def __init__(
        self,
        name: str = "modality_rank",
        modality_names: Optional[List[str]] = None,
        log_every_n_epochs: int = 1,
    ) -> None:
        super().__init__()
        self.name = name
        self.modality_names = modality_names or []
        self.log_every_n_epochs = log_every_n_epochs

        # Reservoir-sampling state per stage
        self._candidates: dict[str, Optional[dict]] = {"fit": None, "validate": None}
        self._batch_counters: dict[str, int] = {"fit": 0, "validate": 0}

    @property
    def state_key(self) -> str:
        return f"ModalityRankLogger[name={self.name}]"

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._candidates["fit"] = None
        self._batch_counters["fit"] = 0

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._candidates["validate"] = None
        self._batch_counters["validate"] = 0

    def _collect_batch(self, outputs: Any, stage: str) -> None:
        if not isinstance(outputs, dict):
            return
        if outputs.get("stage") != stage:
            return

        full_emb = outputs.get("embeddings")
        if full_emb is None:
            return

        active_mask = outputs.get("active_mask")
        if active_mask is None:
            return

        self._batch_counters[stage] += 1
        cnt = self._batch_counters[stage]

        if self._candidates[stage] is None or random.random() < 1.0 / cnt:
            self._candidates[stage] = {
                "full_embeddings": full_emb.detach().cpu(),    # (B, M, D)
                "active_mask": active_mask.detach().cpu(),     # (B, M)
            }

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._collect_batch(outputs, "fit")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._collect_batch(outputs, "validate")

    def _log_results(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        if trainer.global_rank != 0:
            return

        epoch = trainer.current_epoch
        if self.log_every_n_epochs > 1 and epoch % self.log_every_n_epochs != 0:
            return

        data = self._candidates.get(stage)
        if data is None:
            return

        full_emb = data["full_embeddings"]   # (B, M, D)
        active = data["active_mask"]         # (B, M) bool
        B, M, D = full_emb.shape

        # Valid = active (targets are computed with all active modalities)
        valid = active

        ranks = np.full(M, np.nan)
        for i in range(M):
            mask_i = valid[:, i]
            n_active = mask_i.sum().item()
            if n_active < 2:
                continue
            emb_i = full_emb[mask_i, i, :]   # (n_active, D)
            ranks[i] = rankme(emb_i)

        # --- Log scalars ---
        for i in range(M):
            if not np.isnan(ranks[i]):
                mod_name = (
                    self.modality_names[i]
                    if i < len(self.modality_names)
                    else f"mod_{i}"
                )
                pl_module.log(
                    f"{self.name}/{stage}_{mod_name}",
                    ranks[i],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

        # --- Barplot ---
        fig = self._plot_ranks(ranks, epoch, stage, D)
        if fig is not None:
            tb_logger = trainer.logger
            if tb_logger is not None and hasattr(tb_logger, "experiment"):
                exp = tb_logger.experiment
                if hasattr(exp, "add_figure"):
                    exp.add_figure(
                        f"{self.name}/{stage}_barplot",
                        fig,
                        global_step=epoch,
                    )
            plt.close(fig)

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._log_results(trainer, pl_module, "fit")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._log_results(trainer, pl_module, "validate")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _plot_ranks(
        self,
        ranks: np.ndarray,
        epoch: int,
        stage: str,
        hidden_dim: int,
    ) -> Optional[plt.Figure]:
        M = len(ranks)
        if np.isnan(ranks).all():
            return None

        labels = [
            self.modality_names[i] if i < len(self.modality_names) else f"mod_{i}"
            for i in range(M)
        ]

        fig, ax = plt.subplots(figsize=(max(6, M * 0.7), 4))

        valid = ~np.isnan(ranks)
        values = np.where(valid, ranks, 0.0)
        colors = ["#3498db" if v else "#cccccc" for v in valid]

        ax.bar(range(M), values, color=colors, edgecolor="gray", linewidth=0.5)

        # Reference line: maximum possible rank = hidden_dim
        ax.axhline(
            y=hidden_dim, color="red", linestyle="--", linewidth=1, alpha=0.6,
            label=f"max (hidden_dim={hidden_dim})",
        )

        # Annotate bars
        ymax = max(values.max(), hidden_dim) if valid.any() else hidden_dim
        for i, (v, is_valid) in enumerate(zip(values, valid)):
            if is_valid:
                ax.text(
                    i, v + ymax * 0.02, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7,
                )

        ax.set_xticks(range(M))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Effective Rank (RankMe)")
        ax.set_title(
            f"[{stage}] Per-Modality Effective Rank (epoch {epoch})",
            fontsize=10,
        )
        ax.set_ylim(0, ymax * 1.15)
        ax.legend(fontsize=8)

        fig.tight_layout()
        return fig
