"""Cross-modality importance analysis callback.

Produces two diagnostic plots on a batch for both train and validation:

1. **Pairwise Similarity Matrix** (M × M): Cosine similarity between
   embeddings produced by individual modalities in isolation.  Reveals
   which modalities encode redundant vs. complementary information.

2. **Leave-One-Out Impact** (M bars): L2 distance between the full
   embedding and the embedding with each modality removed.  Shows
   the marginal contribution of each modality to the combined
   representation.

These diagnostics run with ``torch.no_grad`` on the first batch of
each stage and are logged every ``log_every_n_epochs`` epochs.
"""

from __future__ import annotations

from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, LightningModule, Trainer


class ModalityImportanceLogger(Callback):
    """Log cross-modality importance diagnostics to TensorBoard.

    Args:
        name:               TensorBoard tag prefix.
        modality_names:     Ordered list of human-readable modality names.
        log_every_n_epochs: Only produce the plots every N validation epochs.
    """

    def __init__(
        self,
        name: str = "modality_importance",
        modality_names: Optional[List[str]] = None,
        log_every_n_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.name = name
        self.modality_names = modality_names or []
        self.log_every_n_epochs = log_every_n_epochs

        self._results: dict[str, Optional[dict]] = {"fit": None, "validate": None}

    @property
    def state_key(self) -> str:
        return f"ModalityImportanceLogger[name={self.name}]"

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._results["fit"] = None

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._results["validate"] = None

    def _compute_importance(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        stage: str,
    ) -> None:
        # Only compute on the first batch, rank 0
        if batch_idx != 0 or trainer.global_rank != 0:
            return

        epoch = trainer.current_epoch
        if self.log_every_n_epochs > 1 and epoch % self.log_every_n_epochs != 0:
            return

        if not isinstance(outputs, dict):
            return

        active_mask = outputs.get("active_mask")
        if active_mask is None:
            return

        # Reconstruct modalities list (same logic as ssl_forward)
        model_cfg = pl_module.hparams["module"]
        modalities = [
            m["name"] + "_x_ptr" if m["name"] + "_x_ptr" in batch else None
            for m in model_cfg["modalities"]
        ]

        model = pl_module.model
        device = next(model.parameters()).device
        active_mask = active_mask.to(device)
        B, M = active_mask.shape

        was_training = model.training
        model.eval()

        with torch.no_grad():
            # 1. Full embedding (all active modalities)
            full_emb = model(batch, modalities, active_mask, active_mask)

            # 2. Single-modality embeddings
            single_embs = {}
            for i in range(M):
                has_i = active_mask[:, i]
                if has_i.sum() == 0:
                    continue
                one_hot_mask = active_mask.clone()
                one_hot_mask[has_i] = False
                one_hot_mask[has_i, i] = True
                emb = model(batch, modalities, one_hot_mask, active_mask)
                single_embs[i] = emb

            # 3. Leave-one-out embeddings
            loo_embs = {}
            for i in range(M):
                loo_mask = active_mask.clone()
                loo_mask[:, i] = False
                empty = loo_mask.sum(dim=1) == 0
                loo_mask[empty] = active_mask[empty]
                emb = model(batch, modalities, loo_mask, active_mask)
                loo_embs[i] = emb

        if was_training:
            model.train()

        self._results[stage] = {
            "full_emb": full_emb.detach().cpu(),
            "single_embs": {k: v.detach().cpu() for k, v in single_embs.items()},
            "loo_embs": {k: v.detach().cpu() for k, v in loo_embs.items()},
            "active_mask": active_mask.detach().cpu(),
            "M": M,
        }

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._compute_importance(trainer, pl_module, outputs, batch, batch_idx, "fit")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._compute_importance(trainer, pl_module, outputs, batch, batch_idx, "validate")

    def _log_results(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.global_rank != 0 or self._results.get(stage) is None:
            return

        data = self._results[stage]
        M = data["M"]
        active = data["active_mask"]
        full_emb = data["full_emb"]
        single_embs = data["single_embs"]
        loo_embs = data["loo_embs"]
        epoch = trainer.current_epoch

        # --- 1. Pairwise Similarity Matrix (M × M) ---
        sim_matrix = np.full((M, M), np.nan)
        for i in single_embs:
            for j in single_embs:
                both_active = active[:, i] & active[:, j]
                if both_active.sum() == 0:
                    continue
                ei = single_embs[i][both_active]
                ej = single_embs[j][both_active]
                cos = F.cosine_similarity(ei, ej, dim=1).mean().item()
                sim_matrix[i, j] = cos

        fig_sim = self._plot_similarity_matrix(sim_matrix, epoch, stage)

        # --- 2. Leave-One-Out Impact (M bars) ---
        loo_impact = np.full(M, np.nan)
        for i in loo_embs:
            col_active = active[:, i]
            if col_active.sum() == 0:
                continue
            diff = ((full_emb - loo_embs[i]) ** 2).mean(dim=1)
            loo_impact[i] = diff[col_active].mean().item()

        fig_loo = self._plot_loo_impact(loo_impact, epoch, stage)

        # --- Log to TensorBoard ---
        tb_logger = trainer.logger
        if tb_logger is not None and hasattr(tb_logger, "experiment"):
            exp = tb_logger.experiment
            if hasattr(exp, "add_figure"):
                if fig_sim is not None:
                    exp.add_figure(
                        f"{self.name}/{stage}_pairwise_similarity",
                        fig_sim,
                        global_step=epoch,
                    )
                if fig_loo is not None:
                    exp.add_figure(
                        f"{self.name}/{stage}_leave_one_out_impact",
                        fig_loo,
                        global_step=epoch,
                    )

        if fig_sim is not None:
            plt.close(fig_sim)
        if fig_loo is not None:
            plt.close(fig_loo)

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._log_results(trainer, pl_module, "fit")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._log_results(trainer, pl_module, "validate")

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def _mod_label(self, i: int) -> str:
        return self.modality_names[i] if i < len(self.modality_names) else f"mod_{i}"

    def _plot_similarity_matrix(
        self, sim_matrix: np.ndarray, epoch: int, stage: str
    ) -> Optional[plt.Figure]:
        M = sim_matrix.shape[0]
        if np.isnan(sim_matrix).all():
            return None

        labels = [self._mod_label(i) for i in range(M)]

        fig, ax = plt.subplots(figsize=(max(6, M * 0.8), max(5, M * 0.7)))

        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad(color="#d9d9d9")

        im = ax.imshow(sim_matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

        # Annotate cells
        for i in range(M):
            for j in range(M):
                if not np.isnan(sim_matrix[i, j]):
                    val = sim_matrix[i, j]
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=7, color=color,
                    )

        ax.set_xticks(range(M))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(M))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Modality j")
        ax.set_ylabel("Modality i")
        ax.set_title(
            f"[{stage}] Single-Modality Pairwise Cosine Similarity (epoch {epoch})",
            fontsize=10,
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cosine similarity")

        fig.tight_layout()
        return fig

    def _plot_loo_impact(
        self, loo_impact: np.ndarray, epoch: int, stage: str
    ) -> Optional[plt.Figure]:
        M = len(loo_impact)
        if np.isnan(loo_impact).all():
            return None

        labels = [self._mod_label(i) for i in range(M)]

        fig, ax = plt.subplots(figsize=(max(6, M * 0.7), 4))

        valid = ~np.isnan(loo_impact)
        values = np.where(valid, loo_impact, 0.0)
        colors = ["#e74c3c" if v else "#cccccc" for v in valid]

        ax.bar(range(M), values, color=colors, edgecolor="gray", linewidth=0.5)

        # Annotate bars
        ymax = values.max() if valid.any() else 1.0
        for i, (v, is_valid) in enumerate(zip(values, valid)):
            if is_valid:
                ax.text(
                    i, v + ymax * 0.02, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=7,
                )

        ax.set_xticks(range(M))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean L2 distance (full vs. without)")
        ax.set_title(
            f"[{stage}] Leave-One-Out Modality Impact (epoch {epoch})", fontsize=10,
        )

        fig.tight_layout()
        return fig
