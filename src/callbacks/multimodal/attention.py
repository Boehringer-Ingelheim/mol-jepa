
from __future__ import annotations

import random
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer


class AttentionWeightLogger(Callback):
    """Log per-modality attention weight heatmaps to TensorBoard.

    Args:
        name:               TensorBoard tag prefix.
        modality_names:     Ordered list of human-readable modality names.
        log_every_n_epochs: Only produce the plot every N validation epochs.
        max_samples:        Cap the number of samples shown in the heatmap
                            (rows are randomly subsampled if the batch is larger).
    """

    def __init__(
        self,
        name: str = "attn_weights",
        modality_names: Optional[List[str]] = None,
        log_every_n_epochs: int = 1,
        max_samples: int = 64,
    ) -> None:
        super().__init__()
        self.name = name
        self.modality_names = modality_names or []
        self.log_every_n_epochs = log_every_n_epochs
        self.max_samples = max_samples

        # Reservoir-sampling state per stage (reset each epoch)
        self._candidates: dict[str, Optional[dict]] = {"fit": None, "validate": None}
        self._batch_counters: dict[str, int] = {"fit": 0, "validate": 0}

    @property
    def state_key(self) -> str:
        return f"AttentionWeightLogger[name={self.name}]"

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
        """Reservoir-sample one batch for *stage*."""
        if not isinstance(outputs, dict):
            return
        if outputs.get("stage") != stage:
            return

        attn_1 = outputs.get("attn_weights_1")
        attn_2 = outputs.get("attn_weights_2")
        if attn_1 is None or attn_2 is None:
            return

        self._batch_counters[stage] += 1
        cnt = self._batch_counters[stage]

        if self._candidates[stage] is None or random.random() < 1.0 / cnt:
            self._candidates[stage] = {
                "attn_weights_1": attn_1.detach().cpu(),
                "attn_weights_2": attn_2.detach().cpu(),
                "active_mask": outputs["active_mask"].detach().cpu(),
                "pass_1_mask": outputs["pass_1_mask"].detach().cpu(),
                "pass_2_mask": outputs["pass_2_mask"].detach().cpu(),
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

    def _log_epoch_end(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.global_rank != 0:
            return

        epoch = trainer.current_epoch
        if self.log_every_n_epochs > 1 and epoch % self.log_every_n_epochs != 0:
            return

        data = self._candidates.get(stage)
        if data is None:
            return

        # --- Log per-modality mean attention (pass 2, scalar) -------------
        self._log_mean_attention(trainer, pl_module, data, stage)

        # --- Plot heatmaps ------------------------------------------------
        for pass_tag, w_key, m_key in [
            ("pass_1", "attn_weights_1", "pass_1_mask"),
            ("pass_2", "attn_weights_2", "pass_2_mask"),
        ]:
            fig = self._make_heatmap(
                attn_weights=data[w_key],
                active_mask=data["active_mask"],
                pass_mask=data[m_key],
                title=f"{self.name}/{stage}_{pass_tag}  (epoch {epoch})",
            )
            if fig is None:
                continue

            tb_logger = trainer.logger
            if tb_logger is not None and hasattr(tb_logger, "experiment"):
                exp = tb_logger.experiment
                if hasattr(exp, "add_figure"):
                    exp.add_figure(
                        f"{self.name}/{stage}_{pass_tag}",
                        fig,
                        global_step=trainer.current_epoch,
                    )
            plt.close(fig)

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._log_epoch_end(trainer, pl_module, "fit")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._log_epoch_end(trainer, pl_module, "validate")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _log_mean_attention(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        data: dict,
        stage: str,
    ) -> None:
        """Log per-modality mean attention weight (pass 2) as scalars."""
        attn = data["attn_weights_2"]          # (B, M)
        active = data["active_mask"]           # (B, M) bool
        pass_mask = data["pass_2_mask"]        # (B, M) bool

        # Only consider cells that are both active *and* in this pass
        valid = active & pass_mask             # (B, M) bool
        attn_masked = attn * valid.float()

        # Mean over samples that actually have the modality
        counts = valid.sum(dim=0).clamp(min=1)  # (M,)
        mean_attn = attn_masked.sum(dim=0) / counts  # (M,)

        for i, w in enumerate(mean_attn.tolist()):
            mod_name = self.modality_names[i] if i < len(self.modality_names) else f"mod_{i}"
            pl_module.log(
                f"{self.name}/{stage}_mean_{mod_name}",
                w,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

    def _make_heatmap(
        self,
        attn_weights: torch.Tensor,
        active_mask: torch.Tensor,
        pass_mask: torch.Tensor,
        title: str,
    ) -> Optional[plt.Figure]:
        """Build a (samples × modalities) heatmap with hatched inactive cells."""
        B, M = attn_weights.shape

        # Subsample rows if the batch is too large for a readable plot
        if B > self.max_samples:
            idx = torch.randperm(B)[: self.max_samples].sort().values
            attn_weights = attn_weights[idx]
            active_mask = active_mask[idx]
            pass_mask = pass_mask[idx]
            B = self.max_samples

        attn_np = attn_weights.numpy()
        active_np = active_mask.numpy().astype(bool)
        pass_np = pass_mask.numpy().astype(bool)

        # Build display matrix: real weights where (active & in pass), NaN elsewhere
        display = np.full_like(attn_np, np.nan)
        in_pass = active_np & pass_np
        display[in_pass] = attn_np[in_pass]

        # Determine color limits from valid cells only
        valid_vals = display[~np.isnan(display)]
        if valid_vals.size == 0:
            return None
        vmin, vmax = float(valid_vals.min()), float(valid_vals.max())
        if vmin == vmax:
            vmax = vmin + 1e-6

        # --- Figure ---
        fig_width = max(6, M * 0.7 + 2)
        fig_height = max(4, B * 0.25 + 2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # 1) Draw the coloured heatmap (NaN cells stay transparent)
        cmap = plt.cm.Blues.copy()
        cmap.set_bad(color="none")  # transparent for NaN
        im = ax.imshow(
            display,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        # 2) Overlay patches for inactive and active-but-not-in-pass cells
        nan_color = "#d9d9d9"
        masked_color = "#C56A6A"
        for row in range(B):
            for col in range(M):
                if not active_np[row, col]:
                    # Inactive modality → gray hatched
                    rect = plt.Rectangle(
                        (col - 0.5, row - 0.5), 1, 1,
                        linewidth=0.5,
                        edgecolor="gray",
                        facecolor=nan_color,
                        hatch="//",
                    )
                    ax.add_patch(rect)
                elif not pass_np[row, col]:
                    # Active but not in this pass → masked color
                    rect = plt.Rectangle(
                        (col - 0.5, row - 0.5), 1, 1,
                        linewidth=0.5,
                        edgecolor="gray",
                        facecolor=masked_color,
                        alpha=0.6,
                    )
                    ax.add_patch(rect)

        # 3) Annotate values for in-pass cells (skip if too many to be readable)
        if B * M <= 300:
            for row in range(B):
                for col in range(M):
                    if in_pass[row, col]:
                        val = attn_np[row, col]
                        text_color = "white" if val > (vmin + vmax) / 2 else "black"
                        ax.text(
                            col, row, f"{val:.2f}",
                            ha="center", va="center",
                            fontsize=6, color=text_color,
                        )

        # --- Axes ---
        mod_labels = [
            self.modality_names[i] if i < len(self.modality_names) else f"mod_{i}"
            for i in range(M)
        ]
        ax.set_xticks(range(M))
        ax.set_xticklabels(mod_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Sample index")
        ax.set_title(title, fontsize=10)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention weight")

        # Legend
        legend_handles = [
            mpatches.Patch(facecolor=nan_color, edgecolor="gray", hatch="//", label="No data"),
            mpatches.Patch(facecolor=masked_color, edgecolor="gray", alpha=0.6, label="Masked"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.15, 1.0),
            fontsize=7,
            frameon=True,
        )

        fig.tight_layout()
        return fig
