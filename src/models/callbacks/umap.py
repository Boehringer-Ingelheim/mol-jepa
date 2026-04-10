from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union, Iterable

import torch
import logging
from lightning.pytorch import Callback, LightningModule, Trainer

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class UMAPConfig:
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 42
    n_components: int = 2  # we log a 2D scatter


class UMAPEmbeddingLogger(Callback):
    """UMAP visualization for embeddings, logged to TensorBoard.

    Features:
      - Maintains an internal FIFO queue of (embeddings, labels)
      - Optionally filters which samples are queued by a target label value
      - Fits UMAP on queued embeddings on validation
      - Logs a scatter plot colored by label to TensorBoard

    Args:
        name: Metric/plot name for logging (TensorBoard tag)
        embedding_key: Key to read embeddings from. (defaults to outputs first, then batch)
        label_key: Key to read labels from. (defaults to batch)
        queue_length: Max number of samples to keep in the queue
        target_shape: Embedding dimensionality (int or iterable)
        label_value: If not None, only samples with (labels == label_value) are queued
        umap: UMAP hyperparameters
        log_every_n_epochs: Only log every N val epochs (default: 1)
        max_points: If not None, subsample queue to at most this many points for UMAP speed
        gather_distributed: If True and DDP is initialized, gather data across ranks (safe with filtering)
    """

    def __init__(
        self,
        name: str,
        embedding_key: str,
        label_key: str,
        queue_length: int,
        target_shape: Union[int, Iterable[int]],
        label_value: Optional[Union[int, float]] = None,
        umap: Optional[UMAPConfig] = None,
        log_every_n_epochs: int = 1,
        max_points: Optional[int] = 5000,
        gather_distributed: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(target_shape, (list, tuple)):
            target_shape = (
                target_shape[0]
                if len(target_shape) == 1
                else int(torch.prod(torch.tensor(target_shape)))
            )

        self.name = name
        self.embedding_key = embedding_key
        self.label_key = label_key
        self.queue_length = int(queue_length)
        self.target_shape = int(target_shape)

        self.label_value = label_value
        self.umap_cfg = umap or UMAPConfig()
        self.log_every_n_epochs = int(log_every_n_epochs)
        self.max_points = max_points
        self.gather_distributed = gather_distributed

        # Internal CPU queues
        self._emb_queue: Optional[torch.Tensor] = None  # (N, D) on CPU
        self._lab_queue: Optional[torch.Tensor] = None  # (N,) on CPU
        self._last_logged_epoch: int = -1

    @property
    def state_key(self) -> str:
        return f"UMAPEmbeddingLogger[name={self.name}]"

    # ---------- helpers ----------
    def _get_from_outputs_or_batch(
        self, outputs: Any, batch: Any, key: str
    ) -> Optional[torch.Tensor]:
        if isinstance(outputs, dict) and key in outputs:
            return outputs[key]
        if isinstance(batch, dict) and key in batch:
            return batch[key]
        return None

    def _append_to_queue(self, emb_cpu: torch.Tensor, lab_cpu: torch.Tensor) -> None:
        """Append to FIFO queue and truncate to queue_length."""
        if emb_cpu.numel() == 0:
            return

        if self._emb_queue is None:
            self._emb_queue = emb_cpu
            self._lab_queue = lab_cpu
        else:
            self._emb_queue = torch.cat([self._emb_queue, emb_cpu], dim=0)
            self._lab_queue = torch.cat([self._lab_queue, lab_cpu], dim=0)

        # Truncate FIFO (keep last queue_length)
        if self._emb_queue.shape[0] > self.queue_length:
            excess = self._emb_queue.shape[0] - self.queue_length
            self._emb_queue = self._emb_queue[excess:]
            self._lab_queue = self._lab_queue[excess:]

    def _gather_across_ranks(
        self, emb_cpu: torch.Tensor, lab_cpu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather variable-length filtered tensors across ranks using all_gather_object."""
        if not self.gather_distributed:
            return emb_cpu, lab_cpu

        try:
            import torch.distributed as dist
        except Exception:
            return emb_cpu, lab_cpu

        if not (dist.is_available() and dist.is_initialized()):
            return emb_cpu, lab_cpu

        world = dist.get_world_size()
        gathered = [None for _ in range(world)]
        dist.all_gather_object(gathered, (emb_cpu, lab_cpu))

        emb_list, lab_list = [], []
        for item in gathered:
            if item is None:
                continue
            e, l = item
            if e is None or e.numel() == 0:
                continue
            emb_list.append(e)
            lab_list.append(l)

        if len(emb_list) == 0:
            return emb_cpu[:0], lab_cpu[:0]

        return torch.cat(emb_list, dim=0), torch.cat(lab_list, dim=0)

    # ---------- Lightning hooks ----------
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Collect on all ranks (we gather later)
        emb = self._get_from_outputs_or_batch(outputs, batch, self.embedding_key)
        lab = self._get_from_outputs_or_batch(outputs, batch, self.label_key)

        if emb is None or lab is None:
            return

        # Flatten embeddings to (B, D) if needed
        if emb.dim() > 2:
            emb = emb.view(emb.shape[0], -1)

        if emb.shape[1] != self.target_shape:
            raise ValueError(
                f"{self.name}: embedding shape mismatch. "
                f"Expected D={self.target_shape}, got {emb.shape} for key '{self.embedding_key}'."
            )

        # Ensure labels are 1D
        if lab.dim() > 1:
            lab = lab.view(lab.shape[0])

        # Filter by label_value if enabled
        if self.label_value is not None:
            keep = lab == self.label_value
            emb = emb[keep]
            lab = lab[keep]

        lab_float = lab.float()
        keep = ~torch.isnan(lab_float)
        emb = emb[keep]
        lab = lab[keep]

        # Move to CPU
        emb_cpu = emb.detach().float().cpu()
        lab_cpu = lab.detach().cpu()

        # Gather across ranks
        emb_cpu, lab_cpu = self._gather_across_ranks(emb_cpu, lab_cpu)

        # Only rank 0 updates queue
        if trainer.global_rank == 0:
            self._append_to_queue(emb_cpu, lab_cpu)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.global_rank != 0:
            return

        epoch = trainer.current_epoch
        if self.log_every_n_epochs > 1 and (epoch % self.log_every_n_epochs != 0):
            return
        if epoch == self._last_logged_epoch:
            return

        if self._emb_queue is None or self._emb_queue.numel() == 0:
            return
        if self._emb_queue.shape[0] < 10:
            return

        emb = self._emb_queue
        lab = self._lab_queue

        # Subsample if needed
        if self.max_points is not None and emb.shape[0] > self.max_points:
            idx = torch.randperm(emb.shape[0])[: self.max_points]
            emb = emb[idx]
            lab = lab[idx]

        # Fit UMAP
        try:
            import umap
        except ImportError as e:
            raise ImportError(
                "UMAPEmbeddingLogger requires `umap-learn`.\n"
                "Install with: pip install umap-learn"
            ) from e

        reducer = umap.UMAP(
            n_neighbors=self.umap_cfg.n_neighbors,
            min_dist=self.umap_cfg.min_dist,
            metric=self.umap_cfg.metric,
            n_components=self.umap_cfg.n_components,
            random_state=self.umap_cfg.random_state,
        )

        with torch.no_grad():
            emb_np = emb.numpy()
            lab_np = lab.numpy()

        if np.isnan(emb_np).all():
            logging.warning(
                f"{self.name}: All embeddings are NaN, skipping UMAP for epoch {epoch}."
            )
            return

        coords = reducer.fit_transform(emb_np)

        fig, ax = plt.subplots(figsize=(7, 6))

        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=lab_np,
            cmap="coolwarm",
            s=200,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5
        )

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(self.label_key)

        fig.tight_layout()

        logger = trainer.logger
        if logger is not None and hasattr(logger, "experiment"):
            exp = logger.experiment
            if hasattr(exp, "add_figure"):
                exp.add_figure(self.name, fig, global_step=trainer.global_step)

        plt.close(fig)

        self._last_logged_epoch = epoch