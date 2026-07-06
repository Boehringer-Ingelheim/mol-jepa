from typing import Union, Iterable

import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from stable_pretraining.callbacks.queue import find_or_create_queue_callback


class CosineSimilarity(Callback):
    """Cosine similarity monitor for self-supervised learning embeddings.

    Supports two modes:
    - 'passes': mean cosine similarity between two forward passes (e.g. two masked views)
    - 'batch': mean pairwise cosine similarity within a single batch (excluding self-pairs)

    Args:
        name: Unique name for this callback instance
        source_1: Key in batch dict for the first embedding
        source_2: Key in batch dict for the second embedding (ignored in 'batch' mode)
        queue_length: Required queue length
        target_shape: Embedding dimensionality
        mode: One of 'passes' or 'batch'
    """

    MODES = ("passes", "batch")

    def __init__(
        self,
        name: str,
        source_1: str,
        source_2: str,
        queue_length: int,
        target_shape: Union[int, Iterable[int]],
        mode: str = "passes",
    ) -> None:
        super().__init__()

        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got '{mode}'")

        if isinstance(target_shape, (list, tuple)):
            target_shape = (
                target_shape[0]
                if len(target_shape) == 1
                else int(torch.prod(torch.tensor(target_shape)))
            )

        self.name = name
        self.source_1 = source_1
        self.source_2 = source_2
        self.queue_length = queue_length
        self.target_shape = target_shape
        self.mode = mode

        self._queue_1 = None
        self._queue_2 = None

    @property
    def state_key(self) -> str:
        return f"CosineSimilarity[name={self.name}]"

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Find or create queue callbacks for both embedding sources."""
        if self._queue_1 is None:
            self._queue_1 = find_or_create_queue_callback(
                trainer,
                self.source_1,
                self.queue_length,
                self.target_shape,
                torch.float32,
                gather_distributed=True,
                create_if_missing=True,
            )
            logging.info(f"{self.name}: Using queue for source_1 '{self.source_1}'")

        if self._queue_2 is None:
            self._queue_2 = find_or_create_queue_callback(
                trainer,
                self.source_2,
                self.queue_length,
                self.target_shape,
                torch.float32,
                gather_distributed=True,
                create_if_missing=True,
            )
            logging.info(f"{self.name}: Using queue for source_2 '{self.source_2}'")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute cosine similarity on the first validation batch only."""
        if batch_idx > 0 or trainer.global_rank != 0:
            return

        logging.info(
            f"{self.name}: Computing cosine similarity (mode='{self.mode}') on first validation batch"
        )

        e1 = self._queue_1.data
        e2 = self._queue_2.data

        for tag, emb in (("source_1", e1), ("source_2", e2)):
            if emb is None or emb.numel() == 0:
                logging.warning(
                    f"{self.name}: Queue '{tag}' is empty or unavailable, skipping"
                )
                return

        with torch.no_grad():
            if self.mode == "passes":
                similarity = F.cosine_similarity(e1, e2, dim=1).mean()

            elif self.mode == "batch":
                # Normalise rows, then gram matrix gives all pairwise cosine sims.
                e1_norm = F.normalize(e1, dim=1)  # (N, D)
                gram = e1_norm @ e1_norm.T  # (N, N)
                # Exclude the diagonal (self-similarity = 1) to avoid bias.
                mask = ~torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)
                similarity = gram[mask].mean()

        pl_module.log(self.name, similarity.item())
