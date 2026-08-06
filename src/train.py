import hydra
import torch
import numpy as np
import logging
import lightning as pl
from omegaconf import DictConfig
import stable_pretraining as spt
from stable_pretraining.callbacks.unused_parameters import LogUnusedParametersOnce

from stable_pretraining.callbacks.factories import (
    EnvironmentDumpCallback,
    SklearnCheckpoint,
    WandbCheckpoint,
)

from data.module import get_data_module
from callbacks.utils import build_callbacks
from models.modules.module import get_module


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def filter_callbacks(callbacks):
    return [
        cb
        for cb in callbacks
        if not isinstance(
            cb,
            (
                EnvironmentDumpCallback,
                SklearnCheckpoint,
                WandbCheckpoint,
                LogUnusedParametersOnce,
            ),
        )
    ]


@hydra.main(config_path="config", config_name="moljepa", version_base="1.1")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(cfg.seed, workers=True)

    logger.info("Loading dataset...")
    data = get_data_module(cfg)

    logger.info("Loading model...")
    module = get_module(cfg)

    logger.info("Initializing trainer...")
    trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.callbacks = filter_callbacks(trainer.callbacks)
    trainer.callbacks += build_callbacks(cfg, module)

    # Start training
    manager = spt.Manager(trainer=trainer, module=module, data=data, seed=cfg.seed)
    manager()

    # Return for HP search
    if hasattr(module, "hp_metric"):
        result = module.hp_metric.item()
        if np.isnan(result):
            logger.warning("HP Metric is NaN, returning inf for optimization.")
            result = float("inf")
        logger.info(f"HP Metric: {result}")
        return result


if __name__ == "__main__":
    main()
