import hydra
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
from models.modules.module import get_module
from models.callbacks.probes import build_probe_callbacks
from models.callbacks.cosine import CosineSimilarity
from models.callbacks.earlystop import EarlyStopping
from models.callbacks.gradnorm import GradientNormLogger
from models.callbacks.umap import UMAPEmbeddingLogger

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
    pl.seed_everything(cfg.seed, workers=True)

    logger.info("Loading dataset...")
    data = get_data_module(cfg)

    logger.info("Loading model...")
    module = get_module(cfg)

    logger.info("Initializing trainer...")
    trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.callbacks = filter_callbacks(trainer.callbacks)
    trainer.callbacks += build_probe_callbacks(cfg, module)
    trainer.callbacks += [
        CosineSimilarity(
            name="cosine_sim_passes",
            source_1="embedding_1",
            source_2="embedding_2",
            queue_length=1024,
            target_shape=cfg.module.moe_encoder.hidden_dim,
            mode="passes",
        ),
        CosineSimilarity(
            name="cosine_sim_batch",
            source_1="embedding_1",
            source_2="embedding_2",
            queue_length=1024,
            target_shape=cfg.module.moe_encoder.hidden_dim,
            mode="batch",
        ),
        GradientNormLogger(norm_type=2, log_on_step=True),
        EarlyStopping(metric_name="validate_loss", patience=200, mode="min"),
        UMAPEmbeddingLogger(
            name="umap_KSOL",
            embedding_key="embedding_1",
            label_key="asap_admet_regression_KSOL",
            queue_length=300,
            target_shape=cfg.module.moe_encoder.hidden_dim,
            label_value=None,
            log_every_n_epochs=1,
            max_points=300,
            gather_distributed=True,
        ),
        UMAPEmbeddingLogger(
            name="umap_LogD",
            embedding_key="embedding_1",
            label_key="asap_admet_regression_LogD",
            queue_length=300,
            target_shape=cfg.module.moe_encoder.hidden_dim,
            label_value=None,
            log_every_n_epochs=1,
            max_points=300,
            gather_distributed=True,
        ),
    ]
    logger.info("Added callbacks...")

    manager = spt.Manager(trainer=trainer, module=module, data=data, seed=cfg.seed)
    # Start training
    manager()

    # Return for HP search
    best_loss = trainer.callback_metrics["eval/probe_expansionrx_regression_LogD_mae"]
    return best_loss.item() if best_loss is not None else None


if __name__ == "__main__":
    main()
