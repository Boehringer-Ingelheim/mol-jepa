from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
import stable_pretraining as spt
from callbacks.probes import build_benchmark_probes
from callbacks.umap import UMAPVisualizer
from callbacks.gradnorm import GradientNorm
from callbacks.earlystop import EarlyStopping
from callbacks.hp_metric import HPMetricLogger
from callbacks.threshstop import ThresholdStopping


def build_diagnostic_callbacks(cfg, module, input_key="embedding_cls"):
    callbacks = [
        # UMAPVisualizer(
        #     name="umap_LogD",
        #     embedding_key=input_key,
        #     label_key="asap_admet_regression_LogD",
        #     queue_length=300,
        #     target_shape=cfg.module.moe_encoder.hidden_dim,
        #     label_value=None,
        #     log_every_n_epochs=1,
        #     max_points=300,
        #     gather_distributed=True,
        # ),
        # GradientNorm(norm_type=2, log_on_step=True),
        spt.callbacks.RankMe(
            name="effective_rank",
            target=input_key,
            queue_length=1000,
            verbose=False,
            target_shape=cfg.module.moe_encoder.hidden_dim,
        ),
    ]

    # Modality-specific ranks
    callbacks += [
        spt.callbacks.RankMe(
            name=f"effective_rank_{m['name']}",
            target=f"embedding_{m['name']}",
            queue_length=1000,
            verbose=False,
            target_shape=cfg.module.moe_encoder.hidden_dim,
        )
        for m in cfg.module.modalities
    ]
    return callbacks


def build_hp_callbacks():
    callbacks = [
        HPMetricLogger(metric_name="eval/probes_mean"),
        EarlyStopping(
            metric_name="eval/probes_mean",
            patience=30,
            mode="min",
        ),
        ThresholdStopping(
            metric_name="eval/probes_mean",
            threshold=0.8,
            epoch_checkpoint=40,
            mode="min",
        ),
        ThresholdStopping(
            metric_name="effective_rank",
            threshold=50,
            epoch_checkpoint=50,
            mode="max",
        ),
    ]
    return callbacks


def is_hp_tuning() -> bool:
    try:
        return HydraConfig.get().mode == RunMode.MULTIRUN
    except Exception:
        return False 


def build_callbacks(cfg, module):
    callbacks = []
    callbacks += build_benchmark_probes(cfg, module)
    callbacks += build_diagnostic_callbacks(cfg, module, input_key="embeddings_cls")
    
    if is_hp_tuning():
        callbacks += build_hp_callbacks()
    return callbacks
