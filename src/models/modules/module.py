import torch
from uri_template import partial

import stable_pretraining as spt
from omegaconf import OmegaConf

from models.modules.jepa import MolJEPA, ssl_forward


def get_module(cfg) -> spt.Module:
    module_cfg = OmegaConf.select(cfg, "module")

    model = MolJEPA(
        modalities_spec=module_cfg.modalities,
        labels_spec=module_cfg.labels,
        label_strategy=module_cfg.label_strategy,
        moe_encoder_spec=module_cfg.moe_encoder,
        expert_encoders_spec=module_cfg.expert_encoders,
    )
    optimizer_cfg = cfg["module"]["optimizer"]

    return spt.Module(
        model=model,
        forward=ssl_forward,
        hparams=cfg,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": optimizer_cfg["lr"],
                "weight_decay": optimizer_cfg["weight_decay"],
                "betas": (0.9, 0.95),
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "step",
        },
    )
