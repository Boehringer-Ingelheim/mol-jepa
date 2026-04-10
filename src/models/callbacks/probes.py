import torch
from torch import nn
from lightning import Callback

import stable_pretraining as spt

from models.losses.masked import MaskedMAE, masked_mae_loss


def extract_labels(benchmark_labels):
    extended_labels = []
    column_names = []
    for datset, spec in benchmark_labels.items():
        label_type = spec["type"]
        labels = spec["labels"]

        for label in labels:
            extended_label = f"{datset}_{label_type}_{label}"
            column_names.append(label)
            extended_labels.append(extended_label)

    return column_names, extended_labels


def build_probe(cfg):
    input_dim = cfg.module.probes.input_dim
    output_dim = cfg.module.probes.output_dim
    hidden_dim = cfg.module.probes.hidden_dim
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def build_online_probes(cfg, module):
    _, extended_labels = extract_labels(cfg.data.benchmark_labels)
    callbacks = []

    for label in extended_labels:
        if "regression" in label:
            callbacks.append(
                spt.callbacks.OnlineProbe(
                    module,
                    name=f"probe_{label}",
                    input="embedding_1",
                    target=label,
                    probe=build_probe(cfg),
                    loss=masked_mae_loss,
                    metrics={
                        "mae": MaskedMAE(),
                    },
                    optimizer={"type": "AdamW", "lr": 1e-3, "weight_decay": 1e-4},
                )
            )
        elif "classification" in label:
            raise NotImplementedError(
                "Classification probes not implemented yet. Requires num classes."
            )
    return callbacks


def build_diagnostic_probe_callbacks(cfg, module):
    callbacks = []
    callbacks.append(
        spt.callbacks.RankMe(
            name="effective_rank",
            target="embedding_1",
            queue_length=1000,
            target_shape=cfg.module.moe_encoder.hidden_dim,
        )
    )
    return callbacks


def build_probe_callbacks(cfg, module):
    callbacks = []
    callbacks.extend(build_online_probes(cfg, module))
    callbacks.extend(build_diagnostic_probe_callbacks(cfg, module))
    return callbacks
