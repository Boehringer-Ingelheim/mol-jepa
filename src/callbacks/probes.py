import math
from typing import List
import torch
from torch import nn
from lightning.pytorch import Callback, LightningModule, Trainer

import stable_pretraining as spt

from models.losses.masked import MaskedMAE, masked_mae_loss
from callbacks.probe_utils import (
    ProbeScatterPlot,
    ModalityLossBarPlot,
    MultimodalWeightLogger,
    BestPredictionSaver,
)


class MultimodalLinearProbe(nn.Module):
    """
    A simple linear probe that takes multiple predicted modalities as inputs
    and computes a weighted sum of their embeddings before applying a linear layer.
    """

    def __init__(self, input_dim, n_modalities, output_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_modalities) / n_modalities)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):  # x: [B, n_mods, D]
        w = torch.softmax(self.weights, dim=0)
        pooled = (x * w[None, :, None]).sum(dim=1)  # [B, D]
        return self.linear(pooled)

    @property
    def importance_scores(self):
        return torch.softmax(self.weights, dim=0).detach()


class MultimodalNonlinearProbe(nn.Module):
    """
    A nonlinear probe that takes multiple predicted modalities as inputs,
    computes a weighted sum of their embeddings, and applies a nonlinear transformation.
    """

    def __init__(self, input_dim, hidden_dim, n_modalities, output_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_modalities) / n_modalities)
        self.out = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):  # x: [B, n_mods, D]
        w = torch.softmax(self.weights, dim=0)
        pooled = (x * w[None, :, None]).sum(dim=1)  # [B, D]
        return self.out(pooled)

    @property
    def importance_scores(self):
        return torch.softmax(self.weights, dim=0).detach()


class TransformerProbe(nn.Module):
    """
    A transformer-based probe that takes multiple predicted modalities as inputs,
    computes a weighted sum of their embeddings, and applies a transformer encoder
    followed by a linear layer to produce the final output.
    """

    def __init__(
        self, n_tokens, token_dim, output_dim=1, n_layers=2, n_heads=4, dropout=0.1
    ):
        super().__init__()
        # token = modality
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.modality_emb = nn.Parameter(torch.zeros(1, n_tokens, token_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.modality_emb, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(token_dim)
        self.head = nn.Linear(token_dim, output_dim)

    def forward(self, x):
        # x: (B, n_tokens, token_dim) or (B, n_tokens * token_dim)
        if x.dim() == 2:
            x = x.view(-1, self.n_tokens, self.token_dim)
        B = x.size(0)
        # Mark zero-imputed (= missing) modality tokens BEFORE adding modality embeddings.
        token_pad_mask = x.abs().sum(dim=-1) == 0  # (B, n_tokens), True = ignore
        x = x + self.modality_emb
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1 + n_tokens, D)
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([cls_pad, token_pad_mask], dim=1)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        cls_out = self.norm(x[:, 0])
        return self.head(cls_out)


def build_nonlinear_probe(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def build_linear_probe(input_dim, output_dim):
    return nn.Linear(input_dim, output_dim)


def build_linear_multimodal_probe(input_dim, output_dim):
    return nn.Linear(input_dim, output_dim)


class ProbeMean(Callback):
    """Log the mean MAE across a set of OnlineProbe callbacks.

    Reads ``eval/{probe_name}_mae`` (and optionally ``train/{probe_name}_mae``)
    from ``trainer.callback_metrics`` at epoch end and logs their average.
    """

    def __init__(self, probe_names: List[str], name: str = "probes_mean") -> None:
        super().__init__()
        self.probe_names = probe_names
        self.name = name

    @property
    def state_key(self) -> str:
        return f"ProbeMean[name={self.name}]"

    def _log_mean(self, trainer, pl_module, stage):
        prefix = "eval" if stage == "validate" else "train"
        metrics = trainer.callback_metrics

        values = []
        for pname in self.probe_names:
            key = f"{prefix}/{pname}_mae"
            if key in metrics:
                v = metrics[key]
                v = v.item() if hasattr(v, "item") else float(v)
                if not math.isnan(v):
                    values.append(v)

        if not values:
            return

        mean_val = sum(values) / len(values)
        pl_module.log(
            f"{prefix}/{self.name}",
            mean_val,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=1,
        )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_mean(trainer, pl_module, "fit")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._log_mean(trainer, pl_module, "validate")


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


def build_benchmark_probes(cfg, module):
    # Extract benchmark names and other properties
    _, extended_labels = extract_labels(cfg.data.benchmark_labels)
    hidden_dim = cfg.module.moe_encoder.hidden_dim
    n_modalities = len(cfg.data.modalities)
    modality_names = [m["name"] for m in cfg.data.modalities]
    probes = cfg.module.probes

    callbacks = []
    agg_linear_probe_names = []
    agg_nonlinear_probe_names = []
    agg_multimodal_linear_probe_names = []
    agg_multimodal_nonlinear_probe_names = []
    agg_multimodal_transformer_probe_names = []
    agg_linear_target_keys = []
    agg_nonlinear_target_keys = []
    agg_multimodal_linear_target_keys = []
    agg_multimodal_nonlinear_target_keys = []
    agg_multimodal_transformer_target_keys = []
    for probe in probes:
        probe_name = probe.name
        variant = probe.variant
        multimodal = probe.multimodal
        output_dim = probe.output_dim
        input_key = "embeddings" if multimodal else "embeddings_cls"

        for label in extended_labels:
            if "regression" in label:
                name = f"probe_{probe_name}_{label}"
                if variant == "linear":
                    if multimodal:
                        probe_model = MultimodalLinearProbe(
                            hidden_dim, n_modalities, output_dim
                        )
                        agg_multimodal_linear_probe_names.append(name)
                        agg_multimodal_linear_target_keys.append(label)
                    else:
                        probe_model = build_linear_probe(hidden_dim, output_dim)

                        agg_linear_probe_names.append(name)
                        agg_linear_target_keys.append(label)
                elif variant == "nonlinear":
                    if multimodal:
                        probe_model = MultimodalNonlinearProbe(
                            hidden_dim, probe.hidden_dim, n_modalities, output_dim
                        )
                        agg_multimodal_nonlinear_probe_names.append(name)
                        agg_multimodal_nonlinear_target_keys.append(label)
                    else:
                        probe_model = build_nonlinear_probe(
                            hidden_dim, probe.hidden_dim, output_dim
                        )
                        agg_nonlinear_probe_names.append(name)
                        agg_nonlinear_target_keys.append(label)
                elif variant == "transformer":
                    if multimodal:
                        probe_model = TransformerProbe(
                            n_tokens=n_modalities,
                            token_dim=hidden_dim,
                            output_dim=output_dim,
                            n_layers=2,
                            n_heads=4,
                        )
                        agg_multimodal_transformer_probe_names.append(name)
                        agg_multimodal_transformer_target_keys.append(label)
                    else:
                        raise ValueError(
                            "Transformer probe only supports multimodal=true"
                        )

                callbacks.append(
                    spt.callbacks.OnlineProbe(
                        module,
                        name=name,
                        input=input_key,
                        target=label,
                        probe=probe_model,
                        loss=masked_mae_loss,
                        metrics={
                            "mae": MaskedMAE(),
                        },
                        optimizer={"type": "AdamW", "lr": 1e-3, "weight_decay": 1e-4},
                        scheduler={"type": "CosineAnnealingLR", "T_max": 100},
                        log_on="epoch",
                    )
                )
            elif "classification" in label:
                raise NotImplementedError(
                    "Classification probes not implemented yet. Requires num classes."
                )

    # Aggregative callbacks
    callbacks.append(
        ProbeMean(probe_names=agg_linear_probe_names, name="probes_linear_mean")
    )
    callbacks.append(
        ProbeMean(probe_names=agg_nonlinear_probe_names, name="probes_nonlinear_mean")
    )
    callbacks.append(
        ProbeMean(
            probe_names=agg_multimodal_linear_probe_names,
            name="probes_multimodal_linear_mean",
        )
    )
    callbacks.append(
        ProbeMean(
            probe_names=agg_multimodal_nonlinear_probe_names,
            name="probes_multimodal_nonlinear_mean",
        )
    )
    callbacks.append(
        ProbeMean(
            probe_names=agg_multimodal_transformer_probe_names,
            name="probes_multimodal_transformer_mean",
        )
    )

    # Weight logger for multimodal probes
    callbacks.append(
        MultimodalWeightLogger(
            probe_names=agg_multimodal_linear_probe_names,
            modality_names=modality_names,
            name="probe_multimodal_linear_importance",
        )
    )
    callbacks.append(
        MultimodalWeightLogger(
            probe_names=agg_multimodal_nonlinear_probe_names,
            modality_names=modality_names,
            name="probe_multimodal_nonlinear_importance",
        )
    )

    # Scatter plots
    callbacks.append(
        ProbeScatterPlot(agg_linear_probe_names, agg_linear_target_keys, name="linear")
    )
    callbacks.append(
        ProbeScatterPlot(
            agg_nonlinear_probe_names, agg_nonlinear_target_keys, name="nonlinear"
        )
    )
    callbacks.append(
        ProbeScatterPlot(
            agg_multimodal_linear_probe_names,
            agg_multimodal_linear_target_keys,
            name="multimodal_linear",
        )
    )
    callbacks.append(
        ProbeScatterPlot(
            agg_multimodal_nonlinear_probe_names,
            agg_multimodal_nonlinear_target_keys,
            name="multimodal_nonlinear",
        )
    )
    callbacks.append(
        ProbeScatterPlot(
            agg_multimodal_transformer_probe_names,
            agg_multimodal_transformer_target_keys,
            name="multimodal_transformer",
        )
    )

    # Save predictions on best validate_loss
    all_probe_names = (
        agg_linear_probe_names
        + agg_nonlinear_probe_names
        + agg_multimodal_linear_probe_names
        + agg_multimodal_nonlinear_probe_names
        + agg_multimodal_transformer_probe_names
    )
    all_target_keys = (
        agg_linear_target_keys
        + agg_nonlinear_target_keys
        + agg_multimodal_linear_target_keys
        + agg_multimodal_nonlinear_target_keys
        + agg_multimodal_transformer_target_keys
    )
    callbacks.append(
        BestPredictionSaver(
            probe_names=all_probe_names,
            target_keys=all_target_keys,
            monitor="validate_loss",
        )
    )

    modality_names = [m["name"] for m in cfg.data.modalities]
    callbacks.append(
        ModalityLossBarPlot(
            modality_names=modality_names,
            stage="validate",
            name="modality_losses",
        )
    )

    return callbacks
