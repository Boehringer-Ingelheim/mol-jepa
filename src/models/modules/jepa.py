from unicodedata import name

import torch
import logging
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Dict
from omegaconf import ListConfig

from models.modules.atoms_encoder import AtomsEncoder
from models.modules.emb_encoder import EmbEncoder
from models.modules.graph_encoder import GraphEncoder
from models.losses.sigreg import SlicedEppsPulley
from models.modules.prediction_head import MultiModalPredictor


class MolJEPA(nn.Module):
    """Molecular Joint Embedding Predictive Architecture"""

    def __init__(
        self,
        modalities_spec: Union[list, ListConfig],
        labels_spec: Union[list, ListConfig],
        moe_encoder_spec: dict,
        expert_encoders_spec: dict,
        train_config: dict,
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.modalities_spec = modalities_spec
        self.labels_spec = labels_spec
        self.moe_encoder_spec = moe_encoder_spec
        self.expert_encoders_spec = expert_encoders_spec
        self.label_strategy = train_config["label_strategy"]
        self.weighted_loss = train_config["weighted_loss"]
        self.loss_projection = train_config["loss_projection"]
        self.cls_sigreg = train_config["cls_sigreg"]
        self.cls_predictor = train_config["cls_predictor"]
        self.hidden_dim = moe_encoder_spec["hidden_dim"]

        # Init components
        if self.weighted_loss:
            self.weight_map = {w["name"]: w["weight"] for w in train_config["weights"]}
        self.apply_label_strategy()
        self.modalities_dict = {mod["name"]: mod for mod in self.modalities_spec}
        self.sigreg = SlicedEppsPulley(num_slices=1024, t_max=3.0, n_points=17)
        self._init_encoders()

        # Cross attention prediction head
        n_modalities = len(self.modalities_spec)
        self.transformer_head = MultiModalPredictor(
            hidden_dim=self.hidden_dim,
            n_heads=moe_encoder_spec["attn_heads"],
            n_layers=moe_encoder_spec["attn_layers"],
            n_modalities=n_modalities + 1,  # +1 for cls token
            dropout=moe_encoder_spec["dropout"],
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        if self.loss_projection:
            self.loss_proj = nn.ModuleList(
                [nn.Linear(self.hidden_dim, 128) for _ in self.modalities_dict]
            )

    def apply_label_strategy(self):
        """Add labels either as modalities or labels with prediction heads."""
        if self.label_strategy == "modality":
            for label_spec in self.labels_spec:
                self.modalities_spec.append(
                    {
                        "name": label_spec["name"],
                        "input": "precomputed",
                        "output": "embedding",
                        "dim": label_spec["dim"],
                    }
                )
        elif self.label_strategy == "label":
            self.prediction_heads = nn.ModuleDict()
            for label_spec in self.labels_spec:
                self.prediction_heads[label_spec["name"]] = nn.Linear(
                    self.hidden_dim, label_spec["dim"]
                )
        else:
            self.logger.warning(
                "Label strategy not yet implemented. Ignoring labels... "
            )

    def _init_encoders(self):
        self.encoders = nn.ModuleDict()
        for mod in self.modalities_spec:
            name = mod["name"]
            output_type = mod["output"]
            if output_type == "graph":
                spec = self.expert_encoders_spec["graph_encoder"]
                self.encoders[name] = GraphEncoder(
                    node_dim=mod["node_dim"],
                    edge_dim=mod["edge_dim"],
                    layers=spec["layers"],
                    layer_type=spec["layer_type"],
                    hidden_dim=spec["hidden_dim"],
                    output_dim=spec["output_dim"],
                    activation=spec["activation"],
                    dropout=spec["dropout"],
                    attn_heads=spec["attn_heads"],
                    pooling=spec["pooling"],
                )
            elif output_type == "atoms":
                spec = self.expert_encoders_spec["atom_encoder"]
                self.encoders[name] = AtomsEncoder(
                    node_dim=mod["node_dim"],
                    layers=spec["layers"],
                    hidden_dim=spec["hidden_dim"],
                    output_dim=spec["output_dim"],
                    attn_heads=spec["attn_heads"],
                    dropout=spec["dropout"],
                )
            else:
                spec = self.expert_encoders_spec["emb_encoder"]
                self.encoders[name] = EmbEncoder(
                    input_dim=mod["dim"],
                    layers=spec["layers"],
                    hidden_dim=spec["hidden_dim"],
                    output_dim=spec["output_dim"],
                    dropout=spec["dropout"],
                )

    def encode(self, batch, modalities, active_mask):
        """Encode all active modalities once. Returns per-sample embeddings."""
        modalities_ext = [
            self.modalities_dict[mod.replace("_x_ptr", "")] if mod is not None else None
            for mod in modalities
        ]

        batch_size = active_mask.size(0)
        hidden_dim = self.hidden_dim
        n_mods = len(modalities_ext)
        embeddings = []

        for i, mod in enumerate(modalities_ext):
            if mod:
                name = mod["name"]
                encoder = self.encoders[name]

                if mod["output"] == "graph":
                    x = batch[f"{name}_x"]
                    edge_index = batch[f"{name}_edge_index"]
                    edge_attr = batch[f"{name}_edge_attr"]
                    b = batch[f"{name}_x_batch"]
                    emb = encoder(x, edge_index, edge_attr, b)
                elif mod["output"] == "atoms":
                    x = batch[f"{name}_x"]
                    b = batch[f"{name}_x_batch"]
                    remapped_batch = torch.unique(b, sorted=True, return_inverse=True)[
                        1
                    ]
                    emb = encoder(x, remapped_batch)
                elif mod["output"] == "embedding":
                    x = batch[f"{name}_x"]
                    emb = encoder(x)

                embeddings.append((i, emb))

        dtype = embeddings[0][1].dtype if embeddings else next(self.parameters()).dtype
        full = torch.zeros(
            batch_size,
            n_mods,
            hidden_dim,
            device=active_mask.device,
            dtype=dtype,
        )
        for col_i, emb in embeddings:
            full[active_mask[:, col_i], col_i] = emb

        return full

    def predict(self, full, pred_mask, active_mask):
        """Run transformer head on pre-computed embeddings with prediction masking."""
        batch_size = pred_mask.size(0)
        device = pred_mask.device

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, full], dim=1)

        cls_col_zero = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        cls_col_one = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

        # Targets get replaced by mask tokens inside the head
        predict_mask_cls = torch.cat([cls_col_zero, ~pred_mask], dim=1)

        # Active = real content. CLS is always active.
        active_mask_cls = torch.cat([cls_col_one, active_mask], dim=1)
        missing_mask_cls = ~active_mask_cls

        if self.cls_predictor:
            # Also hide context modalities from attention so only CLS provides content.
            hide_context = torch.cat([cls_col_zero, pred_mask & active_mask], dim=1)
            missing_mask_cls = missing_mask_cls | hide_context

        out, embeddings = self.transformer_head(
            x, mask=predict_mask_cls, missing_mask=missing_mask_cls
        )
        cls = out[:, 0, :]
        preds = out[:, 1:, :]
        return preds, cls, embeddings


# ----------------------------------------
# --------- Utilities for model ----------
# ----------------------------------------


def ssl_forward(self, batch: Dict, stage: str):
    """
    Self-supervised (masked) forward pass for JEPA called during training.
    self is the stable-pretraining module wrapper, so we need to access the model with self.model
    """
    model_cfg = self.hparams["module"]

    modalities = [
        m["name"] + "_x_ptr" if m["name"] + "_x_ptr" in batch else None
        for m in model_cfg["modalities"]
    ]
    device = next(self.model.parameters()).device
    active_modalities = [
        batch[k][1:] - batch[k][:-1]
        if k is not None
        else torch.zeros(batch.num_graphs, device=device)
        for k in modalities
    ]
    active_mask = (torch.stack(active_modalities) > 0).t()
    pred_mask = ratio_mask(active_mask, model_cfg["train_config"]["masking_ratio"])

    # Encode once, predict from cached embeddings
    targets = self.model.encode(batch, modalities, active_mask)
    predictions, cls, embeddings = self.model.predict(targets, pred_mask, active_mask)

    # Prediction loss
    losses = compute_loss(
        self.model,
        predictions,
        cls,
        targets,
        lamb=model_cfg["train_config"]["lambda"],
        pred_mask=pred_mask,
        active_mask=active_mask,
    )

    # If labels as supervised loss
    if self.model.label_strategy == "label":
        supervised_loss = compute_supervised_loss(self.model, cls, batch, model_cfg)
        losses.update(supervised_loss)
        losses["loss"] += supervised_loss["supervised_loss"]

    out_dict = {
        **losses,
        "stage": stage,
        "batch_size": targets.size(0),
        "embeddings": targets.detach(),
        "embeddings_pred": predictions.detach(),
        "embeddings_cls": cls.detach(),
        "embeddings_z": embeddings.detach(),
        "active_mask": active_mask.detach(),
        "pred_mask": pred_mask.detach(),
        **extract_benchmark_labels(batch),
    }

    # Add modality-specific embeddings for diagnostics
    for i, mod in enumerate(model_cfg["modalities"]):
        name = mod["name"]
        if mod is not None:
            out_dict[f"embedding_{name}"] = targets[:, i, :].detach()
            out_dict[f"embedding_pred_{name}"] = predictions[:, i, :].detach()
        else:
            out_dict[f"embedding_{name}"] = torch.full(
                (targets.size(0), targets.size(2)), float("nan"), device=targets.device
            )

    log_metrics(self, out_dict, stage)
    return out_dict


def get_weights(self, modalities, predictions, pred_mask):
    modality_weights = [
        self.weight_map[m.replace("_x_ptr", "")] if m else 0.0 for m in modalities
    ]
    modality_weights = torch.tensor(
        modality_weights, device=predictions.device, dtype=predictions.dtype
    )
    stacked_weights = modality_weights.unsqueeze(0).repeat(predictions.size(0), 1)
    masked_weights = stacked_weights * pred_mask.to(dtype=predictions.dtype)
    weights = masked_weights.sum(dim=1) / pred_mask.sum(dim=1).clamp(min=1)
    return weights


def compute_supervised_loss(self, cls, batch, model_cfg):

    supervised_losses = {}
    agg_loss = cls.new_zeros(())
    for label_spec in model_cfg["labels"]:
        name = label_spec["name"]
        ptr_key = f"{name}_x_ptr"
        if ptr_key not in batch:
            continue
        ptr = batch[ptr_key]
        present = (ptr[1:] - ptr[:-1]) > 0  # (B,) bool: which samples have this label
        if not bool(present.any()):
            continue
        head = self.prediction_heads[name]
        pred = head(cls[present])
        truth = batch[f"{name}_x"]
        mse_loss = F.mse_loss(pred, truth)
        supervised_losses[f"supervised_loss_{name}"] = mse_loss
        agg_loss = agg_loss + mse_loss

    supervised_losses["supervised_loss"] = agg_loss
    return supervised_losses


def compute_loss(
    self,
    predictions: torch.Tensor,
    cls: torch.Tensor,
    targets: torch.Tensor,
    lamb: float = 0.01,
    pred_mask: torch.Tensor = None,
    active_mask: torch.Tensor = None,
):
    losses = {}
    _zero = torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)

    if self.weighted_loss:
        modalities = [m["name"] for m in self.modalities_spec]
        weights = get_weights(self, modalities, predictions, pred_mask)
    else:
        weights = None

    # For DDP always return all modalities, even if some are empty
    for i in range(predictions.size(1)):
        active_samples = active_mask[:, i]
        if active_samples.sum() < 2:
            # Not enough samples for sigreg, but still register the keys
            losses[f"sigreg_loss_m_{i}"] = _zero
            losses[f"pred_loss_m_{i}"] = _zero
            continue

        # We apply sigreg on the targets, which are more stable
        emb_i = targets[active_samples, i, :]  # (n_active, D)
        sigreg_loss = self.sigreg(emb_i)
        losses[f"sigreg_loss_m_{i}"] = sigreg_loss * lamb

        # Prediction loss
        prediction_samples = ~pred_mask[:, i] & active_mask[:, i]
        if prediction_samples.sum() > 0:
            pred_i = predictions[prediction_samples, i, :]
            target_i = targets[prediction_samples, i, :]

            if self.loss_projection:
                proj = self.loss_proj[i]
                pred_i = proj(pred_i)
                target_i = proj(target_i)

            if weights is not None:
                w_i = weights[prediction_samples].unsqueeze(1)
                l2_loss = F.mse_loss(pred_i * w_i, target_i * w_i)
            else:
                l2_loss = F.mse_loss(pred_i, target_i)
            losses[f"pred_loss_m_{i}"] = l2_loss
        else:
            losses[f"pred_loss_m_{i}"] = _zero

    if self.cls_sigreg:
        # Apply sigreg to cls token as well
        cls_sigreg_loss = self.sigreg(cls)
        losses["sigreg_loss_cls"] = cls_sigreg_loss * lamb

    losses["loss"] = sum(losses.values())
    losses["sigreg_loss"] = sum(v for k, v in losses.items() if "sigreg_loss" in k)
    losses["pred_loss"] = sum(v for k, v in losses.items() if "pred_loss" in k)
    return losses


def ratio_mask(active_mask: torch.Tensor, ratio: float):
    rows, cols = active_mask.shape
    scores = torch.rand(rows, cols, device=active_mask.device)
    scores = scores * active_mask.float()
    mask = scores > ratio

    # Ensure no row is empty
    fallback_mask = (mask.sum(dim=1) == 0) & (active_mask.sum(dim=1) > 0)
    fallback_scores = (
        torch.rand(rows, cols, device=active_mask.device) * active_mask.float()
    )
    fallback_idx = fallback_scores.argmax(dim=1, keepdim=True)
    fallback_values = torch.zeros_like(mask).scatter_(1, fallback_idx, True)

    fallback_mask_all = fallback_mask.repeat(1, cols).reshape(cols, -1).T
    mask = torch.where(fallback_mask_all, fallback_values, mask)
    return mask


def extract_benchmark_labels(batch):
    benchmark_labels = {
        k: batch[k].reshape(-1, 1).float()
        for k in batch.keys()
        if "regression" in k or "classification" in k
    }
    return benchmark_labels


def log_metrics(self, out_dict, stage):
    # Add flag when its weighted loss
    if self.model.weighted_loss:
        stage = f"{stage}_weighted"

    for key, value in out_dict.items():
        if "loss" in key:
            self.log(
                f"{stage}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=key == "loss",
                batch_size=out_dict["batch_size"],
            )

    if hasattr(self, "hp_metric"):
        self.log(
            "hp_metric",
            self.hp_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=out_dict.get("batch_size", None),
        )
