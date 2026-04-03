import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Dict
from omegaconf import ListConfig

from models.modules.atoms_encoder import AtomsEncoder
from models.modules.emb_encoder import EmbEncoder
from models.modules.graph_encoder import GraphEncoder
from models.losses.sigreg import SlicedEppsPulley


class MolJEPA(nn.Module):
    def __init__(
        self,
        modalities_spec: Union[list, ListConfig],
        labels_spec: Union[list, ListConfig],
        label_strategy: str,
        moe_encoder_spec: dict,
        expert_encoders_spec: dict,
    ):
        super().__init__()

        self.modalities_spec = modalities_spec
        self.labels_spec = labels_spec
        self.label_strategy = label_strategy
        self.moe_encoder_spec = moe_encoder_spec
        self.expert_encoders_spec = expert_encoders_spec
        self.apply_label_strategy()
        self.modalities_dict = {mod["name"]: mod for mod in self.modalities_spec}
        self.sigreg = SlicedEppsPulley(num_slices=1024, t_max=3.0, n_points=17)
        self.encoders = nn.ModuleDict()
        for mod in self.modalities_spec:
            name = mod["name"]
            output_type = mod["output"]
            if output_type == "graph":
                spec = expert_encoders_spec["graph_encoder"]
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
                spec = expert_encoders_spec["atom_encoder"]
                self.encoders[name] = AtomsEncoder(
                    node_dim=mod["node_dim"],
                    layers=spec["layers"],
                    hidden_dim=spec["hidden_dim"],
                    output_dim=spec["output_dim"],
                    attn_heads=spec["attn_heads"],
                )
            else:
                spec = expert_encoders_spec["emb_encoder"]
                self.encoders[name] = EmbEncoder(
                    input_dim=mod["dim"],
                    layers=spec["layers"],
                    hidden_dim=spec["hidden_dim"],
                    output_dim=spec["output_dim"],
                    dropout=spec["dropout"],
                )

        hidden_dim = moe_encoder_spec["hidden_dim"]
        self.attn_query = nn.Linear(hidden_dim, 1)

        self.use_prediction_head = moe_encoder_spec.get("use_prediction_head", False)
        if self.use_prediction_head:
            prediction_dim = hidden_dim + len(modalities_spec)
            self.jepa_head = nn.Sequential(
                nn.Linear(prediction_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.use_loss_projection = moe_encoder_spec.get("use_loss_projection", False)
        if self.use_loss_projection:
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, int(hidden_dim / 4)),
            )

    def apply_label_strategy(self):
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
                    self.moe_encoder_spec["hidden_dim"], label_spec["dim"]
                )
        else:
            print("WARN: Label strategy not yet implemented.")

    def pooling(self, x, batch):
        pass

    def forward(self, batch, modalities, mask, active_mask) -> torch.Tensor:
        modalities = [
            self.modalities_dict[mod.replace("_x_ptr", "")] if mod is not None else None
            for mod in modalities
        ]

        # Guard: mask columns must match modality count
        assert mask.size(1) == len(modalities), (
            f"mask has {mask.size(1)} cols but {len(modalities)} modalities"
        )
        assert active_mask.size(1) == len(modalities), (
            f"active_mask has {active_mask.size(1)} cols but {len(modalities)} modalities"
        )

        embeddings = []

        for i, mod in enumerate(modalities):
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

                sparse_mask = mask[:, i][active_mask[:, i]]
                embeddings.append((i, emb[sparse_mask]))

        batch_size = mask.size(0)
        n_cols = mask.size(1)
        hidden_dim = self.moe_encoder_spec["hidden_dim"]
        full = torch.zeros(batch_size, n_cols, hidden_dim, device=mask.device)

        for col_i, emb in embeddings:
            full[mask[:, col_i] & active_mask[:, col_i], col_i] = emb

        # scores = self.attn_query(full).squeeze(-1).masked_fill(~mask, float("-inf"))
        # combined = (torch.softmax(scores, dim=1).unsqueeze(-1) * full).sum(dim=1)
        # TODO: Simple pooling
        combined = full.sum(dim=1)
        return combined


# ----------------------------------------
# --------- Utilities for model ----------
# ----------------------------------------


def ssl_forward(self, batch: Dict, stage: str):
    """
    self is the stable-pretraining module wrapper, so we need to access the model with self.model
    """
    model_cfg = self.hparams["module"]
    data_cfg = self.hparams["data"]

    modalities = [
        m["name"] + "_x_ptr" if m["name"] + "_x_ptr" in batch else None
        for m in model_cfg["modalities"]
    ]
    device = next(self.model.parameters()).device
    active_modalities = [
        batch[k][1:] - batch[k][:-1]
        if k is not None
        else torch.zeros(len(batch), device=device)
        for k in modalities
    ]
    active_mask = (torch.stack(active_modalities) > 0).t()

    if stage == "fit":
        # Create masks for 2 passes
        pass_1_mask, pass_2_mask = sample_masks(
            active_mask,
            model_cfg["moe_encoder"]["masking_strategy"],
            ratio=model_cfg["moe_encoder"]["masking_ratio"],
        )
    else:
        # Use all modalities for validation and testing
        pass_1_mask = active_mask
        pass_2_mask = active_mask

    # Use the other pass mask as features
    embedding_1 = self.model(batch, modalities, pass_1_mask, active_mask)
    embedding_2 = self.model(batch, modalities, pass_2_mask, active_mask)

    # Prediction head
    if self.model.use_prediction_head:
        embedding_1 = self.model.jepa_head(
            torch.cat([embedding_1, pass_2_mask.float()], dim=-1)
        )
        embedding_2 = self.model.jepa_head(
            torch.cat([embedding_2, pass_1_mask.float()], dim=-1)
        )

    # SSL-style projection
    if self.model.use_loss_projection:
        z1 = self.model.projection_head(embedding_1)
        z2 = self.model.projection_head(embedding_2)
    else:
        z1, z2 = embedding_1, embedding_2

    loss, l2_loss, sigreg_loss = compute_loss(
        self.model,
        z1,
        z2,
        lamb=model_cfg["moe_encoder"]["lambda"],
    )

    # Label loss
    if data_cfg.label_strategy == "label":
        for label_name in self.model.prediction_heads.keys():
            if label_name + "_x" not in batch:
                continue
            pred = self.model.prediction_heads[label_name](embedding_1)
            target = batch[label_name + "_x"]
            # Nan mask
            mask = ~torch.isnan(target)
            # TODO: Track batch to make this work
            if mask.sum() > 0:
                label_loss = F.mse_loss(pred[mask], target[mask])
                loss += label_loss

    out_dict = {
        "loss": loss,
        "l2_loss": l2_loss,
        "sigreg_loss": sigreg_loss,
        "stage": stage,
        "batch_size": embedding_1.size(0),
        "embedding_1": embedding_1,
        "embedding_2": embedding_2,
        **extract_benchmark_labels(batch),
    }
    log_metrics(self, out_dict, stage)
    return out_dict


def compute_loss(
    self, embedding_1: torch.Tensor, embedding_2: torch.Tensor, lamb: float = 0.01
):
    l2_loss = F.pairwise_distance(embedding_1, embedding_2, p=2).mean()
    sigreg_loss = self.sigreg(embedding_1) + self.sigreg(embedding_2)
    loss = l2_loss + lamb * sigreg_loss
    return loss, l2_loss, sigreg_loss


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


def sample_masks(
    active_mask: torch.Tensor, strategy="one_hot", ratio: float = 0.1
) -> torch.Tensor:
    if strategy == "one_hot":
        rows, cols = active_mask.shape
        scores = torch.rand(rows, cols, device=active_mask.device)
        scores = scores.masked_fill(~active_mask, float("-inf"))

        # Randomly pick among true ones
        idx = scores.argmax(dim=1)
        onehot = F.one_hot(idx, num_classes=cols).to(dtype=torch.bool)

        # Build masks
        mask1 = onehot & active_mask
        mask2 = active_mask & ~mask1

        # If a row has no active modality, assign it to mask1
        empty_rows = ~mask2.any(dim=1)
        mask2[empty_rows] = mask1[empty_rows]
    elif strategy == "ratio":
        masks = []
        for _ in range(2):
            mask = ratio_mask(active_mask, ratio)
            masks.append(mask)
        mask1, mask2 = masks
    elif strategy == "ratio_first_pass":
        mask1 = ratio_mask(active_mask, ratio)
        mask2 = active_mask.clone()
    else:
        raise NotImplementedError(f"Sampling strategy {strategy} not implemented")
    return mask1, mask2


def extract_benchmark_labels(batch):
    benchmark_labels = {
        k: batch[k].reshape(-1, 1).float()
        for k in batch.keys()
        if "regression" in k or "classification" in k
    }
    return benchmark_labels


def log_metrics(self, out_dict, stage):
    self.log(
        f"{stage}_loss",
        out_dict["loss"],
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        batch_size=out_dict["batch_size"],
    )
    self.log(
        f"{stage}_sigreg_loss",
        out_dict["sigreg_loss"],
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        batch_size=out_dict["batch_size"],
    )
    self.log(
        f"{stage}_l2_loss",
        out_dict["l2_loss"],
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        batch_size=out_dict["batch_size"],
    )
