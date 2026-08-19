import sys
import numpy as np
import torch
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Batch
from contextlib import contextmanager

from models.modules.jepa import MolJEPA
from models.backbones.graph import GraphEncoder
from data.dataset import MultimodalData
from data.collator import from_data_list


class MolJEPAInference(torch.nn.Module):
    def __init__(self, checkpoint_path=None, cfg="moljepa.yaml"):
        super().__init__()

        # Load checkpoint and use the config it was trained with, so the model
        # architecture always matches the saved weights (handles ablations).
        self.checkpoint_path = checkpoint_path
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "hyper_parameters" in ckpt:
            self.cfg_file = "<checkpoint>"
            module_cfg = OmegaConf.to_container(
                ckpt["hyper_parameters"]["module"], resolve=True
            )
            data_cfg = OmegaConf.to_container(
                ckpt["hyper_parameters"]["data"], resolve=True
            )
        else:
            self.cfg_file = f"<enter-your-path>/{cfg}"
            cfg = OmegaConf.load(self.cfg_file)
            module_cfg = OmegaConf.to_container(cfg.module, resolve=True)
            data_cfg = OmegaConf.to_container(cfg.data, resolve=True)

        # Labels are appended to modalities in-place during training and saved
        # that way, so drop labels already present to avoid duplicate encoders.
        modality_names = {m["name"] for m in data_cfg["modalities"]}
        labels_spec = [l for l in data_cfg["labels"] if l["name"] not in modality_names]

        self.model = MolJEPA(
            modalities_spec=data_cfg["modalities"],
            labels_spec=labels_spec,
            moe_encoder_spec=module_cfg["moe_encoder"],
            expert_encoders_spec=module_cfg["expert_encoders"],
            train_config=module_cfg["train_config"],
        )

        state_dict = {
            k.replace("model.", "", 1): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def __str__(self):
        return f"""MolJEPAInference module
                   Checkpoint: {self.checkpoint_path}
                   Config file: {self.cfg_file.split("/")[-1]}
                   Number of modalities: {len(self.model.modalities_spec)}
                   Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}
             """

    def smiles_to_batch(self, smiles_list, embeddings_data=None):
        atom_modalities = {
            m["name"]
            for m in self.model.modalities_spec
            if m.get("input") == "precomputed_node"
        }
        has_graph = any(m.get("output") == "graph" for m in self.model.modalities_spec)
        graph_backbone = GraphEncoder() if has_graph else None

        data_list = []
        follow_batch = set()
        for i, smi in enumerate(smiles_list):
            data = MultimodalData()

            if has_graph:
                graph_feats = graph_backbone.encode(smi)
                data.add_graph(
                    "graph",
                    x=graph_feats["x"],
                    edge_index=graph_feats["edge_index"].long(),
                    edge_attr=graph_feats["edge_attr"],
                )
                follow_batch.add("graph_x")
            else:
                # Graph modality ablated: fall back to ECFP4 (2048-dim).
                data.add_mol_embedding("ecfp", ecfp4(smi))
                follow_batch.add("ecfp_x")

            if embeddings_data is not None and embeddings_data[i]:
                for mod_name, value in embeddings_data[i].items():
                    if isinstance(value, str):
                        emb = torch.from_numpy(np.load(value)).float()
                    else:
                        emb = value.float()
                    if mod_name in atom_modalities:
                        data.add_atom_embedding(mod_name, x=emb)
                    else:
                        data.add_mol_embedding(mod_name, emb)
                    follow_batch.add(f"{mod_name}_x")

            data_list.append(data)

        return from_data_list(Batch, data_list, follow_batch=list(follow_batch))

    def forward(self, smiles_list, return_attn=False, embeddings_data=None):
        device = next(self.model.parameters()).device
        batch = self.smiles_to_batch(smiles_list, embeddings_data=embeddings_data).to(
            device
        )
        batch_size = len(smiles_list)

        # Build active mask
        modalities = [
            f"{m['name']}_x_ptr" if f"{m['name']}_x_ptr" in batch else None
            for m in self.model.modalities_spec
        ]
        active_per_mod = [
            batch[k][1:] - batch[k][:-1]
            if k is not None
            else torch.zeros(batch_size, device=device)
            for k in modalities
        ]
        active_mask = (torch.stack(active_per_mod) > 0).t()

        pred_mask = active_mask.clone()

        captured = []
        with torch.no_grad():
            if return_attn:
                targets = self.model.encode(batch, modalities, active_mask)
                with _capture_attention(
                    self.model.transformer_head.transformer
                ) as captured:
                    predictions, cls, embeddings = self.model.predict(
                        targets, pred_mask, active_mask
                    )
            else:
                targets = self.model.encode(batch, modalities, active_mask)
                predictions, cls, embeddings = self.model.predict(
                    targets, pred_mask, active_mask
                )

        if return_attn:
            # captured: list[Tensor], one per layer, shape [B, n_heads, S, S]
            return predictions, cls, embeddings, list(captured)
        return predictions, cls, embeddings


def ecfp4(smiles, n_bits=2048):
    """ECFP4 (Morgan radius 2) bit vector as a float tensor."""
    mol = Chem.MolFromSmiles(smiles)
    fp = np.zeros(n_bits, dtype=np.float32)
    if mol is not None:
        fp = np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits),
            dtype=np.float32,
        )
    return torch.from_numpy(fp)


@contextmanager
def _capture_attention(transformer):
    """Temporarily patch each TransformerEncoderLayer.forward to capture per-head
    self-attention weights. Assumes norm_first=True (which MultiModalPredictor uses).

    Yields a list that gets populated with one tensor per layer of shape
    [B, n_heads, S, S].
    """
    layers = transformer.layers
    captured = [None] * len(layers)
    originals = [layer.forward for layer in layers]

    def make_patched(layer, idx):
        def patched(src, src_mask=None, src_key_padding_mask=None, is_causal=False):
            h = layer.norm1(src)
            attn_out, attn = layer.self_attn(
                h,
                h,
                h,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            captured[idx] = attn.detach()
            x = src + layer.dropout1(attn_out)
            x = x + layer._ff_block(layer.norm2(x))
            return x

        return patched

    for i, layer in enumerate(layers):
        layer.forward = make_patched(layer, i)
    try:
        yield captured
    finally:
        for layer, orig in zip(layers, originals):
            layer.forward = orig
