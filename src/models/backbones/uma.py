import torch
import numpy as np
from fairchem.core import pretrained_mlip

from models.backbones.base import MoleculeEncoder


@MoleculeEncoder.register("uma")
class UMAEncoder(MoleculeEncoder):
    def __init__(self):
        self.model = pretrained_mlip.get_predict_unit(
            "uma-s-1p2", inference_settings="turbo", device="cuda", workers=1
        ).model

    def encode(self, atoms: str) -> np.ndarray:
        prediction = self.model.forward(atoms)
        per_atom_per_layer_embeddings = prediction["omol_embeddings"]["embeddings"]
        per_atom_last_layer_embeddings = per_atom_per_layer_embeddings[:, -1, :]
        return per_atom_last_layer_embeddings

    def encode_batch(self, atoms_batch: list) -> np.ndarray:
        if len(atoms_batch) == 0:
            return []
        with torch.no_grad():
            predictions = self.model.forward(atoms_batch)
            per_atom_per_layer_embeddings = predictions["omol_embeddings"]["embeddings"]
            per_atom_last_layer_embeddings = per_atom_per_layer_embeddings[:, -1, :]
        return [
            per_atom_last_layer_embeddings[atoms_batch.batch == i]
            for i in range(atoms_batch.num_graphs)
        ]
