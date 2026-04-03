import torch
import numpy as np
from molfeat.calc.atom import AtomCalculator
from molfeat.calc.bond import EdgeMatCalculator
from molfeat.trans.graph import AdjGraphTransformer
from torch_geometric.utils import dense_to_sparse

from models.backbones.base import MoleculeEncoder


@MoleculeEncoder.register("graph")
class GraphEncoder(MoleculeEncoder):
    def __init__(self):
        self.featurizer = AdjGraphTransformer(
                            atom_featurizer=AtomCalculator(),
                            bond_featurizer=EdgeMatCalculator(),
                            explicit_hydrogens=True,
                            self_loop=False,
                            canonical_atom_order=True,
                            dtype=torch.float)

    def encode(self, smiles: str) -> np.ndarray:
        features = self.featurizer(self.featurizer.preprocess([smiles])[0][0])
        adjacency_matrix, atoms, bonds = features[0]
        edge_index = dense_to_sparse(torch.Tensor(adjacency_matrix))[0]

        return {
            "edge_index": edge_index.int(),
            "x": atoms,
            "edge_attr": bonds[edge_index[0], edge_index[1], :]
        }

    def encode_batch(self, smiles_list: list) -> np.ndarray:
        return [self.encode(smiles) for smiles in smiles_list]