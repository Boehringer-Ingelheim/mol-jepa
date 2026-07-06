import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from models.backbones.base import MoleculeEncoder


@MoleculeEncoder.register("ecfp")
class ECFPEncoder(MoleculeEncoder):
    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self.radius = radius
        self.n_bits = n_bits

    def encode(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        return np.array(fp, dtype=np.float32)

    def encode_batch(self, smiles_list: list) -> np.ndarray:
        return np.array([self.encode(s) for s in smiles_list], dtype=np.float32)
