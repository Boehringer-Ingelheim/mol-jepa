import numpy as np

from models.backbones.base import MoleculeEncoder
from logging import getLogger

logger = getLogger(__name__)


@MoleculeEncoder.register("precomputed_node")
class PrecomputedNodeEncoder(MoleculeEncoder):
    def __init__(self):
        pass

    def encode(self, precomputed_path: str) -> np.ndarray:
        try:
            arr = np.load(precomputed_path, allow_pickle=False)
        except ValueError:
            arr = np.load(precomputed_path, allow_pickle=True)
            logger.warning(
                f"{precomputed_path}: {arr.shape} (requires pickle (dtype={arr.dtype}). Save as np.float32 for better performance)"
            )

        return arr

    def encode_batch(self, precomputed_paths: list) -> np.ndarray:
        return np.array([self.encode(path) for path in precomputed_paths])
