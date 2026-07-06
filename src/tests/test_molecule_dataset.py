import sys

sys.path.insert(0, "/home/rottach/phd/p3_jepa/moljepa/src")

from data.dataset import MoleculeDataset
from data.collator import Collater
from data.dataloader import DataLoader

mol_dataset = MoleculeDataset(
    filename="data.csv",
    encoder_spec={
        # "chemgpt": {"input": "smiles", "output": "embedding"},
        "graph": {"input": "smiles", "output": "graph"},
        "uma": {"input": "ase", "output": "atoms"},
        # "boltz": {"input": "precomputed", "output": "embedding"},
        "descriptor": {"input": "precomputed", "output": "embedding"},
    },
)


follow_batch = mol_dataset.follow_batch
include_keys = mol_dataset.include_keys

train_loader = DataLoader(
    mol_dataset,
    batch_size=3,
    shuffle=False,
    follow_batch=follow_batch,
    include_keys=include_keys,
    pin_memory=False,
    num_workers=2,
    persistent_workers=True,
    collate_fn=Collater(mol_dataset, follow_batch=follow_batch, include_keys=include_keys),
)

next(iter(train_loader))
