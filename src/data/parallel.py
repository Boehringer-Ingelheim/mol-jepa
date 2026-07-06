import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from models.backbones.base import MoleculeEncoder

_worker_encoders = None
_worker_modalities_spec = None

def _worker_initializer(modalities_spec):
    """Initialize encoders inside each worker process to avoid pickling errors."""
    global _worker_encoders, _worker_modalities_spec
    _worker_modalities_spec = modalities_spec
    _worker_encoders = {}
    for encoder_name, spec in modalities_spec.items():
        backbone = (
            encoder_name
            if spec["input"] not in ["precomputed", "precomputed_node"]
            else spec["input"]
        )
        _worker_encoders[encoder_name] = MoleculeEncoder.create(backbone)

def _process_row_worker_fn(tensor_data_dir, root_dir, index, row):
    """Processes a single row by encoding its data and saving it as torch object."""
    global _worker_encoders, _worker_modalities_spec
    results = {}
    for encoder_name, spec in _worker_modalities_spec.items():
        save_path = str(Path(tensor_data_dir) / f"{encoder_name}_data_{index}.pt")

        # Check if data already exists as file
        if os.path.isfile(save_path):
            results[encoder_name] = save_path
            continue

        data = encode(
            _worker_encoders[encoder_name], spec, row, root_dir
        )

        if data is not None:
            data = (
                torch.as_tensor(np.array(data, dtype=np.float32))
                if isinstance(data, np.ndarray)
                else data
            )
            torch.save(data, save_path)
            results[encoder_name] = save_path

    return index, results

def encode(encoder, spec, query_data, root_dir):
    """Module-level encoder results function for use in multiprocessing workers."""
    data = None
    if spec["input"] == "smiles":
        data = encoder.encode(query_data["smiles"])
    elif spec["input"] == "ase":
        if pd.isna(query_data["conformer_path"]):
            return None
        xyz_path = str(Path(root_dir) / query_data["conformer_path"])
        try:
            atoms = read(xyz_path)
        except StopIteration:
            raise ValueError(f"XYZ file malformed: {xyz_path}")
        atoms_data = AtomicData.from_ase(atoms, task_name="omol")
        data = encoder.encode(atoms_data)
    elif spec["input"] in ("precomputed", "precomputed_node"):
        precomputed_colname = spec["colname"]
        if pd.isna(query_data[precomputed_colname]):
            return None
        data = encoder.encode(str(Path(root_dir) / query_data[precomputed_colname]))

        # Make sure the precomputed dimensions match the expected dimensions
        expected_dim = spec["dim"]
        actual_dim = data.shape[-1]
        assert (expected_dim == actual_dim), f"Precomputed embedding dimension mismatch for {spec['input']} modality '{precomputed_colname}': expected {expected_dim}, got {actual_dim}"
    return data