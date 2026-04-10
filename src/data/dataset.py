import os
import torch
import logging
import pandas as pd
import numpy as np
import traceback
import multiprocessing as mp
from rdkit import RDLogger
from ase.io import read
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from torch_geometric.data import Dataset
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed

from models.backbones.base import MoleculeEncoder
from models.callbacks.probes import extract_labels

RDLogger.DisableLog("rdApp.warning")


class MoleculeDataset(Dataset):
    def __init__(
        self,
        filename,
        modalities_spec={
            "chemgpt": {"input": "smiles", "output": "embedding"},
            "graph": {"input": "smiles", "output": "graph"},
            "uma": {"input": "ase", "output": "atoms"},
        },
        labels_spec={},
        benchmark_labels_spec={},
        label_strategy="modality",
        metadata_cols=None,
        recreate=False,
        processed_file=None,
        num_workers=8,
    ):
        load_dotenv()
        for var in ["DATA_DIR", "PROCESSED_DATA_DIR"]:
            if var not in os.environ:
                raise EnvironmentError(f"Failed to find required variable: {var}")
        self.root_dir = Path(os.getenv("DATA_DIR"))
        self._processed_dir = Path(os.getenv("PROCESSED_DATA_DIR"))
        self.modalities_spec = self.validate_modalities_spec(modalities_spec)
        self.labels_spec = labels_spec
        self.label_strategy = label_strategy
        self.benchmark_labels_spec = benchmark_labels_spec
        self.apply_label_strategy()
        self.metadata_cols = metadata_cols
        self.filename = filename
        self.filename_unique = f"f={filename.split('.')[0]}_enc={'_'.join(self.modalities_spec.keys()).replace('targets', 't_')}"
        self.tensor_data_dir = self._processed_dir / f"tensors_{self.filename_unique}"
        self.processed_file_dir = "parquets"
        self.processed_file = (
            f"data_table_{self.filename_unique}.parquet"
            if processed_file is None
            else processed_file
        )
        self.data_table = None
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)
        self.follow_batch = []
        self.include_keys = []

        # Enforce recreation
        if recreate and os.path.isfile(
            self.root_dir / self.processed_file_dir / self.processed_file
        ):
            os.remove(self.root_dir / self.processed_file_dir / self.processed_file)
            if os.path.isdir(self.tensor_data_dir):
                self.logger.info(
                    f"Recreate flag set: Removing existing tensor data in {self.tensor_data_dir}..."
                )
                for file in os.listdir(self.tensor_data_dir):
                    os.remove(os.path.join(self.tensor_data_dir, file))
                os.rmdir(self.tensor_data_dir)

        # Make sure all directories exist
        os.makedirs(self._processed_dir, exist_ok=True)
        os.makedirs(self.tensor_data_dir, exist_ok=True)
        os.makedirs(self.root_dir / self.processed_file_dir, exist_ok=True)
        super().__init__(self.root_dir, transform=None, pre_transform=None)
        self.init_include_and_follow_keys()

        self.logger.info(
            f"Loading processed data: {self.root_dir / self.processed_file_dir / self.processed_file}"
        )
        self.data_table = pd.read_parquet(
            self.root_dir / self.processed_file_dir / self.processed_file
        )

    def apply_label_strategy(self):
        if self.label_strategy == "modality":
            for label_name in self.labels_spec.keys():
                self.modalities_spec[label_name] = {
                    "input": "precomputed",
                    "output": "embedding",
                    "colname": self.labels_spec[label_name]["colname"],
                    # Labels are assumed to be precomputed vectors
                    "processing": "multiprocess",
                }

    def validate_modalities_spec(self, modalities_spec: dict):
        VALID_INPUTS = {"smiles", "ase", "precomputed", "precomputed_node"}
        VALID_OUTPUTS = {"embedding", "graph", "atoms"}
        for name, spec in modalities_spec.items():
            if not isinstance(spec, dict):
                raise ValueError(
                    f"Modality '{name}': spec must be a dict, got {type(spec)}"
                )

            if "input" not in spec:
                raise ValueError(f"Modality '{name}': missing required key 'input'")
            if "output" not in spec:
                raise ValueError(f"Modality '{name}': missing required key 'output'")

            if spec["input"] not in VALID_INPUTS:
                raise ValueError(
                    f"Modality '{name}': invalid input '{spec['input']}', must be one of {VALID_INPUTS}"
                )
            if spec["output"] not in VALID_OUTPUTS:
                raise ValueError(
                    f"Modality '{name}': invalid output '{spec['output']}', must be one of {VALID_OUTPUTS}"
                )
        return modalities_spec

    def init_encoder_columns(self, dataset):
        for encoder_name in self.modalities_spec.keys():
            dataset[f"{encoder_name}_tensors_path"] = None

    def init_include_and_follow_keys(self):
        for encoder_name in self.modalities_spec.keys():
            if self.modalities_spec[encoder_name]["output"] in [
                "precomputed",
                "atoms",
                "embedding",
            ]:
                self.include_keys.append(f"{encoder_name}_x")
                self.follow_batch.append(f"{encoder_name}_x")
            elif self.modalities_spec[encoder_name]["output"] == "graph":
                self.include_keys.append(f"{encoder_name}_x")
                self.include_keys.append(f"{encoder_name}_edge_index")
                self.include_keys.append(f"{encoder_name}_edge_attr")
                self.follow_batch.append(f"{encoder_name}_x")

    @property
    def raw_file_names(self):
        return [self.filename]

    @property
    def processed_file_names(self):
        return [self.root_dir / self.processed_file_dir / self.processed_file]

    def get_encoders_state(self, processing_subset="multiprocess"):
        self.logger.info("Initializing encoders for dataset processing...")
        encoders, modalities_spec_subset = self.init_encoders(processing_subset)

        return {"encoders": encoders, "modalities_spec": modalities_spec_subset}

    def init_encoders(self, processing_subset="multiprocess"):
        """
        processing_subset
        - multiprocess: All encoders that are thread-safe and pickleable.
        - batch: All encoders that require batch processing (e.g. due to GPU requirement or non-pickleability).
        """
        encoders = {}
        modalities_spec_subset = {}
        for encoder_name, spec in self.modalities_spec.items():
            if spec["processing"] == processing_subset:
                backbone = (
                    encoder_name
                    if spec["input"] not in ["precomputed", "precomputed_node"]
                    else spec["input"]
                )
                self.logger.info(
                    f"Initializing encoder for {encoder_name} of type {spec['input']}."
                )
                encoders[encoder_name] = MoleculeEncoder.create(backbone)
                modalities_spec_subset[encoder_name] = spec
        return encoders, modalities_spec_subset

    def _get_encoder_results(self, encoder, spec, query_data, root_dir, batch=False):
        data = None
        if spec["input"] == "smiles":
            if batch:
                data = encoder.encode_batch([q["smiles"] for q in query_data]).tolist()
            else:
                data = encoder.encode(query_data["smiles"])
        elif spec["input"] == "ase":
            if batch:
                # Prepare batch data
                filtered_indices = []
                atoms_list = []
                for i, q in enumerate(query_data):
                    if pd.isna(q["conformer_path"]):
                        filtered_indices.append(i)
                        continue
                    try:
                        path = str(root_dir / q["conformer_path"])
                        atoms = read(path)
                        atoms_list.append(AtomicData.from_ase(atoms, task_name="omol"))
                    except Exception as e:
                        filtered_indices.append(i)
                        self.logger.error(
                            f"Index {i}: Failed to read ASE data from {path}: {e}"
                        )
                        traceback.print_exc()

                atoms_list = atomicdata_list_to_batch(atoms_list)
                batch_data = encoder.encode_batch(atoms_list)
                data = [
                    batch_data[i] if i not in filtered_indices else None
                    for i in range(len(query_data))
                ]
            else:
                if pd.isna(query_data["conformer_path"]):
                    return None
                xyz_path = str(root_dir / query_data["conformer_path"])
                try:
                    atoms = read(xyz_path)
                    if not any(atom.symbol == "H" for atom in atoms):
                        self.logger.warning(f"No hydrogens for {xyz_path}")
                except StopIteration:
                    raise ValueError(f"XYZ file malformed: {xyz_path}")
                atoms_data = AtomicData.from_ase(atoms, task_name="omol")
                data = encoder.encode(atoms_data)

        elif spec["input"] in ("precomputed", "precomputed_node"):
            precomputed_colname = spec["colname"]
            if pd.isna(query_data[precomputed_colname]):
                return None
            if batch:
                data = encoder.encode_batch(
                    [str(root_dir / q[precomputed_colname]) for q in query_data]
                )
            else:
                data = encoder.encode(str(root_dir / query_data[precomputed_colname]))

        return data

    def _process_row_worker(self, worker_state, index, row):
        encoders = worker_state["encoders"]
        modalities_spec = worker_state["modalities_spec"]

        results = {}
        for encoder_name, spec in modalities_spec.items():
            data = self._get_encoder_results(
                encoders[encoder_name], spec, row, self.root_dir
            )

            if data is not None:
                save_path = str(
                    self.tensor_data_dir / f"{encoder_name}_data_{index}.pt"
                )
                data = (
                    torch.as_tensor(np.array(data, dtype=np.float32))
                    if isinstance(data, np.ndarray)
                    else data
                )
                torch.save(data, save_path)
                results[encoder_name] = save_path

        return index, results

    def process_sequential(self, dataset):
        rows = [(index, row) for index, row in dataset.iterrows()]
        worker_state = self.get_encoders_state(processing_subset="multiprocess")
        for index, row in tqdm(
            rows, desc="Processing dataset sequentially", dynamic_ncols=True
        ):
            try:
                _, results = self._process_row_worker(worker_state, index, row)
                for encoder_name, save_path in results.items():
                    dataset.at[index, f"{encoder_name}_tensors_path"] = save_path
            except Exception as e:
                self.logger.error(f"Index {index}: {e}")
                traceback.print_exc()

    def process_parallel(self, dataset):
        rows = [(index, row) for index, row in dataset.iterrows()]
        worker_state = self.get_encoders_state(processing_subset="multiprocess")

        pbar = tqdm(
            total=dataset.shape[0], dynamic_ncols=True, desc="Processing dataset"
        )
        ctx = mp.get_context("spawn")
        executor = ProcessPoolExecutor(max_workers=self.num_workers, mp_context=ctx)

        with pbar, executor:
            futures = {
                executor.submit(
                    self._process_row_worker, worker_state, index, row
                ): index
                for index, row in rows
            }

            failed_indices = []
            for future in as_completed(futures):
                try:
                    index, results = future.result()
                    for encoder_name, save_path in results.items():
                        dataset.at[index, f"{encoder_name}_tensors_path"] = save_path
                except Exception as e:
                    pbar.write(f"Index {futures[future]}: {e}")
                    traceback.print_exc()

                    failed_indices.append(futures[future])
                finally:
                    pbar.update(1)
                    pbar.set_postfix({"failed": len(failed_indices)})

    def process(self):
        if self.processed_file_names[0].is_file():
            self.logger.info(
                f"Processed file already exists at {self.processed_file_names[0]}, skipping processing."
            )
            return

        dataset = pd.read_csv(self.root_dir / self.filename)
        self.init_encoder_columns(dataset)

        # --- Step 1: Process all parallelizable encoders ---
        self.logger.info(
            f"Processing {len(dataset)} samples with {self.num_workers} workers..."
        )
        if self.num_workers > 1:
            self.process_parallel(dataset)
        else:
            # Useful for debugging
            self.process_sequential(dataset)

        # --- Step 2: Process other encoders with batch-processing ---
        state = self.get_encoders_state(processing_subset="batch")
        for encoder_name, spec in state["modalities_spec"].items():
            batch_size = spec["batch_size"]
            encoder = state["encoders"][encoder_name]
            rows = list(dataset.iterrows())
            batches = [
                rows[i : i + batch_size] for i in range(0, len(rows), batch_size)
            ]

            for batch in tqdm(batches, desc=f"Processing {encoder_name}"):
                indices = [idx for idx, _ in batch]
                batch_rows = [row for _, row in batch]
                try:
                    batch_data = self._get_encoder_results(
                        encoder, spec, batch_rows, self.root_dir, batch=True
                    )

                    # Update dataset with paths to saved tensors
                    for i, (index, data) in enumerate(zip(indices, batch_data)):
                        try:
                            if data is not None:
                                save_path = str(
                                    self.tensor_data_dir
                                    / f"{encoder_name}_data_{index}.pt"
                                )
                                data = torch.as_tensor(data, dtype=torch.float32)
                                torch.save(data, save_path)
                                dataset.at[index, f"{encoder_name}_tensors_path"] = (
                                    save_path
                                )
                        except Exception as e:
                            self.logger.error(f"Index {index}: {e}")
                            traceback.print_exc()
                except Exception as e:
                    # Batch failed — fall back to per-row processing
                    self.logger.warning(f"Batch failed, falling back to per-row: {e}")
                    for index, row in zip(indices, batch_rows):
                        try:
                            data = self._get_encoder_results(
                                encoder, spec, row, self.root_dir, batch=False
                            )
                            if data is not None:
                                save_path = str(
                                    self.tensor_data_dir
                                    / f"{encoder_name}_data_{index}.pt"
                                )
                                data = torch.as_tensor(data, dtype=torch.float32)
                                torch.save(data, save_path)
                                dataset.at[index, f"{encoder_name}_tensors_path"] = (
                                    save_path
                                )
                        except Exception as e:
                            self.logger.error(f"Index {index}: {e}")
                            traceback.print_exc()

        dataset.to_parquet(self.processed_file_names[0])

    def len(self):
        return self.data_table.shape[0]

    def get(self, idx):
        data = MultimodalData()

        # Add metadata
        metadata = {col: self.data_table.loc[idx, col] for col in self.metadata_cols}
        data.add_metadata(metadata)

        # Add benchmark labels
        column_names, extended_labels = extract_labels(self.benchmark_labels_spec)
        benchmark_metadata = {
            label: self.data_table.loc[idx, col]
            for col, label in zip(column_names, extended_labels)
        }
        data.add_metadata(benchmark_metadata)

        # Get per-encoder data
        for encoder_name, spec in self.modalities_spec.items():
            data_path = self.data_table.loc[idx, f"{encoder_name}_tensors_path"]

            if data_path is None:
                continue

            encoder_data = torch.load(data_path, weights_only=False)

            # Global embedding
            if spec["output"] == "embedding":
                data.add_mol_embedding(encoder_name, encoder_data)

            # Per node embedding
            elif spec["output"] == "atoms":
                data.add_atom_embedding(
                    encoder_name,
                    x=encoder_data,
                )
            # Per node embedding + edge features
            elif spec["output"] == "graph":
                data.add_graph(
                    encoder_name,
                    x=encoder_data["x"],
                    edge_index=encoder_data["edge_index"],
                    edge_attr=encoder_data["edge_attr"],
                )

        # Add labels as metadata
        for label_name in self.labels_spec.keys():
            colname = self.labels_spec[label_name]["colname"]
            labels_path = self.data_table.loc[idx, colname]
            if not pd.isna(labels_path):
                labels = np.load(labels_path)
                data.add_mol_embedding(
                    label_name, torch.tensor(labels, dtype=torch.float32)
                )

        return data


class MultimodalData(Data):
    def __init__(self):
        super().__init__()

    def add_metadata(self, metadata: dict):
        """Add metadata fields to the data object."""
        for key, value in metadata.items():
            setattr(self, key, value)

    def add_mol_embedding(self, encoder_name: str, embedding: torch.Tensor):
        """Global molecule-level embedding, shape (d,)"""
        embedding = embedding.flatten() if embedding.dim() > 1 else embedding
        embedding = embedding.unsqueeze(0) if embedding.dim() == 1 else embedding
        embedding = embedding.float()

        # Zero Imputation - this is mostly for when using target vectors as modalities
        embedding = torch.nan_to_num(embedding, nan=0.0)
        setattr(self, f"{encoder_name}_x", embedding)

    def add_atom_embedding(self, encoder_name: str, x: torch.Tensor):
        """Per-atom embeddings, x shape (n_atoms, d)"""
        setattr(self, f"{encoder_name}_x", x)

    def add_graph(
        self,
        encoder_name: str,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ):
        """Graph representation, x shape (n_atoms, d_a),
        edge_index shape (2, num_edges),
        x shape (num_edges, d_e)"""
        setattr(self, f"{encoder_name}_x", x)
        setattr(self, f"{encoder_name}_edge_index", edge_index)
        setattr(self, f"{encoder_name}_edge_attr", edge_attr)

    def __inc__(self, key, value, *args, **kwargs):
        if key.endswith("edge_index"):
            prefix = key.replace("edge_index", "x")
            if hasattr(self, prefix):
                return getattr(self, prefix).shape[0]
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key.endswith("edge_index"):
            return 1
        return 0
