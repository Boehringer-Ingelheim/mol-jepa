"""Compute UMA embeddings for all conformers in a dataframe and save as .npy files."""

import argparse
import os
import sys
import traceback

import numpy as np
import pandas as pd
import torch
from ase.io import read
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from tqdm import tqdm

# Add the src directory to path so we can import the encoder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))
from models.backbones.uma import UMAEncoder

ROOT_DIR = "<path>/data"
OUTPUT_DIR = "<path>/uma"
BATCH_SIZE = 64
USE_CACHE = True


def main():
    parser = argparse.ArgumentParser(
        description="Compute UMA embeddings for conformers."
    )
    parser.add_argument(
        "--csv", type=str, required=True, help="CSV filename relative to root dir"
    )
    parser.add_argument(
        "--root", type=str, default=ROOT_DIR, help="Root directory for data"
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_DIR, help="Output directory for embeddings"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Batch size for encoding"
    )
    parser.add_argument(
        "--use_cache",
        type=bool,
        default=USE_CACHE,
        help="Whether to use cached embeddings",
    )
    args = parser.parse_args()

    # Create a subdirectory for this CSV's embeddings
    OUTPUT_DIR_FULL = os.path.join(args.output, args.csv.replace(".csv", ""))
    os.makedirs(OUTPUT_DIR_FULL, exist_ok=True)

    # Load dataframe
    csv_path = os.path.join(args.root, args.csv)
    print(f"Loading dataframe from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    df["uma_embedding_path"] = np.nan
    print(f"Loaded {len(df)} rows.")

    # Initialize UMA encoder
    print("Initializing UMA encoder...")
    encoder = UMAEncoder()

    # Process in batches
    indices = df.index.tolist()
    n_batches = (len(indices) + args.batch_size - 1) // args.batch_size
    failed_count = 0

    for batch_start in tqdm(
        range(0, len(indices), args.batch_size),
        total=n_batches,
        desc="Computing UMA embeddings",
    ):
        batch_indices = indices[batch_start : batch_start + args.batch_size]

        # Skip rows that already have embeddings
        if args.use_cache:
            batch_indices_to_process = []
            for idx in batch_indices:
                out_path = os.path.join(OUTPUT_DIR_FULL, f"uma_{idx}.npy")
                if not os.path.exists(out_path):
                    batch_indices_to_process.append(idx)
                else:
                    df["uma_embedding_path"].at[idx] = out_path
        else:
            batch_indices_to_process = batch_indices
        if len(batch_indices_to_process) == 0:
            print(f"Skipping batch {batch_start // args.batch_size + 1}/{n_batches}.")
            continue

        # Load atoms
        atoms_list = []
        valid_indices = []
        for idx in batch_indices_to_process:
            conformer_path = df.at[idx, "conformer_path"]
            if pd.isna(conformer_path):
                continue
            try:
                atoms = read(str(conformer_path))
                atoms_list.append(AtomicData.from_ase(atoms, task_name="omol"))
                valid_indices.append(idx)
            except Exception as e:
                failed_count += 1
                print(
                    f"Index {idx}: Failed to read ASE data from {conformer_path}: {e}"
                )

        if len(atoms_list) == 0:
            print(
                f"No valid conformers in batch {batch_start // args.batch_size + 1}/{n_batches}, skipping."
            )
            continue

        # Try batch encoding first, fall back to individual encoding on failure
        try:
            atoms_batch = atomicdata_list_to_batch(atoms_list)
            batch_results = encoder.encode_batch(atoms_batch)

            for i, idx in enumerate(valid_indices):
                embedding = batch_results[i].cpu().float().numpy()
                out_path = os.path.join(OUTPUT_DIR_FULL, f"uma_{idx}.npy")
                np.save(out_path, embedding)
                df["uma_embedding_path"].at[idx] = out_path

        except Exception as e:
            print(
                f"Batch encoding failed ({e}), falling back to individual encoding..."
            )
            for i, idx in enumerate(valid_indices):
                try:
                    embedding = encoder.encode(atoms_list[i])
                    out_path = os.path.join(OUTPUT_DIR_FULL, f"uma_{idx}.npy")
                    np.save(out_path, embedding)
                    df["uma_embedding_path"].at[idx] = out_path
                except Exception as e2:
                    failed_count += 1
                    print(f"Index {idx}: Individual encoding also failed: {e2}")
                    traceback.print_exc()
    df.to_csv(
        os.path.join(OUTPUT_DIR_FULL, f"{args.csv.replace('.csv', '_uma.csv')}"),
        index=False,
    )
    print(f"Done. Total failures: {failed_count}")


if __name__ == "__main__":
    main()
