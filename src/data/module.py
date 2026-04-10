import logging
import numpy as np
import stable_pretraining as spt

from data.dataset import MoleculeDataset
from data.collator import Collater
from data.dataloader import DataLoader

logger = logging.getLogger(__name__)


def split_dataset(dataset, cfg):
    """Split such that benchmark data ends up in val"""
    benchmark_datasets = cfg.data.benchmark_labels.keys()
    benchmark_label_cols = [
        col for spec in cfg.data.benchmark_labels.values() for col in spec["labels"]
    ]
    # Apply splitting for those who do not have a provided split
    split_mask = dataset.data_table["dataset"].isin(benchmark_datasets) & (
        dataset.data_table["provided_split"].isna()
    )

    # For now, simple random split
    sampled_index = (
        dataset.data_table[split_mask].sample(frac=0.2, random_state=42).index
    )
    test_mask = dataset.data_table.index.isin(sampled_index)

    # Benchmarks with provided splits
    provided_mask = (
        dataset.data_table["dataset"].isin(benchmark_datasets)
        & (dataset.data_table["provided_split"] == "test")
        & (
            dataset.data_table[benchmark_label_cols].notna().any(axis=1)
            if benchmark_label_cols
            else True
        )
    )

    mask = test_mask | provided_mask
    train_index = dataset.data_table[~mask].index
    val_index = dataset.data_table[mask].index

    # Save data to disc
    train_data_table = dataset.data_table.iloc[train_index].reset_index(
        names=["raw_index"]
    )
    val_data_table = dataset.data_table.iloc[val_index].reset_index(names=["raw_index"])
    train_file_name = dataset.filename_unique + "_train.parquet"
    val_file_name = dataset.filename_unique + "_val.parquet"
    train_data_table.to_parquet(
        dataset.root_dir / dataset.processed_file_dir / train_file_name
    )
    val_data_table.to_parquet(
        dataset.root_dir / dataset.processed_file_dir / val_file_name
    )

    # Create new datasets
    train_dataset = MoleculeDataset(
        filename=cfg.data.filename,
        modalities_spec=dataset.modalities_spec,
        labels_spec=dataset.labels_spec,
        label_strategy=cfg.data.label_strategy,
        metadata_cols=cfg.data.metadata_cols,
        benchmark_labels_spec=cfg.data.benchmark_labels,
        recreate=cfg.data.recreate,
        num_workers=cfg.data.num_workers,
        processed_file=train_file_name,
    )

    val_dataset = MoleculeDataset(
        filename=cfg.data.filename,
        modalities_spec=dataset.modalities_spec,
        labels_spec=dataset.labels_spec,
        label_strategy=cfg.data.label_strategy,
        benchmark_labels_spec=cfg.data.benchmark_labels,
        metadata_cols=cfg.data.metadata_cols,
        recreate=cfg.data.recreate,
        num_workers=cfg.data.num_workers,
        processed_file=val_file_name,
    )

    return train_dataset, val_dataset


def build_dataloaders(cfg):
    modalities_spec = {
        item["name"]: {
            "input": item["input"],
            "output": item["output"],
            "colname": item["colname"] if "colname" in item else None,
            "processing": item["processing"] if "processing" in item else None,
            "batch_size": item["batch_size"] if "batch_size" in item else None,
        }
        for item in cfg.data.modalities
    }

    labels_spec = {
        item["name"]: {"colname": item["colname"]} for item in cfg.data.labels
    }

    dataset = MoleculeDataset(
        filename=cfg.data.filename,
        modalities_spec=modalities_spec,
        labels_spec=labels_spec,
        label_strategy=cfg.data.label_strategy,
        benchmark_labels_spec=cfg.data.benchmark_labels,
        metadata_cols=cfg.data.metadata_cols,
        recreate=cfg.data.recreate,
        num_workers=cfg.data.num_workers,
    )
    logger.info(
        f"Loaded dataset with {len(dataset)} samples | Modalities: {list(modalities_spec.keys())} \
            | Labels: {list(labels_spec.keys())}"
    )

    train_dataset, val_dataset = split_dataset(dataset, cfg)
    logger.info(
        f"Train dataset size: {len(train_dataset)} | Val dataset size: {len(val_dataset)}"
    )

    follow_batch = dataset.follow_batch
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        follow_batch=follow_batch,
        num_workers=cfg.data.num_workers,
        pin_memory=False,
        persistent_workers=True,
        collate_fn=Collater(train_dataset, follow_batch=follow_batch),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        follow_batch=follow_batch,
        num_workers=cfg.data.num_workers,
        pin_memory=False,
        persistent_workers=True,
        collate_fn=Collater(val_dataset, follow_batch=follow_batch),
    )

    return train_loader, val_loader


def get_data_module(cfg):
    train_loader, val_loader = build_dataloaders(cfg)
    return spt.data.DataModule(train=train_loader, val=val_loader)
