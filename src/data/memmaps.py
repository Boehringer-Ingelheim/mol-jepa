import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.distributed as dist

from data.utils import is_rank_zero

_PACKED_X_DTYPE = "float32"
_PACKED_EI_DTYPE = "int64"
_PACKED_EA_DTYPE = "float32"


def setup_memmaps(modalities_spec, memmap_dir, logger, source_parquet, filename):
    """Build any missing (embedding + ragged) memmaps.

    Runs on every __init__ so that when the parquet already exists
    (PyG will skip `process()`), we still detect and materialize
    newly-added memmap files.
    """
    missing_embed = missing_memmaps(modalities_spec, memmap_dir, filename)
    missing_ragged = missing_ragged_memmaps(modalities_spec, memmap_dir, filename)
    if not missing_embed and not missing_ragged:
        return
    if is_rank_zero():
        logger.info(
            f"Ensuring memmaps exist (embed_missing={missing_embed}, "
            f"ragged_missing={missing_ragged}) from {source_parquet}"
        )
        dataset_df = pd.read_parquet(source_parquet)
        if missing_embed:
            build_memmaps(dataset_df, memmap_dir, filename, logger, modalities_spec)
        if missing_ragged:
            build_ragged_memmaps(
                dataset_df, memmap_dir, filename, modalities_spec, logger
            )
    memmap_barrier(modalities_spec, memmap_dir, filename)


def build_memmaps(dataset_df, memmap_dir, filename, logger, modalities_spec):
    stem = filename.split(".")[0]
    meta_path = memmap_dir / f"{stem}_meta.json"
    meta = json.load(open(meta_path)) if meta_path.is_file() else {}

    n_samples = len(dataset_df)
    for name, spec in embedding_modalities(modalities_spec).items():
        if memmap_path(name, memmap_dir, filename).is_file():
            continue
        dim = spec["dim"]
        logger.info(f"Building memmap for '{name}' shape=({n_samples}, {dim})")
        mm = np.memmap(
            memmap_path(name, memmap_dir, filename),
            dtype="float32",
            mode="w+",
            shape=(n_samples, dim),
        )
        col = f"{name}_tensors_path"
        for i in range(n_samples):
            path = dataset_df.iloc[i][col] if col in dataset_df.columns else None
            if path and not pd.isna(path):
                try:
                    t = torch.load(path, weights_only=False)
                    mm[i] = t.numpy().flatten()[:dim]
                except Exception:
                    pass
        mm.flush()
        meta[name] = {"shape": [n_samples, dim], "dtype": "float32"}
        del mm

    with open(meta_path, "w") as f:
        json.dump(meta, f)


def build_ragged_memmaps(dataset_df, memmap_dir, filename, modalities_spec, logger):
    """Build ragged memmaps for variable-length modalities (graph, uma).

    Two passes over each modality:
        Pass 1 — computing memmap size and offsets for each sample.
        Pass 2 — storing the data in the memmap.

    Technically, its possible to make this even more efficient by
    dynamically increasing the memmap size as we go.
    """
    ragged = ragged_modalities(modalities_spec)
    if not ragged:
        return

    os.makedirs(memmap_dir, exist_ok=True)
    stem = filename.split(".")[0]
    meta_path = memmap_dir / f"{stem}_ragged_meta.json"

    meta = {}
    if meta_path.is_file():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    n = len(dataset_df)

    for name, spec in ragged.items():
        if (
            name in meta
            and ragged_memmap_path(name, "x", memmap_dir, filename).is_file()
        ):
            # Already built
            continue

        output_type = spec["output"]
        node_dim = int(spec["node_dim"])
        edge_dim = int(spec.get("edge_dim") or 0) if output_type == "graph" else 0
        col = f"{name}_tensors_path"
        paths = (
            dataset_df[col].to_numpy()
            if col in dataset_df.columns
            else np.array([None] * n, dtype=object)
        )

        # --- Pass 1: Determining size and offsets ---
        node_off = np.zeros(n + 1, dtype=np.int64)
        edge_off = np.zeros(n + 1, dtype=np.int64) if output_type == "graph" else None
        logger.info(f"[pack:{name}] sizing {n} samples")
        for i in tqdm(range(n), desc=f"sizing {name}", dynamic_ncols=True):
            p = paths[i]
            if p is None or (isinstance(p, float) and np.isnan(p)):
                continue
            try:
                d = torch.load(p, weights_only=True)
            except Exception as e:
                logger.warning(f"[pack:{name}] sizing[{i}] failed: {e}")
                continue
            if output_type == "atoms":
                node_off[i + 1] = int(d.shape[0])
            else:  # graph
                node_off[i + 1] = int(d["x"].shape[0])
                edge_off[i + 1] = int(d["edge_index"].shape[1])

        np.cumsum(node_off, out=node_off)
        if edge_off is not None:
            np.cumsum(edge_off, out=edge_off)
        total_atoms = int(node_off[-1])
        total_edges = int(edge_off[-1]) if edge_off is not None else 0

        logger.info(
            f"[pack:{name}] total_atoms={total_atoms:,} "
            f"total_edges={total_edges:,} node_dim={node_dim} edge_dim={edge_dim}"
        )

        # --- Build empty memmaps for all variants ---

        # Atom level data
        x_path = ragged_memmap_path(name, "x", memmap_dir, filename)
        x_tmp = x_path.with_suffix(x_path.suffix + ".tmp")
        x_mm = np.memmap(
            x_tmp,
            dtype=_PACKED_X_DTYPE,
            mode="w+",
            shape=(max(total_atoms, 1), node_dim),
        )

        # Edge indices for graphs
        ei_mm = ea_mm = None
        ei_path = ea_path = ei_tmp = ea_tmp = None
        if output_type == "graph":
            ei_path = ragged_memmap_path(name, "ei", memmap_dir, filename)
            ea_path = ragged_memmap_path(name, "ea", memmap_dir, filename)
            ei_tmp = ei_path.with_suffix(ei_path.suffix + ".tmp")
            ea_tmp = ea_path.with_suffix(ea_path.suffix + ".tmp")
            ei_mm = np.memmap(
                ei_tmp,
                dtype=_PACKED_EI_DTYPE,
                mode="w+",
                shape=(2, max(total_edges, 1)),
            )
            ea_mm = np.memmap(
                ea_tmp,
                dtype=_PACKED_EA_DTYPE,
                mode="w+",
                shape=(max(total_edges, 1), edge_dim),
            )

        # --- Pass 2: Fill memmaps using precomputed offsets ---
        for i in tqdm(range(n), desc=f"packing {name}", dynamic_ncols=True):
            p = paths[i]
            if p is None or (isinstance(p, float) and np.isnan(p)):
                continue
            lo, hi = int(node_off[i]), int(node_off[i + 1])
            if hi <= lo:
                continue
            try:
                d = torch.load(p, weights_only=True)
            except Exception as e:
                logger.warning(f"[pack:{name}] fill[{i}] failed: {e}")
                continue
            if output_type == "atoms":
                t = d if isinstance(d, torch.Tensor) else torch.as_tensor(d)
                x_mm[lo:hi] = t.numpy().astype(_PACKED_X_DTYPE, copy=False)
            else:  # graph
                x_mm[lo:hi] = d["x"].numpy().astype(_PACKED_X_DTYPE, copy=False)
                elo, ehi = int(edge_off[i]), int(edge_off[i + 1])
                if ehi > elo:
                    ei_mm[:, elo:ehi] = (
                        d["edge_index"].numpy().astype(_PACKED_EI_DTYPE, copy=False)
                    )
                    ea_mm[elo:ehi] = (
                        d["edge_attr"].numpy().astype(_PACKED_EA_DTYPE, copy=False)
                    )

        x_mm.flush()
        del x_mm
        os.replace(x_tmp, x_path)
        if output_type == "graph":
            ei_mm.flush()
            del ei_mm
            ea_mm.flush()
            del ea_mm
            os.replace(ei_tmp, ei_path)
            os.replace(ea_tmp, ea_path)

        # Save offsets for loading
        np.save(ragged_memmap_path(name, "node_off", memmap_dir, filename), node_off)
        if output_type == "graph":
            np.save(
                ragged_memmap_path(name, "edge_off", memmap_dir, filename), edge_off
            )

        meta[name] = {
            "output": output_type,
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "total_atoms": total_atoms,
            "total_edges": total_edges,
            "x_dtype": _PACKED_X_DTYPE,
            "ei_dtype": _PACKED_EI_DTYPE,
            "ea_dtype": _PACKED_EA_DTYPE,
        }

        # Persist meta incrementally so a crash doesn't lose progress.
        with open(meta_path, "w") as f:
            json.dump(meta, f)


def embedding_modalities(modalities_spec):
    return {
        k: v
        for k, v in modalities_spec.items()
        if v["output"] == "embedding" and v.get("dim")
    }


def ragged_modalities(modalities_spec):
    """Modalities eligible for packed memmap storage (variable-length)."""
    return {
        k: v
        for k, v in modalities_spec.items()
        if v["output"] in ("graph", "atoms") and v.get("node_dim")
    }


def missing_memmaps(modalities_spec, memmap_dir, filename):
    return [
        name
        for name in embedding_modalities(modalities_spec)
        if not memmap_path(name, memmap_dir, filename).is_file()
    ]


def missing_ragged_memmaps(modalities_spec, memmap_dir, filename):
    packed = ragged_modalities(modalities_spec)
    if not packed:
        return []
    stem = filename.split(".")[0]
    meta_path = memmap_dir / f"{stem}_ragged_meta.json"
    meta = {}
    if meta_path.is_file():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    missing = []
    for name, spec in packed.items():
        if name not in meta:
            missing.append(name)
            continue
        if not ragged_memmap_path(name, "x", memmap_dir, filename).is_file():
            missing.append(name)
            continue
        if not ragged_memmap_path(name, "node_off", memmap_dir, filename).is_file():
            missing.append(name)
            continue
        if spec["output"] == "graph":
            if not ragged_memmap_path(name, "ei", memmap_dir, filename).is_file():
                missing.append(name)
                continue
            if not ragged_memmap_path(name, "ea", memmap_dir, filename).is_file():
                missing.append(name)
                continue
            if not ragged_memmap_path(name, "edge_off", memmap_dir, filename).is_file():
                missing.append(name)
                continue
    return missing


def memmap_barrier(modalities_spec, memmap_dir, filename):
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    else:
        import time

        while missing_memmaps(
            modalities_spec, memmap_dir, filename
        ) or missing_ragged_memmaps(modalities_spec, memmap_dir, filename):
            time.sleep(1)


def memmap_path(modality_name, memmap_dir, filename):
    stem = filename.split(".")[0]
    return memmap_dir / f"{stem}_{modality_name}.npy"


def ragged_memmap_path(modality_name, kind, memmap_dir, filename):
    """kind in {'x', 'ei', 'ea', 'node_off', 'edge_off'}."""
    stem = filename.split(".")[0]
    return memmap_dir / f"{stem}_{modality_name}_{kind}.npy"


def open_memmaps(memmap_dir, filename, logger):
    memmaps = {}
    meta_path = memmap_dir / f"{filename.split('.')[0]}_meta.json"
    logger.info(f"Opening memmaps from {meta_path}...")
    if not meta_path.is_file():
        return memmaps
    with open(meta_path) as f:
        meta = json.load(f)
    for name, info in meta.items():
        path = memmap_path(name, memmap_dir, filename)
        if path.is_file():
            memmaps[name] = np.memmap(
                path, dtype=info["dtype"], mode="r", shape=tuple(info["shape"])
            )
    return memmaps


def open_ragged_memmaps(memmap_dir, filename, logger):
    packed_memmaps = {}
    meta_path = memmap_dir / f"{filename.split('.')[0]}_ragged_meta.json"
    if not meta_path.is_file():
        return packed_memmaps
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read ragged meta {meta_path}: {e}")
        return packed_memmaps

    for name, info in meta.items():
        try:
            total_atoms = int(info["total_atoms"])
            node_dim = int(info["node_dim"])

            # Minimal information
            entry = {
                "x": np.memmap(
                    ragged_memmap_path(name, "x", memmap_dir, filename),
                    dtype=info.get("x_dtype", _PACKED_X_DTYPE),
                    mode="r",
                    shape=(total_atoms, node_dim),
                ),
                "node_off": np.load(
                    ragged_memmap_path(name, "node_off", memmap_dir, filename),
                    mmap_mode="r",
                ),
            }

            # Additional information for graph modalities
            if info.get("output") == "graph":
                total_edges = int(info["total_edges"])
                edge_dim = int(info["edge_dim"])
                entry["ei"] = np.memmap(
                    ragged_memmap_path(name, "ei", memmap_dir, filename),
                    dtype=info.get("ei_dtype", _PACKED_EI_DTYPE),
                    mode="r",
                    shape=(2, total_edges),
                )
                entry["ea"] = np.memmap(
                    ragged_memmap_path(name, "ea", memmap_dir, filename),
                    dtype=info.get("ea_dtype", _PACKED_EA_DTYPE),
                    mode="r",
                    shape=(total_edges, edge_dim),
                )
                entry["edge_off"] = np.load(
                    ragged_memmap_path(name, "edge_off", memmap_dir, filename),
                    mmap_mode="r",
                )
            entry["output"] = info.get("output")
            packed_memmaps[name] = entry
        except Exception as e:
            logger.error(f"Failed to open ragged memmap for '{name}': {e}")
    return packed_memmaps
