import json
import copy
import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from pathlib import Path
from torch import nn
from rdkit.Chem import AllChem
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from probes import TransformerProbe
from config import CLUSTER_SPLITS, PUBLIC_SPLIT_DATA


def compute_ecfp(smiles_list, radius=2, n_bits=2048):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp, dtype=np.float32))
        else:
            fps.append(np.zeros(n_bits, dtype=np.float32))
    return np.stack(fps)


def find_checkpoint_for_version(
    version_dir: str,
    cache_root: str = "~/.cache/stable-pretraining/runs",
    which: str | float = "latest",
) -> str | None:
    version_dir = Path(version_dir).expanduser().resolve()
    cache_root = Path(cache_root).expanduser().resolve()

    # All tfevents files. Format: events.out.tfevents.{timestamp}.{host}.{pid}.{idx}
    tfevents = list(version_dir.glob("events.out.tfevents.*"))
    if not tfevents:
        print(f"No tfevents file found in {version_dir}")
        return None

    def _ts(p: Path) -> float:
        return float(p.name.split(".")[3])

    if which == "latest":
        chosen = max(tfevents, key=_ts)
    elif which == "earliest":
        chosen = min(tfevents, key=_ts)
    else:
        target = float(which)
        chosen = min(tfevents, key=lambda p: abs(_ts(p) - target))

    if len(tfevents) > 1:
        print(
            f"Found {len(tfevents)} tfevents files; picked '{chosen.name}' (which={which!r})."
        )
    tb_timestamp = _ts(chosen)
    print(f"TensorBoard tfevents timestamp: {tb_timestamp:.3f}")

    # Scan all sidecar.json files in cache
    sidecars = sorted(cache_root.glob("**/sidecar.json"))
    if not sidecars:
        print(f"No sidecar.json files found under {cache_root}")
        return None

    version_name = version_dir.name
    candidates = []
    for sidecar_path in sidecars:
        try:
            with open(sidecar_path) as f:
                data = json.load(f)
            created_at = data.get("created_at")
            if created_at is None:
                continue
            trainer = data.get("hparams", {}).get("trainer", "")
            m = re.search(r"'version':\s*'([^']+)'", trainer)
            if m is None or m.group(1) != version_name:
                continue
            delta = abs(created_at - tb_timestamp)
            candidates.append((delta, sidecar_path.parent, data.get("run_id", "?")))
        except (json.JSONDecodeError, OSError):
            continue

    if not candidates:
        print(f"No cached run with version '{version_name}' found under {cache_root}.")
        return None

    # Sort by delta (smallest first)
    candidates.sort(key=lambda x: x[0])

    # Log top 3 closest
    print(f"\nTop 3 closest cache runs (out of {len(candidates)} total):")
    print(f"{'Rank':<5} {'Delta':<20} {'Run ID':<15} {'Path'}")
    print("-" * 90)
    for i, (delta, run_dir, run_id) in enumerate(candidates[:3]):
        if delta < 1:
            delta_str = f"{delta * 1000:.1f} ms"
        else:
            delta_str = f"{delta:.2f} s"
        print(f"{i + 1:<5} {delta_str:<20} {run_id:<15} {run_dir}")

    # Best match
    best_delta, best_dir, best_id = candidates[0]
    ckpt_dir = best_dir / "checkpoints"

    if best_delta > 60:
        print(
            f"\nWARNING: Best match has delta of {best_delta:.1f}s — may not be correct."
        )

    if ckpt_dir.is_dir():
        ckpts = list(ckpt_dir.iterdir())
        print(f"\nMatch: run_id={best_id}")
        print(f"Checkpoints dir: {ckpt_dir}")
        print(f"Available checkpoints: {[c.name for c in ckpts]}")
        return str(ckpt_dir)
    else:
        print(f"\nMatch found (run_id={best_id}) but no checkpoints/ directory exists.")
        print(f"Run dir: {best_dir}")
        return None


def load_data_splits(name):
    # Load SDF
    rel_path = CLUSTER_SPLITS[name]
    supplier = Chem.SDMolSupplier(str(rel_path))
    mols = [mol for mol in supplier if mol is not None]
    print(f"Loaded {len(mols)} molecules from {rel_path}")

    # Extract splits
    drop_keys = {"split1", "split2", "split3", "cluster_index"}
    data = []
    for mol in mols:
        split1 = mol.GetProp("split1")
        split2 = mol.GetProp("split2")
        split3 = mol.GetProp("split3")
        smiles = Chem.MolToSmiles(mol)

        props = mol.GetPropsAsDict()
        target_keys = [k for k in props if k not in drop_keys]
        if len(target_keys) != 1:
            raise ValueError(
                f"Expected exactly 1 target property after dropping {drop_keys}, "
                f"got {len(target_keys)}: {target_keys} for molecule {smiles}"
            )
        y = float(props[target_keys[0]])

        data.append({
            "smiles": smiles,
            "split1": split1,
            "split2": split2,
            "split3": split3,
            "y": y,
        })

    return pd.DataFrame(data)


def load_public_split_data(dataset, endpoint):
    public_split_data = pd.read_csv(PUBLIC_SPLIT_DATA)
    subset = public_split_data[public_split_data["dataset"] == dataset]
    subset = subset[["smiles", endpoint, "provided_split"]]
    subset.rename(columns={endpoint: "y", "provided_split": "split1"}, inplace=True)
    return subset


def _get_modality_col_map(model):
    """Get mapping from modality name -> df column name for precomputed modalities."""
    col_map = {}
    for m in model.model.modalities_spec:
        colname = m.get("colname")
        if colname and m.get("input") in ("precomputed", "precomputed_node"):
            col_map[m["name"]] = colname
    return col_map


def _build_embeddings_data(df_slice, model):
    """Build list of dicts mapping modality_name -> file_path for a batch slice."""
    col_map = _get_modality_col_map(model)
    # Only use columns actually present in the dataframe
    col_map = {k: v for k, v in col_map.items() if v in df_slice.columns}

    embeddings_data = []
    for _, row in df_slice.iterrows():
        sample = {}
        for mod_name, col_name in col_map.items():
            if pd.notna(row.get(col_name)):
                sample[mod_name] = row[col_name]
        embeddings_data.append(sample)
    return embeddings_data


def featurize_dataset(df, model, batch_size=512, use_all_modalities=False):
    embeddings = []
    latent_embeddings = []
    cls = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch_smiles = df["smiles"].iloc[i : i + batch_size].tolist()

        embeddings_data = None
        if use_all_modalities:
            embeddings_data = _build_embeddings_data(df.iloc[i : i + batch_size], model)

        with torch.no_grad():
            batch_preds, batch_cls, batch_embeddings = model(
                batch_smiles, embeddings_data=embeddings_data
            )
        embeddings.append(batch_preds.cpu())
        latent_embeddings.append(batch_embeddings.cpu())
        cls.append(batch_cls.cpu())

    # Reshape tensors
    embeddings = torch.vstack(embeddings)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    latent_embeddings = torch.vstack(latent_embeddings)
    latent_embeddings = latent_embeddings.reshape(latent_embeddings.shape[0], -1)

    return embeddings, latent_embeddings, torch.vstack(cls)


def _split_metrics(
    is_cls,
    train_idx,
    test_idx,
    y_true_test,
    y_pred_test,
    y_true_train=None,
    y_pred_train=None,
):
    """Build a per-split result dict (metrics + book-keeping) shared by all CV loops."""
    y_true_test = np.ravel(np.asarray(y_true_test))
    y_pred_test = np.ravel(np.asarray(y_pred_test))
    result = {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "y_true_test": y_true_test.tolist(),
        "y_pred_test": y_pred_test.tolist(),
    }
    if is_cls:
        result["test_roc_auc"] = float(roc_auc_score(y_true_test, y_pred_test))
    else:
        y_true_train = np.ravel(np.asarray(y_true_train))
        y_pred_train = np.ravel(np.asarray(y_pred_train))
        result["test_mae"] = float(mean_absolute_error(y_true_test, y_pred_test))
        result["train_mae"] = float(mean_absolute_error(y_true_train, y_pred_train))
        result["test_r2"] = float(r2_score(y_true_test, y_pred_test))
    return result


def _aggregate_cv(results, variant, is_cls, verbose=True, label=None):
    """Aggregate per-split dicts into mean/std summary, tag the variant, optionally print."""
    label = label or variant
    split_results = [r for r in results.values() if isinstance(r, dict)]

    def _mean_std(key):
        vals = [r[key] for r in split_results]
        return float(np.mean(vals)), float(np.std(vals))

    if is_cls:
        results["mean_test_roc_auc"], results["std_test_roc_auc"] = _mean_std(
            "test_roc_auc"
        )
        results["metric"] = "roc_auc"
        msg = (
            f"[{label}] test ROC-AUC: "
            f"{results['mean_test_roc_auc']:.4f} ± {results['std_test_roc_auc']:.4f}"
        )
    else:
        results["mean_train_mae"], results["std_train_mae"] = _mean_std("train_mae")
        results["mean_test_mae"], results["std_test_mae"] = _mean_std("test_mae")
        results["mean_test_r2"], results["std_test_r2"] = _mean_std("test_r2")
        results["metric"] = "mae"
        msg = (
            f"[{label}] train MAE: {results['mean_train_mae']:.4f} ± {results['std_train_mae']:.4f} "
            f"/ test MAE: {results['mean_test_mae']:.4f} ± {results['std_test_mae']:.4f} "
            f"| R2: {results['mean_test_r2']:.4f} ± {results['std_test_r2']:.4f}"
        )

    results["variant"] = variant
    if verbose:
        print(msg)
    return results


def run_torch_cross_validation(
    df,
    features,
    y,
    variant="linear",
    split_cols=("split1", "split2", "split3"),
    train_value="train",
    test_value="test",
    epochs=300,
    batch_size=256,
    hidden_dim=512,
    lr=1e-3,
    weight_decay=1e-4,
    seed=0,
    verbose=True,
    metric="mae",
    kwargs={},
):
    device = None or ("cuda" if torch.cuda.is_available() else "cpu")
    is_cls = metric == "roc_auc"

    # Load probes
    if variant == "linear":
        probe_fn = nn.Sequential(nn.Linear(features.shape[1], 1))
    elif variant == "nonlinear":
        probe_fn = nn.Sequential(
            nn.Linear(features.shape[1], hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    elif variant == "transformer":
        n_tokens = features.shape[1] // hidden_dim
        assert features.shape[1] == n_tokens * hidden_dim, (
            f"Feature dim {features.shape[1]} not divisible by hidden_dim {hidden_dim}"
        )
        probe_fn = TransformerProbe(n_tokens=n_tokens, token_dim=hidden_dim, **kwargs)

    X = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=device).reshape(
        -1, 1
    )
    assert len(X) == len(y) == len(df), (len(X), len(y), len(df))

    torch.manual_seed(seed)
    results = {}

    for split_col in split_cols:
        labels = df[split_col].astype(str).str.lower().values
        train_idx = np.where(labels == train_value)[0]
        test_idx = np.where(labels == test_value)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            print(
                f"{split_col}: skipping (train={len(train_idx)}, test={len(test_idx)})"
            )
        else:
            if verbose:
                print(f"{split_col}: train={len(train_idx)}, test={len(test_idx)}")

        train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        test_idx_t = torch.as_tensor(test_idx, dtype=torch.long, device=device)

        probe = copy.deepcopy(probe_fn).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss() if is_cls else nn.L1Loss()

        for _ in range(epochs):
            probe.train()
            perm = train_idx_t[torch.randperm(len(train_idx_t), device=device)]
            for j in range(0, len(perm), batch_size):
                b = perm[j : j + batch_size]
                pred = probe(X[b])
                loss = loss_fn(pred, y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()

        probe.eval()
        with torch.no_grad():
            test_logits = probe(X[test_idx_t])
            train_logits = probe(X[train_idx_t])

        if is_cls:
            test_probs = torch.sigmoid(test_logits).cpu().numpy()
            results[split_col] = _split_metrics(
                True,
                train_idx,
                test_idx,
                y_true_test=y[test_idx_t].cpu().numpy(),
                y_pred_test=test_probs,
            )
        else:
            results[split_col] = _split_metrics(
                False,
                train_idx,
                test_idx,
                y_true_test=y[test_idx_t].cpu().numpy(),
                y_pred_test=test_logits.cpu().numpy(),
                y_true_train=y[train_idx_t].cpu().numpy(),
                y_pred_train=train_logits.cpu().numpy(),
            )

    return _aggregate_cv(results, variant, is_cls, verbose)


def _run_batch(ft_model, batch_smiles, device):
    """Forward pass through the model returning CLS embeddings."""
    batch = ft_model.smiles_to_batch(batch_smiles).to(device)
    modalities = [
        f"{m['name']}_x_ptr" if f"{m['name']}_x_ptr" in batch else None
        for m in ft_model.model.modalities_spec
    ]
    bs = len(batch_smiles)
    active_per_mod = [
        batch[k][1:] - batch[k][:-1]
        if k is not None
        else torch.zeros(bs, device=device)
        for k in modalities
    ]
    active_mask = (torch.stack(active_per_mod) > 0).t()
    targets = ft_model.model.encode(batch, modalities, active_mask)
    _, cls, embeddings = ft_model.model.predict(
        targets, active_mask.clone(), active_mask
    )
    return cls


def _finetune_cv(
    df,
    ft_model,
    y,
    variant,
    split_cols,
    train_value,
    test_value,
    epochs,
    batch_size,
    head_hidden_dim,
    lr,
    weight_decay,
    seed,
    verbose,
):
    """Shared cross-validation loop for finetuning and LoRA."""
    device = next(ft_model.parameters()).device
    torch.manual_seed(seed)
    np.random.seed(seed)

    smiles = df["smiles"].astype(str).tolist()
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
    baseline_state = {k: v.detach().clone() for k, v in ft_model.state_dict().items()}

    ft_model.eval()
    with torch.no_grad():
        cls_dim = _run_batch(ft_model, [smiles[0]], device).shape[-1]

    results = {}
    for split_col in split_cols:
        labels = df[split_col].astype(str).str.lower().values
        train_idx = np.where(labels == train_value)[0]
        test_idx = np.where(labels == test_value)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        ft_model.load_state_dict(baseline_state)
        head = nn.Sequential(
            nn.Linear(cls_dim, head_hidden_dim),
            nn.BatchNorm1d(head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 1),
        ).to(device)

        trainable = [p for p in ft_model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable + list(head.parameters()), lr=lr, weight_decay=weight_decay
        )
        loss_fn = nn.L1Loss()

        recent_train = []
        recent_test = []
        pbar = tqdm(range(epochs), desc=f"{split_col}")
        for e in pbar:
            ft_model.train()
            head.train()
            perm = train_idx[np.random.permutation(len(train_idx))]
            epoch_loss = 0.0
            n_batches = 0
            for j in range(0, len(perm), batch_size):
                b = perm[j : j + batch_size]
                pred = head(_run_batch(ft_model, [smiles[k] for k in b], device))
                loss = loss_fn(pred, y_t[b].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1
            recent_train = (recent_train + [epoch_loss / n_batches])[-3:]
            # Quick test loss check
            ft_model.eval()
            head.eval()
            with torch.no_grad():
                test_loss = nn.functional.l1_loss(
                    torch.cat([
                        head(
                            _run_batch(
                                ft_model,
                                [smiles[k] for k in test_idx[j : j + batch_size]],
                                device,
                            )
                        )
                        for j in range(0, len(test_idx), batch_size)
                    ]),
                    y_t[test_idx].to(device),
                ).item()
            recent_test = (recent_test + [test_loss])[-3:]
            pbar.set_postfix_str(
                f"train={'→'.join(f'{l:.3f}' for l in recent_train)} "
                f"test={'→'.join(f'{l:.3f}' for l in recent_test)}"
            )

        ft_model.eval()
        head.eval()
        with torch.no_grad():
            test_pred = torch.cat([
                head(
                    _run_batch(
                        ft_model,
                        [smiles[k] for k in test_idx[j : j + batch_size]],
                        device,
                    )
                ).cpu()
                for j in range(0, len(test_idx), batch_size)
            ])
            train_pred = torch.cat([
                head(
                    _run_batch(
                        ft_model,
                        [smiles[k] for k in train_idx[j : j + batch_size]],
                        device,
                    )
                ).cpu()
                for j in range(0, len(train_idx), batch_size)
            ])

        results[split_col] = _split_metrics(
            False,
            train_idx,
            test_idx,
            y_true_test=y_t[test_idx].numpy(),
            y_pred_test=test_pred.numpy(),
            y_true_train=y_t[train_idx].numpy(),
            y_pred_train=train_pred.numpy(),
        )
        del head, opt
        if device == "cuda":
            torch.cuda.empty_cache()

    return _aggregate_cv(results, variant, is_cls=False, verbose=verbose)


def run_finetune_cross_validation(
    df,
    model,
    y,
    split_cols=("split1", "split2", "split3"),
    train_value="train",
    test_value="test",
    epochs=60,
    batch_size=64,
    head_hidden_dim=512,
    lr=5e-5,
    weight_decay=1e-4,
    seed=0,
    verbose=True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ft_model = copy.deepcopy(model).to(device)
    return _finetune_cv(
        df,
        ft_model,
        y,
        "finetuned",
        split_cols,
        train_value,
        test_value,
        epochs,
        batch_size,
        head_hidden_dim,
        lr,
        weight_decay,
        seed,
        verbose,
    )


def run_lora_cross_validation(
    df,
    model,
    y,
    split_cols=("split1", "split2", "split3"),
    train_value="train",
    test_value="test",
    epochs=60,
    batch_size=64,
    head_hidden_dim=512,
    lr=5e-5,
    weight_decay=1e-4,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    seed=0,
    verbose=True,
):
    from peft import LoraConfig, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ft_model = copy.deepcopy(model).to(device)

    # Apply LoRA only to the shared transformer predictor (not modality encoders)
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["out_proj", "linear1", "linear2"],
        bias="none",
    )
    ft_model.model.transformer_head = get_peft_model(
        ft_model.model.transformer_head, peft_config
    )
    if verbose:
        ft_model.model.transformer_head.print_trainable_parameters()

    return _finetune_cv(
        df,
        ft_model,
        y,
        "lora",
        split_cols,
        train_value,
        test_value,
        epochs,
        batch_size,
        head_hidden_dim,
        lr,
        weight_decay,
        seed,
        verbose,
    )


def run_rf_cross_validation(
    df,
    features,
    y,
    split_cols=("split1", "split2", "split3"),
    train_value="train",
    test_value="test",
    n_estimators=500,
    max_depth=None,
    n_jobs=-1,
    seed=0,
    verbose=True,
    metric="mae",
    **rf_kwargs,
):
    is_cls = metric == "roc_auc"
    X = (
        features.cpu().numpy()
        if isinstance(features, torch.Tensor)
        else np.asarray(features)
    )
    y = np.asarray(y).reshape(-1)
    assert len(X) == len(y) == len(df), (len(X), len(y), len(df))

    results = {}
    feature_importances = {}
    for split_col in split_cols:
        labels = df[split_col].astype(str).str.lower().values
        train_idx = np.where(labels == train_value)[0]
        test_idx = np.where(labels == test_value)[0]

        if is_cls:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=n_jobs,
                random_state=seed,
                **rf_kwargs,
            )
            rf.fit(X[train_idx], y[train_idx].astype(int))
            test_probs = rf.predict_proba(X[test_idx])[:, 1]
            feature_importances[split_col] = rf.feature_importances_
            results[split_col] = _split_metrics(
                True, train_idx, test_idx, y[test_idx], test_probs
            )
        else:
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=n_jobs,
                random_state=seed,
                **rf_kwargs,
            )
            rf.fit(X[train_idx], y[train_idx])
            test_pred = rf.predict(X[test_idx])
            train_pred = rf.predict(X[train_idx])
            feature_importances[split_col] = rf.feature_importances_
            results[split_col] = _split_metrics(
                False,
                train_idx,
                test_idx,
                y[test_idx],
                test_pred,
                y[train_idx],
                train_pred,
            )

    _aggregate_cv(results, "rf", is_cls, verbose, label="RF")
    return results, feature_importances


def run_tabicl_cross_validation(
    df,
    features,
    y,
    split_cols=("split1", "split2", "split3"),
    train_value="train",
    test_value="test",
    seed=0,
    verbose=True,
    metric="mae",
):
    from tabicl import TabICLClassifier, TabICLRegressor

    is_cls = metric == "roc_auc"
    X = (
        features.cpu().numpy()
        if isinstance(features, torch.Tensor)
        else np.asarray(features)
    )
    y_arr = np.asarray(y).reshape(-1)
    assert len(X) == len(y_arr) == len(df), (len(X), len(y_arr), len(df))

    results = {}
    for split_col in split_cols:
        labels = df[split_col].astype(str).str.lower().values
        train_idx = np.where(labels == train_value)[0]
        test_idx = np.where(labels == test_value)[0]

        if is_cls:
            model_icl = TabICLClassifier(random_state=seed)
            model_icl.fit(X[train_idx], y_arr[train_idx].astype(int))
            test_probs = model_icl.predict_proba(X[test_idx])[:, 1]
            results[split_col] = _split_metrics(
                True, train_idx, test_idx, y_arr[test_idx], test_probs
            )
        else:
            model_icl = TabICLRegressor(random_state=seed)
            model_icl.fit(X[train_idx], y_arr[train_idx])
            test_pred = model_icl.predict(X[test_idx])
            train_pred = model_icl.predict(X[train_idx])
            results[split_col] = _split_metrics(
                False,
                train_idx,
                test_idx,
                y_arr[test_idx],
                test_pred,
                y_arr[train_idx],
                train_pred,
            )

    return _aggregate_cv(results, "tabicl", is_cls, verbose, label="TabICL")


def encode_clamp(smiles_list, device="cpu", batch_size=512):
    import clamp as clamp_lib

    clamp_model = clamp_lib.CLAMP(device=device)
    clamp_model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i : i + batch_size]
            embeddings.append(clamp_model.encode_smiles(batch).cpu().numpy())
    return np.vstack(embeddings)


def run_clamp_cross_validation(
    df,
    y,
    split_cols=("split1", "split2", "split3"),
    train_value="train",
    test_value="test",
    seed=0,
    verbose=True,
    metric="mae",
    device="cpu",
    batch_size=512,
    clamp_embeddings=None,
):
    # from xgboost import XGBClassifier
    from tabicl import TabICLRegressor

    is_cls = metric == "roc_auc"
    y_arr = np.asarray(y).reshape(-1)

    if clamp_embeddings is not None:
        X = (
            clamp_embeddings.cpu().numpy()
            if isinstance(clamp_embeddings, torch.Tensor)
            else np.asarray(clamp_embeddings)
        )
    else:
        smiles_list = df["smiles"].astype(str).tolist()
        if verbose:
            print(f"Encoding {len(smiles_list)} SMILES with CLAMP...")
        X = encode_clamp(smiles_list, device=device, batch_size=batch_size)

    assert len(X) == len(y_arr) == len(df), (len(X), len(y_arr), len(df))

    results = {}
    for split_col in split_cols:
        labels = df[split_col].astype(str).str.lower().values
        train_idx = np.where(labels == train_value)[0]
        test_idx = np.where(labels == test_value)[0]

        if is_cls:
            model_icl = TabICLRegressor(random_state=seed)
            model_icl.fit(X[train_idx], y_arr[train_idx])
            test_pred = model_icl.predict(X[test_idx])
            train_pred = model_icl.predict(X[train_idx])
            results[split_col] = _split_metrics(
                False,
                train_idx,
                test_idx,
                y_arr[test_idx],
                test_pred,
                y_arr[train_idx],
                train_pred,
            )
        else:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_dim = X.shape[1]
            mlp = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 1),
            ).to(device)
            opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
            loss_fn = nn.L1Loss()
            X_train_t = torch.tensor(X[train_idx], dtype=torch.float32, device=device)
            y_train_t = torch.tensor(
                y_arr[train_idx], dtype=torch.float32, device=device
            ).reshape(-1, 1)
            mlp.train()
            for _ in range(300):
                perm = torch.randperm(len(X_train_t), device=device)
                for j in range(0, len(perm), 256):
                    b = perm[j : j + 256]
                    opt.zero_grad()
                    loss_fn(mlp(X_train_t[b]), y_train_t[b]).backward()
                    opt.step()
            mlp.eval()
            with torch.no_grad():
                test_pred = (
                    mlp(torch.tensor(X[test_idx], dtype=torch.float32, device=device))
                    .cpu()
                    .numpy()
                    .flatten()
                )
                train_pred = (
                    mlp(torch.tensor(X[train_idx], dtype=torch.float32, device=device))
                    .cpu()
                    .numpy()
                    .flatten()
                )
            results[split_col] = _split_metrics(
                False,
                train_idx,
                test_idx,
                y_arr[test_idx],
                test_pred,
                y_arr[train_idx],
                train_pred,
            )

    return _aggregate_cv(results, "clamp", is_cls, verbose, label="CLAMP")


def process_feature_importances(
    model, importances, modality_size=512, split_cols=("split1", "split2", "split3")
):

    importance_results = []
    for split in split_cols:
        split_importances = importances.get(split)
        modality_names = [m["name"] for m in model.model.modalities_spec]
        n_mods = len(modality_names)

        assert split_importances.shape[0] == n_mods * modality_size, (
            f"Expected {n_mods * modality_size} features, got {split_importances.shape[0]}"
        )

        # Agggregate importance within each modality
        modality_importance = {}
        for i, name in enumerate(modality_names):
            start = i * modality_size
            end = start + modality_size
            modality_importance[name] = split_importances[start:end].sum()

        importance_results.append(
            sorted(modality_importance.items(), key=lambda x: x[1], reverse=True)
        )
    return importance_results
