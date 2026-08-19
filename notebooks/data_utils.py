import io
import json
import base64
import numpy as np
import py3Dmol
import pandas as pd
from IPython.display import display, HTML
import sys
import os
import requests
from contextlib import contextmanager
from rdkit.Chem import Draw
from IPython.display import display, HTML
from ase.io import write
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
import logging
from tdc.utils import retrieve_label_name_list
from tdc.single_pred import ADME, Tox, HTS
from pandas.errors import ParserError
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer


def visualize_mol_grid2d(mols, rows=3, cols=3, figsize=(8, 8)):
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    for ax, mol in zip(axs.flatten(), mols):
        # Force 2D coordinates
        mol_2d = Chem.Mol(mol)
        AllChem.Compute2DCoords(mol_2d)

        img = Draw.MolToImage(mol_2d)
        ax.imshow(img)
        ax.set_title(Chem.MolToSmiles(mol), fontsize=4)
        ax.axis("off")
    plt.show()


def visualize_mol_2d_and_3d(mol, size=400):
    # --- 2D ---
    mol_2d = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_2d)
    img = Draw.MolToImage(mol_2d, size=(size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # --- 3D ---
    mol_3d = Chem.Mol(mol)
    AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol_3d)
    view = py3Dmol.view(width=size, height=size)
    view.addModel(Chem.MolToMolBlock(mol_3d), "sdf")
    view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
    view.setBackgroundColor("white")
    view.zoomTo()
    view_html = view._make_html()

    # --- Combine in HTML table ---
    html = f"""
    <table>
      <tr>
        <td style='text-align:center;vertical-align:top'>
          <b>2D Structure</b><br>
          <img src="data:image/png;base64,{img_b64}" width="{size}" height="{size}">
        </td>
        <td style='text-align:center;vertical-align:top'>
          <b>3D Structure</b><br>
          {view_html}
        </td>
      </tr>
    </table>
    """
    display(HTML(html))


def determine_bonds_with_timeout(xyz_block, charge, timeout=5):
    """Run RDKit bond determination in a separate process."""

    def worker(xyz_block, charge, queue):
        raw_mol = Chem.MolFromXYZBlock(xyz_block)
        mol = Chem.Mol(raw_mol)
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
            queue.put(mol)
        except Exception as e:
            queue.put(e)

    queue = mp.Queue()
    p = mp.Process(target=worker, args=(xyz_block, charge, queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return None

    result = queue.get()
    if isinstance(result, Exception):
        return None
    return result


def process_omol25_data(
    dataset, subset, data_id="biomolecules", multiprocess_timeout=10
):
    failed_indices = []
    data = []
    for index in tqdm(range(subset)):
        atoms = dataset.get_atoms(index)

        if atoms.info["data_id"] == data_id:
            with io.StringIO() as f:
                write(f, atoms, format="xyz")
                f.seek(0)
                xyz_block = f.read()

            mol = determine_bonds_with_timeout(
                xyz_block,
                charge=atoms.info["charge"],
                timeout=multiprocess_timeout,  # seconds
            )

            if mol is None:
                failed_indices.append(index)
                continue

            mol = Chem.RemoveHs(mol)
            smiles = Chem.MolToSmiles(mol)

            # Apply filters
            if "." in smiles or mol.GetNumAtoms() < 10:
                continue

            atoms_data = {
                # --- general info ---
                "smiles": smiles,
                "rdkit_mol": mol,
                "source": atoms.info["source"],
                "data_id": atoms.info["data_id"],
                "sid": atoms.info["sid"],
                "n_basis": atoms.info["n_basis"],
                "composition": atoms.info["composition"],
                # --- Global-level properties ---
                "global_charge": atoms.info["charge"],
                "global_spin": atoms.info["spin"],
                "global_num_atoms": atoms.info["num_atoms"],
                "global_num_electrons": atoms.info["num_electrons"],
                "global_num_ecp_electrons": atoms.info["num_ecp_electrons"],
                "global_n_scf_steps": atoms.info["n_scf_steps"],
                "global_nl_energy": atoms.info["nl_energy"],
                "global_integrated_densities": atoms.info[
                    "integrated_densities"
                ].tolist(),
                "global_homo_energy": atoms.info["homo_energy"].tolist(),
                "global_homo_lumo_gap": atoms.info["homo_lumo_gap"].tolist(),
                "global_s_squared": atoms.info["s_squared"],
                "global_s_squared_dev": atoms.info["s_squared_dev"],
                # --- Atom-level properties ---
                "atom_mulliken_charges": atoms.info["mulliken_charges"].tolist()
                if "mulliken_charges" in atoms.info
                else np.nan,
                "atom_lowdin_charges": atoms.info["lowdin_charges"].tolist()
                if "lowdin_charges" in atoms.info
                else np.nan,
                "atom_nbo_charges": atoms.info["nbo_charges"].tolist()
                if "nbo_charges" in atoms.info
                else np.nan,
            }
            data.append(atoms_data)

    return data, failed_indices


def process_tdc_data():
    tdc_datasets = {
        "adme": [
            # Caco-2 cell effective permeability
            "Caco2_Wang",
            # High permeability (1) or low-to-moderate permeability (0) in PAMPA assay
            "PAMPA_NCATS",
            # Human intestinal absorption (HIA) classification: absorbed (1) or not absorbed (0)
            "HIA_Hou",
            # Activity of Pgp inhibition
            "Pgp_Broccatelli",
            # Rate and extent to which the active ingredient or active moiety is absorbed from a drug produc
            "Bioavailability_Ma",
            # Ability to dissolve in lipid
            "Lipophilicity_AstraZeneca",
            # Aqueous Solubility
            "Solubility_AqSolDB",
            # Hydration free energy in water
            "HydrationFreeEnergy_FreeSolv",
            # Blood-Brain Barrier Penetration
            "BBB_Martins",
            # Percentage of a drug bound to plasma proteins in the blood.
            "PPBR_AZ",
            # The degree of a drug's concentration in body tissue compared to concentration in blood
            "VDss_Lombardo",
            # A drug that can inhibit these enzymes would mean poor metabolism to this drug and other drugs
            "CYP2C19_Veith",
            # Formation and breakdown of various molecules in the liver and CNS
            "CYP2D6_Veith",
            # Important enzyme in the body, mainly found in the liver and in the intestine
            "CYP3A4_Veith",
            # Metabolize some polycyclic aromatic hydrocarbons (PAHs) to carcinogenic intermediate
            "CYP1A2_Veith",
            # Plays a major role in the oxidation of both xenobiotic and endogenous compounds
            "CYP2C9_Veith",
            # Same as above
            "CYP2C9_Substrate_CarbonMangels",
            # Same as above
            "CYP2D6_Substrate_CarbonMangels",
            # Oxidizes small foreign organic molecules
            "CYP3A4_Substrate_CarbonMangels",
            # Concentration of the drug in the body to be reduced by half
            "Half_Life_Obach",
            # Rate at which the active drug is removed from the body
            "Clearance_Hepatocyte_AZ",
        ],
        "tox": [
            # Lethal dose 50
            "LD50_Zhu",
            # Human ether-à-go-go related gene crucial for the coordination of the heart's beating
            "hERG",
            # Percent inhibition at a 10µM concentration.
            ("herg_central", retrieve_label_name_list("herg_central")),
            # hERG (<10uM) and non-hERG (>=10uM) blockers
            "hERG_Karim",
            # short-term bacterial reverse mutation assay to induce genetic alterations
            "AMES",
            # Drug-induced liver injury
            "DILI",
            # Immune reaction in inherently susceptible individuals
            "Skin Reaction",
            # Any substance, radionuclide, or radiation that promotes carcinogenesis, the formation of cancer
            "Carcinogens_Lagunin",
            # Qualitative toxicity measurements such as nuclear receptors and stree response pathways
            ("Tox21", retrieve_label_name_list("Tox21")),
            # Qualitative results of over 600 assays
            ("Toxcast", retrieve_label_name_list("Toxcast")),
            # Drugs that have failed clinical trials due to toxicity
            "ClinTox",
        ],
        "hts": [
            # in-vitro screen in an infected cell-based assay
            "SARSCoV2_Vitro_Touret",
            # Crystallographic fragment screen against SARS-CoV-2 main protease at high resolution
            "SARSCoV2_3CLPro_Diamond",
            # Ability to inhibit HIV replication
            "HIV",
            # Orexin1 Receptor
            "orexin1_receptor_butkiewicz",
            # M1 Muscarinic Receptor Agonists
            "m1_muscarinic_receptor_agonists_butkiewicz",
            # M1 Muscarinic Receptor Antagonists
            "m1_muscarinic_receptor_antagonists_butkiewicz",
            # Potassium Ion Channel Kir2.1
            "potassium_ion_channel_kir2.1_butkiewicz",
            # KCNQ2 Potassium Channel
            "kcnq2_potassium_channel_butkiewicz",
            # Cav3 T-type Calcium Channels
            "cav3_t-type_calcium_channels_butkiewicz",
            # Choline Transporter
            "choline_transporter_butkiewicz",
            # Serine/Threonine Kinase 33
            "serine_threonine_kinase_33_butkiewicz",
            # Tyrosyl-DNA Phosphodiesterase
            "tyrosyl-dna_phosphodiesterase_butkiewicz",
        ],
    }

    data = []
    for data_type in tdc_datasets.keys():
        if data_type == "adme":
            module = ADME
        elif data_type == "tox":
            module = Tox
        elif data_type == "hts":
            module = HTS

        # Load datasets
        for dataset_name in tdc_datasets[data_type]:
            print(f"Loading dataset: {dataset_name} of type {data_type}")
            try:
                if isinstance(dataset_name, str):
                    raw_data = module(name=dataset_name)
                    split = raw_data.get_split()
                    for key, df in split.items():
                        tmp = df.copy()
                        tmp["data_type"] = data_type
                        tmp["split"] = key
                        tmp["dataset_name"] = dataset_name
                        data.append(tmp)
                elif isinstance(dataset_name, tuple):
                    for sub_dataset in dataset_name[1]:
                        print(f"Loading sub-dataset: {sub_dataset}")
                        raw_data = module(name=dataset_name[0], label_name=sub_dataset)
                        split = raw_data.get_split()
                        for key, df in split.items():
                            tmp = df.copy()
                            tmp["data_type"] = data_type
                            tmp["split"] = key
                            tmp["dataset_name"] = dataset_name[0] + "_" + sub_dataset
                            data.append(tmp)
            except ParserError as pe:
                print(f"Failed to load dataset: {dataset_name} (Error: {pe})")
                continue
    return data


@contextmanager
def suppress_all_output():
    old_stdout_obj = sys.stdout
    old_stderr_obj = sys.stderr
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    try:
        # Redirect low-level (FD 1 and 2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        sys.stdout = sys.stderr = os.fdopen(devnull, "w")

        yield
    finally:
        # Restore high-level
        sys.stdout = old_stdout_obj
        sys.stderr = old_stderr_obj

        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)

        os.close(old_stdout_fd)
        os.close(old_stderr_fd)


def process_data_from_sdf(
    sdf_path, xyz_directory, dataset_prefix, identifier, subindex=None
):
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)

    data = []
    for i, mol in tqdm(enumerate(supplier)):
        if mol is not None:
            properties = mol.GetPropsAsDict()

            data_dict = {}
            for prop in list(properties.keys()):
                data_dict[prop] = properties[prop]

            if identifier == "running_index":
                data_dict[identifier] = i

            # Save xyz
            xyz_file_path = (
                (f"{xyz_directory}{dataset_prefix}_{data_dict[identifier]}.xyz")
                if subindex is None
                else f"{xyz_directory}{dataset_prefix}_{data_dict[identifier]}_{subindex}.xyz"
            )
            Chem.MolToXYZFile(mol, xyz_file_path)

            # Validate xyz file exists and is non-empty
            if os.path.isfile(xyz_file_path) and os.path.getsize(xyz_file_path) > 0:
                data_dict["conformer_path"] = xyz_file_path
            else:
                logging.warning(f"XYZ file missing or empty: {xyz_file_path}")
                data_dict["conformer_path"] = None

            # Save sdf
            sdf_file_path = (
                (f"{xyz_directory}{dataset_prefix}_{data_dict[identifier]}.sdf")
                if subindex is None
                else f"{xyz_directory}{dataset_prefix}_{data_dict[identifier]}_{subindex}.sdf"
            )
            # Add the id to the sdf properties for traceability
            mol.SetProp(identifier, str(data_dict[identifier]))
            try:
                writer = Chem.SDWriter(sdf_file_path)
                writer.write(mol)
                writer.close()
                # Validate sdf file exists and is non-empty
                if os.path.isfile(sdf_file_path) and os.path.getsize(sdf_file_path) > 0:
                    data_dict["sdf_path"] = sdf_file_path
                else:
                    logging.warning(
                        f"SDF file missing or empty after write: {sdf_file_path}"
                    )
                    data_dict["sdf_path"] = None
            except Exception as e:
                logging.warning(f"Failed to write SDF {sdf_file_path}: {e}")
                data_dict["sdf_path"] = None

            # Overwrite with canonical smiles
            data_dict["smiles_processed"] = Chem.MolToSmiles(mol)

            data.append(data_dict)

    return pd.DataFrame(data)


def fetch_chembl_activities(chembl_id: str) -> list[dict]:
    all_activities = []

    while True:
        url = (
            f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
            f"?molecule_chembl_id={chembl_id}"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        activities = data.get("activities", [])
        if not activities:
            break

        for act in activities:
            # Quality filters
            if act.get("pchembl_value") is None:
                continue
            if act.get("standard_relation") != "=":
                continue
            if act.get("potential_duplicate") != 0:
                continue
            if act.get("data_validity_comment") is not None:
                continue
            if act.get("standard_flag") != 1:
                continue
            # Assay type filter
            if act.get("assay_type") not in ("B", "F"):
                continue
            # Activity type filter
            VALID_TYPES = {
                "IC50",
                "Ki",
                "Kd",
                "EC50",
                "AC50",
                "Potency",
                "ED50",  # potency
                "t1/2",
                "%bioavailability",
                "logD",
                "Papp",  # ADME
                "LD50",
                "CC50",  # toxicity
                "Solubility",
                "logP",  # physicochemical
            }
            if act.get("standard_type") not in VALID_TYPES:
                continue

            all_activities.append({
                # Identity
                "molecule_chembl_id": act["molecule_chembl_id"],
                "parent_molecule_chembl_id": act[
                    "parent_molecule_chembl_id"
                ],  # deduplication
                "target_chembl_id": act["target_chembl_id"],
                "target_pref_name": act["target_pref_name"],  # human-readable target
                "target_organism": act["target_organism"],  # e.g. Homo sapiens
                # Assay
                "assay_chembl_id": act["assay_chembl_id"],
                "assay_type": act["assay_type"],  # B/F/A/T/P
                "assay_description": act[
                    "assay_description"
                ],  # free text, useful for debugging
                "document_chembl_id": act["document_chembl_id"],  # source publication
                "document_year": act["document_year"],  # data recency
                # Activity
                "standard_type": act["standard_type"],  # IC50, Ki, etc.
                "pchembl_value": float(act["pchembl_value"]),
                "standard_value": act["standard_value"],  # in nM
                "standard_units": act["standard_units"],  # should always be nM
                "standard_relation": act["standard_relation"],  # should always be =
                # Debug/QC flags
                "data_validity_comment": act["data_validity_comment"],  # should be None
                "potential_duplicate": act["potential_duplicate"],  # should be 0
                "standard_flag": act["standard_flag"],  # should be 1
            })

        if data["page_meta"]["next"] is None:
            break

    return all_activities


def fetch_chunk(chunk_df: pd.DataFrame):
    """
    Processes a chunk of molecules and organizes results by Target ID.
    """
    local_dict = {}
    failed = []

    for _, row in chunk_df.iterrows():
        chembl_id = row["ChEMBL ID"]

        try:
            activities = fetch_chembl_activities(chembl_id)

            for act in activities:
                target_id = act["target_chembl_id"]
                if target_id not in local_dict:
                    local_dict[target_id] = []
                local_dict[target_id].append(act)

        except Exception as e:
            failed.append((chembl_id, str(e)))
            continue

    return local_dict, failed


def merge_dicts(dict_list):
    """Merge list of dicts into one."""
    merged = {}
    for d in dict_list:
        for act_id, entries in d.items():
            if act_id not in merged:
                merged[act_id] = []
            merged[act_id].extend(entries)
    return merged


def normalize_activity(row):
    POTENCY_TYPES = {"IC50", "Ki", "Kd", "EC50", "AC50", "Potency", "ED50"}
    TOX_TYPES = {"LD50", "CC50"}

    std_type = row["standard_type"]
    std_value = pd.to_numeric(row["standard_value"], errors="coerce")
    std_units = row["standard_units"]
    pchembl = row["pchembl_value"]

    # --- Potency: prefer pchembl (already -log10 molar) ---
    if std_type in POTENCY_TYPES:
        return pchembl, "pChEMBL (-log10 M)"

    # --- ADME ---
    if std_type == "t1/2":
        # Convert to hours if in minutes
        if std_units == "min":
            return std_value / 60, "h"
        return std_value, "h"

    if std_type == "%bioavailability":
        return std_value, "%"

    if std_type == "logD":
        return std_value, "logD"

    if std_type == "Papp":
        # Convert to 10^-6 cm/s if in cm/s
        if std_units == "cm/s":
            return std_value * 1e6, "10⁻⁶ cm/s"
        return std_value, "10⁻⁶ cm/s"

    # --- Toxicity: convert to uM ---
    if std_type in TOX_TYPES:
        if std_units == "nM":
            return std_value / 1000, "uM"
        if std_units == "mM":
            return std_value * 1000, "uM"
        return std_value, "uM"  # assume uM

    # --- Physicochemical ---
    if std_type == "Solubility":
        if std_units in ("ug/mL", "mg/L"):
            return np.log10(max(std_value, 1e-6)), "log10(ug/mL)"
        if std_units == "mg/mL":
            return np.log10(max(std_value * 1000, 1e-6)), "log10(ug/mL)"
        if std_units == "nM":
            return np.log10(max(std_value, 1e-6)), "log10(nM)"
        return np.log10(max(std_value, 1e-6)), f"log10({std_units})"

    if std_type == "logP":
        return std_value, "logP"

    return np.nan, "unknown"


def plot_histograms_nabla(df_merged, dft_cols):
    n_cols = 4
    n_rows = int(np.ceil(len(dft_cols) / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), dpi=100)
    plt.suptitle("DFT Properties raw", fontsize=12, color="#2c3e50", y=1.01)

    for ax, col in zip(axs.flatten(), dft_cols):
        sns.histplot(
            df_merged[col].dropna(),
            bins=50,
            ax=ax,
            color="#3F51B5",
            alpha=0.7,
            edgecolor="white",
            linewidth=0.3,
            kde=True,
            line_kws={"linewidth": 1.5},
        )
        ax.set_title(col.replace("DFT ", ""), fontsize=8, color="#2c3e50")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused axes
    for ax in axs.flatten()[len(dft_cols) :]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def robust_transform_nabla(df_merged):
    dft_cols = [c for c in df_merged.columns if c.startswith("DFT")]
    plot_histograms_nabla(df_merged, dft_cols)

    # Normalize
    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    df_norm = df_merged.copy()
    df_norm[dft_cols] = qt.fit_transform(df_merged[dft_cols])

    # After normalization
    plot_histograms_nabla(df_norm, dft_cols)

    return df_norm


def process_boltz_data(pred_dir, output_dir, max_idx=1000):
    print(f"Processing up to {max_idx} batches of boltz embeddings ...")
    records = []

    for idx in tqdm(range(max_idx)):
        yamls_dir = os.path.join(pred_dir, f"yamls_{idx}")
        if not os.path.isdir(yamls_dir):
            continue

        predictions_dir = os.path.join(yamls_dir, "predictions")
        if not os.path.isdir(predictions_dir):
            continue

        for protein_folder in os.listdir(predictions_dir):
            protein_path = os.path.join(predictions_dir, protein_folder)
            if not os.path.isdir(protein_path) or "SCN5A" in protein_folder:
                continue

            parts = protein_folder.rsplit("_", 1)
            if len(parts) != 2 or not parts[1].startswith("CHEMBL"):
                continue
            protein_name, chembl_id = parts

            has_affinity = False
            for fname in os.listdir(protein_path):
                if fname.startswith("affinity"):
                    json_path = os.path.join(protein_path, fname)
                    with open(json_path) as f:
                        data = json.load(f)
                    has_affinity = True

                    # Ensemble embeddings
                    emb1 = data.get("affinity_embedding1")
                    emb2 = data.get("affinity_embedding2")
                    combined = np.concatenate([np.array(emb1), np.array(emb2)])
                    emb_path = os.path.join(
                        output_dir, f"boltz_{protein_name}_{chembl_id}.npy"
                    )
                    np.save(emb_path, combined)
                    affinity = data.get("affinity_pred_value")
                elif fname.startswith("confidence"):
                    json_path = os.path.join(protein_path, fname)
                    with open(json_path) as f:
                        data = json.load(f)
                    conf = data.get("confidence_score")

            if has_affinity:
                records.append({
                    "yamls_idx": idx,
                    "protein_name": protein_name,
                    "chembl_id": chembl_id,
                    "boltz_pred": affinity,
                    "embedding_path": emb_path,
                    "structure_confidence": conf,
                })

    df = pd.DataFrame(records)
    return df


def dump_cols_as_array(df, columns_dict, group_name, dir):
    # ---- Metadata
    df_name = df["dataset"].iloc[0]
    df_type, group_type = group_name.split("_")[0], group_name.split("_")[1]
    cols = columns_dict[group_name]
    output_dir = f"{dir}/{group_type}"
    metadata_dir = f"{dir}/metadata"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # ---- Processing
    cols_exist = all(col in df.columns for col in cols)
    if not cols_exist:
        print(
            f"Warning: Not all columns for {group_name} exist in {df_name}. Skipping."
        )
        return

    print(f"Processing {group_name} for {df_name} (n={len(cols)})...")
    metadata_path = os.path.join(metadata_dir, f"{group_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({"columns": cols, "n_cols": len(cols)}, f, indent=2)

    for idx, row in tqdm(df.iterrows()):
        if group_type == "descriptors":
            filename = f"{df_name}_{df_type}_{idx}.npy"
        elif group_type == "targets":
            filename = f"{df_name}_{idx}.npy"

        out_path = os.path.join(output_dir, filename)
        vector = row[cols].values
        assert vector.shape[0] == len(cols)
        np.save(out_path, np.array(vector, dtype=np.float32))
        df.at[idx, f"{group_name}_path"] = out_path

    df.drop(columns=cols, inplace=True)
    return df


def merge_duplicates(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    first_index = group.index[0]

    for col in group.columns:
        # These don't need merge
        values = group[col].dropna().values
        if len(values) == 0:
            continue

        unique_values = set(values)
        if len(unique_values) > 1:
            # For nabla we take nabla, otherwise we just take the first one
            if col == "conformer_path":
                nabla_mask = group[col].astype(str).str.contains("nabla", na=False)
                if nabla_mask.any():
                    chosen_index = group[nabla_mask].index[0]
                else:
                    chosen_index = group[col].dropna().index[0]
                group.at[first_index, col] = group.at[chosen_index, col]
            # Store all dataset origins
            elif col == "dataset":
                group.at[first_index, col] = ",".join(
                    pd.unique(group[col].dropna().astype(str))
                )
            # This includes the target paths, where we also want to keep all available ones
            else:
                group.at[first_index, col] = values[0]
        else:
            group.at[first_index, col] = values[0]

    return group.loc[[first_index]]
