import os
from dotenv import load_dotenv

load_dotenv()

CLUSTER_SPLITS_DIR = os.getenv("CLUSTER_SPLITS_DIR")
BENCHMARKS_DIR = os.getenv("BENCHMARKS_DIR")
PROCESSED_DIR = os.getenv("PROCESSED_DIR")
PUBLIC_SPLIT_DATA = os.path.join(PROCESSED_DIR, "benchmark_data.csv")


PUBLIC_SPLITS = {
    "asap_potency": ["pIC50 (MERS-CoV Mpro)", "pIC50 (SARS-CoV-2 Mpro)"],
    "asap_admet": ["log_HLM", "log_KSOL", "LogD", "log_MDR1-MDCKII", "log_MLM"],
    "biogen_adme": [
        "hlm_clint",
        "mdr1_mdck_er",
        "plasma_protein_binding_human",
        "plasma_protein_binding_rat",
        "rlm_clint",
        "solubility_ph_6_8",
    ],
    "expansionrx": [
        "log_Caco-2 Permeability Efflux",
        "log_Caco-2 Permeability Papp A",
        "log_HLM CLint",
        "log_KSOL",
        "LogD",
        "log_MBPB",
        "log_MGMB",
        "log_MLM CLint",
        "log_MPPB",
    ],
    "pxr": ["pEC50"],
}

CLUSTER_SPLITS = {
    "ASAP_ADMET_HLM": CLUSTER_SPLITS_DIR + "asap_admet/HLM/log_HLM_clustersplits.sdf",
    "ASAP_ADMET_KSOL": CLUSTER_SPLITS_DIR
    + "asap_admet/KSOL/log_KSOL_clustersplits.sdf",
    "ASAP_ADMET_LogD": CLUSTER_SPLITS_DIR + "asap_admet/LogD/LogD_clustersplits.sdf",
    "ASAP_ADMET_MDR1": CLUSTER_SPLITS_DIR
    + "asap_admet/MDR1/log_MDR1-MDCKII_clustersplits.sdf",
    "ASAP_ADMET_MLM": CLUSTER_SPLITS_DIR + "asap_admet/MLM/log_MLM_clustersplits.sdf",
    "ASAP_POTENCY_MERS": CLUSTER_SPLITS_DIR
    + "asap_potency/MERS/asap_potency_MERS_clustersplits.sdf",
    "ASAP_POTENCY_SARS": CLUSTER_SPLITS_DIR
    + "asap_potency/SARS/asap_potency_SARS_clustersplits.sdf",
    "BIOGEN_HLM_Clint": CLUSTER_SPLITS_DIR
    + "biogen/HLM_CLint/biogen_HLM_CLint_clustersplits.sdf",
    "BIOGEN_MDR1": CLUSTER_SPLITS_DIR
    + "biogen/MDR1/biogen_MDR1_MDCK_clustersplits.sdf",
    "BIOGEN_PPBhuman": CLUSTER_SPLITS_DIR
    + "biogen/PPBhuman/biogen_PPBhuman_clustersplits.sdf",
    "BIOGEN_PPBrat": CLUSTER_SPLITS_DIR
    + "biogen/PPBrat/biogen_PPBrat_clustersplits.sdf",
    "BIOGEN_RLM_CLint": CLUSTER_SPLITS_DIR
    + "biogen/RLM_CLint/biogen_RLM_CLint_clustersplits.sdf",
    "BIOGEN_Solubility": CLUSTER_SPLITS_DIR
    + "biogen/solubility/biogen_LOG_SOLUBILITY_clustersplits.sdf",
    "OPENADMET_Caco2_efflux": CLUSTER_SPLITS_DIR
    + "openadmet/Caco2_Perm_Efflux/openadmet_Caco-2_Permeability_Efflux_clustersplits.sdf",
    "OPENADMET_Caco2_pappA": CLUSTER_SPLITS_DIR
    + "openadmet/Caco2_Perm_PappA/openadmet_Caco-2_Permeability_Papp_A_clustersplits.sdf",
    "OPENADMET_HLM_CLint": CLUSTER_SPLITS_DIR
    + "openadmet/HLM_CLint/openadmet_HLM_CLint_clustersplits.sdf",
    "OPENADMET_KSOL": CLUSTER_SPLITS_DIR
    + "openadmet/KSOL/openadmet_KSOL_clustersplits.sdf",
    "OPENADMET_LogD": CLUSTER_SPLITS_DIR
    + "openadmet/LogD/openadmet_LogD_clustersplits.sdf",
    "OPENADMET_MBPB": CLUSTER_SPLITS_DIR
    + "openadmet/MBPB/openadmet_MBPB_clustersplits.sdf",
    "OPENADMET_MGMB": CLUSTER_SPLITS_DIR
    + "openadmet/MGMB/openadmet_MGMB_clustersplits.sdf",
    "OPENADMET_MLM_CLint": CLUSTER_SPLITS_DIR
    + "openadmet/MLM_CLint/openadmet_MLM_CLint_clustersplits.sdf",
    "OPENADMET_MPPB": CLUSTER_SPLITS_DIR
    + "openadmet/MPPB/openadmet_MPPB_clustersplits.sdf",
    "OPENADMET_PXR": BENCHMARKS_DIR + "PXR/PXR_clustersplits.sdf",
}

# Check if each of the files exists
for name, rel_path in CLUSTER_SPLITS.items():
    if not os.path.isfile(rel_path):
        print(f"{name}: MISSING file at {rel_path}")


CV_DS_MAP = {
    "ASAP_ADMET_HLM": "asap_admet_HLM",
    "ASAP_ADMET_KSOL": "asap_admet_KSOL",
    "ASAP_ADMET_LogD": "asap_admet_LogD",
    "ASAP_ADMET_MDR1": "asap_admet_MDR1",
    "ASAP_ADMET_MLM": "asap_admet_MLM",
    "ASAP_POTENCY_MERS": "asap_potency_MERS",
    "ASAP_POTENCY_SARS": "asap_potency_SARS",
    "BIOGEN_HLM_Clint": "biogen_HLM_CLint",
    "BIOGEN_MDR1": "biogen_MDR1",
    "BIOGEN_PPBhuman": "biogen_PPBhuman",
    "BIOGEN_PPBrat": "biogen_PPBrat",
    "BIOGEN_RLM_CLint": "biogen_RLM_CLint",
    "BIOGEN_Solubility": "biogen_solubility",
    "OPENADMET_Caco2_efflux": "openadmet_Caco2_Perm_Efflux",
    "OPENADMET_Caco2_pappA": "openadmet_Caco2_Perm_PappA",
    "OPENADMET_HLM_CLint": "openadmet_HLM_CLint",
    "OPENADMET_KSOL": "openadmet_KSOL",
    "OPENADMET_LogD": "openadmet_LogD",
    "OPENADMET_MBPB": "openadmet_MBPB",
    "OPENADMET_MGMB": "openadmet_MGMB",
    "OPENADMET_MLM_CLint": "openadmet_MLM_CLint",
    "OPENADMET_MPPB": "openadmet_MPPB",
    "OPENADMET_PXR": "PXR",
}


PUBLIC_DS_MAP = {
    "asap_potency_pIC50 (MERS-CoV Mpro)": "pIC50 (MERS-CoV Mpro)",
    "asap_potency_pIC50 (SARS-CoV-2 Mpro)": "pIC50 (SARS-CoV-2 Mpro)",
    "asap_admet_log_HLM": "HLM",
    "asap_admet_log_KSOL": "KSOL",
    "asap_admet_LogD": "LogD",
    "asap_admet_log_MDR1-MDCKII": "MDR1-MDCKII",
    "asap_admet_log_MLM": "MLM",
    "expansionrx_log_Caco-2 Permeability Efflux": "Caco-2 Permeability Efflux",
    "expansionrx_log_Caco-2 Permeability Papp A": "Caco-2 Permeability Papp A",
    "expansionrx_log_HLM CLint": "HLM CLint",
    "expansionrx_log_KSOL": "KSOL",
    "expansionrx_LogD": "LogD",
    "expansionrx_log_MBPB": "MBPB",
    "expansionrx_log_MGMB": "MGMB",
    "expansionrx_log_MLM CLint": "MLM CLint",
    "expansionrx_log_MPPB": "MPPB",
    "pxr_pEC50": "pEC50",
}


BENCHMARK_FAMILIES = {
    "ExpansionRx": [
        "OPENADMET_Caco2_pappA",
        "OPENADMET_Caco2_efflux",
        "OPENADMET_LogD",
        "OPENADMET_KSOL",
        "OPENADMET_HLM_CLint",
        "OPENADMET_MLM_CLint",
        "OPENADMET_MBPB",
        "OPENADMET_MGMB",
        "OPENADMET_MPPB",
    ],
    "ASAP / Polaris": [
        "ASAP_POTENCY_MERS",
        "ASAP_POTENCY_SARS",
        "ASAP_ADMET_LogD",
        "ASAP_ADMET_KSOL",
        "ASAP_ADMET_HLM",
        "ASAP_ADMET_MLM",
        "ASAP_ADMET_MDR1",
    ],
    "PXR": [
        "OPENADMET_PXR",
    ],
    "Biogen ADME": [
        "BIOGEN_Solubility",
        "BIOGEN_HLM_Clint",
        "BIOGEN_RLM_CLint",
        "BIOGEN_PPBhuman",
        "BIOGEN_PPBrat",
        "BIOGEN_MDR1",
    ],
}
