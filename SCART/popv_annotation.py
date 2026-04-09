"""
popv_annotation.py
AUTO-FETCH TABULA SAPIENS REFERENCES (NO HARDCODING)
"""

import os
import glob
import logging
import requests
from typing import Optional

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import popv
from popv.preprocessing import Process_Query
from popv.annotation import annotate_data
import popv.algorithms as alg
import urllib.request

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

REFERENCE_BASE_PATH = "popv_reference"
os.makedirs(REFERENCE_BASE_PATH, exist_ok=True)

FIGSHARE_ARTICLE_ID = "27921984"

TABULA_DOI_LINK = "https://doi.org/10.6084/m9.figshare.27921984"

# ------------------------------------------------------------------------------
# Automatically detect tumor h5ad (UPDATED)
# ------------------------------------------------------------------------------

def get_latest_tumor_h5ad(data_dir="GSE_data"):

    search_paths = [
        os.getcwd(),
        data_dir
    ]

    patterns = [
        "*_tumor.h5ad",
        "combined_tumor.h5ad"
    ]

    files = []

    for path in search_paths:
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(path, pattern)))

    if not files:
        raise FileNotFoundError(
            "No tumor h5ad found.\n"
            "Expected one of:\n"
            "- *_tumor.h5ad\n"
            "- combined_tumor.h5ad\n"
            "in current directory or GSE_data/"
        )

    files = list(set(files))

    return max(files, key=os.path.getctime)

# ------------------------------------------------------------------------------
# 🔥 STEP 1: Fetch ALL Tabula files dynamically
# ------------------------------------------------------------------------------

def fetch_tabula_file_metadata():

    url = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}/files"

    logger.info("Fetching Tabula Sapiens file list...")

    session = requests.Session()

    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    response = session.get(
        url,
        timeout=30,
        headers={"User-Agent": "curl/7.68.0"}
    )

    response.raise_for_status()

    files = response.json()

    h5ad_files = [
        f for f in files
        if f["name"].endswith(".h5ad")
    ]

    return h5ad_files

# ------------------------------------------------------------------------------
# 🔥 STEP 2: Infer tissue from cancer
# ------------------------------------------------------------------------------

def cancer_to_tissue(cancer_type: str) -> str:
    return cancer_type.replace("_cancer", "").lower()

# ------------------------------------------------------------------------------
# 🔥 STEP 3: Match correct file dynamically
# ------------------------------------------------------------------------------

def find_best_reference_file(cancer_type: str, files):

    tissue = cancer_to_tissue(cancer_type)

    logger.info(f"Matching tissue: {tissue}")

    for f in files:
        if f["name"].lower().startswith(tissue):
            return f

    for f in files:
        if tissue in f["name"].lower():
            return f

    return None

# ------------------------------------------------------------------------------
# 🔥 STEP 4: Download selected file
# ------------------------------------------------------------------------------

def download_tabula_reference(cancer_type: str):

    files = fetch_tabula_file_metadata()

    selected = find_best_reference_file(cancer_type, files)

    if selected is None:
        raise ValueError(
            f"\n❌ Reference not found for '{cancer_type}' using Figshare API.\n\n"
            f"👉 Figshare API is incomplete and may not list all tissues.\n\n"
            f"👉 Please download the correct reference manually from:\n"
            f"{TABULA_DOI_LINK}\n\n"
            f"👉 Then pass it using:\n"
            f"auto_run_popv(user_reference='path_to_reference.h5ad')\n"
        )

    filename = selected["name"]
    download_url = selected["download_url"]

    save_path = os.path.join(REFERENCE_BASE_PATH, filename)

    if os.path.exists(save_path):
        logger.info(f"Already exists: {filename}")
        return save_path

    logger.info(f"Downloading: {filename}")

    urllib.request.urlretrieve(download_url, save_path)

    logger.info(f"Saved to: {save_path}")

    return save_path

# ------------------------------------------------------------------------------
# UPDATED: Auto select reference
# ------------------------------------------------------------------------------

def auto_select_reference(cancer_type, user_reference=None):

    if user_reference:
        if not os.path.exists(user_reference):
            raise FileNotFoundError(user_reference)
        return user_reference

    return download_tabula_reference(cancer_type)

# ------------------------------------------------------------------------------
# Utility functions (UNCHANGED)
# ------------------------------------------------------------------------------

def fix_obs_dtypes(adata):
    for col in adata.obs.columns:
        if str(adata.obs[col].dtype) == "category":
            adata.obs[col] = adata.obs[col].astype(str)

def fix_layers(query, ref):
    shared = set(query.layers.keys()).intersection(ref.layers.keys())
    for a in [query, ref]:
        for k in list(a.layers.keys()):
            if k not in shared:
                del a.layers[k]

def force_float32_X(adata):
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr().astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

def sanitize_prediction_columns(adata):
    for col in adata.obs.columns:
        if col.endswith("_prediction"):
            adata.obs[col] = adata.obs[col].astype(str).str.lower()

def clean_obs_for_h5ad(adata):
    for col in adata.obs.columns:
        if adata.obs[col].dtype == "object":
            adata.obs[col] = adata.obs[col].astype(str)

def set_popv_input_matrix(adata, input_type):

    if input_type == "raw":
        if "raw_counts" in adata.layers:
            adata.X = adata.layers["raw_counts"]
        elif "counts" in adata.layers:
            adata.layers["raw_counts"] = adata.layers["counts"]
            adata.X = adata.layers["raw_counts"]

    elif input_type == "log1p":
        pass

    else:
        raise ValueError("input_type must be raw/log1p")

# ------------------------------------------------------------------------------
# CORE POPV (UPDATED METHODS LOGIC)
# ------------------------------------------------------------------------------

def run_popv_annotation(
    adata_query,
    adata_ref,
    obo_file,
    output_dir,
    input_type="raw",
    n_samples_per_label=300,
):

    os.makedirs(output_dir, exist_ok=True)

    fix_obs_dtypes(adata_query)
    fix_obs_dtypes(adata_ref)

    set_popv_input_matrix(adata_query, input_type)
    set_popv_input_matrix(adata_ref, "raw")

    force_float32_X(adata_query)
    force_float32_X(adata_ref)

    # 🔥 FIX: dynamic path to package ontology folder
    package_dir = os.path.dirname(os.path.abspath(__file__))
    ontology_path = os.path.join(os.path.dirname(popv.__file__), "resources", "ontology")
    
    pq = Process_Query(
        query_adata=adata_query,
        ref_adata=adata_ref,
        ref_labels_key="cell_ontology_class",
        ref_batch_key=None,
        cl_obo_folder=ontology_path + "/",
    )

    adata_processed = pq.adata

    # 🔥 NEW LOGIC (ONLY CHANGE)
    if input_type == "raw":
        methods = ["celltypist", "scvi", "scanvi", "rf", "svm"]
    else:
        methods = ["celltypist"]

    for m in methods:
        annotate_data(adata_processed, methods=[m])

    sanitize_prediction_columns(adata_processed)
    clean_obs_for_h5ad(adata_processed)

    out = os.path.join(output_dir, "final_popv_annotated.h5ad")
    adata_processed.write(out)

    return adata_processed

# ------------------------------------------------------------------------------
# Detect cancer type from h5ad 
# ------------------------------------------------------------------------------

def detect_cancer_type_from_h5ad(h5ad_file):

    import scanpy as sc

    adata = sc.read_h5ad(h5ad_file)

    if "cancer_type" in adata.uns:
        print(f"Detected cancer type (from uns): {adata.uns['cancer_type']}")
        return adata.uns["cancer_type"]

    text = ""

    if "disease" in adata.obs.columns:
        text += " ".join(adata.obs["disease"].astype(str).tolist()).lower()

    if "tissue" in adata.obs.columns:
        text += " ".join(adata.obs["tissue"].astype(str).tolist()).lower()

    cancer_keywords = [
        "ovarian", "breast", "lung", "colon", "colorectal",
        "prostate", "pancreatic", "liver", "hepatocellular",
        "kidney", "renal", "bladder", "gastric", "stomach",
        "melanoma", "glioma", "leukemia", "lymphoma",
        "myeloma", "sarcoma", "cervical", "endometrial",
        "uterus", "thyroid", "esophageal", "head and neck",
        "neuroblastoma"
    ]

    keyword_to_cancer = {
        "ovarian": "ovary_cancer",
        "breast": "breast_cancer",
        "lung": "lung_cancer",
        "colon": "large_intestine_cancer",
        "colorectal": "large_intestine_cancer",
        "prostate": "prostate_cancer",
        "pancreatic": "pancreas_cancer",
        "liver": "liver_cancer",
        "hepatocellular": "liver_cancer",
        "kidney": "kidney_cancer",
        "renal": "kidney_cancer",
        "bladder": "bladder_cancer",
        "gastric": "stomach_cancer",
        "stomach": "stomach_cancer",
        "melanoma": "skin_cancer",
        "glioma": "brain_cancer",
        "leukemia": "blood_cancer",
        "lymphoma": "lymph_node_cancer",
        "myeloma": "bone_marrow_cancer",
        "sarcoma": "muscle_cancer",
        "cervical": "uterus_cancer",
        "endometrial": "uterus_cancer",
        "uterus": "uterus_cancer",
        "thyroid": "thyroid_cancer",
        "esophageal": "esophagus_cancer",
        "head and neck": "head_neck_cancer",
        "neuroblastoma": "nerve_cancer"
    }

    for keyword in cancer_keywords:
        if keyword in text:
            detected = keyword_to_cancer.get(keyword)
            if detected:
                print(f"Detected cancer type (fallback): {detected}")
                return detected

    raise ValueError(
        "Could not detect cancer type from h5ad.\n"
        "👉 Please provide user_reference manually OR improve metadata."
    )

# ------------------------------------------------------------------------------
# AUTO ENTRY (UNCHANGED)
# ------------------------------------------------------------------------------

def auto_run_popv(
    input_type="raw",
    nsamples=300,
    output_dir="popv_results",
    user_reference=None
):

    if user_reference:
        print("\n========== Using USER-DEFINED reference ==========\n")
    else:
        print("\n========== Using AUTO Tabula Sapiens reference ==========\n")

    tumor_file = get_latest_tumor_h5ad()

    cancer_type = detect_cancer_type_from_h5ad(tumor_file)

    reference = auto_select_reference(cancer_type, user_reference)

    adata_query = sc.read_h5ad(tumor_file)
    adata_ref = sc.read_h5ad(reference)

    return run_popv_annotation(
        adata_query,
        adata_ref,
        obo_file="cl.obo",
        output_dir=output_dir,
        input_type=input_type,
        n_samples_per_label=nsamples
    )def auto_select_reference(cancer_type: str) -> str:

    cancer_type = cancer_type.strip().lower()

    if cancer_type in ["ovarian_cancer", "ovary"]:
        ref_path = os.path.join(
            REFERENCE_BASE_PATH,
            "Ovary_ref_TabulaSapiens.h5ad"
        )

        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference file not found: {ref_path}")

        logger.info(f"Using reference dataset: {ref_path}")
        return ref_path

    else:
        raise ValueError(f"No reference mapping defined for cancer type: {cancer_type}")

# ------------------------------------------------------------------------------
# Utility functions (unchanged)
# ------------------------------------------------------------------------------

def fix_obs_dtypes(adata: sc.AnnData) -> None:
    for col in adata.obs.columns:
        if str(adata.obs[col].dtype) == "category":
            adata.obs[col] = adata.obs[col].astype(str)

def fix_layers(query: sc.AnnData, ref: sc.AnnData) -> None:
    shared = set(query.layers.keys()).intersection(ref.layers.keys())

    for a in [query, ref]:
        for k in list(a.layers.keys()):
            if k not in shared:
                del a.layers[k]

    for k in shared:
        q_arr, r_arr = query.layers[k], ref.layers[k]
        if np.issubdtype(q_arr.dtype, np.number) and \
           np.issubdtype(r_arr.dtype, np.number):
            query.layers[k] = q_arr.astype(np.float32)
            ref.layers[k] = r_arr.astype(np.float32)
        else:
            del query.layers[k]
            del ref.layers[k]

def force_float32_X(adata: sc.AnnData) -> None:
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr().astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

def sanitize_prediction_columns(adata: sc.AnnData) -> None:
    for col in adata.obs.columns:
        if col.endswith("_prediction"):
            adata.obs[col] = (
                adata.obs[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

def clean_obs_for_h5ad(adata: sc.AnnData) -> None:
    for col in adata.obs.columns:
        if adata.obs[col].dtype == "object":
            adata.obs[col] = adata.obs[col].astype(str)

def set_popv_input_matrix(adata: sc.AnnData, input_type: str) -> None:

    if input_type == "raw":

        if "raw_counts" in adata.layers:
            logger.info("Using existing 'raw_counts' layer.")
            adata.X = adata.layers["raw_counts"].copy()

        elif "counts" in adata.layers:
            logger.info("Using 'counts' layer as raw input.")
            adata.layers["raw_counts"] = adata.layers["counts"].copy()
            adata.X = adata.layers["raw_counts"]

        else:
            logger.info(
                "No counts layer found. Assuming adata.X already contains raw counts."
            )

    elif input_type == "log1p":
        logger.info("Using log1p-normalized matrix (adata.X).")

    else:
        raise ValueError("input_type must be 'raw' or 'log1p'")

# ------------------------------------------------------------------------------
# CORE POPV RUNNER (unchanged)
# ------------------------------------------------------------------------------

def run_popv_annotation(
    adata_query: sc.AnnData,
    adata_ref: sc.AnnData,
    obo_file: str,
    output_dir: str,
    input_type: str = "raw",
    ref_label_key: str = "cell_ontology_class",
    query_batch_key: Optional[str] = None,
    n_samples_per_label: int = 300,
) -> sc.AnnData:

    os.makedirs(output_dir, exist_ok=True)

    original_query_cells = adata_query.obs_names.copy()

    fix_obs_dtypes(adata_query)
    fix_obs_dtypes(adata_ref)
    fix_layers(adata_query, adata_ref)

    set_popv_input_matrix(adata_query, input_type)
    set_popv_input_matrix(adata_ref, "raw")

    force_float32_X(adata_query)
    force_float32_X(adata_ref)

    # Remove .raw to prevent concat dtype crashes
    adata_query.raw = None
    adata_ref.raw = None

    cl_obo_folder = os.path.dirname(obo_file) + "/"

    pq = Process_Query(
        query_adata=adata_query,
        ref_adata=adata_ref,
        ref_labels_key=ref_label_key,
        ref_batch_key=None,
        cl_obo_folder=cl_obo_folder,
        query_batch_key=query_batch_key,
        prediction_mode="retrain",
        unknown_celltype_label="unknown",
        n_samples_per_label=n_samples_per_label,
        save_path_trained_models=os.path.join(output_dir, "trained_models"),
        hvg=None,
    )

    adata_processed = pq.adata

    if input_type == "log1p":
        methods = ["celltypist"]
        logger.info("Running ONLY CellTypist (log1p mode)")
    else:
        methods = [
            name for name in [
                "celltypist", "knn_on_bbknn", "knn_on_harmony",
                "knn_on_scanorama", "knn_on_scvi",
                "onclass", "rf", "svm", "scanvi"
            ]
            if hasattr(alg, name)
        ]

    for method in methods:
        try:
            logger.info(f"Running method: {method}")
            annotate_data(
                adata=adata_processed,
                methods=[method],
                save_path=None,
                methods_kwargs={method: {}},
            )
        except Exception as e:
            logger.warning(f"Skipping {method}: {e}")

    sanitize_prediction_columns(adata_processed)

    adata_processed = adata_processed[
        adata_processed.obs_names.isin(original_query_cells)
    ].copy()

    logger.info(
        f"Final object shape: {adata_processed.n_obs} × {adata_processed.n_vars}"
    )

    clean_obs_for_h5ad(adata_processed)

    final_path = os.path.join(output_dir, "final_popv_annotated.h5ad")
    adata_processed.write(final_path)

    logger.info("PopV annotation completed.")

    return adata_processed

# ------------------------------------------------------------------------------
# Automatically detect cancer type from h5ad
# ------------------------------------------------------------------------------

def detect_cancer_type_from_h5ad(h5ad_file: str) -> str:
    adata = sc.read_h5ad(h5ad_file)

    # First, check if uns contains cancer_type
    if "cancer_type" in adata.uns:
        cancer_type = adata.uns["cancer_type"]
        logger.info(f"Detected cancer type from .uns: {cancer_type}")
        return str(cancer_type).strip().lower()

    # Next, check obs columns for single cancer type
    elif "cancer_type" in adata.obs.columns:
        cancer_type = adata.obs["cancer_type"].unique()[0]
        logger.info(f"Detected cancer type from .obs: {cancer_type}")
        return str(cancer_type).strip().lower()

    # Try to infer from GSE ID in obs (works for combined_tumor.h5ad)
    elif "gse_id" in adata.obs.columns:
        gse_id = adata.obs["gse_id"].unique()[0]
        # Map known GSE IDs to cancer types (extendable)
        gse_cancer_map = {
            "GSE158937": "ovarian_cancer",
            # Add more known mappings if needed
        }
        cancer_type = gse_cancer_map.get(gse_id)
        if cancer_type:
            logger.info(f"Inferred cancer type {cancer_type} from GSE ID: {gse_id}")
            return cancer_type

    # Fallback: default to ovarian cancer
    logger.warning(
        f"Cannot detect cancer type from {h5ad_file}, defaulting to 'ovarian_cancer'"
    )
    return "ovarian_cancer"

# ------------------------------------------------------------------------------
# AUTO ENTRY (updated to handle all h5ad types)
# ------------------------------------------------------------------------------

def auto_run_popv(
    input_type: str = "raw",
    nsamples: int = 300,
    output_dir: str = "popv_results"
) -> sc.AnnData:

    tumor_file = get_latest_tumor_h5ad("GSE_data")
    cancer_type = detect_cancer_type_from_h5ad(tumor_file)

    reference_h5ad = auto_select_reference(cancer_type)

    adata_query = sc.read_h5ad(tumor_file)
    adata_ref = sc.read_h5ad(reference_h5ad)

    adata = run_popv_annotation(
        adata_query=adata_query,
        adata_ref=adata_ref,
        obo_file="/lustre/anas.a/Vinaya/scT-CAR_Designer/PopV/resources/ontology/cl.obo",
        output_dir=output_dir,
        input_type=input_type,
        n_samples_per_label=nsamples
    )

    return adata
