"""
popv_annotation.py
AUTO-FETCH TABULA SAPIENS REFERENCES (NO HARDCODING)
"""

import os
import glob
import logging
import requests
from typing import Optional
import importlib.resources as pkg_resources

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
# Automatically detect tumor h5ad
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
# Fetch Tabula Sapiens file metadata
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
# Infer tissue from cancer
# ------------------------------------------------------------------------------

def cancer_to_tissue(cancer_type: str) -> str:
    return cancer_type.replace("_cancer", "").lower()

# ------------------------------------------------------------------------------
# Match correct reference file
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
# Download reference
# ------------------------------------------------------------------------------

def download_tabula_reference(cancer_type: str):

    files = fetch_tabula_file_metadata()

    selected = find_best_reference_file(cancer_type, files)

    if selected is None:
        raise ValueError(
            f"\nReference not found for '{cancer_type}' using Figshare API.\n\n"
            f"Please download manually from:\n"
            f"{TABULA_DOI_LINK}\n\n"
            f"Then pass using:\n"
            f"user_reference='path_to_reference.h5ad'\n"
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
# Select reference
# ------------------------------------------------------------------------------

def auto_select_reference(cancer_type, user_reference=None):

    if user_reference:
        if not os.path.exists(user_reference):
            raise FileNotFoundError(user_reference)
        return user_reference

    return download_tabula_reference(cancer_type)

# ------------------------------------------------------------------------------
# Utility functions
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
# CORE POPV
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

    # package ontology path (FIXED)
    with pkg_resources.as_file(
        pkg_resources.files("SCART.PopV.resources.ontology")
    ) as ontology_path:

        pq = Process_Query(
            query_adata=adata_query,
            ref_adata=adata_ref,
            ref_labels_key="cell_ontology_class",
            ref_batch_key=None,
            cl_obo_folder=str(ontology_path) + "/",
        )

    adata_processed = pq.adata

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
# Detect cancer type
# ------------------------------------------------------------------------------

def detect_cancer_type_from_h5ad(h5ad_file):

    adata = sc.read_h5ad(h5ad_file)

    if "cancer_type" in adata.uns:
        print(f"Detected cancer type: {adata.uns['cancer_type']}")
        return adata.uns["cancer_type"]

    raise ValueError(
        "Could not detect cancer type from h5ad.\n"
        "Provide user_reference manually."
    )

# ------------------------------------------------------------------------------
# AUTO ENTRY
# ------------------------------------------------------------------------------

def auto_run_popv(
    input_type="raw",
    nsamples=300,
    output_dir="popv_results",
    user_reference=None
):

    tumor_file = get_latest_tumor_h5ad()

    cancer_type = detect_cancer_type_from_h5ad(tumor_file)

    reference = auto_select_reference(
        cancer_type,
        user_reference
    )

    adata_query = sc.read_h5ad(tumor_file)
    adata_ref = sc.read_h5ad(reference)

    return run_popv_annotation(
        adata_query,
        adata_ref,
        obo_file="cl.obo",
        output_dir=output_dir,
        input_type=input_type,
        n_samples_per_label=nsamples
    )
