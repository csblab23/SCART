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
import json

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

    search_paths = [os.getcwd(), data_dir]
    patterns = ["*_tumor.h5ad", "combined_tumor.h5ad"]

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
        )

    files = list(set(files))
    return max(files, key=os.path.getctime)

# ------------------------------------------------------------------------------
# Fetch Tabula Sapiens metadata
# ------------------------------------------------------------------------------

def fetch_tabula_file_metadata():

    url = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}/files"

    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1,
                  status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    response = session.get(url, timeout=30)
    response.raise_for_status()

    files = response.json()
    return [f for f in files if f["name"].endswith(".h5ad")]

# ------------------------------------------------------------------------------
# Reference selection
# ------------------------------------------------------------------------------

def cancer_to_tissue(cancer_type: str) -> str:
    return cancer_type.replace("_cancer", "").lower()

def find_best_reference_file(cancer_type: str, files):

    tissue = cancer_to_tissue(cancer_type)

    for f in files:
        if f["name"].lower().startswith(tissue):
            return f

    for f in files:
        if tissue in f["name"].lower():
            return f

    return None

def download_tabula_reference(cancer_type: str):

    files = fetch_tabula_file_metadata()
    selected = find_best_reference_file(cancer_type, files)

    if selected is None:
        raise ValueError("Reference not found")

    filename = selected["name"]
    download_url = selected["download_url"]
    save_path = os.path.join(REFERENCE_BASE_PATH, filename)

    if os.path.exists(save_path):
        return save_path

    urllib.request.urlretrieve(download_url, save_path)
    return save_path

def auto_select_reference(cancer_type, user_reference=None):

    if user_reference:
        return user_reference

    return download_tabula_reference(cancer_type)

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def fix_obs_dtypes(adata):
    for col in adata.obs.columns:
        if str(adata.obs[col].dtype) == "category":
            adata.obs[col] = adata.obs[col].astype(str)

def force_float32_X(adata):
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr().astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

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

# ------------------------------------------------------------------------------
# ✅ Ontology normalization (SAFE)
# ------------------------------------------------------------------------------

def normalize_predictions_to_ontology(adata, ontology_json_path):

    import difflib

    with open(ontology_json_path) as f:
        cl = json.load(f)

    labels = [n["lbl"] for n in cl["nodes"] if "lbl" in n]
    name_lookup = {l.lower(): l for l in labels}

    def map_label(label):

        if not isinstance(label, str):
            return label

        key = label.strip().lower()

        if key in name_lookup:
            return name_lookup[key]

        match = difflib.get_close_matches(key, name_lookup.keys(), n=1, cutoff=0.8)
        if match:
            return name_lookup[match[0]]

        return label

    for col in adata.obs.columns:
        if col.endswith("_prediction"):
            adata.obs[col] = adata.obs[col].apply(map_label)

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

    ontology_file = pkg_resources.files(
        "SCART.PopV.resources.ontology"
    ).joinpath("cl_popv.json")

    with pkg_resources.as_file(ontology_file) as ontology_json_path:

        pq = Process_Query(
            query_adata=adata_query,
            ref_adata=adata_ref,
            ref_labels_key="cell_ontology_class",
            ref_batch_key=None,
            cl_obo_folder=str(ontology_json_path.parent) + "/",
            n_samples_per_label=n_samples_per_label
        )

        adata_processed = pq.adata

        if input_type == "raw":
            methods = [
                "CELLTYPIST",
                "KNN_BBKNN",
                "KNN_HARMONY",
                "KNN_SCVI",
                "ONCLASS",
                "SCANVI_POPV",
                "Support_Vector",
                "XGboost"
            ]
        else:
            methods = ["CELLTYPIST"]

        successful_methods = []

        for m in methods:

            try:
                annotate_data(adata_processed, methods=[m])

                # ✅ FIX labels AFTER failure-prone step
                normalize_predictions_to_ontology(
                    adata_processed,
                    str(ontology_json_path)
                )

                successful_methods.append(m)

            except Exception as e:

                print(f"⚠️ Fixing labels for {m} and retrying...")

                normalize_predictions_to_ontology(
                    adata_processed,
                    str(ontology_json_path)
                )

                try:
                    annotate_data(adata_processed, methods=[m])
                    successful_methods.append(m)

                except Exception as e2:
                    print(f"Skipping {m}: {e2}")

    # fallback
    if "popv_majority_vote_prediction" not in adata_processed.obs:
        if len(successful_methods) > 0:
            key = f"popv_{successful_methods[0].lower()}_prediction"
            if key in adata_processed.obs:
                adata_processed.obs["popv_majority_vote_prediction"] = adata_processed.obs[key]

    # ❌ REMOVED LOWERCASING (CRITICAL FIX)

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
        return adata.uns["cancer_type"]

    raise ValueError("Could not detect cancer type")

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
    )
