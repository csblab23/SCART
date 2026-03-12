"""
popv_annotation.py

Utilities for:
- Ontology validation and correction
- Layer harmonization
- Raw/log1p handling
- Running PopV annotation pipelines
- Keeping only original query cells
- Production-safe writing (NEW)
"""

import os
import glob
import logging
from typing import Optional

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import popv
from popv.preprocessing import Process_Query
from popv.annotation import annotate_data
import popv.algorithms as alg

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

REFERENCE_BASE_PATH = "/lustre/anas.a/Vinaya/scT-CAR_Designer/popv_reference"

# ------------------------------------------------------------------------------
# Automatically detect tumor h5ad
# ------------------------------------------------------------------------------

def get_latest_tumor_h5ad(data_dir: str = "GSE_data") -> str:
    files = glob.glob(os.path.join(data_dir, "*_tumor.h5ad"))
    if not files:
        raise FileNotFoundError("No *_tumor.h5ad file found in GSE_data")
    latest = max(files, key=os.path.getctime)
    logger.info(f"Using tumor file: {latest}")
    return latest

# ------------------------------------------------------------------------------
# Automatically detect cancer type from h5ad
# ------------------------------------------------------------------------------

def detect_cancer_type_from_h5ad(h5ad_file: str) -> str:
    adata = sc.read_h5ad(h5ad_file)
    if "cancer_type" in adata.uns:
        cancer_type = adata.uns["cancer_type"]
    elif "cancer_type" in adata.obs.columns:
        cancer_type = adata.obs["cancer_type"].unique()[0]
    else:
        raise ValueError(f"Cannot detect cancer type from {h5ad_file}")
    cancer_type = str(cancer_type).strip().lower()
    logger.info(f"Detected cancer type: {cancer_type}")
    return cancer_type

# ------------------------------------------------------------------------------
# Auto select reference
# ------------------------------------------------------------------------------

def auto_select_reference(cancer_type: str) -> str:

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