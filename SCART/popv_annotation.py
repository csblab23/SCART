"""
popv_annotation.py
Module 2 — PopV cell-type annotation

Fixes applied:
  1. Case-normalisation: predictions lower-cased before ontology lookup.
  2. Ontology path: resolves cl.obo from SCART/popv bundle.
  3. Harmony batch fix: KNN_HARMONY skipped when < 2 unique batch values.
  4. Fallback prediction: derived from actually-present obs columns.
  5. GEO download re-used from cached GSE dir.
  6. Minor: bare excepts → specific Exception catches.
  7. Query-cell extraction: only query cells saved after annotation.
  8. Full-gene preservation via SIDECAR h5ad.
  9. Method name auto-discovery using confirmed popv 0.4.2 attribute names.
 10. .X restoration — HVG-aware.
 11. FIX: Method aliases updated to match actual popv 0.4.2 attribute names
         (knn_on_bbknn, knn_on_harmony, knn_on_scanorama, knn_on_scvi, rf)
         as confirmed by sample_popv_annotation.py which successfully ran all 10.
"""

import os
import glob
import logging
import urllib.request
import importlib.resources as pkg_resources

import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import scipy.sparse as sp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import OnClass
import sys
sys.modules["onclass_utils"] = OnClass

import popv
import popv.algorithms as _popv_alg
from popv.preprocessing import Process_Query
from popv.annotation import annotate_data

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

REFERENCE_BASE_PATH = "popv_reference"
os.makedirs(REFERENCE_BASE_PATH, exist_ok=True)

FIGSHARE_ARTICLE_ID = "27921984"
TABULA_DOI_LINK = "https://doi.org/10.6084/m9.figshare.27921984"


# ---------------------------------------------------------------------------
# FIX 11 — corrected method aliases matching popv 0.4.2 actual attribute names
#
# The sample script (which successfully ran all 10 methods) confirms:
#   hasattr(alg, "celltypist")        → True
#   hasattr(alg, "knn_on_bbknn")      → True   (NOT "knn_bbknn")
#   hasattr(alg, "knn_on_harmony")    → True   (NOT "knn_harmony")
#   hasattr(alg, "knn_on_scanorama")  → True   (NOT "knn_scanorama")
#   hasattr(alg, "knn_on_scvi")       → True   (NOT "knn_scvi")
#   hasattr(alg, "rf")                → True   (NOT "random_forest")
#   hasattr(alg, "svm")               → True
#   hasattr(alg, "xgboost")           → True
#   hasattr(alg, "onclass")           → True
#   hasattr(alg, "scanvi")            → True
# ---------------------------------------------------------------------------

# Each inner list: [canonical_key, *aliases_in_priority_order]
# First alias that exists in popv.algorithms wins.
_METHOD_ALIASES = [
    ["CELLTYPIST",    "celltypist"],
    ["KNN_BBKNN",     "knn_on_bbknn",     "knn_bbknn",     "KNN_BBKNN"],
    ["KNN_SCANORAMA", "knn_on_scanorama", "knn_scanorama", "KNN_SCANORAMA"],
    ["KNN_SCVI",      "knn_on_scvi",      "knn_scvi",      "KNN_SCVI"],
    ["KNN_HARMONY",   "knn_on_harmony",   "knn_harmony",   "KNN_HARMONY"],
    ["RANDOM_FOREST", "rf",               "random_forest", "RANDOM_FOREST"],
    ["SVM",           "svm",              "support_vector_machine"],
    ["XGBOOST",       "xgboost",          "XGboost",       "XGBOOST"],
    ["ONCLASS",       "onclass",          "OnClass",       "ONCLASS"],
    ["SCANVI",        "scanvi",           "scanvi_popv",   "SCANVI_POPV"],
]

# Alias sets for special handling
_HARMONY_ALIASES = {
    "KNN_HARMONY", "knn_on_harmony", "knn_harmony", "Knn_Harmony",
}
_ONCLASS_ALIASES = {
    "ONCLASS", "onclass", "OnClass",
}


def _discover_popv_methods() -> dict:
    """
    Inspect popv.algorithms at runtime and return:
        {canonical_name: actual_attribute_name_in_popv.algorithms}

    Priority: first alias in each _METHOD_ALIASES group that exists in
    popv.algorithms wins.  Falls back to the first alias if none found.
    """
    try:
        available = set(dir(_popv_alg))
    except Exception:
        logger.warning("Could not inspect popv.algorithms — using default names.")
        available = set()

    logger.info(f"popv.algorithms attributes: {sorted(available)}")

    found = {}
    for aliases in _METHOD_ALIASES:
        canonical = aliases[0]
        for name in aliases[1:]:          # skip canonical itself
            if name in available:
                found[canonical] = name
                break
        else:
            # None of the aliases matched — log and skip
            logger.warning(
                f"No alias for {canonical} found in popv.algorithms "
                f"(tried: {aliases[1:]}). Method will be skipped."
            )

    if found:
        logger.info(
            f"popv.algorithms discovery: {len(found)} methods available — "
            + ", ".join(f"{k}→{v}" for k, v in found.items())
        )
    else:
        logger.error(
            "popv.algorithms discovery found NOTHING. "
            "Check that popv 0.4.2 is installed correctly."
        )

    return found


# ---------------------------------------------------------------------------
# Locate the most recent tumor h5ad written by Module 1
# ---------------------------------------------------------------------------

def get_latest_tumor_h5ad(data_dir="GSE_data"):
    """Return the most recently created tumor h5ad in cwd or data_dir."""
    search_paths = [os.getcwd(), data_dir]
    patterns = ["*_tumor.h5ad", "combined_tumor.h5ad", "input_tumor.h5ad"]

    files = []
    for path in search_paths:
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(path, pattern)))

    if not files:
        raise FileNotFoundError(
            "No tumor h5ad found.\n"
            "Expected: *_tumor.h5ad | combined_tumor.h5ad | input_tumor.h5ad\n"
            "in current directory or GSE_data/"
        )

    files = list(set(files))
    return max(files, key=os.path.getctime)


# ---------------------------------------------------------------------------
# Figshare metadata fetch
# ---------------------------------------------------------------------------

def fetch_tabula_file_metadata():
    """Return list of h5ad file records from the Figshare article."""
    url = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}/files"
    logger.info("Fetching Tabula Sapiens file list from Figshare …")

    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))

    resp = session.get(url, timeout=30, headers={"User-Agent": "curl/7.68.0"})
    resp.raise_for_status()

    return [f for f in resp.json() if f["name"].endswith(".h5ad")]


def find_best_reference_file(cancer_type: str, files):
    """Match a Figshare file record to the cancer_type string."""
    tissue = cancer_type.replace("_cancer", "").lower().replace("_", " ")
    logger.info(f"Matching tissue keyword: '{tissue}'")

    for f in files:
        if f["name"].lower().startswith(tissue.replace(" ", "_")):
            return f
    for f in files:
        if tissue in f["name"].lower():
            return f
    return None


def download_tabula_reference(cancer_type: str) -> str:
    """Download the matching Tabula Sapiens h5ad and return its local path."""
    files = fetch_tabula_file_metadata()
    selected = find_best_reference_file(cancer_type, files)

    if selected is None:
        raise ValueError(
            f"Reference not found for '{cancer_type}' via Figshare API.\n"
            f"Download manually from: {TABULA_DOI_LINK}\n"
            f"Then pass: user_reference='path_to_reference.h5ad'"
        )

    save_path = os.path.join(REFERENCE_BASE_PATH, selected["name"])
    if os.path.exists(save_path):
        logger.info(f"Reference already cached: {selected['name']}")
        return save_path

    logger.info(f"Downloading reference: {selected['name']} …")
    urllib.request.urlretrieve(selected["download_url"], save_path)
    logger.info(f"Saved to: {save_path}")
    return save_path


def auto_select_reference(cancer_type: str, user_reference=None) -> str:
    if user_reference:
        if not os.path.exists(user_reference):
            raise FileNotFoundError(f"Provided reference not found: {user_reference}")
        return user_reference
    return download_tabula_reference(cancer_type)


# ---------------------------------------------------------------------------
# Detect cancer type stored by Module 1
# ---------------------------------------------------------------------------

def detect_cancer_type_from_h5ad(h5ad_file: str) -> str:
    adata = sc.read_h5ad(h5ad_file)
    if "cancer_type" in adata.uns:
        ct = adata.uns["cancer_type"]
        logger.info(f"Detected cancer type from h5ad: {ct}")
        return ct
    raise ValueError(
        "Could not detect cancer type from h5ad .uns['cancer_type'].\n"
        "Provide user_reference manually via auto_run_popv(user_reference=…)."
    )


# ---------------------------------------------------------------------------
# Data-type helpers
# ---------------------------------------------------------------------------

def _fix_obs_dtypes(adata):
    """Convert categorical obs columns to str to avoid h5ad write errors."""
    for col in adata.obs.columns:
        if str(adata.obs[col].dtype) == "category":
            adata.obs[col] = adata.obs[col].astype(str)


def _clean_obs_for_h5ad(adata):
    """Ensure all object columns are str (no mixed types)."""
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object:
            adata.obs[col] = adata.obs[col].astype(str)


def _force_float32(adata):
    """Cast .X to float32 sparse CSR."""
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr().astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)


def _set_input_matrix(adata, input_type: str):
    """Route raw counts into .X according to input_type."""
    if input_type == "raw":
        if "raw_counts" in adata.layers:
            logger.info("Using existing 'raw_counts' layer for .X.")
            adata.X = adata.layers["raw_counts"].copy()
        elif "counts" in adata.layers:
            logger.info("Using 'counts' layer as raw input.")
            adata.layers["raw_counts"] = adata.layers["counts"].copy()
            adata.X = adata.layers["raw_counts"]
        else:
            logger.info(
                "No counts layer found — assuming .X already contains raw counts."
            )
    elif input_type == "log1p":
        logger.info("log1p mode: .X passed through unchanged.")
    else:
        raise ValueError(f"input_type must be 'raw' or 'log1p', got: {input_type!r}")


# ---------------------------------------------------------------------------
# FIX 1 — case normalisation
# ---------------------------------------------------------------------------

def _build_label_map_from_obo(cl_obo_folder: str):
    """
    Build two dicts from cl.obo (popv OBO flat format):
      label_map   : lowercase → correctly-cased label
      label_to_id : correctly-cased label → CL:XXXXXXX
    """
    obo_candidates = ["cl.obo", "cl_popv.obo"]
    obo_path = None
    for fname in obo_candidates:
        p = os.path.join(cl_obo_folder.rstrip("/"), fname)
        if os.path.exists(p):
            obo_path = p
            break

    if obo_path is None:
        logger.warning(
            f"cl.obo not found in {cl_obo_folder}. "
            "Label normalisation and ontology ID sync will be skipped."
        )
        return {}, {}

    logger.info(f"Building label map from OBO: {obo_path}")

    label_map   = {}
    label_to_id = {}
    current_id  = None
    current_lbl = None

    with open(obo_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip()
            if line == "[Term]":
                current_id = current_lbl = None
            elif line.startswith("id: CL:"):
                current_id = line[4:].strip()
            elif line.startswith("name: "):
                current_lbl = line[6:].strip()
            elif line == "" and current_id and current_lbl:
                label_map[current_lbl.lower()] = current_lbl
                label_to_id[current_lbl]       = current_id
                current_id = current_lbl = None

    if current_id and current_lbl:
        label_map[current_lbl.lower()] = current_lbl
        label_to_id[current_lbl]       = current_id

    logger.info(f"OBO parsed: {len(label_map):,} CL labels loaded.")
    return label_map, label_to_id


def _normalise_predictions(adata, label_map: dict):
    """Replace lowercase predictions with correctly-cased ontology labels."""
    pred_cols = [c for c in adata.obs.columns if c.endswith("_prediction")]
    for col in pred_cols:
        adata.obs[col] = (
            adata.obs[col]
            .astype(str)
            .str.lower()
            .map(lambda v: label_map.get(v, v))
        )


# ---------------------------------------------------------------------------
# FIX 2 — resolve ontology folder
# ---------------------------------------------------------------------------

def _resolve_ontology_folder() -> str:
    """Return the directory containing cl.obo."""
    obo_filenames = ["cl.obo", "cl_popv.obo"]

    candidate_packages = [
        "SCART.PopV.resources.ontology",
        "SCART.PopV.resources",
        "popv.resources.ontology",
        "popv.resources",
        "popv",
    ]

    for pkg in candidate_packages:
        for fname in obo_filenames:
            try:
                f = pkg_resources.files(pkg).joinpath(fname)
                with pkg_resources.as_file(f) as p:
                    if p.exists():
                        logger.info(f"OBO found via importlib ({pkg}): {p}")
                        return str(p.parent) + "/"
            except (ModuleNotFoundError, FileNotFoundError, TypeError, ValueError):
                continue

    walk_roots = []
    try:
        import SCART as _scart_pkg
        walk_roots.append(os.path.dirname(_scart_pkg.__file__))
    except ImportError:
        pass
    try:
        import popv as _popv_pkg
        walk_roots.append(os.path.dirname(_popv_pkg.__file__))
    except ImportError:
        pass

    for pkg_root in walk_roots:
        for root, _, fnames in os.walk(pkg_root):
            for fname in fnames:
                if fname in obo_filenames:
                    logger.info(
                        f"OBO found via filesystem walk: {os.path.join(root, fname)}"
                    )
                    return root + "/"

    raise FileNotFoundError(
        "Could not locate cl.obo.\n"
        "Expected: SCART/PopV/resources/ontology/cl.obo or inside popv package.\n"
        "Check that SCART / popv is installed correctly."
    )


# ---------------------------------------------------------------------------
# FIX 3 — harmony batch guard
# ---------------------------------------------------------------------------

def _check_batch_annotation(adata):
    """Warn if _batch_annotation has fewer than 2 unique values."""
    col = "_batch_annotation"
    if col not in adata.obs.columns:
        logger.warning(f"'{col}' not in adata.obs — KNN_HARMONY may fail.")
        return

    unique_vals = adata.obs[col].unique()
    logger.info(f"'{col}' unique values: {unique_vals}")

    if len(unique_vals) < 2:
        logger.warning(
            f"'{col}' has only 1 unique value ({unique_vals}). "
            "KNN_HARMONY will be skipped."
        )


# ---------------------------------------------------------------------------
# FIX 7 — extract query cells from the combined AnnData after PopV
# ---------------------------------------------------------------------------

def _extract_query_cells(adata_processed, adata_query_original):
    """Return a new AnnData containing ONLY the original query cells."""
    if "_dataset" in adata_processed.obs.columns:
        query_mask = adata_processed.obs["_dataset"] == "query"
        n_query = query_mask.sum()
        logger.info(
            f"_extract_query_cells: '_dataset' found — "
            f"extracting {n_query} of {adata_processed.n_obs} cells."
        )
        if n_query > 0:
            return adata_processed[query_mask].copy()
        logger.warning("'_dataset' == 'query' matched 0 cells; trying fallback.")

    original_names = set(adata_query_original.obs_names)
    mask_by_name   = adata_processed.obs_names.isin(original_names)
    n_matched      = mask_by_name.sum()
    if n_matched > 0:
        logger.info(f"_extract_query_cells: matched {n_matched} cells by obs_names.")
        return adata_processed[mask_by_name].copy()

    if "_reference_labels_annotation" in adata_processed.obs.columns:
        mask_nan = adata_processed.obs["_reference_labels_annotation"].isna()
        n_nan    = mask_nan.sum()
        logger.warning(f"_extract_query_cells: NaN proxy — {n_nan} cells assumed query.")
        if n_nan > 0:
            return adata_processed[mask_nan].copy()

    logger.error("Could not identify query cells — returning full combined object.")
    return adata_processed


def _drop_reference_only_columns(adata):
    """Remove Tabula Sapiens metadata columns from the query output."""
    tabula_ref_cols = {
        "donor", "tissue", "anatomical_position", "method", "cdna_plate",
        "library_plate", "notes", "cdna_well", "old_index", "assay",
        "sample_id", "replicate", "10X_run", "10X_barcode", "ambient_removal",
        "donor_method", "donor_assay", "donor_tissue", "donor_tissue_assay",
        "cell_ontology_class", "cell_ontology_id", "compartment",
        "broad_cell_class", "free_annotation", "manually_annotated",
        "published_2022", "n_genes_by_counts", "total_counts", "total_counts_mt",
        "pct_counts_mt", "total_counts_ercc", "pct_counts_ercc",
        "_scvi_batch", "_scvi_labels", "scvi_leiden_donorassay_full",
        "age", "sex", "ethnicity", "scvi_leiden_res05_tissue", "sample_number",
    }
    cols_to_drop = [c for c in adata.obs.columns if c in tabula_ref_cols]
    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} Tabula reference columns.")
        adata.obs = adata.obs.drop(columns=cols_to_drop)
    return adata


# ---------------------------------------------------------------------------
# FIX 8 — full-gene sidecar helpers
# ---------------------------------------------------------------------------

def _snapshot_full_counts(adata_query: anndata.AnnData):
    """Capture .X as plain sparse matrix + gene name list BEFORE alignment."""
    X = adata_query.X
    full_X = (
        X.tocsr().copy().astype(np.float32) if sp.issparse(X)
        else sp.csr_matrix(np.asarray(X, dtype=np.float32))
    )
    full_var_names = list(adata_query.var_names)
    logger.info(
        f"FIX 8: Snapshot captured — "
        f"{full_X.shape[0]} cells × {full_X.shape[1]} genes."
    )
    return full_X, full_var_names


def _write_full_counts_sidecar(full_X, full_var_names, obs, output_dir) -> str:
    """Write full-gene count matrix as a standalone h5ad. Returns file path."""
    sidecar = anndata.AnnData(
        X   = full_X,
        obs = obs.copy(),
        var = pd.DataFrame(index=full_var_names),
    )
    _clean_obs_for_h5ad(sidecar)

    sidecar_path = os.path.join(output_dir, "full_counts_for_module3.h5ad")
    sidecar.write(sidecar_path)

    size_mb = os.path.getsize(sidecar_path) / 1e6
    logger.info(
        f"FIX 8: Sidecar written → {sidecar_path} "
        f"({sidecar.shape[0]} cells × {sidecar.shape[1]} genes, {size_mb:.1f} MB)."
    )
    return sidecar_path


def _verify_full_counts_sidecar(adata_out: anndata.AnnData) -> None:
    """Log whether the full_counts sidecar was written and registered."""
    sidecar = adata_out.uns.get("full_counts_h5ad_path", None)
    if sidecar and os.path.exists(str(sidecar)):
        size_mb = os.path.getsize(sidecar) / 1e6
        logger.info(
            f"FIX 8 VERIFIED: sidecar present — {sidecar} ({size_mb:.1f} MB). "
            "Module 3: sc.read_h5ad(adata.uns['full_counts_h5ad_path'])"
        )
    else:
        logger.error(
            "FIX 8 FAILED: sidecar missing or path not set. "
            "Module 3 will fall back to 4 000 HVGs."
        )


# ---------------------------------------------------------------------------
# FIX 10 — HVG-aware .X restoration
# ---------------------------------------------------------------------------

def _restore_X_to_hvg_space(
    adata_query_out,
    saved_raw_counts,
    full_X,
    full_var_names,
    input_type,
    common_genes,
):
    """Set adata_query_out.X to original counts in HVG gene space."""
    hvg_names = list(adata_query_out.var_names)
    n_obs     = adata_query_out.n_obs

    def _subset_cols(matrix, gene_list):
        name_to_idx = {g: i for i, g in enumerate(gene_list)}
        col_idx = [name_to_idx[g] for g in hvg_names if g in name_to_idx]
        if len(col_idx) != len(hvg_names):
            logger.warning(
                f".X restoration: {len(col_idx)}/{len(hvg_names)} "
                "HVGs found in source — skipping."
            )
            return None
        m = matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)
        return m[:, col_idx]

    # Priority 1: saved_raw_counts in common-gene space
    if saved_raw_counts is not None and input_type == "raw":
        sub = _subset_cols(saved_raw_counts, common_genes)
        if sub is not None and sub.shape[0] == n_obs:
            adata_query_out.X = sub
            logger.info(
                f".X restored from saved_raw_counts (HVG space: "
                f"{n_obs} cells × {len(hvg_names)} genes)."
            )
            return
        elif sub is not None:
            logger.warning(
                f"saved_raw_counts has {sub.shape[0]} rows vs {n_obs}. Trying full_X."
            )

    # Priority 2: full_X in full-gene space
    sub = _subset_cols(full_X, full_var_names)
    if sub is not None and sub.shape[0] == n_obs:
        adata_query_out.X = sub
        logger.info(
            f".X restored from full_X snapshot (HVG space: "
            f"{n_obs} cells × {len(hvg_names)} genes, {input_type})."
        )
        return
    elif sub is not None:
        logger.warning(
            f"full_X has {sub.shape[0]} rows vs {n_obs}. Cannot assign."
        )

    logger.warning(
        ".X restoration failed — retaining PopV's internal normalised matrix."
    )


# ---------------------------------------------------------------------------
# Core annotation runner
# ---------------------------------------------------------------------------

def run_popv_annotation(
    adata_query,
    adata_ref,
    output_dir: str,
    input_type: str = "raw",
    n_samples_per_label: int = 300,
    drop_reference_columns: bool = True,
):
    """
    Run PopV cell-type annotation and write results to output_dir.

    Output files
    ------------
    output_dir/final_popv_annotated.h5ad   — query cells only
    output_dir/full_counts_for_module3.h5ad — full-gene sidecar
    """
    os.makedirs(output_dir, exist_ok=True)

    adata_query_snapshot = adata_query.copy()

    # --- dtypes -------------------------------------------------------------
    _fix_obs_dtypes(adata_query)
    _fix_obs_dtypes(adata_ref)
    _set_input_matrix(adata_query, input_type)
    _set_input_matrix(adata_ref, "raw")
    _force_float32(adata_query)
    _force_float32(adata_ref)
    adata_query.raw = None
    adata_ref.raw   = None

    # --- FIX 2: resolve cl.obo ----------------------------------------------
    cl_obo_folder = _resolve_ontology_folder()
    logger.info(f"cl_obo_folder: {cl_obo_folder}")

    # --- FIX 1: build label map ---------------------------------------------
    label_map, label_to_id = _build_label_map_from_obo(cl_obo_folder)
    logger.info(f"Loaded {len(label_map):,} ontology labels.")

    _ref_label_col = "cell_ontology_class"
    if _ref_label_col in adata_ref.obs.columns and label_map:
        adata_ref.obs[_ref_label_col] = (
            adata_ref.obs[_ref_label_col]
            .astype(str).str.lower()
            .map(lambda v: label_map.get(v, v))
        )
        valid_labels = set(label_to_id.keys())
        mask_valid   = adata_ref.obs[_ref_label_col].isin(valid_labels)
        n_before     = adata_ref.n_obs
        n_dropped    = (~mask_valid).sum()
        if n_dropped > 0:
            dropped_labels = (
                adata_ref.obs.loc[~mask_valid, _ref_label_col].unique().tolist()
            )
            logger.warning(
                f"Dropping {n_dropped} reference cells with non-ontology labels "
                f"(not in CL label_to_id): {dropped_labels}"
            )
            adata_ref = adata_ref[mask_valid].copy()
        logger.info(
            f"Reference labels normalised — "
            f"{n_before} → {adata_ref.n_obs} cells, "
            f"{adata_ref.obs[_ref_label_col].nunique()} unique labels."
        )

    # --- FIX 8: snapshot full gene space ------------------------------------
    full_X, full_var_names = _snapshot_full_counts(adata_query)

    # --- gene-space alignment -----------------------------------------------
    query_genes  = set(adata_query.var_names)
    ref_genes    = set(adata_ref.var_names)
    common_genes = sorted(query_genes & ref_genes)

    if len(common_genes) == 0:
        raise ValueError(
            "Query and reference share 0 genes. "
            "Check gene identifier type (Ensembl ID vs symbol)."
        )

    n_query_genes = adata_query.n_vars
    n_ref_genes   = adata_ref.n_vars

    _raw_counts_stash = adata_query.layers.pop("raw_counts", None)

    for lk in list(adata_query.layers.keys()):
        del adata_query.layers[lk]
    for lk in list(adata_ref.layers.keys()):
        del adata_ref.layers[lk]

    if len(common_genes) < n_query_genes or len(common_genes) < n_ref_genes:
        logger.info(
            f"Gene-space alignment: query {n_query_genes}, "
            f"ref {n_ref_genes} → {len(common_genes)} common genes."
        )
        adata_query = adata_query[:, common_genes].copy()
        adata_ref   = adata_ref[:, common_genes].copy()

        if _raw_counts_stash is not None:
            original_genes = list(adata_query_snapshot.var_names)
            g2i = {g: i for i, g in enumerate(original_genes)}
            common_idx = [g2i[g] for g in common_genes if g in g2i]
            if sp.issparse(_raw_counts_stash):
                _saved_raw_counts = (
                    _raw_counts_stash[:, common_idx].tocsr().astype(np.float32)
                )
            else:
                _saved_raw_counts = np.asarray(
                    _raw_counts_stash[:, common_idx], dtype=np.float32
                )
            logger.info(f"raw_counts subsetted to {len(common_idx)} common genes.")
        else:
            _saved_raw_counts = None
    else:
        logger.info(f"Gene-space: already aligned to {len(common_genes)} genes.")
        if _raw_counts_stash is not None:
            _saved_raw_counts = (
                _raw_counts_stash.tocsr().astype(np.float32)
                if sp.issparse(_raw_counts_stash)
                else np.asarray(_raw_counts_stash, dtype=np.float32)
            )
        else:
            _saved_raw_counts = None

    logger.info(
        "Layers cleared. raw_counts saved locally; "
        "full_X held as plain matrix for sidecar."
    )

    # --- Process_Query -------------------------------------------------------
    pq = Process_Query(
        query_adata=adata_query,
        ref_adata=adata_ref,
        ref_labels_key="cell_ontology_class",
        ref_batch_key=None,
        cl_obo_folder=cl_obo_folder,
        n_samples_per_label=n_samples_per_label,
    )
    adata_processed = pq.adata

    # --- FIX 3: batch check -------------------------------------------------
    _check_batch_annotation(adata_processed)
    has_two_batches = (
        "_batch_annotation" in adata_processed.obs.columns
        and len(adata_processed.obs["_batch_annotation"].unique()) >= 2
    )

    # --- FIX 11: discover actual method names --------------------------------
    method_map = _discover_popv_methods()

    # --- obsm proxy (harmony shape guard) ------------------------------------
    _harmony_key = "X_pca_harmony_popv"

    class _ObsmProxy:
        def __init__(self, real_obsm, n_obs):
            object.__setattr__(self, "_real", real_obsm)
            object.__setattr__(self, "_n_obs", n_obs)

        def __setitem__(self, key, value):
            if key == _harmony_key:
                arr   = np.array(value)
                n_obs = object.__getattribute__(self, "_n_obs")
                if arr.ndim == 2 and arr.shape[0] != n_obs:
                    logger.warning(
                        f"obsm proxy: correcting {key} shape "
                        f"{arr.shape} → {arr.T.shape}"
                    )
                    arr = arr.T
                elif arr.ndim == 1:
                    logger.warning(f"obsm proxy: {key} is 1-D — skipping.")
                    return
                object.__getattribute__(self, "_real").__setitem__(key, arr)
            else:
                object.__getattribute__(self, "_real").__setitem__(key, value)

        def __getitem__(self, key):
            return object.__getattribute__(self, "_real").__getitem__(key)

        def __contains__(self, key):
            return object.__getattribute__(self, "_real").__contains__(key)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_real"), name)

    # --- ONCLASS dict patch --------------------------------------------------
    def _patch_onclass(label_to_id_map):
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            candidates = [
                ("onclass_utils", "cell_type_nlp_network", "cell_type_nlp"),
                ("onclass_utils", "cell_ontology_graph",   "name2id"),
            ]
            patched = []
            for mod_name, obj_attr, dict_attr in candidates:
                try:
                    import importlib as _il
                    mod = _il.import_module(mod_name)
                    obj = getattr(mod, obj_attr, None)
                    if obj is None:
                        continue
                    d = getattr(obj, dict_attr, None)
                    if not isinstance(d, dict):
                        continue
                    missing = {k: v for k, v in label_to_id_map.items()
                               if k not in d}
                    if missing:
                        d.update(missing)
                        patched.append((d, missing))
                except Exception:
                    continue
            yield
            for d, added in patched:
                for k in added:
                    d.pop(k, None)

        return _ctx()

    # --- per-method runner ---------------------------------------------------
    def _run_method_safe(adata, canonical, actual):
        import unittest.mock as mock

        if actual in _HARMONY_ALIASES or canonical in _HARMONY_ALIASES:
            proxy = _ObsmProxy(adata.obsm, adata.n_obs)
            with mock.patch.object(adata, "obsm", proxy):
                annotate_data(adata, methods=[actual])
        elif actual in _ONCLASS_ALIASES or canonical in _ONCLASS_ALIASES:
            with _patch_onclass(label_to_id):
                annotate_data(adata, methods=[actual])
        else:
            annotate_data(adata, methods=[actual])

    # --- method run order ---------------------------------------------------
    # Mirrors the sample script's confirmed working order for popv 0.4.2.
    # KNN_HARMONY is included only when 2+ batch values exist.
    if input_type == "raw":
        run_order = [
            "CELLTYPIST",
            "KNN_BBKNN",
            "KNN_SCANORAMA",
            "KNN_SCVI",
            "RANDOM_FOREST",
            "SVM",
            "XGBOOST",
            "ONCLASS",
            "SCANVI",
        ]
        # Insert KNN_HARMONY after KNN_SCANORAMA if batches exist
        harmony_canonical = next(
            (c for c in method_map if c == "KNN_HARMONY"), None
        )
        if harmony_canonical and has_two_batches:
            run_order.insert(3, harmony_canonical)
        elif harmony_canonical:
            logger.warning("Skipping KNN_HARMONY: fewer than 2 batch values.")
    else:
        run_order = ["CELLTYPIST"]
        logger.info("log1p mode — running CELLTYPIST only.")

    # Build final list using discovered actual attribute names
    run_list = [
        (canonical, method_map[canonical])
        for canonical in run_order
        if canonical in method_map
    ]

    not_found = [c for c in run_order if c not in method_map]
    if not_found:
        logger.warning(
            f"Methods not found in popv.algorithms and will be skipped: {not_found}"
        )

    logger.info(
        f"Running {len(run_list)} methods: "
        + ", ".join(f"{c}({a})" for c, a in run_list)
    )

    # --- run each method ----------------------------------------------------
    successful_methods = []

    for canonical, actual in run_list:
        try:
            _run_method_safe(adata_processed, canonical, actual)
            if label_map:
                _normalise_predictions(adata_processed, label_map)
            successful_methods.append(actual)
            logger.info(f"✓ {canonical} ({actual}) completed.")
        except Exception as exc:
            logger.warning(
                f"✗ Skipping {canonical} ({actual}): "
                f"{type(exc).__name__}: {exc}"
            )

    logger.info(f"Methods completed: {successful_methods}")

    # --- FIX 4: majority-vote fallback --------------------------------------
    if "popv_majority_vote_prediction" not in adata_processed.obs.columns:
        pred_cols = [
            c for c in adata_processed.obs.columns
            if c.endswith("_prediction")
            and c != "popv_majority_vote_prediction"
            and adata_processed.obs[c].notna().any()
        ]
        if pred_cols:
            logger.warning(
                f"popv_majority_vote_prediction missing — "
                f"using '{pred_cols[0]}' as fallback."
            )
            adata_processed.obs["popv_majority_vote_prediction"] = (
                adata_processed.obs[pred_cols[0]]
            )
        else:
            logger.error("No prediction columns found. Annotation failed.")

    # --- FIX 7: extract query cells -----------------------------------------
    logger.info(
        f"Combined shape: {adata_processed.shape} "
        f"(query {adata_query_snapshot.n_obs} + reference cells)"
    )
    adata_query_out = _extract_query_cells(adata_processed, adata_query_snapshot)
    logger.info(f"Query-only shape: {adata_query_out.shape}")

    # --- FIX 10: restore .X to original counts in HVG space -----------------
    _restore_X_to_hvg_space(
        adata_query_out  = adata_query_out,
        saved_raw_counts = _saved_raw_counts,
        full_X           = full_X,
        full_var_names   = full_var_names,
        input_type       = input_type,
        common_genes     = common_genes,
    )

    # --- FIX 8: write full-gene sidecar -------------------------------------
    snapshot_names   = list(adata_query_snapshot.obs_names)
    query_out_names  = list(adata_query_out.obs_names)
    snap_name_to_idx = {n: i for i, n in enumerate(snapshot_names)}
    row_indices = [
        snap_name_to_idx[n] for n in query_out_names
        if n in snap_name_to_idx
    ]

    if len(row_indices) == adata_query_out.n_obs:
        full_X_out = full_X.tocsr()[row_indices, :]
        logger.info(
            f"FIX 8: full_X reindexed → "
            f"{len(row_indices)} cells × {full_X_out.shape[1]} genes."
        )
    else:
        logger.warning(
            f"FIX 8: Only {len(row_indices)}/{adata_query_out.n_obs} "
            "obs_names matched. Using full_X as-is."
        )
        full_X_out = full_X

    sidecar_path = _write_full_counts_sidecar(
        full_X         = full_X_out,
        full_var_names = full_var_names,
        obs            = adata_query_out.obs,
        output_dir     = output_dir,
    )
    adata_query_out.uns["full_counts_h5ad_path"] = sidecar_path
    _verify_full_counts_sidecar(adata_query_out)

    # --- final clean-up & save ----------------------------------------------
    if drop_reference_columns:
        adata_query_out = _drop_reference_only_columns(adata_query_out)

    _clean_obs_for_h5ad(adata_query_out)

    out_path = os.path.join(output_dir, "final_popv_annotated.h5ad")
    adata_query_out.write(out_path)

    logger.info(
        f"\n{'='*60}\n"
        f"PopV output saved       : {out_path}\n"
        f"Full-gene sidecar       : {sidecar_path}\n"
        f"Shape                   : {adata_query_out.shape}\n"
        f"obs columns             : {list(adata_query_out.obs.columns)}\n"
        f"layers                  : {list(adata_query_out.layers.keys())}\n"
        f"uns keys                : {list(adata_query_out.uns.keys())}\n"
        f"{'='*60}"
    )
    return adata_query_out


# ---------------------------------------------------------------------------
# Auto entry-point
# ---------------------------------------------------------------------------

def auto_run_popv(
    input_type: str = "raw",
    nsamples: int = 300,
    output_dir: str = "popv_results",
    user_reference: str = None,
    drop_reference_columns: bool = True,
    user_popv_prediction: str = None,
):
    """
    Fully automatic entry-point.

    Parameters
    ----------
    input_type : str
        'raw' or 'log1p'.
    nsamples : int
        Cells sampled per label during reference subsampling.
    output_dir : str
        Output directory.
    user_reference : str, optional
        Path to a local reference h5ad (skips Figshare download).
    drop_reference_columns : bool
        Remove Tabula Sapiens metadata columns from the output.
    user_popv_prediction : str, optional
        Path to a pre-computed PopV-annotated h5ad (skips full pipeline).

    Module 3 usage
    --------------
    full_adata = sc.read_h5ad(adata.uns['full_counts_h5ad_path'])
    """
    # ------------------------------------------------------------------
    # Short-circuit: user supplies their own PopV prediction h5ad
    # ------------------------------------------------------------------
    if user_popv_prediction is not None:
        if not os.path.exists(user_popv_prediction):
            raise FileNotFoundError(
                f"user_popv_prediction not found: {user_popv_prediction}"
            )

        logger.info(f"user_popv_prediction provided — loading: {user_popv_prediction}")
        adata = sc.read_h5ad(user_popv_prediction)

        pred_cols = [c for c in adata.obs.columns if c.endswith("_prediction")]
        if not pred_cols:
            raise ValueError(
                f"No '_prediction' columns in {user_popv_prediction}.\n"
                "Supply a valid PopV-annotated h5ad."
            )
        if "popv_majority_vote_prediction" not in adata.obs.columns:
            logger.warning(
                f"'popv_majority_vote_prediction' not found. Found: {pred_cols}"
            )
        else:
            logger.info(
                f"'popv_majority_vote_prediction' present — "
                f"{adata.obs['popv_majority_vote_prediction'].nunique()} labels."
            )

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "final_popv_annotated.h5ad")
        if os.path.abspath(user_popv_prediction) != os.path.abspath(out_path):
            logger.info(f"Copying to: {out_path}")
            adata.write(out_path)
        else:
            logger.info("Already at output path — no copy needed.")

        logger.info(
            f"\n{'='*60}\n"
            f"User PopV prediction loaded: {out_path}\n"
            f"Shape       : {adata.shape}\n"
            f"Pred columns: {pred_cols}\n"
            f"{'='*60}"
        )
        return adata

    # ------------------------------------------------------------------
    # Normal pipeline
    # ------------------------------------------------------------------
    tumor_file = get_latest_tumor_h5ad()
    logger.info(f"Query file: {tumor_file}")

    cancer_type    = detect_cancer_type_from_h5ad(tumor_file)
    primary_cancer = cancer_type.split(",")[0].strip()
    reference_path = auto_select_reference(primary_cancer, user_reference)

    logger.info(f"Loading query    : {tumor_file}")
    logger.info(f"Loading reference: {reference_path}")

    adata_query = sc.read_h5ad(tumor_file)
    adata_ref   = sc.read_h5ad(reference_path)

    return run_popv_annotation(
        adata_query=adata_query,
        adata_ref=adata_ref,
        output_dir=output_dir,
        input_type=input_type,
        n_samples_per_label=nsamples,
        drop_reference_columns=drop_reference_columns,
    )
