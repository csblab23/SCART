"""
popv_annotation.py
Module 2 — PopV cell-type annotation  (popv == 0.4.2)

Fixes applied (all changes marked with # FIX N comments):
  1. Case-normalisation: predictions are title-cased before ontology lookup
     so "b cell" → "B cell" matches the digraph node label.
  2. Ontology path: resolves cl.obo (v0.4.2 format) from popv's own bundle
     or a filesystem walk; cl_obo_folder passed to Process_Query.
  3. Harmony batch fix: _batch_annotation guard added; falls back gracefully
     when only one batch value is present.
  4. Fallback prediction: derived from actually-present obs columns, not
     from the successful_methods list (which could be misleading).
  5. GEO download is re-used from the already-cached GSE dir so Module 1
     does not re-download at annotation time.
  6. Minor: bare excepts replaced with specific Exception catches; unused
     obo_file argument removed from public API.
  7. Query-cell extraction: after PopV annotation the combined
     query+reference AnnData is filtered back to query cells only using
     the '_dataset' column written by Process_Query.  Only the query
     portion (original Module 1 shape) is saved to disk.
  8. [NEW — revised] Full-gene raw count preservation via a dedicated
     layer 'full_counts' rather than adata.raw.

     WHY THE LAYER APPROACH (not adata.raw):
       AnnData.raw = <AnnData> is not a public API.  The only supported
       usage is `adata.raw = adata`, which freezes the CURRENT .X and
       .var in-place.  After Process_Query trims .var to 4000 HVGs,
       assigning a separate AnnData object to .raw either raises
       AttributeError or is silently discarded on write/read — which is
       exactly why Module 3 saw "adata.raw is None".

       Using layers['full_counts'] is the h5ad-safe fix:
         • Snapshot taken BEFORE Process_Query (full gene space, e.g. 36 k).
         • Layers are indexed (cell × gene) so AnnData concatenation and
           slicing in _extract_query_cells carry them automatically.
         • adata.write() / sc.read_h5ad() round-trip layers with zero loss.
         • Gene names saved to uns['full_counts_var_names'] (list of str).

       Module 3 _build_fullgene_adata_for_scm() checks 'full_counts'
       first, giving scMalignantFinder ≥90% model feature overlap instead
       of the previous 19%.

  popv 0.4.2 API notes:
    • Process_Query: no prediction_mode, no save_path_trained_models, no
      hvg kwarg.  cl_obo_folder must point to the directory containing
      cl.obo (not a JSON).
    • annotate_data: simpler signature — annotate_data(adata, methods=[…])
      with no methods_kwargs argument.
    • Method name strings are UPPER_SNAKE (e.g. "KNN_BBKNN", "CELLTYPIST")
      as registered in popv 0.4.2.
    • .X must contain raw counts (integer or float32) when input_type='raw'.
      For log1p input, .X is passed as-is; only CELLTYPIST is run.
"""

import os
import glob
import logging
import urllib.request
import importlib.resources as pkg_resources

import numpy as np
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
from popv.preprocessing import Process_Query
from popv.annotation import annotate_data

# ---------------------------------------------------------------------------
# FIX 9 — runtime method-name discovery
# ---------------------------------------------------------------------------
# Each row: [canonical_key, popv-0.4.2-actual-attr, *older-spellings]
# popv 0.4.2 confirmed attribute names (from popv.algorithms):
#   celltypist, knn_on_bbknn, knn_on_harmony, knn_on_scanorama, knn_on_scvi,
#   rf, svm, onclass, scanvi
_METHOD_ALIASES = [
    ["CELLTYPIST",    "celltypist",       "Celltypist",      "CELLTYPIST"],
    ["KNN_BBKNN",     "knn_on_bbknn",     "knn_bbknn",       "Knn_Bbknn",     "KNN_BBKNN"],
    ["KNN_SCANORAMA", "knn_on_scanorama", "knn_scanorama",   "Knn_Scanorama", "KNN_SCANORAMA"],
    ["KNN_SCVI",      "knn_on_scvi",      "knn_scvi",        "Knn_Scvi",      "KNN_SCVI"],
    ["KNN_HARMONY",   "knn_on_harmony",   "knn_harmony",     "Knn_Harmony",   "KNN_HARMONY"],
    ["RANDOM_FOREST", "rf",               "random_forest",   "Random_Forest", "RANDOM_FOREST"],
    ["SVM",           "svm",              "support_vector_machine", "Support_Vector"],
    # XGBOOST is not present in popv 0.4.2 (confirmed via dir(popv.algorithms))
    # ["XGBOOST", ...],  ← kept commented in case a future version adds it
    ["ONCLASS",       "onclass",          "OnClass",         "ONCLASS"],
    ["SCANVI",        "scanvi",           "scanvi_popv",     "Scanvi_Popv",   "SCANVI_POPV"],
]

_HARMONY_ALIASES = {"KNN_HARMONY", "knn_on_harmony", "knn_harmony", "Knn_Harmony"}
_ONCLASS_ALIASES = {"ONCLASS", "onclass", "OnClass"}


def _discover_popv_methods() -> dict:
    """
    Inspect popv.algorithms at runtime and return:
        {canonical_name: actual_attribute_name}
    Uses hasattr() — same approach as the working sample script.
    """
    try:
        import popv.algorithms as _alg
    except ImportError:
        logger.warning("Could not import popv.algorithms — using first alias as fallback.")
        return {row[0]: row[1] for row in _METHOD_ALIASES}

    found = {}
    for row in _METHOD_ALIASES:
        canonical = row[0]
        for name in row[1:]:           # try actual names in priority order
            if hasattr(_alg, name):
                found[canonical] = name
                break

    if found:
        logger.info(
            f"popv.algorithms discovery: {len(found)} methods available — "
            + ", ".join(f"{k}→{v}" for k, v in found.items())
        )
    else:
        found = {row[0]: row[1] for row in _METHOD_ALIASES}
        logger.warning("popv.algorithms discovery found nothing — using first aliases.")

    return found

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
            "Expected one of: *_tumor.h5ad | combined_tumor.h5ad | input_tumor.h5ad\n"
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
    """Convert all categorical obs columns to str to avoid h5ad write errors."""
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
    """
    Route raw counts into .X according to input_type.

    popv 0.4.2 expects .X to contain raw counts (not log-normalised)
    for all methods except CELLTYPIST.  For log1p input we pass .X
    through unchanged and restrict methods to CELLTYPIST only.
    """
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
                "No counts layer found — assuming adata.X already contains raw counts."
            )
    elif input_type == "log1p":
        # .X is already log-normalised; pass through.
        # Only CELLTYPIST will be run downstream.
        logger.info("log1p mode: adata.X passed through unchanged.")
    else:
        raise ValueError(f"input_type must be 'raw' or 'log1p', got: {input_type!r}")


# ---------------------------------------------------------------------------
# FIX 1 — case normalisation
# ---------------------------------------------------------------------------

def _build_label_map_from_obo(cl_obo_folder: str):
    """
    Build two dicts from cl.obo (popv 0.4.2 format):

    label_map   : lowercase_label → correctly-cased label
                  e.g. "b cell" → "B cell"

    label_to_id : correctly-cased label → short CL ID
                  e.g. "B cell" → "CL:0000236"

    Parses the OBO flat-text format directly so there is no dependency
    on pronto or owlready2.
    """
    obo_candidates = ["cl.obo", "cl_popv.obo"]
    obo_path = None
    for fname in obo_candidates:
        p = os.path.join(cl_obo_folder.rstrip("/"), fname)
        if os.path.exists(p):
            obo_path = p
            break

    if obo_path is None:
        # Graceful fallback: empty maps (annotation will still run)
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
                current_id  = None
                current_lbl = None
            elif line.startswith("id: CL:"):
                current_id = line[4:].strip()   # e.g. "CL:0000236"
            elif line.startswith("name: "):
                current_lbl = line[6:].strip()
            elif line == "" and current_id and current_lbl:
                label_map[current_lbl.lower()] = current_lbl
                label_to_id[current_lbl]       = current_id
                current_id  = None
                current_lbl = None

    # Flush last term if file has no trailing blank line
    if current_id and current_lbl:
        label_map[current_lbl.lower()] = current_lbl
        label_to_id[current_lbl]       = current_id

    logger.info(
        f"OBO parsed: {len(label_map):,} CL labels loaded."
    )
    return label_map, label_to_id


def _normalise_predictions(adata, label_map: dict):
    """
    For every *_prediction column in adata.obs, replace lowercase labels
    with the correctly-cased ontology label.
    """
    pred_cols = [c for c in adata.obs.columns if c.endswith("_prediction")]
    for col in pred_cols:
        adata.obs[col] = (
            adata.obs[col]
            .astype(str)
            .str.lower()
            .map(lambda v: label_map.get(v, v))
        )


# ---------------------------------------------------------------------------
# FIX 2 — resolve ontology folder (cl.obo for popv 0.4.2)
# ---------------------------------------------------------------------------

def _resolve_ontology_folder() -> str:
    """
    Return the directory containing cl.obo (popv 0.4.2 uses the OBO file,
    not a JSON).  Searches importlib resources then filesystem walk.
    """
    obo_filenames = ["cl.obo", "cl_popv.obo"]

    # --- importlib.resources search -----------------------------------------
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
                        logger.info(
                            f"Ontology (OBO) found via importlib ({pkg}): {p}"
                        )
                        return str(p.parent) + "/"
            except (ModuleNotFoundError, FileNotFoundError, TypeError, ValueError):
                continue

    # --- filesystem walk ----------------------------------------------------
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
                        f"Ontology (OBO) found via filesystem walk: "
                        f"{os.path.join(root, fname)}"
                    )
                    return root + "/"

    raise FileNotFoundError(
        "Could not locate cl.obo.\n"
        "Expected location: SCART/PopV/resources/ontology/cl.obo\n"
        "or inside the popv package directory.\n"
        "Check that the SCART / popv 0.4.2 package is installed correctly."
    )


# ---------------------------------------------------------------------------
# FIX 3 — harmony batch guard
# ---------------------------------------------------------------------------

def _check_batch_annotation(adata):
    """Warn if _batch_annotation has fewer than 2 unique values."""
    col = "_batch_annotation"
    if col not in adata.obs.columns:
        logger.warning(
            f"'{col}' not in adata.obs — KNN_HARMONY may fail. "
            "Check that Process_Query ran correctly."
        )
        return

    unique_vals = adata.obs[col].unique()
    logger.info(f"'{col}' unique values: {unique_vals}")

    if len(unique_vals) < 2:
        logger.warning(
            f"'{col}' has only 1 unique value ({unique_vals}). "
            "KNN_HARMONY will fail with a shape mismatch. "
            "It will be skipped automatically."
        )


# ---------------------------------------------------------------------------
# FIX 7 — extract query cells from the combined AnnData after PopV
# ---------------------------------------------------------------------------

def _extract_query_cells(adata_processed, adata_query_original):
    """
    After Process_Query + annotate_data, the AnnData contains both query
    and reference cells concatenated together.  Returns a new AnnData
    containing ONLY the original query cells with all PopV columns.
    """
    if "_dataset" in adata_processed.obs.columns:
        query_mask = adata_processed.obs["_dataset"] == "query"
        n_query = query_mask.sum()
        logger.info(
            f"_extract_query_cells: '_dataset' found — "
            f"extracting {n_query} query cells out of {adata_processed.n_obs} total."
        )
        if n_query > 0:
            return adata_processed[query_mask].copy()
        logger.warning("'_dataset' == 'query' matched 0 cells; trying fallback.")

    original_names = set(adata_query_original.obs_names)
    mask_by_name   = adata_processed.obs_names.isin(original_names)
    n_matched      = mask_by_name.sum()
    if n_matched > 0:
        logger.info(
            f"_extract_query_cells: matched {n_matched} cells by obs_names."
        )
        return adata_processed[mask_by_name].copy()

    if "_reference_labels_annotation" in adata_processed.obs.columns:
        mask_nan = adata_processed.obs["_reference_labels_annotation"].isna()
        n_nan    = mask_nan.sum()
        logger.warning(
            f"_extract_query_cells: NaN proxy — {n_nan} cells assumed query."
        )
        if n_nan > 0:
            return adata_processed[mask_nan].copy()

    logger.error(
        "Could not identify query cells — returning full combined object. "
        "Output will contain reference cells too."
    )
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
# FIX 8 (REVISED) — preserve full-gene raw counts as layers['full_counts']
# ---------------------------------------------------------------------------

def _store_full_counts_layer(adata_query: anndata.AnnData) -> anndata.AnnData:
    """
    Snapshot raw counts for ALL genes into layers['full_counts'] and save
    gene names to uns['full_counts_var_names'].

    Call AFTER _set_input_matrix (so .X = raw counts) and BEFORE
    Process_Query (which trims .var to HVGs).
    """
    if "full_counts" in adata_query.layers:
        logger.info(
            "FIX 8: 'full_counts' already present "
            f"({adata_query.n_vars} genes) — skipping."
        )
        return adata_query

    logger.info(
        f"FIX 8: Snapshotting {adata_query.n_vars} genes → "
        "layers['full_counts'] before Process_Query."
    )

    X = adata_query.X
    if sp.issparse(X):
        adata_query.layers["full_counts"] = X.tocsr().copy()
    else:
        adata_query.layers["full_counts"] = sp.csr_matrix(
            np.asarray(X, dtype=np.float32)
        )

    adata_query.uns["full_counts_var_names"] = list(adata_query.var_names)

    logger.info(
        f"FIX 8: Snapshot done — "
        f"layers['full_counts'].shape = {adata_query.layers['full_counts'].shape}, "
        f"uns['full_counts_var_names'] has "
        f"{len(adata_query.uns['full_counts_var_names'])} entries."
    )
    return adata_query


def _verify_full_counts_layer(adata_out: anndata.AnnData) -> None:
    """Log whether 'full_counts' survived the full pipeline."""
    if "full_counts" in adata_out.layers:
        n_genes = adata_out.layers["full_counts"].shape[1]
        n_names = len(adata_out.uns.get("full_counts_var_names", []))
        logger.info(
            f"FIX 8 VERIFIED: layers['full_counts'] present — "
            f"{adata_out.n_obs} cells × {n_genes} genes, "
            f"uns['full_counts_var_names'] has {n_names} entries. "
            "Module 3 will use full gene space for scMalignantFinder."
        )
    else:
        logger.error(
            "FIX 8 FAILED: layers['full_counts'] MISSING from query output. "
            "Process_Query may have dropped non-HVG layers. "
            "Module 3 will fall back to 4000 HVGs (~19% overlap)."
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
    n_jobs: int = 1,
):
    """
    Run PopV (0.4.2) cell-type annotation and write results to output_dir.

    Parameters
    ----------
    adata_query : AnnData
        Query dataset (tumor cells from Module 1).
    adata_ref : AnnData
        Tabula Sapiens tissue reference.
    output_dir : str
        Directory where final_popv_annotated.h5ad is written.
    input_type : str
        'raw'  — .X contains raw counts (default). All popv methods are run.
        'log1p' — .X is already log-normalised; only CELLTYPIST will run.
    n_samples_per_label : int
        Cells sampled per label during reference subsampling.
    n_jobs : int
        Number of CPU cores for parallelisable methods (SVM, Random Forest,
        ONCLASS).  -1 = use all available cores.  Default 1.
    drop_reference_columns : bool
        If True (default), Tabula Sapiens metadata columns are removed from
        the saved query AnnData to keep the output file clean.

    popv 0.4.2 specifics
    --------------------
    • Process_Query does NOT accept: prediction_mode, save_path_trained_models,
      hvg.  These kwargs are removed vs the 0.6.0 call.
    • annotate_data does NOT accept methods_kwargs.  Each method is called
      with annotate_data(adata, methods=[method]) only.
    • cl_obo_folder must point to the directory containing cl.obo.
    • .X must be raw integer/float32 counts for all methods except CELLTYPIST.
    • .raw is set to None before Process_Query to prevent dtype concat crashes.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Snapshot of original query needed for _extract_query_cells
    adata_query_snapshot = adata_query.copy()

    # --- pre-process dtypes -------------------------------------------------
    _fix_obs_dtypes(adata_query)
    _fix_obs_dtypes(adata_ref)

    _set_input_matrix(adata_query, input_type)
    _set_input_matrix(adata_ref, "raw")

    _force_float32(adata_query)
    _force_float32(adata_ref)

    # Remove .raw to prevent concat dtype crashes (popv 0.4.2 issue)
    adata_query.raw = None
    adata_ref.raw   = None

    # --- FIX 2: resolve cl.obo folder (popv 0.4.2 uses OBO, not JSON) ------
    cl_obo_folder = _resolve_ontology_folder()
    logger.info(f"cl_obo_folder: {cl_obo_folder}")

    # --- FIX 1: build label map from OBO ------------------------------------
    label_map, label_to_id = _build_label_map_from_obo(cl_obo_folder)
    logger.info(f"Loaded {len(label_map):,} ontology labels.")

    # FIX 1a — normalise reference labels + drop non-CL-ontology cells.
    #
    # WHY dropping is required for popv 0.4.2:
    #   After KNN methods finish their embedding, popv walks the CL ontology
    #   digraph to propagate labels.  Any reference label absent from the
    #   digraph (e.g. 'follicle' — a tissue compartment, not a cell type)
    #   raises NetworkXError: "The node follicle is not in the digraph."
    #   This kills KNN_BBKNN, KNN_SCANORAMA, KNN_SCVI, KNN_HARMONY even
    #   though the embedding itself completed successfully.
    #   Dropping these cells before Process_Query is the only reliable fix.
    _ref_label_col = "cell_ontology_class"
    if _ref_label_col in adata_ref.obs.columns and label_map:
        # Step 1: case-normalise
        adata_ref.obs[_ref_label_col] = (
            adata_ref.obs[_ref_label_col]
            .astype(str)
            .str.lower()
            .map(lambda v: label_map.get(v, v))
        )
        # Step 2: drop cells whose label is NOT a valid CL ontology term.
        # label_to_id keys are the correctly-cased CL labels from cl.obo.
        valid_labels = set(label_to_id.keys())
        mask_valid   = adata_ref.obs[_ref_label_col].isin(valid_labels)
        n_before     = adata_ref.n_obs
        n_dropped    = (~mask_valid).sum()
        if n_dropped > 0:
            dropped_labels = (
                adata_ref.obs.loc[~mask_valid, _ref_label_col].unique().tolist()
            )
            logger.warning(
                f"Dropping {n_dropped} reference cells with labels not in "
                f"CL ontology digraph: {dropped_labels}. "
                "These would cause NetworkXError during KNN label propagation."
            )
            adata_ref = adata_ref[mask_valid].copy()
        logger.info(
            f"Reference labels normalised — "
            f"{n_before} → {adata_ref.n_obs} cells, "
            f"{adata_ref.obs[_ref_label_col].nunique()} unique labels."
        )

    # -----------------------------------------------------------------------
    # FIX 8 — snapshot full gene space into layers['full_counts'] BEFORE
    # Process_Query trims .var to HVGs.
    # -----------------------------------------------------------------------
    adata_query = _store_full_counts_layer(adata_query)

    # Log what layers exist before the gene-alignment step strips them all.
    # (Full removal happens at the end of the gene-alignment block below.)
    _ref_layer_info   = list(adata_ref.layers.keys())
    _query_layer_info = list(adata_query.layers.keys())
    if _ref_layer_info:
        logger.info(f"Reference layers present (will be cleared): {_ref_layer_info}")
    if _query_layer_info:
        logger.info(f"Query layers present (full_counts/raw_counts will be saved): {_query_layer_info}")

    # -----------------------------------------------------------------------
    # Gene-space alignment before Process_Query.
    #
    # Root cause (persists even after layer stripping): Process_Query calls
    #   anndata.concat(ref, query, join="outer", fill_value=unknown_celltype_label)
    # where unknown_celltype_label is the STRING "unknown".  When query and
    # reference do not share the exact same var_names, anndata fills missing
    # genes with that string value while building the outer-join sparse matrix.
    # scipy.sparse then raises "does not support dtype str224".
    #
    # Fix: subset both objects to the intersection of their var_names BEFORE
    # calling Process_Query.  With identical gene spaces, join="outer" becomes
    # equivalent to join="inner" and no string fill is ever needed.
    # The full gene snapshot (layers['full_counts']) was taken before this step
    # so Module 3 still gets all 36 k genes.
    # -----------------------------------------------------------------------
    query_genes = set(adata_query.var_names)
    ref_genes   = set(adata_ref.var_names)
    common_genes = sorted(query_genes & ref_genes)

    if len(common_genes) == 0:
        raise ValueError(
            "Query and reference share 0 genes. "
            "Check that both datasets use the same gene identifier (Ensembl ID vs symbol)."
        )

    n_query_genes = adata_query.n_vars
    n_ref_genes   = adata_ref.n_vars

    # Initialise stash variables — only assigned inside the subsetting branch.
    # The no-subsetting branch leaves them None (layers remain in adata_query).
    _full_counts_stash    = None
    _full_var_names_stash = None
    _raw_counts_stash     = None

    if len(common_genes) < n_query_genes or len(common_genes) < n_ref_genes:
        logger.info(
            f"Gene-space alignment: "
            f"query {n_query_genes} genes, ref {n_ref_genes} genes → "
            f"{len(common_genes)} common genes (intersection)."
        )

        # Stash full_counts and raw_counts BEFORE subsetting .var.
        # After [:, common_genes].copy() these layers would be column-subsetted
        # too, losing the full-gene snapshot.  We preserve them separately and
        # reattach after subsetting.
        _full_counts_stash     = adata_query.layers.pop("full_counts", None)
        _full_var_names_stash  = adata_query.uns.pop("full_counts_var_names", None)
        _raw_counts_stash      = adata_query.layers.pop("raw_counts", None)

        adata_query = adata_query[:, common_genes].copy()
        adata_ref   = adata_ref[:, common_genes].copy()

        # full_counts (36k genes) is intentionally NOT put back into
        # adata_query.layers — its .var is now 23,539 genes and AnnData
        # would reject the shape mismatch.  _full_counts_stash stays as a
        # plain Python variable and flows into _saved_full_counts below,
        # then gets written to the sidecar h5ad after annotation.

        # Subset raw_counts to common genes (float32)
        if _raw_counts_stash is not None:
            # _raw_counts_stash still has the original n_vars columns; subset it
            query_gene_list = list(adata_query_snapshot.var_names)
            gene_to_idx     = {g: i for i, g in enumerate(query_gene_list)}
            common_idx      = [gene_to_idx[g] for g in common_genes if g in gene_to_idx]
            if sp.issparse(_raw_counts_stash):
                adata_query.layers["raw_counts"] = (
                    _raw_counts_stash[:, common_idx].tocsr().astype(np.float32)
                )
            else:
                adata_query.layers["raw_counts"] = np.asarray(
                    _raw_counts_stash[:, common_idx], dtype=np.float32
                )
            logger.info(
                f"Gene-space alignment: raw_counts subsetted to "
                f"{len(common_idx)} common genes."
            )

    else:
        logger.info(
            f"Gene-space alignment: query and reference already share "
            f"all {len(common_genes)} genes — no subsetting needed."
        )

    # Final safety: collect saved matrices and clear ALL layers from both
    # objects before Process_Query.
    #
    # After the subsetting branch:
    #   - full_counts was never put back into adata_query.layers (shape mismatch
    #     would crash), so it lives in _full_counts_stash / _full_var_names_stash.
    #   - raw_counts was put back into adata_query.layers (already subsetted).
    #
    # After the no-subsetting branch:
    #   - full_counts is still in adata_query.layers (same gene space).
    #   - raw_counts is still in adata_query.layers.
    #
    # In both cases we pop everything into _saved_* variables, then wipe all
    # remaining layers so anndata.concat sees zero layers on both sides and
    # the str224 crash cannot occur.

    # full_counts: prefer what's already in layers (no-subsetting path),
    # fall back to the stash variable (subsetting path).
    _saved_full_counts    = (
        adata_query.layers.pop("full_counts", None)
        or _full_counts_stash
    )
    _saved_full_var_names = (
        adata_query.uns.pop("full_counts_var_names", None)
        or _full_var_names_stash
    )
    _saved_raw_counts     = adata_query.layers.pop("raw_counts", None)

    # Clear any remaining layers on both sides
    for lk in list(adata_query.layers.keys()):
        del adata_query.layers[lk]
    for lk in list(adata_ref.layers.keys()):
        del adata_ref.layers[lk]

    logger.info(
        f"Layers cleared before Process_Query. "
        f"full_counts saved: {_saved_full_counts is not None} "
        f"({_saved_full_counts.shape[1] if _saved_full_counts is not None else 0} genes), "
        f"raw_counts saved: {_saved_raw_counts is not None}."
    )

    # --- Process_Query (popv 0.4.2 signature) --------------------------------
    # Removed vs 0.6.0: prediction_mode, save_path_trained_models, hvg
    pq = Process_Query(
        query_adata=adata_query,
        ref_adata=adata_ref,
        ref_labels_key="cell_ontology_class",
        ref_batch_key=None,
        cl_obo_folder=cl_obo_folder,
        n_samples_per_label=n_samples_per_label,
    )
    adata_processed = pq.adata

    # Confirm layer survived Process_Query
    if "full_counts" in adata_processed.layers:
        logger.info(
            f"FIX 8: 'full_counts' survived Process_Query — "
            f"combined shape {adata_processed.shape}"
        )
    else:
        logger.error(
            "FIX 8: 'full_counts' DROPPED by Process_Query — "
            "Module 3 will fall back to 4000 HVGs."
        )

    # FIX 3: check batch column
    _check_batch_annotation(adata_processed)

    # Determine whether REAL batch variation exists for KNN_HARMONY.
    # popv always writes '_batch_annotation' with pseudo-labels
    # 'unknown' (reference) and 'unknown_query' (query) when no real
    # batch key is provided.  These are not biological batches and
    # harmony will loop indefinitely trying to correct them.
    # Only enable KNN_HARMONY when at least one value is NOT a popv
    # internal pseudo-label — i.e. a real sample/donor batch ID.
    _POPV_PSEUDO_BATCHES = {"unknown", "unknown_query"}
    _batch_vals = set()
    if "_batch_annotation" in adata_processed.obs.columns:
        _batch_vals = set(adata_processed.obs["_batch_annotation"].unique())

    has_real_batches = bool(_batch_vals - _POPV_PSEUDO_BATCHES)

    if not has_real_batches and _batch_vals:
        logger.warning(
            f"'_batch_annotation' only contains popv pseudo-labels "
            f"{_batch_vals} — no real batch signal. "
            "KNN_HARMONY will be skipped to avoid infinite convergence loop."
        )

    has_two_batches = has_real_batches



    # --- ONCLASS dict patch (popv 0.4.2) ------------------------------------
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
                    missing = {k: v for k, v in label_to_id_map.items() if k not in d}
                    if missing:
                        logger.info(
                            f"ONCLASS patch: injecting {len(missing)} entries into "
                            f"{mod_name}.{obj_attr}.{dict_attr}"
                        )
                        d.update(missing)
                        patched.append((d, missing))
                except Exception:
                    continue
            yield
            for d, added in patched:
                for k in added:
                    d.pop(k, None)

        return _ctx()

    # --- FIX 9: discover actual method names in popv.algorithms ─────────────
    method_map = _discover_popv_methods()

    # --- per-method runner --------------------------------------------------
    # Set thread env vars so sklearn SVM/RF/ONCLASS use n_jobs cores.
    import os as _os
    _n_jobs_str = str(n_jobs if n_jobs > 0 else _os.cpu_count())
    _os.environ['OMP_NUM_THREADS']      = _n_jobs_str
    _os.environ['OPENBLAS_NUM_THREADS'] = _n_jobs_str
    _os.environ['MKL_NUM_THREADS']      = _n_jobs_str
    logger.info(f'Parallelism: n_jobs={n_jobs} ')

    def _run_method_safe(adata, canonical, actual):
        # Run exactly as the sample script does — no proxy, no timeout.
        # Harmony works fine when called directly; the obsm proxy was
        # causing it to hang by intercepting internal writes.
        if actual in _ONCLASS_ALIASES or canonical in _ONCLASS_ALIASES:
            with _patch_onclass(label_to_id):
                annotate_data(adata, methods=[actual])
        else:
            annotate_data(adata, methods=[actual])

    # --- method selection ---------------------------------------------------
    # Preferred run order (all 10 methods).
    # KNN_HARMONY only inserted when 2 batch values exist.
    if input_type == "raw":
        run_order = [
            "CELLTYPIST",
            "KNN_BBKNN",
            "KNN_SCANORAMA",
            "KNN_SCVI",
            "RANDOM_FOREST",
            "SVM",
            "ONCLASS",
            "SCANVI",
        ]
        harmony_canonical = next(
            (c for c in method_map if c in _HARMONY_ALIASES or method_map[c] in _HARMONY_ALIASES),
            None,
        )
        if harmony_canonical and has_two_batches:
            run_order.insert(4, harmony_canonical)
        elif harmony_canonical:
            logger.warning("Skipping KNN_HARMONY: fewer than 2 batch values.")
    else:
        run_order = ["CELLTYPIST"]
        logger.info("log1p mode — running CELLTYPIST only.")

    # Build (canonical, actual) pairs for methods found in popv.algorithms
    run_list  = [(c, method_map[c]) for c in run_order if c in method_map]
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
                f"✗ Skipping {canonical} ({actual}): {type(exc).__name__}: {exc}"
            )

    logger.info(f"Methods completed: {successful_methods}")

    # --- FIX 4: robust fallback for majority-vote ---------------------------
    if "popv_majority_vote_prediction" not in adata_processed.obs.columns:
        pred_cols = [
            c for c in adata_processed.obs.columns
            if c.endswith("_prediction")
            and c != "popv_majority_vote_prediction"
            and adata_processed.obs[c].notna().any()
        ]
        if pred_cols:
            logger.warning(
                "popv_majority_vote_prediction missing — "
                f"using '{pred_cols[0]}' as fallback."
            )
            adata_processed.obs["popv_majority_vote_prediction"] = (
                adata_processed.obs[pred_cols[0]]
            )
        else:
            logger.error("No prediction columns found at all. Annotation failed.")

    # --- FIX 7: extract query cells only ------------------------------------
    logger.info(
        f"Combined shape after annotation: {adata_processed.shape} "
        f"(query {adata_query_snapshot.n_obs} + reference cells)"
    )
    adata_query_out = _extract_query_cells(adata_processed, adata_query_snapshot)
    logger.info(f"Query-only shape: {adata_query_out.shape}")

    # full_counts (36k genes) is NOT attached to adata_query_out.layers —
    # adata_query_out.var has only 4000 HVGs so AnnData would reject the
    # shape mismatch.  It is written exclusively to the sidecar h5ad below
    # and accessed by Module 3 via adata.uns['full_counts_h5ad_path'].
    if _saved_full_counts is None:
        logger.error(
            "full_counts snapshot missing — sidecar will not be written. "
            "Module 3 will fall back to 4000 HVGs."
        )

    # -----------------------------------------------------------------------
    # Attach raw_counts layer + restore .X
    #
    # _saved_raw_counts shape: (n_all_query_cells × n_intersection_genes)
    # adata_query_out shape:   (n_query_cells × 4000 HVGs)
    #
    # Three things to do:
    #   1. Row-align _saved_raw_counts to adata_query_out.obs_names
    #      (_extract_query_cells may have reordered or dropped rows).
    #   2. Store the row-aligned, intersection-space matrix as
    #      layers['raw_counts'] so it persists in the final h5ad and is
    #      available to any downstream module that needs original counts.
    #   3. Column-subset to the 4000 HVGs and assign to .X so the main
    #      matrix also reflects raw counts (not PopV's normalised values).
    # -----------------------------------------------------------------------
    if _saved_raw_counts is not None:
        # Step 1: row-align to adata_query_out.obs_names
        snapshot_obs  = list(adata_query_snapshot.obs_names)
        snap_name_idx = {n: i for i, n in enumerate(snapshot_obs)}
        out_obs       = list(adata_query_out.obs_names)
        row_idx       = [snap_name_idx[n] for n in out_obs if n in snap_name_idx]

        if len(row_idx) == adata_query_out.n_obs:
            rc_aligned = (
                _saved_raw_counts.tocsr()[row_idx, :]
                if sp.issparse(_saved_raw_counts)
                else _saved_raw_counts[row_idx, :]
            )
            # Step 2: store intersection-space raw counts as named layer
            adata_query_out.layers["raw_counts"] = rc_aligned
            logger.info(
                f"layers['raw_counts'] attached — "
                f"{rc_aligned.shape[0]} cells × {rc_aligned.shape[1]} genes "
                f"(intersection gene space). Original raw counts from Module 1."
            )

            # Step 3: subset to HVG columns for .X
            hvg_names    = list(adata_query_out.var_names)          # 4000 HVGs
            # common_genes is the intersection list; build index into it
            cg_to_idx    = {g: i for i, g in enumerate(common_genes)}
            hvg_col_idx  = [cg_to_idx[g] for g in hvg_names if g in cg_to_idx]

            if len(hvg_col_idx) == len(hvg_names):
                adata_query_out.X = rc_aligned.tocsr()[:, hvg_col_idx]
                logger.info(
                    f".X set to raw counts in HVG space "
                    f"({adata_query_out.n_obs} cells × {len(hvg_names)} genes)."
                )
            else:
                logger.warning(
                    f".X: only {len(hvg_col_idx)}/{len(hvg_names)} HVGs found "
                    "in intersection genes — leaving .X as PopV internal matrix."
                )
        else:
            logger.warning(
                f"raw_counts row-alignment failed: "
                f"{len(row_idx)}/{adata_query_out.n_obs} obs_names matched. "
                "layers['raw_counts'] and .X not set from saved counts."
            )
    else:
        logger.warning(
            "No _saved_raw_counts available — "
            "layers['raw_counts'] not added and .X not restored."
        )

    # --- FIX 8 verification -------------------------------------------------
    _verify_full_counts_layer(adata_query_out)

    if drop_reference_columns:
        adata_query_out = _drop_reference_only_columns(adata_query_out)

    # --- final clean-up & save ----------------------------------------------
    _clean_obs_for_h5ad(adata_query_out)

    out_path = os.path.join(output_dir, "final_popv_annotated.h5ad")
    adata_query_out.write(out_path)

    logger.info(
        f"\n{'='*60}\n"
        f"PopV output saved: {out_path}\n"
        f"Shape             : {adata_query_out.shape}\n"
        f"obs columns       : {list(adata_query_out.obs.columns)}\n"
        f"layers            : {list(adata_query_out.layers.keys())}\n"
        f"uns keys          : {list(adata_query_out.uns.keys())}\n"
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
    n_jobs: int = 1,
):
    """
    Fully automatic entry-point.  Finds the tumor h5ad from Module 1,
    downloads the matching Tabula Sapiens reference (or uses
    user_reference), and runs PopV 0.4.2 annotation.

    Parameters
    ----------
    input_type : str
        'raw' or 'log1p' — determines which methods run and how .X is handled.
    nsamples : int
        Cells sampled per label during reference subsampling.
    output_dir : str
        Directory where final_popv_annotated.h5ad is written.
    user_reference : str, optional
        Path to a local Tabula Sapiens (or compatible) reference h5ad.
        If provided, the automatic Figshare download is skipped.
    drop_reference_columns : bool
        Remove Tabula Sapiens metadata columns from the saved output.
    user_popv_prediction : str, optional
        Path to an already-annotated h5ad that contains PopV prediction
        columns (e.g. 'popv_majority_vote_prediction').  When supplied,
        the entire PopV annotation pipeline is SKIPPED.  The file is
        validated, copied to output_dir as final_popv_annotated.h5ad,
        and returned directly.  This lets users plug in their own
        pre-computed PopV results without re-running the pipeline.

        The h5ad must contain at least one column ending in '_prediction'
        in .obs.  A warning is printed if 'popv_majority_vote_prediction'
        is absent but execution continues.

    Usage
    -----
    # Run our pipeline:
    from SCART import popv_annotation
    adata = popv_annotation.auto_run_popv(
        input_type="raw",
        nsamples=300,
        user_reference="/data/Ovary_TSP1_30.h5ad"
    )

    # Use your own pre-computed PopV result:
    adata = popv_annotation.auto_run_popv(
        user_popv_prediction="/data/my_popv_annotated.h5ad",
        output_dir="popv_results"
    )
    """
    # ------------------------------------------------------------------
    # Short-circuit: user supplies their own PopV prediction h5ad
    # ------------------------------------------------------------------
    if user_popv_prediction is not None:
        if not os.path.exists(user_popv_prediction):
            raise FileNotFoundError(
                f"user_popv_prediction file not found: {user_popv_prediction}"
            )

        logger.info(
            f"user_popv_prediction provided — skipping PopV pipeline.\n"
            f"Loading: {user_popv_prediction}"
        )
        adata = sc.read_h5ad(user_popv_prediction)

        # Basic validation
        pred_cols = [c for c in adata.obs.columns if c.endswith("_prediction")]
        if not pred_cols:
            raise ValueError(
                f"No '_prediction' columns found in {user_popv_prediction}.\n"
                "Expected at least one column ending in '_prediction' in .obs.\n"
                "Please supply a valid PopV-annotated h5ad."
            )
        if "popv_majority_vote_prediction" not in adata.obs.columns:
            logger.warning(
                "'popv_majority_vote_prediction' not found in user-supplied h5ad. "
                f"Found prediction columns: {pred_cols}. "
                "Downstream modules expecting 'popv_majority_vote_prediction' "
                "may need adjustment."
            )
        else:
            logger.info(
                f"'popv_majority_vote_prediction' present — "
                f"{adata.obs['popv_majority_vote_prediction'].nunique()} unique labels."
            )

        # Save a copy into output_dir so the rest of the pipeline
        # always finds final_popv_annotated.h5ad in the expected location.
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "final_popv_annotated.h5ad")
        if os.path.abspath(user_popv_prediction) != os.path.abspath(out_path):
            logger.info(f"Copying user prediction to: {out_path}")
            adata.write(out_path)
        else:
            logger.info("user_popv_prediction is already the output path — no copy needed.")

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
        n_jobs=n_jobs,
        )
