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
  8. [NEW — sidecar h5ad] Full-gene raw/log1p count preservation via a
     SEPARATE sidecar h5ad file rather than layers['full_counts'].

     WHY SIDECAR (not adata.raw, not layers):
       After gene-space alignment (query ∩ reference intersection), adata_query
       is subsetted to e.g. 23,539 common genes.  AnnData enforces that every
       layer must match the parent's .var shape — so layers['full_counts']
       (36,601 genes) cannot be reattached to the 23,539-gene object.
       adata.raw has the same constraint.

       Sidecar approach:
         • Snapshot taken BEFORE gene-space alignment (full gene space, 36 k).
         • Written as output_dir/full_counts_for_module3.h5ad after annotation.
         • Path stored in adata.uns['full_counts_h5ad_path'] so Module 3 can
           find it automatically with sc.read_h5ad(adata.uns[...]).
         • Zero shape-mismatch risk — separate file, no AnnData parent coupling.
         • Gene names available via the sidecar's .var_names directly.

       Module 3 _build_fullgene_adata_for_scm() checks uns['full_counts_h5ad_path']
       first, giving scMalignantFinder ≥90% model feature overlap.

     .X RESTORATION:
       After Process_Query + annotate_data, .X holds PopV's internal HVG-trimmed
       normalised matrix.  The final output restores .X to the user's original
       raw / log1p counts aligned to the current HVG .var (intersection genes).
       The raw_counts layer is subsetted to common genes before Process_Query
       and reattached to adata_query_out after annotation.

  popv 0.4.2 API notes:
    • Process_Query: no prediction_mode, no save_path_trained_models, no hvg kwarg.
      cl_obo_folder must point to the directory containing cl.obo (not a JSON).
    • annotate_data: annotate_data(adata, methods=[…]) — no methods_kwargs.
    • Method name strings are UPPER_SNAKE (e.g. "KNN_BBKNN", "CELLTYPIST").
    • .X must contain raw counts (integer or float32) for input_type='raw'.
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

    Parses the OBO flat-text format directly — no pronto/owlready2 dependency.
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
                current_id  = None
                current_lbl = None
            elif line.startswith("id: CL:"):
                current_id = line[4:].strip()
            elif line.startswith("name: "):
                current_lbl = line[6:].strip()
            elif line == "" and current_id and current_lbl:
                label_map[current_lbl.lower()] = current_lbl
                label_to_id[current_lbl]       = current_id
                current_id  = None
                current_lbl = None

    if current_id and current_lbl:
        label_map[current_lbl.lower()] = current_lbl
        label_to_id[current_lbl]       = current_id

    logger.info(f"OBO parsed: {len(label_map):,} CL labels loaded.")
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
    Return the directory containing cl.obo (popv 0.4.2 uses the OBO file).
    Searches importlib resources then filesystem walk.
    """
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
                        logger.info(
                            f"Ontology (OBO) found via importlib ({pkg}): {p}"
                        )
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
            "KNN_HARMONY will fail with a shape mismatch — it will be skipped."
        )


# ---------------------------------------------------------------------------
# FIX 7 — extract query cells from the combined AnnData after PopV
# ---------------------------------------------------------------------------

def _extract_query_cells(adata_processed, adata_query_original):
    """
    After Process_Query + annotate_data, the AnnData contains both query
    and reference cells.  Returns a new AnnData with ONLY the query cells.

    Strategy (in order of preference):
      1. adata.obs['_dataset'] == 'query'
      2. Match obs_names against the original query obs_names
      3. adata.obs['_reference_labels_annotation'].isna() proxy
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
# FIX 8 (SIDECAR) — preserve full-gene counts as a separate h5ad file
# ---------------------------------------------------------------------------

def _snapshot_full_counts(adata_query: anndata.AnnData):
    """
    Capture a snapshot of the full gene-space expression matrix (.X) BEFORE
    gene-space alignment trims .var to the query ∩ reference intersection.

    Returns
    -------
    full_X : scipy.sparse.csr_matrix
        Raw (or log1p) count matrix — shape (n_cells × n_full_genes).
    full_var_names : list[str]
        Gene names matching the columns of full_X.

    Call AFTER _set_input_matrix (so .X = raw/log1p counts) and BEFORE
    the gene-space alignment / Process_Query call.
    """
    X = adata_query.X
    if sp.issparse(X):
        full_X = X.tocsr().copy().astype(np.float32)
    else:
        full_X = sp.csr_matrix(np.asarray(X, dtype=np.float32))

    full_var_names = list(adata_query.var_names)

    logger.info(
        f"FIX 8: Full-gene snapshot captured — "
        f"{full_X.shape[0]} cells × {full_X.shape[1]} genes."
    )
    return full_X, full_var_names


def _write_full_counts_sidecar(
    full_X,
    full_var_names: list,
    obs: pd.DataFrame,
    output_dir: str,
) -> str:
    """
    Write the full-gene count matrix as a standalone h5ad sidecar file.

    Parameters
    ----------
    full_X : sparse matrix (n_query_cells × n_full_genes)
    full_var_names : list of gene names (length n_full_genes)
    obs : pd.DataFrame — cell metadata from the final query AnnData
    output_dir : str

    Returns
    -------
    str : absolute path to the written sidecar file.
    """
    # obs may have more or fewer rows than full_X if _extract_query_cells
    # was used.  Guard against shape mismatch.
    if full_X.shape[0] != len(obs):
        logger.warning(
            f"FIX 8: full_X has {full_X.shape[0]} cells but obs has {len(obs)} rows. "
            "Using obs index as-is; cell order assumed consistent."
        )

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
        f"FIX 8: Full-gene sidecar written → {sidecar_path} "
        f"({sidecar.shape[0]} cells × {sidecar.shape[1]} genes, {size_mb:.1f} MB)."
    )
    return sidecar_path


def _verify_full_counts_sidecar(adata_out: anndata.AnnData) -> None:
    """Log whether the full_counts sidecar was registered and exists on disk."""
    sidecar = adata_out.uns.get("full_counts_h5ad_path", None)
    if sidecar and os.path.exists(str(sidecar)):
        size_mb = os.path.getsize(sidecar) / 1e6
        logger.info(
            f"FIX 8 VERIFIED: full_counts sidecar present — {sidecar} "
            f"({size_mb:.1f} MB). "
            "Module 3 reads sc.read_h5ad(adata.uns['full_counts_h5ad_path'])."
        )
    else:
        logger.error(
            "FIX 8 FAILED: full_counts sidecar MISSING or path not set in uns. "
            "Module 3 will fall back to 4000 HVGs (~19% model feature overlap)."
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
    Run PopV (0.4.2) cell-type annotation and write results to output_dir.

    Parameters
    ----------
    adata_query : AnnData
        Query dataset (tumor cells from Module 1).
    adata_ref : AnnData
        Tabula Sapiens tissue reference.
    output_dir : str
        Directory where final_popv_annotated.h5ad and
        full_counts_for_module3.h5ad are written.
    input_type : str
        'raw'  — .X contains raw counts (default). All popv methods are run.
        'log1p' — .X is already log-normalised; only CELLTYPIST will run.
    n_samples_per_label : int
        Cells sampled per label during reference subsampling.
    drop_reference_columns : bool
        If True (default), Tabula Sapiens metadata columns are removed from
        the saved query AnnData.

    Output files
    ------------
    output_dir/final_popv_annotated.h5ad
        Query cells only, .X = original raw/log1p counts (HVG/intersection
        gene space), .obs contains all PopV prediction columns.
        uns['full_counts_h5ad_path'] points to the sidecar below.

    output_dir/full_counts_for_module3.h5ad
        Full-gene count matrix (all genes before intersection subsetting).
        Module 3 reads this for scMalignantFinder (≥90% model overlap).
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

    # --- FIX 2: resolve cl.obo folder ---------------------------------------
    cl_obo_folder = _resolve_ontology_folder()
    logger.info(f"cl_obo_folder: {cl_obo_folder}")

    # --- FIX 1: build label map from OBO ------------------------------------
    label_map, label_to_id = _build_label_map_from_obo(cl_obo_folder)
    logger.info(f"Loaded {len(label_map):,} ontology labels.")

    # FIX 1a — normalise reference labels (case only; popv 0.4.2 handles
    # unknown labels internally via its OBO digraph)
    _ref_label_col = "cell_ontology_class"
    if _ref_label_col in adata_ref.obs.columns and label_map:
        adata_ref.obs[_ref_label_col] = (
            adata_ref.obs[_ref_label_col]
            .astype(str)
            .str.lower()
            .map(lambda v: label_map.get(v, v))
        )
        logger.info(
            f"Reference labels case-normalised — "
            f"{adata_ref.n_obs} cells, "
            f"{adata_ref.obs[_ref_label_col].nunique()} unique labels."
        )

    # -----------------------------------------------------------------------
    # FIX 8 — snapshot full gene space BEFORE gene-space alignment.
    #
    # We capture full_X and full_var_names here as plain Python objects
    # (not attached to any AnnData layer) so there is NO shape-mismatch
    # risk when adata_query is later subsetted to intersection genes.
    # They will be written as a sidecar h5ad after annotation completes.
    # -----------------------------------------------------------------------
    full_X, full_var_names = _snapshot_full_counts(adata_query)

    # -----------------------------------------------------------------------
    # Gene-space alignment before Process_Query.
    #
    # Root cause: Process_Query calls anndata.concat(ref, query, join="outer",
    # fill_value="unknown").  When var_names differ, anndata fills missing genes
    # with the string "unknown" which scipy.sparse rejects ("dtype str224").
    #
    # Fix: subset both objects to the intersection of var_names so the outer
    # join never needs to fill any values.
    #
    # The full-gene snapshot above was already captured as plain matrices —
    # it is completely unaffected by this subsetting step.
    # -----------------------------------------------------------------------
    query_genes  = set(adata_query.var_names)
    ref_genes    = set(adata_ref.var_names)
    common_genes = sorted(query_genes & ref_genes)

    if len(common_genes) == 0:
        raise ValueError(
            "Query and reference share 0 genes. "
            "Check that both datasets use the same gene identifier "
            "(Ensembl ID vs symbol)."
        )

    n_query_genes = adata_query.n_vars
    n_ref_genes   = adata_ref.n_vars

    if len(common_genes) < n_query_genes or len(common_genes) < n_ref_genes:
        logger.info(
            f"Gene-space alignment: "
            f"query {n_query_genes} genes, ref {n_ref_genes} genes → "
            f"{len(common_genes)} common genes (intersection)."
        )

        # Stash raw_counts layer BEFORE subsetting so we can re-index it
        # to the common gene set after the copy.
        _raw_counts_stash = adata_query.layers.pop("raw_counts", None)

        # Clear ALL layers from both objects before subsetting to prevent
        # anndata.concat inside Process_Query from attempting to align
        # layers across mismatched gene spaces (str224 crash).
        for lk in list(adata_query.layers.keys()):
            del adata_query.layers[lk]
        for lk in list(adata_ref.layers.keys()):
            del adata_ref.layers[lk]

        adata_query = adata_query[:, common_genes].copy()
        adata_ref   = adata_ref[:, common_genes].copy()

        # Reattach raw_counts subsetted to common genes
        if _raw_counts_stash is not None:
            original_gene_list = list(adata_query_snapshot.var_names)
            gene_to_idx        = {g: i for i, g in enumerate(original_gene_list)}
            common_idx         = [
                gene_to_idx[g] for g in common_genes if g in gene_to_idx
            ]
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
                f"{len(common_idx)} common genes and reattached."
            )
    else:
        logger.info(
            f"Gene-space alignment: query and reference already share "
            f"all {len(common_genes)} genes — no subsetting needed."
        )
        # Still clear layers to prevent str224 crash from any stray layers
        for lk in list(adata_query.layers.keys()):
            if lk != "raw_counts":
                del adata_query.layers[lk]
        for lk in list(adata_ref.layers.keys()):
            del adata_ref.layers[lk]

    # Save and remove raw_counts from query before Process_Query — anndata.concat
    # would try to align it across both datasets (ref has no raw_counts layer)
    # and hit the str224 crash.  We restore it to adata_query_out after annotation.
    _saved_raw_counts = adata_query.layers.pop("raw_counts", None)

    # Final safety: no layers on either side when entering Process_Query
    for lk in list(adata_query.layers.keys()):
        del adata_query.layers[lk]
    for lk in list(adata_ref.layers.keys()):
        del adata_ref.layers[lk]

    logger.info(
        "All layers cleared from query and reference before Process_Query. "
        "raw_counts saved locally for reattachment after annotation. "
        "full_counts snapshot held as plain matrix for sidecar write."
    )

    # --- Process_Query (popv 0.4.2 signature) --------------------------------
    pq = Process_Query(
        query_adata=adata_query,
        ref_adata=adata_ref,
        ref_labels_key="cell_ontology_class",
        ref_batch_key=None,
        cl_obo_folder=cl_obo_folder,
        n_samples_per_label=n_samples_per_label,
    )
    adata_processed = pq.adata

    # FIX 3: check batch column
    _check_batch_annotation(adata_processed)
    has_two_batches = (
        "_batch_annotation" in adata_processed.obs.columns
        and len(adata_processed.obs["_batch_annotation"].unique()) >= 2
    )

    # --- popv 0.4.2 obsm proxy (harmony shape guard) ------------------------
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
                    logger.warning(
                        f"obsm proxy: {key} is 1-D — KNN_HARMONY skipped."
                    )
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
                    missing = {
                        k: v for k, v in label_to_id_map.items() if k not in d
                    }
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

    # --- per-method runner (popv 0.4.2) -------------------------------------
    def _run_method_safe(adata, method):
        import unittest.mock as mock

        if method == "KNN_HARMONY":
            real_obsm = adata.obsm
            proxy     = _ObsmProxy(real_obsm, adata.n_obs)
            with mock.patch.object(adata, "obsm", proxy):
                annotate_data(adata, methods=[method])
        elif method == "ONCLASS":
            with _patch_onclass(label_to_id):
                annotate_data(adata, methods=[method])
        else:
            annotate_data(adata, methods=[method])

    # --- method selection ---------------------------------------------------
    if input_type == "raw":
        methods = [
            "CELLTYPIST",
            "KNN_BBKNN",
            "KNN_SCVI",
            "ONCLASS",
            "SCANVI_POPV",
            "Support_Vector",
            "XGboost",
        ]
        if has_two_batches:
            methods.insert(2, "KNN_HARMONY")
        else:
            logger.warning("Skipping KNN_HARMONY: fewer than 2 batch values.")
    else:
        methods = ["CELLTYPIST"]
        logger.info("log1p mode — running CELLTYPIST only.")

    # --- run each method ----------------------------------------------------
    successful_methods = []

    for method in methods:
        try:
            _run_method_safe(adata_processed, method)
            if label_map:
                _normalise_predictions(adata_processed, label_map)
            successful_methods.append(method)
            logger.info(f"✓ {method} completed.")
        except Exception as exc:
            logger.warning(f"✗ Skipping {method}: {type(exc).__name__}: {exc}")

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

    # --- Restore original .X (raw or log1p) into the final output -----------
    # After Process_Query + annotate_data, .X holds PopV's internal HVG-trimmed
    # normalised matrix.  We restore the user's original expression values
    # aligned to the current .var (intersection/HVG gene space).
    #
    # Priority:
    #   1. layers['raw_counts'] (already column-subsetted to common genes above)
    #   2. full_X snapshot column-subsetted to current .var genes
    #   3. Leave .X as-is (PopV internal) with a warning
    logger.info("Restoring original .X to the final query output …")
    _restored = False

    if _saved_raw_counts is not None and input_type == "raw":
        # _saved_raw_counts is already aligned to common_genes — same gene space
        # as adata_query_out.var after Process_Query HVG selection.
        # Guard: ensure row count matches (should always match after extraction).
        if _saved_raw_counts.shape[0] == adata_query_out.n_obs:
            adata_query_out.X = (
                _saved_raw_counts.tocsr()
                if sp.issparse(_saved_raw_counts)
                else sp.csr_matrix(_saved_raw_counts)
            )
            logger.info(
                f".X restored from saved raw_counts "
                f"({adata_query_out.n_obs} cells × {_saved_raw_counts.shape[1]} genes)."
            )
            _restored = True
        else:
            logger.warning(
                f"raw_counts row mismatch: saved {_saved_raw_counts.shape[0]} rows "
                f"vs query_out {adata_query_out.n_obs} cells. Trying full_X fallback."
            )

    if not _restored:
        # Subset full_X to the current .var genes
        full_name_to_idx = {g: i for i, g in enumerate(full_var_names)}
        hvg_names        = list(adata_query_out.var_names)
        col_indices      = [
            full_name_to_idx[g] for g in hvg_names if g in full_name_to_idx
        ]

        if len(col_indices) == len(hvg_names):
            subset_X = full_X.tocsr()[:, col_indices]
            if subset_X.shape[0] == adata_query_out.n_obs:
                adata_query_out.X = subset_X
                logger.info(
                    f".X restored from full_X snapshot "
                    f"({input_type}, {len(col_indices)} intersection genes)."
                )
                _restored = True
            else:
                logger.warning(
                    f"full_X row mismatch: {subset_X.shape[0]} rows "
                    f"vs query_out {adata_query_out.n_obs} cells. "
                    "Leaving .X as PopV internal matrix."
                )
        else:
            logger.warning(
                f"Gene index mismatch: only {len(col_indices)}/{len(hvg_names)} "
                "current .var genes found in full_var_names. "
                "Leaving .X as PopV internal matrix."
            )

    if not _restored:
        logger.warning(
            "Could not restore original .X — no suitable source available. "
            ".X contains PopV's internal normalised HVG matrix."
        )

    # --- FIX 8: write full-gene sidecar h5ad --------------------------------
    # full_X holds the ORIGINAL expression matrix (all genes, all query cells)
    # captured before gene-space alignment.  It has the same cell order as
    # adata_query_snapshot (pre-extraction); we need to match it to the final
    # query cell set.
    #
    # Cell order alignment:
    #   full_X rows correspond to adata_query_snapshot.obs_names.
    #   adata_query_out.obs_names is a subset (query cells only).
    #   We reindex full_X rows to match adata_query_out.
    logger.info("Writing full-gene sidecar for Module 3 …")

    snapshot_names   = list(adata_query_snapshot.obs_names)
    query_out_names  = list(adata_query_out.obs_names)
    snap_name_to_idx = {n: i for i, n in enumerate(snapshot_names)}

    row_indices = [snap_name_to_idx[n] for n in query_out_names
                   if n in snap_name_to_idx]

    if len(row_indices) == adata_query_out.n_obs:
        full_X_reindexed = full_X.tocsr()[row_indices, :]
        logger.info(
            f"FIX 8: Reindexed full_X to {len(row_indices)} query cells "
            f"× {full_X_reindexed.shape[1]} full genes."
        )
    else:
        logger.warning(
            f"FIX 8: Only {len(row_indices)}/{adata_query_out.n_obs} query "
            "obs_names found in full_X snapshot. Using full_X as-is (may include "
            "reference cells if _extract_query_cells fell back to full object)."
        )
        full_X_reindexed = full_X

    sidecar_path = _write_full_counts_sidecar(
        full_X        = full_X_reindexed,
        full_var_names= full_var_names,
        obs           = adata_query_out.obs,
        output_dir    = output_dir,
    )

    # Register path so Module 3 can locate the sidecar automatically
    adata_query_out.uns["full_counts_h5ad_path"] = sidecar_path

    # --- FIX 8 verification -------------------------------------------------
    _verify_full_counts_sidecar(adata_query_out)

    # --- drop reference metadata cols if requested --------------------------
    if drop_reference_columns:
        adata_query_out = _drop_reference_only_columns(adata_query_out)

    # --- final clean-up & save ----------------------------------------------
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
    Fully automatic entry-point.  Finds the tumor h5ad from Module 1,
    downloads the matching Tabula Sapiens reference (or uses
    user_reference), and runs PopV 0.4.2 annotation.

    Parameters
    ----------
    input_type : str
        'raw' or 'log1p'.
    nsamples : int
        Cells sampled per label during reference subsampling.
    output_dir : str
        Directory where output files are written:
          • final_popv_annotated.h5ad
          • full_counts_for_module3.h5ad  (full gene space for Module 3)
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
        and returned directly.

    Usage
    -----
    # Run the full pipeline:
    from SCART import popv_annotation
    adata = popv_annotation.auto_run_popv(
        input_type="raw",
        nsamples=300,
        user_reference="/data/Ovary_TSP1_30.h5ad"
    )

    # Use a pre-computed PopV result:
    adata = popv_annotation.auto_run_popv(
        user_popv_prediction="/data/my_popv_annotated.h5ad",
        output_dir="popv_results"
    )

    # Module 3 accesses full gene space via:
    full_adata = sc.read_h5ad(adata.uns['full_counts_h5ad_path'])
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

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "final_popv_annotated.h5ad")
        if os.path.abspath(user_popv_prediction) != os.path.abspath(out_path):
            logger.info(f"Copying user prediction to: {out_path}")
            adata.write(out_path)
        else:
            logger.info(
                "user_popv_prediction is already the output path — no copy needed."
            )

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
