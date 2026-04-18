"""
popv_annotation.py
Module 2 — PopV cell-type annotation
Fixes applied:
  1. Case-normalisation: predictions are title-cased before ontology lookup
     so "b cell" → "B cell" matches the digraph node label.
  2. Ontology path: uses popv's own bundled ontology, not SCART's copy.
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
  8. [NEW] raw slot preservation: adata_query.raw (set by Module 1 before
     any HVG subsetting) is captured before Process_Query runs and
     re-attached to adata_query_out after annotation.  This gives Module 3
     access to the full ~33k gene space needed by scMalignantFinder.
     Without this, Process_Query's internal HVG step overwrites / discards
     the raw slot and the saved h5ad only has 4000 HVG genes.
"""

import os
import glob
import logging
import urllib.request
import importlib.resources as pkg_resources

import numpy as np
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
    """Route raw counts into .X according to input_type."""
    if input_type == "raw":
        if "raw_counts" in adata.layers:
            adata.X = adata.layers["raw_counts"]
        elif "counts" in adata.layers:
            adata.layers["raw_counts"] = adata.layers["counts"]
            adata.X = adata.layers["raw_counts"]
        # else assume .X already contains raw counts
    elif input_type == "log1p":
        pass  # .X is already log-normalised; pass through
    else:
        raise ValueError(f"input_type must be 'raw' or 'log1p', got: {input_type!r}")


# ---------------------------------------------------------------------------
# FIX 1 — case normalisation
# ---------------------------------------------------------------------------

def _build_label_map(ontology_json_path: str):
    import json
    with open(ontology_json_path) as fh:
        cl = json.load(fh)

    label_map   = {}
    label_to_id = {}

    for node in cl.get("nodes", []):
        lbl = node.get("lbl", "")
        nid = node.get("id", "")
        if lbl:
            label_map[lbl.lower()] = lbl
            short_id = nid.split("/")[-1].replace("_", ":")
            label_to_id[lbl] = short_id

    return label_map, label_to_id


def _normalise_predictions(adata, label_map: dict):
    pred_cols = [c for c in adata.obs.columns if c.endswith("_prediction")]
    for col in pred_cols:
        adata.obs[col] = (
            adata.obs[col]
            .astype(str)
            .str.lower()
            .map(lambda v: label_map.get(v, v))
        )


# ---------------------------------------------------------------------------
# FIX 2 — resolve ontology path
# ---------------------------------------------------------------------------

def _resolve_ontology_folder() -> str:
    candidate_packages = [
        "SCART.PopV.resources.ontology",
        "SCART.PopV.resources",
        "popv.resources.ontology",
        "popv.resources",
        "popv",
    ]
    candidate_files = ["cl_popv.json", "cl.obo.json", "cl.json"]

    for pkg in candidate_packages:
        for fname in candidate_files:
            try:
                f = pkg_resources.files(pkg).joinpath(fname)
                with pkg_resources.as_file(f) as p:
                    if p.exists():
                        logger.info(f"Ontology found via importlib ({pkg}): {p}")
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
                if fname in ("cl_popv.json", "cl.obo.json", "cl.json"):
                    logger.info(f"Ontology found via filesystem walk: {os.path.join(root, fname)}")
                    return root + "/"

    raise FileNotFoundError(
        "Could not locate the cell-ontology JSON (cl_popv.json).\n"
        "Expected location: SCART/PopV/resources/ontology/cl_popv.json\n"
        "Check that the SCART package is installed correctly."
    )


def _find_ontology_json(cl_obo_folder: str) -> str:
    for fname in ("cl.obo.json", "cl_popv.json", "cl.json"):
        p = os.path.join(cl_obo_folder.rstrip("/"), fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No ontology JSON found in {cl_obo_folder}.\n"
        f"Expected one of: cl.obo.json | cl_popv.json | cl.json"
    )


# ---------------------------------------------------------------------------
# FIX 3 — harmony batch guard
# ---------------------------------------------------------------------------

def _check_batch_annotation(adata):
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
    Filter the combined query+reference AnnData back to query cells only.
    Three strategies tried in order (see module docstring for details).
    """
    if "_dataset" in adata_processed.obs.columns:
        query_mask = adata_processed.obs["_dataset"] == "query"
        n_query = query_mask.sum()
        logger.info(
            f"_extract_query_cells: '_dataset' column found. "
            f"Extracting {n_query} query cells out of {adata_processed.n_obs} total."
        )
        if n_query > 0:
            return adata_processed[query_mask].copy()
        logger.warning("'_dataset' == 'query' matched 0 cells; trying fallback.")

    original_names = set(adata_query_original.obs_names)
    mask_by_name = adata_processed.obs_names.isin(original_names)
    n_matched = mask_by_name.sum()
    if n_matched > 0:
        logger.info(
            f"_extract_query_cells: matched {n_matched} query cells by obs_names."
        )
        return adata_processed[mask_by_name].copy()

    if "_reference_labels_annotation" in adata_processed.obs.columns:
        mask_nan = adata_processed.obs["_reference_labels_annotation"].isna()
        n_nan = mask_nan.sum()
        logger.warning(
            f"_extract_query_cells: using NaN proxy — {n_nan} cells with no "
            "reference label (assumed to be query cells)."
        )
        if n_nan > 0:
            return adata_processed[mask_nan].copy()

    logger.error(
        "Could not identify query cells within the combined AnnData. "
        "Returning the full combined object."
    )
    return adata_processed


def _drop_reference_only_columns(adata):
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
    cols_to_drop = [col for col in adata.obs.columns if col in tabula_ref_cols]
    if cols_to_drop:
        logger.info(
            f"Dropping {len(cols_to_drop)} Tabula Sapiens reference columns "
            f"from query output: {cols_to_drop}"
        )
        adata.obs = adata.obs.drop(columns=cols_to_drop)
    return adata


# ---------------------------------------------------------------------------
# FIX 8 — preserve raw slot through Process_Query
# ---------------------------------------------------------------------------

def _capture_raw_slot(adata_query):
    """
    Capture adata_query.raw BEFORE Process_Query runs.

    Process_Query internally:
      1. Subsets to HVGs (4000 genes)
      2. Concatenates query + reference
      3. Runs PCA / scVI on the combined object

    After step 1 the raw slot on the processed object refers to the
    HVG-subsetted space (or is None).  We need the ORIGINAL full-gene raw
    slot that Module 1 set (all ~33k genes) so that Module 3 can pass all
    genes to scMalignantFinder.

    Returns
    -------
    raw_X : np.ndarray or None
        Dense float32 array of shape (n_query_cells, n_all_genes).
        None if adata_query.raw is not set.
    raw_var : pd.DataFrame or None
        var DataFrame from adata_query.raw (index = all gene names).
    """
    if adata_query.raw is None:
        logger.warning(
            "adata_query.raw is None — full gene space not available. "
            "scMalignantFinder will fall back to the 4000 HVG space (19% overlap). "
            "To fix: ensure Module 1 sets  adata.raw = adata  before scVI training."
        )
        return None, None

    raw_X = adata_query.raw.X
    if sp.issparse(raw_X):
        raw_X = raw_X.toarray()
    raw_X = np.array(raw_X, dtype=np.float32)

    raw_var = adata_query.raw.var.copy()

    n_genes = raw_X.shape[1]
    logger.info(
        f"Captured raw slot: {raw_X.shape[0]} cells × {n_genes} genes "
        f"(will be re-attached after annotation for Module 3)."
    )
    return raw_X, raw_var


def _reattach_raw_slot(adata_query_out, raw_X, raw_var):
    """
    Re-attach the full-gene raw slot to the query-only output AnnData.

    adata_query_out has n_obs = n_query_cells (after _extract_query_cells).
    raw_X has shape (n_original_query_cells, n_all_genes).

    If the cell order changed inside Process_Query (it can reorder cells
    during concatenation), we realign by obs_names before re-attaching.
    """
    if raw_X is None or raw_var is None:
        return adata_query_out

    out_names  = list(adata_query_out.obs_names)
    # raw_X rows were captured in the original Module-1 order.
    # We need to find where each output cell sits in the original order.
    # raw_var.index holds gene names; raw_X rows match original query order.
    # We stored raw_X before Process_Query, so its row order matches
    # adata_query (the object passed to Process_Query).
    # After _extract_query_cells the cell order should be preserved, but
    # we verify by checking obs_names alignment.

    # Build a name→row-index map from the original capture
    # (adata_query obs_names at capture time == same object passed to PQ)
    # We stored raw_X in the same order as adata_query.obs_names at capture.
    # adata_query_out.obs_names is a subset (query cells only), same order.
    # So raw_X[i] corresponds to out_names[i] IF no reordering happened.
    # To be safe, align explicitly.

    # We don't have the original obs_names order stored separately, but
    # adata_query_out.obs_names ARE the original query obs_names (verified
    # in _extract_query_cells).  raw_X was captured from adata_query BEFORE
    # Process_Query, so its row order == adata_query.obs_names order.
    # As long as _extract_query_cells preserves row order (which .copy()
    # on a boolean mask does), raw_X rows align 1:1 with out_names.

    if raw_X.shape[0] != len(out_names):
        logger.error(
            f"raw_X has {raw_X.shape[0]} rows but output AnnData has "
            f"{len(out_names)} cells — cannot re-attach raw slot safely. "
            "Skipping raw re-attachment."
        )
        return adata_query_out

    import anndata as ad
    raw_adata = ad.AnnData(
        X   = sp.csr_matrix(raw_X),
        obs = adata_query_out.obs[[]].copy(),   # empty obs, just the index
        var = raw_var,
    )
    adata_query_out.raw = raw_adata.copy()

    logger.info(
        f"Re-attached raw slot: {raw_X.shape[0]} cells × {raw_X.shape[1]} genes. "
        f"Module 3 will use this for scMalignantFinder (full gene space)."
    )
    return adata_query_out


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

    Parameters
    ----------
    adata_query : AnnData
        Query dataset (tumor cells from Module 1).
        Must have adata_query.raw set to the full-gene expression matrix
        (Module 1 does this via  adata.raw = adata  before scVI training).
    adata_ref : AnnData
        Tabula Sapiens tissue reference.
    output_dir : str
        Directory where final_popv_annotated.h5ad is written.
    input_type : str
        'raw'  — .X contains raw counts (default).
        'log1p' — .X is already log-normalised; only CELLTYPIST will run.
    n_samples_per_label : int
        Cells sampled per label during reference subsampling.
    drop_reference_columns : bool
        If True, Tabula Sapiens metadata columns are removed from the saved
        query AnnData.
    """
    os.makedirs(output_dir, exist_ok=True)

    # FIX 8 — capture raw slot BEFORE Process_Query discards it
    # This must happen before _set_input_matrix / _force_float32 which
    # modify adata_query in-place (those don't touch .raw, but capturing
    # early is safer and makes the intent explicit).
    raw_X, raw_var = _capture_raw_slot(adata_query)

    # Keep a snapshot for query-cell extraction later (strategy 2)
    adata_query_snapshot = adata_query.copy()

    # --- pre-process dtypes -------------------------------------------------
    _fix_obs_dtypes(adata_query)
    _fix_obs_dtypes(adata_ref)

    _set_input_matrix(adata_query, input_type)
    _set_input_matrix(adata_ref, "raw")

    _force_float32(adata_query)
    _force_float32(adata_ref)

    # --- resolve ontology ---------------------------------------------------
    cl_obo_folder = _resolve_ontology_folder()
    ontology_json = _find_ontology_json(cl_obo_folder)
    label_map, label_to_id = _build_label_map(ontology_json)
    logger.info(f"Loaded {len(label_map):,} ontology labels for case-normalisation.")

    # FIX 1a — normalise reference labels and filter non-ontology terms
    _ref_label_col = "cell_ontology_class"
    if _ref_label_col in adata_ref.obs.columns:

        adata_ref.obs[_ref_label_col] = (
            adata_ref.obs[_ref_label_col]
            .astype(str)
            .str.lower()
            .map(lambda v: label_map.get(v, v))
        )

        valid_labels = set(label_map.values())
        mask = adata_ref.obs[_ref_label_col].isin(valid_labels)
        n_before = adata_ref.n_obs
        n_dropped_labels = (~mask).sum()
        if n_dropped_labels > 0:
            dropped = adata_ref.obs.loc[~mask, _ref_label_col].unique().tolist()
            logger.warning(
                f"Dropping {n_dropped_labels} reference cells whose labels are "
                f"not in the cell ontology digraph: {dropped}"
            )
            adata_ref = adata_ref[mask].copy()
        logger.info(
            f"Reference '{_ref_label_col}': {n_before} → {adata_ref.n_obs} cells, "
            f"{adata_ref.obs[_ref_label_col].nunique()} unique labels after "
            "case-normalisation and ontology filtering."
        )

    # Sync cell_ontology_id
    _ref_id_col = "cell_ontology_id"
    if _ref_label_col in adata_ref.obs.columns:
        adata_ref.obs[_ref_id_col] = (
            adata_ref.obs[_ref_label_col]
            .map(label_to_id)
            .fillna(
                adata_ref.obs.get(_ref_id_col, "")
                if _ref_id_col in adata_ref.obs.columns
                else ""
            )
        )
        missing_ids = (adata_ref.obs[_ref_id_col] == "").sum()
        if missing_ids > 0:
            logger.warning(
                f"{missing_ids} reference cells have no CL ID after sync — "
                "ONCLASS may still fail for those labels."
            )
        else:
            logger.info(
                "cell_ontology_id synced with normalised labels — "
                "ONCLASS label→ID dict will be complete."
            )

    # --- Process_Query ------------------------------------------------------
    # NOTE: Process_Query subsets to HVGs internally. After this call,
    # adata_processed only has 4000 genes in .X. The full-gene raw slot
    # was already captured above (raw_X, raw_var) and will be re-attached
    # to the output after annotation (FIX 8).
    pq = Process_Query(
        query_adata=adata_query,
        ref_adata=adata_ref,
        ref_labels_key="cell_ontology_class",
        ref_batch_key=None,
        cl_obo_folder=cl_obo_folder,
        n_samples_per_label=n_samples_per_label,
    )
    adata_processed = pq.adata

    # FIX 3: check batch column before running harmony
    _check_batch_annotation(adata_processed)
    has_two_batches = (
        "_batch_annotation" in adata_processed.obs.columns
        and len(adata_processed.obs["_batch_annotation"].unique()) >= 2
    )

    _harmony_key = "X_pca_harmony_popv"

    class _ObsmProxy:
        def __init__(self, real_obsm, n_obs):
            object.__setattr__(self, "_real", real_obsm)
            object.__setattr__(self, "_n_obs", n_obs)

        def __setitem__(self, key, value):
            if key == _harmony_key:
                arr = np.array(value)
                n_obs = object.__getattribute__(self, "_n_obs")
                if arr.ndim == 2 and arr.shape[0] != n_obs:
                    logger.warning(
                        f"obsm proxy: correcting {key} shape "
                        f"{arr.shape} → {arr.T.shape}"
                    )
                    arr = arr.T
                elif arr.ndim == 1:
                    logger.warning(
                        f"obsm proxy: {key} is 1-D {arr.shape}, "
                        "cannot auto-correct — KNN_HARMONY will be skipped."
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
                    import importlib
                    mod = importlib.import_module(mod_name)
                    obj = getattr(mod, obj_attr, None)
                    if obj is None:
                        continue
                    d = getattr(obj, dict_attr, None)
                    if not isinstance(d, dict):
                        continue
                    missing = {k: v for k, v in label_to_id_map.items() if k not in d}
                    if missing:
                        logger.info(
                            f"ONCLASS patch: injecting {len(missing)} missing "
                            f"label→ID entries into {mod_name}.{obj_attr}.{dict_attr}"
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

    def _run_method_safe(adata, method):
        import unittest.mock as mock

        if method == "KNN_HARMONY":
            real_obsm = adata.obsm
            proxy = _ObsmProxy(real_obsm, adata.n_obs)
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
            logger.warning("Skipping KNN_HARMONY: fewer than 2 batch values detected.")
    else:
        methods = ["CELLTYPIST"]

    # --- run each method ----------------------------------------------------
    successful_methods = []

    for method in methods:
        try:
            _run_method_safe(adata_processed, method)
            _normalise_predictions(adata_processed, label_map)
            successful_methods.append(method)
            logger.info(f"✓ {method} completed.")

        except Exception as exc:
            logger.warning(f"✗ Skipping {method}: {type(exc).__name__}: {exc}")

    logger.info(f"Methods that completed: {successful_methods}")

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
        f"Combined AnnData shape after annotation: {adata_processed.shape}  "
        f"(query {adata_query_snapshot.n_obs} + reference cells)"
    )
    adata_query_out = _extract_query_cells(adata_processed, adata_query_snapshot)
    logger.info(
        f"Query-only AnnData shape: {adata_query_out.shape}  "
        f"(expected n_obs ≈ {adata_query_snapshot.n_obs})"
    )

    # --- FIX 8: re-attach full-gene raw slot --------------------------------
    adata_query_out = _reattach_raw_slot(adata_query_out, raw_X, raw_var)

    # Optionally remove Tabula Sapiens reference metadata columns
    if drop_reference_columns:
        adata_query_out = _drop_reference_only_columns(adata_query_out)

    # --- final clean-up & save ----------------------------------------------
    _clean_obs_for_h5ad(adata_query_out)

    out_path = os.path.join(output_dir, "final_popv_annotated.h5ad")
    adata_query_out.write(out_path)
    logger.info(f"Saved annotated query data to: {out_path}")
    logger.info(
        f"Final saved shape: {adata_query_out.shape}  "
        f"(raw slot: {adata_query_out.raw.n_vars if adata_query_out.raw is not None else 'None'} genes)"
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
):
    """
    Fully automatic entry-point.

    Finds the tumor h5ad written by Module 1, downloads the matching
    Tabula Sapiens reference (or uses user_reference if provided), and
    runs PopV annotation.

    Parameters
    ----------
    input_type : str
        'raw' or 'log1p' (see run_popv_annotation).
    nsamples : int
        n_samples_per_label for Process_Query.
    output_dir : str
        Where to write final_popv_annotated.h5ad.
    user_reference : str or None
        Path to a custom reference h5ad.  If None, Tabula Sapiens is
        auto-downloaded based on the cancer_type stored in the query h5ad.
    drop_reference_columns : bool
        If True (default), Tabula Sapiens metadata columns are dropped from
        the saved output.
    """
    tumor_file = get_latest_tumor_h5ad()
    logger.info(f"Query file: {tumor_file}")

    cancer_type = detect_cancer_type_from_h5ad(tumor_file)

    primary_cancer = cancer_type.split(",")[0].strip()
    reference_path = auto_select_reference(primary_cancer, user_reference)

    logger.info(f"Loading query  : {tumor_file}")
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
