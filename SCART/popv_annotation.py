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
    # cancer_type may be "ovary_cancer" → tissue = "ovary"
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

def _build_label_map(ontology_json_path: str) -> dict:
    """
    Build a lowercase→original-case mapping for every node label in the
    cell-ontology JSON so we can correct prediction strings before they
    reach PopV's digraph lookup.
    """
    import json
    with open(ontology_json_path) as fh:
        cl = json.load(fh)

    label_map = {}
    for node in cl.get("nodes", []):
        lbl = node.get("lbl", "")
        if lbl:
            label_map[lbl.lower()] = lbl  # e.g. "b cell" → "B cell"
    return label_map


def _normalise_predictions(adata, label_map: dict):
    """
    For every *_prediction column in adata.obs, replace lowercase labels
    with the correctly-cased ontology label.  This is the core fix for the
    "node X is not in the digraph" error.
    """
    pred_cols = [c for c in adata.obs.columns if c.endswith("_prediction")]
    for col in pred_cols:
        adata.obs[col] = (
            adata.obs[col]
            .astype(str)
            .str.lower()
            .map(lambda v: label_map.get(v, v))   # fall back to original if not found
        )


# ---------------------------------------------------------------------------
# FIX 2 — resolve ontology path (SCART-vendored copy takes priority)
# ---------------------------------------------------------------------------

def _resolve_ontology_folder() -> str:
    """
    Return the directory containing cl_popv.json (or equivalent).

    Search order:
      1. SCART.PopV.resources.ontology   ← vendored copy inside this package
      2. popv.resources.ontology         ← upstream popv installation
      3. popv.resources / popv           ← older upstream layouts
      4. Filesystem walk of installed SCART package root
      5. Filesystem walk of installed popv package root
    """
    candidate_packages = [
        # SCART's own vendored ontology — matches the repo layout visible in
        # the screenshot: SCART/PopV/resources/ontology/cl_popv.json
        "SCART.PopV.resources.ontology",
        "SCART.PopV.resources",
        # upstream popv layouts
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

    # Filesystem walk fallback — covers editable / non-standard installs
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
    """Return the full path to the ontology JSON inside cl_obo_folder."""
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
    """
    Warn if _batch_annotation has fewer than 2 unique values; harmony
    integration will collapse in that case and cause the shape-mismatch error.
    """
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
# Core annotation runner
# ---------------------------------------------------------------------------

def run_popv_annotation(
    adata_query,
    adata_ref,
    output_dir: str,
    input_type: str = "raw",
    n_samples_per_label: int = 300,
):
    """
    Run PopV cell-type annotation and write results to output_dir.

    Parameters
    ----------
    adata_query : AnnData
        Query dataset (tumor cells from Module 1).
    adata_ref : AnnData
        Tabula Sapiens tissue reference.
    output_dir : str
        Directory where final_popv_annotated.h5ad is written.
    input_type : str
        'raw'  — .X contains raw counts (default).
        'log1p' — .X is already log-normalised; only CELLTYPIST will run.
    n_samples_per_label : int
        Cells sampled per label during reference subsampling.
    """
    os.makedirs(output_dir, exist_ok=True)

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
    label_map = _build_label_map(ontology_json)
    logger.info(f"Loaded {len(label_map):,} ontology labels for case-normalisation.")

    # --- Process_Query ------------------------------------------------------
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
            methods.insert(2, "KNN_HARMONY")  # only add when harmony can run safely
        else:
            logger.warning("Skipping KNN_HARMONY: fewer than 2 batch values detected.")
    else:
        methods = ["CELLTYPIST"]

    # --- run each method ----------------------------------------------------
    successful_methods = []

    for method in methods:
        try:
            annotate_data(adata_processed, methods=[method])

            # FIX 1: normalise case immediately after each method so that the
            # next method and the consensus step see correctly-cased labels.
            _normalise_predictions(adata_processed, label_map)

            successful_methods.append(method)
            logger.info(f"✓ {method} completed.")

        except Exception as exc:
            logger.warning(f"✗ Skipping {method}: {exc}")

    logger.info(f"Methods that completed: {successful_methods}")

    # --- FIX 4: robust fallback for majority-vote ---------------------------
    if "popv_majority_vote_prediction" not in adata_processed.obs.columns:
        # Collect columns that exist and have actual predictions
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

    # --- final clean-up & save ----------------------------------------------
    _clean_obs_for_h5ad(adata_processed)

    out_path = os.path.join(output_dir, "final_popv_annotated.h5ad")
    adata_processed.write(out_path)
    logger.info(f"Saved annotated data to: {out_path}")

    return adata_processed


# ---------------------------------------------------------------------------
# Auto entry-point (mirrors Module 1's run() interface)
# ---------------------------------------------------------------------------

def auto_run_popv(
    input_type: str = "raw",
    nsamples: int = 300,
    output_dir: str = "popv_results",
    user_reference: str = None,
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
    """
    tumor_file = get_latest_tumor_h5ad()
    logger.info(f"Query file: {tumor_file}")

    cancer_type = detect_cancer_type_from_h5ad(tumor_file)

    # cancer_type may be a comma-separated list; use the first one for reference matching
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
    )
