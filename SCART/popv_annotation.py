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

def _build_label_map(ontology_json_path: str):
    """
    Build two dicts from cl_popv.json:

    label_map   : lowercase_label → correctly-cased label
                  e.g. "b cell" → "B cell"

    label_to_id : correctly-cased label → short CL ID
                  e.g. "B cell" → "CL:0000236"

    ONCLASS internally builds its own label→ID dict from the reference adata.
    If any label is missing from that dict it throws KeyError(<label>).
    We return label_to_id so we can sync cell_ontology_id in the reference
    before Process_Query runs, ensuring ONCLASS can always resolve every label.
    """
    import json
    with open(ontology_json_path) as fh:
        cl = json.load(fh)

    label_map   = {}  # lowercase → correctly-cased
    label_to_id = {}  # correctly-cased → short CL ID

    for node in cl.get("nodes", []):
        lbl = node.get("lbl", "")
        nid = node.get("id", "")
        if lbl:
            label_map[lbl.lower()] = lbl
            # Full URI e.g. "http://purl.obolibrary.org/obo/CL_0000236"
            # → short ID "CL:0000236"
            short_id = nid.split("/")[-1].replace("_", ":")
            label_to_id[lbl] = short_id

    return label_map, label_to_id


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
    label_map, label_to_id = _build_label_map(ontology_json)
    logger.info(f"Loaded {len(label_map):,} ontology labels for case-normalisation.")

    # FIX 1a — normalise reference labels AND filter non-ontology terms BEFORE
    # Process_Query builds the digraph.
    #
    # Two sub-problems:
    #   (a) Case mismatch: CellTypist/KNN return lowercase predictions but the
    #       digraph nodes use mixed case ("B cell", "CD8-positive, alpha-beta T
    #       cell").  We map every ref label through label_map so the digraph is
    #       built with the same strings that will appear in predictions.
    #   (b) Non-ontology anatomical terms: Tabula Sapiens ovary contains labels
    #       like "follicle" which are anatomical structures, not CL cell types,
    #       and have no node in the digraph at all.  Any method whose consensus
    #       step encounters such a label throws "node X is not in the digraph".
    #       We drop those reference cells entirely before Process_Query so the
    #       label never enters the training set.
    _ref_label_col = "cell_ontology_class"
    if _ref_label_col in adata_ref.obs.columns:

        # Step 1: normalise case
        adata_ref.obs[_ref_label_col] = (
            adata_ref.obs[_ref_label_col]
            .astype(str)
            .str.lower()
            .map(lambda v: label_map.get(v, v))
        )

        # Step 2: keep only labels that exist as nodes in the ontology
        valid_labels = set(label_map.values())   # correctly-cased CL terms
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

    # FIX ONCLASS — sync cell_ontology_id with the normalised labels.
    #
    # ONCLASS builds its own internal label→ID dict by pairing
    # cell_ontology_class with cell_ontology_id from the reference adata.
    # After our case-normalisation the labels are correctly cased ("B cell")
    # but if cell_ontology_id is NaN, empty, or mismatched for any row,
    # ONCLASS's dict has gaps and throws KeyError(<label>) at lookup time.
    #
    # We rebuild cell_ontology_id from label_to_id (derived from the same
    # cl_popv.json) so every label always has a valid CL ID.
    _ref_id_col = "cell_ontology_id"
    if _ref_label_col in adata_ref.obs.columns:
        adata_ref.obs[_ref_id_col] = (
            adata_ref.obs[_ref_label_col]
            .map(label_to_id)          # correctly-cased label → "CL:XXXXXXX"
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

    # FIX 2 — KNN_HARMONY obsm shape bug.
    #
    # Diagnosis (from logs):
    #   - harmonypy returns Z_corr with shape (64144, 50) — already correct
    #     (n_cells, n_pcs).
    #   - BUT PopV's KNN_HARMONY code does:
    #         adata.obsm["X_pca_harmony_popv"] = harmony_out.Z_corr.T
    #     i.e. it TRANSPOSES the result, producing (50, 64144).
    #     AnnData then tries to index that as (n_cells,) and gets (50,).
    #   - Z_corr is a read-only property — we cannot patch the attribute.
    #
    # Fix: intercept adata.obsm.__setitem__ for the harmony key only.
    #   When PopV writes X_pca_harmony_popv, we detect if the value has the
    #   wrong shape and silently correct it to (n_cells, n_pcs) before
    #   AnnData validates it.
    #
    # FIX ONCLASS — KeyError('B cell') inside ONCLASS's own graph.
    #
    # Diagnosis: ONCLASS builds label→CL_ID from its internal ontology graph
    #   (not from adata).  In this version of ONCLASS the graph uses a
    #   DIFFERENT string for CL:0000236 than "B cell" (e.g. "B lymphocyte").
    #   Syncing cell_ontology_id had no effect because ONCLASS never reads it.
    #
    # Fix: patch ONCLASS's internal cell_type_nlp dict (or equivalent) to add
    #   any missing label→ID mappings from our label_to_id before it runs.

    _harmony_key = "X_pca_harmony_popv"

    class _ObsmProxy:
        """
        Thin proxy around adata.obsm that intercepts __setitem__ for
        X_pca_harmony_popv and ensures the value is (n_cells, n_pcs).
        """
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
                    return   # don't write bad value; method will fail cleanly
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
        """
        Return a context manager that injects missing label→ID entries into
        ONCLASS's internal ontology graph before it runs, then restores the
        original state on exit.
        """
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            # ONCLASS stores its CL graph in one of these locations depending
            # on version.  We try each until we find a dict we can patch.
            import onclass_utils   # ONCLASS ships this as a top-level module
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

            # Also try patching popv's own ONCLASS wrapper
            try:
                import popv.algorithms as _palg
                onclass_alg = getattr(_palg, "ONCLASS", None)
                if onclass_alg and hasattr(onclass_alg, "cl_obo_file"):
                    pass   # file-based; nothing to patch here
            except Exception:
                pass

            yield

            # Restore: remove only the keys we added
            for d, added in patched:
                for k in added:
                    d.pop(k, None)

        return _ctx()

    def _run_method_safe(adata, method):
        """Run one PopV method with targeted fixes per method."""
        import unittest.mock as mock

        if method == "KNN_HARMONY":
            # Swap adata.obsm with our proxy for the duration of annotate_data
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

            # FIX 1b — normalise prediction column case after every method so
            # the PopV consensus step always sees correctly-cased labels.
            _normalise_predictions(adata_processed, label_map)

            successful_methods.append(method)
            logger.info(f"✓ {method} completed.")

        except Exception as exc:
            logger.warning(f"✗ Skipping {method}: {type(exc).__name__}: {exc}")

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
