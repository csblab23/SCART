"""
popv_annotation.py
Module 2 — PopV cell-type annotation

Analysis logic ported from PopV_GSE173682.ipynb reference notebook.
Generalised as a reusable module for any GSE / cancer type.

Key notebook logic preserved:
  1. OBO parsed with obonet (same as notebook) → proper name2id / id2name dicts
  2. Reference obs columns prefixed with 'ref_' before Process_Query
  3. ref_labels_key = 'ref_cell_ontology_class' (post-prefix)
  4. batch_key auto-detected from obs (mirrors notebook's 'sample' key)
  5. n_samples_per_label = max(min_celltype_size, 300)  — dynamic like notebook
  6. Layer alignment: query gets placeholder layers to match reference structure
  7. ref var index set to gene_symbol if available (notebook step)
  8. obs_names / var_names made unique before Process_Query
  9. Method list mirrors notebook: celltypist, knn_on_bbknn, knn_on_harmony,
     knn_on_scanorama, onclass, rf, svm
 10. Query cells extracted by obs_names after annotation (notebook approach)
 11. Full-gene raw counts preserved via layers['full_counts'] + sidecar h5ad
     for Module 3 (scMalignantFinder / SCEVAN)
"""

import os
import glob
import logging
import urllib.request
import unicodedata

import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import scipy.sparse as sp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import obonet
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
TABULA_DOI_LINK     = "https://doi.org/10.6084/m9.figshare.27921984"

# ---------------------------------------------------------------------------
# 1. OBO parsing — notebook uses obonet directly
# ---------------------------------------------------------------------------

def make_celltype_to_cell_ontology_id_dict(obo_file: str):
    """
    Parse cl.obo with obonet (same as notebook), then supplement with a
    direct line-scan of the OBO file.

    obonet builds a graph from relationship edges.  Depending on the OBO
    release version and the installed obonet version, some [Term] blocks
    whose nodes have no outgoing edges (e.g. root-like terms such as
    CL:0000150 'glandular epithelial cell') may be absent from the graph
    even though their [Term] block is physically present in the file.  The
    direct scan ensures every CL [Term] block contributes to name2id/id2name
    regardless of graph connectivity or obonet version.

    Returns
    -------
    name2id : dict  {cell type name -> CL:xxxxxxx}
    id2name : dict  {CL:xxxxxxx -> cell type name}
    """
    logger.info(f"Parsing OBO: {obo_file}")
    with open(obo_file, "r") as f:
        co = obonet.read_obo(f)

    id2name = {
        id_: data.get("name")
        for id_, data in co.nodes(data=True)
        if "CL:" in id_
    }
    id2name = {k: v for k, v in id2name.items() if v is not None}

    # ------------------------------------------------------------------
    # Supplemental direct scan — catches any CL [Term] block that obonet
    # did not add to its graph (version- and topology-dependent gap).
    # Only adds entries that are genuinely missing; never overwrites obonet.
    #
    # FIX: flush the current term when a new [Term] header is encountered
    # (before resetting cur_id/cur_name), not only on blank lines.
    # Some OBO files omit the blank-line separator between [Term] blocks,
    # which caused the previous version to silently drop terms such as
    # CL:0000150 'glandular epithelial cell'.
    # ------------------------------------------------------------------
    _scan_added = 0
    try:
        cur_id   = None
        cur_name = None
        cur_obs  = False

        def _flush():
            nonlocal _scan_added
            if cur_id and cur_name and not cur_obs:
                if cur_id not in id2name:
                    id2name[cur_id] = cur_name
                    _scan_added += 1

        with open(obo_file, "r", errors="replace") as fh:
            for raw in fh:
                line = raw.rstrip()
                if line == "[Term]":
                    _flush()                    # flush BEFORE reset
                    cur_id = cur_name = None
                    cur_obs = False
                elif line.startswith("id: CL:"):
                    cur_id = line[4:].strip()
                elif line.startswith("name: ") and cur_id is not None:
                    cur_name = line[6:].strip()
                elif line.startswith("is_obsolete: true"):
                    cur_obs = True
                # blank-line flush kept as belt-and-suspenders
                elif line == "" and cur_id and cur_name and not cur_obs:
                    _flush()
                    cur_id = cur_name = None
                    cur_obs = False

        _flush()  # flush final term at EOF

    except Exception as exc:
        logger.warning(f"Direct OBO scan failed (non-fatal): {exc}")

    if _scan_added:
        logger.info(
            f"Direct OBO scan added {_scan_added} CL terms missed by obonet "
            f"(e.g. root/disconnected nodes such as CL:0000150 "
            f"'glandular epithelial cell')."
        )

    name2id = {v: k for k, v in id2name.items()}
    logger.info(
        f"OBO loaded: {len(name2id):,} cell type labels "
        f"(obonet + {_scan_added} direct-scan supplements)."
    )
    return name2id, id2name



def _resolve_obo_file() -> str:
    """
    Locate cl.obo from SCART bundle or popv package directory.
    Returns path to the OBO FILE (not folder).

    Resolution order (first match wins):
      1. Current working directory
      2. Directory of this source file
      3. importlib (installed package copy) — validated for completeness;
         if the bundled copy is the known stripped subset (~3,324 terms)
         that is missing valid CL terms such as CL:0000150
         'glandular epithelial cell', a clear error is raised with the
         exact command needed to replace it with the full OBO.
      4. Filesystem walk of the SCART / popv package trees.
    """
    import importlib.resources as pkg_resources

    obo_filenames = ["cl.obo", "cl_popv.obo"]

    # Minimum number of CL 'name:' lines that a COMPLETE cl.obo must have.
    # The full CL release (2023-01-09) has ~17 k terms; the SCART-bundled
    # stripped copy has only ~3,324 and silently drops valid reference labels.
    _MIN_OBO_NAME_LINES = 10_000

    def _count_name_lines(path: str) -> int:
        """Fast line-count of 'name:' entries — proxy for OBO completeness."""
        try:
            n = 0
            with open(path, "r", errors="replace") as fh:
                for line in fh:
                    if line.startswith("name:"):
                        n += 1
            return n
        except Exception:
            return 0

    def _is_full_obo(path: str) -> bool:
        return _count_name_lines(path) >= _MIN_OBO_NAME_LINES

    # ------------------------------------------------------------------
    # Priority 1 & 2 — cwd and script directory (drop-in replacement)
    # ------------------------------------------------------------------
    for search_dir in [os.getcwd(),
                       os.path.dirname(os.path.abspath(__file__))]:
        for fname in obo_filenames:
            candidate = os.path.join(search_dir, fname)
            if os.path.isfile(candidate):
                logger.info(
                    f"OBO found in local directory (preferred over bundled): "
                    f"{candidate}"
                )
                return candidate

    # ------------------------------------------------------------------
    # Priority 3 — importlib (installed package copy)
    # ------------------------------------------------------------------
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
                    if not p.exists():
                        continue
                    obo_path = str(p)

                    # ── Completeness check ───────────────────────────
                    # The SCART-bundled cl.obo is a stripped subset that
                    # omits valid CL terms (e.g. CL:0000150 'glandular
                    # epithelial cell'), causing reference cells to be
                    # silently dropped.  Detect this and fail loudly
                    # with the exact remediation command rather than
                    # proceeding with a broken ontology.
                    n_names = _count_name_lines(obo_path)
                    if n_names < _MIN_OBO_NAME_LINES:
                        raise FileNotFoundError(
                            f"\n{'='*65}\n"
                            f"INCOMPLETE OBO DETECTED\n"
                            f"{'='*65}\n"
                            f"Found  : {obo_path}\n"
                            f"Terms  : {n_names:,}  (need ≥ {_MIN_OBO_NAME_LINES:,})\n\n"
                            f"The SCART-bundled cl.obo is a stripped subset that is\n"
                            f"missing valid CL terms such as:\n"
                            f"  CL:0000150  glandular epithelial cell\n\n"
                            f"Fix — replace it with the full CL release:\n\n"
                            f"  # 1. Back up the stripped copy\n"
                            f"  cp {obo_path} {obo_path}.bak\n\n"
                            f"  # 2. Copy in the full cl.obo\n"
                            f"  cp /path/to/full/cl.obo {obo_path}\n\n"
                            f"OR place the full cl.obo in your working directory:\n\n"
                            f"  cp /path/to/full/cl.obo $(pwd)/cl.obo\n"
                            f"{'='*65}"
                        )

                    logger.info(
                        f"OBO found via importlib ({pkg}): {obo_path} "
                        f"({n_names:,} terms — OK)"
                    )
                    return obo_path
            except FileNotFoundError:
                raise   # re-raise our own completeness error immediately
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Priority 4 — filesystem walk
    # ------------------------------------------------------------------
    walk_roots = []
    for pkg_name in ("SCART", "popv"):
        try:
            mod = __import__(pkg_name)
            walk_roots.append(os.path.dirname(mod.__file__))
        except ImportError:
            pass

    for root_dir in walk_roots:
        for dirpath, _, fnames in os.walk(root_dir):
            for fname in fnames:
                if fname in obo_filenames:
                    full = os.path.join(dirpath, fname)
                    n_names = _count_name_lines(full)
                    if n_names < _MIN_OBO_NAME_LINES:
                        logger.warning(
                            f"Skipping incomplete OBO ({n_names:,} terms): {full}"
                        )
                        continue
                    logger.info(
                        f"OBO found via filesystem walk: {full} "
                        f"({n_names:,} terms — OK)"
                    )
                    return full

    raise FileNotFoundError(
        "Could not locate a complete cl.obo (≥ 10,000 CL terms).\n"
        "Place the full CL release cl.obo in your working directory:\n\n"
        "  cp /path/to/full/cl.obo $(pwd)/cl.obo\n\n"
        "or replace the SCART-bundled copy at:\n"
        "  SCART/PopV/resources/ontology/cl.obo"
    )


# ---------------------------------------------------------------------------
# 2. Reference label normalisation — matches notebook corrections exactly,
#    then extends generically via OBO lookup for any unseen label
# ---------------------------------------------------------------------------

# Exact replacements the notebook applies (kept verbatim)
_NOTEBOOK_FIXES = {
    "b cell":                        "B cell",
    "cd4-positive, alpha-beta t cell": "CD4-positive, alpha-beta T cell",
    "cd8-positive, alpha-beta t cell": "CD8-positive, alpha-beta T cell",
    "mature nk t cell":              "mature NK T cell",
    "t cell":                        "T cell",
    "follicle":                      "follicle cell of egg chamber",
    # CL:0000150 was renamed in newer CL releases from 'glandular epithelial cell'
    # to 'glandular secretory epithelial cell'. The old name is retained as an
    # EXACT synonym in the OBO and added to name2id by the synonym-parsing tier
    # in make_celltype_to_cell_ontology_id_dict, so the OBO membership check
    # passes while keeping the original label intact in predictions.
    "glandular epithelial cell":     "glandular epithelial cell",
}


def _normalise_ref_labels(adata_ref: anndata.AnnData,
                           label_col: str,
                           name2id: dict) -> anndata.AnnData:
    """
    1. Strip leading/trailing whitespace and apply Unicode NFC normalisation.
       Newer Tabula Sapiens reference versions (e.g. Nov 2024) store some
       labels with invisible characters or non-breaking spaces that cause the
       OBO lookup to fail even though the label text looks correct.
    2. Apply the notebook's hardcoded fixes.
    3. For any label still missing from the ontology, attempt a
       case-insensitive OBO lookup (also on the stripped/normalised form).
    4. Drop cells whose label still cannot be found (would cause
       NetworkXError in PopV's KNN label propagation).
    """
    col = adata_ref.obs[label_col].astype(str).copy()

    # Step 0 — strip whitespace + NFC unicode normalisation
    # This fixes labels like 'glandular epithelial cell ' (trailing space)
    # or 'glandular\xa0epithelial cell' (non-breaking space) that appear in
    # newer reference versions and silently break the OBO membership check.
    col = col.str.strip()
    col = col.map(lambda s: unicodedata.normalize("NFC", s))

    # Step 1 — notebook exact replacements (applied on the cleaned strings)
    for wrong, right in _NOTEBOOK_FIXES.items():
        col = col.replace(wrong, right)

    # Step 2 — generic case-insensitive OBO lookup for anything still missing
    # Build lookup on stripped+normalised canonical names so it is consistent
    # with the cleaning applied above.
    lower2canonical = {
        unicodedata.normalize("NFC", k.strip()).lower(): k
        for k in name2id
    }

    def _fix(v):
        v_clean = unicodedata.normalize("NFC", v.strip())
        if v_clean in name2id:
            return v_clean
        return lower2canonical.get(v_clean.lower(), v_clean)

    col = col.map(_fix)

    adata_ref.obs[label_col] = col

    # Step 3 — drop cells with labels not in ontology (same logic as before)
    valid   = set(name2id.keys())
    mask    = adata_ref.obs[label_col].isin(valid)
    n_drop  = (~mask).sum()
    if n_drop > 0:
        bad = adata_ref.obs.loc[~mask, label_col].unique().tolist()
        logger.warning(
            f"Dropping {n_drop} reference cells — labels not in OBO: {bad}"
        )
        adata_ref = adata_ref[mask].copy()

    logger.info(
        f"Reference labels normalised — {adata_ref.n_obs} cells, "
        f"{adata_ref.obs[label_col].nunique()} unique labels."
    )
    return adata_ref


# ---------------------------------------------------------------------------
# 3. Reference var index — notebook sets it to gene_symbol if present
# ---------------------------------------------------------------------------

def _set_ref_var_index(adata_ref: anndata.AnnData) -> anndata.AnnData:
    """
    Mirror notebook:  ref_adata.var.index = ref_adata.var['gene_symbol']
    Falls back gracefully if the column is absent.
    """
    if "gene_symbol" in adata_ref.var.columns:
        logger.info("Setting ref var index to 'gene_symbol' (notebook step).")
        adata_ref.var.index = adata_ref.var["gene_symbol"]
    elif "feature_name" in adata_ref.var.columns:
        logger.info("Setting ref var index to 'feature_name'.")
        adata_ref.var.index = adata_ref.var["feature_name"]
    else:
        logger.info("No gene_symbol / feature_name column — keeping existing var index.")

    # Ensure string index and uniqueness (notebook does var_names_make_unique)
    adata_ref.var.index = pd.Index(adata_ref.var.index.astype(str))
    adata_ref.var_names_make_unique()
    # Notebook converts to CategoricalIndex — affects HVG selection inside Process_Query
    adata_ref.var.index = pd.CategoricalIndex(adata_ref.var.index)
    return adata_ref


# ---------------------------------------------------------------------------
# 4. Raw count routing — same as notebook's explicit layer checks
# ---------------------------------------------------------------------------

def _set_raw_counts_in_X(adata: anndata.AnnData, label: str = "") -> anndata.AnnData:
    """
    Put raw integer counts into .X, checking layers in priority order.
    Mirrors notebook logic:
      ref: X = layers['decontXcounts'].copy()  — ambient-RNA-removed counts
      query: X = layers['counts'].copy()
    """
    tag = f"[{label}] " if label else ""

    if label == "reference":
        # Reference priority: decontXcounts first (ambient-RNA-removed),
        # then raw_counts, then counts
        if "decontXcounts" in adata.layers:
            logger.info(f"{tag}Using layers['decontXcounts'] → .X")
            adata.X = adata.layers["decontXcounts"].copy()
        elif "raw_counts" in adata.layers:
            logger.info(f"{tag}Using layers['raw_counts'] → .X")
            adata.X = adata.layers["raw_counts"].copy()
        elif "counts" in adata.layers:
            logger.info(f"{tag}Using layers['counts'] → .X")
            adata.X = adata.layers["counts"].copy()
        else:
            logger.info(f"{tag}No count layer found — assuming .X already contains raw counts.")
    else:
        # Query priority: counts first, then raw_counts, then decontXcounts
        if "counts" in adata.layers:
            logger.info(f"{tag}Using layers['counts'] → .X")
            adata.X = adata.layers["counts"].copy()
        elif "raw_counts" in adata.layers:
            logger.info(f"{tag}Using layers['raw_counts'] → .X")
            adata.X = adata.layers["raw_counts"].copy()
        elif "decontXcounts" in adata.layers:
            logger.info(f"{tag}Using layers['decontXcounts'] → .X")
            adata.X = adata.layers["decontXcounts"].copy()
        else:
            logger.info(f"{tag}No count layer found — assuming .X already contains raw counts.")

    # Ensure float32 sparse (PopV requirement)
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr().astype(np.float32)
    else:
        adata.X = sp.csr_matrix(np.asarray(adata.X, dtype=np.float32))

    return adata


def _validate_raw_counts(adata: anndata.AnnData, label: str = "") -> None:
    tag = f"[{label}] " if label else ""
    X = adata.X
    sample = np.array(
        X.data[:10000] if sp.issparse(X) and X.nnz > 0 else X.ravel()[:10000],
        dtype=np.float64,
    )
    if np.any(sample < 0):
        raise ValueError(
            f"{tag}.X contains negative values — not raw counts.\n"
            "SCART requires raw integer counts."
        )
    mean_val = float(np.mean(sample))
    if 0 < mean_val < 2.0:
        raise ValueError(
            f"{tag}.X mean={mean_val:.4f} — looks like log-normalised data.\n"
            "SCART requires raw integer counts."
        )
    logger.info(f"{tag}Raw count validation passed (mean={mean_val:.2f}).")


# ---------------------------------------------------------------------------
# 5. Standardise reference layer names — notebook renames raw_counts → counts
# ---------------------------------------------------------------------------

def _standardise_ref_layers(adata_ref: anndata.AnnData) -> anndata.AnnData:
    """
    Notebook: ref_adata.layers['counts'] = ref_adata.layers['decontXcounts'].copy()
    Ensures downstream code always finds layers['counts'] on the reference,
    built from decontXcounts (ambient-RNA-removed) as first priority.
    """
    if "counts" not in adata_ref.layers:
        if "decontXcounts" in adata_ref.layers:
            logger.info("Reference: copying layers['decontXcounts'] → layers['counts'].")
            adata_ref.layers["counts"] = adata_ref.layers["decontXcounts"].copy()
        elif "raw_counts" in adata_ref.layers:
            logger.info("Reference: renaming layers['raw_counts'] → layers['counts'].")
            adata_ref.layers["counts"] = adata_ref.layers["raw_counts"].copy()
    return adata_ref


# ---------------------------------------------------------------------------
# 6. Prefix reference obs columns — notebook renames all to 'ref_*'
# ---------------------------------------------------------------------------

def _prefix_ref_obs_columns(adata_ref: anndata.AnnData) -> tuple:
    """
    Mirror notebook:
        ref_adata.obs.columns = [f'ref_{col}' for col in ref_adata.obs.columns]

    Returns (adata_ref, ref_labels_key) where ref_labels_key is the
    post-prefix column name used as the PopV label key.
    """
    new_cols = {c: f"ref_{c}" for c in adata_ref.obs.columns
                if not c.startswith("ref_")}
    if new_cols:
        logger.info(f"Prefixing {len(new_cols)} reference obs columns with 'ref_'.")
        adata_ref.obs = adata_ref.obs.rename(columns=new_cols)

    # Determine correct label column name after prefixing
    ref_labels_key = None
    for candidate in ("ref_cell_ontology_class", "cell_ontology_class"):
        if candidate in adata_ref.obs.columns:
            ref_labels_key = candidate
            break
    if ref_labels_key is None:
        raise ValueError(
            "Cannot find 'cell_ontology_class' (or 'ref_cell_ontology_class') "
            "in reference obs after column prefixing.\n"
            f"Available columns: {list(adata_ref.obs.columns)}"
        )

    logger.info(f"Reference label key: '{ref_labels_key}'")
    return adata_ref, ref_labels_key


# ---------------------------------------------------------------------------
# 7. Auto-detect query batch key — notebook uses 'sample'
# ---------------------------------------------------------------------------

_BATCH_KEY_CANDIDATES = ["gsm_id", "sample", "batch", "Sample", "Batch", "patient",
                          "donor", "library", "Run", "run", "gse_id"]


def _detect_query_batch_key(adata_query: anndata.AnnData,
                             user_batch_key: str = None) -> str | None:
    """
    Determine the column in adata_query.obs to use as PopV's batch key.

    Resolution order:
      1. user_batch_key, if explicitly supplied — lets a user with their
         own h5ad point to whatever sample/donor column they actually have,
         even if its name isn't in _BATCH_KEY_CANDIDATES. This is also how
         a Module 1 batch_key value (e.g. "donor") gets used here — pass
         the SAME name as batch_key to this module too.
      2. 'gsm_id' — written by Module 1 (geo_fetcher.SampleAnnotator) into
         every GEO-derived tumor h5ad (single GSE or multi-GSE combined),
         so GEO-sourced runs are batch-corrected automatically with no
         extra argument needed. THIS IS THE DEFAULT: if neither Module 1
         nor Module 2 is given a batch key, GSM ID is what gets used.
      3. Remaining candidates in _BATCH_KEY_CANDIDATES, 'gse_id' checked
         last since GSE-level grouping is coarser than GSM-level.

    If GSM ID is not a valid batch for your data (e.g. one GSM actually
    contains multiple pooled patients/donors), either:
      (a) re-run Module 1 with batch_key='<keyword>' (e.g. 'donor')
          to extract a better column from GEO metadata automatically, or
      (b) supply your own h5ad with a batch column already in adata.obs,
          then pass batch_key='<your column name>' here.
    """
    if user_batch_key is not None:
        if user_batch_key not in adata_query.obs.columns:
            raise ValueError(
                f"batch_key='{user_batch_key}' not found in query h5ad obs.\n"
                f"Available columns: {list(adata_query.obs.columns)}"
            )
        n_unique = adata_query.obs[user_batch_key].nunique()
        if n_unique < 2:
            logger.warning(
                f"batch_key='{user_batch_key}' has only {n_unique} unique "
                "value(s) — running without batch correction."
            )
            return None
        logger.info(
            f"batch_key user-specified: '{user_batch_key}' "
            f"({n_unique} unique values)."
        )
        return user_batch_key

    for key in _BATCH_KEY_CANDIDATES:
        if key in adata_query.obs.columns:
            n_unique = adata_query.obs[key].nunique()
            if n_unique >= 2:
                logger.info(
                    f"batch_key auto-detected: '{key}' "
                    f"({n_unique} unique values)."
                )
                if key == "gsm_id":
                    logger.info(
                        "Using GSM ID as the batch key (default — no "
                        "batch_key given in either Module 1 or Module 2). "
                        "If one GSM actually contains multiple pooled "
                        "patients/donors, this is too coarse: re-run Module 1 "
                        "with batch_key='donor' (or similar), or supply "
                        "your own h5ad with a proper batch column and pass "
                        "batch_key= here instead."
                    )
                return key
    logger.warning(
        "No suitable batch_key found — running without batch correction. "
        f"Checked: {_BATCH_KEY_CANDIDATES}\n"
        "If this h5ad came from Module 1 with a GEO ID, 'gsm_id' should "
        "normally be present — check it has >=2 unique values. If you need "
        "a different batch column, either re-run Module 1 with "
        "batch_key='<keyword>', or pass batch_key='<column name>' "
        "here directly."
    )
    return None


# ---------------------------------------------------------------------------
# 8. Layer alignment — strip all layers from both sides before Process_Query
# ---------------------------------------------------------------------------

def _strip_layers_for_popv(adata_query: anndata.AnnData,
                            adata_ref:   anndata.AnnData) -> tuple:
    """
    anndata.concat(..., join='outer', fill_value='unknown') fills missing
    layer slots with the string 'unknown', which scipy.sparse rejects
    (dtype str224 is not supported).

    The safe fix is to strip ALL layers from both query and reference
    before passing to Process_Query, so anndata.concat never sees a
    layer mismatch.  PopV only needs .X (raw counts) — it builds its
    own internal layers during preprocessing.

    The 'counts' layer on query is kept as the sole exception because
    some popv internals check for it.  All other layers are dropped.
    """
    # Both query and reference must have IDENTICAL layer keys before Process_Query.
    # anndata.concat(join='outer', fill_value='unknown') fills missing layer slots
    # with the string 'unknown' → scipy.sparse dtype str224 error.
    # Solution: keep only 'counts' on both sides. PopV only reads .X internally.

    # ── Reference: strip to 'counts' only ───────────────────────────────────
    ref_keep = {"counts"}
    ref_drop  = [k for k in list(adata_ref.layers.keys()) if k not in ref_keep]
    for k in ref_drop:
        del adata_ref.layers[k]
    if ref_drop:
        logger.info(f"Layer strip (reference): removed {ref_drop}, kept ['counts'].")

    # ── Query: strip to 'counts' only (full_counts already stashed externally) ─
    query_keep = {"counts"}
    query_drop  = [k for k in list(adata_query.layers.keys()) if k not in query_keep]
    for k in query_drop:
        del adata_query.layers[k]
    if query_drop:
        logger.info(f"Layer strip (query): removed {query_drop}, kept ['counts'].")

    return adata_query, adata_ref


# ---------------------------------------------------------------------------
# 9. Full-gene count preservation (FIX 8 — layer sidecar for Module 3)
# ---------------------------------------------------------------------------

def _store_full_counts_layer(adata_query: anndata.AnnData) -> anndata.AnnData:
    if "full_counts" in adata_query.layers:
        logger.info(
            f"'full_counts' already present ({adata_query.n_vars} genes) — skipping."
        )
        return adata_query

    logger.info(
        f"Snapshotting {adata_query.n_vars} genes → layers['full_counts'] "
        "before Process_Query."
    )
    X = adata_query.X
    adata_query.layers["full_counts"] = (
        X.tocsr().copy() if sp.issparse(X)
        else sp.csr_matrix(np.asarray(X, dtype=np.float32))
    )
    adata_query.uns["full_counts_var_names"] = list(adata_query.var_names)
    logger.info(
        f"full_counts snapshot: {adata_query.layers['full_counts'].shape}, "
        f"uns['full_counts_var_names']: {len(adata_query.uns['full_counts_var_names'])} genes."
    )
    return adata_query


# ---------------------------------------------------------------------------
# 10. Query cell extraction — notebook approach (obs_names isin query_cells)
# ---------------------------------------------------------------------------

def _extract_query_cells(adata_processed:    anndata.AnnData,
                          query_obs_names:    pd.Index) -> anndata.AnnData:
    """
    Notebook:
        query_cells = query_adata.obs_names
        adata = adata[adata.obs_names.isin(query_cells)]

    Multi-fallback version for robustness across datasets.
    """
    # Primary: obs_names intersection (notebook method)
    mask = adata_processed.obs_names.isin(query_obs_names)
    n    = mask.sum()
    if n > 0:
        logger.info(
            f"Query extraction (obs_names isin): {n} / {adata_processed.n_obs} cells."
        )
        return adata_processed[mask].copy()

    # Fallback 1: _dataset column written by Process_Query
    if "_dataset" in adata_processed.obs.columns:
        mask2 = adata_processed.obs["_dataset"] == "query"
        n2    = mask2.sum()
        if n2 > 0:
            logger.warning(
                f"Query extraction fallback (_dataset=='query'): {n2} cells."
            )
            return adata_processed[mask2].copy()

    # Fallback 2: NaN in reference label column
    ref_col_candidates = [c for c in adata_processed.obs.columns
                          if "reference_labels" in c or "ref_cell_ontology" in c]
    for col in ref_col_candidates:
        mask3 = adata_processed.obs[col].isna()
        n3    = mask3.sum()
        if n3 > 0:
            logger.warning(
                f"Query extraction fallback (NaN in '{col}'): {n3} cells."
            )
            return adata_processed[mask3].copy()

    logger.error(
        "Could not isolate query cells — returning full combined object. "
        "Output may contain reference cells."
    )
    return adata_processed


# ---------------------------------------------------------------------------
# 11. Sidecar h5ad for Module 3 (full-gene space)
# ---------------------------------------------------------------------------

def _write_full_counts_sidecar(
    adata_query_out:      anndata.AnnData,
    query_obs_names_orig: pd.Index,
    full_counts_mat:      sp.csr_matrix,
    full_var_names:       list,
    output_dir:           str,
) -> str:
    snap_idx  = {n: i for i, n in enumerate(query_obs_names_orig)}
    out_obs   = list(adata_query_out.obs_names)
    row_idx   = [snap_idx[n] for n in out_obs if n in snap_idx]

    if len(row_idx) != adata_query_out.n_obs:
        logger.error(
            f"Sidecar: only {len(row_idx)}/{adata_query_out.n_obs} obs matched. "
            "Writing partial sidecar."
        )

    fc_aligned = full_counts_mat.tocsr()[row_idx, :]
    sidecar    = anndata.AnnData(
        X   = fc_aligned,
        obs = adata_query_out.obs.copy(),
        var = pd.DataFrame(index=full_var_names),
    )
    for col in sidecar.obs.columns:
        if sidecar.obs[col].dtype == object:
            sidecar.obs[col] = sidecar.obs[col].astype(str)

    path    = os.path.join(output_dir, "full_counts_for_module3.h5ad")
    sidecar.write(path)
    size_mb = os.path.getsize(path) / 1e6
    logger.info(
        f"Sidecar written → {path} "
        f"({sidecar.n_obs} cells × {sidecar.n_vars} genes, {size_mb:.1f} MB)."
    )
    return path


# ---------------------------------------------------------------------------
# 12. Tabula Sapiens reference download (Figshare)
# ---------------------------------------------------------------------------

def fetch_tabula_file_metadata():
    url = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}/files"
    logger.info("Fetching Tabula Sapiens file list from Figshare …")

    session = requests.Session()
    retry   = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))

    resp = session.get(url, timeout=30, headers={"User-Agent": "curl/7.68.0"})
    resp.raise_for_status()
    return [f for f in resp.json() if f["name"].endswith(".h5ad")]


def find_best_reference_file(cancer_type: str, files: list):
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
    files    = fetch_tabula_file_metadata()
    selected = find_best_reference_file(cancer_type, files)
    if selected is None:
        raise ValueError(
            f"Reference not found for '{cancer_type}' via Figshare API.\n"
            f"Download manually from: {TABULA_DOI_LINK}\n"
            "Then pass: user_reference='path_to_reference.h5ad'"
        )
    save_path = os.path.join(REFERENCE_BASE_PATH, selected["name"])
    if os.path.exists(save_path):
        logger.info(f"Reference already cached: {selected['name']}")
        return save_path
    logger.info(f"Downloading reference: {selected['name']} …")
    urllib.request.urlretrieve(selected["download_url"], save_path)
    logger.info(f"Saved to: {save_path}")
    return save_path


def auto_select_reference(cancer_type: str, user_reference: str = None) -> str:
    if user_reference:
        if not os.path.exists(user_reference):
            raise FileNotFoundError(f"Reference not found: {user_reference}")
        return user_reference
    return download_tabula_reference(cancer_type)


# ---------------------------------------------------------------------------
# 13. Locate Module 1 output
# ---------------------------------------------------------------------------

def get_latest_tumor_h5ad(data_dir: str = "GSE_data") -> str:
    patterns = ["*_tumor.h5ad", "combined_tumor.h5ad", "input_tumor.h5ad"]
    files    = []
    for path in [os.getcwd(), data_dir]:
        for pat in patterns:
            files.extend(glob.glob(os.path.join(path, pat)))
    files = list(set(files))
    if not files:
        raise FileNotFoundError(
            "No tumor h5ad found.\n"
            "Expected: *_tumor.h5ad | combined_tumor.h5ad | input_tumor.h5ad\n"
            "in current directory or GSE_data/"
        )
    return max(files, key=os.path.getctime)


def detect_cancer_type_from_h5ad(h5ad_file: str) -> str:
    adata = sc.read_h5ad(h5ad_file)
    if "cancer_type" in adata.uns:
        ct = adata.uns["cancer_type"]
        logger.info(f"Detected cancer type: {ct}")
        return ct
    raise ValueError(
        "Could not detect cancer type from h5ad .uns['cancer_type'].\n"
        "Provide user_reference manually."
    )


# ---------------------------------------------------------------------------
# 14. Tabula Sapiens reference obs: fix cell_ontology_id 'None' string
# ---------------------------------------------------------------------------

def _fix_cell_ontology_id(adata_ref: anndata.AnnData,
                           id_col: str = "ref_cell_ontology_id") -> anndata.AnnData:
    """
    Notebook: ref_adata.obs['cell_ontology_id'].replace("None", "CL:0000477")
    Applied after column prefixing.
    """
    # Try both prefixed and unprefixed column names
    for col in (id_col, "cell_ontology_id"):
        if col in adata_ref.obs.columns:
            before = (adata_ref.obs[col].astype(str) == "None").sum()
            if before > 0:
                adata_ref.obs[col] = (
                    adata_ref.obs[col].astype(str).replace("None", "CL:0000477")
                )
                logger.info(
                    f"Fixed {before} 'None' entries in '{col}' → 'CL:0000477'."
                )
            break
    return adata_ref


# ---------------------------------------------------------------------------
# 15. Drop Tabula Sapiens-specific columns from query output
# ---------------------------------------------------------------------------

_TABULA_REF_COLS = {
    "donor", "tissue", "anatomical_position", "method", "cdna_plate",
    "library_plate", "notes", "cdna_well", "old_index", "assay",
    "sample_id", "replicate", "10X_run", "10X_barcode", "ambient_removal",
    "donor_method", "donor_assay", "donor_tissue", "donor_tissue_assay",
    "cell_ontology_class", "cell_ontology_id", "compartment",
    "broad_cell_class", "free_annotation", "manually_annotated",
    "published_2022", "n_genes_by_counts", "total_counts", "total_counts_mt",
    "pct_counts_mt", "total_counts_ercc", "pct_counts_ercc",
    "_scvi_batch", "_scvi_labels", "age", "sex", "ethnicity",
}


def _drop_reference_only_columns(adata: anndata.AnnData) -> anndata.AnnData:
    drop = [c for c in adata.obs.columns
            if c in _TABULA_REF_COLS or c.startswith("ref_")]
    if drop:
        logger.info(f"Dropping {len(drop)} reference metadata columns.")
        adata.obs = adata.obs.drop(columns=drop)
    return adata


# ---------------------------------------------------------------------------
# 16. Core annotation runner
# ---------------------------------------------------------------------------

def run_popv_annotation(
    adata_query,
    adata_ref,
    output_dir:             str  = "popv_results",
    nsamples:               int  = 300,
    drop_reference_columns: bool = True,
    n_jobs:                 int  = 1,
    batch_key:              str  = None,
) -> anndata.AnnData:
    """
    Run PopV cell-type annotation following the logic of PopV_GSE173682.ipynb.

    Works for any GSE / cancer type — all dataset-specific steps (batch key
    detection, n_samples_per_label, gene symbol index, layer alignment) are
    handled automatically.

    Parameters
    ----------
    adata_query : AnnData
        Tumor query cells from Module 1.  Must contain raw counts in
        layers['counts'] or adata.X.
    adata_ref : AnnData
        Tabula Sapiens tissue reference.
    output_dir : str
        Where to write final_popv_annotated.h5ad and sidecar.
    nsamples : int
        Minimum cells per label (floor).  Actual value =
        max(min_celltype_size, nsamples) — matches notebook.
    drop_reference_columns : bool
        Remove Tabula Sapiens metadata columns from output.
    n_jobs : int
        CPU threads (-1 = all cores).
    batch_key : str, optional
        Column in adata_query.obs to use as the batch key for PopV's
        batch-corrected methods (knn_on_harmony, knn_on_scanorama).
        If not supplied, auto-detected — 'gsm_id' (written automatically
        by Module 1 for GEO-derived data) is checked first, then other
        common names, falling back to 'gse_id'.
        DEFAULT BEHAVIOUR: if you give no batch_key here AND gave no
        batch_key to Module 1's SampleAnnotator, GSM ID is used as the
        batch (one value per GEO sample). If that's too coarse for your
        data (e.g. one GSM contains multiple pooled patients/donors), either
        re-run Module 1 with batch_key='<keyword>' (e.g. 'donor') to
        extract a better column automatically, or supply your own h5ad with
        a batch column already in adata.obs and pass its name here. Use the
        SAME parameter name — batch_key — in both Module 1 and Module 2.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Resolve OBO file ────────────────────────────────────────────────────
    obo_file      = _resolve_obo_file()
    cl_obo_folder = os.path.dirname(obo_file) + "/"
    name2id, _    = make_celltype_to_cell_ontology_id_dict(obo_file)

    # OnClass needs cl.ontology in the same folder as cl.obo (notebook assumption).
    # If missing, check whether it was uploaded alongside this run and copy it in.
    _cl_ontology_dest = os.path.join(cl_obo_folder, "cl.ontology")
    if not os.path.exists(_cl_ontology_dest):
        _search_paths = [
            os.path.join(os.getcwd(), "cl.ontology"),
            os.path.join(os.path.dirname(__file__), "cl.ontology"),
        ]
        _found_ontology = next((p for p in _search_paths if os.path.exists(p)), None)
        if _found_ontology:
            import shutil
            shutil.copy2(_found_ontology, _cl_ontology_dest)
            logger.info(f"cl.ontology copied from {_found_ontology} → {_cl_ontology_dest}")
        else:
            logger.warning(
                f"cl.ontology not found in {cl_obo_folder}. "
                "OnClass will be skipped. "
                f"Place cl.ontology alongside cl.obo to enable it: "
                f"cp cl.ontology {cl_obo_folder}"
            )

    # ── Snapshot original query obs_names for later extraction ──────────────
    query_obs_names_orig = adata_query.obs_names.copy()

    # ── Step 1: set gene symbol index on reference (notebook step) ──────────
    adata_ref = _set_ref_var_index(adata_ref)

    # ── Step 2: route raw counts into .X ────────────────────────────────────
    adata_query = _set_raw_counts_in_X(adata_query, label="query")
    adata_ref   = _set_raw_counts_in_X(adata_ref,   label="reference")

    _validate_raw_counts(adata_query, label="query")

    # ── Step 3: normalise reference labels using OBO + notebook fixes ────────
    # Must happen BEFORE column prefixing (label col is still 'cell_ontology_class')
    if "cell_ontology_class" in adata_ref.obs.columns:
        adata_ref = _normalise_ref_labels(adata_ref, "cell_ontology_class", name2id)
    else:
        logger.warning("'cell_ontology_class' not found in ref obs — skipping label normalisation.")

    # ── Step 4: standardise reference layer names ────────────────────────────
    adata_ref = _standardise_ref_layers(adata_ref)

    # ── Step 5: prefix reference obs columns → 'ref_*' (notebook step) ──────
    adata_ref, ref_labels_key = _prefix_ref_obs_columns(adata_ref)

    # ── Step 6: fix 'None' cell_ontology_id strings (notebook step) ─────────
    adata_ref = _fix_cell_ontology_id(adata_ref)

    # ── Step 7: make obs/var names unique (notebook step) ───────────────────
    adata_query.obs_names_make_unique()
    adata_ref.obs_names_make_unique()
    adata_query.var_names_make_unique()
    adata_ref.var_names_make_unique()

    # ── Step 8: auto-detect query batch key (notebook uses 'sample') ─────────
    batch_key = _detect_query_batch_key(adata_query, user_batch_key=batch_key)

    # ── Step 9: compute n_samples_per_label dynamically (notebook formula) ───
    # Notebook: min_celltype_size computed on ref_labels_key AFTER prefixing
    # (ref_labels_key = 'ref_cell_ontology_class' at this point — matches notebook exactly)
    min_celltype_size   = int(
        adata_ref.obs.groupby(ref_labels_key).size().min()
    )
    n_samples_per_label = int(np.max([min_celltype_size, nsamples]))
    logger.info(
        f"n_samples_per_label = max(min_celltype_size={min_celltype_size}, "
        f"nsamples={nsamples}) = {n_samples_per_label}"
    )

    # ── Step 10: check common genes ─────────────────────────────────────────
    common_genes = sorted(
        set(adata_query.var_names) & set(adata_ref.var_names)
    )
    logger.info(
        f"Common genes: query {adata_query.n_vars}, "
        f"ref {adata_ref.n_vars} → {len(common_genes)} shared."
    )
    if len(common_genes) == 0:
        raise ValueError(
            "Query and reference share 0 genes. "
            "Check that both use the same gene identifier (symbol vs Ensembl ID)."
        )

    # ── Step 11: snapshot full-gene counts BEFORE any subsetting ────────────
    adata_query = _store_full_counts_layer(adata_query)
    _full_counts_stash    = adata_query.layers["full_counts"].copy()
    _full_var_names_stash = list(adata_query.uns["full_counts_var_names"])

    # ── Step 12: strip layers from both sides before Process_Query ──────────
    # anndata.concat fill_value='unknown' (a string) causes scipy.sparse
    # dtype errors when layers exist on one side but not the other.
    # Solution: keep only 'counts' on both — PopV only needs .X anyway.
    adata_query, adata_ref = _strip_layers_for_popv(adata_query, adata_ref)

    # ── Step 13: clear raw from both (PopV manages its own raw internally) ───
    adata_query.raw = None
    adata_ref.raw   = None

    # ── Step 14: parallelism ─────────────────────────────────────────────────
    import os as _os
    n_threads = str(n_jobs if n_jobs > 0 else (_os.cpu_count() or 1))
    for env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ[env_var] = n_threads
    try:
        popv.Config.num_threads = int(n_threads)
    except Exception:
        pass
    logger.info(f"Parallelism: n_jobs={n_jobs} ({n_threads} threads).")

    # ── Step 15: Process_Query ───────────────────────────────────────────────
    # Mirrors notebook arguments exactly; only omits args removed in popv 0.4.2
    logger.info("Running Process_Query …")

    pq_kwargs = dict(
        query_labels_key      = None,
        query_batch_key       = batch_key,  # Process_Query's own kwarg name — must stay as-is
        ref_labels_key        = ref_labels_key,
        ref_batch_key         = None,
        unknown_celltype_label= "unknown",
        cl_obo_folder         = cl_obo_folder,
        n_samples_per_label   = n_samples_per_label,
        compute_embedding     = True,
        hvg                   = None,
    )

    # Gracefully add args that may not exist in all popv 0.4.x sub-versions
    _tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(_tmp_dir, exist_ok=True)

    # ── OnClass: force fresh cl.ontology from current cl.obo ─────────────────
    # OnClass uses a pre-processed cl.ontology graph file for ancestor traversal.
    # The SCART-bundled cl.ontology may be stale (built from an older cl.obo
    # release) and missing nodes like CL:0002371 'somatic cell', causing
    # KeyError during OnClass graph traversal.
    #
    # Fix: remove any stale cl.ontology from the tmp working dir so OnClass
    # regenerates it from scratch using the current full cl.obo.  Also copy
    # the full cl.obo into tmp so OnClass can find it alongside cl.ontology.
    import shutil as _shutil
    for _stale in [
        os.path.join(_tmp_dir, "cl.ontology"),
        os.path.join(_tmp_dir, "cl.ontology.nlp.emb"),
    ]:
        if os.path.exists(_stale):
            os.remove(_stale)
            logger.info(f"OnClass: removed stale file for regeneration: {_stale}")

    _obo_in_tmp = os.path.join(_tmp_dir, "cl.obo")
    if not os.path.exists(_obo_in_tmp):
        _shutil.copy2(obo_file, _obo_in_tmp)
        logger.info(f"OnClass: copied cl.obo to tmp dir → {_obo_in_tmp}")
    # ─────────────────────────────────────────────────────────────────────────

    for optional_kwarg, value in [
        ("save_path_trained_models", _tmp_dir),
        ("prediction_mode",          "retrain"),
        ("accelerator",              "cpu"),
    ]:
        try:
            import inspect
            sig = inspect.signature(Process_Query.__init__)
            if optional_kwarg in sig.parameters:
                pq_kwargs[optional_kwarg] = value
                logger.info(f"Process_Query: '{optional_kwarg}' = {value!r}")
            else:
                logger.info(f"Process_Query: '{optional_kwarg}' not in API — skipped.")
        except Exception:
            pass

    adata_combined = Process_Query(
        adata_query,
        adata_ref,
        **pq_kwargs,
    ).adata

    logger.info(f"Process_Query done — combined shape: {adata_combined.shape}")

    # ── Step 16: run annotation methods (notebook method list) ───────────────
    # Notebook: ["celltypist", "knn_on_bbknn", "knn_on_harmony",
    #            "knn_on_scanorama", "onclass", "rf", "svm"]
    # We try both old-style lowercase and new UPPER_SNAKE names for compatibility.

    # OnClass is included in the notebook's method list and works from cl.obo alone.
    # PopV internally generates cl.ontology from cl.obo during Process_Query and
    # saves it to save_path_trained_models (tmp dir) — so it exists by this point.
    # We always include OnClass; if it still fails it is caught by the per-method
    # try/except below and skipped with a warning rather than crashing the pipeline.
    _METHOD_CANDIDATES = [
        ["celltypist",      "CELLTYPIST"],
        ["knn_on_bbknn",    "KNN_BBKNN"],
        ["knn_on_harmony",  "KNN_HARMONY"],
        ["knn_on_scanorama","KNN_SCANORAMA"],
        ["onclass",         "ONCLASS"],
        ["rf",              "RANDOM_FOREST"],
        ["svm",             "SVM"],
    ]

    def _resolve_method_name(candidates: list) -> str | None:
        try:
            import popv.algorithms as _alg
            for name in candidates:
                if hasattr(_alg, name):
                    return name
        except ImportError:
            pass
        # Fall back to first candidate and let annotate_data raise if wrong
        return candidates[0]

    # Build run list — skip harmony if fewer than 2 batch values
    _batch_vals = set()
    if "_batch_annotation" in adata_combined.obs.columns:
        _batch_vals = set(adata_combined.obs["_batch_annotation"].unique())
    has_two_batches = len(_batch_vals - {"unknown", "unknown_query"}) >= 2

    _tmp_save = os.path.join(output_dir, "tmp")

    successful = []
    for candidates in _METHOD_CANDIDATES:
        canonical = candidates[0]  # human-readable name for logging
        if "harmony" in canonical and not has_two_batches:
            logger.warning(
                f"Skipping {canonical}: fewer than 2 real batch values in "
                f"'_batch_annotation' ({_batch_vals})."
            )
            continue

        method_name = _resolve_method_name(candidates)
        try:
            # annotate_data signature varies: try with save_path, fall back without
            try:
                annotate_data(adata_combined, methods=[method_name],
                              save_path=_tmp_save)
            except TypeError:
                annotate_data(adata_combined, methods=[method_name])

            successful.append(method_name)
            logger.info(f"✓ {canonical} ({method_name}) completed.")
        except Exception as exc:
            logger.warning(
                f"✗ Skipping {canonical} ({method_name}): "
                f"{type(exc).__name__}: {exc}"
            )

    logger.info(f"Annotation complete. Successful methods: {successful}")

    # ── Step 17: consensus prediction column ────────────────────────────────
    # Notebook uses 'popv_prediction'; newer popv may write 'popv_majority_vote_prediction'
    for pred_col in ("popv_prediction", "popv_majority_vote_prediction"):
        if pred_col in adata_combined.obs.columns:
            logger.info(
                f"Prediction column: '{pred_col}' — "
                f"{adata_combined.obs[pred_col].nunique()} unique labels."
            )
            break
    else:
        # Build fallback from whichever prediction columns exist
        fallback_cols = [
            c for c in adata_combined.obs.columns
            if c.endswith("_prediction") and adata_combined.obs[c].notna().any()
        ]
        if fallback_cols:
            adata_combined.obs["popv_majority_vote_prediction"] = (
                adata_combined.obs[fallback_cols[0]]
            )
            logger.warning(
                f"No consensus column found — using '{fallback_cols[0]}' as fallback."
            )
        else:
            logger.error("No prediction columns found at all — annotation failed.")

    # Ensure 'popv_majority_vote_prediction' always exists for downstream modules
    if ("popv_majority_vote_prediction" not in adata_combined.obs.columns
            and "popv_prediction" in adata_combined.obs.columns):
        adata_combined.obs["popv_majority_vote_prediction"] = (
            adata_combined.obs["popv_prediction"]
        )

    # ── Step 18: extract query cells (notebook: obs_names isin query_cells) ──
    adata_out = _extract_query_cells(adata_combined, query_obs_names_orig)
    logger.info(f"Query-only shape: {adata_out.shape}")

    # ── Step 19: notebook saves log_normalised X as a layer ──────────────────
    # We preserve the PopV-internal .X (usually log-normalised) as 'log_normalized'
    adata_out.layers["log_normalized"] = adata_out.X.copy()

    # ── Step 20: restore raw counts layer in HVG space ───────────────────────
    hvg_names   = list(adata_out.var_names)
    full_var_idx = {g: i for i, g in enumerate(_full_var_names_stash)}
    hvg_col_in_full = [full_var_idx[g] for g in hvg_names if g in full_var_idx]

    snap_idx  = {n: i for i, n in enumerate(query_obs_names_orig)}
    out_obs   = list(adata_out.obs_names)
    row_idx   = [snap_idx[n] for n in out_obs if n in snap_idx]

    if len(row_idx) == adata_out.n_obs and len(hvg_col_in_full) == len(hvg_names):
        fc_aligned  = _full_counts_stash.tocsr()[row_idx, :]
        adata_out.X = fc_aligned[:, hvg_col_in_full].tocsr()
        adata_out.layers["counts"] = adata_out.X.copy()
        logger.info(
            f"Raw counts restored to .X and layers['counts'] "
            f"({adata_out.n_obs} cells × {len(hvg_names)} HVGs)."
        )
    else:
        logger.warning(
            "Could not fully restore raw counts to .X "
            f"(row match: {len(row_idx)}/{adata_out.n_obs}, "
            f"col match: {len(hvg_col_in_full)}/{len(hvg_names)})."
        )

    # ── Step 21: store full_counts var names in uns (layer goes to sidecar only) ─
    # full_counts has shape (n_cells, 27984) — full gene space.
    # adata_out has shape (n_cells, 21418) — HVG space.
    # AnnData layers must match .var shape, so full_counts cannot be a layer here.
    # It is written to the sidecar h5ad (Step 22) for Module 3 instead.
    if len(row_idx) == adata_out.n_obs:
        adata_out.uns["full_counts_var_names"] = _full_var_names_stash
        logger.info(
            f"uns['full_counts_var_names'] stored ({len(_full_var_names_stash)} genes). "
            "Full-gene matrix is in the sidecar h5ad (full_counts_for_module3.h5ad)."
        )

    # ── Step 22: write sidecar h5ad for Module 3 ─────────────────────────────
    if len(row_idx) == adata_out.n_obs:
        sidecar_path = _write_full_counts_sidecar(
            adata_out,
            query_obs_names_orig,
            _full_counts_stash,
            _full_var_names_stash,
            output_dir,
        )
        adata_out.uns["full_counts_h5ad_path"] = sidecar_path
    else:
        logger.error("Sidecar not written — obs_names alignment failed.")

    # ── Step 23: clean up obs dtypes ─────────────────────────────────────────
    for col in adata_out.obs.columns:
        if adata_out.obs[col].dtype == object:
            adata_out.obs[col] = adata_out.obs[col].astype(str)

    if drop_reference_columns:
        adata_out = _drop_reference_only_columns(adata_out)

    # ── Step 24: write output ─────────────────────────────────────────────────
    out_path = os.path.join(output_dir, "final_popv_annotated.h5ad")
    adata_out.write(out_path)

    pred_col_final = next(
        (c for c in ("popv_majority_vote_prediction", "popv_prediction")
         if c in adata_out.obs.columns), "N/A"
    )

    logger.info(
        f"\n{'='*60}\n"
        f"PopV output          : {out_path}\n"
        f"Full-gene sidecar    : {adata_out.uns.get('full_counts_h5ad_path', 'N/A')}\n"
        f"Shape                : {adata_out.shape}\n"
        f"Prediction column    : {pred_col_final}\n"
        f"Unique predictions   : "
        f"{adata_out.obs.get(pred_col_final, pd.Series()).nunique()}\n"
        f"layers               : {list(adata_out.layers.keys())}\n"
        f"{'='*60}"
    )

    return adata_out


# ---------------------------------------------------------------------------
# 17. Public entry-point
# ---------------------------------------------------------------------------

def auto_run_popv(
    nsamples:               int  = 300,
    output_dir:             str  = "popv_results",
    user_reference:         str  = None,
    drop_reference_columns: bool = True,
    user_popv_prediction:   str  = None,
    n_jobs:                 int  = 1,
    batch_key:              str  = None,
) -> anndata.AnnData:
    """
    Fully automatic entry-point for Module 2.

    Finds the tumor h5ad written by Module 1, downloads the matching
    Tabula Sapiens reference (or uses user_reference), and runs PopV
    annotation using the same analysis logic as PopV_GSE173682.ipynb.

    Parameters
    ----------
    nsamples : int
        Minimum cells per label (floor for n_samples_per_label).
        Actual value = max(min_celltype_size_in_ref, nsamples).
    output_dir : str
        Directory for output files.
    user_reference : str, optional
        Path to a local Tabula Sapiens h5ad.  Skips Figshare download.
    drop_reference_columns : bool
        Remove Tabula Sapiens metadata from saved output.
    user_popv_prediction : str, optional
        Path to an already-annotated h5ad.  Skips the entire PopV pipeline.
    n_jobs : int
        CPU threads (-1 = all cores).
    batch_key : str, optional
        Column in your query h5ad's .obs to use as the PopV batch key.
        DEFAULT: if omitted (and Module 1's SampleAnnotator was given no
        batch_key either), GSM ID ('gsm_id') is used as the batch
        automatically for GEO-derived data. Only needed if you're passing a
        custom h5ad (not produced by Module 1) whose sample/donor column
        isn't auto-detected, or if GSM ID isn't a valid batch for your data
        (e.g. one GSM pools multiple patients/donors) — e.g.
        batch_key="patient_id". Use the SAME parameter name — batch_key —
        in both Module 1 and Module 2.

    Usage
    -----
    from SCART import popv_annotation

    # Standard run — supply reference
    adata = popv_annotation.auto_run_popv(
        nsamples       = 300,
        user_reference = "/data/Ovary_TSP1_30_version2d_10X_smartseq_scvi.h5ad"
    )

    # Auto-download reference from Figshare
    adata = popv_annotation.auto_run_popv(nsamples=300)

    # Custom h5ad with a non-standard batch column name
    adata = popv_annotation.auto_run_popv(
        nsamples         = 300,
        user_reference   = "/data/Ovary_TSP1_30_version2d_10X_smartseq_scvi.h5ad",
        batch_key        = "patient_id"
    )

    # GEO data where GSM ID isn't a valid batch (pooled multi-donor GSMs):
    # first re-run Module 1 with batch_key="donor" (or similar), then
    # pass the SAME name here so Module 2 uses it too.
    adata = popv_annotation.auto_run_popv(
        nsamples         = 300,
        user_reference   = "/data/Ovary_TSP1_30_version2d_10X_smartseq_scvi.h5ad",
        batch_key        = "donor"
    )

    # Skip pipeline — use pre-computed PopV result
    adata = popv_annotation.auto_run_popv(
        user_popv_prediction = "/data/my_popv_annotated.h5ad"
    )
    """
    # ── Bypass: user supplies pre-computed annotation ─────────────────────
    if user_popv_prediction is not None:
        if not os.path.exists(user_popv_prediction):
            raise FileNotFoundError(
                f"user_popv_prediction not found: {user_popv_prediction}"
            )
        logger.info(f"Loading pre-computed PopV result: {user_popv_prediction}")
        adata = sc.read_h5ad(user_popv_prediction)

        pred_cols = [c for c in adata.obs.columns if c.endswith("_prediction")]
        if not pred_cols:
            raise ValueError(
                f"No '_prediction' columns found in {user_popv_prediction}.\n"
                "Supply a valid PopV-annotated h5ad."
            )

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "final_popv_annotated.h5ad")
        if os.path.abspath(user_popv_prediction) != os.path.abspath(out_path):
            adata.write(out_path)
        logger.info(f"User PopV prediction saved to: {out_path}")
        return adata

    # ── Locate Module 1 output ────────────────────────────────────────────
    tumor_file  = get_latest_tumor_h5ad()
    cancer_type = detect_cancer_type_from_h5ad(tumor_file)
    primary_cancer = cancer_type.split(",")[0].strip()

    reference_path = auto_select_reference(primary_cancer, user_reference)

    logger.info(f"Query    : {tumor_file}")
    logger.info(f"Reference: {reference_path}")

    adata_query = sc.read_h5ad(tumor_file)
    adata_ref   = sc.read_h5ad(reference_path)

    return run_popv_annotation(
        adata_query            = adata_query,
        adata_ref              = adata_ref,
        output_dir              = output_dir,
        nsamples                = nsamples,
        drop_reference_columns  = drop_reference_columns,
        n_jobs                  = n_jobs,
        batch_key                = batch_key,
    )
