"""
preprocessing.py
Module 3 — Preprocessing, malignancy detection, and surfaceome DEG

═══════════════════════════════════════════════════════════════════════════════
CORRECT BIOLOGICAL PIPELINE DESIGN
═══════════════════════════════════════════════════════════════════════════════

Step 1  Load the full PopV-annotated h5ad (all cell types, e.g. 15202 cells).
Step 2  Extract epithelial cells only → apply QC filters if set in Module 1.
Step 3  Run scMalignantFinder + SCEVAN on epithelial cells.
Step 4  Keep ONLY malignant epithelial cells.
Step 5  From adata_full, take all NON-EPITHELIAL cells as the "rest" group.
Step 6  DEG: malignant epithelial vs non-epithelial rest (surfaceome genes).
Step 7  Binarise ONLY the malignant epithelial AnnData, store DEG, save.

═══════════════════════════════════════════════════════════════════════════════
RAW COUNT HANDLING
═══════════════════════════════════════════════════════════════════════════════

CORRECTION: popv_annotation.py has no input_type parameter and no log1p
mode — it only accepts raw integer counts. _set_raw_counts_in_X() routes
counts into .X (layers['counts'] -> 'raw_counts' -> 'scvi_counts', reference
side also checks 'decontXcounts' first), and _validate_raw_counts() then
RAISES if the result looks log-normalised (any negative values, or mean
between 0 and 2.0). So Module 2 always runs on raw counts, always snapshots
them to layers['full_counts'] before HVG subsetting (_store_full_counts_layer),
and always writes layers['counts'] (HVG-space) plus the full-gene sidecar
full_counts_for_module3.h5ad in its output. There is no partial/log1p path
to handle here.

_get_raw_counts_from_adata() below still uses a priority-ordered fallback
(layers['counts'] -> 'raw_counts' -> 'scvi_counts' -> adata.raw.X -> adata.X)
purely as defensive coding — e.g. if PopV is bypassed via
auto_run_popv(user_popv_prediction=...) with a hand-supplied h5ad that
doesn't follow the same layer convention. Under the normal Module 1 -> 2 -> 3
flow, layers['counts'] is always present and always raw.

═══════════════════════════════════════════════════════════════════════════════
SCEVAN REFERENCE — HOW IT WORKS
═══════════════════════════════════════════════════════════════════════════════

SCEVAN requires integer raw counts from:
  (a) A query AnnData  — the epithelial cells to classify (tumour vs normal)
  (b) A reference AnnData — KNOWN NORMAL epithelial cells as CNV baseline

The reference is loaded from reference_h5ad (Tabula Sapiens or user-supplied).

Epithelial cells are extracted from the reference using ONE of three methods,
controlled by scevan_ref_cell_col and scevan_ref_epithelial_values:

  Method A — Default (Tabula Sapiens, cell_ontology_class column):
    scevan_ref_cell_col        = "cell_ontology_class"   (default)
    scevan_ref_epithelial_values = None                  (default)
    → substring match: any value containing "epithelial cell" (case-insensitive)
    → captures: "ovarian surface epithelial cell", "glandular epithelial cell", etc.

  Method B — User supplies exact label list:
    scevan_ref_cell_col        = "cell_type"             (your column name)
    scevan_ref_epithelial_values = ["Normal Epithelial cells"]
    → exact match: only cells whose column value is in the provided list

  Method C — User supplies a pre-filtered reference (already epithelial only):
    scevan_ref_cell_col        = None
    scevan_ref_epithelial_values = None
    → uses the entire reference h5ad as-is (no filtering)

After selection, the reference is subsampled to scevan_ref_max_cells (default=500).
Setting scevan_ref_max_cells=None uses ALL available reference epithelial cells.

Gene alignment follows the notebook approach:
  1. Common genes between query and reference are found FIRST.
  2. Both matrices are subset to common genes BEFORE building the count CSV.
  3. The combined matrix (query + REF_ prefixed reference) is passed to SCEVAN.
  4. Normal barcodes CSV lists only the REF_ prefixed cells.

═══════════════════════════════════════════════════════════════════════════════
KEY FIXES
═══════════════════════════════════════════════════════════════════════════════

FIX 1   Full-gene route priority for scMalignantFinder.
FIX 2   SCEVAN reference selection now mirrors Input_SCEVAN.ipynb exactly:
        - common genes found FIRST (query ∩ reference)
        - both subsetted BEFORE matrix extraction
        - ref_max_cells default raised to 500 (notebook uses all 476)
        - user can supply scevan_ref_cell_col + scevan_ref_epithelial_values
          for custom references, or pass a pre-filtered reference directly.
FIX 3   DEG uses pvals_adj (BH-adjusted).
FIX 5   _get_raw_counts_from_adata prefers layers['counts'] → raw → .X.
FIX 6   scMalignantFinder predictions aligned by obs_names.
FIX 8   SCEVAN results saved in full detail in adata.uns['scevan_results'].
FIX 9   Correct DEG: malignant epithelial vs NON-EPITHELIAL cells.
FIX 10  SCEVAN runs via fresh Rscript subprocess.
        Rscript resolved via sys.executable bin dir FIRST — the only
        reliable signal inside a Jupyter kernel. CONDA_PREFIX and PATH
        are fallbacks only.
FIX 11  Epithelial cell selection uses str.contains("epithelial cell")
        (case-insensitive) instead of str.endswith("epithelial cell"),
        so "ovarian surface epithelial cell", "glandular epithelial cell",
        "lung epithelial cell", etc. are all captured correctly.
QC FIX  QC thresholds read from adata.uns['qc_params'] (set by Module 1).
        If absent, QC is SKIPPED ENTIRELY.

═══════════════════════════════════════════════════════════════════════════════
CLAUDE EDIT — Step 8b + 3-way final gene intersection
═══════════════════════════════════════════════════════════════════════════════

FIX 12  New Step 8b (between the existing Step 8 DEG and Step 9 gene
        subsetting): calls run_cancer_composition_step() (defined further
        down in this same file) on a full-gene-space snapshot of the
        malignant cells (captured right after Step 5, before Step 7's
        surfaceome subsetting) against reference_h5ad — the SAME single
        Tabula Sapiens h5ad the user already supplies for SCEVAN. There is
        deliberately no separate healthy-reference argument: reference_h5ad
        is the one h5ad this whole module takes from the user, reused
        wherever a healthy reference is needed. This Harmony-integrates
        tumor vs. healthy, saves before/after UMAP QC plots, and computes a
        per-gene Cancer Composition Score (Tumor_Z - Healthy_Z on % cells
        expressing).
FIX 13  Step 9's gene subsetting is now a 3-way intersection between THREE
        INDEPENDENT, PARALLEL criteria (none gates the others beforehand):
            {DEG-passing genes (Step 8, log2FC / pvals_adj, full gene space)}
          ∩ {Cancer_Composition >= cc_score_threshold (Step 8b, full gene space)}
          ∩ {surfaceome genes (GESP list, Step 7 — list membership only)}
        Step 7 no longer pre-restricts adata_mal/adata_rest to surfaceome
        genes before DEG runs — Step 8 and Step 8b both operate on the full
        common gene space; surfaceome is intersected in only at the end,
        on equal footing with the other two.
        adata.uns['final_gene_selection'] records each set's size and the
        final intersection for traceability. If run_cancer_composition=False
        or no healthy reference is available, this falls back to a 2-way
        DEG ∩ surfaceome intersection.
"""

import os
import sys
import glob
import shutil
import logging
import tempfile
import subprocess
import resource

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Auto-detect paths
# ═══════════════════════════════════════════════════════════════════════════

def _find_scart_resource(relative_path):
    try:
        import SCART as _scart
        candidate = os.path.join(os.path.dirname(_scart.__file__), relative_path)
        if os.path.exists(candidate):
            return candidate
    except ImportError:
        pass
    return None


def _auto_scmalignant_model():
    path = _find_scart_resource("external/scMalignantFinder/model")
    if path is None:
        raise FileNotFoundError(
            "Could not auto-detect scMalignantFinder model.\n"
            "Pass scmalignant_model_dir= explicitly.\n"
            "Expected: <scart_root>/external/scMalignantFinder/model/"
        )
    return path


def _auto_surfaceome_path():
    for candidate in (
        "GESP_surfaceome_gene.csv",
        "GESP/GESP_surfaceome_gene.csv",
        "data/GESP_surfaceome_gene.csv",
        "resources/GESP_surfaceome_gene.csv",
    ):
        path = _find_scart_resource(candidate)
        if path is not None:
            return path
    raise FileNotFoundError(
        "Could not auto-detect GESP surfaceome CSV.\n"
        "Expected at: <scart_root>/GESP_surfaceome_gene.csv\n"
        "Pass surfaceome_path= explicitly."
    )


def _auto_tumor_h5ad():
    """Auto-detect Module 1 tumor h5ad from cwd and GSE_data/."""
    patterns    = ["*_tumor.h5ad", "combined_tumor.h5ad", "input_tumor.h5ad"]
    search_dirs = [os.getcwd(), os.path.join(os.getcwd(), "GSE_data")]
    files = []
    for d in search_dirs:
        for pat in patterns:
            files.extend(glob.glob(os.path.join(d, pat)))
    files = list(set(files))
    if not files:
        return None
    found = max(files, key=os.path.getctime)
    logger.info(f"Auto-detected Module 1 tumor h5ad: {found}")
    return found


def _auto_popv_h5ad():
    """Auto-detect Module 2 PopV output h5ad."""
    for c in [
        os.path.join("popv_results", "final_popv_annotated.h5ad"),
        "final_popv_annotated.h5ad",
    ]:
        if os.path.exists(c):
            return c
    return None


# ═══════════════════════════════════════════════════════════════════════════
# QC parameter reader
# ═══════════════════════════════════════════════════════════════════════════

def _read_qc_params(adata):
    qc = adata.uns.get("qc_params", None)
    if qc is None:
        return (None, None, False,
                "SKIPPED — 'qc_params' not found in adata.uns.")
    min_genes = qc.get("min_genes", None)
    max_mt    = qc.get("max_mt",    None)
    if min_genes is not None:
        min_genes = int(min_genes)
    if max_mt is not None:
        max_mt = float(max_mt)
    qc_active = (min_genes is not None) or (max_mt is not None)
    parts = []
    if min_genes is not None:
        parts.append(f"min_genes={min_genes}")
    if max_mt is not None:
        parts.append(f"max_mt={max_mt}")
    source = (
        "adata.uns['qc_params'] — " + ", ".join(parts)
        if qc_active
        else "'qc_params' present but both values None — QC SKIPPED."
    )
    return min_genes, max_mt, qc_active, source


# ═══════════════════════════════════════════════════════════════════════════
# FIX 5 — raw count extractor
# ═══════════════════════════════════════════════════════════════════════════

def _get_raw_counts_from_adata(adata, context=""):
    """
    Return a dense float32 (cells × genes) raw count matrix from adata.

    Priority:
      1. layers['counts']      — written by popv_annotation.py (raw mode)
      2. layers['raw_counts']  — alternate name
      3. layers['scvi_counts'] — scVI internal counts
      4. adata.raw.X           — legacy raw slot
      5. adata.X               — last resort (WARNING: may be log1p)
    """
    tag = f"[{context}] " if context else ""

    for lyr in ("counts", "raw_counts", "scvi_counts"):
        if lyr in adata.layers:
            logger.info(f"{tag}Raw counts source: layers['{lyr}']")
            X = adata.layers[lyr]
            if sp.issparse(X):
                X = X.toarray()
            return np.array(X, dtype=np.float32)

    if adata.raw is not None:
        logger.info(f"{tag}Raw counts source: adata.raw.X ({adata.raw.n_vars} genes)")
        X = adata.raw.X
        if sp.issparse(X):
            X = X.toarray()
        return np.array(X, dtype=np.float32)

    logger.warning(
        f"{tag}No raw counts layer found — using adata.X as-is. "
        "Under the normal Module 1->2->3 flow this shouldn't happen "
        "(PopV always writes layers['counts']); this object may have "
        "bypassed PopV or come from a hand-supplied h5ad. "
        "QC metrics and CNA tools require true raw counts. "
        "Consider providing tumor_h5ad= to load raw counts from Module 1."
    )
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return np.array(X, dtype=np.float32)


def _get_raw_matrix(adata):
    """
    Return dense float64 (cells × genes) raw count matrix.
    Kept for backward-compat with SCEVAN helpers that use float64.
    Wraps _get_raw_counts_from_adata.
    """
    return _get_raw_counts_from_adata(adata).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# FIX 1 — full-gene AnnData for scMalignantFinder
# ═══════════════════════════════════════════════════════════════════════════

def _build_fullgene_adata_for_scm(adata, feature_tsv, tumor_h5ad_path=None):
    model_features = set(
        pd.read_csv(feature_tsv, sep="\t", header=None)[0].tolist()
    )
    n_model = len(model_features)

    def _pct(names):
        return len(set(names) & model_features) / n_model * 100

    def _make_adata(X, obs, var_ref):
        if sp.issparse(X):
            X = X.toarray()
        X   = np.array(X, dtype=np.float32)
        var = var_ref if isinstance(var_ref, pd.DataFrame) \
              else pd.DataFrame(index=list(var_ref))
        af  = sc.AnnData(X=X, obs=obs.copy(), var=var)
        sc.pp.normalize_total(af, target_sum=1e4)
        sc.pp.log1p(af)
        af.X = sp.csr_matrix(af.X)
        return af

    # Route A-layer: use layers['counts'] from popv output (raw mode)
    for lyr in ("counts", "raw_counts"):
        if lyr in adata.layers:
            ov = _pct(adata.var_names)
            logger.info(
                f"Route A-layer (layers['{lyr}']): {adata.n_vars} genes, {ov:.1f}% overlap"
            )
            if ov >= 50:
                af = _make_adata(adata.layers[lyr], adata.obs, adata.var)
                logger.info(f"scMalignantFinder → Route A-layer ({af.n_vars} genes).")
                return af
            logger.warning(
                f"Route A-layer overlap {ov:.1f}% < 50% — trying full-gene routes."
            )
            break

    # Route A-new: full_counts layer with var names in uns (36k genes)
    if "full_counts" in adata.layers:
        vn = adata.uns.get("full_counts_var_names")
        if vn is not None and len(vn) == adata.layers["full_counts"].shape[1]:
            ov = _pct(vn)
            logger.info(f"Route A-new: {len(vn)} genes, {ov:.1f}% model overlap")
            if ov >= 50:
                af = _make_adata(adata.layers["full_counts"], adata.obs, vn)
                logger.info(f"scMalignantFinder → Route A-new ({af.n_vars} genes).")
                return af
            logger.warning(f"Route A-new overlap {ov:.1f}% < 50% — trying A-rescue.")
        else:
            logger.warning("full_counts gene names mismatch — trying A-rescue.")

    # Route A-rescue: read from Module 1 tumor h5ad (always has full gene space)
    rescue = tumor_h5ad_path or _auto_tumor_h5ad()
    if rescue and os.path.exists(rescue):
        logger.info(f"Route A-rescue: {rescue}")
        try:
            m1     = sc.read_h5ad(rescue)
            shared = sorted(set(adata.obs_names) & set(m1.obs_names))
            if m1.raw is not None and len(shared) > 0:
                ov = _pct(m1.raw.var_names)
                logger.info(f"Route A-rescue (m1.raw): {m1.raw.n_vars} genes, {ov:.1f}%")
                if ov >= 50:
                    order = [c for c in adata.obs_names if c in set(shared)]
                    sub   = m1[order]
                    af    = _make_adata(sub.raw.X, adata.obs.loc[order], sub.raw.var)
                    logger.info(f"scMalignantFinder → Route A-rescue ({af.n_vars} genes).")
                    return af
            for lyr in ("counts", "raw_counts", "scvi_counts"):
                if lyr in m1.layers and len(shared) > 0:
                    ov = _pct(m1.var_names)
                    logger.info(
                        f"Route A-rescue (m1.layers['{lyr}']): {m1.n_vars} genes, {ov:.1f}%"
                    )
                    if ov >= 50:
                        order = [c for c in adata.obs_names if c in set(shared)]
                        sub   = m1[order]
                        af    = _make_adata(sub.layers[lyr], adata.obs.loc[order], sub.var)
                        logger.info(f"scMalignantFinder → Route A-rescue (layers['{lyr}']).")
                        return af
                    break
        except Exception as exc:
            logger.warning(f"Route A-rescue failed: {exc}")

    # Route A-old
    if adata.raw is not None:
        ov = _pct(adata.raw.var_names)
        logger.info(f"Route A-old (adata.raw): {adata.raw.n_vars} genes, {ov:.1f}%")
        if ov >= 50:
            af = _make_adata(adata.raw.X, adata.obs, adata.raw.var)
            logger.info("scMalignantFinder → Route A-old.")
            return af
        logger.warning(f"Route A-old overlap {ov:.1f}% < 50%.")

    # Route B
    if "full_var_names" in adata.uns:
        fv  = list(adata.uns["full_var_names"])
        ov  = _pct(fv)
        logger.info(f"Route B (uns): {len(fv)} genes, {ov:.1f}%")
        for lyr in ("scvi_counts", "raw_counts", "counts"):
            if lyr in adata.layers:
                X = adata.layers[lyr]
                if sp.issparse(X):
                    X = X.toarray()
                if X.shape[1] == len(fv) and ov >= 50:
                    af = _make_adata(X, adata.obs, fv)
                    logger.info(f"scMalignantFinder → Route B (layers['{lyr}']).")
                    return af

    # Route C — last resort (HVGs from .X — already log-normalised)
    ov_hvg = _pct(adata.var_names)
    logger.warning(
        f"All routes failed. Falling back to {adata.n_vars} HVGs ({ov_hvg:.1f}% overlap).\n"
        "Place GSE*_tumor.h5ad in cwd or re-run Module 2 with FIX 8."
    )
    return adata.copy()


# ═══════════════════════════════════════════════════════════════════════════
# FIX 10 — Rscript resolver (sys.executable first)
# ═══════════════════════════════════════════════════════════════════════════

def _find_rscript():
    py_bin = os.path.dirname(os.path.abspath(sys.executable))
    cand   = os.path.join(py_bin, "Rscript")
    if os.path.isfile(cand):
        logger.info(f"Rscript found via sys.executable dir: {cand}")
        return cand

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        cand = os.path.join(conda_prefix, "bin", "Rscript")
        if os.path.isfile(cand):
            logger.warning(
                f"Rscript found via CONDA_PREFIX (may be wrong env in Jupyter): {cand}\n"
                f"  CONDA_PREFIX  = {conda_prefix}\n"
                f"  sys.executable= {sys.executable}"
            )
            return cand

    cand = shutil.which("Rscript")
    if cand:
        logger.warning(f"Rscript found via PATH (may be wrong env): {cand}")
        return cand

    return None


def _get_r_home(rscript_bin):
    try:
        r_home = subprocess.check_output(
            [rscript_bin, "--vanilla", "-e", "cat(R.home())"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return r_home
    except Exception as exc:
        logger.warning(f"Could not determine R_HOME from {rscript_bin}: {exc}")
        return None


def _build_r_env(r_home):
    env = os.environ.copy()
    if r_home:
        env["R_HOME"] = r_home
        r_lib = os.path.join(r_home, "lib")
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            r_lib + (os.pathsep + existing_ld if existing_ld else "")
        )
        logger.info(f"Subprocess R_HOME → {r_home}")
    return env


# ═══════════════════════════════════════════════════════════════════════════
# SCEVAN reference preparation (mirrors Input_SCEVAN.ipynb exactly)
# ═══════════════════════════════════════════════════════════════════════════

def _prepare_scevan_reference(
    adata_ref_full,
    ref_cell_col,
    ref_epithelial_values,
    ref_max_cells,
):
    if ref_cell_col is None:
        n_before = adata_ref_full.n_obs
        logger.info(
            f"SCEVAN reference: scevan_ref_cell_col=None — "
            f"using entire reference as-is ({n_before} cells)."
        )
        adata_ref_ep = adata_ref_full.copy()

    elif ref_cell_col not in adata_ref_full.obs.columns:
        logger.warning(
            f"SCEVAN reference: column '{ref_cell_col}' not found in reference obs.\n"
            f"  Available columns: {list(adata_ref_full.obs.columns)}\n"
            f"  Falling back to using full reference ({adata_ref_full.n_obs} cells)."
        )
        n_before = adata_ref_full.n_obs
        adata_ref_ep = adata_ref_full.copy()

    elif ref_epithelial_values is not None:
        values_set = set(ref_epithelial_values)
        ep_mask = adata_ref_full.obs[ref_cell_col].isin(values_set)
        n_before = ep_mask.sum()
        logger.info(
            f"SCEVAN reference: exact match on '{ref_cell_col}' "
            f"for values {ref_epithelial_values} → {n_before} cells."
        )
        if n_before == 0:
            unique_vals = adata_ref_full.obs[ref_cell_col].value_counts().to_string()
            raise ValueError(
                f"SCEVAN reference: no cells matched for column='{ref_cell_col}' "
                f"with values={ref_epithelial_values}.\n"
                f"Unique values in '{ref_cell_col}':\n{unique_vals}\n\n"
                f"Fix: update scevan_ref_epithelial_values= to match your labels,\n"
                f"or set scevan_ref_cell_col=None to use the full reference."
            )
        adata_ref_ep = adata_ref_full[ep_mask].copy()

    else:
        ep_mask = adata_ref_full.obs[ref_cell_col].astype(str).str.contains(
            "epithelial cell", case=False, na=False
        )
        n_before = ep_mask.sum()
        logger.info(
            f"SCEVAN reference: substring match (contains 'epithelial cell') "
            f"on '{ref_cell_col}' → {n_before} cells."
        )
        if n_before == 0:
            unique_vals = adata_ref_full.obs[ref_cell_col].value_counts().to_string()
            raise ValueError(
                f"SCEVAN reference: no cells found containing 'epithelial cell' "
                f"in column '{ref_cell_col}'.\n"
                f"Unique values:\n{unique_vals}\n\n"
                f"Fix: supply scevan_ref_epithelial_values=['YourLabelHere'] and "
                f"scevan_ref_cell_col='{ref_cell_col}',\n"
                f"or set scevan_ref_cell_col=None to use the full reference."
            )
        adata_ref_ep = adata_ref_full[ep_mask].copy()

    if ref_cell_col is not None and ref_cell_col in adata_ref_full.obs.columns:
        matched_counts = adata_ref_ep.obs[ref_cell_col].value_counts()
        print(f"  Reference epithelial cells selected ({n_before} total):")
        for lbl, cnt in matched_counts.items():
            print(f"    {lbl}: {cnt}")

    if ref_max_cells is not None and adata_ref_ep.n_obs > ref_max_cells:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(adata_ref_ep.n_obs, size=ref_max_cells, replace=False)
        adata_ref_ep = adata_ref_ep[np.sort(idx)].copy()
        print(f"  Reference subsampled to {ref_max_cells} cells (from {n_before}).")
    else:
        print(f"  Using all {adata_ref_ep.n_obs} reference epithelial cells (no subsampling).")

    return adata_ref_ep, n_before


# ═══════════════════════════════════════════════════════════════════════════
# SCEVAN subprocess runner — notebook-aligned
# ═══════════════════════════════════════════════════════════════════════════

def _run_scevan(
    adata_query,
    adata_ref,
    ref_cell_col="cell_ontology_class",
    ref_epithelial_values=None,
    ref_max_cells=500,
    sample_name="SCEVAN_run",
    organism="human",
    par_cores=1,
    subclones=False,
    batch_size=3000,
    save_dir=None,
):
    q_barcodes   = np.array(adata_query.obs_names)
    empty_result = pd.DataFrame({
        "barcode"           : list(q_barcodes),
        "scevan_prediction" : "not.defined",
    })

    rscript_bin = _find_rscript()
    if rscript_bin is None:
        logger.error("Rscript not found. SCEVAN skipped.")
        return empty_result

    r_home  = _get_r_home(rscript_bin)
    sub_env = _build_r_env(r_home)

    try:
        check = subprocess.run(
            [rscript_bin, "--vanilla", "-e",
             "if (!requireNamespace('SCEVAN', quietly=TRUE)) "
             "{ cat('NOT_INSTALLED'); quit(status=1) } else { cat('OK') }"],
            capture_output=True, text=True, env=sub_env,
        )
        if "NOT_INSTALLED" in check.stdout or check.returncode != 0:
            raise ImportError(
                f"R package 'SCEVAN' is not installed in the R used by:\n"
                f"  {rscript_bin}\n"
                f"R home: {r_home}\n"
                f"Install it with:\n"
                f"  {rscript_bin} -e \"devtools::install_github('miccec/yaGST')\"\n"
                f"  {rscript_bin} -e \"devtools::install_github('AntonioDeFalco/SCEVAN')\""
            )
        logger.info(f"SCEVAN verified OK in {rscript_bin}")
    except ImportError:
        raise
    except Exception as exc:
        logger.error(f"SCEVAN verification failed: {exc}")
        return empty_result

    print("  Preparing SCEVAN reference cells...")
    try:
        adata_ref_ep, n_ref_before = _prepare_scevan_reference(
            adata_ref_full       = adata_ref,
            ref_cell_col         = ref_cell_col,
            ref_epithelial_values= ref_epithelial_values,
            ref_max_cells        = ref_max_cells,
        )
    except ValueError as exc:
        logger.error(f"SCEVAN reference preparation failed:\n{exc}")
        return empty_result

    if adata_ref_ep.n_obs == 0:
        logger.warning("SCEVAN: no reference cells after filtering. Skipping.")
        return empty_result

    q_gene_index   = adata_query.var_names
    ref_gene_index = adata_ref_ep.var_names

    common_gene_index = q_gene_index.intersection(ref_gene_index)
    n_common = len(common_gene_index)
    logger.info(
        f"SCEVAN gene overlap: {adata_query.n_vars} query genes ∩ "
        f"{adata_ref_ep.n_vars} ref genes = {n_common} common genes."
    )
    print(
        f"  Gene overlap: {adata_query.n_vars} query ∩ "
        f"{adata_ref_ep.n_vars} reference = {n_common} common genes."
    )

    if n_common < 200:
        raise ValueError(
            f"Only {n_common} common genes between query and reference. "
            "Need >= 200. Both datasets must use HGNC gene symbols.\n"
            f"  Query gene examples:     {list(q_gene_index[:5])}\n"
            f"  Reference gene examples: {list(ref_gene_index[:5])}"
        )

    adata_query_sub = adata_query[:, common_gene_index].copy()
    adata_ref_sub   = adata_ref_ep[:, common_gene_index].copy()

    logger.info("SCEVAN: extracting raw counts from query and reference (after gene alignment)...")
    mat_query = _get_raw_counts_from_adata(adata_query_sub, "SCEVAN-query")
    mat_ref   = _get_raw_counts_from_adata(adata_ref_sub,   "SCEVAN-ref")

    q_barcodes_arr = np.array(adata_query_sub.obs_names)
    r_barcodes     = np.array(["REF_" + b for b in adata_ref_sub.obs_names])
    common_genes   = np.array(common_gene_index)

    mat_combined = np.vstack([mat_query, mat_ref]).T
    all_barcodes = np.concatenate([q_barcodes_arr, r_barcodes])

    print(
        f"  Combined matrix: {mat_combined.shape[0]} genes × "
        f"{mat_combined.shape[1]} cells "
        f"({len(q_barcodes_arr)} query + {len(r_barcodes)} ref)"
    )

    _tmpdir_created = False
    if save_dir is None:
        save_dir = tempfile.mkdtemp(prefix="scart_scevan_")
        _tmpdir_created = True
    else:
        os.makedirs(save_dir, exist_ok=True)

    try:
        counts_csv    = os.path.join(save_dir, "scevan_counts.csv")
        norm_csv      = os.path.join(save_dir, "normal_barcodes.csv")
        driver_r      = os.path.join(save_dir, "run_scevan.R")
        results_csv   = os.path.join(save_dir, "scevan_full_results.csv")
        malignant_csv = os.path.join(save_dir, "scevan_malignant_cells.csv")

        logger.info(f"SCEVAN: writing count matrix ({mat_combined.shape}) ...")
        count_df = pd.DataFrame(
            mat_combined,
            index   = common_genes,
            columns = all_barcodes,
        )
        count_df.to_csv(counts_csv)
        logger.info(f"SCEVAN count matrix written: {counts_csv}")

        pd.Series(r_barcodes.tolist()).to_csv(norm_csv, index=False, header=False)
        logger.info(f"SCEVAN normal barcodes written: {norm_csv} ({len(r_barcodes)} cells)")

        subclones_r  = "TRUE" if subclones else "FALSE"
        fixed_norm_r = "TRUE"

        with open(driver_r, "w") as f:
            f.write(f"""\
suppressPackageStartupMessages({{
  library(SCEVAN)
  library(Matrix)
}})

original_fn <- get("classifyTumorCells", envir = asNamespace("SCEVAN"))
modified_fn <- original_fn
body_text   <- deparse(body(original_fn))
body_text   <- gsub("parallel::parLapply\\\\(cl,", "lapply(", body_text)
body_text   <- gsub("parLapply\\\\(cl,", "lapply(", body_text)
new_body    <- parse(text = paste(body_text, collapse = "\\n"))
body(modified_fn) <- as.call(c(as.name("{{"), new_body))
environment(modified_fn) <- asNamespace("SCEVAN")
assignInNamespace("classifyTumorCells", modified_fn, "SCEVAN")

cat("Loading count matrix...\\n")
count_mat  <- read.csv("{counts_csv}", row.names = 1, check.names = FALSE)
count_mat  <- as.matrix(count_mat)
cat("Matrix dims:", nrow(count_mat), "genes x", ncol(count_mat), "cells\\n")

normal_cells <- readLines("{norm_csv}")
cat("Normal reference cells:", length(normal_cells), "\\n")

query_cells <- setdiff(colnames(count_mat), normal_cells)
batch_size  <- {batch_size}
batches     <- split(query_cells, ceiling(seq_along(query_cells) / batch_size))
all_results <- list()

cat("Total query cells  :", length(query_cells), "\\n")
cat("Batch size         :", batch_size, "\\n")
cat("Total batches      :", length(batches), "\\n")

for (i in seq_along(batches)) {{
  save_file <- file.path("{save_dir}", paste0("scevan_batch_", i, "_results.csv"))

  if (file.exists(save_file)) {{
    cat("Batch", i, "already done — loading from file\\n")
    all_results[[i]] <- read.csv(save_file, row.names = 1, check.names = FALSE)
    next
  }}

  cat("\\n--- Batch", i, "of", length(batches),
      "| Started:", format(Sys.time()), "---\\n")

  batch_cells     <- c(normal_cells, batches[[i]])
  count_mat_batch <- count_mat[, batch_cells]

  tryCatch({{
    res <- pipelineCNA(
      count_mtx          = count_mat_batch,
      norm_cell          = normal_cells,
      sample             = paste0("{sample_name}_batch_", i),
      par_cores          = {par_cores},
      SUBCLONES          = {subclones_r},
      FIXED_NORMAL_CELLS = {fixed_norm_r},
      organism           = "{organism}"
    )

    res_query <- res[rownames(res) %in% batches[[i]], , drop = FALSE]

    write.csv(res_query, save_file)
    all_results[[i]] <- res_query

    cat("Batch", i, "done |",
        nrow(res_query), "cells |",
        sum(res_query$class == "tumor"), "tumor |",
        "Finished:", format(Sys.time()), "\\n")

  }}, error = function(e) {{
    cat("Batch", i, "failed:", conditionMessage(e), "\\n")
  }})

  gc()
}}

cat("\\nCombining all batches...\\n")
final_results <- do.call(rbind, Filter(Negate(is.null), all_results))

cat("\\n=== Final Results ===\\n")
print(table(final_results$class))
cat("Total classified:", nrow(final_results), "\\n")

write.csv(final_results, "{results_csv}")
write.csv(
  data.frame(cell = rownames(final_results[final_results$class == "tumor", ])),
  "{malignant_csv}",
  row.names = FALSE
)
cat("Done!\\n")
""")

        logger.info(f"SCEVAN driver script written: {driver_r}")
        logger.info("SCEVAN: launching Rscript (this may take ~15-20 min per batch)...")

        run_result = subprocess.run(
            [rscript_bin, "--vanilla", driver_r],
            capture_output=True, text=True, env=sub_env, cwd=save_dir,
        )
        for line in run_result.stdout.strip().splitlines():
            logger.info(f"[SCEVAN] {line}")
        for line in run_result.stderr.strip().splitlines():
            logger.debug(f"[SCEVAN stderr] {line}")

        if run_result.returncode != 0:
            logger.error(
                f"SCEVAN Rscript exited {run_result.returncode}.\n"
                + "\n".join(run_result.stderr.strip().splitlines()[-40:])
            )
            return empty_result

        if not os.path.exists(results_csv):
            logger.error(f"SCEVAN completed but results CSV not found: {results_csv}")
            return empty_result

        pred_df   = pd.read_csv(results_csv, index_col=0)
        logger.info(f"SCEVAN results columns: {list(pred_df.columns)}")

        class_col = "class" if "class" in pred_df.columns else pred_df.columns[0]

        pred_df = pred_df.reset_index().rename(columns={"index": "barcode"})
        pred_df = pred_df[~pred_df["barcode"].astype(str).str.startswith("REF_")]
        pred_df = pred_df.rename(columns={class_col: "scevan_prediction"})[
            ["barcode", "scevan_prediction"]
        ]
        pred_df["scevan_prediction"] = (
            pred_df["scevan_prediction"]
            .astype(str).str.strip().str.lower()
            .replace("nan", "not.defined")
            .fillna("not.defined")
        )

        result_full = pd.DataFrame({"barcode": list(q_barcodes_arr)})
        result_full = result_full.merge(pred_df, on="barcode", how="left")
        result_full["scevan_prediction"] = (
            result_full["scevan_prediction"].fillna("not.defined")
        )

        logger.info(
            "SCEVAN predictions:\n"
            + result_full["scevan_prediction"].value_counts().to_string()
        )
        return result_full

    except Exception as exc:
        logger.error(f"SCEVAN subprocess runner failed: {exc}")
        logger.exception("Full traceback:")
        return empty_result
    finally:
        if _tmpdir_created:
            try:
                shutil.rmtree(save_dir)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# CLAUDE EDIT — Step 8b: Cancer Composition Score (tumor vs. healthy
# reference), Harmony integration + before/after UMAP QC plot.
# Merged inline (single-file Module 3) rather than a separate import.
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# Raw count extraction (local copy — kept self-contained, mirrors the
# priority order used elsewhere in Module 3: layers['counts'] -> 'raw_counts'
# -> 'scvi_counts' -> 'raw_for_cna' -> .raw.X -> .X. 'raw_for_cna' is a
# defensive addition: it's set explicitly in this module's own Step 3
# (always genuinely raw, guaranteed present on adata_mal_fullgene) and
# covers the edge case where Module 2's own layers['counts'] restoration
# didn't succeed — 'counts' is still checked first and preferred when
# present, since it's correct under normal operation.
# ═══════════════════════════════════════════════════════════════════════════

def _get_raw_counts(adata, context=""):
    tag = f"[{context}] " if context else ""

    # CLAUDE EDIT — memory fix: keep sparse input sparse. The old version
    # always called .toarray() here, producing a dense copy even though
    # both call sites (build_tumor_healthy_combined) immediately re-wrap
    # the result in sp.csr_matrix() — a completely wasted dense round-trip.
    # For a ~49k-cell x 23.5k-gene healthy reference, that one .toarray()
    # call materializes an unnecessary ~4.6 GB dense array right before
    # Harmony/KMeans need their own memory — a very plausible cause of a
    # kernel dying with no traceback (OS OOM-killer, not a Python error).
    # sp.csr_matrix() accepts an already-sparse input safely and cheaply
    # (a format/dtype conversion, not a densify), so callers are unaffected.
    def _as_sparse_or_dense(X):
        if sp.issparse(X):
            return sp.csr_matrix(X, dtype=np.float32)
        return np.array(X, dtype=np.float32)

    for lyr in ("counts", "raw_counts", "scvi_counts", "raw_for_cna"):
        if lyr in adata.layers:
            logger.info(f"{tag}Raw counts source: layers['{lyr}']")
            return _as_sparse_or_dense(adata.layers[lyr])
    if adata.raw is not None:
        logger.info(f"{tag}Raw counts source: adata.raw.X ({adata.raw.n_vars} genes)")
        return _as_sparse_or_dense(adata.raw.X)
    logger.warning(
        f"{tag}No raw counts layer found on this object — using .X as-is. "
        "If this is already log-normalised, the composition score will be wrong."
    )
    return _as_sparse_or_dense(adata.X)


# ═══════════════════════════════════════════════════════════════════════════
# Batch-column auto-detection (mirrors popv_annotation.py's
# _detect_query_batch_key, kept as a local copy for module independence)
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_BATCH_CANDIDATES = [
    "gsm_id", "sample", "batch", "Sample", "Batch", "patient",
    "donor", "library", "Run", "run", "gse_id",
]


def _detect_batch_column(adata, user_key=None, candidates=None, context=""):
    tag = f"[{context}] " if context else ""
    candidates = candidates or _DEFAULT_BATCH_CANDIDATES

    if user_key is not None:
        if user_key not in adata.obs.columns:
            raise ValueError(
                f"{tag}batch key '{user_key}' not found in obs.\n"
                f"Available columns: {list(adata.obs.columns)}"
            )
        logger.info(f"{tag}batch key (user-specified): '{user_key}'")
        return user_key

    for key in candidates:
        if key in adata.obs.columns and adata.obs[key].nunique() >= 2:
            logger.info(f"{tag}batch key auto-detected: '{key}' "
                        f"({adata.obs[key].nunique()} unique values)")
            return key

    logger.warning(
        f"{tag}No batch column auto-detected (checked {candidates}) — "
        "all cells from this side will be treated as a single batch."
    )
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — build the combined tumor + healthy-reference AnnData
# ═══════════════════════════════════════════════════════════════════════════

def build_tumor_healthy_combined(
    adata_tumor_fullgene,
    healthy_reference_h5ad,
    tumor_batch_key=None,
    healthy_batch_key=None,
):
    """
    Combine malignant epithelial (tumor) cells with an external healthy
    reference on their common genes. Mirrors Ovarian_adata_prep.ipynb's
    merge logic (Data_Type tagging, batch tagging, inner-join on common
    genes, raw counts preserved in layers['counts']).
    """
    print(f"  Loading healthy reference: {healthy_reference_h5ad}")
    adata_healthy = sc.read_h5ad(healthy_reference_h5ad)

    common_genes = adata_tumor_fullgene.var_names.intersection(adata_healthy.var_names)
    print(
        f"  Tumor genes: {adata_tumor_fullgene.n_vars}  |  "
        f"Healthy genes: {adata_healthy.n_vars}  |  Common genes: {len(common_genes)}"
    )
    if len(common_genes) == 0:
        raise ValueError(
            "No common genes between malignant cells and healthy reference.\n"
            "Check both use the same gene identifier (HGNC symbol vs Ensembl ID)."
        )

    adata_t = adata_tumor_fullgene[:, common_genes].copy()
    adata_h = adata_healthy[:, common_genes].copy()

    # CLAUDE EDIT — memory fix: the full-size healthy reference (all
    # 61,806 genes in a real run, not just the ~23,539 common ones) is no
    # longer needed once adata_h has been extracted above. Explicitly
    # delete + collect here instead of waiting for adata_healthy to fall
    # out of scope at the end of this function, so that memory is freed
    # BEFORE the concat/Harmony steps below need their own.
    del adata_healthy
    import gc as _gc
    _gc.collect()

    adata_t.obs["Data_Type"] = "Tumor"
    adata_h.obs["Data_Type"] = "Healthy"

    t_key = _detect_batch_column(adata_t, tumor_batch_key, context="tumor")
    h_key = _detect_batch_column(
        adata_h, healthy_batch_key,
        candidates=["donor"] + _DEFAULT_BATCH_CANDIDATES,
        context="healthy",
    )
    adata_t.obs["batch"] = (adata_t.obs[t_key].astype(str) if t_key else "tumor_all")
    adata_h.obs["batch"] = (adata_h.obs[h_key].astype(str) if h_key else "healthy_all")

    raw_t = _get_raw_counts(adata_t, "tumor")
    raw_h = _get_raw_counts(adata_h, "healthy")
    adata_t.X = sp.csr_matrix(raw_t)
    adata_h.X = sp.csr_matrix(raw_h)

    adata_combined = sc.concat(
        [adata_t, adata_h],
        axis=0,
        join="inner",
        merge="same",
        label=None,
        keys=["Tumor", "Healthy"],
        index_unique="-",
    )
    adata_combined.var_names_make_unique()
    adata_combined.layers["counts"] = adata_combined.X.copy()

    for col in adata_combined.obs.columns:
        if adata_combined.obs[col].dtype == object:
            adata_combined.obs[col] = adata_combined.obs[col].astype(str)

    return adata_combined


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — HVG selection + scaling, for the Harmony/PCA/UMAP embedding ONLY
# (the Cancer Composition Score itself is computed later on the full
# common-gene space preserved in adata_combined.layers['counts'])
# ═══════════════════════════════════════════════════════════════════════════

def prepare_for_harmony(adata_combined, n_top_genes=3000):
    # CLAUDE EDIT — memory fix: no full-size copy before HVG subsetting.
    # The old version did adata_combined.copy() at the FULL common-gene
    # count (e.g. 23,539 in a real run) before ever reducing to HVGs,
    # meaning two full-size objects (the original adata_combined plus this
    # copy) sat in memory simultaneously right up until the HVG subset —
    # the single largest avoidable memory spike in this step. Normalizing
    # + log1p in place on adata_combined is safe: only .X is touched
    # (normalize_total/log1p never touch other layers by default), and
    # compute_cancer_composition_score() — called later on this same
    # adata_combined object — always re-derives .X fresh from
    # layers['counts'] regardless of what .X currently holds, so it does
    # not care that .X was mutated here.
    sc.pp.normalize_total(adata_combined, target_sum=1e4)
    sc.pp.log1p(adata_combined)

    # FIX vs. original notebook: flavor='seurat_v3' requires raw counts.
    # The notebook passed already-log1p'd .X (silent scanpy warning). Here
    # we point it at the preserved raw-counts layer explicitly.
    sc.pp.highly_variable_genes(
        adata_combined,
        batch_key="batch",
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
        layer="counts",
    )
    print(f"  Selected {int(adata_combined.var['highly_variable'].sum())} HVGs for Harmony/UMAP embedding")

    # The ONLY full-materializing copy in this function now happens here —
    # already reduced to n_top_genes, not the full common-gene count.
    adata_hvg = adata_combined[:, adata_combined.var["highly_variable"]].copy()
    import gc as _gc
    _gc.collect()
    sc.pp.scale(adata_hvg, max_value=10)
    return adata_hvg


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Harmony integration (ported from run_harmony_integration.py)
# ═══════════════════════════════════════════════════════════════════════════

def run_harmony_integration(
    adata_hvg,
    batch_key="batch",
    n_pcs=50,
    max_iter_harmony=10,
    theta=None,
    seed=0,
    svd_solver="arpack",
    n_jobs=1,
):
    import harmonypy

    # CLAUDE EDIT — thread-limiting, mirrors popv_annotation.py's Step 14
    # exactly (same env vars, same pattern). Without this, numpy/sklearn's
    # BLAS backend defaults to using ALL available cores for PCA and for
    # harmonypy's internal sklearn.KMeans centroid initialization. On a
    # multi-core HPC node with a per-job memory limit, that oversubscription
    # (each thread/worker duplicating chunks of the data) is a classic
    # silent OOM-killer trigger — the process is killed with no Python
    # traceback, which looks exactly like "the kernel just died".
    import os as _os
    n_threads = str(n_jobs if n_jobs > 0 else (_os.cpu_count() or 1))
    for env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ[env_var] = n_threads
    print(f"  Parallelism: n_jobs={n_jobs} ({n_threads} threads) for PCA + Harmony")

    print(f"  Running PCA (n_comps={n_pcs}, svd_solver={svd_solver})")
    sc.pp.pca(adata_hvg, n_comps=n_pcs, svd_solver=svd_solver)

    harmony_kwargs = {"max_iter_harmony": max_iter_harmony, "random_state": seed}
    if theta is not None:
        harmony_kwargs["theta"] = theta

    print(f"  Running Harmony (vars_use=['{batch_key}'], "
          f"max_iter_harmony={max_iter_harmony}, theta={theta}, random_state={seed})")
    ho = harmonypy.run_harmony(
        adata_hvg.obsm["X_pca"], adata_hvg.obs, vars_use=[batch_key], **harmony_kwargs,
    )

    Z = np.asarray(ho.Z_corr)
    if Z.shape == (adata_hvg.n_obs, n_pcs):
        adata_hvg.obsm["X_pca_harmony"] = Z
    elif Z.shape == (n_pcs, adata_hvg.n_obs):
        adata_hvg.obsm["X_pca_harmony"] = Z.T
    else:
        raise RuntimeError(f"Unexpected Harmony output shape: {Z.shape}")

    return adata_hvg


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — UMAP before (unintegrated PCA) vs. after (Harmony) + QC plot
# ═══════════════════════════════════════════════════════════════════════════

def _scatter(ax, coords, labels, title, legend=True):
    import matplotlib.pyplot as plt

    labels = pd.Categorical(pd.Series(labels).astype(str))
    cmap = plt.get_cmap("tab20", max(len(labels.categories), 1))
    for i, cat in enumerate(labels.categories):
        mask = np.asarray(labels == cat)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=3, color=cmap(i), label=cat)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if legend and len(labels.categories) <= 15:
        ax.legend(markerscale=4, fontsize=6, loc="best", frameon=False)


def plot_umap_before_after(adata_hvg, save_dir, sample_name="cancer_composition"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("  Computing UMAP — unintegrated (X_pca)")
    sc.pp.neighbors(adata_hvg, use_rep="X_pca", key_added="unintegrated")
    sc.tl.umap(adata_hvg, neighbors_key="unintegrated")
    umap_unintegrated = adata_hvg.obsm["X_umap"].copy()

    print("  Computing UMAP — Harmony-integrated (X_pca_harmony)")
    sc.pp.neighbors(adata_hvg, use_rep="X_pca_harmony", key_added="harmony")
    sc.tl.umap(adata_hvg, neighbors_key="harmony")
    umap_harmony = adata_hvg.obsm["X_umap"].copy()

    adata_hvg.obsm["X_umap_unintegrated"] = umap_unintegrated
    adata_hvg.obsm["X_umap_harmony"]      = umap_harmony

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    _scatter(axes[0, 0], umap_unintegrated, adata_hvg.obs["Data_Type"], "Unintegrated — Data_Type")
    _scatter(axes[0, 1], umap_unintegrated, adata_hvg.obs["batch"],     "Unintegrated — batch")
    _scatter(axes[1, 0], umap_harmony,      adata_hvg.obs["Data_Type"], "Harmony-integrated — Data_Type")
    _scatter(axes[1, 1], umap_harmony,      adata_hvg.obs["batch"],     "Harmony-integrated — batch")
    fig.suptitle("Tumor vs. Healthy Reference — before/after Harmony integration", fontsize=12)
    fig.tight_layout()

    pdf_path = os.path.join(save_dir, f"{sample_name}_umap_before_after.pdf")
    png_path = os.path.join(save_dir, f"{sample_name}_umap_before_after.png")
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)

    return pdf_path, png_path


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — Cancer Composition Score (tumor-vs-healthy detection-rate proxy)
# ═══════════════════════════════════════════════════════════════════════════

def compute_cancer_composition_score(adata_combined, save_dir=None, sample_name="cancer_composition"):
    adata_cc = adata_combined.copy()
    adata_cc.X = adata_cc.layers["counts"].copy()
    sc.pp.normalize_total(adata_cc, target_sum=1e6)

    tumor   = (adata_cc.obs["Data_Type"] == "Tumor").to_numpy()
    healthy = (adata_cc.obs["Data_Type"] == "Healthy").to_numpy()
    print(f"  Tumor cells: {tumor.sum()}  |  Healthy cells: {healthy.sum()}")

    X = adata_cc.X
    if sp.issparse(X):
        tumor_pct   = np.asarray((X[tumor]   > 0).mean(axis=0)).ravel() * 100
        healthy_pct = np.asarray((X[healthy] > 0).mean(axis=0)).ravel() * 100
    else:
        tumor_pct   = (X[tumor]   > 0).mean(axis=0) * 100
        healthy_pct = (X[healthy] > 0).mean(axis=0) * 100

    cancer_composition_preview = pd.DataFrame({
        "Gene"        : adata_cc.var_names,
        "Tumor_pct"   : tumor_pct,
        "Healthy_pct" : healthy_pct,
    })
    print(cancer_composition_preview.head().to_string(index=False))

    tumor_z   = (tumor_pct   - tumor_pct.mean())   / tumor_pct.std()
    healthy_z = (healthy_pct - healthy_pct.mean()) / healthy_pct.std()
    cc_score  = tumor_z - healthy_z

    # Strict parity with script 3: scores are also written back onto the
    # combined AnnData's .var (script 3 wrote adata.var[...] = ...), not
    # just returned as a standalone dataframe.
    adata_combined.var["Tumor_pct"]         = tumor_pct
    adata_combined.var["Healthy_pct"]       = healthy_pct
    adata_combined.var["Tumor_Z"]           = tumor_z
    adata_combined.var["Healthy_Z"]         = healthy_z
    adata_combined.var["Cancer_Composition"] = cc_score

    cc_df = pd.DataFrame({
        "Gene"        : adata_cc.var_names,
        "Tumor_pct"   : tumor_pct,
        "Healthy_pct" : healthy_pct,
        "Tumor_Z"     : tumor_z,
        "Healthy_Z"   : healthy_z,
        "Cancer_Composition": cc_score,
    })
    cc_df["Cancer_Composition_Scaled"] = (
        (cc_df["Cancer_Composition"] - cc_df["Cancer_Composition"].min())
        / (cc_df["Cancer_Composition"].max() - cc_df["Cancer_Composition"].min())
    )
    cc_df = cc_df.sort_values("Cancer_Composition", ascending=False).reset_index(drop=True)

    print("\n  Top 20 genes by Cancer_Composition (tumor-preferential):")
    print(cc_df.head(20).to_string(index=False))

    if save_dir is not None:
        out_csv = os.path.join(save_dir, f"{sample_name}_cancer_composition_scores.csv")
        cc_df.to_csv(out_csv, index=False)
        print(f"\n  Cancer Composition scores (all genes) saved to: {out_csv}")

    return cc_df


# ═══════════════════════════════════════════════════════════════════════════
# Memory checkpoint logging — diagnostic instrumentation
# ═══════════════════════════════════════════════════════════════════════════

def _log_mem(label):
    """
    Print the process's PEAK resident memory so far (Linux: ru_maxrss is
    reported in KB; this function converts to GB). This is cumulative/
    monotonically non-decreasing across the whole process, not a snapshot
    of current usage — so consecutive calls show exactly how much peak
    memory GREW between checkpoints, which is what we need to pinpoint
    where inside Step 8b memory actually balloons, rather than only
    knowing the final number at the moment of an OOM-kill.
    """
    try:
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_gb = peak_kb / (1024 ** 2)
        print(f"  [mem] {label}: peak RSS so far = {peak_gb:.2f} GB")
    except Exception as exc:
        print(f"  [mem] {label}: could not read memory usage ({exc})")


# ═══════════════════════════════════════════════════════════════════════════
# Public entry point — called from preprocessing.py's Step 8b
# ═══════════════════════════════════════════════════════════════════════════

def run_cancer_composition_step(
    adata_tumor_fullgene,
    healthy_reference_h5ad,
    save_dir,
    tumor_batch_key=None,
    healthy_batch_key=None,
    n_top_genes=3000,
    n_pcs=50,
    max_iter_harmony=10,
    theta=None,
    seed=0,
    n_jobs=1,
    cc_score_threshold=0.5,
    sample_name="cancer_composition",
):
    """
    Full sub-pipeline: build tumor+healthy combined object -> HVG select ->
    Harmony integrate -> UMAP before/after -> Cancer Composition Score.

    n_jobs : int
        CPU threads for PCA + Harmony (default 1 — deliberately
        conservative). Passed to OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/
        MKL_NUM_THREADS the same way Module 2 already does. Raise this only
        if you know you have the RAM headroom for it: sklearn's KMeans
        (used internally by harmonypy) duplicates data across threads/
        workers, and on a shared multi-core HPC node with a per-job memory
        cap, an unconstrained thread count is a common cause of the kernel
        silently dying (OOM-killed) with no Python traceback.

    Returns
    -------
    cc_df    : pd.DataFrame, every common gene, sorted by Cancer_Composition desc.
    cc_genes : set[str], genes with Cancer_Composition >= cc_score_threshold.
    plot_paths : (pdf_path, png_path) for the before/after UMAP figure.
    """
    print("\n--- Step 8b: Cancer Composition Score (tumor vs. healthy reference) ---")
    os.makedirs(save_dir, exist_ok=True)
    _log_mem("Step 8b start")

    # CLAUDE EDIT — global seed, matching run_harmony_integration.py's
    # set_seed() exactly. Previously only harmonypy's own random_state was
    # seeded (inside run_harmony_integration() below); sc.tl.umap() is also
    # stochastic and depends on sc.settings.seed / numpy's global RNG for
    # reproducibility, which was NOT being set — so the before/after UMAP
    # plots weren't actually reproducible run-to-run despite passing seed=.
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    sc.settings.seed = seed

    adata_combined = build_tumor_healthy_combined(
        adata_tumor_fullgene, healthy_reference_h5ad,
        tumor_batch_key=tumor_batch_key, healthy_batch_key=healthy_batch_key,
    )
    print(f"  Combined: {adata_combined.n_obs} cells x {adata_combined.n_vars} common genes")
    print(f"  Data_Type counts: {adata_combined.obs['Data_Type'].value_counts().to_dict()}")
    _log_mem("after build_tumor_healthy_combined (healthy ref loaded + merged)")

    adata_hvg = prepare_for_harmony(adata_combined, n_top_genes=n_top_genes)
    _log_mem("after prepare_for_harmony (normalize + HVG select + scale)")

    adata_hvg = run_harmony_integration(
        adata_hvg, batch_key="batch", n_pcs=n_pcs,
        max_iter_harmony=max_iter_harmony, theta=theta, seed=seed, n_jobs=n_jobs,
    )
    _log_mem("after run_harmony_integration (PCA + Harmony + KMeans — the crash point)")

    plot_paths = plot_umap_before_after(adata_hvg, save_dir, sample_name=sample_name)
    print(f"  UMAP (unintegrated vs Harmony-integrated) saved to:\n"
          f"    {plot_paths[0]}\n    {plot_paths[1]}")
    _log_mem("after plot_umap_before_after")

    cc_df = compute_cancer_composition_score(adata_combined, save_dir=save_dir, sample_name=sample_name)
    _log_mem("after compute_cancer_composition_score")

    cc_genes = set(cc_df.loc[cc_df["Cancer_Composition"] >= cc_score_threshold, "Gene"])
    print(f"\n  Genes with Cancer_Composition >= {cc_score_threshold}: {len(cc_genes)}")

    return cc_df, cc_genes, plot_paths


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_preprocessing_pipeline(
    adata=None,
    popv_path=None,
    log2fc_threshold=1.0,
    pval_adj_threshold=0.05,
    reference_h5ad=None,
    tumor_h5ad=None,
    save_dir=None,
    scmalignant_model_dir=None,
    surfaceome_path=None,
    malignant_strategy="intersection",
    scevan_ref_max_cells=500,
    scevan_ref_cell_col="cell_ontology_class",
    scevan_ref_epithelial_values=None,
    scevan_sample_name="SCEVAN_run",
    scevan_organism="human",
    scevan_par_cores=1,
    scevan_subclones=False,
    scevan_batch_size=3000,
    # CLAUDE EDIT — Step 8b (Cancer Composition Score) parameters.
    # No separate healthy-reference path here on purpose: reference_h5ad
    # (already required above, for SCEVAN) is the ONE h5ad the user supplies
    # for this whole module — it's reused as the healthy reference here too.
    run_cancer_composition=True,
    cc_score_threshold=0.5,
    cc_n_top_genes=3000,
    cc_n_pcs=50,
    cc_max_iter_harmony=10,
    cc_theta=None,
    cc_seed=0,
    cc_n_jobs=1,
    cc_tumor_batch_key=None,
    cc_healthy_batch_key=None,
):
    print("\n========== START ==========\n")

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "preprocessing_results")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory: {save_dir}")

    if scmalignant_model_dir is None:
        scmalignant_model_dir = _auto_scmalignant_model()
    logger.info(f"scMalignantFinder model: {scmalignant_model_dir}")

    if surfaceome_path is None:
        surfaceome_path = _auto_surfaceome_path()
    logger.info(f"Surfaceome (GESP) path: {surfaceome_path}")

    # STEP 1 — Load full PopV h5ad
    if adata is None:
        auto_popv = _auto_popv_h5ad()
        for path in ([popv_path] if popv_path else []) + ([auto_popv] if auto_popv else []):
            if path and os.path.exists(path):
                print(f"Loading PopV output: {path}")
                adata = sc.read_h5ad(path)
                break
        if adata is None:
            raise FileNotFoundError(
                "Could not auto-detect PopV output.\n"
                "Expected: popv_results/final_popv_annotated.h5ad\n"
                "Pass adata= or popv_path= explicitly."
            )

    adata_full = adata.copy()
    print(f"Full dataset loaded: {adata_full.n_obs} cells × {adata_full.n_vars} genes")

    _skip_popv  = adata_full.uns.get("skip_popv", False)
    _manual_col = adata_full.uns.get("manual_annotation_col", None)

    if _skip_popv:
        print(
            f"\n  *** Manual annotation mode detected ***\n"
            f"  adata.uns['skip_popv']            = True\n"
            f"  adata.uns['manual_annotation_col'] = '{_manual_col}'\n"
            f"  PopV was skipped — using 'popv_majority_vote_prediction' "
            f"copied from column '{_manual_col}' by SampleAnnotator.\n"
        )
        if "popv_majority_vote_prediction" not in adata_full.obs.columns:
            raise ValueError(
                "adata.uns['skip_popv'] is True but "
                "'popv_majority_vote_prediction' is missing from adata.obs.\n"
                f"Expected it to be a copy of obs column '{_manual_col}'.\n"
                "Re-run SampleAnnotator with manual_annotation_col= to regenerate "
                "the h5ad, or add 'popv_majority_vote_prediction' manually."
            )
        _label_counts = adata_full.obs["popv_majority_vote_prediction"].value_counts()
        _epi_labels   = [
            l for l in _label_counts.index
            if "epithelial cell" in str(l).lower()
        ]
        print(
            f"  Label summary (popv_majority_vote_prediction):\n"
            + _label_counts.to_string()
            + f"\n\n  Epithelial labels that will be selected in Step 3:\n"
            + ("  " + "\n  ".join(_epi_labels) if _epi_labels
               else "  *** NONE FOUND — check your label names! ***")
            + "\n"
        )
        if not _epi_labels:
            raise ValueError(
                "Manual annotation mode: no labels containing 'epithelial cell' "
                f"found in 'popv_majority_vote_prediction'.\n"
                f"Unique labels present: {list(_label_counts.index)}\n"
                "Epithelial labels must contain the phrase 'epithelial cell' "
                "(case-insensitive), e.g.:\n"
                "  'epithelial cell'\n"
                "  'ovarian surface epithelial cell'\n"
                "  'glandular epithelial cell'\n"
                "Please rename your epithelial labels in the source h5ad and "
                "re-run SampleAnnotator."
            )
    else:
        if "popv_majority_vote_prediction" not in adata_full.obs.columns:
            raise ValueError(
                "'popv_majority_vote_prediction' not found in adata.obs.\n"
                "Expected this column from Module 2 (PopV annotation).\n"
                "If you want to use your own annotations, re-run Module 1 "
                "(SampleAnnotator) with manual_annotation_col= set."
            )

    _raw_layers_present = [
        l for l in ("counts", "raw_counts", "scvi_counts")
        if l in adata_full.layers
    ]
    if _raw_layers_present:
        print(f"Raw count layers available: {_raw_layers_present} "
              f"(expected — PopV always outputs raw counts)")
    else:
        print(
            "WARNING: No raw count layers found in PopV output.\n"
            "  This is unexpected under the normal Module 1->2->3 flow "
            "(PopV always writes layers['counts']) — this h5ad may have "
            "bypassed PopV or come from elsewhere.\n"
            "  Steps requiring raw counts (QC, SCEVAN, scMalignantFinder)\n"
            "  will attempt Route A-rescue from the Module 1 tumor h5ad.\n"
            f"  Provide tumor_h5ad= explicitly or place GSE*_tumor.h5ad in cwd."
        )

    # STEP 2 — Read QC thresholds
    min_genes, max_mt, qc_active, qc_source = _read_qc_params(adata_full)

    # STEP 3 — Extract epithelial cells
    # CLAUDE EDIT: QC (min_genes / max_mt) is intentionally NOT re-applied
    # here. Module 1 already filtered cells against these exact thresholds
    # before this data ever reached Module 2/3 — re-filtering here dropped
    # cells a second time against the same numbers (a "double QC"). The
    # qc_params read above (Step 2) is still carried through to the final
    # output's uns['qc_params'] purely as a record of what Module 1 did.
    print("\n--- Step 3: Epithelial selection ---")

    labels  = adata_full.obs["popv_majority_vote_prediction"].astype(str)
    ep_mask = labels.str.contains("epithelial cell", case=False, na=False)

    print(f"Epithelial cells (contains 'epithelial cell'): "
          f"{ep_mask.sum()} / {adata_full.n_obs} total")
    print(f"Non-epithelial cells (will be 'rest' group for DEG): {(~ep_mask).sum()}")

    matched_labels = labels[ep_mask].unique()
    logger.info(f"Epithelial labels matched: {sorted(matched_labels)}")

    adata_epi = adata_full[ep_mask].copy()

    print("Setting up raw counts for epithelial cells...")
    _raw_epi = _get_raw_counts_from_adata(adata_epi, "epithelial-raw-setup")
    adata_epi.X                     = sp.csr_matrix(_raw_epi)
    adata_epi.layers["raw_for_cna"] = adata_epi.X.copy()
    print(f"  layers['raw_for_cna'] stored ({adata_epi.n_obs} × {adata_epi.n_vars})")

    adata_epi.var_names_make_unique()
    sc.pp.normalize_total(adata_epi, target_sum=1e4)
    sc.pp.log1p(adata_epi)

    # STEP 4a — scMalignantFinder
    print("\n--- Step 4a: scMalignantFinder ---")
    feature_tsv = os.path.join(scmalignant_model_dir, "ordered_feature.tsv")

    _raw_layers_epi = [l for l in ("counts", "raw_counts") if l in adata_epi.layers]
    if _raw_layers_epi:
        print(f"Gene-space: Route A-layer (layers['{_raw_layers_epi[0]}'], "
              f"{adata_epi.n_vars} HVGs)")
    elif "full_counts" in adata_epi.layers and adata_epi.uns.get("full_counts_var_names"):
        print(f"Gene-space: Route A-new ({len(adata_epi.uns['full_counts_var_names'])} genes)")
    else:
        rescue = tumor_h5ad or _auto_tumor_h5ad()
        if rescue and os.path.exists(rescue):
            print(f"Gene-space: Route A-rescue ({rescue})")
        elif adata_epi.raw is not None:
            print(f"Gene-space: Route A-old (adata.raw, {adata_epi.raw.n_vars} genes)")
        else:
            print("Gene-space: Route C (HVG fallback — low overlap expected)")

    adata_scm = _build_fullgene_adata_for_scm(adata_epi, feature_tsv, tumor_h5ad)
    print(f"  Gene space used: {adata_scm.n_vars} genes")

    import importlib.util as _ilu

    _scm_pkg_dir      = os.path.dirname(scmalignant_model_dir)
    _scm_external_dir = os.path.dirname(_scm_pkg_dir)
    _classifier_py    = os.path.join(_scm_pkg_dir, "classifier.py")
    _init_py          = os.path.join(_scm_pkg_dir, "__init__.py")

    def _load_module_from_file(mod_name, filepath):
        if mod_name in sys.modules:
            return sys.modules[mod_name]
        _spec = _ilu.spec_from_file_location(mod_name, filepath)
        _mod  = _ilu.module_from_spec(_spec)
        sys.modules[mod_name] = _mod
        _spec.loader.exec_module(_mod)
        return _mod

    _clf_mod = None
    if os.path.isfile(_classifier_py):
        logger.info(f"scMalignantFinder: loading via classifier.py ({_classifier_py})")
        _clf_mod = _load_module_from_file("scMalignantFinder.classifier", _classifier_py)
    elif os.path.isfile(_init_py):
        logger.info(f"scMalignantFinder: loading via __init__.py ({_init_py})")
        _clf_mod = _load_module_from_file("scMalignantFinder", _init_py)
    else:
        logger.warning(f"scMalignantFinder not found in {_scm_pkg_dir} — sys.path fallback.")
        _inserted = _scm_external_dir not in sys.path
        if _inserted:
            sys.path.insert(0, _scm_external_dir)
        try:
            import scMalignantFinder as _scm_pkg
            _clf_mod = _scm_pkg
        except ImportError as _exc:
            raise ImportError(
                f"Could not import scMalignantFinder.\n"
                f"  classifier.py: {_classifier_py}\n"
                f"  __init__.py:   {_init_py}\n"
                f"  Error: {_exc}"
            ) from _exc
        finally:
            if _inserted and _scm_external_dir in sys.path:
                sys.path.remove(_scm_external_dir)

    model = _clf_mod.scMalignantFinder(
        test_input          = adata_scm,
        celltype_annotation = False,
        pretrain_dir        = scmalignant_model_dir,
        feature_path        = feature_tsv,
        norm_type           = False,
    )
    model.load()
    result_scm = model.predict()

    scm_col = "scMalignantFinder_prediction"
    if result_scm.obs_names.equals(adata_epi.obs_names):
        adata_epi.obs[scm_col] = result_scm.obs[scm_col].values
    else:
        logger.warning("scMalignantFinder obs_names differ — aligning by index.")
        adata_epi.obs[scm_col] = (
            result_scm.obs[scm_col]
            .reindex(adata_epi.obs_names)
            .fillna("Unknown")
            .values
        )

    print("scMalignantFinder completed:")
    print(adata_epi.obs[scm_col].value_counts().to_string())

    # STEP 4b — SCEVAN
    scevan_available = False
    scevan_result_df = None

    if malignant_strategy in ("scevan", "intersection"):
        if reference_h5ad is None:
            print(
                "\nWarning: SCEVAN skipped — no reference_h5ad provided.\n"
                "  Falling back to scMalignantFinder only.\n"
                "  Pass reference_h5ad= to enable SCEVAN."
            )
            malignant_strategy = "scMalignant"
        else:
            _rs = _find_rscript()
            _rh = _get_r_home(_rs) if _rs else "NOT FOUND"
            print(
                f"\n--- Step 4b: SCEVAN (subprocess) ---\n"
                f"  Reference h5ad:           {reference_h5ad}\n"
                f"  ref_cell_col:             {scevan_ref_cell_col}\n"
                f"  ref_epithelial_values:    {scevan_ref_epithelial_values}\n"
                f"  ref_max_cells:            {scevan_ref_max_cells} "
                f"({'all available' if scevan_ref_max_cells is None else 'max'})\n"
                f"  sample_name:              {scevan_sample_name}\n"
                f"  organism:                 {scevan_organism}\n"
                f"  par_cores:                {scevan_par_cores}\n"
                f"  subclones:                {scevan_subclones}\n"
                f"  batch_size:               {scevan_batch_size}\n"
                f"  Rscript:                  {_rs}\n"
                f"  R home:                   {_rh}"
            )

            if not _raw_layers_present:
                print(
                    "\n  WARNING: No raw count layers in PopV output "
                    "(unexpected — PopV always outputs raw counts; this "
                    "h5ad may have bypassed PopV).\n"
                    "  SCEVAN requires integer raw counts for reliable CNV inference.\n"
                    "  Results may be unreliable."
                )

            try:
                adata_raw_cna   = adata_epi.copy()
                adata_raw_cna.X = adata_epi.layers["raw_for_cna"]

                adata_ref_full  = sc.read_h5ad(reference_h5ad)

                scevan_work_dir = os.path.join(save_dir, "scevan")
                os.makedirs(scevan_work_dir, exist_ok=True)

                scevan_result_df = _run_scevan(
                    adata_query             = adata_raw_cna,
                    adata_ref               = adata_ref_full,
                    ref_cell_col            = scevan_ref_cell_col,
                    ref_epithelial_values   = scevan_ref_epithelial_values,
                    ref_max_cells           = scevan_ref_max_cells,
                    sample_name             = scevan_sample_name,
                    organism                = scevan_organism,
                    par_cores               = scevan_par_cores,
                    subclones               = scevan_subclones,
                    batch_size              = scevan_batch_size,
                    save_dir                = scevan_work_dir,
                )

                bc_to_pred = dict(zip(
                    scevan_result_df["barcode"],
                    scevan_result_df["scevan_prediction"],
                ))
                adata_epi.obs["scevan_prediction"] = [
                    bc_to_pred.get(b, "not.defined") for b in adata_epi.obs_names
                ]
                scevan_available = True
                print("\nSCEVAN completed.")
                print("  Prediction counts:")
                print(adata_epi.obs["scevan_prediction"].value_counts().to_string())

            except Exception as exc:
                print(f"\nWarning: SCEVAN failed — {type(exc).__name__}: {exc}\n"
                      "  Falling back to scMalignantFinder only.")
                logger.exception("SCEVAN error:")
                malignant_strategy = "scMalignant"

    # STEP 4c — Combine malignancy calls
    print("\n--- Step 4c: Combine malignancy calls ---")
    scm_mal = adata_epi.obs[scm_col].str.lower() == "malignant"

    if scevan_available:
        sv_mal = adata_epi.obs["scevan_prediction"].str.lower() == "tumor"
        if malignant_strategy == "intersection":
            malignant_mask = scm_mal & sv_mal
            strategy_label = "intersection (scMalignantFinder AND SCEVAN)"
        elif malignant_strategy == "scevan":
            malignant_mask = sv_mal
            strategy_label = "SCEVAN only"
        else:
            malignant_mask = scm_mal
            strategy_label = "scMalignantFinder only"
    else:
        malignant_mask = scm_mal
        strategy_label = "scMalignantFinder only"

    adata_epi.obs["final_malignant"] = malignant_mask.map(
        {True: "malignant", False: "non-malignant"}
    )
    print(f"Strategy: {strategy_label}")
    print(f"  Malignant epithelial:     {malignant_mask.sum()}")
    print(f"  Non-malignant epithelial: {(~malignant_mask).sum()}  ← these will be REMOVED")

    # STEP 5 — Keep ONLY malignant epithelial cells
    print("\n--- Step 5: Retain malignant epithelial cells only ---")
    adata_mal = adata_epi[malignant_mask].copy()
    print(f"Malignant epithelial cells retained: {adata_mal.n_obs}")
    print(f"Non-malignant epithelial cells removed: {(~malignant_mask).sum()}")

    if adata_mal.n_obs == 0:
        raise ValueError(
            "No malignant cells found after filtering.\n"
            "Check scMalignantFinder model path and gene overlap."
        )

    # CLAUDE EDIT — full-gene-space snapshot for Step 8b, taken right after
    # Step 5, before Step 9's final gene subsetting touches adata_mal.
    adata_mal_fullgene = adata_mal.copy()

    # STEP 6 — Non-epithelial "rest" group
    print("\n--- Step 6: Extract non-epithelial 'rest' group ---")
    rest_mask  = ~ep_mask
    adata_rest = adata_full[rest_mask].copy()
    print(f"Non-epithelial 'rest' cells: {adata_rest.n_obs}")

    _raw_rest    = _get_raw_counts_from_adata(adata_rest, "rest-group")
    adata_rest.X = sp.csr_matrix(_raw_rest)
    sc.pp.normalize_total(adata_rest, target_sum=1e4)
    sc.pp.log1p(adata_rest)

    # STEP 7 — Load surfaceome gene list (GESP file)
    # CLAUDE EDIT: this used to SUBSET adata_mal/adata_rest to surfaceome
    # genes before DEG ran, so Step 8's Wilcoxon test only ever saw ~3,568
    # surfaceome genes. Per your correction, surfaceome membership is now a
    # separate, independent criterion — not a pre-filter. Step 8 (DEG) and
    # Step 8b (Cancer Composition) both now run on the FULL common gene
    # space; the surfaceome list is intersected in at the very end
    # (Step 9), alongside DEG and Cancer Composition, as the 3rd of 3 equal
    # criteria.
    print("\n--- Step 7: Load surfaceome gene list (GESP file) ---")
    surfaceome         = pd.read_csv(surfaceome_path)
    surfaceome.columns = surfaceome.columns.str.strip()
    surf_genes          = surfaceome["Gene"].astype(str).tolist()
    surf_genes_set       = set(surf_genes)
    print(f"Surfaceome genes in GESP file: {len(surf_genes_set)}")

    surf_in_mal = adata_mal.var_names.intersection(surf_genes_set)
    print(f"  Of those, present in the malignant-cell gene space: {len(surf_in_mal)} "
          f"(membership only — NOT filtered here, see Step 9)\n")

    # STEP 8 — DEG (now runs on the FULL common gene space, unrestricted)
    print("--- Step 8: DEG — malignant epithelial vs non-epithelial rest ---")

    adata_mal.obs["deg_group"]  = "malignant_epithelial"
    adata_rest.obs["deg_group"] = "non_epithelial_rest"

    # CLAUDE EDIT — capture before freeing adata_rest later; only the cell
    # count is needed downstream (uns['deg_params']), not the object itself.
    n_rest_cells = adata_rest.n_obs

    adata_deg = sc.concat([adata_mal, adata_rest], join="outer", label=None)
    adata_deg.obs_names_make_unique()
    adata_deg.var = adata_mal.var.copy()

    # CLAUDE EDIT — memory fix: same anti-pattern as _get_raw_counts (now
    # fixed above) — densify -> nan_to_num -> resparsify wasted a full
    # dense copy (for 7,528 cells x 23,539 genes here, ~700MB+) purely to
    # sanitize NaNs. scanpy's rank_genes_groups works fine on sparse input;
    # NaN-cleaning only needs to touch the sparse matrix's .data array
    # (just the non-zero values), never the full dense matrix.
    if sp.issparse(adata_deg.X):
        adata_deg.X = adata_deg.X.tocsr().astype(np.float32)
        np.nan_to_num(adata_deg.X.data, copy=False, nan=0.0)
    else:
        adata_deg.X = sp.csr_matrix(
            np.nan_to_num(np.asarray(adata_deg.X, dtype=np.float32), nan=0.0)
        )

    print(f"DEG AnnData: {adata_deg.n_obs} cells × {adata_deg.n_vars} genes")
    print(f"  malignant_epithelial: "
          f"{(adata_deg.obs['deg_group'] == 'malignant_epithelial').sum()}")
    print(f"  non_epithelial_rest:  "
          f"{(adata_deg.obs['deg_group'] == 'non_epithelial_rest').sum()}")

    sc.tl.rank_genes_groups(
        adata_deg,
        groupby="deg_group",
        groups=["malignant_epithelial"],
        reference="non_epithelial_rest",
        method="wilcoxon",
        key_added="rank_genes_groups",
    )
    deg = sc.get.rank_genes_groups_df(adata_deg, group="malignant_epithelial")
    print(f"\nTotal DEG candidates: {deg.shape[0]}")
    print(f"Applying filters: log2FC > {log2fc_threshold}, pvals_adj < {pval_adj_threshold}")

    # CLAUDE EDIT — memory fix: adata_deg and adata_rest are not referenced
    # anywhere after this point (deg's stats are already extracted above;
    # n_rest_cells was captured earlier) — free them explicitly now,
    # before Step 8b's much larger Harmony/PCA/KMeans allocations begin,
    # instead of letting them sit in memory unused for the rest of the run.
    del adata_deg, adata_rest
    import gc as _gc
    _gc.collect()

    filtered_deg = deg[
        (deg["logfoldchanges"] > log2fc_threshold) &
        (deg["pvals_adj"]      < pval_adj_threshold)
    ]

    if filtered_deg.shape[0] == 0:
        print("WARNING: 0 DEGs passed the filter.\n"
              "  Try: log2fc_threshold=0.5 or pval_adj_threshold=0.10")
    else:
        print(f"DEGs retained: {filtered_deg.shape[0]}")
        print("\nTop 10 DEGs (malignant epithelial vs non-epithelial rest):")
        print(filtered_deg.head(10).to_string(index=False))

    # CLAUDE EDIT — STEP 8b: Cancer Composition Score (tumor vs. healthy
    # reference), Harmony-integrated, with before/after UMAP QC plots.
    # Uses reference_h5ad directly — the same Tabula Sapiens file already
    # required above for SCEVAN. No separate healthy-reference argument.
    cc_df    = None
    cc_genes = None
    cc_plot_paths = None
    _cc_healthy_ref = reference_h5ad

    if run_cancer_composition and _cc_healthy_ref is None:
        print(
            "\nWarning: Step 8b skipped — no reference_h5ad provided.\n"
            "  Pass reference_h5ad= (the same Tabula Sapiens file used for "
            "SCEVAN) to enable the Cancer Composition Score."
        )
    elif run_cancer_composition:
        try:
            cc_save_dir = os.path.join(save_dir, "cancer_composition")
            cc_df, cc_genes, cc_plot_paths = run_cancer_composition_step(
                adata_tumor_fullgene   = adata_mal_fullgene,
                healthy_reference_h5ad = _cc_healthy_ref,
                save_dir               = cc_save_dir,
                tumor_batch_key        = cc_tumor_batch_key,
                healthy_batch_key      = cc_healthy_batch_key,
                n_top_genes            = cc_n_top_genes,
                n_pcs                  = cc_n_pcs,
                max_iter_harmony       = cc_max_iter_harmony,
                theta                  = cc_theta,
                seed                   = cc_seed,
                n_jobs                 = cc_n_jobs,
                cc_score_threshold     = cc_score_threshold,
                sample_name            = scevan_sample_name,
            )
        except Exception as exc:
            print(f"\nWarning: Step 8b (Cancer Composition Score) failed — "
                  f"{type(exc).__name__}: {exc}\n"
                  "  Continuing without it — Step 9 will fall back to "
                  "DEG ∩ surfaceome only.")
            logger.exception("Cancer Composition Score error:")
            cc_df, cc_genes, cc_plot_paths = None, None, None

    # STEP 9 — Subset to final gene list (3-way intersection), binarise, store, save
    print("\n--- Step 9: Subset to final gene list, binarise and save ---")

    deg_genes = filtered_deg["names"].tolist() if filtered_deg.shape[0] > 0 else None

    selection_sets = {}
    if deg_genes is not None:
        selection_sets["DEG malignant-vs-rest (Step 8)"] = set(deg_genes)
    if cc_genes is not None:
        selection_sets[f"Cancer_Composition >= {cc_score_threshold} (Step 8b)"] = cc_genes
    selection_sets["surfaceome gene list (Step 7)"] = surf_genes_set

    for label, gset in selection_sets.items():
        print(f"  {label}: {len(gset)} genes")

    if len(selection_sets) >= 2:
        final_gene_list = set.intersection(*selection_sets.values())
    else:
        final_gene_list = next(iter(selection_sets.values()))

    print(f"  Final intersection ({' ∩ '.join(selection_sets.keys())}): "
          f"{len(final_gene_list)} genes")

    if len(final_gene_list) == 0:
        print(
            "WARNING: 0 genes passed the final intersection — "
            "saving all surfaceome genes (no gene subset) instead.\n"
            "  Consider relaxing log2fc_threshold / pval_adj_threshold / "
            "cc_score_threshold."
        )
        final_gene_list = surf_genes_set

    final_in_mal = adata_mal.var_names.intersection(final_gene_list)
    adata_mal    = adata_mal[:, final_in_mal].copy()
    print(f"Malignant cells subset to {len(final_in_mal)} final genes.")

    adata_mal.X = (
        np.array(
            adata_mal.X.toarray() if sp.issparse(adata_mal.X) else adata_mal.X
        ) > 0
    ).astype(np.int8)
    adata_mal.X = sp.csr_matrix(adata_mal.X)
    print("Expression converted to binary (0/1).")

    adata_mal.uns["filtered_deg"] = filtered_deg.reset_index(drop=True)
    adata_mal.uns["all_deg"]      = deg.reset_index(drop=True)
    adata_mal.uns["deg_params"]   = {
        "comparison"         : "malignant_epithelial vs non_epithelial_rest",
        "log2fc_threshold"   : log2fc_threshold,
        "pval_adj_threshold" : pval_adj_threshold,
        "method"             : "wilcoxon",
        "n_malignant"        : int(adata_mal.n_obs),
        "n_rest"             : int(n_rest_cells),
        "n_genes_tested"     : int(adata_mal.var_names.shape[0]),
        "n_surfaceome_genes_total"      : int(len(surf_genes_set)),
        "n_surfaceome_genes_in_dataset" : int(len(surf_in_mal)),
        "n_filtered_deg"     : int(filtered_deg.shape[0]),
    }
    adata_mal.uns["qc_params"] = (
        {"min_genes": min_genes, "max_mt": max_mt} if qc_active else None
    )

    # CLAUDE EDIT — store Step 8b + final 3-way selection for traceability
    adata_mal.uns["cancer_composition_scores"] = (
        cc_df.reset_index(drop=True) if cc_df is not None else None
    )
    adata_mal.uns["cc_params"] = {
        "ran"                 : cc_df is not None,
        "healthy_reference"   : _cc_healthy_ref,
        "cc_score_threshold"  : cc_score_threshold,
        "n_cc_genes"          : int(len(cc_genes)) if cc_genes is not None else None,
        "umap_plot_paths"     : cc_plot_paths,
    }
    adata_mal.uns["final_gene_selection"] = {
        "sets_used"   : {label: int(len(gset)) for label, gset in selection_sets.items()},
        "n_final"     : int(len(final_gene_list)),
        "fell_back_to_surfaceome_only": len(final_gene_list) == len(surf_genes_set)
                                          and len(selection_sets) >= 2
                                          and final_gene_list == surf_genes_set,
    }

    if scevan_result_df is not None:
        final_barcodes = set(adata_mal.obs_names)
        scevan_stored  = scevan_result_df.copy()
        scevan_stored["in_final_output"] = scevan_stored["barcode"].isin(final_barcodes)
        adata_mal.uns["scevan_results"] = scevan_stored
        print(
            f"\nSCEVAN results stored in adata.uns['scevan_results']:\n"
            f"  Shape: {scevan_stored.shape[0]} rows × {scevan_stored.shape[1]} columns\n"
            f"  Columns: {list(scevan_stored.columns)}\n"
            f"  Prediction summary:\n"
            + scevan_stored["scevan_prediction"].value_counts().to_string()
            + f"\n  Cells in final output: {scevan_stored['in_final_output'].sum()}"
        )
    else:
        adata_mal.uns["scevan_results"] = None
        print("\nSCEVAN was not run — adata.uns['scevan_results'] = None")

    for col in adata_mal.obs.columns:
        if adata_mal.obs[col].dtype == object:
            adata_mal.obs[col] = adata_mal.obs[col].astype(str)

    final_path = os.path.join(save_dir, "final_tumor.h5ad")
    adata_mal.write(final_path)
    print(f"\nFinal object saved to: {final_path}")
    print("\n========== PREPROCESSING COMPLETED ==========\n")
    return adata_mal
