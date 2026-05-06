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
RAW COUNT HANDLING (input_type from Module 2)
═══════════════════════════════════════════════════════════════════════════════

Module 2 (PopV) can be run with two input_type modes:

  input_type='raw'   → all 8 methods run; Module 2 saves layers['counts']
                       in final_popv_annotated.h5ad (original integer counts
                       subsetted to 4000 HVGs, restored by popv_annotation.py).

  input_type='log1p' → only CELLTYPIST runs; .X was already log-normalised
                       when passed to Module 2; NO raw counts layer is saved
                       because Module 2 never had integer counts to snapshot.

This module handles both cases via _get_raw_counts_from_adata():

  Priority order for extracting raw counts from adata_epi / adata_rest:
    1. layers['counts']      ← written by updated popv_annotation.py (raw mode)
    2. layers['raw_counts']  ← alternate name
    3. layers['scvi_counts'] ← scVI internal counts (less preferred)
    4. adata.raw.X           ← legacy path
    5. adata.X               ← last resort; may be log1p — a WARNING is logged

  If only log1p data is available (input_type='log1p' path), steps that
  REQUIRE raw counts (QC, CNA, scMalignantFinder normalisation) will still
  work because _build_fullgene_adata_for_scm() reads from the Module 1
  tumor h5ad (Route A-rescue) which always has layers['counts'] with the
  original integer counts from GEO.

═══════════════════════════════════════════════════════════════════════════════
KEY FIXES
═══════════════════════════════════════════════════════════════════════════════

FIX 1   Full-gene route priority for scMalignantFinder.
FIX 2   SCEVAN reference subsampled to scevan_ref_max_cells (default 100).
FIX 3   DEG uses pvals_adj (BH-adjusted).
FIX 5   _get_raw_counts_from_adata prefers layers['counts'] → raw → .X.
FIX 6   scMalignantFinder predictions aligned by obs_names.
FIX 8   SCEVAN results saved in full detail in adata.uns['scevan_results'].
FIX 9   Correct DEG: malignant epithelial vs NON-EPITHELIAL cells.
FIX 10  SCEVAN runs via fresh Rscript subprocess.
        Rscript resolved via sys.executable bin dir FIRST — the only
        reliable signal inside a Jupyter kernel. CONDA_PREFIX and PATH
        are fallbacks only.
QC FIX  QC thresholds read from adata.uns['qc_params'] (set by Module 1).
        If absent, QC is SKIPPED ENTIRELY.
"""

import os
import sys
import glob
import shutil
import logging
import tempfile
import subprocess

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
        "GESP/GESP_surfaceome_gene.csv",
        "data/GESP_surfaceome_gene.csv",
        "resources/GESP_surfaceome_gene.csv",
    ):
        path = _find_scart_resource(candidate)
        if path is not None:
            return path
    raise FileNotFoundError(
        "Could not auto-detect GESP surfaceome CSV.\n"
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

    For the log1p mode (when PopV was run with input_type='log1p'), layers
    ['counts'] will be absent from the PopV output.  In that case Route 4/5
    is used here, and callers that need true raw counts should fall back to
    reading Module 1's GSE*_tumor.h5ad directly (done in
    _build_fullgene_adata_for_scm via Route A-rescue).
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
        f"{tag}No raw counts layer found — using adata.X. "
        "If PopV was run with input_type='log1p', this may be log-normalised. "
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
    Kept for backward-compat with CopyKAT/SCEVAN helpers that use float64.
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

    # Route A-new: use layers['counts'] from popv output (raw mode)
    # This is now the FIRST priority since popv_annotation.py saves
    # layers['counts'] with the original Module 1 raw counts (4000 HVGs).
    # If that overlap is too low, fall through to rescue routes.
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

    # Route C — last resort (4000 HVGs from .X — already log-normalised by Step 3)
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
    """
    Return the Rscript binary for the active conda environment.

    Priority:
      1. dirname(sys.executable)/Rscript  ← MOST reliable in Jupyter kernels
      2. $CONDA_PREFIX/bin/Rscript        ← fallback; unreliable in Jupyter
      3. shutil.which("Rscript")          ← PATH fallback, last resort
    """
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
# SCEVAN subprocess runner (replaces CopyKAT)
# ═══════════════════════════════════════════════════════════════════════════

def _run_scevan(
    adata_query,
    adata_ref,
    ref_epithelial_key="cell_ontology_class",
    ref_epithelial_values=None,
    ref_max_cells=100,
    sample_name="SCEVAN_run",
    organism="human",
    par_cores=1,
    subclones=False,
    batch_size=3000,
    save_dir=None,
):
    """
    Run SCEVAN via a fresh Rscript subprocess.

    Data flow (mirroring Input_SCEVAN.ipynb + SCEVAN.ipynb):
      1. Python: extract raw counts from query epithelial + normal ref cells
      2. Python: subset to common genes, combine, write genes×cells CSV
      3. Python: write normal barcodes list
      4. Python: write R driver script that runs pipelineCNA() in batches
      5. Rscript subprocess: run driver, write per-batch CSV + scevan_full_results.csv
      6. Python: read results, return per-cell prediction DataFrame

    Parameters
    ----------
    adata_query : AnnData
        Epithelial query cells (from adata_epi after QC).
        Raw counts must be recoverable via layers['counts'] / 'raw_counts'
        or adata.raw.X.
    adata_ref : AnnData
        Tabula Sapiens (or user) reference h5ad — same file used in Module 2.
    ref_epithelial_key : str
        obs column in adata_ref containing cell type labels.
    ref_epithelial_values : list of str
        Labels in ref_epithelial_key to use as normal reference.
    ref_max_cells : int
        Max normal reference cells to subsample (FIX 2).
    sample_name : str
        Prefix for SCEVAN output files.
    organism : str
        'human' or 'mouse'.
    par_cores : int
        Cores per batch passed to pipelineCNA().
    subclones : bool
        Whether to infer subclones.
    batch_size : int
        Query cells per batch (default 3000, matching SCEVAN.ipynb).
    save_dir : str or None
        Directory to write intermediate files and results.

    Returns
    -------
    pd.DataFrame  columns: barcode, scevan_prediction
                  values:  "tumor" | "normal" | "filtered" | "not.defined"
    """
    if ref_epithelial_values is None:
        ref_epithelial_values = ["epithelial cell", "glandular epithelial cell",
                                 "ovarian surface epithelial cell"]

    q_barcodes   = np.array(adata_query.obs_names)
    empty_result = pd.DataFrame({
        "barcode"           : list(q_barcodes),
        "scevan_prediction" : "not.defined",
    })

    # Locate Rscript
    rscript_bin = _find_rscript()
    if rscript_bin is None:
        logger.error("Rscript not found. SCEVAN skipped.")
        return empty_result

    r_home  = _get_r_home(rscript_bin)
    sub_env = _build_r_env(r_home)

    # Verify SCEVAN is installed
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

    # ── FIX 2: filter reference to normal epithelial cells + subsample ────
    if ref_epithelial_key in adata_ref.obs.columns:
        ref_vals_lower = [v.lower() for v in ref_epithelial_values]
        ep_mask        = adata_ref.obs[ref_epithelial_key].str.lower().isin(ref_vals_lower)
        adata_ref_ep   = adata_ref[ep_mask].copy() if ep_mask.any() else adata_ref.copy()
        logger.info(f"SCEVAN reference epithelial cells before subsample: {adata_ref_ep.n_obs}")
    else:
        logger.warning(
            f"ref_epithelial_key '{ref_epithelial_key}' not in adata_ref.obs — "
            "using full reference."
        )
        adata_ref_ep = adata_ref.copy()

    if adata_ref_ep.n_obs > ref_max_cells:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(adata_ref_ep.n_obs, size=ref_max_cells, replace=False)
        adata_ref_ep = adata_ref_ep[np.sort(idx)].copy()
        logger.info(f"SCEVAN reference subsampled to {ref_max_cells} cells.")

    if adata_ref_ep.n_obs == 0:
        logger.warning("SCEVAN: no reference cells after filtering. Skipping.")
        return empty_result

    # ── Extract raw counts ────────────────────────────────────────────────
    logger.info("SCEVAN: extracting raw counts from query and reference...")
    mat_query = _get_raw_counts_from_adata(adata_query, "SCEVAN-query").T   # genes × cells
    mat_ref   = _get_raw_counts_from_adata(adata_ref_ep, "SCEVAN-ref").T    # genes × cells

    q_genes = np.array(adata_query.var_names)
    r_genes = (
        np.array(adata_ref_ep.raw.var_names)
        if adata_ref_ep.raw is not None and adata_ref_ep.raw.n_vars >= mat_ref.shape[0]
        else np.array(adata_ref_ep.var_names)
    )
    if mat_ref.shape[0] != len(r_genes):
        r_genes = np.array(adata_ref_ep.var_names)
        mat_ref = _get_raw_counts_from_adata(adata_ref_ep, "SCEVAN-ref-retry").T

    common_genes = np.intersect1d(q_genes, r_genes)
    logger.info(f"SCEVAN common genes: {len(common_genes)}")

    if len(common_genes) < 200:
        raise ValueError(
            f"Only {len(common_genes)} common genes between query and reference. "
            "Need >= 200. Both datasets must use HGNC gene symbols."
        )

    q_idx = np.where(np.isin(q_genes, common_genes))[0]
    r_idx = np.where(np.isin(r_genes, common_genes))[0]

    # Normal reference barcodes (prefixed so they are identifiable)
    r_barcodes = np.array(["REF_" + b for b in adata_ref_ep.obs_names])
    q_barcodes_arr = np.array(adata_query.obs_names)

    # Combined genes × cells matrix (query + ref), matching Input_SCEVAN.ipynb
    mat_combined = np.hstack([
        mat_query[q_idx, :],
        mat_ref[r_idx, :],
    ])
    all_barcodes = np.concatenate([q_barcodes_arr, r_barcodes])

    logger.info(
        f"SCEVAN combined matrix: {mat_combined.shape[0]} genes × "
        f"{mat_combined.shape[1]} cells "
        f"({len(q_barcodes_arr)} query + {len(r_barcodes)} ref)"
    )

    if save_dir is None:
        save_dir = tempfile.mkdtemp(prefix="scart_scevan_")
        _tmpdir_created = True
    else:
        os.makedirs(save_dir, exist_ok=True)
        _tmpdir_created = False

    try:
        counts_csv    = os.path.join(save_dir, "scevan_counts.csv")
        norm_csv      = os.path.join(save_dir, "normal_barcodes.csv")
        driver_r      = os.path.join(save_dir, "run_scevan.R")
        results_csv   = os.path.join(save_dir, "scevan_full_results.csv")
        malignant_csv = os.path.join(save_dir, "scevan_malignant_cells.csv")

        # Write genes × cells count CSV (mirrors Input_SCEVAN.ipynb count_df)
        logger.info(f"SCEVAN: writing count matrix ({mat_combined.shape}) ...")
        count_df = pd.DataFrame(
            mat_combined,
            index=common_genes,
            columns=all_barcodes,
        )
        count_df.to_csv(counts_csv)
        logger.info(f"SCEVAN count matrix written: {counts_csv}")

        # Write normal barcodes (mirrors normal_barcodes.csv in Input_SCEVAN.ipynb)
        pd.Series(r_barcodes.tolist()).to_csv(norm_csv, index=False, header=False)
        logger.info(f"SCEVAN normal barcodes written: {norm_csv} ({len(r_barcodes)} cells)")

        # Write R driver script (mirrors SCEVAN.ipynb logic with batch processing)
        subclones_r   = "TRUE"  if subclones   else "FALSE"
        fixed_norm_r  = "TRUE"

        with open(driver_r, "w") as f:
            f.write(f"""\
suppressPackageStartupMessages({{
  library(SCEVAN)
  library(Matrix)
}})

# ── Patch classifyTumorCells to use lapply instead of parLapply ─────────
original_fn <- get("classifyTumorCells", envir = asNamespace("SCEVAN"))
modified_fn <- original_fn
body_text   <- deparse(body(original_fn))
body_text   <- gsub("parallel::parLapply\\\\(cl,", "lapply(", body_text)
body_text   <- gsub("parLapply\\\\(cl,", "lapply(", body_text)
new_body    <- parse(text = paste(body_text, collapse = "\\n"))
body(modified_fn) <- as.call(c(as.name("{{"), new_body))
environment(modified_fn) <- asNamespace("SCEVAN")
assignInNamespace("classifyTumorCells", modified_fn, "SCEVAN")

# ── Load data ────────────────────────────────────────────────────────────
cat("Loading count matrix...\\n")
count_mat  <- read.csv("{counts_csv}", row.names = 1, check.names = FALSE)
count_mat  <- as.matrix(count_mat)
cat("Matrix dims:", nrow(count_mat), "genes x", ncol(count_mat), "cells\\n")

normal_cells <- readLines("{norm_csv}")
cat("Normal reference cells:", length(normal_cells), "\\n")

# ── Batch setup (mirrors SCEVAN.ipynb) ───────────────────────────────────
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

    # Keep only query cells (exclude normal ref rows)
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

# ── Combine all batches ───────────────────────────────────────────────────
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

        # ── Parse results ─────────────────────────────────────────────────
        pred_df = pd.read_csv(results_csv, index_col=0)
        logger.info(f"SCEVAN results columns: {list(pred_df.columns)}")

        # 'class' column contains 'tumor' / 'normal' / 'filtered'
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

        # Left-join to ensure all query cells are represented
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
    # SCEVAN parameters (replaces CopyKAT)
    scevan_ref_epithelial_key="cell_ontology_class",
    scevan_ref_epithelial_values=None,
    scevan_ref_max_cells=100,
    scevan_sample_name="SCEVAN_run",
    scevan_organism="human",
    scevan_par_cores=1,
    scevan_subclones=False,
    scevan_batch_size=3000,
):
    """
    Full preprocessing pipeline — see module docstring for details.

    QC thresholds (min_genes, max_mt) are read from adata.uns['qc_params']
    written by Module 1.  If absent, QC is skipped entirely.

    Raw counts handling:
      - If PopV was run with input_type='raw', layers['counts'] is present
        in final_popv_annotated.h5ad and is used automatically.
      - If PopV was run with input_type='log1p', layers['counts'] is absent.
        _get_raw_counts_from_adata() will fall through to adata.raw or .X
        (with a warning), and scMalignantFinder will use Route A-rescue
        (Module 1 tumor h5ad) for full-gene raw counts.
      - SCEVAN always needs raw integer counts; if only log1p data is
        available, the pipeline prints a warning and SCEVAN may produce
        unreliable CNV calls.

    malignant_strategy: 'intersection' | 'scMalignant' | 'scevan'

    SCEVAN runs via Rscript subprocess using the same logic as SCEVAN.ipynb
    (batched pipelineCNA calls with normal reference cells included in each
    batch, classifyTumorCells patched to use lapply instead of parLapply).
    """
    if scevan_ref_epithelial_values is None:
        scevan_ref_epithelial_values = [
            "epithelial cell",
            "glandular epithelial cell",
            "ovarian surface epithelial cell",
        ]

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

    # Report what raw count source is available
    _raw_layers_present = [
        l for l in ("counts", "raw_counts", "scvi_counts")
        if l in adata_full.layers
    ]
    if _raw_layers_present:
        print(f"Raw count layers available: {_raw_layers_present} "
              f"(PopV was run with input_type='raw')")
    else:
        print(
            "WARNING: No raw count layers found in PopV output.\n"
            "  PopV was likely run with input_type='log1p'.\n"
            "  Steps requiring raw counts (QC, SCEVAN, scMalignantFinder)\n"
            "  will attempt Route A-rescue from the Module 1 tumor h5ad.\n"
            f"  Provide tumor_h5ad= explicitly or place GSE*_tumor.h5ad in cwd."
        )

    # STEP 2 — Read QC thresholds
    min_genes, max_mt, qc_active, qc_source = _read_qc_params(adata_full)

    # STEP 3 — Extract epithelial cells → QC
    print("\n--- Step 3: Epithelial selection" +
          (" + QC ---" if qc_active else " ---"))

    labels  = adata_full.obs["popv_majority_vote_prediction"].astype(str)
    ep_mask = labels.str.endswith("epithelial cell")
    print(f"Epithelial cells: {ep_mask.sum()} / {adata_full.n_obs} total")
    print(f"Non-epithelial cells (will be 'rest' group for DEG): {(~ep_mask).sum()}")

    adata_epi = adata_full[ep_mask].copy()
    before_qc = adata_epi.n_obs

    if qc_active:
        # QC requires raw counts: try layers['counts'] first, fall back to .X
        # Note: sc.pp.calculate_qc_metrics uses .X, so we temporarily set
        # .X to raw counts for accurate n_genes_by_counts / pct_counts_mt.
        _raw_for_qc = _get_raw_counts_from_adata(adata_epi, "QC")
        _X_backup   = adata_epi.X.copy()
        adata_epi.X = sp.csr_matrix(_raw_for_qc)

        adata_epi.var["mt"] = adata_epi.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata_epi, qc_vars=["mt"], inplace=True)
        print(f"Mean MT% BEFORE QC: {adata_epi.obs['pct_counts_mt'].mean():.2f}")

        # Restore .X after metrics are computed
        adata_epi.X = _X_backup

        filters = np.ones(adata_epi.n_obs, dtype=bool)
        if min_genes is not None:
            filters &= adata_epi.obs["n_genes_by_counts"] > min_genes
        if max_mt is not None:
            filters &= adata_epi.obs["pct_counts_mt"] < max_mt
        adata_epi = adata_epi[filters].copy()
        filter_desc = "  ".join(
            ([f"min_genes>{min_genes}"] if min_genes is not None else []) +
            ([f"max_mt<{max_mt}"]       if max_mt    is not None else [])
        )
        print(f"Epithelial cells after QC: {adata_epi.n_obs}  "
              f"(removed {before_qc - adata_epi.n_obs}  |  {filter_desc})")
        print(f"Mean MT% AFTER QC:  {adata_epi.obs['pct_counts_mt'].mean():.2f}\n")
    else:
        print(f"QC filtering SKIPPED — all {adata_epi.n_obs} epithelial cells proceed.\n")

    # Set .X to raw counts for downstream tools that need them (SCEVAN, scMalignantFinder)
    # layers['raw_for_cna'] stores integer counts before normalisation.
    print("Setting up raw counts for epithelial cells...")
    _raw_epi = _get_raw_counts_from_adata(adata_epi, "epithelial-raw-setup")
    adata_epi.X                  = sp.csr_matrix(_raw_epi)
    adata_epi.layers["raw_for_cna"] = adata_epi.X.copy()
    print(f"  layers['raw_for_cna'] stored ({adata_epi.n_obs} × {adata_epi.n_vars})")

    # Now normalise .X for scMalignantFinder (Route C fallback uses log1p .X)
    adata_epi.var_names_make_unique()
    sc.pp.normalize_total(adata_epi, target_sum=1e4)
    sc.pp.log1p(adata_epi)

    # STEP 4a — scMalignantFinder
    print("\n--- Step 4a: scMalignantFinder ---")
    feature_tsv = os.path.join(scmalignant_model_dir, "ordered_feature.tsv")

    # Report which gene-space route will be used
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
            print("Gene-space: Route C (4000-HVG fallback — low overlap expected)")

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

    # norm_type=False: adata_scm already log-normalised by _build_fullgene_adata_for_scm
    model = _clf_mod.scMalignantFinder(
        test_input          = adata_scm,
        celltype_annotation = False,
        pretrain_dir        = scmalignant_model_dir,
        feature_path        = feature_tsv,
        norm_type           = False,
    )
    model.load()
    result_scm = model.predict()

    # FIX 6 — align by obs_names
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

    # STEP 4b — SCEVAN (replaces CopyKAT)
    scevan_available  = False
    scevan_result_df  = None

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
                f"  Reference h5ad:       {reference_h5ad}\n"
                f"  ref_epithelial_key:   {scevan_ref_epithelial_key}\n"
                f"  ref_epithelial_vals:  {scevan_ref_epithelial_values}\n"
                f"  ref_max_cells:        {scevan_ref_max_cells}\n"
                f"  sample_name:          {scevan_sample_name}\n"
                f"  organism:             {scevan_organism}\n"
                f"  par_cores:            {scevan_par_cores}\n"
                f"  subclones:            {scevan_subclones}\n"
                f"  batch_size:           {scevan_batch_size}\n"
                f"  Rscript:              {_rs}\n"
                f"  R home:               {_rh}"
            )

            # Warn if raw counts may not be available for SCEVAN
            if not _raw_layers_present:
                print(
                    "\n  WARNING: No raw count layers in PopV output "
                    "(input_type='log1p' path).\n"
                    "  SCEVAN requires integer raw counts for reliable CNV inference.\n"
                    "  The pipeline will attempt to use adata.raw or adata.X — "
                    "results may be unreliable."
                )

            try:
                # Use raw counts for SCEVAN (layers['raw_for_cna'] = integer counts)
                adata_raw_cna   = adata_epi.copy()
                adata_raw_cna.X = adata_epi.layers["raw_for_cna"]
                adata_ref_full  = sc.read_h5ad(reference_h5ad)

                # SCEVAN intermediate files go into save_dir/scevan/
                scevan_work_dir = os.path.join(save_dir, "scevan")
                os.makedirs(scevan_work_dir, exist_ok=True)

                scevan_result_df = _run_scevan(
                    adata_query             = adata_raw_cna,
                    adata_ref               = adata_ref_full,
                    ref_epithelial_key      = scevan_ref_epithelial_key,
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
        # SCEVAN labels tumor cells as "tumor"
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

    # STEP 6 — Non-epithelial "rest" group
    print("\n--- Step 6: Extract non-epithelial 'rest' group ---")
    rest_mask  = ~ep_mask
    adata_rest = adata_full[rest_mask].copy()
    print(f"Non-epithelial 'rest' cells: {adata_rest.n_obs}")

    # Set .X to raw counts then normalise for DEG
    _raw_rest = _get_raw_counts_from_adata(adata_rest, "rest-group")
    adata_rest.X = sp.csr_matrix(_raw_rest)
    sc.pp.normalize_total(adata_rest, target_sum=1e4)
    sc.pp.log1p(adata_rest)

    # STEP 7 — Surfaceome filter
    print("\n--- Step 7: Surfaceome filter (GESP file) ---")
    surfaceome        = pd.read_csv(surfaceome_path)
    surfaceome.columns = surfaceome.columns.str.strip()
    surf_genes        = surfaceome["Gene"].astype(str).tolist()
    print(f"Surfaceome genes in GESP file: {len(surf_genes)}")

    surf_in_mal  = adata_mal.var_names.intersection(surf_genes)
    adata_mal    = adata_mal[:, surf_in_mal].copy()
    print(f"Surfaceome genes in malignant cells: {len(surf_in_mal)}")

    surf_in_rest = adata_rest.var_names.intersection(surf_genes)
    adata_rest   = adata_rest[:, surf_in_rest].copy()
    print(f"Surfaceome genes in rest cells: {len(surf_in_rest)}")

    surf_common = surf_in_mal.intersection(surf_in_rest)
    adata_mal   = adata_mal[:, surf_common].copy()
    adata_rest  = adata_rest[:, surf_common].copy()
    print(f"Common surfaceome genes (used for DEG): {len(surf_common)}\n")

    # STEP 8 — DEG
    print("--- Step 8: DEG — malignant epithelial vs non-epithelial rest ---")

    adata_mal.obs["deg_group"]  = "malignant_epithelial"
    adata_rest.obs["deg_group"] = "non_epithelial_rest"

    adata_deg = sc.concat([adata_mal, adata_rest], join="outer", label=None)
    adata_deg.obs_names_make_unique()
    adata_deg.var = adata_mal.var.copy()

    if sp.issparse(adata_deg.X):
        adata_deg.X = adata_deg.X.toarray()
    adata_deg.X = np.nan_to_num(np.array(adata_deg.X, dtype=np.float32), nan=0.0)
    adata_deg.X = sp.csr_matrix(adata_deg.X)

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

    # STEP 9 — Binarise, store, save
    print("\n--- Step 9: Binarise malignant cells and save ---")

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
        "n_rest"             : int(adata_rest.n_obs),
        "n_surfaceome_genes" : int(len(surf_common)),
        "n_filtered_deg"     : int(filtered_deg.shape[0]),
    }
    adata_mal.uns["qc_params"] = (
        {"min_genes": min_genes, "max_mt": max_mt} if qc_active else None
    )

    # FIX 8 — store SCEVAN results in full detail
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
