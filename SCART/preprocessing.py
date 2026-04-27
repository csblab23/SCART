"""
preprocessing.py
Module 3 — Preprocessing, malignancy detection, and surfaceome DEG

═══════════════════════════════════════════════════════════════════════════════
CORRECT BIOLOGICAL PIPELINE DESIGN
═══════════════════════════════════════════════════════════════════════════════

Step 1  Load the full PopV-annotated h5ad (all cell types, e.g. 15202 cells).
        Save this as adata_full — it is the complete dataset.

Step 2  Extract epithelial cells only from adata_full → apply QC filters
        (only if QC thresholds were set in Module 1).
        QC thresholds (min_genes, max_mt) are read automatically from
        adata.uns['qc_params'] written by Module 1 (geo_fetcher.py).
        If 'qc_params' is absent (user did not set thresholds), QC is
        skipped entirely and all epithelial cells proceed.

Step 3  Run scMalignantFinder + CopyKAT on the epithelial cells.
        Combine predictions → final_malignant column on the epithelial adata.
        Strategy: intersection (scMalignantFinder AND CopyKAT must both agree).

Step 4  Keep ONLY the malignant epithelial cells (drop non-malignant epithelial).
        These are the true tumour cells.

Step 5  From adata_full, take all NON-EPITHELIAL cells as the "rest" group.
        Apply the same surfaceome gene filter to both groups.

Step 6  DEG:  malignant epithelial cells  vs  all non-epithelial cells
        (both filtered to surfaceome genes from GESP file)
        This is the correct comparison: tumour surface markers vs stromal/immune.

Step 7  Binarise ONLY the malignant epithelial AnnData, store DEG results, save.

═══════════════════════════════════════════════════════════════════════════════
QC PARAMETER FLOW
═══════════════════════════════════════════════════════════════════════════════

Module 1 (geo_fetcher.py)
  SampleAnnotator("GSE…")                        → QC disabled, key absent
  SampleAnnotator("GSE…", min_genes=200)          → only gene count filter
  SampleAnnotator("GSE…", max_mt=40)              → only MT filter
  SampleAnnotator("GSE…", min_genes=200, max_mt=40) → both filters active

Module 2 (popv_annotation.py)
  Passes the h5ad through — uns keys including 'qc_params' are preserved.

Module 3 (this file)
  Reads adata.uns['qc_params'] automatically.
  If the key is ABSENT  → QC step is SKIPPED (all epithelial cells kept).
  If the key is PRESENT → only the thresholds that are not None are applied.
  Logs clearly which path was taken.

═══════════════════════════════════════════════════════════════════════════════
DATA FLOW ACROSS MODULES
═══════════════════════════════════════════════════════════════════════════════

Module 1 (geo_fetcher.py)
  Saves:  GSE*_tumor.h5ad
  Contains: adata.layers['counts']          raw integer counts, full gene space
            adata.raw = adata               same data frozen
            adata.uns['cancer_type']
            adata.uns['qc_params']          only present when user set thresholds

Module 2 (popv_annotation.py)
  Reads:  GSE*_tumor.h5ad
  Saves:  popv_results/final_popv_annotated.h5ad
  Contains (after FIX 8 layer approach):
            adata.layers['full_counts']          raw counts, full gene space
            adata.uns['full_counts_var_names']   list of gene names
            adata.uns['qc_params']               passed through from Module 1
            adata.layers['scvi_counts']          4000 HVG subset
            obs['popv_majority_vote_prediction'] cell-type labels

Module 3 (this file)
  Reads:  popv_results/final_popv_annotated.h5ad  (auto-detected)
  Saves:  preprocessing_results/final_tumor.h5ad
            → malignant epithelial cells only
            → obs: scMalignantFinder_prediction, copykat_prediction,
                   copykat_cna_signal, copykat_gmm_label, final_malignant
            → uns: copykat_results (full detail DataFrame)
                   filtered_deg, all_deg, deg_params, cancer_type, qc_params

═══════════════════════════════════════════════════════════════════════════════
KEY FIXES
═══════════════════════════════════════════════════════════════════════════════

FIX 1   Full-gene route priority for scMalignantFinder:
        A-new (layers['full_counts']) → A-rescue (auto-detect Module 1 h5ad)
        → A-old (adata.raw) → C (4000-HVG fallback)

FIX 2   CopyKAT reference subsampled to copykat_ref_max_cells (default 100).

FIX 3   DEG uses pvals_adj (BH-adjusted), not raw pvals.

FIX 5   _get_raw_matrix prefers adata.raw over HVG layers (more genes).

FIX 6   scMalignantFinder predictions aligned by obs_names, not positional.

FIX 8   CopyKAT results saved in full detail:
        Per-cell: cna_signal, gmm_label stored as obs columns.
        Full table: adata.uns['copykat_results'] DataFrame.

FIX 9   CORRECT DEG DESIGN:
        Malignant epithelial cells vs all NON-EPITHELIAL cells from adata_full.
        Both groups surfaceome-filtered before DEG.
        Non-malignant epithelial cells are REMOVED (not used in any group).

QC FIX  min_genes and max_mt removed from the public API of
        run_preprocessing_pipeline().  Both values are read from
        adata.uns['qc_params'] (written by Module 1).
        If the key is absent the QC step is SKIPPED ENTIRELY — no defaults
        are applied.  This makes the user's choice in Module 1 authoritative.
"""

import os
import glob
import logging

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
    """
    Read QC thresholds from adata.uns['qc_params'] (written by Module 1).

    Returns
    -------
    min_genes : int or None
        None  → do not apply a gene-count filter.
    max_mt    : float or None
        None  → do not apply an MT-percentage filter.
    qc_active : bool
        True  → at least one filter will be applied.
        False → QC step is skipped entirely.
    source    : str
        Human-readable description of where the values came from.
    """
    qc = adata.uns.get("qc_params", None)

    if qc is None:
        return (
            None, None, False,
            "SKIPPED — 'qc_params' not found in adata.uns. "
            "Re-run Module 1 with SampleAnnotator(min_genes=…, max_mt=…) "
            "to enable QC filtering."
        )

    min_genes = qc.get("min_genes", None)
    max_mt    = qc.get("max_mt",    None)

    # Normalise types if values are present
    if min_genes is not None:
        min_genes = int(min_genes)
    if max_mt is not None:
        max_mt = float(max_mt)

    qc_active = (min_genes is not None) or (max_mt is not None)

    if qc_active:
        parts = []
        if min_genes is not None:
            parts.append(f"min_genes={min_genes}")
        if max_mt is not None:
            parts.append(f"max_mt={max_mt}")
        source = (
            "adata.uns['qc_params'] (set by Module 1) — "
            + ", ".join(parts)
        )
    else:
        source = (
            "'qc_params' key present but both values are None — QC SKIPPED."
        )

    return min_genes, max_mt, qc_active, source


# ═══════════════════════════════════════════════════════════════════════════
# FIX 5 — raw count extractor (adata.raw preferred over HVG layers)
# ═══════════════════════════════════════════════════════════════════════════

def _get_raw_matrix(adata):
    """
    Return dense float64 (cells × genes) raw count matrix.
    Prefers adata.raw when it has more genes than any layer.
    """
    if adata.raw is not None:
        raw_n   = adata.raw.n_vars
        layer_n = max(
            (adata.layers[l].shape[1]
             for l in ("full_counts", "scvi_counts", "raw_counts", "counts")
             if l in adata.layers),
            default=0,
        )
        if raw_n > layer_n:
            logger.info(f"Raw counts: adata.raw ({raw_n} genes > best layer {layer_n})")
            X = adata.raw.X
            if sp.issparse(X):
                X = X.toarray()
            return np.array(X, dtype=np.float64)

    for lyr in ("full_counts", "scvi_counts", "raw_counts", "counts"):
        if lyr in adata.layers:
            logger.info(f"Raw counts: layers['{lyr}']")
            X = adata.layers[lyr]
            if sp.issparse(X):
                X = X.toarray()
            return np.array(X, dtype=np.float64)

    src = adata.raw.X if adata.raw is not None else adata.X
    logger.warning("Using adata.X as raw counts — may be log-normalised.")
    if sp.issparse(src):
        src = src.toarray()
    return np.array(src, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# FIX 1 — full-gene AnnData for scMalignantFinder
# ═══════════════════════════════════════════════════════════════════════════

def _build_fullgene_adata_for_scm(adata, feature_tsv, tumor_h5ad_path=None):
    """
    Return a log-normalised AnnData covering the full original gene space.

    Route priority:
      A-new     layers['full_counts'] + uns['full_counts_var_names']
      A-rescue  Module 1 tumor h5ad (auto-detected or explicit)
      A-old     adata.raw
      B         uns['full_var_names']
      C         4000-HVG fallback (last resort)
    """
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

    # Route A-new
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

    # Route A-rescue
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
                        af    = _make_adata(
                            sub.layers[lyr], adata.obs.loc[order], sub.var
                        )
                        logger.info(
                            f"scMalignantFinder → Route A-rescue (layers['{lyr}'])."
                        )
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

    # Route C — last resort
    ov_hvg = _pct(adata.var_names)
    logger.warning(
        f"All routes failed. Falling back to {adata.n_vars} HVGs ({ov_hvg:.1f}% overlap).\n"
        "Place GSE*_tumor.h5ad in cwd or re-run Module 2 with FIX 8."
    )
    return adata.copy()


# ═══════════════════════════════════════════════════════════════════════════
# CopyKAT runner
# ═══════════════════════════════════════════════════════════════════════════

def _run_copykat(
    adata_query,
    adata_ref,
    genome="hg20",
    id_type="S",
    ngene_chr=10,
    win_size=50,
    ks_cut=0.1,
    distance="euclidean",
    n_cores=2,
    plot_genes=True,
    output_seg=False,
    ref_max_cells=100,
    sam_name="copykat_run",
    ref_epithelial_key="cell_ontology_class",
    ref_epithelial_values=None,
):
    """
    Run CopyKAT via rpy2 to identify aneuploid (malignant) cells.

    Mirrors the sample workflow from copykat_sample_code.odt:
      1. Filter reference to epithelial cells (normal reference).
      2. Subsample reference to ref_max_cells (FIX 2).
      3. Extract raw counts from both query and reference.
      4. Find common genes; build combined genes × cells matrix.
      5. Prefix reference barcodes with "REF_" to avoid collision.
      6. Run copykat() in R with user-controlled parameters.
      7. Parse aneuploid.pred output → per-cell prediction DataFrame.

    Parameters
    ----------
    adata_query : AnnData
        Epithelial cells (query / tumour candidates).
    adata_ref : AnnData
        Tabula Sapiens (or other normal reference) h5ad.
    genome : str
        Genome version passed to copykat(). Default "hg20".
    id_type : str
        Gene ID type: "S" (symbol) or "E" (Ensembl). Default "S".
    ngene_chr : int
        Minimum genes per chromosome. Default 10.
    win_size : int
        Sliding window size for smoothing. Default 50.
    ks_cut : float
        KS statistic cut-off for classifying aneuploid vs diploid. Default 0.1.
    distance : str
        Distance metric for hierarchical clustering. Default "euclidean".
    n_cores : int
        Number of parallel cores for copykat. Default 2.
    plot_genes : bool
        Whether copykat should produce gene-level plots. Default True.
    output_seg : bool
        Whether copykat should write segment files. Default False.
    ref_max_cells : int
        Maximum normal reference cells (FIX 2). Default 100.
    sam_name : str
        Sample name prefix for copykat output files. Default "copykat_run".
    ref_epithelial_key : str
        obs column in adata_ref used to identify normal epithelial cells.
        Default "cell_ontology_class".
    ref_epithelial_values : list of str or None
        Values in ref_epithelial_key that mark normal epithelial cells.
        Default ["epithelial cell"].

    Returns
    -------
    pd.DataFrame  columns: barcode, copykat_prediction
                           (values: "aneuploid" | "diploid" | "not.defined")
                  All query barcodes are present; missing ones get "not.defined".
    """
    if ref_epithelial_values is None:
        ref_epithelial_values = ["epithelial cell"]

    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        err = str(exc)
        if any(s in err for s in ("R_getVar", "undefined symbol", "R_ClosureEnv")):
            raise ImportError(
                "rpy2/R version mismatch.\n"
                "Fix:\n  conda activate scart\n"
                "  conda remove rpy2 --force\n"
                "  conda install -c conda-forge rpy2\n"
                f"Original: {err}"
            ) from exc
        raise

    try:
        importr("copykat")
    except Exception as exc:
        raise ImportError(
            "R package 'copykat' not found.\n"
            "In R: devtools::install_github('navinlabcode/copykat')"
        ) from exc

    q_barcodes = np.array(adata_query.obs_names)

    # Pre-build safe fallback DataFrame returned on any critical failure
    empty_result = pd.DataFrame({
        "barcode"            : list(q_barcodes),
        "copykat_prediction" : "not.defined",
    })

    # ------------------------------------------------------------------
    # FIX 2 — filter reference to normal epithelial cells and subsample
    # ------------------------------------------------------------------
    if ref_epithelial_key in adata_ref.obs.columns:
        ref_vals_lower = [v.lower() for v in ref_epithelial_values]
        ep_mask = adata_ref.obs[ref_epithelial_key].str.lower().isin(ref_vals_lower)
        adata_ref_ep = adata_ref[ep_mask].copy() if ep_mask.any() else adata_ref.copy()
        logger.info(
            f"CopyKAT reference epithelial cells: {adata_ref_ep.n_obs} "
            f"(key='{ref_epithelial_key}', values={ref_epithelial_values})"
        )
    else:
        logger.warning(
            f"ref_epithelial_key '{ref_epithelial_key}' not found in reference obs. "
            "Using full reference as normal cells."
        )
        adata_ref_ep = adata_ref.copy()

    if adata_ref_ep.n_obs > ref_max_cells:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(adata_ref_ep.n_obs, size=ref_max_cells, replace=False)
        adata_ref_ep = adata_ref_ep[np.sort(idx)].copy()
        logger.info(f"CopyKAT reference subsampled to {ref_max_cells} cells.")

    if adata_ref_ep.n_obs == 0:
        logger.warning("CopyKAT: no reference cells found after filtering. Skipping.")
        return empty_result

    # ------------------------------------------------------------------
    # Extract raw integer count matrices  (cells × genes → transpose → genes × cells)
    # ------------------------------------------------------------------
    logger.info("CopyKAT: extracting raw counts from query ...")
    mat_query = _get_raw_matrix(adata_query)      # cells × genes
    mat_query = mat_query.T                        # genes × cells

    logger.info("CopyKAT: extracting raw counts from reference ...")
    mat_ref = _get_raw_matrix(adata_ref_ep)        # cells × genes
    mat_ref = mat_ref.T                            # genes × cells

    # Gene names
    q_genes = np.array(adata_query.var_names)
    r_genes = (
        np.array(adata_ref_ep.raw.var_names)
        if adata_ref_ep.raw is not None
        and adata_ref_ep.raw.n_vars > max(
            (adata_ref_ep.layers[l].shape[1]
             for l in ("scvi_counts", "raw_counts", "counts")
             if l in adata_ref_ep.layers),
            default=0,
        )
        else np.array(adata_ref_ep.var_names)
    )

    # Align mat_ref rows to r_genes if raw was used
    if adata_ref_ep.raw is not None and adata_ref_ep.raw.n_vars == mat_ref.shape[0]:
        pass   # already aligned
    # If _get_raw_matrix returned adata.raw but r_genes is var_names, realign
    if mat_ref.shape[0] != len(r_genes):
        logger.warning(
            f"CopyKAT: ref matrix rows ({mat_ref.shape[0]}) != r_genes ({len(r_genes)}). "
            "Falling back to adata_ref_ep.var_names for gene labelling."
        )
        r_genes = np.array(adata_ref_ep.var_names)
        mat_ref_raw = _get_raw_matrix(adata_ref_ep)
        mat_ref = mat_ref_raw.T

    # ------------------------------------------------------------------
    # Common genes
    # ------------------------------------------------------------------
    common_genes = np.intersect1d(q_genes, r_genes)
    n_common     = len(common_genes)
    logger.info(f"CopyKAT common genes: {n_common}")

    if n_common < 200:
        raise ValueError(
            f"Only {n_common} common genes between query and reference. Need >= 200.\n"
            "Both datasets must use HGNC gene symbols."
        )
    if n_common < 2000:
        logger.warning(f"Only {n_common} common genes — CopyKAT prefers a larger gene set.")

    q_idx = np.where(np.isin(q_genes,  common_genes))[0]
    r_idx = np.where(np.isin(r_genes,  common_genes))[0]

    mat_query_sub = mat_query[q_idx, :]   # common_genes × query_cells
    mat_ref_sub   = mat_ref[r_idx,   :]   # common_genes × ref_cells

    # Prefix reference barcodes to avoid barcode collision
    q_barcodes_sub = np.array(adata_query.obs_names)
    r_barcodes_sub = np.array(["REF_" + b for b in adata_ref_ep.obs_names])

    mat_combined  = np.hstack([mat_query_sub, mat_ref_sub])
    all_barcodes  = np.concatenate([q_barcodes_sub, r_barcodes_sub])
    normal_cells  = r_barcodes_sub.tolist()

    logger.info(
        f"CopyKAT combined matrix: {mat_combined.shape[0]} genes × "
        f"{mat_combined.shape[1]} cells "
        f"({len(q_barcodes_sub)} query + {len(r_barcodes_sub)} ref)"
    )

    # ------------------------------------------------------------------
    # Transfer to R and run copykat()
    # ------------------------------------------------------------------
    logger.info("CopyKAT: transferring matrix to R ...")

    # Flatten column-major (Fortran order) to match R matrix layout
    ro.globalenv["ck_mat_flat"]    = ro.FloatVector(mat_combined.flatten(order="F").tolist())
    ro.globalenv["ck_n_rows"]      = ro.IntVector([mat_combined.shape[0]])
    ro.globalenv["ck_n_cols"]      = ro.IntVector([mat_combined.shape[1]])
    ro.globalenv["ck_gene_names"]  = ro.StrVector(common_genes.tolist())
    ro.globalenv["ck_barcodes"]    = ro.StrVector(all_barcodes.tolist())
    ro.globalenv["ck_norm_cells"]  = ro.StrVector(normal_cells)
    ro.globalenv["ck_id_type"]     = ro.StrVector([id_type])
    ro.globalenv["ck_ngene_chr"]   = ro.IntVector([ngene_chr])
    ro.globalenv["ck_win_size"]    = ro.IntVector([win_size])
    ro.globalenv["ck_ks_cut"]      = ro.FloatVector([ks_cut])
    ro.globalenv["ck_distance"]    = ro.StrVector([distance])
    ro.globalenv["ck_n_cores"]     = ro.IntVector([n_cores])
    ro.globalenv["ck_plot_genes"]  = ro.StrVector(["TRUE" if plot_genes  else "FALSE"])
    ro.globalenv["ck_output_seg"]  = ro.StrVector(["TRUE" if output_seg  else "FALSE"])
    ro.globalenv["ck_genome"]      = ro.StrVector([genome])
    ro.globalenv["ck_sam_name"]    = ro.StrVector([sam_name])

    logger.info("CopyKAT: running copykat() in R ...")
    try:
        ro.r("""
            suppressPackageStartupMessages(library(copykat))

            r_mat <- matrix(ck_mat_flat, nrow = ck_n_rows, ncol = ck_n_cols)
            rownames(r_mat) <- ck_gene_names
            colnames(r_mat) <- ck_barcodes

            copykat.result <- copykat(
                rawmat          = r_mat,
                id.type         = ck_id_type,
                ngene.chr       = ck_ngene_chr,
                win.size        = ck_win_size,
                KS.cut          = ck_ks_cut,
                sam.name        = ck_sam_name,
                distance        = ck_distance,
                norm.cell.names = ck_norm_cells,
                output.seg      = ck_output_seg,
                plot.genes      = ck_plot_genes,
                genome          = ck_genome,
                n.cores         = ck_n_cores
            )
        """)
    except Exception as exc:
        logger.error(f"CopyKAT R execution failed: {exc}")
        return empty_result

    # ------------------------------------------------------------------
    # Parse copykat prediction output from R
    # ------------------------------------------------------------------
    try:
        ro.r("""
            ck_pred <- copykat.result$prediction
        """)
        ck_pred_r = ro.globalenv["ck_pred"]

        # Convert R data.frame to pandas
        import rpy2.robjects as ro2
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        pred_df = pandas2ri.rpy2py(ck_pred_r)

        logger.info(f"CopyKAT raw prediction columns: {list(pred_df.columns)}")
        logger.info(
            "CopyKAT raw prediction counts:\n"
            + pred_df["copykat.pred"].value_counts().to_string()
            if "copykat.pred" in pred_df.columns
            else str(pred_df.head())
        )

        # Normalise column names — copykat may use 'cell.names' or the index
        if "cell.names" in pred_df.columns:
            barcode_col = "cell.names"
        elif "barcodes" in pred_df.columns:
            barcode_col = "barcodes"
        else:
            # Fall back: use the DataFrame index as barcodes
            pred_df = pred_df.reset_index()
            pred_df = pred_df.rename(columns={"index": "cell.names"})
            barcode_col = "cell.names"

        pred_col = "copykat.pred" if "copykat.pred" in pred_df.columns else pred_df.columns[-1]

        # Map copykat labels: "aneuploid" → "aneuploid", "diploid" → "diploid"
        # Filter out REF_ prefixed barcodes — those are normal reference cells
        pred_df = pred_df[~pred_df[barcode_col].str.startswith("REF_")].copy()
        pred_df = pred_df.rename(columns={
            barcode_col : "barcode",
            pred_col    : "copykat_prediction",
        })[["barcode", "copykat_prediction"]]
        pred_df["copykat_prediction"] = (
            pred_df["copykat_prediction"]
            .str.strip()
            .str.lower()
            .fillna("not.defined")
        )

        # Align to all original query barcodes (left-merge keeps order)
        result_full = pd.DataFrame({"barcode": list(q_barcodes)})
        result_full = result_full.merge(pred_df, on="barcode", how="left")
        result_full["copykat_prediction"] = (
            result_full["copykat_prediction"].fillna("not.defined")
        )

        logger.info(
            "CopyKAT predictions (query cells only):\n"
            + result_full["copykat_prediction"].value_counts().to_string()
        )
        return result_full

    except Exception as exc:
        logger.error(f"CopyKAT result parsing failed: {exc}")
        return empty_result


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_preprocessing_pipeline(
    adata=None,
    popv_path=None,
    # DEG
    log2fc_threshold=1.0,
    pval_adj_threshold=0.05,
    # paths
    reference_h5ad=None,
    tumor_h5ad=None,
    save_dir=None,
    scmalignant_model_dir=None,
    surfaceome_path=None,
    # malignancy
    malignant_strategy="intersection",
    # CopyKAT parameters (all user-configurable from the calling script)
    copykat_genome="hg20",
    copykat_id_type="S",
    copykat_ngene_chr=10,
    copykat_win_size=50,
    copykat_ks_cut=0.1,
    copykat_distance="euclidean",
    copykat_n_cores=2,
    copykat_plot_genes=True,
    copykat_output_seg=False,
    copykat_ref_max_cells=100,
    copykat_sam_name="copykat_run",
    copykat_ref_epithelial_key="cell_ontology_class",
    copykat_ref_epithelial_values=None,
):
    """
    Full preprocessing pipeline.

    QC THRESHOLDS (min_genes, max_mt)
    ----------------------------------
    These are NOT parameters of this function.  They are set once in Module 1::

        # QC disabled (default) — all epithelial cells pass through
        annotator = SampleAnnotator("GSE…")

        # Both thresholds
        annotator = SampleAnnotator("GSE…", min_genes=200, max_mt=40)

        # Gene count only
        annotator = SampleAnnotator("GSE…", min_genes=300)

        # MT only
        annotator = SampleAnnotator("GSE…", max_mt=25)

    Module 1 stores them in ``adata.uns['qc_params']`` of every h5ad it
    writes.  Module 3 reads that key automatically.

    If ``qc_params`` is absent, the QC filtering step is SKIPPED ENTIRELY —
    no default values are silently applied.

    PIPELINE FLOW
    -------------
    1.  Load full PopV h5ad (all cell types)  → adata_full
    2.  Read QC thresholds from adata.uns['qc_params']
    3.  Extract epithelial cells → apply QC if thresholds are set
    4.  scMalignantFinder + CopyKAT on epithelial cells
    5.  Keep ONLY malignant epithelial cells (non-malignant dropped)
    6.  adata_full non-epithelial cells → "rest" comparison group
    7.  Surfaceome filter (GESP genes) applied to BOTH groups
    8.  DEG: malignant epithelial vs non-epithelial rest
    9.  Binarise malignant-only cells, save

    Parameters
    ----------
    adata : AnnData or None
        Full PopV output. Auto-loaded if None.
    popv_path : str or None
        Explicit PopV h5ad path.
    log2fc_threshold : float   DEG log2FC cutoff. Default 1.0.
    pval_adj_threshold : float DEG BH-adjusted p-value cutoff. Default 0.05.
    reference_h5ad : str or None
        Tabula Sapiens h5ad for CopyKAT normal reference.
        CopyKAT skipped if None.
    tumor_h5ad : str or None
        Module 1 h5ad for Route A-rescue. Auto-detected if None.
    save_dir : str or None
        Output directory. Default 'preprocessing_results/' in cwd.
    scmalignant_model_dir : str or None  Auto-detected from SCART.
    surfaceome_path : str or None        Auto-detected from SCART GESP file.
    malignant_strategy : str
        'intersection' — malignant only if BOTH tools agree (default)
        'scMalignant'  — scMalignantFinder only
        'copykat'      — CopyKAT only (requires reference_h5ad)
    copykat_genome : str
        Genome version for copykat(). Default "hg20".
    copykat_id_type : str
        Gene ID type: "S" (symbol) or "E" (Ensembl). Default "S".
    copykat_ngene_chr : int
        Minimum genes per chromosome for copykat. Default 10.
    copykat_win_size : int
        Sliding window size for CNA smoothing. Default 50.
    copykat_ks_cut : float
        KS statistic cut-off for aneuploid/diploid classification. Default 0.1.
    copykat_distance : str
        Distance metric for hierarchical clustering. Default "euclidean".
    copykat_n_cores : int
        Parallel cores for copykat. Default 2.
    copykat_plot_genes : bool
        Whether copykat produces gene-level plots. Default True.
    copykat_output_seg : bool
        Whether copykat writes segment files. Default False.
    copykat_ref_max_cells : int
        Maximum normal reference cells (FIX 2). Default 100.
    copykat_sam_name : str
        Sample name prefix for copykat output files. Default "copykat_run".
    copykat_ref_epithelial_key : str
        obs column in reference h5ad used to identify normal epithelial cells.
        Default "cell_ontology_class".
    copykat_ref_epithelial_values : list of str or None
        Values in copykat_ref_epithelial_key that mark normal epithelial cells.
        Default ["epithelial cell"].

    Returns
    -------
    AnnData  Malignant epithelial cells only, surfaceome-filtered, binarised.
             DEG stored in adata.uns['filtered_deg'] and adata.uns['all_deg'].
             CopyKAT full results in adata.uns['copykat_results'].
             QC params echoed in adata.uns['qc_params'] (None if QC skipped).
    """
    if copykat_ref_epithelial_values is None:
        copykat_ref_epithelial_values = ["epithelial cell"]

    print("\n========== START ==========\n")

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # STEP 1 — Load full PopV h5ad  → adata_full
    # ------------------------------------------------------------------
    if adata is None:
        auto_popv = _auto_popv_h5ad()
        cands = (
            ([popv_path]   if popv_path   else []) +
            ([auto_popv]   if auto_popv   else [])
        )
        for path in cands:
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

    # Keep a clean copy of the FULL dataset for the "rest" group later
    adata_full = adata.copy()
    print(f"Full dataset loaded: {adata_full.n_obs} cells × {adata_full.n_vars} genes")

    # ------------------------------------------------------------------
    # STEP 2 — Read QC thresholds written by Module 1
    # ------------------------------------------------------------------
    min_genes, max_mt, qc_active, qc_source = _read_qc_params(adata_full)

    # ------------------------------------------------------------------
    # STEP 3 — Extract epithelial cells → QC (conditional)
    # ------------------------------------------------------------------
    print("\n--- Step 3: Epithelial selection" +
          (" + QC ---" if qc_active else " ---"))

    labels  = adata_full.obs["popv_majority_vote_prediction"].astype(str)
    ep_mask = labels.str.endswith("epithelial cell")
    print(f"Epithelial cells: {ep_mask.sum()} / {adata_full.n_obs} total")
    print(f"Non-epithelial cells (will be 'rest' group for DEG): {(~ep_mask).sum()}")

    adata_epi = adata_full[ep_mask].copy()
    before_qc = adata_epi.n_obs

    if qc_active:
        adata_epi.var["mt"] = adata_epi.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata_epi, qc_vars=["mt"], inplace=True)
        print(f"Mean MT% BEFORE QC: {adata_epi.obs['pct_counts_mt'].mean():.2f}")

        # Build filter: apply only the thresholds that are not None
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
        print(
            f"Epithelial cells after QC: {adata_epi.n_obs}  "
            f"(removed {before_qc - adata_epi.n_obs}  |  {filter_desc})"
        )
        print(f"Mean MT% AFTER QC:  {adata_epi.obs['pct_counts_mt'].mean():.2f}\n")
    else:
        print(f"QC filtering SKIPPED — all {adata_epi.n_obs} epithelial cells proceed.\n")

    # Route raw counts → .X; snapshot for CopyKAT
    print("Detecting raw count source for epithelial cells...")
    for lyr in ("scvi_counts", "raw_counts", "counts"):
        if lyr in adata_epi.layers:
            print(f"  Using layers['{lyr}'] as raw counts.")
            adata_epi.X = adata_epi.layers[lyr].copy()
            break
    else:
        if adata_epi.raw is not None:
            print("  Using adata.raw.X as raw counts.")
            adata_epi.X = adata_epi.raw.X.copy()
        else:
            print("  No raw layer — assuming .X is raw counts.")

    adata_epi.var_names_make_unique()
    adata_epi.layers["raw_for_cna"] = adata_epi.X.copy()
    sc.pp.normalize_total(adata_epi, target_sum=1e4)
    sc.pp.log1p(adata_epi)

    # ------------------------------------------------------------------
    # STEP 4a — scMalignantFinder
    # ------------------------------------------------------------------
    print("\n--- Step 4a: scMalignantFinder ---")
    feature_tsv = os.path.join(scmalignant_model_dir, "ordered_feature.tsv")

    # Report gene-space route
    if "full_counts" in adata_epi.layers and adata_epi.uns.get("full_counts_var_names"):
        print(
            f"Gene-space: Route A-new "
            f"({len(adata_epi.uns['full_counts_var_names'])} genes)"
        )
    else:
        rescue = tumor_h5ad or _auto_tumor_h5ad()
        if rescue and os.path.exists(rescue):
            print(f"Gene-space: Route A-rescue ({rescue})")
        elif adata_epi.raw is not None:
            print(
                f"Gene-space: Route A-old (adata.raw, {adata_epi.raw.n_vars} genes)"
            )
        else:
            print("Gene-space: Route C (4000-HVG fallback — ~19% model overlap)")

    adata_scm = _build_fullgene_adata_for_scm(adata_epi, feature_tsv, tumor_h5ad)
    print(f"  Gene space used: {adata_scm.n_vars} genes")

    # scMalignantFinder is bundled inside SCART as a sub-package, NOT installed.
    #
    # Directory layout:
    #   <scart_root>/external/scMalignantFinder/
    #       __init__.py          ← always present
    #       classifier.py        ← present in newer installs; class lives here
    #       model/               ← this is scmalignant_model_dir
    #
    # Loading strategy (tries all routes, most specific first):
    #   Route 1  classifier.py exists → load it directly via importlib
    #   Route 2  __init__.py exists   → load it directly via importlib
    #   Route 3  fallback             → add …/external to sys.path and import
    import sys as _sys
    import importlib.util as _ilu

    _scm_pkg_dir      = os.path.dirname(scmalignant_model_dir)   # …/external/scMalignantFinder
    _scm_external_dir = os.path.dirname(_scm_pkg_dir)             # …/external
    _classifier_py    = os.path.join(_scm_pkg_dir, "classifier.py")
    _init_py          = os.path.join(_scm_pkg_dir, "__init__.py")

    def _load_module_from_file(mod_name, filepath):
        """Load a Python file as a module by absolute path."""
        if mod_name in _sys.modules:
            return _sys.modules[mod_name]
        _spec = _ilu.spec_from_file_location(mod_name, filepath)
        _mod  = _ilu.module_from_spec(_spec)
        _sys.modules[mod_name] = _mod
        _spec.loader.exec_module(_mod)
        return _mod

    _clf_mod = None

    # Route 1 — classifier.py (newer installs)
    if os.path.isfile(_classifier_py):
        logger.info(f"scMalignantFinder: loading via classifier.py ({_classifier_py})")
        _clf_mod = _load_module_from_file("scMalignantFinder.classifier", _classifier_py)

    # Route 2 — __init__.py (older installs where class is defined there)
    elif os.path.isfile(_init_py):
        logger.info(f"scMalignantFinder: loading via __init__.py ({_init_py})")
        _clf_mod = _load_module_from_file("scMalignantFinder", _init_py)

    # Route 3 — sys.path fallback
    else:
        logger.warning(
            f"scMalignantFinder: neither classifier.py nor __init__.py found in "
            f"{_scm_pkg_dir} — attempting sys.path fallback."
        )
        _inserted = _scm_external_dir not in _sys.path
        if _inserted:
            _sys.path.insert(0, _scm_external_dir)
        try:
            import importlib as _il
            import scMalignantFinder as _scm_pkg
            _clf_mod = _scm_pkg
        except ImportError as _exc:
            raise ImportError(
                f"Could not import scMalignantFinder from any route.\n"
                f"  classifier.py checked: {_classifier_py}\n"
                f"  __init__.py checked:   {_init_py}\n"
                f"  sys.path fallback dir: {_scm_external_dir}\n"
                f"  Original error: {_exc}"
            ) from _exc
        finally:
            if _inserted and _scm_external_dir in _sys.path:
                _sys.path.remove(_scm_external_dir)

    # pretrain_dir  — directory containing model.joblib + ordered_feature.tsv
    # norm_type=False — adata_scm is already log-normalised by _build_fullgene_adata_for_scm;
    #                   passing True would double-normalise and corrupt the counts.
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

    # ------------------------------------------------------------------
    # STEP 4b — CopyKAT
    # ------------------------------------------------------------------
    copykat_available = False
    copykat_result_df = None

    if malignant_strategy in ("copykat", "intersection"):
        if reference_h5ad is None:
            print(
                "\nWarning: CopyKAT skipped — no reference_h5ad provided.\n"
                "  Falling back to scMalignantFinder only."
            )
            malignant_strategy = "scMalignant"
        else:
            print(
                f"\n--- Step 4b: CopyKAT ---\n"
                f"  Reference:           {reference_h5ad}\n"
                f"  Genome:              {copykat_genome}\n"
                f"  id.type:             {copykat_id_type}\n"
                f"  ngene.chr:           {copykat_ngene_chr}\n"
                f"  win.size:            {copykat_win_size}\n"
                f"  KS.cut:              {copykat_ks_cut}\n"
                f"  distance:            {copykat_distance}\n"
                f"  n.cores:             {copykat_n_cores}\n"
                f"  plot.genes:          {copykat_plot_genes}\n"
                f"  output.seg:          {copykat_output_seg}\n"
                f"  ref_max_cells:       {copykat_ref_max_cells}\n"
                f"  sam_name:            {copykat_sam_name}\n"
                f"  ref_epithelial_key:  {copykat_ref_epithelial_key}\n"
                f"  ref_epithelial_vals: {copykat_ref_epithelial_values}"
            )
            try:
                adata_raw_cna   = adata_epi.copy()
                adata_raw_cna.X = adata_epi.layers["raw_for_cna"]
                adata_ref_full  = sc.read_h5ad(reference_h5ad)

                copykat_result_df = _run_copykat(
                    adata_query              = adata_raw_cna,
                    adata_ref                = adata_ref_full,
                    genome                   = copykat_genome,
                    id_type                  = copykat_id_type,
                    ngene_chr                = copykat_ngene_chr,
                    win_size                 = copykat_win_size,
                    ks_cut                   = copykat_ks_cut,
                    distance                 = copykat_distance,
                    n_cores                  = copykat_n_cores,
                    plot_genes               = copykat_plot_genes,
                    output_seg               = copykat_output_seg,
                    ref_max_cells            = copykat_ref_max_cells,
                    sam_name                 = copykat_sam_name,
                    ref_epithelial_key       = copykat_ref_epithelial_key,
                    ref_epithelial_values    = copykat_ref_epithelial_values,
                )

                # Store per-cell CopyKAT predictions on adata_epi
                bc_to_pred = dict(zip(
                    copykat_result_df["barcode"],
                    copykat_result_df["copykat_prediction"],
                ))
                adata_epi.obs["copykat_prediction"] = [
                    bc_to_pred.get(b, "not.defined") for b in adata_epi.obs_names
                ]

                copykat_available = True

                print("\nCopyKAT completed.")
                print("  Prediction counts:")
                print(adata_epi.obs["copykat_prediction"].value_counts().to_string())

            except Exception as exc:
                print(
                    f"\nWarning: CopyKAT failed — {type(exc).__name__}: {exc}\n"
                    "  Falling back to scMalignantFinder only."
                )
                logger.exception("CopyKAT error:")
                malignant_strategy = "scMalignant"

    # ------------------------------------------------------------------
    # STEP 4c — Combine malignancy calls → final_malignant
    # ------------------------------------------------------------------
    print("\n--- Step 4c: Combine malignancy calls ---")
    scm_mal = adata_epi.obs[scm_col].str.lower() == "malignant"

    if copykat_available:
        # CopyKAT labels aneuploid cells as malignant
        ck_mal = adata_epi.obs["copykat_prediction"].str.lower() == "aneuploid"
        if malignant_strategy == "intersection":
            malignant_mask  = scm_mal & ck_mal
            strategy_label  = "intersection (scMalignantFinder AND CopyKAT)"
        elif malignant_strategy == "copykat":
            malignant_mask  = ck_mal
            strategy_label  = "CopyKAT only"
        else:
            malignant_mask  = scm_mal
            strategy_label  = "scMalignantFinder only"
    else:
        malignant_mask = scm_mal
        strategy_label = "scMalignantFinder only"

    adata_epi.obs["final_malignant"] = malignant_mask.map(
        {True: "malignant", False: "non-malignant"}
    )
    print(f"Strategy: {strategy_label}")
    print(f"  Malignant epithelial:     {malignant_mask.sum()}")
    print(f"  Non-malignant epithelial: {(~malignant_mask).sum()}  ← these will be REMOVED")

    # ------------------------------------------------------------------
    # STEP 5 — Keep ONLY malignant epithelial cells
    # ------------------------------------------------------------------
    print("\n--- Step 5: Retain malignant epithelial cells only ---")
    adata_mal = adata_epi[malignant_mask].copy()
    print(f"Malignant epithelial cells retained: {adata_mal.n_obs}")
    print(f"Non-malignant epithelial cells removed: {(~malignant_mask).sum()}")

    if adata_mal.n_obs == 0:
        raise ValueError(
            "No malignant cells found after filtering.\n"
            "Check scMalignantFinder model path and gene overlap."
        )

    # ------------------------------------------------------------------
    # STEP 6 — Non-epithelial "rest" group from adata_full
    # ------------------------------------------------------------------
    print("\n--- Step 6: Extract non-epithelial 'rest' group ---")
    rest_mask  = ~ep_mask    # non-epithelial cells from the full dataset
    adata_rest = adata_full[rest_mask].copy()
    print(f"Non-epithelial 'rest' cells: {adata_rest.n_obs}")

    # Normalise the rest group for DEG
    for lyr in ("scvi_counts", "raw_counts", "counts"):
        if lyr in adata_rest.layers:
            adata_rest.X = adata_rest.layers[lyr].copy()
            break
    else:
        if adata_rest.raw is not None:
            adata_rest.X = adata_rest.raw.X.copy()
    sc.pp.normalize_total(adata_rest, target_sum=1e4)
    sc.pp.log1p(adata_rest)

    # ------------------------------------------------------------------
    # STEP 7 — Surfaceome filter (GESP genes) applied to BOTH groups
    # ------------------------------------------------------------------
    print("\n--- Step 7: Surfaceome filter (GESP file) ---")
    surfaceome   = pd.read_csv(surfaceome_path)
    surfaceome.columns = surfaceome.columns.str.strip()
    surf_genes   = surfaceome["Gene"].astype(str).tolist()
    print(f"Surfaceome genes in GESP file: {len(surf_genes)}")

    surf_in_mal  = adata_mal.var_names.intersection(surf_genes)
    adata_mal    = adata_mal[:, surf_in_mal].copy()
    print(f"Surfaceome genes in malignant cells: {len(surf_in_mal)}")

    surf_in_rest = adata_rest.var_names.intersection(surf_genes)
    adata_rest   = adata_rest[:, surf_in_rest].copy()
    print(f"Surfaceome genes in rest cells: {len(surf_in_rest)}")

    surf_common  = surf_in_mal.intersection(surf_in_rest)
    adata_mal    = adata_mal[:, surf_common].copy()
    adata_rest   = adata_rest[:, surf_common].copy()
    print(f"Common surfaceome genes (used for DEG): {len(surf_common)}\n")

    # ------------------------------------------------------------------
    # STEP 8 — DEG: malignant epithelial vs non-epithelial rest
    #          (FIX 9: correct biological comparison)
    # ------------------------------------------------------------------
    print("--- Step 8: DEG — malignant epithelial vs non-epithelial rest ---")

    adata_mal.obs["deg_group"]  = "malignant_epithelial"
    adata_rest.obs["deg_group"] = "non_epithelial_rest"

    adata_deg = sc.concat(
        [adata_mal, adata_rest],
        join="outer",
        label=None,
    )
    adata_deg.obs_names_make_unique()
    adata_deg.var = adata_mal.var.copy()

    if sp.issparse(adata_deg.X):
        adata_deg.X = adata_deg.X.toarray()
    adata_deg.X = np.nan_to_num(np.array(adata_deg.X, dtype=np.float32), nan=0.0)
    adata_deg.X = sp.csr_matrix(adata_deg.X)

    print(f"DEG AnnData: {adata_deg.n_obs} cells × {adata_deg.n_vars} genes")
    print(
        f"  malignant_epithelial: "
        f"{(adata_deg.obs['deg_group'] == 'malignant_epithelial').sum()}"
    )
    print(
        f"  non_epithelial_rest:  "
        f"{(adata_deg.obs['deg_group'] == 'non_epithelial_rest').sum()}"
    )

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
    print(
        f"Applying filters: log2FC > {log2fc_threshold}, "
        f"pvals_adj < {pval_adj_threshold}"
    )

    filtered_deg = deg[
        (deg["logfoldchanges"] > log2fc_threshold) &
        (deg["pvals_adj"]      < pval_adj_threshold)
    ]

    if filtered_deg.shape[0] == 0:
        print(
            "WARNING: 0 DEGs passed the filter.\n"
            "  Try: log2fc_threshold=0.5 or pval_adj_threshold=0.10"
        )
    else:
        print(f"DEGs retained: {filtered_deg.shape[0]}")
        print("\nTop 10 DEGs (malignant epithelial vs non-epithelial rest):")
        print(filtered_deg.head(10).to_string(index=False))

    # ------------------------------------------------------------------
    # STEP 9 — Binarise ONLY malignant cells; store results; save
    # ------------------------------------------------------------------
    print("\n--- Step 9: Binarise malignant cells and save ---")

    adata_mal.X = (
        np.array(
            adata_mal.X.toarray() if sp.issparse(adata_mal.X) else adata_mal.X
        ) > 0
    ).astype(np.int8)
    adata_mal.X = sp.csr_matrix(adata_mal.X)
    print("Expression converted to binary (0/1).")

    # Store DEG results
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

    # Echo QC params — store None if QC was skipped so downstream knows
    adata_mal.uns["qc_params"] = (
        {"min_genes": min_genes, "max_mt": max_mt}
        if qc_active else None
    )

    # FIX 8 — store full CopyKAT results
    if copykat_result_df is not None:
        final_barcodes      = set(adata_mal.obs_names)
        copykat_stored      = copykat_result_df.copy()
        copykat_stored["in_final_output"] = copykat_stored["barcode"].isin(
            final_barcodes
        )
        adata_mal.uns["copykat_results"] = copykat_stored

        print(
            f"\nCopyKAT results stored in adata.uns['copykat_results']:\n"
            f"  Shape: {copykat_stored.shape[0]} rows × "
            f"{copykat_stored.shape[1]} columns\n"
            f"  Columns: {list(copykat_stored.columns)}\n"
            f"  Prediction summary (all epithelial cells before filtering):\n"
            + copykat_stored["copykat_prediction"].value_counts().to_string()
            + f"\n  Cells in final malignant output: "
            f"{copykat_stored['in_final_output'].sum()}"
        )
    else:
        adata_mal.uns["copykat_results"] = None
        print("\nCopyKAT was not run — adata.uns['copykat_results'] = None")

    # Clean string columns
    for col in adata_mal.obs.columns:
        if adata_mal.obs[col].dtype == object:
            adata_mal.obs[col] = adata_mal.obs[col].astype(str)

    final_path = os.path.join(save_dir, "final_tumor.h5ad")
    adata_mal.write(final_path)

    print(f"\nFinal object saved to: {final_path}")
    print("\n========== PREPROCESSING COMPLETED ==========\n")
    return adata_mal
