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
COPYKAT DESIGN NOTES
═══════════════════════════════════════════════════════════════════════════════

CopyKAT (Copy number Karyotyping of Aneuploid Tumours) infers copy number
alterations from scRNA-seq data to distinguish aneuploid tumour cells from
diploid normal cells.

BIOLOGICALLY CORRECT APPROACH USED HERE:
  - Query cells:     epithelial cells only (from query h5ad)
  - Reference cells: normal epithelial cells from Tabula Sapiens h5ad
                     (prefixed REF_ to identify them as known normals)
  - CopyKAT uses norm.cell.names to anchor the diploid baseline precisely
  - Predictions returned for query cells only (REF_ rows stripped)

WHY THIS IS BETTER THAN NO-REFERENCE MODE (Script 1 equivalent):
  - Without a reference, CopyKAT guesses the diploid baseline internally
    from whatever cells look "most normal" — unreliable in tumour samples
  - With known normal epithelial reference cells the diploid baseline is
    anchored to the correct tissue/cell-type context
  - False-positive rate (immune/stromal labelled aneuploid) is drastically
    reduced by running on epithelial cells only

SPEED OPTIMISATIONS APPLIED:
  copykat_n_cores          : parallelise DLM smoothing (set to available CPUs)
  copykat_ref_max_cells    : subsample reference to ≤100 cells (paper shows
                             ~50–100 normals are sufficient for baseline)
  copykat_win_size         : larger window (50) = fewer DLM iterations per gene
  copykat_ngene_chr        : higher threshold (10) = more low-quality cells
                             removed before the expensive smoothing step

═══════════════════════════════════════════════════════════════════════════════
QC PARAMETER FLOW
═══════════════════════════════════════════════════════════════════════════════

Module 1 (geo_fetcher.py)
  SampleAnnotator("GSE…")                           → QC disabled, key absent
  SampleAnnotator("GSE…", min_genes=200)            → only gene count filter
  SampleAnnotator("GSE…", max_mt=40)                → only MT filter
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
                   final_malignant
            → uns: copykat_results (full detail DataFrame)
                   filtered_deg, all_deg, deg_params, cancer_type, qc_params

═══════════════════════════════════════════════════════════════════════════════
KEY FIXES (inherited from previous version)
═══════════════════════════════════════════════════════════════════════════════

FIX 1   Full-gene route priority for scMalignantFinder:
        A-new (layers['full_counts']) → A-rescue (auto-detect Module 1 h5ad)
        → A-old (adata.raw) → C (4000-HVG fallback)

FIX 2   CopyKAT reference subsampled to copykat_ref_max_cells (default 100).

FIX 3   DEG uses pvals_adj (BH-adjusted), not raw pvals.

FIX 5   _get_raw_matrix prefers adata.raw over HVG layers (more genes).

FIX 6   scMalignantFinder predictions aligned by obs_names, not positional.

FIX 9   CORRECT DEG DESIGN:
        Malignant epithelial cells vs all NON-EPITHELIAL cells from adata_full.
        Both groups surfaceome-filtered before DEG.
        Non-malignant epithelial cells are REMOVED (not used in any group).

QC FIX  min_genes and max_mt removed from the public API of
        run_preprocessing_pipeline().  Both values are read from
        adata.uns['qc_params'] (written by Module 1).
        If the key is absent the QC step is SKIPPED ENTIRELY.
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
    max_mt    : float or None
    qc_active : bool
    source    : str
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
# CopyKAT runner (replaces inferCNA)
# ═══════════════════════════════════════════════════════════════════════════

def _run_copykat(
    adata_query,
    adata_ref,
    output_dir,
    sam_name        = "copykat_run",
    genome          = "hg20",
    id_type         = "S",
    ngene_chr       = 10,
    win_size        = 50,
    ks_cut          = 0.1,
    distance        = "euclidean",
    n_cores         = 4,
    plot_genes      = True,
    output_seg      = False,
    low_dr          = 0.05,
    up_dr           = 0.2,
    ref_max_cells   = 100,
):
    """
    Run CopyKAT via rpy2 using reference-guided mode.

    Biologically correct approach:
      - Query cells:  epithelial cells from the tumour sample
      - Reference:    normal epithelial cells from Tabula Sapiens, subsampled
                      to ref_max_cells (50–100 is sufficient for the diploid
                      baseline; more than that just slows the DLM step)
      - All REF_ prefixed barcodes are passed to norm.cell.names
      - Predictions are returned for query cells only (REF_ rows dropped)

    Speed optimisations vs vanilla CopyKAT:
      - ref_max_cells=100  : much fewer cells for baseline, same accuracy
      - win_size=50        : larger smoothing window → fewer DLM iterations
      - ngene_chr=10       : stricter early filter → fewer cells to smooth
      - n_cores=4          : parallel DLM (set higher if CPUs available)

    Parameters
    ----------
    adata_query    : AnnData  — epithelial cells (query), raw UMI counts in .X
                               or a raw layer
    adata_ref      : AnnData  — Tabula Sapiens reference
    output_dir     : str      — directory for CopyKAT output files
    sam_name       : str      — prefix for output filenames
    genome         : str      — "hg20" (=hg38), "hg19", or "mm10"
    id_type        : str      — "S" (gene Symbol) or "E" (Ensembl)
    ngene_chr      : int      — min genes per chr per cell (higher → faster)
    win_size       : int      — smoothing window (15–150; higher → faster)
    ks_cut         : float    — segmentation sensitivity (0.05–0.15)
    distance       : str      — "euclidean", "pearson", or "spearman"
    n_cores        : int      — CPU cores for DLM parallelisation
    plot_genes     : bool     — include gene names in heatmap PDF
    output_seg     : bool     — write .seg file for IGV
    low_dr         : float    — min fraction of cells expressing a gene
    up_dr          : float    — max fraction of cells expressing a gene
    ref_max_cells  : int      — max reference cells to subsample to

    Returns
    -------
    pd.DataFrame  columns: cell.names, copykat.pred
                           copykat.pred ∈ {"aneuploid", "diploid",
                                           "not.defined"}
                  Rows for query cells only (REF_ rows removed).
    """
    try:
        import rpy2.robjects as ro
    except ImportError as exc:
        raise ImportError(
            "rpy2 is required to run CopyKAT.\n"
            "conda install -c conda-forge rpy2"
        ) from exc

    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    # ── 1. Extract raw counts from query ──────────────────────────────────
    logger.info("CopyKAT: extracting query raw counts ...")
    raw_q = _get_raw_matrix(adata_query)           # cells × genes
    query_barcodes = list(adata_query.obs_names)
    query_genes    = list(adata_query.var_names)
    logger.info(
        f"  Query matrix: {raw_q.shape[0]} cells × {raw_q.shape[1]} genes"
    )

    # ── 2. Extract raw counts from reference, filter to normal epithelial ──
    EPITHELIAL = {
        "epithelial cell",
        "glandular epithelial cell",
        "ovarian surface epithelial cell",
    }

    if "cell_ontology_class" in adata_ref.obs.columns:
        ep_mask  = adata_ref.obs["cell_ontology_class"].str.lower().isin(EPITHELIAL)
        ref_epi  = adata_ref[ep_mask].copy() if ep_mask.any() else adata_ref.copy()
        logger.info(
            f"CopyKAT: reference epithelial cells (before subsampling): {ref_epi.n_obs}"
        )
    else:
        logger.warning(
            "CopyKAT: 'cell_ontology_class' column absent — using full reference."
        )
        ref_epi = adata_ref.copy()

    # FIX 2 — subsample reference (50–100 normal cells are sufficient)
    if ref_epi.n_obs > ref_max_cells:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(ref_epi.n_obs, size=ref_max_cells, replace=False)
        ref_epi = ref_epi[np.sort(idx)].copy()
        logger.info(f"CopyKAT: reference subsampled to {ref_max_cells} cells.")

    logger.info("CopyKAT: extracting reference raw counts ...")
    raw_r      = _get_raw_matrix(ref_epi)          # cells × genes
    ref_genes  = list(ref_epi.var_names)
    logger.info(
        f"  Reference matrix: {raw_r.shape[0]} cells × {raw_r.shape[1]} genes"
    )

    # ── 3. Find common genes → combine matrices ────────────────────────────
    q_gene_idx = {g: i for i, g in enumerate(query_genes)}
    r_gene_idx = {g: i for i, g in enumerate(ref_genes)}
    common     = sorted(set(query_genes) & set(ref_genes))
    logger.info(f"CopyKAT: common genes between query and reference: {len(common)}")

    if len(common) < 200:
        raise ValueError(
            f"Only {len(common)} common genes between query and reference.\n"
            "Both h5ad files must use HGNC gene symbols.\n"
            "Minimum 200 common genes required for CopyKAT."
        )

    q_idx = np.array([q_gene_idx[g] for g in common])
    r_idx = np.array([r_gene_idx[g] for g in common])

    mat_q_sub  = raw_q[:, q_idx]                   # query cells × common genes
    mat_r_sub  = raw_r[:, r_idx]                   # ref cells   × common genes

    # CopyKAT expects genes × cells
    mat_q_t    = mat_q_sub.T                       # common genes × query cells
    mat_r_t    = mat_r_sub.T                       # common genes × ref cells

    # Prefix reference barcodes to mark them as known normals
    ref_barcodes  = ["REF_" + b for b in ref_epi.obs_names]
    mat_combined  = np.hstack([mat_q_t, mat_r_t])  # genes × (query + ref) cells
    all_barcodes  = query_barcodes + ref_barcodes

    n_genes, n_cells = mat_combined.shape
    logger.info(
        f"CopyKAT: combined matrix → {n_genes} genes × {n_cells} cells "
        f"({len(query_barcodes)} query + {len(ref_barcodes)} ref)"
    )

    # ── 4. Transfer to R and run copykat() ────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    abs_outdir = os.path.abspath(output_dir)

    logger.info("CopyKAT: transferring matrix to R ...")
    ro.globalenv["ck_flat"]       = ro.FloatVector(
        mat_combined.flatten(order="F").tolist()
    )
    ro.globalenv["ck_nrow"]       = ro.IntVector([n_genes])
    ro.globalenv["ck_ncol"]       = ro.IntVector([n_cells])
    ro.globalenv["ck_genes"]      = ro.StrVector(common)
    ro.globalenv["ck_cells"]      = ro.StrVector(all_barcodes)
    ro.globalenv["ck_norm_cells"] = ro.StrVector(ref_barcodes)
    ro.globalenv["ck_sam"]        = ro.StrVector([sam_name])
    ro.globalenv["ck_outdir"]     = ro.StrVector([abs_outdir])
    ro.globalenv["ck_id_type"]    = ro.StrVector([id_type])
    ro.globalenv["ck_ngene_chr"]  = ro.IntVector([ngene_chr])
    ro.globalenv["ck_win"]        = ro.IntVector([win_size])
    ro.globalenv["ck_ks"]         = ro.FloatVector([ks_cut])
    ro.globalenv["ck_dist"]       = ro.StrVector([distance])
    ro.globalenv["ck_genome"]     = ro.StrVector([genome])
    ro.globalenv["ck_ncores"]     = ro.IntVector([n_cores])
    ro.globalenv["ck_plot_genes"] = ro.StrVector(["TRUE" if plot_genes  else "FALSE"])
    ro.globalenv["ck_out_seg"]    = ro.StrVector(["TRUE" if output_seg  else "FALSE"])
    ro.globalenv["ck_low_dr"]     = ro.FloatVector([low_dr])
    ro.globalenv["ck_up_dr"]      = ro.FloatVector([up_dr])

    logger.info("CopyKAT: running copykat() in R — this may take several minutes ...")
    ro.r("""
        suppressPackageStartupMessages(library(copykat))

        rawmat           <- matrix(ck_flat, nrow=ck_nrow, ncol=ck_ncol)
        rownames(rawmat) <- ck_genes
        colnames(rawmat) <- ck_cells

        old_wd <- getwd()
        setwd(ck_outdir)

        copykat.result <- copykat(
            rawmat          = rawmat,
            id.type         = ck_id_type,
            ngene.chr       = ck_ngene_chr,
            win.size        = ck_win,
            KS.cut          = ck_ks,
            sam.name        = ck_sam,
            distance        = ck_dist,
            norm.cell.names = ck_norm_cells,
            output.seg      = ck_out_seg,
            plot.genes      = ck_plot_genes,
            genome          = ck_genome,
            n.cores         = ck_ncores,
            LOW.DR          = ck_low_dr,
            UP.DR           = ck_up_dr,
            cell.line       = "no"
        )

        setwd(old_wd)
        ck_pred <- as.data.frame(copykat.result$prediction)
    """)

    # ── 5. Pull predictions back to Python ────────────────────────────────
    with localconverter(ro.default_converter + pandas2ri.converter):
        pred_df_all = ro.conversion.rpy2py(ro.globalenv["ck_pred"]).reset_index(drop=True)

    # Normalise column name (varies by rpy2 version)
    pred_col = "copykat.pred"
    if pred_col not in pred_df_all.columns:
        pred_col = [c for c in pred_df_all.columns if "pred" in c.lower()][0]

    # Drop REF_ rows — return query cells only
    pred_df = (
        pred_df_all[~pred_df_all["cell.names"].str.startswith("REF_")]
        .copy()
        .reset_index(drop=True)
    )

    # Ensure the column is always named "copykat.pred" for downstream code
    if pred_col != "copykat.pred":
        pred_df = pred_df.rename(columns={pred_col: "copykat.pred"})

    # Save a copy of the predictions CSV alongside other CopyKAT output files
    out_csv = os.path.join(abs_outdir, f"{sam_name}_predictions.csv")
    pred_df.to_csv(out_csv, index=False)

    aneuploid   = (pred_df["copykat.pred"] == "aneuploid").sum()
    diploid     = (pred_df["copykat.pred"] == "diploid").sum()
    not_defined = (pred_df["copykat.pred"] == "not.defined").sum()

    logger.info(
        f"\nCopyKAT completed\n"
        f"  Aneuploid  (malignant) : {aneuploid}\n"
        f"  Diploid    (normal)    : {diploid}\n"
        f"  Not defined            : {not_defined}\n"
        f"  Total query cells      : {len(pred_df)}\n"
        f"  Predictions CSV        : {out_csv}"
    )

    return pred_df


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
    # malignancy strategy
    malignant_strategy="intersection",
    # CopyKAT parameters
    copykat_genome       = "hg20",
    copykat_id_type      = "S",
    copykat_ngene_chr    = 10,
    copykat_win_size     = 50,
    copykat_ks_cut       = 0.1,
    copykat_distance     = "euclidean",
    copykat_n_cores      = 4,
    copykat_plot_genes   = True,
    copykat_output_seg   = False,
    copykat_low_dr       = 0.05,
    copykat_up_dr        = 0.2,
    copykat_ref_max_cells= 100,
    copykat_sam_name     = "copykat_run",
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

    If ``qc_params`` is absent, the QC filtering step is SKIPPED ENTIRELY.

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
    log2fc_threshold : float
        DEG log2FC cutoff. Default 1.0.
    pval_adj_threshold : float
        DEG BH-adjusted p-value cutoff. Default 0.05.
    reference_h5ad : str or None
        Tabula Sapiens h5ad for CopyKAT normal reference.
        CopyKAT skipped and falls back to scMalignantFinder only if None.
    tumor_h5ad : str or None
        Module 1 h5ad for Route A-rescue. Auto-detected if None.
    save_dir : str or None
        Output directory. Default 'preprocessing_results/' in cwd.
    scmalignant_model_dir : str or None
        Auto-detected from SCART.
    surfaceome_path : str or None
        Auto-detected from SCART GESP file.
    malignant_strategy : str
        'intersection' — malignant only if BOTH tools agree (default)
        'scMalignant'  — scMalignantFinder only
        'copykat'      — CopyKAT only (requires reference_h5ad)
    copykat_genome : str
        "hg20" (=hg38, default), "hg19", or "mm10".
    copykat_id_type : str
        "S" = gene Symbol (default), "E" = Ensembl ID.
    copykat_ngene_chr : int
        Min genes per chromosome per cell. Higher = faster (default 10).
    copykat_win_size : int
        DLM smoothing window size. Larger = faster (default 50, range 15–150).
    copykat_ks_cut : float
        Segmentation sensitivity (default 0.1, range 0.05–0.15).
    copykat_distance : str
        Clustering distance: "euclidean" (default), "pearson", "spearman".
    copykat_n_cores : int
        CPU cores for DLM parallelisation (default 4).
    copykat_plot_genes : bool
        Include gene names in heatmap PDF (default True).
    copykat_output_seg : bool
        Save .seg file for IGV viewer (default False).
    copykat_low_dr : float
        Min fraction of cells expressing a gene (default 0.05).
    copykat_up_dr : float
        Max fraction of cells expressing a gene (default 0.2).
    copykat_ref_max_cells : int
        Max reference cells to subsample to (default 100).
        50–100 normal cells are sufficient for a stable diploid baseline.
    copykat_sam_name : str
        Prefix for CopyKAT output filenames (default "copykat_run").

    Returns
    -------
    AnnData  Malignant epithelial cells only, surfaceome-filtered, binarised.
             DEG stored in adata.uns['filtered_deg'] and adata.uns['all_deg'].
             CopyKAT full results in adata.uns['copykat_results'].
             QC params echoed in adata.uns['qc_params'] (None if QC skipped).
    """
    print("\n========== START ==========\n")

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "preprocessing_results")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory: {save_dir}")

    # CopyKAT output goes into a sub-folder to keep files organised
    copykat_output_dir = os.path.join(save_dir, "copykat_results")

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

    # Route raw counts → .X
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

    # Keep a raw snapshot for CopyKAT (must be UMI counts, not log-normalised)
    adata_epi.layers["raw_for_cna"] = adata_epi.X.copy()

    sc.pp.normalize_total(adata_epi, target_sum=1e4)
    sc.pp.log1p(adata_epi)

    # ------------------------------------------------------------------
    # STEP 4a — scMalignantFinder
    # ------------------------------------------------------------------
    print("\n--- Step 4a: scMalignantFinder ---")
    feature_tsv = os.path.join(scmalignant_model_dir, "ordered_feature.tsv")

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

    from scMalignantFinder import classifier
    model = classifier.scMalignantFinder(
        test_input          = adata_scm,
        celltype_annotation = False,
        pretrain_path       = scmalignant_model_dir,
        feature_path        = feature_tsv,
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
    # STEP 4b — CopyKAT (replaces inferCNA)
    # ------------------------------------------------------------------
    copykat_available = False
    copykat_pred_df   = None

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
                f"  Reference      : {reference_h5ad}\n"
                f"  Ref subsampled : ≤{copykat_ref_max_cells} normal epithelial cells\n"
                f"  Genome         : {copykat_genome}\n"
                f"  Win size       : {copykat_win_size}  (larger = faster DLM)\n"
                f"  ngene_chr      : {copykat_ngene_chr}  (higher = more early filtering)\n"
                f"  n_cores        : {copykat_n_cores}\n"
                f"  Strategy       : {malignant_strategy}"
            )
            try:
                # Build a raw-count AnnData for CopyKAT
                adata_raw_cna   = adata_epi.copy()
                adata_raw_cna.X = adata_epi.layers["raw_for_cna"]

                adata_ref_full  = sc.read_h5ad(reference_h5ad)

                copykat_pred_df = _run_copykat(
                    adata_query     = adata_raw_cna,
                    adata_ref       = adata_ref_full,
                    output_dir      = copykat_output_dir,
                    sam_name        = copykat_sam_name,
                    genome          = copykat_genome,
                    id_type         = copykat_id_type,
                    ngene_chr       = copykat_ngene_chr,
                    win_size        = copykat_win_size,
                    ks_cut          = copykat_ks_cut,
                    distance        = copykat_distance,
                    n_cores         = copykat_n_cores,
                    plot_genes      = copykat_plot_genes,
                    output_seg      = copykat_output_seg,
                    low_dr          = copykat_low_dr,
                    up_dr           = copykat_up_dr,
                    ref_max_cells   = copykat_ref_max_cells,
                )

                # Map CopyKAT predictions onto adata_epi obs
                bc_to_ckpred = dict(
                    zip(copykat_pred_df["cell.names"],
                        copykat_pred_df["copykat.pred"])
                )
                adata_epi.obs["copykat_prediction"] = [
                    bc_to_ckpred.get(b, "not.defined")
                    for b in adata_epi.obs_names
                ]

                copykat_available = True

                print("\nCopyKAT completed.")
                print("  Prediction counts (query cells):")
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
        # CopyKAT uses "aneuploid" to label malignant cells
        ck_mal = adata_epi.obs["copykat_prediction"].str.lower() == "aneuploid"

        if malignant_strategy == "intersection":
            malignant_mask = scm_mal & ck_mal
            strategy_label = "intersection (scMalignantFinder AND CopyKAT)"
        elif malignant_strategy == "copykat":
            malignant_mask = ck_mal
            strategy_label = "CopyKAT only"
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
            "Check scMalignantFinder model path and gene overlap.\n"
            "If using 'intersection' strategy, consider switching to "
            "'scMalignant' or 'copykat' to see individual tool results."
        )

    # ------------------------------------------------------------------
    # STEP 6 — Non-epithelial "rest" group from adata_full
    # ------------------------------------------------------------------
    print("\n--- Step 6: Extract non-epithelial 'rest' group ---")
    rest_mask  = ~ep_mask
    adata_rest = adata_full[rest_mask].copy()
    print(f"Non-epithelial 'rest' cells: {adata_rest.n_obs}")

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

    # Echo QC params
    adata_mal.uns["qc_params"] = (
        {"min_genes": min_genes, "max_mt": max_mt}
        if qc_active else None
    )

    # Store CopyKAT results (replaces infercna_results)
    if copykat_pred_df is not None:
        final_barcodes = set(adata_mal.obs_names)
        ck_stored      = copykat_pred_df.copy()
        ck_stored["in_final_output"] = ck_stored["cell.names"].isin(final_barcodes)
        adata_mal.uns["copykat_results"] = ck_stored

        print(
            f"\nCopyKAT results stored in adata.uns['copykat_results']:\n"
            f"  Shape: {ck_stored.shape[0]} rows × {ck_stored.shape[1]} columns\n"
            f"  Columns: {list(ck_stored.columns)}\n"
            f"  Prediction summary (all epithelial query cells):\n"
            + ck_stored["copykat.pred"].value_counts().to_string()
            + f"\n  Cells in final malignant output: "
            f"{ck_stored['in_final_output'].sum()}"
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
