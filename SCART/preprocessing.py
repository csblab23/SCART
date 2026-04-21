"""
preprocessing.py
Module 3 — Preprocessing, malignancy detection, and surfaceome DEG

═══════════════════════════════════════════════════════════════════════════════
CORRECT BIOLOGICAL PIPELINE DESIGN
═══════════════════════════════════════════════════════════════════════════════

Step 1  Load the full PopV-annotated h5ad (all cell types, e.g. 15202 cells).
        Save this as adata_full — it is the complete dataset.

Step 2  Extract epithelial cells only from adata_full → apply QC filters.
        QC thresholds (min_genes, max_mt) are read automatically from
        adata.uns['qc_params'] written by Module 1 (geo_fetcher.py).
        These are the cells we want to classify as malignant or not.

Step 3  Run scMalignantFinder + inferCNA on the epithelial cells.
        Combine predictions → final_malignant column on the epithelial adata.
        Strategy: intersection (scMalignantFinder AND inferCNA must both agree).

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
  SampleAnnotator("GSE…", min_genes=200, max_mt=40) stores QC thresholds in
  adata.uns['qc_params'] = {"min_genes": 200, "max_mt": 40.0} of every h5ad
  it writes.

Module 2 (popv_annotation.py)
  Passes the h5ad through — uns keys including 'qc_params' are preserved.

Module 3 (this file)
  Reads adata.uns['qc_params'] automatically.
  Logs which values it will use so the user can verify them.
  Falls back to defaults (min_genes=200, max_mt=40) if the key is absent
  (e.g. when adata was passed directly without going through Module 1).

═══════════════════════════════════════════════════════════════════════════════
DATA FLOW ACROSS MODULES
═══════════════════════════════════════════════════════════════════════════════

Module 1 (geo_fetcher.py)
  Saves:  GSE*_tumor.h5ad
  Contains: adata.layers['counts']          raw integer counts, full gene space
            adata.raw = adata               same data frozen
            adata.uns['cancer_type']
            adata.uns['qc_params']          {"min_genes": …, "max_mt": …}

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
            → obs: scMalignantFinder_prediction, infercna_prediction,
                   infercna_cna_signal, infercna_cna_cor,
                   infercna_gmm_label, final_malignant
            → uns: infercna_results (full detail DataFrame)
                   filtered_deg, all_deg, deg_params, cancer_type, qc_params

═══════════════════════════════════════════════════════════════════════════════
KEY FIXES
═══════════════════════════════════════════════════════════════════════════════

FIX 1   Full-gene route priority for scMalignantFinder:
        A-new (layers['full_counts']) → A-rescue (auto-detect Module 1 h5ad)
        → A-old (adata.raw) → C (4000-HVG fallback)

FIX 2   inferCNA reference subsampled to infercna_ref_max_cells (default 2000).

FIX 3   DEG uses pvals_adj (BH-adjusted), not raw pvals.

FIX 4   inferCNA n auto-capped to (n_common_genes - 1).

FIX 5   _get_raw_matrix prefers adata.raw over HVG layers (more genes).

FIX 6   scMalignantFinder predictions aligned by obs_names, not positional.

FIX 7   inferCNA findMalignant() scalop-incompatibility workaround.
        scalop >= 0.2.4 removed split_by_sample_names; we bypass findMalignant
        entirely and run a Python-side 2-component GMM on (cnaSignal, cnaCor).

FIX 8   inferCNA results saved in full detail:
        Per-cell: cna_signal, cna_cor, gmm_label stored as obs columns.
        Full table: adata.uns['infercna_results'] DataFrame.

FIX 9   CORRECT DEG DESIGN:
        Malignant epithelial cells vs all NON-EPITHELIAL cells from adata_full.
        Both groups surfaceome-filtered before DEG.
        Non-malignant epithelial cells are REMOVED (not used in any group).

QC FIX  min_genes and max_mt removed from the public API of
        run_preprocessing_pipeline().  Both values are read from
        adata.uns['qc_params'] (written by Module 1).  Sensible defaults
        (200 / 40) are used as a fallback so the pipeline still works when
        an h5ad was not produced by Module 1.
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

# ── QC fallback defaults (used only when uns['qc_params'] is absent) ───────
_DEFAULT_MIN_GENES = 200
_DEFAULT_MAX_MT    = 40.0


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

    Falls back to package defaults if the key is absent so the pipeline
    works even when adata was not produced by Module 1.

    Returns
    -------
    min_genes : int
    max_mt    : float
    source    : str   Human-readable description of where the values came from.
    """
    qc = adata.uns.get("qc_params", None)

    if qc is not None:
        min_genes = int(qc.get("min_genes", _DEFAULT_MIN_GENES))
        max_mt    = float(qc.get("max_mt",    _DEFAULT_MAX_MT))
        source    = "adata.uns['qc_params'] (set by Module 1)"
    else:
        min_genes = _DEFAULT_MIN_GENES
        max_mt    = _DEFAULT_MAX_MT
        source    = (
            f"package defaults ({_DEFAULT_MIN_GENES} / {_DEFAULT_MAX_MT}) — "
            "'qc_params' key not found in adata.uns. "
            "Re-run Module 1 with SampleAnnotator(min_genes=…, max_mt=…) "
            "to propagate custom thresholds automatically."
        )

    return min_genes, max_mt, source


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
# FIX 7 — Python-side GMM replacing inferCNA findMalignant()
# ═══════════════════════════════════════════════════════════════════════════

def _python_find_malignant(cna_df, q_barcodes, signal_threshold=0.9):
    """
    Python replacement for inferCNA's findMalignant() which crashes when
    scalop >= 0.2.4 removes split_by_sample_names.

    Algorithm (mirrors the original R implementation):
      1. cnaSignal = mean of top-(1-signal_threshold) absolute CNA values per cell
      2. cnaCor    = Pearson correlation of each cell's profile vs tumour-mean profile
      3. 2-component GMM on (cnaSignal, cnaCor)
      4. Higher-signal cluster → 'malignant'

    FIX 8: returns a DataFrame with per-cell scores for full result storage.

    Returns
    -------
    pd.DataFrame with columns:
        barcode             cell barcode
        cna_signal          cnaSignal value
        cna_cor             cnaCor value
        gmm_label           raw GMM cluster (0 or 1)
        infercna_prediction  'malignant' | 'non-malignant' | 'not.defined'
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError(
            "scikit-learn required for Python-side GMM.\n"
            "conda install -c conda-forge scikit-learn"
        )

    q_cols    = [c for c in q_barcodes if c in cna_df.columns]
    cna_query = cna_df[q_cols].values      # genes × cells
    n_genes, n_cells = cna_query.shape

    empty_row = pd.DataFrame({
        "barcode"             : list(q_barcodes),
        "cna_signal"          : np.nan,
        "cna_cor"             : np.nan,
        "gmm_label"           : -1,
        "infercna_prediction" : "not.defined",
    })

    if n_cells == 0:
        logger.warning("GMM: no query barcodes in CNA matrix.")
        return empty_row

    # Step 1 — cnaSignal
    top_k      = max(1, int(n_genes * (1.0 - signal_threshold)))
    abs_cna    = np.abs(cna_query)
    top_vals   = np.sort(abs_cna, axis=0)[-top_k:, :]
    cna_signal = top_vals.mean(axis=0)           # (n_cells,)

    # Step 2 — cnaCor
    tumour_mean = cna_query.mean(axis=1)          # (n_genes,)
    cna_cor     = np.array([
        np.corrcoef(cna_query[:, i], tumour_mean)[0, 1]
        for i in range(n_cells)
    ])
    cna_cor = np.nan_to_num(cna_cor, nan=0.0)

    # Step 3 — GMM
    X_gmm = np.column_stack([cna_signal, cna_cor])
    try:
        gmm    = GaussianMixture(n_components=2, covariance_type="full",
                                 random_state=42, max_iter=300)
        gmm.fit(X_gmm)
        labels = gmm.predict(X_gmm)

        # Step 4 — higher mean cnaSignal → malignant
        mean0       = cna_signal[labels == 0].mean()
        mean1       = cna_signal[labels == 1].mean()
        mal_cluster = 1 if mean1 > mean0 else 0

        pred_labels = np.where(labels == mal_cluster, "malignant", "non-malignant")

        result_df = pd.DataFrame({
            "barcode"             : q_cols,
            "cna_signal"          : cna_signal,
            "cna_cor"             : cna_cor,
            "gmm_label"           : labels.astype(int),
            "infercna_prediction" : pred_labels,
        })

        # Reindex to match all q_barcodes (some may be missing from CNA cols)
        result_full = pd.DataFrame({"barcode": list(q_barcodes)})
        result_full = result_full.merge(result_df, on="barcode", how="left")
        result_full["infercna_prediction"] = (
            result_full["infercna_prediction"].fillna("not.defined")
        )
        result_full["gmm_label"] = result_full["gmm_label"].fillna(-1).astype(int)

        logger.info(
            "Python GMM findMalignant results:\n"
            + result_full["infercna_prediction"].value_counts().to_string()
        )
        return result_full

    except Exception as exc:
        logger.warning(f"GMM failed: {exc}. Returning all 'not.defined'.")
        return empty_row.assign(barcode=list(q_barcodes))


# ═══════════════════════════════════════════════════════════════════════════
# inferCNA runner
# ═══════════════════════════════════════════════════════════════════════════

def _run_infercna(
    adata_query,
    adata_ref,
    genome="hg19",
    n=5000,
    noise=0.1,
    signal_threshold=0.9,
    ref_max_cells=2000,
):
    """
    Run inferCNA.

    FIX 4: n auto-capped to (n_common - 1).
    FIX 5: _get_raw_matrix prefers adata.raw.
    FIX 7: findMalignant bypassed; Python GMM used instead.
    FIX 8: returns full per-cell DataFrame (scores + labels).

    Returns
    -------
    pd.DataFrame  columns: barcode, cna_signal, cna_cor, gmm_label,
                           infercna_prediction
                  indexed 0..n_query-1, barcode matches adata_query.obs_names
    """
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
        importr("infercna")
    except Exception as exc:
        raise ImportError(
            "R package 'infercna' not found.\n"
            "In R: devtools::install_github('jlaffy/infercna')"
        ) from exc

    # FIX 2 — subsample reference epithelial cells
    EPITHELIAL = {
        "epithelial cell",
        "glandular epithelial cell",
        "ovarian surface epithelial cell",
    }
    if "cell_ontology_class" in adata_ref.obs.columns:
        ep_mask      = adata_ref.obs["cell_ontology_class"].str.lower().isin(EPITHELIAL)
        adata_ref_ep = adata_ref[ep_mask].copy() if ep_mask.any() else adata_ref.copy()
        logger.info(f"inferCNA reference epithelial cells: {adata_ref_ep.n_obs}")
    else:
        adata_ref_ep = adata_ref.copy()

    if adata_ref_ep.n_obs > ref_max_cells:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(adata_ref_ep.n_obs, size=ref_max_cells, replace=False)
        adata_ref_ep = adata_ref_ep[np.sort(idx)].copy()
        logger.info(f"Reference subsampled to {ref_max_cells} cells.")

    # Log-CPM matrices (FIX 5)
    def _to_log_cpm(obj):
        X  = _get_raw_matrix(obj)
        rs = X.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        return np.log1p(X / rs * 1e6).T   # genes × cells

    logger.info("inferCNA: building query log-CPM ...")
    mat_query = _to_log_cpm(adata_query)
    logger.info("inferCNA: building reference log-CPM ...")
    mat_ref   = _to_log_cpm(adata_ref_ep)

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

    common   = np.intersect1d(q_genes, r_genes)
    n_common = len(common)
    logger.info(f"inferCNA common genes: {n_common}")

    if n_common < 200:
        raise ValueError(
            f"Only {n_common} common genes. Need >= 200.\n"
            "Both datasets must use HGNC gene symbols."
        )
    if n_common < 2000:
        logger.warning(f"Only {n_common} common genes — inferCNA prefers 5000+.")

    # FIX 4 — auto-cap n
    n_safe = min(n, n_common - 1)
    if n_safe < n:
        logger.warning(f"FIX 4: n capped {n} → {n_safe} (n_common={n_common}).")

    q_idx = np.where(np.isin(q_genes, common))[0]
    r_idx = np.where(np.isin(r_genes, common))[0]

    mat_combined = np.hstack([mat_query[q_idx, :], mat_ref[r_idx, :]])
    sub_genes    = q_genes[q_idx]
    q_barcodes   = np.array(adata_query.obs_names)
    ref_barcodes = np.array(["REF_" + b for b in adata_ref_ep.obs_names])
    all_barcodes = np.concatenate([q_barcodes, ref_barcodes])

    logger.info(
        f"inferCNA matrix: {mat_combined.shape[0]} genes × {mat_combined.shape[1]} cells "
        f"({len(q_barcodes)} query + {len(ref_barcodes)} ref), n_safe={n_safe}"
    )

    # Transfer to R and run infercna() ONLY — NOT findMalignant() (FIX 7)
    logger.info("inferCNA: transferring to R ...")
    ro.globalenv["mat_flat"]     = ro.FloatVector(mat_combined.flatten(order="F").tolist())
    ro.globalenv["n_rows"]       = ro.IntVector([mat_combined.shape[0]])
    ro.globalenv["n_cols"]       = ro.IntVector([mat_combined.shape[1]])
    ro.globalenv["all_barcodes"] = ro.StrVector(all_barcodes.tolist())
    ro.globalenv["ref_barcodes"] = ro.StrVector(ref_barcodes.tolist())
    ro.globalenv["gene_names"]   = ro.StrVector(sub_genes.tolist())
    ro.globalenv["n_genes"]      = ro.IntVector([n_safe])
    ro.globalenv["noise_val"]    = ro.FloatVector([noise])
    ro.globalenv["genome_name"]  = ro.StrVector([genome])

    logger.info(
        "inferCNA: running R (useGenome + infercna only — findMalignant bypassed) ..."
    )
    ro.r("""
        suppressPackageStartupMessages(library(infercna))
        r_mat <- matrix(mat_flat, nrow=n_rows, ncol=n_cols)
        rownames(r_mat) <- gene_names
        colnames(r_mat) <- all_barcodes
        useGenome(genome_name)
        cna <- infercna(
            m        = r_mat,
            refCells = list(normal_ref = ref_barcodes),
            n        = n_genes,
            noise    = noise_val,
            isLog    = TRUE,
            verbose  = FALSE
        )
    """)

    # Pull CNA matrix back to Python
    cna_r = ro.globalenv["cna"]
    try:
        cna_cols   = list(cna_r.colnames)
        cna_rows   = list(cna_r.rownames)
        cna_values = np.array(cna_r).reshape(len(cna_rows), len(cna_cols), order="F")
        cna_df     = pd.DataFrame(cna_values, index=cna_rows, columns=cna_cols)
        logger.info(
            f"CNA matrix retrieved: {cna_df.shape[0]} genes × {cna_df.shape[1]} cells"
        )
    except Exception as exc:
        logger.error(f"Could not convert CNA matrix: {exc}")
        return pd.DataFrame({
            "barcode"            : list(q_barcodes),
            "cna_signal"         : np.nan,
            "cna_cor"            : np.nan,
            "gmm_label"          : -1,
            "infercna_prediction": "not.defined",
        })

    # FIX 7 — Python GMM replaces findMalignant()
    logger.info("FIX 7: Python-side GMM findMalignant replacement ...")
    result_df = _python_find_malignant(cna_df, q_barcodes, signal_threshold)
    logger.info(
        "inferCNA predictions:\n"
        + result_df["infercna_prediction"].value_counts().to_string()
    )
    return result_df


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
    # inferCNA
    infercna_genome="hg19",
    infercna_n=5000,
    infercna_noise=0.1,
    infercna_signal_threshold=0.9,
    infercna_ref_max_cells=2000,
):
    """
    Full preprocessing pipeline.

    QC THRESHOLDS (min_genes, max_mt)
    ----------------------------------
    These are NO LONGER parameters of this function.  They are set once in
    Module 1::

        annotator = SampleAnnotator("GSE…", min_genes=200, max_mt=40)

    Module 1 stores them in ``adata.uns['qc_params']`` of every h5ad it
    writes.  Module 3 reads them from that key automatically, so you never
    need to repeat them here.

    If ``qc_params`` is absent (e.g. you supplied an h5ad that did not come
    from Module 1), the pipeline falls back to ``min_genes=200, max_mt=40``
    and prints a warning.

    PIPELINE FLOW
    -------------
    1.  Load full PopV h5ad (all cell types)  → adata_full
    2.  Read QC thresholds from adata.uns['qc_params']
    3.  Extract epithelial cells → QC
    4.  scMalignantFinder + inferCNA on epithelial cells
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
        Tabula Sapiens h5ad for inferCNA normal reference.
        inferCNA skipped if None.
    tumor_h5ad : str or None
        Module 1 h5ad for Route A-rescue. Auto-detected if None.
    save_dir : str or None
        Output directory. Default 'preprocessing_results/' in cwd.
    scmalignant_model_dir : str or None  Auto-detected from SCART.
    surfaceome_path : str or None        Auto-detected from SCART GESP file.
    malignant_strategy : str
        'intersection' — malignant only if BOTH tools agree (default)
        'scMalignant'  — scMalignantFinder only
        'infercna'     — inferCNA only (requires reference_h5ad)
    infercna_genome : str   'hg19' (default) or 'hg38'. String key, not file.
    infercna_n : int        CNA top genes. Default 5000. Auto-capped (FIX 4).
    infercna_noise : float  Noise floor. Default 0.1.
    infercna_signal_threshold : float  GMM cnaSignal top fraction. Default 0.9.
    infercna_ref_max_cells : int       Max reference cells. Default 2000.

    Returns
    -------
    AnnData  Malignant epithelial cells only, surfaceome-filtered, binarised.
             DEG stored in adata.uns['filtered_deg'] and adata.uns['all_deg'].
             inferCNA full results in adata.uns['infercna_results'].
             QC params echoed in adata.uns['qc_params'].
    """
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
    min_genes, max_mt, qc_source = _read_qc_params(adata_full)

    print(f"\n--- Step 2: QC thresholds ---")
    print(f"  Source    : {qc_source}")
    print(f"  min_genes : {min_genes}")
    print(f"  max_mt    : {max_mt}")

    # ------------------------------------------------------------------
    # STEP 3 — Extract epithelial cells → QC
    # ------------------------------------------------------------------
    print("\n--- Step 3: Epithelial selection + QC ---")
    labels  = adata_full.obs["popv_majority_vote_prediction"].astype(str)
    ep_mask = labels.str.endswith("epithelial cell")
    print(f"Epithelial cells: {ep_mask.sum()} / {adata_full.n_obs} total")
    print(
        f"Non-epithelial cells (will be 'rest' group for DEG): {(~ep_mask).sum()}"
    )

    adata_epi = adata_full[ep_mask].copy()

    adata_epi.var["mt"] = adata_epi.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata_epi, qc_vars=["mt"], inplace=True)
    print(f"Mean MT% BEFORE QC: {adata_epi.obs['pct_counts_mt'].mean():.2f}")
    before_qc = adata_epi.n_obs
    adata_epi = adata_epi[
        (adata_epi.obs["n_genes_by_counts"] > min_genes) &
        (adata_epi.obs["pct_counts_mt"]     < max_mt)
    ].copy()
    print(
        f"Epithelial cells after QC: {adata_epi.n_obs}  "
        f"(removed {before_qc - adata_epi.n_obs}  |  "
        f"min_genes>{min_genes}, max_mt<{max_mt})"
    )
    print(f"Mean MT% AFTER QC:  {adata_epi.obs['pct_counts_mt'].mean():.2f}\n")

    # Route raw counts → .X; snapshot for inferCNA
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
    # STEP 4b — inferCNA (FIX 7: scalop-safe Python GMM)
    # ------------------------------------------------------------------
    infercna_available = False
    infercna_result_df = None

    if malignant_strategy in ("infercna", "intersection"):
        if reference_h5ad is None:
            print(
                "\nWarning: inferCNA skipped — no reference_h5ad provided.\n"
                "  Falling back to scMalignantFinder only."
            )
            malignant_strategy = "scMalignant"
        else:
            print(
                f"\n--- Step 4b: inferCNA ---\n"
                f"  Reference: {reference_h5ad}\n"
                f"  Reference subsampled to <={infercna_ref_max_cells} cells\n"
                f"  n auto-capped, scalop-safe Python GMM"
            )
            try:
                adata_raw_cna   = adata_epi.copy()
                adata_raw_cna.X = adata_epi.layers["raw_for_cna"]
                adata_ref_full  = sc.read_h5ad(reference_h5ad)

                infercna_result_df = _run_infercna(
                    adata_query      = adata_raw_cna,
                    adata_ref        = adata_ref_full,
                    genome           = infercna_genome,
                    n                = infercna_n,
                    noise            = infercna_noise,
                    signal_threshold = infercna_signal_threshold,
                    ref_max_cells    = infercna_ref_max_cells,
                )

                # FIX 8 — store detailed inferCNA results on adata_epi
                bc_to_pred   = dict(zip(infercna_result_df["barcode"],
                                        infercna_result_df["infercna_prediction"]))
                bc_to_signal = dict(zip(infercna_result_df["barcode"],
                                        infercna_result_df["cna_signal"]))
                bc_to_cor    = dict(zip(infercna_result_df["barcode"],
                                        infercna_result_df["cna_cor"]))
                bc_to_gmm    = dict(zip(infercna_result_df["barcode"],
                                        infercna_result_df["gmm_label"]))

                adata_epi.obs["infercna_prediction"]  = [
                    bc_to_pred.get(b,   "not.defined") for b in adata_epi.obs_names
                ]
                adata_epi.obs["infercna_cna_signal"]  = [
                    bc_to_signal.get(b, np.nan)         for b in adata_epi.obs_names
                ]
                adata_epi.obs["infercna_cna_cor"]     = [
                    bc_to_cor.get(b,    np.nan)         for b in adata_epi.obs_names
                ]
                adata_epi.obs["infercna_gmm_label"]   = [
                    bc_to_gmm.get(b,    -1)             for b in adata_epi.obs_names
                ]

                infercna_available = True

                print("\ninferCNA completed.")
                print("  Prediction counts:")
                print(adata_epi.obs["infercna_prediction"].value_counts().to_string())
                print(
                    f"\n  cna_signal stats:  "
                    f"mean={adata_epi.obs['infercna_cna_signal'].mean():.4f}  "
                    f"std={adata_epi.obs['infercna_cna_signal'].std():.4f}"
                )
                print(
                    f"  cna_cor stats:     "
                    f"mean={adata_epi.obs['infercna_cna_cor'].mean():.4f}  "
                    f"std={adata_epi.obs['infercna_cna_cor'].std():.4f}"
                )

            except Exception as exc:
                print(
                    f"\nWarning: inferCNA failed — {type(exc).__name__}: {exc}\n"
                    "  Falling back to scMalignantFinder only."
                )
                logger.exception("inferCNA error:")
                malignant_strategy = "scMalignant"

    # ------------------------------------------------------------------
    # STEP 4c — Combine malignancy calls → final_malignant
    # ------------------------------------------------------------------
    print("\n--- Step 4c: Combine malignancy calls ---")
    scm_mal = adata_epi.obs[scm_col].str.lower() == "malignant"

    if infercna_available:
        cna_mal = adata_epi.obs["infercna_prediction"].str.lower() == "malignant"
        if malignant_strategy == "intersection":
            malignant_mask  = scm_mal & cna_mal
            strategy_label  = "intersection (scMalignantFinder AND inferCNA)"
        elif malignant_strategy == "infercna":
            malignant_mask  = cna_mal
            strategy_label  = "inferCNA only"
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
    print("  Cell types in rest group:")
    print(
        adata_rest.obs["popv_majority_vote_prediction"]
        .value_counts()
        .head(15)
        .to_string()
    )

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

    # Echo the QC params used so they stay with the output file
    adata_mal.uns["qc_params"] = {"min_genes": min_genes, "max_mt": max_mt}

    # FIX 8 — store full inferCNA results
    if infercna_result_df is not None:
        final_barcodes         = set(adata_mal.obs_names)
        infercna_stored        = infercna_result_df.copy()
        infercna_stored["in_final_output"] = infercna_stored["barcode"].isin(
            final_barcodes
        )
        adata_mal.uns["infercna_results"] = infercna_stored

        print(
            f"\ninferCNA results stored in adata.uns['infercna_results']:\n"
            f"  Shape: {infercna_stored.shape[0]} rows × "
            f"{infercna_stored.shape[1]} columns\n"
            f"  Columns: {list(infercna_stored.columns)}\n"
            f"  Prediction summary (all epithelial cells before filtering):\n"
            + infercna_stored["infercna_prediction"].value_counts().to_string()
            + f"\n  Cells in final malignant output: "
            f"{infercna_stored['in_final_output'].sum()}"
        )
    else:
        adata_mal.uns["infercna_results"] = None
        print("\ninferCNA was not run — adata.uns['infercna_results'] = None")

    # Clean string columns
    for col in adata_mal.obs.columns:
        if adata_mal.obs[col].dtype == object:
            adata_mal.obs[col] = adata_mal.obs[col].astype(str)

    final_path = os.path.join(save_dir, "final_tumor.h5ad")
    adata_mal.write(final_path)

    # ------------------------------------------------------------------
    # Final summary report
    # ------------------------------------------------------------------
    print(f"\nFinal object saved to: {final_path}")
    print(
        f"\n{'═'*60}\n"
        f"PIPELINE SUMMARY\n"
        f"{'═'*60}\n"
        f"Full dataset (Module 2 output):    {adata_full.n_obs} cells\n"
        f"Epithelial cells (pre-QC):         {ep_mask.sum()}\n"
        f"Epithelial cells (post-QC):        {before_qc}  →  {adata_epi.n_obs}\n"
        f"  QC: min_genes>{min_genes}, max_mt<{max_mt}  [{qc_source}]\n"
        f"Malignant epithelial (kept):       {adata_mal.n_obs}\n"
        f"Non-malignant epithelial (removed):{(~malignant_mask).sum()}\n"
        f"Non-epithelial rest (DEG ref):     {adata_rest.n_obs}\n"
        f"{'─'*60}\n"
        f"Malignancy strategy:               {strategy_label}\n"
        f"Surfaceome genes (GESP):           {len(surf_genes)} → {len(surf_common)} common\n"
        f"DEG comparison:                    malignant epithelial vs non-epithelial rest\n"
        f"DEGs passing filter:               {filtered_deg.shape[0]}\n"
        f"{'─'*60}\n"
        f"Saved adata shape:                 {adata_mal.shape}\n"
        f"  obs columns: {list(adata_mal.obs.columns)}\n"
        f"  uns keys:    {list(adata_mal.uns.keys())}\n"
        f"  layers:      {list(adata_mal.layers.keys())}\n"
        f"{'═'*60}"
    )
    print("\n========== PREPROCESSING COMPLETED ==========\n")
    return adata_mal
