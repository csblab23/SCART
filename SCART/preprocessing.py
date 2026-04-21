"""
preprocessing.py
Module 3 — Preprocessing, malignancy detection, and surfaceome DEG

Data flow across modules
------------------------
Module 1 (geo_fetcher.py)
  Saves:  GSE*_tumor.h5ad
  Contains: adata.layers['counts']  (raw integer counts, full gene space ~33k)
            adata.raw = adata        (same data frozen in .raw)
            adata.uns['cancer_type']

Module 2 (popv_annotation.py)
  Reads:  GSE*_tumor.h5ad
  Saves:  popv_results/final_popv_annotated.h5ad
  Contains (after FIX 8 layer approach):
            adata.layers['full_counts']          (raw counts, full gene space)
            adata.uns['full_counts_var_names']   (list of gene names)
            adata.layers['scvi_counts']          (4000 HVG subset)
            popv_majority_vote_prediction column

Module 3 (this file)
  Reads:  popv_results/final_popv_annotated.h5ad  (auto-detected)
  Uses:   layers['full_counts']  → Route A-new  (Module 2 FIX 8 path)
          Auto-found Module 1 h5ad from cwd     → Route A-rescue (no user input)
          adata.raw                              → Route A-old
          4000 HVG fallback                     → Route C

Key fixes
---------
FIX 1  Full-gene rescue priority: A-new → A-rescue (auto-detected) → A-old → C.
       tumor_h5ad is OPTIONAL — auto-detected from cwd/GSE_data/ without
       any user input.

FIX 2  inferCNA reference subsampled to infercna_ref_max_cells=2000.

FIX 3  DEG uses pvals_adj (BH-adjusted) not raw pvals.

FIX 4  inferCNA: auto-cap n to (n_common_genes - 1).

FIX 5  inferCNA reference: prefer adata.raw over HVG layers.

FIX 6  scMalignantFinder output aligned by obs_names not positional .values.

FIX 7  [NEW] inferCNA findMalignant() scalop incompatibility workaround.
       The error "'split_by_sample_names' is not an exported object from
       'namespace:scalop'" means a newer scalop version removed that symbol.
       Fix: downgrade scalop to the version inferCNA was built against, OR
       call infercna's CNA signal/correlation metrics directly and do the
       bimodal clustering in Python (scipy GMM), bypassing findMalignant().
       This module implements the Python-side fallback automatically.

FIX 8  [NEW] Normal-cell fallback when scMalignantFinder classifies
       nearly all cells as malignant (<10 normal cells).
       In high-purity tumor samples scMalignantFinder is correct to call
       99%+ cells malignant, but this leaves too few "normal" cells for a
       Wilcoxon DEG test.
       Fallback strategy:
         (a) Use reference epithelial cells from the Tabula Sapiens h5ad
             as the "normal" comparison group — biologically more meaningful
             than the rare misclassified cells in the query.
         (b) Persist the reference epithelial cells in adata so DEG has
             two meaningful groups.
       If reference_h5ad is None the pipeline still runs; it simply warns
       that the DEG will be uninformative with <10 normal cells.
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


# ===========================================================================
# Auto-detect paths
# ===========================================================================

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
            "Could not auto-detect scMalignantFinder model directory.\n"
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
        "Could not auto-detect surfaceome CSV inside SCART package.\n"
        "Pass surfaceome_path= explicitly."
    )


def _auto_tumor_h5ad():
    """
    Auto-detect the Module 1 tumor h5ad from cwd and GSE_data/.
    Returns the most recently created match, or None if nothing found.
    """
    patterns   = ["*_tumor.h5ad", "combined_tumor.h5ad", "input_tumor.h5ad"]
    search_dirs = [os.getcwd(), os.path.join(os.getcwd(), "GSE_data")]
    files = []
    for d in search_dirs:
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(d, pattern)))
    files = list(set(files))
    if not files:
        return None
    found = max(files, key=os.path.getctime)
    logger.info(f"Auto-detected Module 1 tumor h5ad: {found}")
    return found


def _auto_popv_h5ad():
    """Auto-detect the Module 2 PopV output h5ad."""
    candidates = [
        os.path.join("popv_results", "final_popv_annotated.h5ad"),
        "final_popv_annotated.h5ad",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# ===========================================================================
# FIX 1 — Build full-gene AnnData for scMalignantFinder
# ===========================================================================

def _build_fullgene_adata_for_scm(adata, feature_tsv, tumor_h5ad_path=None):
    """
    Return a log-normalised AnnData covering the full original gene space.

    Route priority
    --------------
    A-new     layers['full_counts'] + uns['full_counts_var_names']
              (Module 2 FIX 8 — primary path)
    A-rescue  Module 1 tumor h5ad auto-detected from cwd/GSE_data/
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

    def _make_adata(X, obs, var_names_or_df):
        if sp.issparse(X):
            X = X.toarray()
        X   = X.astype(np.float32)
        var = var_names_or_df if isinstance(var_names_or_df, pd.DataFrame) \
              else pd.DataFrame(index=list(var_names_or_df))
        af  = sc.AnnData(X=X, obs=obs, var=var)
        sc.pp.normalize_total(af, target_sum=1e4)
        sc.pp.log1p(af)
        af.X = sp.csr_matrix(af.X)
        return af

    # Route A-new
    if "full_counts" in adata.layers:
        var_names = adata.uns.get("full_counts_var_names", None)
        if var_names is not None and \
                len(var_names) == adata.layers["full_counts"].shape[1]:
            ov = _pct(var_names)
            logger.info(
                f"Route A-new (layers['full_counts']): "
                f"{len(var_names)} genes, {ov:.1f}% overlap"
            )
            if ov >= 50:
                af = _make_adata(adata.layers["full_counts"], adata.obs.copy(), var_names)
                logger.info(f"scMalignantFinder → Route A-new ({af.n_vars} genes, {ov:.1f}% overlap).")
                return af
            logger.warning(f"Route A-new overlap {ov:.1f}% < 50% — trying Route A-rescue.")
        else:
            logger.warning("layers['full_counts'] gene names mismatch — trying Route A-rescue.")

    # Route A-rescue
    rescue_path = tumor_h5ad_path or _auto_tumor_h5ad()
    if rescue_path is not None and os.path.exists(rescue_path):
        logger.info(f"Route A-rescue: loading from {rescue_path}")
        try:
            adata_m1 = sc.read_h5ad(rescue_path)
            shared = sorted(set(adata.obs_names) & set(adata_m1.obs_names))

            if adata_m1.raw is not None and len(shared) > 0:
                ov = _pct(adata_m1.raw.var_names)
                logger.info(f"Route A-rescue (adata_m1.raw): {adata_m1.raw.n_vars} genes, {ov:.1f}% overlap")
                if ov >= 50:
                    current_order = [c for c in adata.obs_names if c in set(shared)]
                    m1_sub = adata_m1[current_order]
                    af = _make_adata(m1_sub.raw.X, adata.obs.loc[current_order].copy(), m1_sub.raw.var.copy())
                    logger.info(f"scMalignantFinder → Route A-rescue ({af.n_vars} genes, {ov:.1f}% overlap).")
                    return af

            for lyr in ("counts", "raw_counts", "scvi_counts"):
                if lyr in adata_m1.layers and len(shared) > 0:
                    ov = _pct(adata_m1.var_names)
                    logger.info(f"Route A-rescue (layers['{lyr}']): {adata_m1.n_vars} genes, {ov:.1f}% overlap")
                    if ov >= 50:
                        current_order = [c for c in adata.obs_names if c in set(shared)]
                        m1_sub = adata_m1[current_order]
                        af = _make_adata(m1_sub.layers[lyr], adata.obs.loc[current_order].copy(), m1_sub.var.copy())
                        logger.info(f"scMalignantFinder → Route A-rescue (layers['{lyr}'], {af.n_vars} genes).")
                        return af
                    break
        except Exception as exc:
            logger.warning(f"Route A-rescue failed: {exc}. Trying Route A-old.")

    # Route A-old
    if adata.raw is not None:
        ov = _pct(adata.raw.var_names)
        logger.info(f"Route A-old (adata.raw): {adata.raw.n_vars} genes, {ov:.1f}% overlap")
        if ov >= 50:
            af = _make_adata(adata.raw.X, adata.obs.copy(), adata.raw.var.copy())
            logger.info("scMalignantFinder → Route A-old.")
            return af
        logger.warning(f"Route A-old overlap {ov:.1f}% < 50% — trying Route B.")

    # Route B
    if "full_var_names" in adata.uns:
        full_var = list(adata.uns["full_var_names"])
        ov       = _pct(full_var)
        logger.info(f"Route B (uns): {len(full_var)} genes, {ov:.1f}% overlap")
        for lyr in ("scvi_counts", "raw_counts", "counts"):
            if lyr in adata.layers:
                X = adata.layers[lyr]
                if sp.issparse(X):
                    X = X.toarray()
                if X.shape[1] == len(full_var) and ov >= 50:
                    af = _make_adata(X, adata.obs.copy(), full_var)
                    logger.info(f"scMalignantFinder → Route B (layers['{lyr}']).")
                    return af

    # Route C — fallback
    ov_hvg = _pct(adata.var_names)
    logger.warning(
        f"All routes failed. Falling back to {adata.n_vars} HVGs ({ov_hvg:.1f}% overlap).\n"
        "Ensure GSE*_tumor.h5ad is in cwd, or re-run Module 2 with FIX 8."
    )
    return adata.copy()


# ===========================================================================
# FIX 5 — Helper: raw count matrix (adata.raw first, then layers)
# ===========================================================================

def _get_raw_matrix(adata):
    """
    Return a dense float64 (cells x genes) array of raw integer counts.
    Prefers adata.raw over HVG layers when adata.raw has more genes.
    """
    if adata.raw is not None:
        raw_n   = adata.raw.n_vars
        layer_n = max(
            (adata.layers[l].shape[1]
             for l in ("full_counts", "scvi_counts", "raw_counts", "counts")
             if l in adata.layers),
            default=0
        )
        if raw_n > layer_n:
            logger.info(f"Raw counts from adata.raw ({raw_n} genes > best layer {layer_n} genes)")
            X = adata.raw.X
            if sp.issparse(X):
                X = X.toarray()
            return np.array(X, dtype=np.float64)

    for lyr in ("full_counts", "scvi_counts", "raw_counts", "counts"):
        if lyr in adata.layers:
            logger.info(f"Raw counts from adata.layers['{lyr}']")
            X = adata.layers[lyr]
            if sp.issparse(X):
                X = X.toarray()
            return np.array(X, dtype=np.float64)

    if adata.raw is not None:
        logger.info("Raw counts from adata.raw.X (fallback)")
        X = adata.raw.X
    else:
        logger.warning("No raw layer or adata.raw — using adata.X (may be log-normalised).")
        X = adata.X

    if sp.issparse(X):
        X = X.toarray()
    return np.array(X, dtype=np.float64)


# ===========================================================================
# FIX 7 — inferCNA findMalignant Python-side GMM fallback
# ===========================================================================

def _python_find_malignant(cna_scores_df, q_barcodes, signal_threshold=0.9):
    """
    Python-side bimodal GMM replacement for inferCNA's findMalignant().

    Called when findMalignant() crashes with the scalop error:
      'split_by_sample_names' is not an exported object from 'namespace:scalop'

    The original findMalignant does:
      1. Compute cnaSignal  = mean of top-N absolute CNA values per cell
      2. Compute cnaCor     = Pearson correlation of each cell's CNA profile
                              vs the tumour-average CNA profile
      3. Fit a 2-component Gaussian mixture to (cnaSignal, cnaCor)
      4. Label the high-signal / high-correlation cluster as 'malignant'

    We replicate steps 1–4 using scipy.stats and sklearn.mixture.

    Parameters
    ----------
    cna_scores_df : pd.DataFrame
        genes × cells CNA matrix (R cna object converted to pandas).
        Rows = genes, columns = query+ref cell barcodes.
    q_barcodes : array-like
        Query cell barcodes (used to subset cna_scores_df).
    signal_threshold : float
        Top fraction of genes used for cnaSignal (default 0.9 → top 10%).

    Returns
    -------
    pd.Series  Index = q_barcodes, values = 'malignant' | 'non-malignant'
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError(
            "scikit-learn is required for the Python-side findMalignant fallback.\n"
            "Install: conda install -c conda-forge scikit-learn"
        )

    # Subset to query cells only
    q_cols     = [c for c in q_barcodes if c in cna_scores_df.columns]
    cna_query  = cna_scores_df[q_cols].values   # genes × cells
    n_genes, n_cells = cna_query.shape

    if n_cells == 0:
        logger.warning("Python findMalignant: no query barcodes in CNA matrix.")
        return pd.Series("not.defined", index=q_barcodes, name="infercna_prediction")

    # Step 1: cnaSignal = mean of top-(1-signal_threshold) absolute CNA values
    top_k    = max(1, int(n_genes * (1.0 - signal_threshold)))
    abs_cna  = np.abs(cna_query)                          # genes × cells
    top_vals = np.sort(abs_cna, axis=0)[-top_k:, :]      # top-k rows
    cna_signal = top_vals.mean(axis=0)                    # shape (n_cells,)

    # Step 2: cnaCor = correlation of each cell vs tumour-mean CNA profile
    tumour_mean = cna_query.mean(axis=1)                  # shape (n_genes,)
    cna_cor     = np.array([
        np.corrcoef(cna_query[:, i], tumour_mean)[0, 1]
        for i in range(n_cells)
    ])
    cna_cor = np.nan_to_num(cna_cor, nan=0.0)

    # Step 3: 2-component GMM on (cnaSignal, cnaCor)
    X_gmm = np.column_stack([cna_signal, cna_cor])
    try:
        gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            random_state=42,
            max_iter=200,
        )
        gmm.fit(X_gmm)
        labels = gmm.predict(X_gmm)   # 0 or 1

        # Step 4: the malignant cluster has higher mean cnaSignal
        mean0 = cna_signal[labels == 0].mean()
        mean1 = cna_signal[labels == 1].mean()
        mal_cluster = 1 if mean1 > mean0 else 0

        pred_labels = np.where(labels == mal_cluster, "malignant", "non-malignant")
        preds = pd.Series(pred_labels, index=q_cols, name="infercna_prediction")
        logger.info(
            "Python GMM findMalignant:\n" + preds.value_counts().to_string()
        )
        return preds.reindex(q_barcodes, fill_value="not.defined")

    except Exception as exc:
        logger.warning(f"Python GMM failed: {exc}. All cells labelled 'not.defined'.")
        return pd.Series("not.defined", index=q_barcodes, name="infercna_prediction")


# ===========================================================================
# inferCNA  (via rpy2)
# ===========================================================================

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
    Run inferCNA (official tutorial step order).

    FIX 4: n auto-capped to (n_common_genes - 1).
    FIX 5: _get_raw_matrix() prefers adata.raw for the reference.
    FIX 7: if findMalignant() crashes with the scalop symbol error,
           automatically falls back to the Python-side GMM implementation.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        err_str = str(exc)
        if any(s in err_str for s in ("R_getVar", "undefined symbol", "R_ClosureEnv")):
            raise ImportError(
                "rpy2 compiled against a different R version.\n"
                "Fix:\n"
                "  conda activate scart\n"
                "  conda remove rpy2 --force\n"
                "  conda install -c conda-forge rpy2\n"
                f"Original error: {err_str}"
            ) from exc
        raise ImportError(
            f"rpy2 import failed: {exc}\n"
            "Install: conda install -c conda-forge rpy2\n"
            "R package: devtools::install_github('jlaffy/infercna')"
        ) from exc

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

    # Build log-CPM matrices (FIX 5)
    def _to_log_cpm(adata_obj):
        X  = _get_raw_matrix(adata_obj)
        rs = X.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        return np.log1p(X / rs * 1e6).T   # genes × cells

    logger.info("inferCNA: building query log-CPM matrix ...")
    mat_query = _to_log_cpm(adata_query)
    logger.info("inferCNA: building reference log-CPM matrix ...")
    mat_ref   = _to_log_cpm(adata_ref_ep)

    q_genes = np.array(adata_query.var_names)
    r_genes = np.array(adata_ref_ep.raw.var_names) \
              if adata_ref_ep.raw is not None and \
                 adata_ref_ep.raw.n_vars > max(
                     (adata_ref_ep.layers[l].shape[1]
                      for l in ("scvi_counts", "raw_counts", "counts")
                      if l in adata_ref_ep.layers),
                     default=0) \
              else np.array(adata_ref_ep.var_names)

    common   = np.intersect1d(q_genes, r_genes)
    n_common = len(common)
    logger.info(f"inferCNA common genes: {n_common}")

    if n_common < 200:
        raise ValueError(
            f"Only {n_common} common genes. Need >= 200.\n"
            "Check both datasets use HGNC gene symbols."
        )
    if n_common < 2000:
        logger.warning(f"Only {n_common} common genes — inferCNA works best with 5000+.")

    # FIX 4 — auto-cap n
    n_safe = min(n, n_common - 1)
    if n_safe < n:
        logger.warning(
            f"FIX 4: infercna_n={n} capped to {n_safe} "
            f"(must be < n_common_genes={n_common})."
        )

    q_idx = np.where(np.isin(q_genes, common))[0]
    r_idx = np.where(np.isin(r_genes,  common))[0]

    mat_combined = np.hstack([mat_query[q_idx, :], mat_ref[r_idx, :]])
    sub_genes    = q_genes[q_idx]
    q_barcodes   = np.array(adata_query.obs_names)
    ref_barcodes = np.array(["REF_" + b for b in adata_ref_ep.obs_names])
    all_barcodes = np.concatenate([q_barcodes, ref_barcodes])

    logger.info(
        f"inferCNA matrix: {mat_combined.shape[0]} genes x "
        f"{mat_combined.shape[1]} cells "
        f"({len(q_barcodes)} query + {len(ref_barcodes)} ref), n_safe={n_safe}"
    )

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

    logger.info("inferCNA: running R CNA inference (useGenome + infercna) ...")
    # -----------------------------------------------------------------------
    # FIX 7: Run useGenome() + infercna() ONLY in R. Do NOT call
    # findMalignant() in R because it depends on scalop::split_by_sample_names
    # which was removed in scalop >= 0.2.4.
    # The CNA matrix is pulled back to Python and the bimodal GMM clustering
    # is done by _python_find_malignant() instead.
    # -----------------------------------------------------------------------
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
            f"CNA matrix retrieved: {cna_df.shape[0]} genes x "
            f"{cna_df.shape[1]} cells"
        )
    except Exception as exc:
        logger.error(f"Could not convert R CNA matrix to Python: {exc}")
        return pd.Series("not.defined", index=q_barcodes, name="infercna_prediction")

    # FIX 7 — Python-side GMM instead of R findMalignant()
    logger.info("FIX 7: Running Python-side GMM findMalignant replacement ...")
    preds = _python_find_malignant(cna_df, q_barcodes, signal_threshold=signal_threshold)
    logger.info("inferCNA predictions:\n" + preds.value_counts().to_string())
    return preds


# ===========================================================================
# FIX 8 — Normal-cell fallback using reference epithelial cells
# ===========================================================================

def _build_reference_normal_adata(reference_h5ad, query_var_names, n_normal_cells=500):
    """
    Extract epithelial cells from the Tabula Sapiens reference h5ad and
    return them as a log-normalised AnnData aligned to query_var_names.

    Used when scMalignantFinder leaves fewer than MIN_CELLS_FOR_DEG normal
    cells in the query — which happens with high-purity tumor samples.

    The returned AnnData has:
      obs['final_malignant'] = 'normal'
      obs['source']          = 'reference_epithelial'

    It is concatenated with the query adata so DEG has two meaningful groups.

    Parameters
    ----------
    reference_h5ad   : str    Path to the Tabula Sapiens tissue h5ad.
    query_var_names  : Index  Gene names present in the query adata.
    n_normal_cells   : int    Max reference cells to include (default 500).

    Returns
    -------
    AnnData or None
    """
    EPITHELIAL = {
        "epithelial cell",
        "glandular epithelial cell",
        "ovarian surface epithelial cell",
    }
    try:
        logger.info(f"FIX 8: Loading reference for normal-cell supplement: {reference_h5ad}")
        ref = sc.read_h5ad(reference_h5ad)

        if "cell_ontology_class" in ref.obs.columns:
            ep_mask = ref.obs["cell_ontology_class"].str.lower().isin(EPITHELIAL)
            ref_ep  = ref[ep_mask].copy() if ep_mask.any() else ref.copy()
        else:
            ref_ep = ref.copy()

        if ref_ep.n_obs == 0:
            logger.warning("FIX 8: No epithelial cells in reference — skipping supplement.")
            return None

        # Subsample
        if ref_ep.n_obs > n_normal_cells:
            rng = np.random.default_rng(seed=0)
            idx = rng.choice(ref_ep.n_obs, size=n_normal_cells, replace=False)
            ref_ep = ref_ep[np.sort(idx)].copy()

        logger.info(f"FIX 8: Using {ref_ep.n_obs} reference epithelial cells as normal group.")

        # Get raw counts from ref
        X = _get_raw_matrix(ref_ep)

        # Align to query var names
        ref_gene_names = (
            ref_ep.raw.var_names
            if ref_ep.raw is not None and ref_ep.raw.n_vars > ref_ep.n_vars
            else ref_ep.var_names
        )
        ref_gene_names = pd.Index(ref_gene_names)
        shared         = query_var_names.intersection(ref_gene_names)

        if len(shared) == 0:
            logger.warning("FIX 8: No shared genes between reference and query — skipping.")
            return None

        q_idx = query_var_names.get_indexer(shared)
        r_idx = ref_gene_names.get_indexer(shared)

        X_shared = X[:, r_idx].astype(np.float32)

        # Build a full-query-gene-space matrix (zeros for missing genes)
        X_full      = np.zeros((ref_ep.n_obs, len(query_var_names)), dtype=np.float32)
        X_full[:, q_idx] = X_shared

        ref_adata = sc.AnnData(
            X   = sp.csr_matrix(X_full),
            obs = pd.DataFrame(index=[f"REF_NORMAL_{i}" for i in range(ref_ep.n_obs)]),
            var = pd.DataFrame(index=query_var_names),
        )
        sc.pp.normalize_total(ref_adata, target_sum=1e4)
        sc.pp.log1p(ref_adata)

        ref_adata.obs["final_malignant"] = "normal"
        ref_adata.obs["source"]          = "reference_epithelial"
        ref_adata.obs["gsm_id"]          = "reference"
        ref_adata.obs["gse_id"]          = "reference"

        logger.info(f"FIX 8: Reference normal AnnData built: {ref_adata.shape}")
        return ref_adata

    except Exception as exc:
        logger.warning(f"FIX 8: Could not build reference normal cells: {exc}")
        return None


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_preprocessing_pipeline(
    adata=None,
    popv_path=None,
    min_genes=200,
    max_mt=40.0,
    log2fc_threshold=1.0,
    pval_adj_threshold=0.05,
    reference_h5ad=None,
    tumor_h5ad=None,
    save_dir=None,
    scmalignant_model_dir=None,
    surfaceome_path=None,
    malignant_strategy="union",
    infercna_genome="hg19",
    infercna_n=5000,
    infercna_noise=0.1,
    infercna_signal_threshold=0.9,
    infercna_ref_max_cells=2000,
    min_normal_cells=10,
    ref_normal_supplement_cells=500,
):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    adata : AnnData or None
        If None, auto-loads from popv_results/final_popv_annotated.h5ad.
    popv_path : str or None
        Explicit path to the PopV output h5ad (overrides auto-detection).
    min_genes : int        Minimum genes per cell (QC). Default 200.
    max_mt : float         Maximum mitochondrial %. Default 40.
    log2fc_threshold : float   Log2FC cutoff for DEG. Default 1.0.
    pval_adj_threshold : float BH-adjusted p-value cutoff. Default 0.05.
    reference_h5ad : str or None
        Tabula Sapiens reference for inferCNA AND for FIX 8 normal-cell
        supplement. inferCNA + supplement both skipped if None.
    tumor_h5ad : str or None
        Module 1 output h5ad for Route A-rescue full-gene space.
        OPTIONAL — auto-detected from cwd and GSE_data/ if not given.
    save_dir : str or None    Output dir. Default 'preprocessing_results/'.
    scmalignant_model_dir : str or None   Auto-detected from SCART.
    surfaceome_path : str or None         Auto-detected from SCART.
    malignant_strategy : str
        'union' | 'intersection' | 'scMalignant' | 'infercna'
    infercna_genome : str   'hg19' (default) or 'hg38'. String key, not file.
    infercna_n : int        Top genes for CNA. Default 5000. Auto-capped (FIX 4).
    infercna_noise : float  Noise floor. Default 0.1.
    infercna_signal_threshold : float  Default 0.9.
    infercna_ref_max_cells : int  Reference cells passed to inferCNA. Default 2000.
    min_normal_cells : int
        If fewer than this many normal cells remain after malignancy calling,
        FIX 8 supplements with reference epithelial cells. Default 10.
    ref_normal_supplement_cells : int
        Number of reference epithelial cells to add when FIX 8 triggers.
        Default 500.
    """
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
    logger.info(f"Surfaceome path: {surfaceome_path}")

    # Auto-load PopV output
    if adata is None:
        candidates = []
        if popv_path:
            candidates.append(popv_path)
        auto = _auto_popv_h5ad()
        if auto:
            candidates.append(auto)
        for path in candidates:
            if os.path.exists(path):
                print(f"Loading PopV output: {path}")
                adata = sc.read_h5ad(path)
                break
        if adata is None:
            raise FileNotFoundError(
                "Could not auto-detect PopV output.\n"
                "Expected: popv_results/final_popv_annotated.h5ad\n"
                "Pass adata= or popv_path= explicitly."
            )

    # Report gene-space status
    if "full_counts" in adata.layers and adata.uns.get("full_counts_var_names"):
        n_full = len(adata.uns["full_counts_var_names"])
        print(f"Route A-new: layers['full_counts'] ({n_full} genes from Module 2 FIX 8)")
    else:
        rescue = tumor_h5ad or _auto_tumor_h5ad()
        if rescue and os.path.exists(rescue):
            print(f"Route A-rescue: {rescue} (auto-detected Module 1 h5ad)")
        elif adata.raw is not None:
            print(f"Route A-old: adata.raw ({adata.raw.n_vars} genes)")
        else:
            print(
                "WARNING: No full-gene source found.\n"
                "  scMalignantFinder will use 4000 HVGs (~19% overlap).\n"
                "  Place GSE*_tumor.h5ad in cwd, or update Module 2 to FIX 8."
            )

    print(f"Initial cells: {adata.n_obs}")

    # ------------------------------------------------------------------
    # 1. Select epithelial cells
    # ------------------------------------------------------------------
    labels  = adata.obs["popv_majority_vote_prediction"].astype(str)
    ep_mask = labels.str.endswith("epithelial cell")
    adata   = adata[ep_mask].copy()
    print(f"Epithelial cells retained: {adata.n_obs}")
    print(f"Cells removed:             {(~ep_mask).sum()}\n")

    # ------------------------------------------------------------------
    # 2. QC
    # ------------------------------------------------------------------
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    print(f"Mean MT% BEFORE QC: {adata.obs['pct_counts_mt'].mean():.2f}")
    before_qc = adata.n_obs
    adata = adata[
        (adata.obs["n_genes_by_counts"] > min_genes) &
        (adata.obs["pct_counts_mt"] < max_mt)
    ].copy()
    print(f"Cells after QC:     {adata.n_obs}")
    print(f"Cells removed:      {before_qc - adata.n_obs}")
    print(f"Mean MT% AFTER QC:  {adata.obs['pct_counts_mt'].mean():.2f}\n")

    # ------------------------------------------------------------------
    # 3. Route raw counts into .X; snapshot for inferCNA
    # ------------------------------------------------------------------
    print("Detecting raw count source...")
    for lyr in ("scvi_counts", "raw_counts", "counts"):
        if lyr in adata.layers:
            print(f"Using adata.layers['{lyr}'] as raw counts.")
            adata.X = adata.layers[lyr].copy()
            break
    else:
        if adata.raw is not None:
            print("Using adata.raw.X as raw counts.")
            adata.X = adata.raw.X.copy()
        else:
            print("No raw layer — assuming adata.X is raw counts.")

    adata.var_names_make_unique()
    adata.layers["raw_for_cna"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ------------------------------------------------------------------
    # 4. scMalignantFinder
    # ------------------------------------------------------------------
    print("Running scMalignantFinder ...")
    feature_tsv = os.path.join(scmalignant_model_dir, "ordered_feature.tsv")
    print("  Building full-gene matrix ...")
    adata_scm = _build_fullgene_adata_for_scm(adata, feature_tsv, tumor_h5ad)
    print(f"  Gene space: {adata_scm.n_vars} genes")

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
    scm_pred_col = "scMalignantFinder_prediction"
    if result_scm.obs_names.equals(adata.obs_names):
        adata.obs[scm_pred_col] = result_scm.obs[scm_pred_col].values
    else:
        logger.warning("scMalignantFinder obs_names differ — aligning by index.")
        adata.obs[scm_pred_col] = (
            result_scm.obs[scm_pred_col]
            .reindex(adata.obs_names)
            .fillna("Unknown")
            .values
        )

    print("scMalignantFinder completed.")
    print(adata.obs[scm_pred_col].value_counts().to_string(), "\n")

    # ------------------------------------------------------------------
    # 5. inferCNA  (FIX 7: scalop-safe via Python GMM)
    # ------------------------------------------------------------------
    infercna_available = False
    if malignant_strategy in ("infercna", "union", "intersection"):
        if reference_h5ad is None:
            print(
                "Warning: inferCNA skipped — no reference_h5ad provided.\n"
                "  Falling back to scMalignantFinder only."
            )
            malignant_strategy = "scMalignant"
        else:
            print(
                f"Running inferCNA "
                f"(reference subsampled to <={infercna_ref_max_cells} cells, "
                f"n auto-capped, scalop-safe GMM) ..."
            )
            try:
                adata_raw_cna   = adata.copy()
                adata_raw_cna.X = adata.layers["raw_for_cna"]
                adata_ref_full  = sc.read_h5ad(reference_h5ad)

                infercna_preds = _run_infercna(
                    adata_query      = adata_raw_cna,
                    adata_ref        = adata_ref_full,
                    genome           = infercna_genome,
                    n                = infercna_n,
                    noise            = infercna_noise,
                    signal_threshold = infercna_signal_threshold,
                    ref_max_cells    = infercna_ref_max_cells,
                )
                adata.obs["infercna_prediction"] = (
                    infercna_preds.reindex(adata.obs_names, fill_value="not.defined").values
                )
                infercna_available = True
                print("inferCNA completed.")
                print(adata.obs["infercna_prediction"].value_counts().to_string(), "\n")

            except Exception as exc:
                print(
                    f"Warning: inferCNA failed — {type(exc).__name__}: {exc}\n"
                    "  Falling back to scMalignantFinder only."
                )
                logger.exception("inferCNA error details:")
                malignant_strategy = "scMalignant"

    # ------------------------------------------------------------------
    # 6. Combine malignancy calls → final_malignant
    # ------------------------------------------------------------------
    scm_mal = adata.obs[scm_pred_col].str.lower() == "malignant"
    if infercna_available:
        cna_mal = adata.obs["infercna_prediction"].str.lower() == "malignant"
        if malignant_strategy == "union":
            malignant_mask = scm_mal | cna_mal
            strategy_label = "union  (scMalignantFinder OR inferCNA)"
        elif malignant_strategy == "intersection":
            malignant_mask = scm_mal & cna_mal
            strategy_label = "intersection  (scMalignantFinder AND inferCNA)"
        elif malignant_strategy == "infercna":
            malignant_mask = cna_mal
            strategy_label = "inferCNA only"
        else:
            malignant_mask = scm_mal
            strategy_label = "scMalignantFinder only"
    else:
        malignant_mask = scm_mal
        strategy_label = "scMalignantFinder only"

    adata.obs["final_malignant"] = malignant_mask.map({True: "malignant", False: "normal"})
    adata.obs["source"]          = "query"
    print(f"Malignancy strategy: {strategy_label}")
    print(f"  Malignant: {malignant_mask.sum()} | Normal: {(~malignant_mask).sum()}\n")

    # ------------------------------------------------------------------
    # FIX 8 — supplement with reference normal cells if query normals < threshold
    # ------------------------------------------------------------------
    n_normal_query = (adata.obs["final_malignant"] == "normal").sum()
    ref_normal_adata = None

    if n_normal_query < min_normal_cells:
        print(
            f"FIX 8: Only {n_normal_query} normal cells in query "
            f"(threshold = {min_normal_cells})."
        )
        if reference_h5ad is not None:
            print(
                f"  Supplementing with up to {ref_normal_supplement_cells} "
                "reference epithelial cells as normal comparison group ..."
            )
            ref_normal_adata = _build_reference_normal_adata(
                reference_h5ad,
                adata.var_names,
                n_normal_cells=ref_normal_supplement_cells,
            )
            if ref_normal_adata is not None:
                print(
                    f"  Added {ref_normal_adata.n_obs} reference normal cells.\n"
                    "  DEG will compare malignant query cells vs reference epithelial cells."
                )
            else:
                print("  Could not build reference normal AnnData.")
        else:
            print(
                "  reference_h5ad not provided — cannot supplement.\n"
                "  DEG will be unreliable with so few normal cells.\n"
                "  Pass reference_h5ad= to enable normal-cell supplement."
            )

    # ------------------------------------------------------------------
    # 7. Surfaceome filter
    # ------------------------------------------------------------------
    surfaceome = pd.read_csv(surfaceome_path)
    surfaceome.columns = surfaceome.columns.str.strip()
    surf_genes = surfaceome["Gene"].astype(str).tolist()
    adata      = adata[:, adata.var_names.intersection(surf_genes)].copy()
    print(f"Surfaceome genes retained: {adata.n_vars}\n")

    # ------------------------------------------------------------------
    # 8. Concatenate reference normal cells if FIX 8 triggered
    # ------------------------------------------------------------------
    if ref_normal_adata is not None:
        # Align reference adata to the now-surfaceome-filtered gene space
        shared_surf = adata.var_names.intersection(ref_normal_adata.var_names)
        ref_normal_surf = ref_normal_adata[:, shared_surf].copy()

        # Expand to full surfaceome var space (zeros for missing genes)
        X_ref_full = np.zeros(
            (ref_normal_surf.n_obs, adata.n_vars), dtype=np.float32
        )
        ref_idx = adata.var_names.get_indexer(shared_surf)
        if sp.issparse(ref_normal_surf.X):
            X_ref_full[:, ref_idx] = ref_normal_surf.X.toarray()
        else:
            X_ref_full[:, ref_idx] = np.array(ref_normal_surf.X)

        ref_surf_adata = sc.AnnData(
            X   = sp.csr_matrix(X_ref_full),
            obs = ref_normal_surf.obs[["final_malignant", "source"]].copy(),
            var = adata.var.copy(),
        )

        # Ensure query adata has the same obs columns for concatenation
        if "source" not in adata.obs.columns:
            adata.obs["source"] = "query"

        adata_for_deg = sc.concat(
            [adata, ref_surf_adata],
            join="outer",
            label=None,
        )
        adata_for_deg.obs_names_make_unique()
        adata_for_deg.var = adata.var.copy()
        print(
            f"Concatenated AnnData for DEG: "
            f"{adata_for_deg.n_obs} cells x {adata_for_deg.n_vars} genes\n"
            f"  ({adata.n_obs} query + {ref_surf_adata.n_obs} reference normal)"
        )
    else:
        adata_for_deg = adata

    # ------------------------------------------------------------------
    # 9. DEG  (FIX 3: pvals_adj not raw pvals)
    # ------------------------------------------------------------------
    group_counts = adata_for_deg.obs["final_malignant"].value_counts()
    min_group    = int(group_counts.min())
    print(f"Malignant/normal cell counts for DEG: {dict(group_counts)}")

    if min_group < min_normal_cells:
        print(
            f"WARNING: Smallest group has only {min_group} cells "
            f"(need >= {min_normal_cells} for reliable Wilcoxon DEG).\n"
            "  All surfaceome genes saved as candidates.\n"
            "  Provide reference_h5ad= to enable normal-cell supplement."
        )
        all_surf     = adata_for_deg.var_names.tolist()
        filtered_deg = pd.DataFrame({"names": all_surf, "group": "malignant"})
        deg          = filtered_deg.copy()
    else:
        sc.tl.rank_genes_groups(
            adata_for_deg,
            groupby="final_malignant",
            method="wilcoxon",
            key_added="rank_genes_groups",
        )
        deg = sc.get.rank_genes_groups_df(adata_for_deg, group=None)
        print(f"Total DEG candidates: {deg.shape[0]}")
        print(
            f"Applying filters: log2FC > {log2fc_threshold}, "
            f"pvals_adj < {pval_adj_threshold}"
        )
        filtered_deg = deg[
            (deg["logfoldchanges"] > log2fc_threshold) &
            (deg["pvals_adj"]      < pval_adj_threshold)    # FIX 3
        ]
        if filtered_deg.shape[0] == 0:
            print(
                "WARNING: 0 DEGs passed the filter.\n"
                "  Try: log2fc_threshold=0.5 or pval_adj_threshold=0.10"
            )

    # Store results on query-only adata (not the padded adata_for_deg)
    adata.uns["filtered_deg"] = filtered_deg
    adata.uns["all_deg"]      = deg
    adata.uns["deg_params"]   = {
        "log2fc_threshold"   : log2fc_threshold,
        "pval_adj_threshold" : pval_adj_threshold,
        "method"             : "wilcoxon",
        "normal_source"      : "reference_epithelial" if ref_normal_adata else "query",
    }
    print(f"Final DE genes retained: {filtered_deg.shape[0]}\n")

    # ------------------------------------------------------------------
    # 10. Binarise  (query adata only)
    # ------------------------------------------------------------------
    adata.X = (adata.X > 0).astype(int)
    print("Expression converted to binary (0/1).\n")

    # ------------------------------------------------------------------
    # 11. Save
    # ------------------------------------------------------------------
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object:
            adata.obs[col] = adata.obs[col].astype(str)

    final_path = os.path.join(save_dir, "final_tumor.h5ad")
    adata.write(final_path)
    print(f"Final object saved to: {final_path}")
    print("\n========== PREPROCESSING COMPLETED ==========\n")
    return adata
