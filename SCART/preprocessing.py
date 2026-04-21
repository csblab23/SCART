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
       tumor_h5ad is now OPTIONAL — auto-detected from cwd/GSE_data/ without
       any user input.  Just leave the Module 1 h5ad in the working directory.

FIX 2  inferCNA reference subsampled to infercna_ref_max_cells=2000.

FIX 3  DEG uses pvals_adj (BH-adjusted) not raw pvals.

FIX 4  inferCNA: auto-cap n to (n_common_genes - 1) — prevents the R error
       "<ngenes> cannot be larger than nrow(m)".

FIX 5  inferCNA reference: prefer adata.raw over HVG layers.

FIX 6  scMalignantFinder output aligned by obs_names not positional .values.
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

    Search order mirrors geo_fetcher.py output conventions:
      1. *_tumor.h5ad in cwd          (e.g. GSE158937_tumor.h5ad)
      2. combined_tumor.h5ad in cwd
      3. input_tumor.h5ad in cwd
      4. Same patterns inside GSE_data/

    Returns the most recently created match, or None if nothing found.
    Used as Route A-rescue when Module 2 layers['full_counts'] is absent.
    """
    patterns = ["*_tumor.h5ad", "combined_tumor.h5ad", "input_tumor.h5ad"]
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
    """
    Auto-detect the Module 2 PopV output h5ad.

    Search order:
      1. popv_results/final_popv_annotated.h5ad  (default Module 2 output dir)
      2. final_popv_annotated.h5ad in cwd
    """
    candidates = [
        os.path.join("popv_results", "final_popv_annotated.h5ad"),
        "final_popv_annotated.h5ad",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# ===========================================================================
# FIX 1 - Build full-gene AnnData for scMalignantFinder
# ===========================================================================

def _build_fullgene_adata_for_scm(adata, feature_tsv, tumor_h5ad_path=None):
    """
    Return a log-normalised AnnData covering the full original gene space.

    Route priority
    --------------
    A-new     layers['full_counts'] in current adata (from Module 2 FIX 8).
              Primary path when Module 2 is up to date.

    A-rescue  Module 1 tumor h5ad. Auto-detected from cwd/GSE_data/ with no
              user input required. Activated when A-new is unavailable.

    A-old     adata.raw (earlier Module 2 approach).

    B         uns['full_var_names'] (SCART-internal fallback).

    C         4000-HVG fallback with loud warning.
    """
    model_features = set(
        pd.read_csv(feature_tsv, sep="\t", header=None)[0].tolist()
    )
    n_model = len(model_features)

    def _pct(names):
        return len(set(names) & model_features) / n_model * 100

    def _make_adata(X, obs, var_names_or_df):
        """
        Wrap raw counts in a log-normalised AnnData.
        Returns CSR sparse so scMalignantFinder can call .todense().
        """
        if sp.issparse(X):
            X = X.toarray()
        X = X.astype(np.float32)
        var = var_names_or_df if isinstance(var_names_or_df, pd.DataFrame) \
              else pd.DataFrame(index=list(var_names_or_df))
        af = sc.AnnData(X=X, obs=obs, var=var)
        sc.pp.normalize_total(af, target_sum=1e4)
        sc.pp.log1p(af)
        af.X = sp.csr_matrix(af.X)
        return af

    # ----------------------------------------------------------------
    # Route A-new: layers['full_counts'] written by Module 2 FIX 8
    # ----------------------------------------------------------------
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
                af = _make_adata(
                    adata.layers["full_counts"],
                    adata.obs.copy(),
                    var_names
                )
                logger.info(
                    f"scMalignantFinder using Route A-new "
                    f"({af.n_vars} genes, {ov:.1f}% overlap)."
                )
                return af
            logger.warning(
                f"Route A-new overlap {ov:.1f}% < 50% — trying Route A-rescue."
            )
        else:
            logger.warning(
                "layers['full_counts'] present but gene names mismatch or missing. "
                "Trying Route A-rescue."
            )

    # ----------------------------------------------------------------
    # Route A-rescue: auto-detect Module 1 tumor h5ad (no user input needed)
    # ----------------------------------------------------------------
    rescue_path = tumor_h5ad_path or _auto_tumor_h5ad()

    if rescue_path is not None and os.path.exists(rescue_path):
        logger.info(f"Route A-rescue: loading from {rescue_path}")
        try:
            adata_m1 = sc.read_h5ad(rescue_path)

            # Prefer adata.raw (geo_fetcher sets adata.raw = adata)
            if adata_m1.raw is not None:
                raw_var = adata_m1.raw.var_names
                ov = _pct(raw_var)
                logger.info(
                    f"Route A-rescue (adata_m1.raw): "
                    f"{adata_m1.raw.n_vars} genes, {ov:.1f}% overlap"
                )
                if ov >= 50:
                    shared = adata.obs_names.intersection(adata_m1.obs_names)
                    if len(shared) > 0:
                        current_order = [c for c in adata.obs_names if c in set(shared)]
                        m1_sub = adata_m1[current_order]
                        af = _make_adata(
                            m1_sub.raw.X,
                            adata.obs.loc[current_order].copy(),
                            m1_sub.raw.var.copy()
                        )
                        logger.info(
                            f"scMalignantFinder using Route A-rescue "
                            f"({af.n_vars} genes, {ov:.1f}% overlap)."
                        )
                        return af
                    logger.warning(
                        "Route A-rescue: no shared obs_names — "
                        "trying layers inside Module 1 h5ad."
                    )

            # Fallback within rescue: layers['counts']
            for lyr in ("counts", "raw_counts", "scvi_counts"):
                if lyr in adata_m1.layers:
                    raw_var = adata_m1.var_names
                    ov = _pct(raw_var)
                    logger.info(
                        f"Route A-rescue (layers['{lyr}']): "
                        f"{len(raw_var)} genes, {ov:.1f}% overlap"
                    )
                    if ov >= 50:
                        shared = adata.obs_names.intersection(adata_m1.obs_names)
                        if len(shared) > 0:
                            current_order = [c for c in adata.obs_names if c in set(shared)]
                            m1_sub = adata_m1[current_order]
                            af = _make_adata(
                                m1_sub.layers[lyr],
                                adata.obs.loc[current_order].copy(),
                                m1_sub.var.copy()
                            )
                            logger.info(
                                f"scMalignantFinder using Route A-rescue "
                                f"(layers['{lyr}'], {af.n_vars} genes, {ov:.1f}% overlap)."
                            )
                            return af
                    break

        except Exception as exc:
            logger.warning(f"Route A-rescue failed: {exc}. Trying Route A-old.")

    elif rescue_path is not None:
        logger.warning(f"Route A-rescue path not found: {rescue_path!r}")
    else:
        logger.info(
            "Route A-rescue: no Module 1 h5ad found in cwd or GSE_data/. "
            "Trying Route A-old."
        )

    # ----------------------------------------------------------------
    # Route A-old: adata.raw
    # ----------------------------------------------------------------
    if adata.raw is not None:
        ov = _pct(adata.raw.var_names)
        logger.info(
            f"Route A-old (adata.raw): {adata.raw.n_vars} genes, {ov:.1f}% overlap"
        )
        if ov >= 50:
            af = _make_adata(adata.raw.X, adata.obs.copy(), adata.raw.var.copy())
            logger.info("scMalignantFinder using Route A-old.")
            return af
        logger.warning(f"Route A-old overlap {ov:.1f}% < 50% — trying Route B.")

    # ----------------------------------------------------------------
    # Route B: uns['full_var_names']
    # ----------------------------------------------------------------
    if "full_var_names" in adata.uns:
        full_var = list(adata.uns["full_var_names"])
        ov = _pct(full_var)
        logger.info(f"Route B (uns): {len(full_var)} genes, {ov:.1f}% overlap")
        for lyr in ("scvi_counts", "raw_counts", "counts"):
            if lyr in adata.layers:
                X = adata.layers[lyr]
                if sp.issparse(X):
                    X = X.toarray()
                if X.shape[1] == len(full_var) and ov >= 50:
                    af = _make_adata(X, adata.obs.copy(), full_var)
                    logger.info(f"scMalignantFinder using Route B (layers['{lyr}']).")
                    return af

    # ----------------------------------------------------------------
    # Route C — 4000 HVG fallback
    # ----------------------------------------------------------------
    ov_hvg = _pct(adata.var_names)
    auto_rescue = _auto_tumor_h5ad()
    logger.warning(
        f"All routes failed. Falling back to {adata.n_vars} HVGs "
        f"({ov_hvg:.1f}% overlap).\n"
        "Diagnosis:\n"
        "  1. Is Module 2 up to date with FIX 8?\n"
        "     Check adata.uns.get('full_counts_var_names') is not None\n"
        "  2. Is the Module 1 h5ad in cwd or GSE_data/?\n"
        f"     Auto-search found: {auto_rescue}"
    )
    return adata.copy()


# ===========================================================================
# FIX 5 - Helper: raw count matrix (adata.raw first, then layers)
# ===========================================================================

def _get_raw_matrix(adata):
    """
    Return a dense float64 (cells x genes) array of raw integer counts.

    FIX 5: Check adata.raw BEFORE layers.
    The Tabula Sapiens reference h5ad has adata.raw with ~30k genes but
    also HVG layers with only 4000 genes. Checking adata.raw first gives
    ~15-25k common genes with the query for proper CNA inference.
    """
    if adata.raw is not None:
        raw_n = adata.raw.n_vars
        layer_n = max(
            (adata.layers[l].shape[1]
             for l in ("full_counts", "scvi_counts", "raw_counts", "counts")
             if l in adata.layers),
            default=0
        )
        if raw_n > layer_n:
            logger.info(
                f"Raw counts from adata.raw ({raw_n} genes > "
                f"best layer {layer_n} genes)"
            )
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
        logger.warning(
            "No raw layer or adata.raw — using adata.X which may be "
            "log-normalised. inferCNA results may be unreliable."
        )
        X = adata.X

    if sp.issparse(X):
        X = X.toarray()
    return np.array(X, dtype=np.float64)


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
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        raise ImportError(
            "rpy2 is required to run inferCNA.\n"
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

    # FIX 2 - subsample reference epithelial cells
    EPITHELIAL = {
        "epithelial cell",
        "glandular epithelial cell",
        "ovarian surface epithelial cell",
    }
    if "cell_ontology_class" in adata_ref.obs.columns:
        ep_mask = adata_ref.obs["cell_ontology_class"].str.lower().isin(EPITHELIAL)
        adata_ref_ep = adata_ref[ep_mask].copy() if ep_mask.any() else adata_ref.copy()
        logger.info(f"inferCNA reference epithelial cells: {adata_ref_ep.n_obs}")
    else:
        adata_ref_ep = adata_ref.copy()

    if adata_ref_ep.n_obs > ref_max_cells:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(adata_ref_ep.n_obs, size=ref_max_cells, replace=False)
        adata_ref_ep = adata_ref_ep[np.sort(idx)].copy()
        logger.info(f"Reference subsampled to {ref_max_cells} cells.")

    # Build log-CPM matrices (FIX 5: _get_raw_matrix checks .raw first)
    def _to_log_cpm(adata_obj):
        X  = _get_raw_matrix(adata_obj)
        rs = X.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        return np.log1p(X / rs * 1e6).T

    logger.info("inferCNA: building query log-CPM matrix ...")
    mat_query = _to_log_cpm(adata_query)
    logger.info("inferCNA: building reference log-CPM matrix ...")
    mat_ref   = _to_log_cpm(adata_ref_ep)

    q_genes = np.array(adata_query.var_names)
    if adata_ref_ep.raw is not None and \
            adata_ref_ep.raw.n_vars > max(
                (adata_ref_ep.layers[l].shape[1]
                 for l in ("scvi_counts", "raw_counts", "counts")
                 if l in adata_ref_ep.layers),
                default=0):
        r_genes = np.array(adata_ref_ep.raw.var_names)
    else:
        r_genes = np.array(adata_ref_ep.var_names)

    common   = np.intersect1d(q_genes, r_genes)
    n_common = len(common)
    logger.info(f"inferCNA common genes: {n_common}")

    if n_common < 200:
        raise ValueError(
            f"Only {n_common} common genes. Need >= 200.\n"
            "Check both datasets use HGNC gene symbols."
        )
    if n_common < 2000:
        logger.warning(
            f"Only {n_common} common genes — inferCNA works best with 5000+."
        )

    # FIX 4 - auto-cap n
    n_safe = min(n, n_common - 1)
    if n_safe < n:
        logger.warning(
            f"FIX 4: infercna_n={n} capped to {n_safe} "
            f"(must be < n_common_genes={n_common})."
        )

    q_idx = np.where(np.isin(q_genes, common))[0]
    r_idx = np.where(np.isin(r_genes, common))[0]

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
    ro.globalenv["sig_thresh"]   = ro.FloatVector([signal_threshold])
    ro.globalenv["genome_name"]  = ro.StrVector([genome])

    logger.info("inferCNA: running R steps ...")
    ro.r("""
        library(infercna)
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
        cnaM <- cna[, !colnames(cna) %in% ref_barcodes, drop = FALSE]
        n_query    <- ncol(cna) - length(ref_barcodes)
        n_ref      <- length(ref_barcodes)
        sample_vec <- c(rep("tumor", n_query), rep("normal", n_ref))
        modes <- tryCatch(
            findMalignant(
                cna              = cna,
                signal.threshold = sig_thresh,
                samples          = sample_vec,
                excludeFromAvg   = ref_barcodes
            ),
            error = function(e) { message("findMalignant error: ", conditionMessage(e)); NULL }
        )
    """)

    modes = ro.globalenv["modes"]
    is_null_or_false = (
        modes is ro.rinterface.NULL
        or (hasattr(modes, "typeof") and str(modes.typeof) == "logical")
        or not hasattr(modes, "names")
        or modes.names is None
    )

    if is_null_or_false:
        logger.warning(
            "inferCNA findMalignant() returned NULL/FALSE.\n"
            "All query cells labelled 'not.defined'.\n"
            "Try: infercna_signal_threshold=0.75"
        )
        return pd.Series("not.defined", index=q_barcodes, name="infercna_prediction")

    label_map = {}
    for key in list(modes.names):
        label = "malignant" if "malignant" in key.lower() else "non-malignant"
        for bc in list(modes.rx2(key)):
            label_map[bc] = label

    preds = pd.Series(
        [label_map.get(bc, "not.defined") for bc in q_barcodes],
        index=q_barcodes,
        name="infercna_prediction",
    ).reindex(q_barcodes, fill_value="not.defined")

    logger.info("inferCNA predictions:\n" + preds.value_counts().to_string())
    return preds


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
        Tabula Sapiens reference for inferCNA. inferCNA skipped if None.
    tumor_h5ad : str or None
        Module 1 output h5ad for Route A-rescue full-gene space.
        OPTIONAL — auto-detected from cwd and GSE_data/ if not given.
        Leave the GSE*_tumor.h5ad in the working directory and this
        parameter does not need to be set at all.
    save_dir : str or None    Output dir. Default 'preprocessing_results/'.
    scmalignant_model_dir : str or None   Auto-detected from SCART.
    surfaceome_path : str or None         Auto-detected from SCART.
    malignant_strategy : str
        'union' | 'intersection' | 'scMalignant' | 'infercna'
    infercna_genome : str   'hg19' (default) or 'hg38'. String key, not file.
    infercna_n : int        Top genes for CNA. Default 5000. Auto-capped.
    infercna_noise : float  Noise floor. Default 0.1.
    infercna_signal_threshold : float  Default 0.9. Lower to 0.75 if needed.
    infercna_ref_max_cells : int  Reference cells passed to R. Default 2000.
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
                "Pass adata= or popv_path= explicitly if file is elsewhere."
            )

    # Report gene-space status
    if "full_counts" in adata.layers and \
            adata.uns.get("full_counts_var_names") is not None:
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
                "  Ensure GSE*_tumor.h5ad is in cwd, or update Module 2 to FIX 8."
            )

    print(f"Initial cells: {adata.n_obs}")

    # 1. Select epithelial cells
    labels  = adata.obs["popv_majority_vote_prediction"].astype(str)
    ep_mask = labels.str.endswith("epithelial cell")
    adata   = adata[ep_mask].copy()
    print(f"Epithelial cells retained: {adata.n_obs}")
    print(f"Cells removed:             {(~ep_mask).sum()}\n")

    # 2. QC
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

    # 3. Route raw counts into .X; snapshot for inferCNA
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

    # 4. scMalignantFinder
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

    # FIX 6 - align by obs_names
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

    # 5. inferCNA
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
                f"(reference subsampled to <=|{infercna_ref_max_cells} cells, "
                f"n auto-capped) ..."
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
                adata.obs["infercna_prediction"] = infercna_preds.values
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

    # 6. Combine malignancy calls
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
    print(f"Malignancy strategy: {strategy_label}")
    print(f"  Malignant: {malignant_mask.sum()} | Normal: {(~malignant_mask).sum()}\n")

    # 7. Surfaceome filter
    surfaceome = pd.read_csv(surfaceome_path)
    surfaceome.columns = surfaceome.columns.str.strip()
    surf_genes = surfaceome["Gene"].astype(str).tolist()
    adata      = adata[:, adata.var_names.intersection(surf_genes)].copy()
    print(f"Surfaceome genes retained: {adata.n_vars}\n")

    # 8. DEG
    group_counts = adata.obs["final_malignant"].value_counts()
    min_group    = group_counts.min()
    print(f"Malignant/normal cell counts for DEG: {dict(group_counts)}")
    MIN_CELLS_FOR_DEG = 10

    if min_group < MIN_CELLS_FOR_DEG:
        print(
            f"WARNING: Smallest group has only {min_group} cells "
            f"(need >= {MIN_CELLS_FOR_DEG} for reliable Wilcoxon DEG).\n"
            "  All surfaceome genes saved as candidates.\n"
            "  Enable inferCNA (provide reference_h5ad=) to improve the split."
        )
        all_surf     = adata.var_names.tolist()
        filtered_deg = pd.DataFrame({"names": all_surf, "group": "malignant"})
        deg          = filtered_deg.copy()
    else:
        sc.tl.rank_genes_groups(
            adata, groupby="final_malignant", method="wilcoxon",
            key_added="rank_genes_groups",
        )
        deg = sc.get.rank_genes_groups_df(adata, group=None)
        print(f"Total DEG candidates: {deg.shape[0]}")
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

    adata.uns["filtered_deg"] = filtered_deg
    adata.uns["all_deg"]      = deg
    adata.uns["deg_params"]   = {
        "log2fc_threshold"  : log2fc_threshold,
        "pval_adj_threshold": pval_adj_threshold,
        "method"            : "wilcoxon",
    }
    print(f"Final DE genes retained: {filtered_deg.shape[0]}\n")

    # 9. Binarise
    adata.X = (adata.X > 0).astype(int)
    print("Expression converted to binary (0/1).\n")

    # 10. Save
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object:
            adata.obs[col] = adata.obs[col].astype(str)

    final_path = os.path.join(save_dir, "final_tumor.h5ad")
    adata.write(final_path)
    print(f"Final object saved to: {final_path}")
    print("\n========== PREPROCESSING COMPLETED ==========\n")
    return adata
