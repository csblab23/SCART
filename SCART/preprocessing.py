"""
preprocessing.py
Module 3 — Preprocessing, malignancy detection, and surfaceome DEG

GitHub: https://github.com/navinlabcode/SCART

Fixes in this revision
----------------------
FIX 1 — scMalignantFinder full-gene rescue without re-running Module 2
  The current final_popv_annotated.h5ad was generated before the
  _reattach_raw_slot fix, so adata.raw is None and only 4000 HVGs
  are available (19% model overlap).
  New Route A-rescue: loads the original Module 1 tumor h5ad (e.g.
  GSE158937_tumor.h5ad) which still has adata.raw with all ~33k genes.
  Pass tumor_h5ad= to the pipeline to activate this route.
  Auto-detected by searching for *_tumor.h5ad / combined_tumor.h5ad
  in the current directory if not provided explicitly.

FIX 2 — inferCNA speed: reference subsampling
  Loading the full Tabula Sapiens ovary h5ad (~49k cells) and
  transposing it to a log-CPM genes×cells matrix was the "stuck" step.
  New parameter infercna_ref_max_cells (default 2000) subsamples the
  reference epithelial cells to at most this number before passing to R.
  inferCNA's refCorrect() only needs a stable baseline average —
  2000 normal cells is more than sufficient for that.

FIX 3 — DEG uses pvals_adj (BH-adjusted) not raw pvals (carried over).

All hardcoded paths removed (carried over from previous revision).
"""

import os
import glob
import logging
import importlib.resources as pkg_resources

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===========================================================================
# Auto-detect paths from the installed SCART package
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
    Find the Module 1 tumor h5ad in the current working directory.
    Searched in order: *_tumor.h5ad, combined_tumor.h5ad, input_tumor.h5ad
    Returns the most recently created match, or None if nothing found.
    """
    patterns = ["*_tumor.h5ad", "combined_tumor.h5ad", "input_tumor.h5ad"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(os.getcwd(), pattern)))
        files.extend(glob.glob(os.path.join(os.getcwd(), "GSE_data", pattern)))
    files = list(set(files))
    if not files:
        return None
    return max(files, key=os.path.getctime)


# ===========================================================================
# FIX 1 — Build full-gene AnnData for scMalignantFinder
# ===========================================================================

def _build_fullgene_adata_for_scm(adata, feature_tsv, tumor_h5ad_path=None):
    """
    Return a log-normalised AnnData covering the full original gene space.

    Route A-rescue  Load original Module 1 tumor h5ad → adata.raw (~33k genes)
                    This is the primary fix for the 19% overlap problem when
                    Module 2 was run before the _reattach_raw_slot fix.
                    Activated when tumor_h5ad_path is provided (or auto-detected).

    Route A-new     layers['full_counts'] + uns['full_counts_var_names']
                    Written by Module 2 FIX 8 (future runs).

    Route A-old     adata.raw  (if Module 2 FIX 8 was applied)

    Route B         uns['full_var_names']

    Route C         4000-HVG fallback with warning
    """
    model_features = set(
        pd.read_csv(feature_tsv, sep="\t", header=None)[0].tolist()
    )
    n_model = len(model_features)

    def _pct(names):
        return len(set(names) & model_features) / n_model * 100

    def _make_adata(X, obs, var_names_or_df):
        """
        Normalise a raw count matrix and wrap in AnnData.

        scMalignantFinder._make_predictions() calls .todense() on adata.X,
        which only works on scipy sparse matrices — not numpy arrays.
        After normalize_total + log1p the matrix becomes a dense numpy array,
        so we convert back to CSR before returning.
        """
        if sp.issparse(X):
            X = X.toarray()
        X = X.astype(np.float32)
        var = var_names_or_df if isinstance(var_names_or_df, pd.DataFrame) \
              else pd.DataFrame(index=list(var_names_or_df))
        af = sc.AnnData(X=X, obs=obs, var=var)
        sc.pp.normalize_total(af, target_sum=1e4)
        sc.pp.log1p(af)
        # Convert to CSR sparse so scMalignantFinder can call .todense()
        af.X = sp.csr_matrix(af.X)
        return af

    # ----------------------------------------------------------------
    # Route A-rescue: reload Module 1 tumor h5ad raw slot
    # ----------------------------------------------------------------
    if tumor_h5ad_path is not None and os.path.exists(tumor_h5ad_path):
        logger.info(f"Route A-rescue: loading raw slot from {tumor_h5ad_path}")
        try:
            adata_m1 = sc.read_h5ad(tumor_h5ad_path)
            # Module 1 sets adata.raw = adata before any processing
            raw_src = None
            raw_var = None

            if adata_m1.raw is not None:
                raw_var = adata_m1.raw.var_names
                ov = _pct(raw_var)
                logger.info(
                    f"Route A-rescue (adata_m1.raw): "
                    f"{adata_m1.raw.n_vars} genes, {ov:.1f}% overlap"
                )
                if ov >= 50:
                    # Align cells: Module 1 h5ad may have more cells than
                    # the QC-filtered adata; subset to matching obs_names.
                    shared = adata.obs_names.intersection(adata_m1.obs_names)
                    if len(shared) == 0:
                        logger.warning(
                            "Route A-rescue: no shared obs_names between "
                            "current adata and Module 1 h5ad. "
                            "Trying adata_m1.X directly."
                        )
                        # Fall through to X-based approach below
                    else:
                        m1_sub = adata_m1[shared].copy()
                        # Re-order to match current adata cell order
                        current_order = [
                            c for c in adata.obs_names if c in set(shared)
                        ]
                        m1_sub = m1_sub[current_order]
                        X_raw = m1_sub.raw.X
                        af = _make_adata(X_raw, adata.obs.copy(), m1_sub.raw.var.copy())
                        logger.info(
                            f"scMalignantFinder using Route A-rescue "
                            f"({af.n_vars} genes, {ov:.1f}% overlap)."
                        )
                        return af

            # Fallback inside rescue: use adata_m1.layers['counts'] or .X
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
                            current_order = [
                                c for c in adata.obs_names if c in set(shared)
                            ]
                            m1_sub = adata_m1[current_order]
                            X_raw = m1_sub.layers[lyr]
                            af = _make_adata(
                                X_raw,
                                adata.obs.copy(),
                                m1_sub.var.copy()
                            )
                            logger.info(
                                f"scMalignantFinder using Route A-rescue "
                                f"(layers['{lyr}'], {af.n_vars} genes, {ov:.1f}% overlap)."
                            )
                            return af
                    break

        except Exception as exc:
            logger.warning(f"Route A-rescue failed: {exc}. Trying next route.")
    else:
        if tumor_h5ad_path is not None:
            logger.warning(
                f"Route A-rescue: provided tumor_h5ad={tumor_h5ad_path!r} not found."
            )

    # ----------------------------------------------------------------
    # Route A-new: layers['full_counts']
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
            logger.warning(f"Route A-new overlap {ov:.1f}% < 50% — trying A-old.")

    # ----------------------------------------------------------------
    # Route A-old: adata.raw
    # ----------------------------------------------------------------
    if adata.raw is not None:
        ov = _pct(adata.raw.var_names)
        logger.info(
            f"Route A-old (adata.raw): {adata.raw.n_vars} genes, {ov:.1f}% overlap"
        )
        if ov >= 50:
            af = _make_adata(
                adata.raw.X,
                adata.obs.copy(),
                adata.raw.var.copy()
            )
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
                    logger.info(
                        f"scMalignantFinder using Route B (layers['{lyr}'])."
                    )
                    return af

    # ----------------------------------------------------------------
    # Route C — fallback (19% overlap)
    # ----------------------------------------------------------------
    ov_hvg = _pct(adata.var_names)
    logger.warning(
        f"All routes failed. Falling back to {adata.n_vars} HVGs "
        f"({ov_hvg:.1f}% overlap).\n"
        "To fix without re-running Module 2:\n"
        "  Pass tumor_h5ad='/path/to/GSE158937_tumor.h5ad' to the pipeline.\n"
        "For future runs: update Module 2 (popv_annotation.py FIX 8)."
    )
    return adata.copy()


# ===========================================================================
# Helper: extract raw count matrix
# ===========================================================================

def _get_raw_matrix(adata):
    """Return a dense float64 (cells × genes) array of raw integer counts."""
    for lyr in ("full_counts", "scvi_counts", "raw_counts", "counts"):
        if lyr in adata.layers:
            logger.info(f"Raw counts from adata.layers['{lyr}']")
            X = adata.layers[lyr]
            break
    else:
        if adata.raw is not None:
            logger.info("Raw counts from adata.raw.X")
            X = adata.raw.X
        else:
            logger.info("No raw layer — assuming adata.X is raw counts")
            X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return np.array(X, dtype=np.float64)


# ===========================================================================
# inferCNA  (via rpy2) — official tutorial step order
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
    Run inferCNA following the official tutorial step order:
    https://rdrr.io/github/jlaffy/infercna/f/vignettes/infercna_tutorial.Rmd

    Step 1  useGenome()   — set built-in chromosome coordinate table
    Step 2  infercna()    — CNA inference on combined (query + ref) matrix
    Step 3  strip ref     — remove reference columns from cna result
    Step 4  findMalignant()  — bimodal Gaussian fitting on full cna

    FIX 2 — reference subsampling
    The Tabula Sapiens ovary h5ad has ~49k cells.  Building a genes×cells
    matrix for all of them took several minutes and appeared to "hang".
    refCorrect() only needs a stable average of the normal baseline;
    ref_max_cells=2000 epithelial reference cells is more than sufficient.
    Subsampling is done BEFORE building the R matrix, keeping the
    Python-side work small.

    Parameters
    ----------
    adata_query      : AnnData  Query epithelial cells (raw counts).
    adata_ref        : AnnData  Normal reference (Tabula Sapiens tissue h5ad).
    genome           : str      'hg19' (default, built-in) or 'hg38'.
    n                : int      Most-variable genes to keep (default 5000).
    noise            : float    Exclude genes with range < noise (default 0.1).
    signal_threshold : float    Top fraction for cnaSignal/cnaCor (default 0.9).
    ref_max_cells    : int      Max reference epithelial cells to pass to R
                                (default 2000). Reduces memory and runtime.

    Returns
    -------
    pd.Series  Index = all query barcodes.
               Values = 'malignant' | 'non-malignant' | 'not.defined'
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects import default_converter, numpy2ri
    except ImportError as exc:
        raise ImportError(
            "rpy2 is required to run inferCNA.\n"
            "Install: pip install rpy2\n"
            "R package: devtools::install_github('jlaffy/infercna')"
        ) from exc

    try:
        importr("infercna")
    except Exception as exc:
        raise ImportError(
            "R package 'infercna' not found.\n"
            "Install in R:\n"
            "  install.packages('devtools')\n"
            "  devtools::install_github('jlaffy/infercna')"
        ) from exc

    # ------------------------------------------------------------------
    # FIX 2 — subsample reference epithelial cells
    # ------------------------------------------------------------------
    EPITHELIAL = {
        "epithelial cell",
        "glandular epithelial cell",
        "ovarian surface epithelial cell",
    }
    if "cell_ontology_class" in adata_ref.obs.columns:
        ep_mask = adata_ref.obs["cell_ontology_class"].str.lower().isin(EPITHELIAL)
        adata_ref_ep = adata_ref[ep_mask].copy() if ep_mask.any() else adata_ref.copy()
    else:
        adata_ref_ep = adata_ref.copy()

    # Subsample to ref_max_cells
    if adata_ref_ep.n_obs > ref_max_cells:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(adata_ref_ep.n_obs, size=ref_max_cells, replace=False)
        adata_ref_ep = adata_ref_ep[np.sort(idx)].copy()
        logger.info(
            f"inferCNA reference subsampled to {ref_max_cells} epithelial cells "
            f"(was {ep_mask.sum() if 'cell_ontology_class' in adata_ref.obs.columns else adata_ref.n_obs})."
        )
    else:
        logger.info(f"inferCNA reference: {adata_ref_ep.n_obs} epithelial cells")

    # ------------------------------------------------------------------
    # Build log-CPM matrices (genes × cells)
    # ------------------------------------------------------------------
    def _to_log_cpm(adata_obj):
        X  = _get_raw_matrix(adata_obj)
        rs = X.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        return np.log1p(X / rs * 1e6).T   # genes × cells

    logger.info("inferCNA: building query log-CPM matrix ...")
    mat_query = _to_log_cpm(adata_query)
    logger.info("inferCNA: building reference log-CPM matrix ...")
    mat_ref   = _to_log_cpm(adata_ref_ep)

    # ------------------------------------------------------------------
    # Align to common genes
    # ------------------------------------------------------------------
    q_genes = np.array(adata_query.var_names)
    r_genes = np.array(adata_ref_ep.var_names)
    common  = np.intersect1d(q_genes, r_genes)
    logger.info(f"inferCNA common genes: {len(common)}")

    if len(common) < 200:
        raise ValueError(
            f"Only {len(common)} common genes between query and reference. "
            "Check that both use HGNC gene symbols."
        )

    q_idx = np.where(np.isin(q_genes, common))[0]
    r_idx = np.where(np.isin(r_genes, common))[0]

    mat_combined = np.hstack([mat_query[q_idx, :], mat_ref[r_idx, :]])
    sub_genes    = q_genes[q_idx]

    q_barcodes   = np.array(adata_query.obs_names)
    ref_barcodes = np.array(["REF_" + b for b in adata_ref_ep.obs_names])
    all_barcodes = np.concatenate([q_barcodes, ref_barcodes])

    logger.info(
        f"inferCNA combined matrix: {mat_combined.shape[0]} genes × "
        f"{mat_combined.shape[1]} cells "
        f"({len(q_barcodes)} query + {len(ref_barcodes)} ref)"
    )

    # ------------------------------------------------------------------
    # Pass data to R and run inferCNA steps via a single R block
    # (avoids repeated Python↔R round-trips and rpy2 version issues)
    # ------------------------------------------------------------------
    with localconverter(default_converter + numpy2ri.converter):
        r_mat = ro.r.matrix(
            ro.FloatVector(mat_combined.flatten(order="F")),
            nrow=mat_combined.shape[0],
            ncol=mat_combined.shape[1],
        )

    # Assign to R global environment so the inline R block can see them
    ro.globalenv["r_mat"]        = r_mat
    ro.globalenv["all_barcodes"] = ro.StrVector(all_barcodes.tolist())
    ro.globalenv["ref_barcodes"] = ro.StrVector(ref_barcodes.tolist())
    ro.globalenv["gene_names"]   = ro.StrVector(sub_genes.tolist())
    ro.globalenv["n_genes"]      = ro.IntVector([n])
    ro.globalenv["noise_val"]    = ro.FloatVector([noise])
    ro.globalenv["sig_thresh"]   = ro.FloatVector([signal_threshold])
    ro.globalenv["genome_name"]  = ro.StrVector([genome])

    # All four tutorial steps in one R call
    ro.r("""
        library(infercna)

        # STEP 1 — useGenome
        useGenome(genome_name)

        # Attach row/col names to the matrix
        rownames(r_mat) <- gene_names
        colnames(r_mat) <- all_barcodes

        # STEP 2 — infercna on combined matrix (query + ref)
        cna <- infercna(
            m        = r_mat,
            refCells = list(normal_ref = ref_barcodes),
            n        = n_genes,
            noise    = noise_val,
            isLog    = TRUE,
            verbose  = FALSE
        )

        # STEP 3 — strip reference columns -> cnaM (query cells only)
        cnaM <- cna[, !colnames(cna) %in% ref_barcodes, drop = FALSE]

        # STEP 4 — findMalignant on FULL cna
        # samples = per-cell vector; excludeFromAvg = ref barcodes
        n_query <- ncol(cna) - length(ref_barcodes)
        n_ref   <- length(ref_barcodes)
        sample_vec <- c(rep("tumor", n_query), rep("normal", n_ref))

        modes <- tryCatch(
            findMalignant(
                cna            = cna,
                signal.threshold = sig_thresh,
                samples        = sample_vec,
                excludeFromAvg = ref_barcodes
            ),
            error = function(e) {
                message("findMalignant error: ", conditionMessage(e))
                NULL
            }
        )
    """)

    modes = ro.globalenv["modes"]

    # ------------------------------------------------------------------
    # Parse R result → Python Series indexed by all query barcodes
    # ------------------------------------------------------------------
    # modes is NULL / FALSE when bimodal fitting fails
    is_null_or_false = (
        modes is ro.rinterface.NULL
        or (hasattr(modes, "typeof") and str(modes.typeof) == "logical")
        or not hasattr(modes, "names")
        or modes.names is None
    )

    if is_null_or_false:
        logger.warning(
            "inferCNA findMalignant() returned NULL/FALSE — bimodal fit "
            "did not converge (likely unimodal CNA distribution).\n"
            "All query cells labelled 'not.defined'.\n"
            "Try lowering infercna_signal_threshold (e.g. 0.75) or "
            "infercna_n (e.g. 3000)."
        )
        return pd.Series(
            "not.defined",
            index=q_barcodes,
            name="infercna_prediction",
        )

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
    # QC
    min_genes=200,
    max_mt=40.0,
    # DEG  (FIX 3: pvals_adj)
    log2fc_threshold=1.0,
    pval_adj_threshold=0.05,
    # paths (all auto-detected if not given)
    reference_h5ad=None,
    tumor_h5ad=None,
    save_dir=None,
    scmalignant_model_dir=None,
    surfaceome_path=None,
    # malignancy logic
    malignant_strategy="union",
    # inferCNA parameters
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
        If None, auto-loads from popv_path or
        'popv_results/final_popv_annotated.h5ad'.
    popv_path : str or None
        Explicit path to the PopV output h5ad.
    min_genes : int        Minimum genes per cell (QC). Default 200.
    max_mt : float         Maximum mitochondrial % (QC). Default 40.
    log2fc_threshold : float
        Log2FC cutoff for DEG (default 1.0 = 2-fold change).
    pval_adj_threshold : float
        BH-adjusted p-value cutoff for DEG (default 0.05).
        If 0 DEGs result, try 0.10 or lower log2fc_threshold.
    reference_h5ad : str or None
        Tabula Sapiens tissue-matched reference for inferCNA.
        Same file used in the PopV module works fine.
        If None, inferCNA is skipped.
    tumor_h5ad : str or None
        Path to the original Module 1 output h5ad (e.g. GSE158937_tumor.h5ad).
        Used by Route A-rescue to recover the full ~33k gene space for
        scMalignantFinder without re-running Module 2.
        Auto-detected from the current directory if not given.
    save_dir : str or None
        Output directory. Defaults to 'preprocessing_results/' in cwd.
    scmalignant_model_dir : str or None
        Auto-detected from SCART package if not given.
    surfaceome_path : str or None
        Auto-detected from SCART package if not given.
    malignant_strategy : str
        'union' | 'intersection' | 'scMalignant' | 'infercna'
    infercna_genome : str
        'hg19' (default, bundled in R package) or 'hg38'.
        This is a string KEY — not a file path.
    infercna_n : int
        Most-variable genes to retain before CNA inference (default 5000).
    infercna_noise : float
        Exclude genes with expression range < noise (default 0.1).
    infercna_signal_threshold : float
        Top fraction for cnaSignal/cnaCor (default 0.9).
        Lower to 0.75 if findMalignant() returns not.defined for all cells.
    infercna_ref_max_cells : int
        Maximum reference epithelial cells passed to R (default 2000).
        Reduces memory and runtime; 2000 cells gives a stable baseline.

    Returns
    -------
    AnnData  Binary expression matrix over surfaceome DEGs.
    """
    print("\n========== START ==========\n")

    # --- Resolve paths ------------------------------------------------------
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

    # Auto-detect Module 1 tumor h5ad for Route A-rescue
    if tumor_h5ad is None:
        tumor_h5ad = _auto_tumor_h5ad()
        if tumor_h5ad is not None:
            logger.info(f"Auto-detected tumor h5ad for Route A-rescue: {tumor_h5ad}")
        else:
            logger.info(
                "No tumor h5ad found in current directory. "
                "Route A-rescue disabled. Pass tumor_h5ad= to enable it."
            )

    # --- Auto-load adata ----------------------------------------------------
    if adata is None:
        for path in [popv_path,
                     "popv_results/final_popv_annotated.h5ad",
                     "final_popv_annotated.h5ad"]:
            if path and os.path.exists(path):
                print(f"Loading PopV output: {path}")
                adata = sc.read_h5ad(path)
                break
        if adata is None:
            raise FileNotFoundError(
                "Could not auto-detect PopV output. "
                "Pass adata= or popv_path= explicitly."
            )

    # Report gene-space status
    if tumor_h5ad and os.path.exists(tumor_h5ad):
        print(f"Route A-rescue enabled: {tumor_h5ad}")
    elif "full_counts" in adata.layers:
        print(f"Route A-new available: layers['full_counts']")
    elif adata.raw is not None:
        print(f"Route A-old available: adata.raw ({adata.raw.n_vars} genes)")
    else:
        print(
            "WARNING: No full-gene source found.\n"
            "  scMalignantFinder will fall back to 4000 HVGs (~19% overlap).\n"
            "  Add  tumor_h5ad='GSE158937_tumor.h5ad'  to the calling script."
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
    # 2. Quality control
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
    adata.layers["raw_for_cna"] = adata.X.copy()   # snapshot before log-norm

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
    adata.obs["scMalignantFinder_prediction"] = (
        result_scm.obs["scMalignantFinder_prediction"].values
    )
    print("scMalignantFinder completed.")
    print(adata.obs["scMalignantFinder_prediction"].value_counts().to_string(), "\n")

    # ------------------------------------------------------------------
    # 5. inferCNA
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
                f"(reference subsampled to {infercna_ref_max_cells} cells) ..."
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

    # ------------------------------------------------------------------
    # 6. Combine malignancy calls → final_malignant
    # ------------------------------------------------------------------
    scm_mal = adata.obs["scMalignantFinder_prediction"].str.lower() == "malignant"

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

    adata.obs["final_malignant"] = malignant_mask.map(
        {True: "malignant", False: "normal"}
    )
    print(f"Malignancy strategy: {strategy_label}")
    print(f"  Malignant: {malignant_mask.sum()} | Normal: {(~malignant_mask).sum()}\n")

    # ------------------------------------------------------------------
    # 7. Surfaceome filter
    # ------------------------------------------------------------------
    surfaceome = pd.read_csv(surfaceome_path)
    surfaceome.columns = surfaceome.columns.str.strip()
    surf_genes = surfaceome["Gene"].astype(str).tolist()
    adata      = adata[:, adata.var_names.intersection(surf_genes)].copy()
    print(f"Surfaceome genes retained: {adata.n_vars}\n")

    # ------------------------------------------------------------------
    # 8. DEG  (FIX 3: pvals_adj not raw pvals)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 9. Binarise
    # ------------------------------------------------------------------
    adata.X = (adata.X > 0).astype(int)
    print("Expression converted to binary (0/1).\n")

    # ------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object:
            adata.obs[col] = adata.obs[col].astype(str)

    final_path = os.path.join(save_dir, "final_tumor.h5ad")
    adata.write(final_path)
    print(f"Final object saved to: {final_path}")
    print("\n========== PREPROCESSING COMPLETED ==========\n")
    return adata
