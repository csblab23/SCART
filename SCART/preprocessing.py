"""
preprocessing.py
Module 3 — Preprocessing, malignancy detection, and surfaceome DEG

Malignancy detection uses TWO complementary CNV-based methods:
  1. scMalignantFinder  — deep-learning classifier (expression-based, fast)
  2. inferCNA           — CNA inference via rolling-mean smoothing over
                          chromosomally-ordered genes; no clustering needed,
                          produces per-cell cnaSignal + cnaCor scores and a
                          binary malignant/non-malignant call via bimodal
                          Gaussian fitting (findMalignant).

Why inferCNA instead of CopyKAT?
  - inferCNA uses the same mathematical framework (Tirosh/Patel lineage) but
    is lighter: no hierarchical clustering of 50k+ cells, no KS-test boundary.
  - findMalignant() fits bimodal Gaussians to cnaSignal × cnaCor and returns
    the malignant mode directly — no manual threshold tuning required.
  - inferCNA runs in ~5 minutes on 8k epithelial cells vs ~2 hours for CopyKAT
    on the same data.

A cell is labelled malignant if EITHER method calls it malignant (union
strategy, configurable via malignant_strategy).

inferCNA is run via rpy2 (R package).  If rpy2 / the R inferCNA package are
not available the step is skipped gracefully and the pipeline continues with
scMalignantFinder alone.
"""

import os
import logging
import tempfile

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SURFACEOME_PATH = (
    "/lustre/anas.a/Vinaya/scT-CAR_Designer/GESP/GESP_surfaceome_gene.csv"
)
SCMALIGNANT_MODEL = (
    "/home/igib/anaconda3/envs/scart/lib/python3.10/site-packages/"
    "SCART/external/scMalignantFinder/model"
)
SAVE_DIR = "/lustre/anas.a/Vinaya/scT-CAR_Designer/preprocessed_input"
os.makedirs(SAVE_DIR, exist_ok=True)


# ===========================================================================
# Build full-gene AnnData for scMalignantFinder
# ===========================================================================

def _build_fullgene_adata_for_scm(adata, feature_tsv: str):
    """
    Return a log-normalised AnnData covering the full original gene space.

    Route A  — adata.raw  (set by Module 2's _reattach_raw_slot fix)
    Route B  — adata.uns['full_var_names'] (SCART-specific, future-proofing)
    Route C  — fallback: 4000 HVG adata as-is, with a warning

    The returned object is used ONLY for scMalignantFinder.
    The caller's adata (4000 HVGs) is not modified.
    """
    model_features = set(
        pd.read_csv(feature_tsv, sep="\t", header=None)[0].tolist()
    )
    n_model = len(model_features)

    def _pct(names):
        return len(set(names) & model_features) / n_model * 100

    # --- Route A: adata.raw -------------------------------------------------
    if adata.raw is not None:
        raw_var_names = adata.raw.var_names
        ov = _pct(raw_var_names)
        logger.info(
            f"Route A (adata.raw): {adata.raw.n_vars} genes, "
            f"{ov:.1f}% model overlap"
        )
        if ov >= 50:
            X_raw = adata.raw.X
            if sp.issparse(X_raw):
                X_raw = X_raw.toarray()
            adata_full = sc.AnnData(
                X   = X_raw.astype(np.float32),
                obs = adata.obs.copy(),
                var = adata.raw.var.copy(),
            )
            sc.pp.normalize_total(adata_full, target_sum=1e4)
            sc.pp.log1p(adata_full)
            logger.info("Using Route A (adata.raw) for scMalignantFinder.")
            return adata_full
        else:
            logger.warning(
                f"Route A: adata.raw present but only {ov:.1f}% model overlap. "
                "adata.raw may still be HVG-filtered. Trying Route B."
            )

    # --- Route B: uns['full_var_names'] -------------------------------------
    if "full_var_names" in adata.uns:
        full_var = list(adata.uns["full_var_names"])
        ov = _pct(full_var)
        logger.info(
            f"Route B (uns['full_var_names']): {len(full_var)} genes, "
            f"{ov:.1f}% model overlap"
        )
        for layer in ("scvi_counts", "raw_counts", "counts"):
            if layer in adata.layers:
                X_layer = adata.layers[layer]
                if sp.issparse(X_layer):
                    X_layer = X_layer.toarray()
                if X_layer.shape[1] == len(full_var) and ov >= 50:
                    adata_full = sc.AnnData(
                        X   = X_layer.astype(np.float32),
                        obs = adata.obs.copy(),
                        var = pd.DataFrame(index=full_var),
                    )
                    sc.pp.normalize_total(adata_full, target_sum=1e4)
                    sc.pp.log1p(adata_full)
                    logger.info(
                        f"Using Route B (uns['full_var_names'] + "
                        f"layers['{layer}']) for scMalignantFinder."
                    )
                    return adata_full

    # --- Route C: fallback --------------------------------------------------
    ov_hvg = _pct(adata.var_names)
    logger.warning(
        f"Could not find full-gene matrix (Routes A and B failed). "
        f"Falling back to {adata.n_vars} HVGs ({ov_hvg:.1f}% model overlap). "
        f"scMalignantFinder results will be unreliable.\n"
        f"Ensure Module 2 (popv_annotation.py) is the fixed version that "
        f"calls _reattach_raw_slot() before saving final_popv_annotated.h5ad."
    )
    return adata.copy()


# ===========================================================================
# Helper: extract raw count matrix
# ===========================================================================

def _get_raw_matrix(adata):
    """Return a dense float64 array (cells x genes) of raw integer counts."""
    for layer in ("scvi_counts", "raw_counts", "counts"):
        if layer in adata.layers:
            logger.info(f"Raw counts sourced from adata.layers['{layer}']")
            X = adata.layers[layer]
            break
    else:
        if adata.raw is not None:
            logger.info("Raw counts sourced from adata.raw.X")
            X = adata.raw.X
        else:
            logger.info("No dedicated raw layer — assuming adata.X is raw counts")
            X = adata.X

    if sp.issparse(X):
        X = X.toarray()
    return np.array(X, dtype=np.float64)


# ===========================================================================
# inferCNA integration (via rpy2)
# ===========================================================================

def _run_infercna(
    adata_query,
    adata_ref,
    genome: str = "hg19",
    n_top_genes: int = 5000,
    noise: float = 0.1,
    window: int = 101,
    signal_threshold: float = 0.9,
    n_cores: int = 4,
):
    """
    Run inferCNA via rpy2 and return a pd.Series of malignancy predictions
    indexed by ALL query cell barcodes.

    Prediction values: 'malignant' | 'non-malignant' | 'not.defined'

    How inferCNA works (brief)
    --------------------------
    1. orderGenes()    — genes are sorted by chromosomal position (hg19/hg38).
    2. infercna()      — for each cell, expression values are smoothed with a
                         rolling mean of window size `n` across the ordered
                         gene list.  This converts the expression profile into
                         a CNA profile.  If reference (normal) cells are given,
                         refCorrect() subtracts their average to produce
                         absolute rather than relative CNA values.
    3. cnaSignal()     — mean of squared CNA values across the genome per cell.
                         High signal → many / large copy-number changes →
                         likely malignant.
    4. cnaCor()        — Pearson correlation of each cell's CNA profile against
                         the tumour-average CNA profile.  Malignant cells are
                         self-similar; normal cells are not.
    5. findMalignant() — fits bimodal Gaussian distributions to cnaSignal and
                         cnaCor.  If two modes are found and are compatible,
                         the lower mode = non-malignant, upper mode = malignant.
                         Returns a named list: list(nonmalignant=..., malignant=...).

    Parameters
    ----------
    adata_query : AnnData
        Epithelial query cells with raw counts in `scvi_counts` / `raw_counts`
        layer (or .X if no layer present).
    adata_ref : AnnData
        Normal reference (Tabula Sapiens); epithelial cells are extracted
        internally using cell_ontology_class.
    genome : str
        Genome build for gene ordering: 'hg19' | 'hg38'.
        inferCNA ships hg19 by default; hg38 requires addGenome().
    n_top_genes : int
        n parameter passed to infercna() — number of most variable genes to
        keep before CNA inference (reduces noise).
    noise : float
        noise parameter in infercna() — genes with expression range < noise
        across all cells are excluded from the CNA calculation.
    window : int
        Rolling-mean window size (infercna calls it n internally, mapped to
        the runMean window).  Default 101 matches the original Tirosh method.
    signal_threshold : float
        Top fraction of genes used for cnaSignal / cnaCor calculation.
        0.9 = top 10% of genes by CNA signal (recommended in the tutorial).
    n_cores : int
        Not used by inferCNA itself (it is single-threaded) but kept for
        API consistency with the rest of the pipeline.

    Returns
    -------
    pd.Series
        Index = query barcodes, values = 'malignant' | 'non-malignant' |
        'not.defined'.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
        numpy2ri.activate()
    except ImportError as exc:
        raise ImportError(
            "rpy2 is required to run inferCNA.\n"
            "Install: pip install rpy2\n"
            "R package: devtools::install_github('jlaffy/infercna')"
        ) from exc

    try:
        infercna_r = importr("infercna")
    except Exception as exc:
        raise ImportError(
            "R package 'infercna' not found.\n"
            "Install in R:\n"
            "  install.packages('devtools')\n"
            "  devtools::install_github('jlaffy/infercna')"
        ) from exc

    base_r = importr("base")

    # ------------------------------------------------------------------
    # 1. Build the log-normalised expression matrix for inferCNA
    #    inferCNA expects: genes x cells, log-normalised (e.g. log2(CPM/10+1))
    #    We use log1p(CPM) which is close enough; isLog=TRUE is passed to
    #    infercna() to tell it the data is already in log space.
    # ------------------------------------------------------------------
    # Query matrix
    X_query = _get_raw_matrix(adata_query)          # cells x genes
    # Normalise to CPM and log1p
    row_sums = X_query.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X_query_norm = np.log1p(X_query / row_sums * 1e6)  # log1p(CPM)
    mat_query = X_query_norm.T                          # genes x cells

    # Reference matrix — extract epithelial cells
    epithelial_terms = {
        "epithelial cell",
        "glandular epithelial cell",
        "ovarian surface epithelial cell",
    }
    if "cell_ontology_class" in adata_ref.obs.columns:
        ep_mask = adata_ref.obs["cell_ontology_class"].str.lower().isin(
            epithelial_terms
        )
        adata_ref_ep = adata_ref[ep_mask].copy() if ep_mask.any() else adata_ref
        logger.info(
            f"inferCNA reference: {ep_mask.sum()} epithelial cells used as normal"
        )
    else:
        adata_ref_ep = adata_ref

    X_ref = _get_raw_matrix(adata_ref_ep)
    row_sums_ref = X_ref.sum(axis=1, keepdims=True)
    row_sums_ref[row_sums_ref == 0] = 1
    X_ref_norm = np.log1p(X_ref / row_sums_ref * 1e6)
    mat_ref = X_ref_norm.T                              # genes x cells

    # ------------------------------------------------------------------
    # 2. Find common genes between query and reference, align gene order
    # ------------------------------------------------------------------
    query_genes = np.array(adata_query.var_names)
    ref_genes   = np.array(adata_ref_ep.var_names)
    common_genes = np.intersect1d(query_genes, ref_genes)
    logger.info(f"inferCNA common genes: {len(common_genes)}")

    if len(common_genes) < 200:
        raise ValueError(
            f"Only {len(common_genes)} common genes between query and reference. "
            "inferCNA needs at least ~200 genes to be reliable. "
            "Check that both use HGNC gene symbols."
        )

    q_idx = np.where(np.isin(query_genes, common_genes))[0]
    r_idx = np.where(np.isin(ref_genes,   common_genes))[0]

    mat_query_sub = mat_query[q_idx, :]
    mat_ref_sub   = mat_ref[r_idx,   :]
    sub_genes     = query_genes[q_idx]

    # ------------------------------------------------------------------
    # 3. Pass data to R
    # ------------------------------------------------------------------
    query_barcodes = np.array(adata_query.obs_names)
    ref_barcodes   = np.array(["REF_" + b for b in adata_ref_ep.obs_names])

    # Combined matrix: genes x (query_cells + ref_cells)
    mat_combined  = np.hstack([mat_query_sub, mat_ref_sub])
    all_barcodes  = np.concatenate([query_barcodes, ref_barcodes])

    r_mat = ro.r.matrix(
        ro.FloatVector(mat_combined.flatten(order="F")),
        nrow=mat_combined.shape[0],
        ncol=mat_combined.shape[1],
        dimnames=ro.ListVector([
            ro.StrVector(sub_genes.tolist()),
            ro.StrVector(all_barcodes.tolist()),
        ]),
    )

    # refCells argument: named list with one entry 'normal' = REF_ barcodes
    r_ref_cells = ro.ListVector({
        "normal": ro.StrVector(ref_barcodes.tolist())
    })

    # ------------------------------------------------------------------
    # 4. Set genome and run infercna()
    # ------------------------------------------------------------------
    infercna_r.useGenome(genome)

    logger.info("Running infercna() — CNA inference ...")
    cna = infercna_r.infercna(
        m        = r_mat,
        refCells = r_ref_cells,
        n        = n_top_genes,
        noise    = noise,
        isLog    = True,
        verbose  = False,
    )

    # ------------------------------------------------------------------
    # 5. Find malignant cells with findMalignant()
    #    findMalignant fits bimodal Gaussians to cnaSignal x cnaCor.
    #    It returns a list: list(nonmalignant=c(...), malignant=c(...))
    #    or FALSE if the bimodal fit fails.
    # ------------------------------------------------------------------
    logger.info("Running findMalignant() — bimodal Gaussian fitting ...")

    # Use the full combined cna matrix (query + ref rows) for a better
    # average tumour profile, but pass ref barcodes as excludeFromAvg
    # so they don't skew the tumour average.
    try:
        modes = infercna_r.findMalignant(
            cna              = cna,
            signal_threshold = signal_threshold,
            samples          = "query",            # sample name label
            excludeFromAvg   = ro.StrVector(ref_barcodes.tolist()),
        )
    except Exception:
        # findMalignant may return FALSE if bimodal fit fails —
        # rpy2 raises an exception when R returns FALSE as a non-list.
        modes = None

    # ------------------------------------------------------------------
    # 6. Convert R result to Python Series indexed by query barcodes
    # ------------------------------------------------------------------
    if modes is None or ro.rinterface.NULL == modes:
        logger.warning(
            "inferCNA findMalignant() returned NULL/FALSE — "
            "bimodal fitting failed (likely unimodal CNA distribution). "
            "All query cells labelled 'not.defined'."
        )
        return pd.Series(
            "not.defined",
            index=query_barcodes,
            name="infercna_prediction",
        )

    # Convert R list to Python dict of cell-name sets
    try:
        result_dict = {
            str(key): list(modes.rx2(key))
            for key in list(modes.names)
        }
    except Exception as exc:
        logger.warning(
            f"inferCNA: could not parse findMalignant output: {exc}. "
            "All query cells labelled 'not.defined'."
        )
        return pd.Series(
            "not.defined",
            index=query_barcodes,
            name="infercna_prediction",
        )

    # Build label map: barcode → prediction string
    label_map = {}
    for mode_name, barcodes in result_dict.items():
        label = "malignant" if "malignant" in mode_name.lower() else "non-malignant"
        for bc in barcodes:
            label_map[bc] = label

    # Map to query barcodes; cells not in findMalignant output → 'not.defined'
    # (These are reference cells or cells excluded from the fit)
    preds = pd.Series(
        [label_map.get(bc, "not.defined") for bc in query_barcodes],
        index=query_barcodes,
        name="infercna_prediction",
    )

    # Drop REF_ cells from the output — keep query cells only
    preds = preds[~preds.index.str.startswith("REF_")]

    # Reindex against query barcodes to guarantee full coverage
    preds = preds.reindex(query_barcodes, fill_value="not.defined")

    n_mal = (preds == "malignant").sum()
    n_nor = (preds == "non-malignant").sum()
    n_und = (preds == "not.defined").sum()
    logger.info(
        f"inferCNA predictions — malignant: {n_mal}, "
        f"non-malignant: {n_nor}, not.defined: {n_und}"
    )

    return preds


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_preprocessing_pipeline(
    adata=None,
    popv_path: str = None,
    min_genes: int = 200,
    max_mt: float = 40.0,
    log2fc_threshold: float = 2.0,
    pval_threshold: float = 0.5,
    reference_h5ad: str = None,
    n_cores: int = 4,
    malignant_strategy: str = "union",
    infercna_params: dict = None,
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
    min_genes : int
        Minimum genes per cell (QC filter).
    max_mt : float
        Maximum mitochondrial % per cell (QC filter).
    log2fc_threshold : float
        Log2 fold-change cutoff for DEG.
    pval_threshold : float
        P-value cutoff for DEG.
    reference_h5ad : str or None
        Path to the normal reference h5ad (Tabula Sapiens or equivalent).
        Required for inferCNA.  If None, inferCNA is skipped.
    n_cores : int
        Passed through for API consistency (inferCNA is single-threaded).
    malignant_strategy : str
        'union'        — malignant if EITHER method says so (recommended)
        'intersection' — malignant only if BOTH agree (more specific)
        'scMalignant'  — scMalignantFinder only
        'infercna'     — inferCNA only (requires reference_h5ad)
    infercna_params : dict or None
        Optional inferCNA parameter overrides. Valid keys:
          genome           : 'hg19' (default) | 'hg38'
          n_top_genes      : int, default 5000
          noise            : float, default 0.1
          window           : int, default 101
          signal_threshold : float, default 0.9
        Example: {"genome": "hg38", "signal_threshold": 0.85}

    Returns
    -------
    AnnData
        Binary expression matrix over surfaceome DEGs, with obs columns:
          scMalignantFinder_prediction, infercna_prediction (if run),
          final_malignant.
        adata.uns['filtered_deg'] contains the DEG result table.
    """
    print("\n========== START ==========\n")

    # ------------------------------------------------------------------
    # Auto-load adata
    # ------------------------------------------------------------------
    if adata is None:
        candidates = [
            popv_path,
            "popv_results/final_popv_annotated.h5ad",
            "final_popv_annotated.h5ad",
        ]
        for path in candidates:
            if path and os.path.exists(path):
                print(f"Loading POPV output (auto): {path}")
                adata = sc.read_h5ad(path)
                break
        if adata is None:
            raise FileNotFoundError(
                "Could not auto-detect POPV output. "
                "Pass adata= or popv_path= explicitly."
            )

    if adata.raw is not None:
        print(f"adata.raw detected: {adata.raw.n_vars} genes "
              f"(scMalignantFinder will use full gene space via Route A)")
    else:
        print("WARNING: adata.raw is None — scMalignantFinder will use "
              "4000 HVGs only (19% feature overlap). Re-run Module 2 with "
              "the fixed popv_annotation.py to resolve this.")

    initial_cells = adata.n_obs
    print(f"Initial cells: {initial_cells}")

    # ------------------------------------------------------------------
    # 1. Select epithelial cells
    # ------------------------------------------------------------------
    labels  = adata.obs["popv_majority_vote_prediction"].astype(str)
    ep_mask = labels.str.endswith("epithelial cell")
    adata   = adata[ep_mask].copy()
    print(f"Epithelial cells retained: {adata.n_obs}")
    print(f"Cells removed:             {initial_cells - adata.n_obs}\n")

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
    # 3. Route raw counts into .X and snapshot for inferCNA
    # ------------------------------------------------------------------
    print("Detecting raw count source...")
    for layer in ("scvi_counts", "raw_counts", "counts"):
        if layer in adata.layers:
            print(f"Using adata.layers['{layer}'] as raw counts.")
            adata.X = adata.layers[layer].copy()
            break
    else:
        if adata.raw is not None:
            print("Using adata.raw.X as raw counts.")
            adata.X = adata.raw.X.copy()
        else:
            print("No raw layer found — assuming adata.X is raw counts.")

    adata.var_names_make_unique()

    # Snapshot raw integer counts for inferCNA BEFORE log-normalisation
    adata.layers["raw_for_cna"] = adata.X.copy()

    # Normalise for classifiers and DEG
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ------------------------------------------------------------------
    # 4. scMalignantFinder  (Route A works via Module 2 raw-slot fix)
    # ------------------------------------------------------------------
    print("Running scMalignantFinder ...")
    feature_tsv = os.path.join(SCMALIGNANT_MODEL, "ordered_feature.tsv")

    print("  Building full-gene matrix for scMalignantFinder ...")
    adata_scm = _build_fullgene_adata_for_scm(adata, feature_tsv)
    print(f"  Gene space for classifier: {adata_scm.n_vars} genes")

    from scMalignantFinder import classifier
    model = classifier.scMalignantFinder(
        test_input          = adata_scm,
        celltype_annotation = False,
        pretrain_path       = SCMALIGNANT_MODEL,
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
    _icna_defaults = dict(
        genome           = "hg19",
        n_top_genes      = 5000,
        noise            = 0.1,
        window           = 101,
        signal_threshold = 0.9,
        n_cores          = n_cores,
    )
    if infercna_params:
        _icna_defaults.update(infercna_params)

    if malignant_strategy in ("infercna", "union", "intersection"):
        if reference_h5ad is None:
            print(
                "Warning: inferCNA skipped — no reference_h5ad provided.\n"
                "  Falling back to scMalignantFinder only."
            )
            malignant_strategy = "scMalignant"
        else:
            print("Running inferCNA ...")
            try:
                adata_raw_cna   = adata.copy()
                adata_raw_cna.X = adata.layers["raw_for_cna"]
                adata_ref_full  = sc.read_h5ad(reference_h5ad)

                infercna_preds = _run_infercna(
                    adata_query = adata_raw_cna,
                    adata_ref   = adata_ref_full,
                    **_icna_defaults,
                )
                adata.obs["infercna_prediction"] = infercna_preds.values
                infercna_available = True
                print("inferCNA completed.")
                print(
                    adata.obs["infercna_prediction"].value_counts().to_string(),
                    "\n"
                )
            except Exception as exc:
                print(
                    f"Warning: inferCNA failed: {type(exc).__name__}: {exc}\n"
                    "  Falling back to scMalignantFinder only."
                )
                logger.exception("inferCNA error details:")
                malignant_strategy = "scMalignant"

    # ------------------------------------------------------------------
    # 6. Combine malignancy calls
    # ------------------------------------------------------------------
    scm_mal = (
        adata.obs["scMalignantFinder_prediction"].str.lower() == "malignant"
    )

    if infercna_available:
        cna_mal = adata.obs["infercna_prediction"].str.lower() == "malignant"

        if malignant_strategy == "union":
            malignant_mask = scm_mal | cna_mal
            strategy_label = "union (scMalignantFinder OR inferCNA)"
        elif malignant_strategy == "intersection":
            malignant_mask = scm_mal & cna_mal
            strategy_label = "intersection (scMalignantFinder AND inferCNA)"
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
    surfaceome = pd.read_csv(SURFACEOME_PATH)
    surfaceome.columns = surfaceome.columns.str.strip()
    surf_genes = surfaceome["Gene"].astype(str).tolist()
    common     = adata.var_names.intersection(surf_genes)
    adata      = adata[:, common].copy()
    print(f"Surfaceome genes retained: {adata.n_vars}\n")

    # ------------------------------------------------------------------
    # 8. DEG (malignant vs normal)
    # ------------------------------------------------------------------
    sc.tl.rank_genes_groups(
        adata, groupby="final_malignant", method="wilcoxon"
    )
    result_deg   = sc.get.rank_genes_groups_df(adata, group=None)
    filtered_deg = result_deg[
        (result_deg["logfoldchanges"] > log2fc_threshold) &
        (result_deg["pvals"] < pval_threshold)
    ]
    adata.uns["filtered_deg"] = filtered_deg
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

    final_path = os.path.join(SAVE_DIR, "final_tumor.h5ad")
    adata.write(final_path)
    print(f"Final object saved to:\n{final_path}")
    print("\n========== PREPROCESSING COMPLETED ==========\n")

    return adata
