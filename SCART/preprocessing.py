"""
preprocessing.py
Module 3 — Preprocessing, malignancy detection, and surfaceome DEG

Root-cause summary (from diagnose_preprocessing.py v2)
-------------------------------------------------------
ISSUE 1 — scMalignantFinder 81% missing features
  var_names are HGNC symbols — format was never the problem.
  The 4000 genes are scVI HVGs selected by Process_Query inside Module 2.
  Only ~19% of scMalignantFinder's 2707 required genes fall in that HVG set.

  Fix location: Module 2 (popv_annotation.py).
    _capture_raw_slot()  — grabs adata_query.raw BEFORE Process_Query runs.
    _reattach_raw_slot() — puts it back on adata_query_out AFTER annotation.
  The saved final_popv_annotated.h5ad now has adata.raw with the full
  ~33k gene space (set by Module 1 via  adata.raw = adata).

  _build_fullgene_adata_for_scm() here tries Route A (adata.raw) first
  and will now succeed.  Routes B and C are kept as safety fallbacks.

ISSUE 2 — CopyKAT KeyError on obs_names
  copykat drops cells failing min.gene.per.cell internally.
  Fix: .reindex(query_barcodes, fill_value='not.defined') inside
  _run_copykat() so every query barcode always gets a label.

CONFIRMED GOOD
  - var_names are HGNC symbols — no renaming needed.
  - layers['scvi_counts'] is integer raw counts (max=146) — suitable for CopyKAT.
  - epithelial filter (.endswith('epithelial cell')) catches both label types.
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
    # adata.X is already log-normalised at this point in the pipeline
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
# CopyKAT integration (via rpy2)
# ===========================================================================

def _run_copykat(
    adata_query,
    adata_ref,
    sam_name: str = "copykat_run",
    id_type: str = "S",
    ngene_chr: int = 5,
    win_size: int = 25,
    ks_cut: float = 0.1,
    distance: str = "euclidean",
    genome: str = "hg20",
    n_cores: int = 4,
    output_dir: str = None,
):
    """
    Run CopyKAT via rpy2.

    Returns a pd.Series indexed by ALL query obs_names.
    Cells dropped by copykat's internal filter are filled with 'not.defined'.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
        numpy2ri.activate()
    except ImportError as exc:
        raise ImportError(
            "rpy2 is required to run CopyKAT.\n"
            "Install: pip install rpy2\n"
            "R package: devtools::install_github('navinlabcode/copykat')"
        ) from exc

    try:
        copykat_r = importr("copykat")
    except Exception as exc:
        raise ImportError(
            "R package 'copykat' not found.\n"
            "Install in R: devtools::install_github('navinlabcode/copykat')"
        ) from exc

    # --- 1. Raw matrices (genes x cells) ------------------------------------
    mat_query = _get_raw_matrix(adata_query).T

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
        logger.info(f"CopyKAT reference: {ep_mask.sum()} epithelial cells")
    else:
        adata_ref_ep = adata_ref

    mat_ref = _get_raw_matrix(adata_ref_ep).T

    # --- 2. Common genes ----------------------------------------------------
    query_genes  = np.array(adata_query.var_names)
    ref_genes    = np.array(adata_ref_ep.var_names)
    common_genes = np.intersect1d(query_genes, ref_genes)
    logger.info(f"CopyKAT common genes: {len(common_genes)}")

    if len(common_genes) < 100:
        raise ValueError(
            f"Only {len(common_genes)} common genes between query and reference. "
            "Check that both use HGNC gene symbols."
        )

    q_idx = np.where(np.isin(query_genes, common_genes))[0]
    r_idx = np.where(np.isin(ref_genes,   common_genes))[0]
    q_order = np.argsort(query_genes[q_idx])
    r_order = np.argsort(ref_genes[r_idx])

    mat_query_sub = mat_query[q_idx, :][q_order, :]
    mat_ref_sub   = mat_ref[r_idx,   :][r_order, :]
    sorted_genes  = query_genes[q_idx][q_order]

    # --- 3. Prefix ref barcodes and combine ---------------------------------
    query_barcodes = np.array(adata_query.obs_names)
    ref_barcodes   = np.array(["REF_" + b for b in adata_ref_ep.obs_names])

    mat_combined = np.hstack([mat_query_sub, mat_ref_sub])
    all_barcodes = np.concatenate([query_barcodes, ref_barcodes])

    # --- 4. Pass to R -------------------------------------------------------
    r_mat = ro.r.matrix(
        ro.FloatVector(mat_combined.flatten(order="F")),
        nrow=mat_combined.shape[0],
        ncol=mat_combined.shape[1],
        dimnames=ro.ListVector([
            ro.StrVector(sorted_genes.tolist()),
            ro.StrVector(all_barcodes.tolist()),
        ]),
    )
    r_normal_cells = ro.StrVector(ref_barcodes.tolist())

    use_dir = output_dir or tempfile.mkdtemp(prefix="copykat_")
    original_dir = os.getcwd()
    os.chdir(use_dir)

    try:
        result = copykat_r.copykat(
            rawmat          = r_mat,
            id_type         = id_type,
            ngene_chr       = ngene_chr,
            win_size        = win_size,
            KS_cut          = ks_cut,
            sam_name        = sam_name,
            distance        = distance,
            norm_cell_names = r_normal_cells,
            output_seg      = "FALSE",
            plot_genes      = "TRUE",
            genome          = genome,
            n_cores         = n_cores,
        )
    finally:
        os.chdir(original_dir)

    # --- 5. Robust barcode reindex (FIX 2) ----------------------------------
    pred_df = pandas2ri.rpy2py(result.rx2("prediction"))
    pred_df.columns = [c.strip() for c in pred_df.columns]

    if "cell.names" in pred_df.columns:
        pred_df = pred_df.set_index("cell.names")

    pred_df = pred_df[~pred_df.index.str.startswith("REF_")]

    n_returned = len(pred_df)
    full_preds = pred_df["copykat.pred"].reindex(
        query_barcodes, fill_value="not.defined"
    )
    n_dropped = len(query_barcodes) - n_returned
    if n_dropped > 0:
        logger.warning(
            f"CopyKAT dropped {n_dropped} query cells internally "
            f"(min.gene.per.cell filter) — labelled 'not.defined'."
        )

    full_preds.name = "copykat_prediction"
    logger.info("CopyKAT predictions:\n" + full_preds.value_counts().to_string())
    return full_preds


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
    copykat_params: dict = None,
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
        Required for CopyKAT.  If None, CopyKAT is skipped.
    n_cores : int
        CPU cores for CopyKAT.
    malignant_strategy : str
        'union'        — malignant if EITHER method says so (recommended)
        'intersection' — malignant only if BOTH agree (more specific)
        'scMalignant'  — scMalignantFinder only
        'copykat'      — CopyKAT only (requires reference_h5ad)
    copykat_params : dict or None
        Optional CopyKAT parameter overrides. Valid keys:
          id_type, ngene_chr, win_size, ks_cut, distance, genome
        Example: {"ks_cut": 0.05, "win_size": 50, "distance": "pearson"}

    Returns
    -------
    AnnData
        Binary expression matrix over surfaceome DEGs, with obs columns:
          scMalignantFinder_prediction, copykat_prediction (if run),
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

    # Report raw slot status immediately so the user knows which route will run
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
    # 3. Route raw counts into .X and snapshot for CopyKAT
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

    # Snapshot raw integer counts for CopyKAT BEFORE log-normalisation
    adata.layers["raw_for_copykat"] = adata.X.copy()

    # Normalise for classifiers and DEG
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ------------------------------------------------------------------
    # 4. scMalignantFinder  (Route A now works via Module 2 fix)
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
    # 5. CopyKAT  (reindex fix inside _run_copykat)
    # ------------------------------------------------------------------
    copykat_available = False
    _ck_defaults = dict(
        id_type   = "S",
        ngene_chr = 5,
        win_size  = 25,
        ks_cut    = 0.1,
        distance  = "euclidean",
        genome    = "hg20",
    )
    if copykat_params:
        _ck_defaults.update(copykat_params)

    if malignant_strategy in ("copykat", "union", "intersection"):
        if reference_h5ad is None:
            print(
                "Warning: CopyKAT skipped — no reference_h5ad provided.\n"
                "  Falling back to scMalignantFinder only."
            )
            malignant_strategy = "scMalignant"
        else:
            print("Running CopyKAT ...")
            try:
                adata_raw_ck   = adata.copy()
                adata_raw_ck.X = adata.layers["raw_for_copykat"]
                adata_ref_full = sc.read_h5ad(reference_h5ad)

                copykat_preds = _run_copykat(
                    adata_query = adata_raw_ck,
                    adata_ref   = adata_ref_full,
                    n_cores     = n_cores,
                    **_ck_defaults,
                )
                adata.obs["copykat_prediction"] = copykat_preds.values
                copykat_available = True
                print("CopyKAT completed.")
                print(
                    adata.obs["copykat_prediction"].value_counts().to_string(),
                    "\n"
                )
            except Exception as exc:
                print(
                    f"Warning: CopyKAT failed: {type(exc).__name__}: {exc}\n"
                    "  Falling back to scMalignantFinder only."
                )
                logger.exception("CopyKAT error details:")
                malignant_strategy = "scMalignant"

    # ------------------------------------------------------------------
    # 6. Combine malignancy calls
    # ------------------------------------------------------------------
    scm_mal = (
        adata.obs["scMalignantFinder_prediction"].str.lower() == "malignant"
    )

    if copykat_available:
        ck_mal = adata.obs["copykat_prediction"].str.lower() == "aneuploid"

        if malignant_strategy == "union":
            malignant_mask = scm_mal | ck_mal
            strategy_label = "union (scMalignantFinder OR CopyKAT)"
        elif malignant_strategy == "intersection":
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
