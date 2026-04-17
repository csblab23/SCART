"""
preprocessing.py
Module 3 — Preprocessing, malignancy detection, and surfaceome DEG

Malignancy detection uses TWO complementary methods:
  1. scMalignantFinder  — deep-learning classifier (single-sample, fast)
  2. CopyKAT            — CNV-based inference (requires normal reference cells)

A cell is labelled malignant if EITHER method calls it malignant
(union strategy, configurable via malignant_strategy).

CopyKAT is run via rpy2 because the canonical implementation is an R package.
If rpy2 / the R copykat package are not available, CopyKAT is skipped and a
warning is printed; the pipeline continues with scMalignantFinder alone.
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
# Paths — edit these to match your installation
# ---------------------------------------------------------------------------
SURFACEOME_PATH = (
    "/lustre/anas.a/Vinaya/scT-CAR_Designer/GESP/GESP_surfaceome_gene.csv"
)
SCMALIGNANT_MODEL = "/lustre/anas.a/scMalignantFinder/model/"
SAVE_DIR = "/lustre/anas.a/Vinaya/scT-CAR_Designer/preprocessed_input"
os.makedirs(SAVE_DIR, exist_ok=True)


# ===========================================================================
# Helper: extract raw count matrix from an AnnData
# ===========================================================================

def _get_raw_matrix(adata):
    """
    Return a dense numpy array of raw integer counts (cells × genes).
    Priority: layers['scvi_counts'] > layers['raw_counts'] > layers['counts']
    > adata.raw.X > adata.X (assumed raw if nothing else found).
    """
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
            logger.info("No dedicated raw layer found — assuming adata.X is raw counts")
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
    n_cores: int = 1,
    output_dir: str = None,
):
    """
    Run CopyKAT via rpy2 and return a Series of predictions indexed by
    the query cell barcodes.

    Prediction values: 'aneuploid' (malignant), 'diploid' (normal),
    or 'not.defined'.

    Parameters
    ----------
    adata_query : AnnData
        Epithelial query cells (already QC-filtered).
    adata_ref : AnnData
        Normal reference (Tabula Sapiens or equivalent); epithelial cells
        are extracted internally using cell_ontology_class.
    sam_name : str
        Prefix for copykat output files.
    id_type : str
        Gene ID type: 'S' = gene symbol, 'E' = Ensembl.
    ngene_chr : int
        Minimum genes per chromosome segment (copykat ngene.chr).
    win_size : int
        Smoothing window size (copykat win.size).
    ks_cut : float
        KS statistic cutoff (copykat KS.cut).
    distance : str
        Distance metric for hierarchical clustering ('euclidean' | 'pearson' | 'spearman').
    genome : str
        Reference genome build ('hg20' = hg38 | 'hg19' | 'mm10').
    n_cores : int
        Parallel cores for copykat.
    output_dir : str or None
        Directory where copykat writes its output files. Defaults to a
        temp directory that is cleaned up automatically.

    Returns
    -------
    pd.Series
        Index = query barcodes, values = copykat prediction strings.
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
            "Install it with:  pip install rpy2\n"
            "and make sure the R 'copykat' package is installed:\n"
            "  install.packages('devtools')\n"
            "  devtools::install_github('navinlabcode/copykat')"
        ) from exc

    try:
        copykat_r = importr("copykat")
    except Exception as exc:
        raise ImportError(
            "R package 'copykat' not found.\n"
            "Install it in R with:\n"
            "  devtools::install_github('navinlabcode/copykat')"
        ) from exc

    # ------------------------------------------------------------------
    # 1. Extract raw count matrices (genes × cells — copykat convention)
    # ------------------------------------------------------------------
    mat_query = _get_raw_matrix(adata_query)   # cells × genes
    mat_query = mat_query.T                    # → genes × cells

    # Reference: keep only epithelial cells that match copykat's purpose
    epithelial_terms = (
        "epithelial cell",
        "glandular epithelial cell",
        "ovarian surface epithelial cell",
    )
    if "cell_ontology_class" in adata_ref.obs.columns:
        ep_mask = adata_ref.obs["cell_ontology_class"].str.lower().isin(
            [t.lower() for t in epithelial_terms]
        )
        adata_ref_ep = adata_ref[ep_mask].copy() if ep_mask.any() else adata_ref
        logger.info(
            f"CopyKAT reference: {ep_mask.sum()} epithelial cells from "
            f"{adata_ref.n_obs} total"
        )
    else:
        adata_ref_ep = adata_ref

    mat_ref = _get_raw_matrix(adata_ref_ep)    # cells × genes
    mat_ref = mat_ref.T                        # → genes × cells

    # ------------------------------------------------------------------
    # 2. Align to common genes
    # ------------------------------------------------------------------
    query_genes = np.array(adata_query.var_names)
    ref_genes   = np.array(adata_ref_ep.var_names)
    common_genes = np.intersect1d(query_genes, ref_genes)
    logger.info(f"CopyKAT common genes: {len(common_genes)}")

    q_idx = np.where(np.isin(query_genes, common_genes))[0]
    r_idx = np.where(np.isin(ref_genes,   common_genes))[0]

    mat_query_sub = mat_query[q_idx, :]
    mat_ref_sub   = mat_ref[r_idx,   :]

    # Gene order must match between query and ref sub-matrices
    q_order = np.argsort(query_genes[q_idx])
    r_order = np.argsort(ref_genes[r_idx])
    mat_query_sub = mat_query_sub[q_order, :]
    mat_ref_sub   = mat_ref_sub[r_order,   :]
    sorted_genes  = query_genes[q_idx][q_order]

    # ------------------------------------------------------------------
    # 3. Prefix reference barcodes and combine
    # ------------------------------------------------------------------
    query_barcodes = np.array(adata_query.obs_names)

    # BUG FIX 1: use list comprehension instead of string + array
    ref_barcodes = np.array(["REF_" + bc for bc in adata_ref_ep.obs_names])

    mat_combined = np.hstack([mat_query_sub, mat_ref_sub])
    all_barcodes = np.concatenate([query_barcodes, ref_barcodes])

    # ------------------------------------------------------------------
    # 4. Convert to R matrix and run copykat
    # ------------------------------------------------------------------
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
        # BUG FIX 2: use dot-notation parameter names via **{} dict
        # because rpy2 cannot map Python underscores to R dots automatically
        result = copykat_r.copykat(
            **{
                "rawmat":          r_mat,
                "id.type":         id_type,
                "ngene.chr":       ngene_chr,
                "win.size":        win_size,
                "KS.cut":          ks_cut,
                "sam.name":        sam_name,
                "distance":        distance,
                "norm.cell.names": r_normal_cells,
                "output.seg":      "FALSE",
                "plot.genes":      "TRUE",
                "genome":          genome,
                "n.cores":         n_cores,
            }
        )
    finally:
        os.chdir(original_dir)

    # ------------------------------------------------------------------
    # 5. Extract prediction table for query cells only
    # ------------------------------------------------------------------
    pred_df = pandas2ri.rpy2py(result.rx2("prediction"))
    pred_df.columns = [c.strip() for c in pred_df.columns]
    pred_df = pred_df.set_index("cell.names")

    # Keep only query barcodes (drop REF_ cells)
    query_preds = pred_df.loc[
        pred_df.index.isin(query_barcodes), "copykat.pred"
    ]
    # Fill any missing query barcodes as 'not.defined'
    query_series = pd.Series(
        query_preds.reindex(query_barcodes).fillna("not.defined").values,
        index=query_barcodes,
        name="copykat_prediction",
    )

    logger.info(
        "CopyKAT predictions:\n"
        + query_series.value_counts().to_string()
    )
    return query_series


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_preprocessing_pipeline(
    adata,
    min_genes: int = 200,
    max_mt: float = 40.0,
    log2fc_threshold: float = 2.0,
    pval_threshold: float = 0.5,
    # --- CopyKAT parameters ---
    reference_h5ad: str = None,
    copykat_id_type: str = "S",
    copykat_ngene_chr: int = 5,
    copykat_win_size: int = 25,
    copykat_ks_cut: float = 0.1,
    copykat_distance: str = "euclidean",
    copykat_genome: str = "hg20",
    n_cores: int = 1,
    malignant_strategy: str = "union",
):
    """
    Full preprocessing pipeline: epithelial selection → QC → malignancy
    detection (scMalignantFinder + CopyKAT) → surfaceome filter → DEG →
    binarise → save.

    Parameters
    ----------
    adata : AnnData
        Full annotated object from Module 2 (PopV output).
    min_genes : int
        Minimum genes per cell (QC).
    max_mt : float
        Maximum mitochondrial % per cell (QC).
    log2fc_threshold : float
        Log2 fold-change cutoff for DEG filter.
    pval_threshold : float
        P-value cutoff for DEG filter.
    reference_h5ad : str or None
        Path to the normal reference h5ad (e.g. Tabula Sapiens ovary).
        Required for CopyKAT. If None, CopyKAT is skipped.
    copykat_id_type : str
        Gene ID type for copykat: 'S' (symbol) or 'E' (Ensembl).
    copykat_ngene_chr : int
        Minimum genes per chromosome segment (copykat ngene.chr).
    copykat_win_size : int
        Smoothing window size (copykat win.size).
    copykat_ks_cut : float
        KS statistic cutoff for aneuploid/diploid boundary (copykat KS.cut).
    copykat_distance : str
        Distance metric for copykat clustering ('euclidean' | 'pearson' | 'spearman').
    copykat_genome : str
        Genome build ('hg20' = hg38 | 'hg19' | 'mm10').
    n_cores : int
        CPU cores for CopyKAT (copykat default is 1).
    malignant_strategy : str
        How to combine scMalignantFinder and CopyKAT calls:
          'union'        — malignant if EITHER method says so (more sensitive)
          'intersection' — malignant only if BOTH methods agree (more specific)
          'scMalignant'  — use scMalignantFinder only
          'copykat'      — use CopyKAT only (requires reference_h5ad)

    Returns
    -------
    AnnData
        Preprocessed object with binary expression matrix and DEG stored
        in adata.uns['filtered_deg'].
    """

    print("\n========== STARTING PREPROCESSING ==========\n")
    initial_cells = adata.n_obs
    print(f"Initial cells: {initial_cells}")

    # ------------------------------------------------------------------
    # 1. Select epithelial cells
    # ------------------------------------------------------------------
    labels = adata.obs["popv_majority_vote_prediction"].astype(str)
    epithelial_mask = labels.str.endswith("epithelial cell")
    adata = adata[epithelial_mask].copy()
    print(f"Epithelial cells retained: {adata.n_obs}")
    print(f"Cells removed:             {initial_cells - adata.n_obs}\n")

    # ------------------------------------------------------------------
    # 2. Quality control
    # ------------------------------------------------------------------
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    print(
        f"Mean mitochondrial % BEFORE filter: "
        f"{adata.obs['pct_counts_mt'].mean():.2f}"
    )
    before_qc = adata.n_obs
    adata = adata[
        (adata.obs["n_genes_by_counts"] > min_genes) &
        (adata.obs["pct_counts_mt"] < max_mt)
    ].copy()
    print(f"Cells after QC:            {adata.n_obs}")
    print(f"Cells removed in QC:       {before_qc - adata.n_obs}")
    print(
        f"Mean mitochondrial % AFTER filter:  "
        f"{adata.obs['pct_counts_mt'].mean():.2f}\n"
    )

    # ------------------------------------------------------------------
    # 3. Route raw counts into .X, then normalise for classifiers
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
            print(
                "No dedicated raw layer found. "
                "Assuming adata.X already contains raw counts."
            )

    adata.var_names_make_unique()

    # Store a clean copy of raw counts for CopyKAT BEFORE log-normalisation
    adata.layers["raw_for_copykat"] = adata.X.copy()

    # Normalise for scMalignantFinder and DEG
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ------------------------------------------------------------------
    # 4. scMalignantFinder
    # ------------------------------------------------------------------
    print("Running scMalignantFinder …")
    from scMalignantFinder import classifier

    model = classifier.scMalignantFinder(
        test_input=adata,
        celltype_annotation=False,
        pretrain_path=SCMALIGNANT_MODEL,
        feature_path=os.path.join(SCMALIGNANT_MODEL, "ordered_feature.tsv"),
    )
    model.load()
    result_scm = model.predict()
    adata.obs["scMalignantFinder_prediction"] = result_scm.obs[
        "scMalignantFinder_prediction"
    ]
    print("scMalignantFinder completed.")
    print(adata.obs["scMalignantFinder_prediction"].value_counts().to_string(), "\n")

    # ------------------------------------------------------------------
    # 5. CopyKAT (CNV-based, requires a normal reference)
    # ------------------------------------------------------------------
    copykat_available = False

    if malignant_strategy in ("copykat", "union", "intersection"):
        if reference_h5ad is None:
            print(
                "⚠ CopyKAT skipped: no reference_h5ad provided.\n"
                "  Pass reference_h5ad='/path/to/normal_ref.h5ad' to enable it.\n"
                "  malignant_strategy falls back to 'scMalignant'."
            )
            malignant_strategy = "scMalignant"
        else:
            print(f"Running CopyKAT with reference: {reference_h5ad} …")
            try:
                # Restore raw counts into a temporary AnnData for CopyKAT
                adata_raw = adata.copy()
                adata_raw.X = adata.layers["raw_for_copykat"]

                adata_ref_full = sc.read_h5ad(reference_h5ad)

                copykat_preds = _run_copykat(
                    adata_query  = adata_raw,
                    adata_ref    = adata_ref_full,
                    id_type      = copykat_id_type,
                    ngene_chr    = copykat_ngene_chr,
                    win_size     = copykat_win_size,
                    ks_cut       = copykat_ks_cut,
                    distance     = copykat_distance,
                    genome       = copykat_genome,
                    n_cores      = n_cores,
                )
                adata.obs["copykat_prediction"] = copykat_preds.values
                copykat_available = True
                print("CopyKAT completed.")
                print(adata.obs["copykat_prediction"].value_counts().to_string(), "\n")

            except Exception as exc:
                print(
                    f"⚠ CopyKAT failed: {type(exc).__name__}: {exc}\n"
                    f"  malignant_strategy falls back to 'scMalignant'."
                )
                logger.exception("CopyKAT error details:")
                malignant_strategy = "scMalignant"

    # ------------------------------------------------------------------
    # 6. Combine malignancy calls
    # ------------------------------------------------------------------
    # scMalignantFinder convention: 'malignant' | 'normal'
    # CopyKAT convention:           'aneuploid' | 'diploid' | 'not.defined'

    scm_mal = adata.obs["scMalignantFinder_prediction"].str.lower() == "malignant"

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
    print(f"  Malignant cells: {malignant_mask.sum()}")
    print(f"  Normal cells:    {(~malignant_mask).sum()}\n")

    # ------------------------------------------------------------------
    # 7. Surfaceome filter
    # ------------------------------------------------------------------
    surfaceome = pd.read_csv(SURFACEOME_PATH)
    surfaceome.columns = surfaceome.columns.str.strip()
    surf_genes = surfaceome["Gene"].astype(str).tolist()
    common = adata.var_names.intersection(surf_genes)
    adata = adata[:, common].copy()
    print(f"Surfaceome genes retained: {adata.n_vars}\n")

    # ------------------------------------------------------------------
    # 8. Differential expression (malignant vs normal)
    # ------------------------------------------------------------------
    sc.tl.rank_genes_groups(
        adata,
        groupby="final_malignant",
        method="wilcoxon",
    )
    result_deg = sc.get.rank_genes_groups_df(adata, group=None)
    filtered_deg = result_deg[
        (result_deg["logfoldchanges"] > log2fc_threshold) &
        (result_deg["pvals"] < pval_threshold)
    ]
    adata.uns["filtered_deg"] = filtered_deg
    print(f"Final DE genes retained: {filtered_deg.shape[0]}\n")

    # ------------------------------------------------------------------
    # 9. Binarise from raw counts (not log-normalised X)
    # ------------------------------------------------------------------
    # BUG FIX 3: binarise from raw_for_copykat layer, not from log-normalised X.
    # After surfaceome filtering the layer is subsetted automatically with adata,
    # so we can safely read it here.
    adata.X = (adata.layers["raw_for_copykat"] > 0).astype(int)
    print("Expression converted to binary (0/1).\n")

    # ------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------
    final_path = os.path.join(SAVE_DIR, "final_tumor.h5ad")

    # Clean object columns before writing
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object:
            adata.obs[col] = adata.obs[col].astype(str)

    adata.write(final_path)
    print(f"Final object saved to:\n{final_path}")
    print("\n========== PREPROCESSING COMPLETED ==========\n")

    return adata
