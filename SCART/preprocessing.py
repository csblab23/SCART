"""
preprocessing.py
Module 3 — Preprocessing, malignancy detection, and surfaceome DEG
"""

import os
import sys
import logging
import tempfile

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — UPDATED TO REPO-RELATIVE (GitHub-safe)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# allow import from SCART/external
sys.path.append(os.path.join(BASE_DIR, "external"))

SURFACEOME_PATH = os.path.join(
    BASE_DIR,
    "GESP",
    "GESP_surfaceome_gene.csv"
)

SCMALIGNANT_MODEL = os.path.join(
    BASE_DIR,
    "external",
    "scMalignantFinder",
    "model"
)

SAVE_DIR = os.path.join(
    BASE_DIR,
    "preprocessed_input"
)

os.makedirs(SAVE_DIR, exist_ok=True)


# ===========================================================================
# Helper: extract raw count matrix from an AnnData
# ===========================================================================

def _get_raw_matrix(adata):
    """
    Return a dense numpy array of raw integer counts (cells × genes).
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
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
        numpy2ri.activate()
    except ImportError as exc:
        raise ImportError("rpy2 is required") from exc

    try:
        copykat_r = importr("copykat")
    except Exception as exc:
        raise ImportError("copykat R package not found") from exc

    mat_query = _get_raw_matrix(adata_query).T

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
    else:
        adata_ref_ep = adata_ref

    mat_ref = _get_raw_matrix(adata_ref_ep).T

    query_genes = np.array(adata_query.var_names)
    ref_genes   = np.array(adata_ref_ep.var_names)
    common_genes = np.intersect1d(query_genes, ref_genes)

    q_idx = np.where(np.isin(query_genes, common_genes))[0]
    r_idx = np.where(np.isin(ref_genes,   common_genes))[0]

    mat_query_sub = mat_query[q_idx, :]
    mat_ref_sub   = mat_ref[r_idx,   :]

    q_order = np.argsort(query_genes[q_idx])
    r_order = np.argsort(ref_genes[r_idx])

    mat_query_sub = mat_query_sub[q_order, :]
    mat_ref_sub   = mat_ref_sub[r_order,   :]

    sorted_genes  = query_genes[q_idx][q_order]

    query_barcodes = np.array(adata_query.obs_names)
    ref_barcodes = np.array(["REF_" + bc for bc in adata_ref_ep.obs_names])

    mat_combined = np.hstack([mat_query_sub, mat_ref_sub])
    all_barcodes = np.concatenate([query_barcodes, ref_barcodes])

    r_mat = ro.r.matrix(
        ro.FloatVector(mat_combined.flatten(order="F")),
        nrow=mat_combined.shape[0],
        ncol=mat_combined.shape[1],
    )

    r_normal_cells = ro.StrVector(ref_barcodes.tolist())

    use_dir = output_dir or tempfile.mkdtemp(prefix="copykat_")
    original_dir = os.getcwd()
    os.chdir(use_dir)

    try:
        result = copykat_r.copykat(
            rawmat=r_mat,
            id_type=id_type,
            sam_name=sam_name,
            distance=distance,
            genome=genome,
            n_cores=n_cores,
        )
    finally:
        os.chdir(original_dir)

    pred_df = pandas2ri.rpy2py(result.rx2("prediction"))
    pred_df = pred_df.set_index("cell.names")

    query_preds = pred_df.loc[
        pred_df.index.isin(query_barcodes), "copykat.pred"
    ]

    return pd.Series(
        query_preds.reindex(query_barcodes).fillna("not.defined").values,
        index=query_barcodes,
    )


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_preprocessing_pipeline(
    adata,
    min_genes: int = 200,
    max_mt: float = 40.0,
    log2fc_threshold: float = 2.0,
    pval_threshold: float = 0.5,
    reference_h5ad: str = None,
    malignant_strategy: str = "union",
):

    print("\n========== STARTING PREPROCESSING ==========\n")

    labels = adata.obs["popv_majority_vote_prediction"].astype(str)
    epithelial_mask = labels.str.endswith("epithelial cell")
    adata = adata[epithelial_mask].copy()

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    adata = adata[
        (adata.obs["n_genes_by_counts"] > min_genes) &
        (adata.obs["pct_counts_mt"] < max_mt)
    ].copy()

    for layer in ("scvi_counts", "raw_counts", "counts"):
        if layer in adata.layers:
            adata.X = adata.layers[layer].copy()
            break

    adata.layers["raw_for_copykat"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # scMalignantFinder
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

    # Surfaceome
    surfaceome = pd.read_csv(SURFACEOME_PATH)
    surf_genes = surfaceome["Gene"].astype(str).tolist()
    common = adata.var_names.intersection(surf_genes)
    adata = adata[:, common].copy()

    sc.tl.rank_genes_groups(
        adata,
        groupby="scMalignantFinder_prediction",
        method="wilcoxon",
    )

    result_deg = sc.get.rank_genes_groups_df(adata, group=None)

    filtered_deg = result_deg[
        (result_deg["logfoldchanges"] > log2fc_threshold) &
        (result_deg["pvals"] < pval_threshold)
    ]

    adata.uns["filtered_deg"] = filtered_deg

    adata.X = (adata.layers["raw_for_copykat"] > 0).astype(int)

    final_path = os.path.join(SAVE_DIR, "final_tumor.h5ad")
    adata.write(final_path)

    print(f"Saved to: {final_path}")
    print("\n========== PREPROCESSING DONE ==========\n")

    return adata
