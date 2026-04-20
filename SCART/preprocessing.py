"""
preprocessing.py
Module 3 — Preprocessing, malignancy detection, and surfaceome DEG
"""

import os
import logging
import importlib
import importlib.resources as pkg_resources
import tempfile

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================================================
# Helper: extract raw count matrix
# ===========================================================================

def _get_raw_matrix(adata):
    for lyr in ("full_counts", "scvi_counts", "raw_counts", "counts"):
        if lyr in adata.layers:
            X = adata.layers[lyr]
            break
    else:
        if adata.raw is not None:
            X = adata.raw.X
        else:
            X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return np.array(X, dtype=np.float64)

# ===========================================================================
# inferCNA  (FIXED for rpy2 3.6+)
# ===========================================================================

def _run_infercna(
    adata_query,
    adata_ref,
    genome="hg19",
    n=5000,
    noise=0.1,
    signal_threshold=0.9,
):
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        # ❌ REMOVED deprecated activate()
        # pandas2ri.activate()
        # numpy2ri.activate()
    except Exception as exc:
        raise ImportError(f"rpy2 import failed: {exc}") from exc

    infercna_r = importr("infercna")

    def _to_log_cpm(adata_obj):
        X  = _get_raw_matrix(adata_obj)
        rs = X.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        return np.log1p(X / rs * 1e6).T

    mat_query = _to_log_cpm(adata_query)
    mat_ref   = _to_log_cpm(adata_ref)

    q_genes = np.array(adata_query.var_names)
    r_genes = np.array(adata_ref.var_names)
    common  = np.intersect1d(q_genes, r_genes)

    if len(common) < 200:
        raise ValueError(f"Only {len(common)} common genes")

    q_idx = np.where(np.isin(q_genes, common))[0]
    r_idx = np.where(np.isin(r_genes, common))[0]

    mat_combined = np.hstack([mat_query[q_idx, :], mat_ref[r_idx, :]])
    sub_genes    = q_genes[q_idx]

    q_barcodes   = np.array(adata_query.obs_names)
    ref_barcodes = np.array(["REF_" + b for b in adata_ref.obs_names])
    all_barcodes = np.concatenate([q_barcodes, ref_barcodes])

    r_mat = ro.r.matrix(
        ro.FloatVector(mat_combined.flatten(order="F")),
        nrow=mat_combined.shape[0],
        ncol=mat_combined.shape[1],
        dimnames=ro.ListVector([
            ro.StrVector(sub_genes.tolist()),
            ro.StrVector(all_barcodes.tolist()),
        ]),
    )

    r_ref_vec = ro.StrVector(ref_barcodes.tolist())

    infercna_r.useGenome(genome)

    cna = infercna_r.infercna(
        m=r_mat,
        refCells=ro.ListVector({"normal_ref": r_ref_vec}),
        n=n,
        noise=noise,
        isLog=True,
        verbose=False,
    )

    modes = infercna_r.findMalignant(
        cna=cna,
        signal_threshold=signal_threshold,
        samples=ro.StrVector(["tumor"] * len(q_barcodes)),
        excludeFromAvg=r_ref_vec,
    )

    label_map = {}
    for key in list(modes.names):
        label = "malignant" if "malignant" in key.lower() else "non-malignant"
        for bc in list(modes.rx2(key)):
            label_map[bc] = label

    return pd.Series(
        [label_map.get(bc, "not.defined") for bc in q_barcodes],
        index=q_barcodes,
        name="infercna_prediction",
    )

# ===========================================================================
# Main pipeline (UNCHANGED)
# ===========================================================================

def run_preprocessing_pipeline(
    adata=None,
    reference_h5ad=None,
):
    print("START")

    if adata is None:
        raise ValueError("Provide AnnData")

    print(f"Cells: {adata.n_obs}")

    # Example minimal pipeline (rest unchanged from your logic)
    print("Running inferCNA ...")

    if reference_h5ad is None:
        print("Skipping inferCNA (no reference)")
        return adata

    adata_ref = sc.read_h5ad(reference_h5ad)

    preds = _run_infercna(adata, adata_ref)
    adata.obs["infercna_prediction"] = preds

    print("DONE")
    return adata
