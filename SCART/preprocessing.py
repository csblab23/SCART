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
# Paths — repo-relative
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(BASE_DIR, "external"))

SURFACEOME_PATH = os.path.join(
    BASE_DIR, "GESP", "GESP_surfaceome_gene.csv"
)

SCMALIGNANT_MODEL = os.path.join(
    BASE_DIR, "external", "scMalignantFinder", "model"
)

SAVE_DIR = os.path.join(BASE_DIR, "preprocessed_input")
os.makedirs(SAVE_DIR, exist_ok=True)


# ===========================================================================
# Helper
# ===========================================================================

def _get_raw_matrix(adata):
    for layer in ("scvi_counts", "raw_counts", "counts"):
        if layer in adata.layers:
            X = adata.layers[layer]
            break
    else:
        X = adata.raw.X if adata.raw is not None else adata.X

    if sp.issparse(X):
        X = X.toarray()
    return np.array(X, dtype=np.float64)


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_preprocessing_pipeline(
    adata=None,   # 🔥 changed
    min_genes: int = 200,
    max_mt: float = 40.0,
    log2fc_threshold: float = 2.0,
    pval_threshold: float = 0.5,
    reference_h5ad: str = None,
    n_cores: int = 1,
    malignant_strategy: str = "union",
):

    print("\n========== STARTING PREPROCESSING ==========\n")

    # 🔥 AUTO-LOAD POPV OUTPUT
    if adata is None:
        popv_path = os.path.join(
            BASE_DIR, "..", "popv_results", "final_popv_annotated.h5ad"
        )
        popv_path = os.path.abspath(popv_path)

        if not os.path.exists(popv_path):
            raise FileNotFoundError(
                f"POPV output not found at:\n{popv_path}"
            )

        print(f"Auto-loading POPV output:\n{popv_path}\n")
        adata = sc.read_h5ad(popv_path)

    initial_cells = adata.n_obs

    # ------------------------------------------------------------------
    # 1. Epithelial
    # ------------------------------------------------------------------
    labels = adata.obs["popv_majority_vote_prediction"].astype(str)
    adata = adata[labels.str.endswith("epithelial cell")].copy()

    # ------------------------------------------------------------------
    # 2. QC
    # ------------------------------------------------------------------
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    adata = adata[
        (adata.obs["n_genes_by_counts"] > min_genes) &
        (adata.obs["pct_counts_mt"] < max_mt)
    ].copy()

    # ------------------------------------------------------------------
    # 3. Raw counts
    # ------------------------------------------------------------------
    for layer in ("scvi_counts", "raw_counts", "counts"):
        if layer in adata.layers:
            adata.X = adata.layers[layer].copy()
            break

    adata.layers["raw_for_copykat"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ------------------------------------------------------------------
    # 4. scMalignantFinder
    # ------------------------------------------------------------------
    from scMalignantFinder import classifier

    model = classifier.scMalignantFinder(
        test_input=adata,
        celltype_annotation=False,
        pretrain_path=SCMALIGNANT_MODEL,
        feature_path=os.path.join(SCMALIGNANT_MODEL, "ordered_feature.tsv"),
    )
    model.load()
    result = model.predict()

    adata.obs["scMalignantFinder_prediction"] = result.obs[
        "scMalignantFinder_prediction"
    ]

    # ------------------------------------------------------------------
    # 5. Surfaceome
    # ------------------------------------------------------------------
    surfaceome = pd.read_csv(SURFACEOME_PATH)
    surf_genes = surfaceome["Gene"].astype(str).tolist()
    adata = adata[:, adata.var_names.intersection(surf_genes)].copy()

    # ------------------------------------------------------------------
    # 6. DEG
    # ------------------------------------------------------------------
    sc.tl.rank_genes_groups(
        adata,
        groupby="scMalignantFinder_prediction",
        method="wilcoxon",
    )

    df = sc.get.rank_genes_groups_df(adata, group=None)

    adata.uns["filtered_deg"] = df[
        (df["logfoldchanges"] > log2fc_threshold) &
        (df["pvals"] < pval_threshold)
    ]

    # ------------------------------------------------------------------
    # 7. Binarise
    # ------------------------------------------------------------------
    adata.X = (adata.layers["raw_for_copykat"] > 0).astype(int)

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    out = os.path.join(SAVE_DIR, "final_tumor.h5ad")
    adata.write(out)

    print(f"\nSaved: {out}")
    print("\n========== DONE ==========\n")

    return adata
