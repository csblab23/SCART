"""
preprocessing.py
"""

import os
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SURFACEOME_PATH = "/lustre/anas.a/Vinaya/scT-CAR_Designer/GESP/GESP_surfaceome_gene.csv"
SCMALIGNANT_MODEL = "/lustre/anas.a/scMalignantFinder/model/"

SAVE_DIR = "/lustre/anas.a/Vinaya/scT-CAR_Designer/preprocessed_input"
os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================================================
# Main Pipeline
# ==========================================================

def run_preprocessing_pipeline(
    adata,
    min_genes=200,
    max_mt=40,
    log2fc_threshold=2,
    pval_threshold=0.5,
):

    print("\n========== STARTING PREPROCESSING ==========\n")
    initial_cells = adata.n_obs
    print(f"Initial cells: {initial_cells}")

    # --------------------------------------------------
    # 1️⃣ Select epithelial cells
    # --------------------------------------------------

    labels = adata.obs["popv_majority_vote_prediction"].astype(str)
    epithelial_mask = labels.str.endswith("epithelial cell")
    adata = adata[epithelial_mask].copy()

    print(f"Epithelial cells retained: {adata.n_obs}")
    print(f"Cells removed: {initial_cells - adata.n_obs}\n")

    # --------------------------------------------------
    # 2️⃣ Quality Control
    # --------------------------------------------------

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    print(f"Mean mitochondrial % BEFORE filter: {adata.obs['pct_counts_mt'].mean():.2f}")

    before_qc = adata.n_obs

    adata = adata[
        (adata.obs["n_genes_by_counts"] > min_genes) &
        (adata.obs["pct_counts_mt"] < max_mt)
    ].copy()

    print(f"Cells after QC: {adata.n_obs}")
    print(f"Cells removed in QC: {before_qc - adata.n_obs}")
    print(f"Mean mitochondrial % AFTER filter: {adata.obs['pct_counts_mt'].mean():.2f}\n")

    # --------------------------------------------------
    # 3️⃣ Use raw counts + standard normalization
    # --------------------------------------------------

    print("Detecting raw count source...")

    if "scvi_counts" in adata.layers:
        print("Using adata.layers['scvi_counts'] as raw counts.")
        adata.X = adata.layers["scvi_counts"].copy()

    elif "raw_counts" in adata.layers:
        print("Using adata.layers['raw_counts'] as raw counts.")
        adata.X = adata.layers["raw_counts"].copy()

    elif "counts" in adata.layers:
        print("Using adata.layers['counts'] as raw counts.")
        adata.X = adata.layers["counts"].copy()

    elif adata.raw is not None:
        print("Using adata.raw.X as raw counts.")
        adata.X = adata.raw.X.copy()

    else:
        print("No dedicated raw layer found. Assuming adata.X already contains raw counts.")

    adata.var_names_make_unique()

    # ✅ Required for rank_genes_groups
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # --------------------------------------------------
    # 4️⃣ scMalignantFinder (UNCHANGED)
    # --------------------------------------------------

    from scMalignantFinder import classifier

    model = classifier.scMalignantFinder(
        test_input=adata,
        celltype_annotation=False,
        pretrain_path=SCMALIGNANT_MODEL,
        feature_path=os.path.join(SCMALIGNANT_MODEL, "ordered_feature.tsv"),
    )

    model.load()
    result = model.predict()

    adata.obs["scMalignantFinder_prediction"] = \
        result.obs["scMalignantFinder_prediction"]

    print("scMalignantFinder completed.\n")

    # --------------------------------------------------
    # 5️⃣ Surfaceome filter (UNCHANGED)
    # --------------------------------------------------

    surfaceome = pd.read_csv(SURFACEOME_PATH)
    surfaceome.columns = surfaceome.columns.str.strip()
    genes = surfaceome["Gene"].astype(str).tolist()

    common = adata.var_names.intersection(genes)
    adata = adata[:, common].copy()

    print(f"Surfaceome genes retained: {adata.n_vars}\n")

    # --------------------------------------------------
    # 6️⃣ DEG (UNCHANGED)
    # --------------------------------------------------

    sc.tl.rank_genes_groups(
        adata,
        groupby="scMalignantFinder_prediction",
        method="wilcoxon"
    )

    result = sc.get.rank_genes_groups_df(adata, group=None)

    filtered = result[
        (result["logfoldchanges"] > log2fc_threshold) &
        (result["pvals"] < pval_threshold)
    ]

    adata.uns["filtered_deg"] = filtered

    print(f"Final DE genes retained: {filtered.shape[0]}\n")

    # --------------------------------------------------
    # 7️⃣ Binarize (UNCHANGED)
    # --------------------------------------------------

    adata.X = (adata.X > 0).astype(int)
    print("Expression converted to binary (0/1).\n")

    # --------------------------------------------------
    # 8️⃣ Save final object
    # --------------------------------------------------

    final_path = os.path.join(SAVE_DIR, "final_tumor.h5ad")
    adata.write(final_path)

    print(f"Final object saved to:\n{final_path}")
    print("\n========== PREPROCESSING COMPLETED ==========\n")

    return adata