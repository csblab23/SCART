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
# CopyKAT helper  (NEW)
# ==========================================================

def _run_copykat(adata_raw_counts, n_cores=4, sam_name="copykat_run"):
    """
    Run CopyKAT (R package) on a raw-count AnnData via rpy2.

    Parameters
    ----------
    adata_raw_counts : AnnData
        AnnData whose .X contains raw integer UMI counts
        (cells × genes, BEFORE normalisation/log1p).
    n_cores : int
        Number of parallel cores to pass to copykat().
    sam_name : str
        Sample name prefix for copykat output files.

    Returns
    -------
    pd.Series
        Index = cell barcodes, values = "aneuploid" | "diploid" | "not.defined".
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
    except ImportError as e:
        raise ImportError(
            "rpy2 is required to run CopyKAT. "
            "Install it with:  pip install rpy2\n"
            "Also ensure the R copykat package is installed:\n"
            "  R -e \"devtools::install_github('navinlabcode/copykat')\""
        ) from e

    # ── Build genes-in-rows, cells-in-columns raw matrix ──────────────────
    X = adata_raw_counts.X
    if sparse.issparse(X):
        X = X.toarray()

    raw_df = pd.DataFrame(
        X.T,                                    # genes × cells
        index=adata_raw_counts.var_names,
        columns=adata_raw_counts.obs_names,
    )

    # ── Transfer to R ──────────────────────────────────────────────────────
    r_copykat = importr("copykat")
    r_base    = importr("base")

    r_mat = pandas2ri.py2rpy(raw_df)

    # ── Run copykat ────────────────────────────────────────────────────────
    logger.info("Running CopyKAT (this may take several minutes)…")
    copykat_result = r_copykat.copykat(
        rawmat      = r_mat,
        id_type     = "S",          # gene symbols
        ngene_chr   = ro.IntVector([5]),
        win_size    = ro.IntVector([25]),
        KS_cut      = ro.FloatVector([0.1]),
        sam_name    = sam_name,
        distance    = "euclidean",
        genome      = "hg20",
        n_cores     = ro.IntVector([n_cores]),
    )

    # ── Extract prediction table ───────────────────────────────────────────
    pred_r  = r_base.as_data_frame(copykat_result.rx2("prediction"))
    pred_df = pandas2ri.rpy2py(pred_r)         # columns: cell.names, copykat.pred

    # Keep only defined predictions; fill missing cells as "not.defined"
    pred_series = pred_df.set_index("cell.names")["copykat.pred"]
    pred_series = pred_series.reindex(adata_raw_counts.obs_names, fill_value="not.defined")

    logger.info(
        "CopyKAT done. aneuploid=%d  diploid=%d  not.defined=%d",
        (pred_series == "aneuploid").sum(),
        (pred_series == "diploid").sum(),
        (pred_series == "not.defined").sum(),
    )
    return pred_series


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
    # 4b️⃣ CopyKAT — run on raw counts, combine with scMalignantFinder  (NEW)
    # --------------------------------------------------

    # CopyKAT needs raw (un-normalised) counts → use the layer that was
    # detected in step 3, stored before normalisation/log1p was applied.
    print("Running CopyKAT for CNV-based malignancy inference…")

    # Recover raw counts: prefer a stored layer, fall back to .raw
    if "scvi_counts" in adata.layers:
        adata_for_copykat = adata.copy()
        adata_for_copykat.X = adata.layers["scvi_counts"].copy()
    elif "raw_counts" in adata.layers:
        adata_for_copykat = adata.copy()
        adata_for_copykat.X = adata.layers["raw_counts"].copy()
    elif "counts" in adata.layers:
        adata_for_copykat = adata.copy()
        adata_for_copykat.X = adata.layers["counts"].copy()
    elif adata.raw is not None:
        adata_for_copykat = adata.copy()
        adata_for_copykat.X = adata.raw.X.copy()
    else:
        # adata.X has already been normalised/log1p-ed at this point;
        # warn the user but proceed — copykat will still run, just less ideal
        logger.warning(
            "No raw count layer found for CopyKAT. "
            "CopyKAT will use the normalised matrix, which may reduce accuracy."
        )
        adata_for_copykat = adata.copy()

    copykat_pred = _run_copykat(adata_for_copykat, n_cores=4, sam_name="tumor_copykat")

    # Store individual CopyKAT labels on the main object
    adata.obs["copykat_prediction"] = copykat_pred.values

    # ── Consensus: malignant only when BOTH tools agree ───────────────────
    scmal_is_malignant  = adata.obs["scMalignantFinder_prediction"] == "malignant"
    copykat_is_aneuploid = adata.obs["copykat_prediction"] == "aneuploid"

    adata.obs["consensus_malignant"] = (scmal_is_malignant & copykat_is_aneuploid)

    n_consensus = adata.obs["consensus_malignant"].sum()
    print(f"scMalignantFinder malignant calls : {scmal_is_malignant.sum()}")
    print(f"CopyKAT aneuploid calls           : {copykat_is_aneuploid.sum()}")
    print(f"Consensus malignant (both agree)  : {n_consensus}\n")

    # Keep only consensus-malignant cells for downstream steps
    before_consensus = adata.n_obs
    adata = adata[adata.obs["consensus_malignant"]].copy()
    print(f"Cells retained after consensus filter: {adata.n_obs}")
    print(f"Cells removed by consensus filter    : {before_consensus - adata.n_obs}\n")

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
