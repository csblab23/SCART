"""
preprocessing.py
"""

import os
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import urllib.request
import zipfile
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SURFACEOME_PATH = "/lustre/anas.a/Vinaya/scT-CAR_Designer/GESP/GESP_surfaceome_gene.csv"

SAVE_DIR = "./preprocessed_output"
os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================================================
# NEW: Auto-download scMalignantFinder model
# ==========================================================

def _ensure_scmalignant_model():
    """
    Download scMalignantFinder model from GitHub if not present.
    """
    model_dir = os.path.expanduser("~/.scart/scmalignant_model")

    required_file = os.path.join(model_dir, "ordered_feature.tsv")

    if os.path.exists(required_file):
        return model_dir

    os.makedirs(model_dir, exist_ok=True)

    logger.info("Downloading scMalignantFinder model from GitHub...")

    # GitHub repo zip
    url = "https://github.com/Jonyyqn/scMalignantFinder/archive/refs/heads/main.zip"

    tmp_zip = os.path.join(tempfile.gettempdir(), "scmalignant_model.zip")

    urllib.request.urlretrieve(url, tmp_zip)

    with zipfile.ZipFile(tmp_zip, 'r') as zip_ref:
        zip_ref.extractall(tempfile.gettempdir())

    extracted_path = os.path.join(
        tempfile.gettempdir(),
        "scMalignantFinder-main",
        "model"
    )

    # Move model files
    for f in os.listdir(extracted_path):
        src = os.path.join(extracted_path, f)
        dst = os.path.join(model_dir, f)
        if not os.path.exists(dst):
            os.rename(src, dst)

    logger.info(f"Model downloaded to: {model_dir}")

    return model_dir


# ==========================================================
# CopyKAT helper
# ==========================================================

def _run_copykat(adata_raw_counts, n_cores=4, sam_name="copykat_run"):

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
    except ImportError as e:
        raise ImportError(
            "rpy2 is required to run CopyKAT.\n"
            "pip install rpy2\n"
            "R -e \"devtools::install_github('navinlabcode/copykat')\""
        ) from e

    X = adata_raw_counts.X
    if sparse.issparse(X):
        X = X.toarray()

    raw_df = pd.DataFrame(
        X.T,
        index=adata_raw_counts.var_names,
        columns=adata_raw_counts.obs_names,
    )

    r_copykat = importr("copykat")
    r_base = importr("base")

    r_mat = pandas2ri.py2rpy(raw_df)

    logger.info("Running CopyKAT...")
    copykat_result = r_copykat.copykat(
        rawmat=r_mat,
        id_type="S",
        ngene_chr=ro.IntVector([5]),
        win_size=ro.IntVector([25]),
        KS_cut=ro.FloatVector([0.1]),
        sam_name=sam_name,
        distance="euclidean",
        genome="hg38",   # FIXED
        n_cores=ro.IntVector([n_cores]),
    )

    pred_r = r_base.as_data_frame(copykat_result.rx2("prediction"))
    pred_df = pandas2ri.rpy2py(pred_r)

    pred_series = pred_df.set_index("cell.names")["copykat.pred"]
    pred_series = pred_series.reindex(
        adata_raw_counts.obs_names, fill_value="not.defined"
    )

    return pred_series


# ==========================================================
# Main Pipeline (UNCHANGED LOGIC)
# ==========================================================

def run_preprocessing_pipeline(
    adata,
    min_genes=200,
    max_mt=40,
    log2fc_threshold=2,
    pval_threshold=0.5,
):

    # 🔥 NEW: ensure model exists
    SCMALIGNANT_MODEL = _ensure_scmalignant_model()

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

    if "scvi_counts" in adata.layers:
        adata.X = adata.layers["scvi_counts"].copy()
    elif "raw_counts" in adata.layers:
        adata.X = adata.layers["raw_counts"].copy()
    elif "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    elif adata.raw is not None:
        adata.X = adata.raw.X.copy()

    adata.var_names_make_unique()

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
    result = model.predict()

    adata.obs["scMalignantFinder_prediction"] = \
        result.obs["scMalignantFinder_prediction"]

    print("scMalignantFinder completed.\n")

    # CopyKAT
    adata_for_copykat = adata.copy()
    copykat_pred = _run_copykat(adata_for_copykat)

    adata.obs["copykat_prediction"] = copykat_pred.values

    scmal_is_malignant = adata.obs["scMalignantFinder_prediction"] == "malignant"
    copykat_is_aneuploid = adata.obs["copykat_prediction"] == "aneuploid"

    adata.obs["consensus_malignant"] = (
        scmal_is_malignant & copykat_is_aneuploid
    )

    adata = adata[adata.obs["consensus_malignant"]].copy()

    surfaceome = pd.read_csv(SURFACEOME_PATH)
    genes = surfaceome["Gene"].astype(str).tolist()

    common = adata.var_names.intersection(genes)
    adata = adata[:, common].copy()

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

    adata.X = (adata.X > 0).astype(int)

    final_path = os.path.join(SAVE_DIR, "final_tumor.h5ad")
    adata.write(final_path)

    print("\n========== PREPROCESSING COMPLETED ==========\n")

    return adata
