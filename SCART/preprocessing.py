"""
preprocessing.py
Module 3 — Preprocessing + scMalignantFinder + CopyKAT
"""

import os
import sys
import logging
import subprocess
import tempfile
import glob

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "external"))

SURFACEOME_PATH = os.path.join(BASE_DIR, "GESP", "GESP_surfaceome_gene.csv")


# =========================================================
# ✅ PERMANENT FIX (ROBUST MODEL PATH)
# =========================================================
SCMALIGNANT_MODEL = os.path.join(
    os.path.dirname(__file__),
    "external",
    "scMalignantFinder",
    "model"
)

# hard fail early if broken install
assert os.path.exists(os.path.join(SCMALIGNANT_MODEL, "model.joblib")), \
    "scMalignantFinder model missing at expected path"

SAVE_DIR = os.path.join(BASE_DIR, "preprocessed_input")
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# Helpers
# =========================================================
def _to_dense(X):
    return X.toarray() if sp.issparse(X) else X


def _auto_find_popv():
    search_paths = [os.getcwd(), "popv_results"]

    files = []
    for path in search_paths:
        files.extend(glob.glob(os.path.join(path, "final_popv_annotated.h5ad")))

    if not files:
        raise FileNotFoundError(
            "No POPV output found.\n"
            "Expected: popv_results/final_popv_annotated.h5ad\n"
            "Run Module 2 first OR pass popv_path"
        )

    files = list(set(files))
    return max(files, key=os.path.getctime)


def run_copykat(adata, ref_path, n_cores=4, copykat_params=None):

    print("\nRunning CopyKAT...\n")

    default_params = {
        "id_type": "S",
        "ngene_chr": 5,
        "win_size": 25,
        "KS_cut": 0.1,
        "sam_name": "sample",
        "distance": "euclidean",
        "genome": "hg20"
    }

    if copykat_params is not None:
        default_params.update(copykat_params)

    ref = sc.read_h5ad(ref_path)

    mat_main = _to_dense(adata.layers.get("counts", adata.X)).T
    mat_ref = _to_dense(ref.layers.get("counts", ref.X)).T

    genes_main = adata.var_names
    genes_ref = ref.var_names

    common = np.intersect1d(genes_main, genes_ref)

    mat_main = mat_main[[genes_main.get_loc(g) for g in common], :]
    mat_ref = mat_ref[[genes_ref.get_loc(g) for g in common], :]

    ref_cells = ["REF_" + c for c in ref.obs_names]

    combined = np.concatenate([mat_main, mat_ref], axis=1)
    cell_names = list(adata.obs_names) + ref_cells

    tmp_dir = tempfile.mkdtemp()

    mat_file = os.path.join(tmp_dir, "matrix.csv")
    pd.DataFrame(combined, index=common, columns=cell_names).to_csv(mat_file)

    r_script = os.path.join(tmp_dir, "run_copykat.R")

    with open(r_script, "w") as f:
        f.write(f"""
library(copykat)

data <- read.csv("{mat_file}", row.names=1)

res <- copykat(
  rawmat = data,
  id.type = "{default_params['id_type']}",
  ngene.chr = {default_params['ngene_chr']},
  win.size = {default_params['win_size']},
  KS.cut = {default_params['KS_cut']},
  sam.name = "{default_params['sam_name']}",
  distance = "{default_params['distance']}",
  norm.cell.names = colnames(data)[grep("^REF_", colnames(data))],
  output.seg = "FALSE",
  plot.genes = "TRUE",
  genome = "{default_params['genome']}",
  n.cores = {n_cores}
)

write.csv(res$prediction, file="{tmp_dir}/copykat_pred.csv")
""")

    subprocess.run(["Rscript", r_script], check=True)

    pred = pd.read_csv(os.path.join(tmp_dir, "copykat_pred.csv"), index_col=0)

    return pred


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_preprocessing_pipeline(
    adata=None,
    popv_path=None,
    min_genes=200,
    max_mt=40,
    log2fc_threshold=2,
    pval_threshold=0.5,
    reference_h5ad=None,
    n_cores=4,
    malignant_strategy="union",
    copykat_params=None,
):

    print("\n========== START ==========\n")

    if adata is None:

        if popv_path is not None:
            print(f"Loading POPV output (user): {popv_path}")
        else:
            popv_path = _auto_find_popv()
            print(f"Loading POPV output (auto): {popv_path}")

        adata = sc.read_h5ad(popv_path)

    labels = adata.obs["popv_majority_vote_prediction"].astype(str)
    adata = adata[labels.str.contains("epithelial", case=False)].copy()

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    adata = adata[
        (adata.obs["n_genes_by_counts"] > min_genes) &
        (adata.obs["pct_counts_mt"] < max_mt)
    ].copy()

    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()

    adata.layers["raw_for_copykat"] = adata.X.copy()

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # =====================================================
    # FIXED scMalignantFinder PATH USAGE
    # =====================================================
    print("SCMALIGNANT_MODEL:", SCMALIGNANT_MODEL)
    print("MODEL EXISTS:", os.path.exists(SCMALIGNANT_MODEL))

    from scMalignantFinder import classifier

    model = classifier.scMalignantFinder(
        test_input=adata,
        celltype_annotation=False,
        pretrain_path=SCMALIGNANT_MODEL,
        feature_path=os.path.join(SCMALIGNANT_MODEL, "ordered_feature.tsv"),
    )

    model.load()
    res = model.predict()

    adata.obs["scMF"] = res.obs["scMalignantFinder_prediction"]

    if reference_h5ad is None:
        raise ValueError("reference_h5ad required for CopyKAT")

    copykat_pred = run_copykat(
        adata,
        reference_h5ad,
        n_cores,
        copykat_params
    )

    copykat_pred = copykat_pred.loc[adata.obs_names]
    adata.obs["copykat"] = copykat_pred["copykat.pred"]

    if malignant_strategy == "union":
        adata.obs["malignant"] = (
            (adata.obs["scMF"] == "malignant") |
            (adata.obs["copykat"] == "aneuploid")
        )
    else:
        adata.obs["malignant"] = (
            (adata.obs["scMF"] == "malignant") &
            (adata.obs["copykat"] == "aneuploid")
        )

    surf = pd.read_csv(SURFACEOME_PATH)["Gene"].tolist()
    adata = adata[:, adata.var_names.intersection(surf)].copy()

    sc.tl.rank_genes_groups(adata, groupby="malignant", method="wilcoxon")

    out = os.path.join(SAVE_DIR, "final_tumor.h5ad")
    adata.write(out)

    print(f"\nSaved: {out}")
    print("\n========== PREPROCESSING DONE ==========\n")

    return adata
