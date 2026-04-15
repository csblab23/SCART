"""
preprocessing.py — Module 3 of SCART Pipeline

Input:  AnnData returned by Module 2 (popv_annotation.auto_run_popv)
        which must contain adata.obs["popv_majority_vote_prediction"].
        If that column is missing, the function will raise a clear error
        with instructions on how to fix it.

Reference file (user-supplied):
        The SAME .h5ad reference that was passed to Module 2 as
        `user_reference` is optionally forwarded here for CopyKAT
        (normal-cell reference) and scMalignantFinder.
        Pass it via: run_preprocessing_pipeline(..., reference_h5ad="path.h5ad")
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

# ── paths ──────────────────────────────────────────────────────────────────────
SURFACEOME_PATH = "/lustre/anas.a/Vinaya/scT-CAR_Designer/GESP/GESP_surfaceome_gene.csv"
SAVE_DIR = "./preprocessed_output"
os.makedirs(SAVE_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Guard: verify popv column is present before doing anything else
# ══════════════════════════════════════════════════════════════════════════════

def _assert_popv_ready(adata):
    """
    Raise a helpful error if the popv column is missing.

    Common causes
    -------------
    1. Module 2 (popv_annotation.auto_run_popv) was never run.
    2. Module 2 ran but returned a different object from what was passed here.
    3. The adata was re-loaded from disk after Module 2 but the obs column
       was not saved / the wrong file was loaded.

    Fix
    ---
    Make sure you assign the return value of Module 2 and pass it directly:

        adata = popv_annotation.auto_run_popv(
            input_type="raw",
            nsamples=300,
            user_reference="Ovary_TSP1_30_...h5ad"
        )

        adata_preprocessed = preprocessing.run_preprocessing_pipeline(
            adata=adata,          # <─ same object returned above
            ...
        )
    """
    col = "popv_majority_vote_prediction"
    if col not in adata.obs.columns:
        available = list(adata.obs.columns)
        raise KeyError(
            f"\n\n{'='*60}\n"
            f"MISSING COLUMN: '{col}'\n"
            f"{'='*60}\n"
            f"Module 3 expects this column to be present in adata.obs.\n"
            f"It is added by Module 2 (popv_annotation.auto_run_popv).\n\n"
            f"Available columns in adata.obs:\n  {available}\n\n"
            f"HOW TO FIX:\n"
            f"  1. Run Module 2 first and capture its return value:\n"
            f"       adata = popv_annotation.auto_run_popv(...)\n"
            f"  2. Pass THAT adata directly to Module 3:\n"
            f"       preprocessing.run_preprocessing_pipeline(adata=adata, ...)\n"
            f"  3. If you saved/loaded adata in between, make sure you saved\n"
            f"     with adata.write('file.h5ad') AFTER Module 2 completed.\n"
            f"{'='*60}\n"
        )


# ══════════════════════════════════════════════════════════════════════════════
# scMalignantFinder — auto-download model
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_scmalignant_model():
    """
    Download scMalignantFinder model from GitHub if not already present.
    Returns the local model directory path.
    """
    model_dir = os.path.expanduser("~/.scart/scmalignant_model")
    required_file = os.path.join(model_dir, "ordered_feature.tsv")

    if os.path.exists(required_file):
        logger.info(f"scMalignantFinder model found at: {model_dir}")
        return model_dir

    os.makedirs(model_dir, exist_ok=True)
    logger.info("Downloading scMalignantFinder model from GitHub...")

    url = "https://github.com/Jonyyqn/scMalignantFinder/archive/refs/heads/main.zip"
    tmp_zip = os.path.join(tempfile.gettempdir(), "scmalignant_model.zip")
    urllib.request.urlretrieve(url, tmp_zip)

    with zipfile.ZipFile(tmp_zip, "r") as zip_ref:
        zip_ref.extractall(tempfile.gettempdir())

    extracted_model = os.path.join(
        tempfile.gettempdir(), "scMalignantFinder-main", "model"
    )

    for f in os.listdir(extracted_model):
        src = os.path.join(extracted_model, f)
        dst = os.path.join(model_dir, f)
        if not os.path.exists(dst):
            os.rename(src, dst)

    logger.info(f"scMalignantFinder model downloaded to: {model_dir}")
    return model_dir


# ══════════════════════════════════════════════════════════════════════════════
# CopyKAT
# ══════════════════════════════════════════════════════════════════════════════

def _extract_normal_barcodes_from_reference(reference_h5ad: str) -> list:
    """
    Load the user-supplied reference h5ad and return a list of cell barcodes
    that are annotated as normal / non-malignant.  These are passed to CopyKAT
    as its `norm.cell.names` argument so it can calibrate CNV inference.

    The function looks, in order, for:
      • adata.obs["cell_type"] containing "normal" or "epithelial"
      • adata.obs["popv_majority_vote_prediction"] containing "epithelial cell"
      • adata.obs["malignant"] == "normal"
    If none of those columns exist every barcode is returned (CopyKAT will
    infer normals automatically from the data, which is its default behaviour).
    """
    logger.info(f"Loading reference for CopyKAT normal cells: {reference_h5ad}")
    ref = sc.read_h5ad(reference_h5ad)

    obs = ref.obs

    for col in ["cell_type", "popv_majority_vote_prediction", "malignant"]:
        if col not in obs.columns:
            continue

        vals = obs[col].astype(str).str.lower()

        if col == "malignant":
            mask = vals == "normal"
        else:
            mask = vals.str.contains("normal") | vals.str.contains("epithelial")

        barcodes = obs.index[mask].tolist()
        if barcodes:
            logger.info(
                f"Found {len(barcodes)} normal-cell barcodes "
                f"from column '{col}' in reference."
            )
            return barcodes

    logger.warning(
        "Could not identify normal cells in reference by column name. "
        "CopyKAT will run without an explicit normal reference "
        "(auto-inference mode)."
    )
    return []


def _run_copykat(
    adata_raw_counts,
    n_cores: int = 4,
    sam_name: str = "copykat_run",
    normal_barcodes: list = None,
):
    """
    Run CopyKAT via rpy2.

    Parameters
    ----------
    adata_raw_counts : AnnData
        Must hold RAW integer counts in .X.
    n_cores : int
        Parallel cores for CopyKAT.
    sam_name : str
        Prefix for CopyKAT output files.
    normal_barcodes : list or None
        Cell barcodes known to be normal (from the reference h5ad).
        Pass [] or None to let CopyKAT infer normals automatically.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
    except ImportError as e:
        raise ImportError(
            "rpy2 is required to run CopyKAT.\n"
            "  pip install rpy2\n"
            "  R -e \"devtools::install_github('navinlabcode/copykat')\""
        ) from e

    # ── build genes × cells matrix ────────────────────────────────────────────
    X = adata_raw_counts.X
    if sparse.issparse(X):
        X = X.toarray()

    # CopyKAT expects genes as rows, cells as columns
    raw_df = pd.DataFrame(
        X.T,
        index=adata_raw_counts.var_names,
        columns=adata_raw_counts.obs_names,
    )

    r_copykat = importr("copykat")
    r_base    = importr("base")

    r_mat = pandas2ri.py2rpy(raw_df)

    # ── normal cell reference ─────────────────────────────────────────────────
    if normal_barcodes:
        # keep only those barcodes that are actually in this query dataset
        valid_normals = [b for b in normal_barcodes
                         if b in adata_raw_counts.obs_names]
        if valid_normals:
            r_norm = ro.StrVector(valid_normals)
            logger.info(
                f"Passing {len(valid_normals)} normal barcodes to CopyKAT."
            )
        else:
            logger.warning(
                "None of the reference normal barcodes appear in the query "
                "dataset.  CopyKAT will run in auto-inference mode."
            )
            r_norm = ro.StrVector([])
    else:
        r_norm = ro.StrVector([])

    logger.info("Running CopyKAT (this may take several minutes)...")

    copykat_result = r_copykat.copykat(
        rawmat        = r_mat,
        id_type       = "S",
        norm_cell_names = r_norm,     # empty → auto-inference
        ngene_chr     = ro.IntVector([5]),
        win_size      = ro.IntVector([25]),
        KS_cut        = ro.FloatVector([0.1]),
        sam_name      = sam_name,
        distance      = "euclidean",
        genome        = "hg38",
        n_cores       = ro.IntVector([n_cores]),
    )

    pred_r  = r_base.as_data_frame(copykat_result.rx2("prediction"))
    pred_df = pandas2ri.rpy2py(pred_r)

    pred_series = pred_df.set_index("cell.names")["copykat.pred"]
    pred_series = pred_series.reindex(
        adata_raw_counts.obs_names, fill_value="not.defined"
    )

    return pred_series


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_preprocessing_pipeline(
    adata,
    min_genes: int   = 200,
    max_mt: float    = 40,
    log2fc_threshold: float = 2,
    pval_threshold: float   = 0.5,
    reference_h5ad: str     = None,
    n_cores: int     = 4,
):
    """
    Module 3 — Preprocessing & malignant-cell identification.

    Parameters
    ----------
    adata : AnnData
        Output of Module 2 (popv_annotation.auto_run_popv).
        Must contain adata.obs["popv_majority_vote_prediction"].

    min_genes : int
        Minimum genes per cell (QC filter).

    max_mt : float
        Maximum % mitochondrial reads allowed per cell (QC filter).

    log2fc_threshold : float
        Minimum log2 fold-change for DEG filtering.

    pval_threshold : float
        Maximum p-value for DEG filtering.

    reference_h5ad : str or None
        Path to the SAME reference .h5ad that was passed to Module 2
        (e.g. "Ovary_TSP1_30_...h5ad").
        Used to extract normal-cell barcodes for CopyKAT.
        If None, CopyKAT infers normal cells automatically.
        scMalignantFinder uses its own pre-trained model (auto-downloaded).

    n_cores : int
        Number of CPU cores for CopyKAT.

    Returns
    -------
    AnnData
        Filtered, malignant, surfaceome-restricted AnnData with DEG results
        stored in adata.uns["filtered_deg"].
        Also written to ./preprocessed_output/final_tumor.h5ad.
    """

    # ── 0. validate input ─────────────────────────────────────────────────────
    _assert_popv_ready(adata)

    # ── 1. ensure scMalignantFinder model ─────────────────────────────────────
    SCMALIGNANT_MODEL = _ensure_scmalignant_model()

    print("\n========== STARTING PREPROCESSING ==========\n")

    # ── 2. keep only epithelial cells (PopV label) ────────────────────────────
    labels         = adata.obs["popv_majority_vote_prediction"].astype(str)
    epithelial_mask = labels.str.endswith("epithelial cell")
    n_total        = adata.n_obs
    adata          = adata[epithelial_mask].copy()
    print(
        f"[Step 1] Epithelial filter: {adata.n_obs} / {n_total} cells retained."
    )

    if adata.n_obs == 0:
        raise ValueError(
            "No epithelial cells found after filtering on "
            "'popv_majority_vote_prediction'.  "
            "Check that Module 2 completed successfully and that the tissue "
            "type has epithelial cells (e.g. some blood cancers will not)."
        )

    # ── 3. QC filtering ───────────────────────────────────────────────────────
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    before = adata.n_obs
    adata  = adata[
        (adata.obs["n_genes_by_counts"] > min_genes) &
        (adata.obs["pct_counts_mt"]     < max_mt)
    ].copy()
    print(
        f"[Step 2] QC filter (min_genes={min_genes}, max_mt={max_mt}%): "
        f"{adata.n_obs} / {before} cells retained."
    )

    if adata.n_obs == 0:
        raise ValueError(
            "All cells were removed by QC filtering.  "
            "Consider relaxing min_genes or max_mt."
        )

    # ── 4. restore raw counts layer ───────────────────────────────────────────
    # We keep a copy of raw integer counts for CopyKAT before normalisation.
    for layer in ["scvi_counts", "raw_counts", "counts"]:
        if layer in adata.layers:
            adata.X = adata.layers[layer].copy()
            logger.info(f"Restored raw counts from layer '{layer}'.")
            break
    else:
        if adata.raw is not None:
            adata.X = adata.raw[adata.obs_names, adata.var_names].X.copy()
            logger.info("Restored raw counts from adata.raw.")
        else:
            logger.warning(
                "No raw count layer found.  Using current adata.X as-is. "
                "CopyKAT results may be unreliable if the matrix is "
                "already normalised."
            )

    adata_raw = adata.copy()   # save for CopyKAT (needs integer counts)

    adata.var_names_make_unique()

    # ── 5. normalise & log-transform ─────────────────────────────────────────
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("[Step 3] Normalisation & log1p complete.")

    # ── 6. scMalignantFinder ─────────────────────────────────────────────────
    print("[Step 4] Running scMalignantFinder...")
    try:
        from scMalignantFinder import classifier

        model = classifier.scMalignantFinder(
            test_input       = adata,
            celltype_annotation = False,
            pretrain_path    = SCMALIGNANT_MODEL,
            feature_path     = os.path.join(
                SCMALIGNANT_MODEL, "ordered_feature.tsv"
            ),
        )
        model.load()
        result = model.predict()
        adata.obs["scMalignantFinder_prediction"] = \
            result.obs["scMalignantFinder_prediction"]
        print("    scMalignantFinder completed.\n")

    except Exception as e:
        logger.error(f"scMalignantFinder failed: {e}")
        raise RuntimeError(
            "scMalignantFinder encountered an error.  See traceback above.\n"
            "The model is auto-downloaded from GitHub; ensure you have an "
            "internet connection and that the scMalignantFinder package is "
            "installed:\n"
            "  pip install scMalignantFinder"
        ) from e

    # ── 7. CopyKAT ────────────────────────────────────────────────────────────
    print("[Step 5] Running CopyKAT...")

    # Extract normal-cell reference barcodes from the user-supplied reference
    normal_barcodes = []
    if reference_h5ad is not None:
        if os.path.exists(reference_h5ad):
            normal_barcodes = _extract_normal_barcodes_from_reference(
                reference_h5ad
            )
        else:
            logger.warning(
                f"reference_h5ad path not found: {reference_h5ad}  "
                "CopyKAT will run without an explicit normal reference."
            )
    else:
        logger.info(
            "No reference_h5ad supplied.  "
            "CopyKAT will infer normal cells automatically."
        )

    try:
        copykat_pred = _run_copykat(
            adata_raw_counts = adata_raw,
            n_cores          = n_cores,
            sam_name         = "copykat_run",
            normal_barcodes  = normal_barcodes,
        )
        adata.obs["copykat_prediction"] = copykat_pred.values
        print("    CopyKAT completed.\n")

    except Exception as e:
        logger.error(f"CopyKAT failed: {e}")
        raise RuntimeError(
            "CopyKAT encountered an error.  See traceback above.\n"
            "Make sure rpy2 and the R package copykat are installed:\n"
            "  pip install rpy2\n"
            "  R -e \"devtools::install_github('navinlabcode/copykat')\""
        ) from e

    # ── 8. consensus malignancy filter ────────────────────────────────────────
    scmal_malignant  = adata.obs["scMalignantFinder_prediction"] == "malignant"
    copykat_aneuploid = adata.obs["copykat_prediction"]          == "aneuploid"

    adata.obs["consensus_malignant"] = scmal_malignant & copykat_aneuploid

    before = adata.n_obs
    adata  = adata[adata.obs["consensus_malignant"]].copy()
    print(
        f"[Step 6] Consensus malignant filter "
        f"(scMalignantFinder=malignant AND CopyKAT=aneuploid): "
        f"{adata.n_obs} / {before} cells retained."
    )

    if adata.n_obs == 0:
        raise ValueError(
            "No cells passed the consensus malignancy filter "
            "(scMalignantFinder=malignant AND CopyKAT=aneuploid).  "
            "Check that your input adata contains tumour cells and that "
            "both tools ran successfully."
        )

    # ── 9. surfaceome gene filter ─────────────────────────────────────────────
    print("[Step 7] Filtering to surfaceome genes...")
    if not os.path.exists(SURFACEOME_PATH):
        raise FileNotFoundError(
            f"Surfaceome gene list not found at:\n  {SURFACEOME_PATH}\n"
            "Update SURFACEOME_PATH at the top of preprocessing.py."
        )

    surfaceome = pd.read_csv(SURFACEOME_PATH)
    surf_genes  = surfaceome["Gene"].astype(str).tolist()
    common      = adata.var_names.intersection(surf_genes)
    adata       = adata[:, common].copy()
    print(f"    {len(common)} surfaceome genes retained.")

    if len(common) == 0:
        raise ValueError(
            "No surfaceome genes found in the data after filtering.  "
            "Check that the gene names in your AnnData match the 'Gene' "
            "column in the surfaceome CSV (e.g. HGNC symbols)."
        )

    # ── 10. differential expression ───────────────────────────────────────────
    print("[Step 8] Running rank_genes_groups (Wilcoxon)...")
    sc.tl.rank_genes_groups(
        adata,
        groupby = "scMalignantFinder_prediction",
        method  = "wilcoxon",
    )

    result   = sc.get.rank_genes_groups_df(adata, group=None)
    filtered = result[
        (result["logfoldchanges"] > log2fc_threshold) &
        (result["pvals"]          < pval_threshold)
    ]

    adata.uns["filtered_deg"] = filtered
    print(f"    {len(filtered)} DEGs passed thresholds "
          f"(log2FC > {log2fc_threshold}, p < {pval_threshold}).")

    # ── 11. binarise expression ───────────────────────────────────────────────
    adata.X = (adata.X > 0).astype(int)

    # ── 12. save ──────────────────────────────────────────────────────────────
    final_path = os.path.join(SAVE_DIR, "final_tumor.h5ad")
    adata.write(final_path)
    print(f"\n[Done] Saved final AnnData to: {final_path}")
    print("\n========== PREPROCESSING COMPLETED ==========\n")

    return adata
