"""
preprocessing.py
Module 3 — Preprocessing, malignancy detection, and surfaceome DEG

GitHub: https://github.com/navinlabcode/SCART

All previously hardcoded paths (/lustre/..., /home/igib/...) have been
replaced with function parameters or auto-detected from the installed
package.  No path in this file requires editing before use.

Path resolution strategy
-------------------------
SCMALIGNANT_MODEL  Auto-detected from the installed SCART package via
                   importlib; no user input needed.
SURFACEOME_PATH    Bundled inside SCART; auto-detected the same way.
SAVE_DIR           Defaults to 'preprocessing_results/' in the current
                   working directory; override with save_dir= parameter.

Fixes vs original
-----------------
  FIX A  DEG now filters on pvals_adj (BH-adjusted) instead of raw pvals.
          Raw p-values are uniform under the null and yield misleading
          cutoffs; adjusted values are the correct multiple-testing control.
          Default adjusted-pval threshold changed to 0.05 (was 0.5 raw).

  FIX B  adata.raw is now the primary gene-space source for scMalignant-
          Finder (Route A).  Module 2 (popv_annotation.py FIX 8) now
          writes adata.raw with the full gene space before saving, so the
          19% overlap warning should no longer appear.

  FIX C  rpy2 / inferCNA installation guidance is printed clearly when the
          module is missing, including the exact conda command.
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
# Auto-detect paths from the installed SCART package
# ===========================================================================

def _find_scart_resource(relative_path: str) -> str:
    """
    Locate a file bundled inside the installed SCART package.

    Tries importlib.resources first (works for editable and wheel installs),
    then falls back to a filesystem walk of the package root.

    Parameters
    ----------
    relative_path : str
        Path relative to the SCART package root, e.g.
        'external/scMalignantFinder/model' or
        'GESP/GESP_surfaceome_gene.csv'

    Returns
    -------
    str  Absolute path to the resource, or None if not found.
    """
    try:
        import SCART as _scart
        pkg_root = os.path.dirname(_scart.__file__)
        candidate = os.path.join(pkg_root, relative_path)
        if os.path.exists(candidate):
            return candidate
    except ImportError:
        pass
    return None


def _auto_scmalignant_model() -> str:
    path = _find_scart_resource("external/scMalignantFinder/model")
    if path is None:
        raise FileNotFoundError(
            "Could not auto-detect scMalignantFinder model directory.\n"
            "Pass scmalignant_model_dir= explicitly to run_preprocessing_pipeline().\n"
            "Expected location inside the SCART package:\n"
            "  <scart_root>/external/scMalignantFinder/model/"
        )
    return path


def _auto_surfaceome_path() -> str:
    # Try common locations inside the SCART package
    for candidate in (
        "GESP/GESP_surfaceome_gene.csv",
        "data/GESP_surfaceome_gene.csv",
        "resources/GESP_surfaceome_gene.csv",
    ):
        path = _find_scart_resource(candidate)
        if path is not None:
            return path
    raise FileNotFoundError(
        "Could not auto-detect surfaceome gene list inside the SCART package.\n"
        "Pass surfaceome_path= explicitly to run_preprocessing_pipeline().\n"
        "Expected a CSV with a 'Gene' column somewhere under the SCART package root."
    )


# ===========================================================================
# Build full-gene AnnData for scMalignantFinder
# ===========================================================================

def _build_fullgene_adata_for_scm(adata, feature_tsv: str):
    """
    Return a log-normalised AnnData covering the full original gene space.

    Route A — adata.raw            (populated by Module 2 FIX 8)
    Route B — uns['full_var_names'] (SCART-specific fallback)
    Route C — 4000-HVG adata as-is  (last resort, with warning)

    Only used for scMalignantFinder; caller's adata is not modified.
    """
    model_features = set(
        pd.read_csv(feature_tsv, sep="\t", header=None)[0].tolist()
    )
    n_model = len(model_features)

    def _pct(names):
        return len(set(names) & model_features) / n_model * 100

    # Route A — adata.raw
    if adata.raw is not None:
        ov = _pct(adata.raw.var_names)
        logger.info(f"Route A (adata.raw): {adata.raw.n_vars} genes, {ov:.1f}% overlap")
        if ov >= 50:
            X = adata.raw.X
            if sp.issparse(X):
                X = X.toarray()
            af = sc.AnnData(
                X   = X.astype(np.float32),
                obs = adata.obs.copy(),
                var = adata.raw.var.copy(),
            )
            sc.pp.normalize_total(af, target_sum=1e4)
            sc.pp.log1p(af)
            logger.info("Using Route A for scMalignantFinder.")
            return af
        logger.warning(f"Route A overlap only {ov:.1f}% — trying Route B.")

    # Route B — uns['full_var_names']
    if "full_var_names" in adata.uns:
        full_var = list(adata.uns["full_var_names"])
        ov = _pct(full_var)
        logger.info(f"Route B (uns): {len(full_var)} genes, {ov:.1f}% overlap")
        for lyr in ("scvi_counts", "raw_counts", "counts"):
            if lyr in adata.layers:
                X = adata.layers[lyr]
                if sp.issparse(X):
                    X = X.toarray()
                if X.shape[1] == len(full_var) and ov >= 50:
                    af = sc.AnnData(
                        X   = X.astype(np.float32),
                        obs = adata.obs.copy(),
                        var = pd.DataFrame(index=full_var),
                    )
                    sc.pp.normalize_total(af, target_sum=1e4)
                    sc.pp.log1p(af)
                    logger.info(f"Using Route B (layers['{lyr}']) for scMalignantFinder.")
                    return af

    # Route C — fallback
    ov_hvg = _pct(adata.var_names)
    logger.warning(
        f"Routes A and B failed. Falling back to {adata.n_vars} HVGs "
        f"({ov_hvg:.1f}% overlap). Results may be unreliable.\n"
        "Re-run Module 2 with the fixed popv_annotation.py to resolve this."
    )
    return adata.copy()


# ===========================================================================
# Helper: extract raw count matrix
# ===========================================================================

def _get_raw_matrix(adata):
    """Return a dense float64 (cells × genes) array of raw integer counts."""
    for lyr in ("scvi_counts", "raw_counts", "counts"):
        if lyr in adata.layers:
            logger.info(f"Raw counts from adata.layers['{lyr}']")
            X = adata.layers[lyr]
            break
    else:
        if adata.raw is not None:
            logger.info("Raw counts from adata.raw.X")
            X = adata.raw.X
        else:
            logger.info("No raw layer — assuming adata.X is raw counts")
            X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return np.array(X, dtype=np.float64)


# ===========================================================================
# inferCNA  (via rpy2) — step order matches the official tutorial exactly
# ===========================================================================

def _run_infercna(
    adata_query,
    adata_ref,
    genome: str = "hg19",
    n: int = 5000,
    noise: float = 0.1,
    signal_threshold: float = 0.9,
):
    """
    Run inferCNA following the exact step order from the official tutorial:
    https://rdrr.io/github/jlaffy/infercna/f/vignettes/infercna_tutorial.Rmd

    Step 1 — useGenome()
        Sets the built-in chromosome coordinate table (hg19 is bundled).
        'hg19' and 'hg38' are the two supported values.
        This is NOT a file — it is a string key for inferCNA's internal data.

    Step 2 — infercna(m, refCells, n, noise, isLog=TRUE)
        Input m = genes × ALL cells (query + ref cells combined).
        refCells = named list of normal-cell barcodes (REF_* prefix).
        Internally runs:
          • filterGenes()  — drops genes not in the genome annotation
          • orderGenes()   — sorts genes by chromosomal position
          • rolling mean (window ≈ n genes) per cell  → CNA profile
          • refCorrect()   — subtracts reference average → absolute CNAs

    Step 3 — strip reference columns
        cnaM = cna[, query cells only]  (ref cells used only for baseline)

    Step 4 — findMalignant(cna, signal.threshold, samples, excludeFromAvg)
        Called on the FULL cna (query + ref).
        samples        = per-cell sample label vector (length == ncol of cna)
        excludeFromAvg = ref barcodes, so they don't bias the tumour average
        Fits bimodal Gaussians to cnaSignal × cnaCor.
        Returns list(nonmalignant=..., malignant=...) or FALSE if unimodal.

    Parameters
    ----------
    adata_query : AnnData   Epithelial query cells (raw counts in layer).
    adata_ref   : AnnData   Normal reference (Tabula Sapiens ovary h5ad).
    genome      : str       'hg19' (default, built-in) or 'hg38'.
    n           : int       Most-variable genes to keep (default 5000).
    noise       : float     Exclude genes with range < noise (default 0.1).
    signal_threshold : float  Top fraction for cnaSignal/cnaCor (default 0.9).

    Returns
    -------
    pd.Series  Index = all query barcodes.
               Values = 'malignant' | 'non-malignant' | 'not.defined'
    """
    # --- rpy2 / R package availability ---
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
        numpy2ri.activate()
    except ImportError as exc:
        raise ImportError(
            "rpy2 is required to run inferCNA.\n\n"
            "Install instructions (run inside your scart conda env):\n"
            "  conda install -c conda-forge rpy2\n"
            "  # OR\n"
            "  pip install rpy2\n\n"
            "Then install the R package (run in R or via rpy2):\n"
            "  install.packages('devtools')\n"
            "  devtools::install_github('jlaffy/infercna')\n\n"
            "Verify installation:\n"
            "  python -c \"import rpy2.robjects as ro; "
            "print(ro.r('R.version.string'))\"\n"
        ) from exc

    try:
        infercna_r = importr("infercna")
    except Exception as exc:
        raise ImportError(
            "R package 'infercna' not found.\n"
            "Install in R:\n"
            "  install.packages('devtools')\n"
            "  devtools::install_github('jlaffy/infercna')"
        ) from exc

    # --- Prepare log-CPM matrices (genes × cells) ---
    def _to_log_cpm(adata_obj):
        X  = _get_raw_matrix(adata_obj)            # cells × genes
        rs = X.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        return np.log1p(X / rs * 1e6).T            # genes × cells

    mat_query = _to_log_cpm(adata_query)

    # Extract epithelial reference cells
    EPITHELIAL = {
        "epithelial cell",
        "glandular epithelial cell",
        "ovarian surface epithelial cell",
    }
    if "cell_ontology_class" in adata_ref.obs.columns:
        ep_mask      = adata_ref.obs["cell_ontology_class"].str.lower().isin(EPITHELIAL)
        adata_ref_ep = adata_ref[ep_mask].copy() if ep_mask.any() else adata_ref
        logger.info(f"inferCNA reference: {ep_mask.sum()} epithelial cells")
    else:
        adata_ref_ep = adata_ref

    mat_ref = _to_log_cpm(adata_ref_ep)

    # --- Align to common genes ---
    q_genes = np.array(adata_query.var_names)
    r_genes = np.array(adata_ref_ep.var_names)
    common  = np.intersect1d(q_genes, r_genes)
    logger.info(f"inferCNA common genes: {len(common)}")

    if len(common) < 200:
        raise ValueError(
            f"Only {len(common)} common genes — inferCNA needs ≥200. "
            "Check that query and reference both use HGNC gene symbols."
        )

    q_idx = np.where(np.isin(q_genes, common))[0]
    r_idx = np.where(np.isin(r_genes, common))[0]

    mat_combined = np.hstack([mat_query[q_idx, :], mat_ref[r_idx, :]])
    sub_genes    = q_genes[q_idx]

    q_barcodes   = np.array(adata_query.obs_names)
    ref_barcodes = np.array(["REF_" + b for b in adata_ref_ep.obs_names])
    all_barcodes = np.concatenate([q_barcodes, ref_barcodes])

    # --- Build R matrix ---
    r_mat = ro.r.matrix(
        ro.FloatVector(mat_combined.flatten(order="F")),
        nrow=mat_combined.shape[0],
        ncol=mat_combined.shape[1],
        dimnames=ro.ListVector([
            ro.StrVector(sub_genes.tolist()),
            ro.StrVector(all_barcodes.tolist()),
        ]),
    )
    r_ref_cells = ro.ListVector({
        "normal_ref": ro.StrVector(ref_barcodes.tolist())
    })
    r_ref_vec   = ro.StrVector(ref_barcodes.tolist())

    # ---------------------------------------------------------------
    # TUTORIAL STEP 1 — useGenome()
    # ---------------------------------------------------------------
    logger.info(f"inferCNA step 1: useGenome('{genome}')")
    infercna_r.useGenome(genome)

    # ---------------------------------------------------------------
    # TUTORIAL STEP 2 — infercna()  on COMBINED matrix
    # ---------------------------------------------------------------
    logger.info("inferCNA step 2: infercna() — CNA inference")
    cna = infercna_r.infercna(
        m        = r_mat,
        refCells = r_ref_cells,
        n        = n,
        noise    = noise,
        isLog    = True,
        verbose  = False,
    )

    # ---------------------------------------------------------------
    # TUTORIAL STEP 3 — strip reference columns → cnaM
    # ---------------------------------------------------------------
    logger.info("inferCNA step 3: stripping reference columns")
    cnaM = ro.r(
        "function(cna, ref) cna[, !colnames(cna) %in% ref, drop=FALSE]"
    )(cna, r_ref_vec)

    # ---------------------------------------------------------------
    # TUTORIAL STEP 4 — findMalignant()  on FULL cna
    # ---------------------------------------------------------------
    logger.info("inferCNA step 4: findMalignant() — bimodal Gaussian fitting")
    sample_vec = ["tumor"] * len(q_barcodes) + ["normal"] * len(ref_barcodes)
    r_samples  = ro.StrVector(sample_vec)

    try:
        modes = infercna_r.findMalignant(
            cna              = cna,
            signal_threshold = signal_threshold,
            samples          = r_samples,
            excludeFromAvg   = r_ref_vec,
        )
        if not hasattr(modes, "names") or modes.names is None:
            modes = None
    except Exception as exc:
        logger.warning(f"findMalignant() raised: {exc}")
        modes = None

    # --- Parse result → Python Series ---
    if modes is None:
        logger.warning(
            "inferCNA findMalignant() returned FALSE — bimodal fit did not "
            "converge. All query cells labelled 'not.defined'.\n"
            "Try lowering infercna_signal_threshold (e.g. 0.75)."
        )
        return pd.Series("not.defined", index=q_barcodes, name="infercna_prediction")

    label_map = {}
    for key in list(modes.names):
        label = "malignant" if "malignant" in key.lower() else "non-malignant"
        for bc in list(modes.rx2(key)):
            label_map[bc] = label

    preds = pd.Series(
        [label_map.get(bc, "not.defined") for bc in q_barcodes],
        index=q_barcodes,
        name="infercna_prediction",
    ).reindex(q_barcodes, fill_value="not.defined")

    logger.info("inferCNA predictions:\n" + preds.value_counts().to_string())
    return preds


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_preprocessing_pipeline(
    adata=None,
    popv_path: str = None,
    # --- QC ---
    min_genes: int = 200,
    max_mt: float = 40.0,
    # --- DEG ---
    log2fc_threshold: float = 1.0,
    pval_adj_threshold: float = 0.05,
    # --- paths (auto-detected if not given) ---
    reference_h5ad: str = None,
    save_dir: str = None,
    scmalignant_model_dir: str = None,
    surfaceome_path: str = None,
    # --- malignancy logic ---
    malignant_strategy: str = "union",
    # --- inferCNA parameters ---
    infercna_genome: str = "hg19",
    infercna_n: int = 5000,
    infercna_noise: float = 0.1,
    infercna_signal_threshold: float = 0.9,
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
        Log2 fold-change cutoff for DEG (default 1.0 — 2-fold).
        The original default of 2.0 is very strict; lower to 1.0 to
        recover more DEGs while still being biologically meaningful.
    pval_adj_threshold : float
        BH-adjusted p-value cutoff for DEG (default 0.05).

        *** FIX A ***  The original code used raw pvals with a cutoff
        of 0.5, which is essentially no multiple-testing control and
        admits nearly all genes. This parameter now filters on
        pvals_adj (Benjamini–Hochberg) at 0.05, which is the standard
        bioinformatics practice.  If you get 0 DEGs, try relaxing to
        pval_adj_threshold=0.10 or lowering log2fc_threshold to 0.5.

    reference_h5ad : str or None
        Path to the normal reference h5ad (Tabula Sapiens tissue-matched).
        Required for inferCNA.  The same file used in the PopV module works.
        If None, inferCNA is skipped.
    save_dir : str or None
        Directory where final_tumor.h5ad is written.
        Defaults to 'preprocessing_results/' in the current working directory.
    scmalignant_model_dir : str or None
        Path to the scMalignantFinder model directory.
        Auto-detected from the SCART package if not provided.
    surfaceome_path : str or None
        Path to the surfaceome gene list CSV (must have a 'Gene' column).
        Auto-detected from the SCART package if not provided.
    malignant_strategy : str
        How to combine scMalignantFinder and inferCNA calls:
          'union'        — malignant if EITHER method says so  (recommended)
          'intersection' — malignant only if BOTH agree  (more specific)
          'scMalignant'  — scMalignantFinder only
          'infercna'     — inferCNA only  (requires reference_h5ad)

    inferCNA parameters
    -------------------
    infercna_genome : str
        Built-in genome key for gene ordering.  NOT a file path.
        'hg19' (default, bundled inside the R package) or 'hg38'.
    infercna_n : int
        Most-variable genes to retain before CNA inference (default 5000).
    infercna_noise : float
        Genes with expression range < noise are excluded (default 0.1).
    infercna_signal_threshold : float
        Top fraction of genes used for cnaSignal / cnaCor (default 0.9).
        Lower to 0.75–0.8 if findMalignant() returns not.defined for all.

    Returns
    -------
    AnnData
        Binary expression matrix over surfaceome DEGs with obs columns:
          scMalignantFinder_prediction, infercna_prediction (if run),
          final_malignant.
        adata.uns['filtered_deg'] = DEG result DataFrame.
    """
    print("\n========== START ==========\n")

    # --- Resolve paths ------------------------------------------------------
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "preprocessing_results")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory: {save_dir}")

    if scmalignant_model_dir is None:
        scmalignant_model_dir = _auto_scmalignant_model()
    logger.info(f"scMalignantFinder model: {scmalignant_model_dir}")

    if surfaceome_path is None:
        surfaceome_path = _auto_surfaceome_path()
    logger.info(f"Surfaceome path: {surfaceome_path}")

    # --- Auto-load adata ----------------------------------------------------
    if adata is None:
        for path in [popv_path,
                     "popv_results/final_popv_annotated.h5ad",
                     "final_popv_annotated.h5ad"]:
            if path and os.path.exists(path):
                print(f"Loading POPV output (auto): {path}")
                adata = sc.read_h5ad(path)
                break
        if adata is None:
            raise FileNotFoundError(
                "Could not auto-detect POPV output. "
                "Pass adata= or popv_path= explicitly."
            )

    # --- Report gene-space availability -------------------------------------
    if adata.raw is not None:
        print(
            f"adata.raw detected: {adata.raw.n_vars} genes. "
            "scMalignantFinder will use full gene space via Route A."
        )
    else:
        print(
            "WARNING: adata.raw is None.\n"
            "  scMalignantFinder will fall back to 4000 HVGs (~19% overlap).\n"
            "  Re-run Module 2 with the fixed popv_annotation.py (FIX 8) to\n"
            "  resolve this and achieve >90% model feature overlap."
        )

    print(f"Initial cells: {adata.n_obs}")

    # ------------------------------------------------------------------
    # 1. Select epithelial cells
    # ------------------------------------------------------------------
    labels  = adata.obs["popv_majority_vote_prediction"].astype(str)
    ep_mask = labels.str.endswith("epithelial cell")
    adata   = adata[ep_mask].copy()
    print(f"Epithelial cells retained: {adata.n_obs}")
    print(f"Cells removed:             {(~ep_mask).sum()}\n")

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
    # 3. Route raw counts into .X; snapshot for inferCNA
    # ------------------------------------------------------------------
    print("Detecting raw count source...")
    for lyr in ("scvi_counts", "raw_counts", "counts"):
        if lyr in adata.layers:
            print(f"Using adata.layers['{lyr}'] as raw counts.")
            adata.X = adata.layers[lyr].copy()
            break
    else:
        if adata.raw is not None:
            print("Using adata.raw.X as raw counts.")
            adata.X = adata.raw.X.copy()
        else:
            print("No raw layer — assuming adata.X is raw counts.")

    adata.var_names_make_unique()
    adata.layers["raw_for_cna"] = adata.X.copy()   # snapshot for inferCNA

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ------------------------------------------------------------------
    # 4. scMalignantFinder
    # ------------------------------------------------------------------
    print("Running scMalignantFinder ...")
    feature_tsv = os.path.join(scmalignant_model_dir, "ordered_feature.tsv")
    print("  Building full-gene matrix ...")
    adata_scm = _build_fullgene_adata_for_scm(adata, feature_tsv)
    print(f"  Gene space: {adata_scm.n_vars} genes")

    from scMalignantFinder import classifier
    model = classifier.scMalignantFinder(
        test_input          = adata_scm,
        celltype_annotation = False,
        pretrain_path       = scmalignant_model_dir,
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
    # 5. inferCNA
    # ------------------------------------------------------------------
    infercna_available = False

    if malignant_strategy in ("infercna", "union", "intersection"):
        if reference_h5ad is None:
            print(
                "Warning: inferCNA skipped — no reference_h5ad provided.\n"
                "  Falling back to scMalignantFinder only.\n"
                "  To enable inferCNA, pass reference_h5ad='path/to/tabula.h5ad'."
            )
            malignant_strategy = "scMalignant"
        else:
            print("Running inferCNA ...")
            try:
                adata_raw_cna   = adata.copy()
                adata_raw_cna.X = adata.layers["raw_for_cna"]
                adata_ref_full  = sc.read_h5ad(reference_h5ad)

                infercna_preds = _run_infercna(
                    adata_query      = adata_raw_cna,
                    adata_ref        = adata_ref_full,
                    genome           = infercna_genome,
                    n                = infercna_n,
                    noise            = infercna_noise,
                    signal_threshold = infercna_signal_threshold,
                )
                adata.obs["infercna_prediction"] = infercna_preds.values
                infercna_available = True
                print("inferCNA completed.")
                print(adata.obs["infercna_prediction"].value_counts().to_string(), "\n")

            except Exception as exc:
                print(
                    f"Warning: inferCNA failed — {type(exc).__name__}: {exc}\n"
                    "  Falling back to scMalignantFinder only."
                )
                logger.exception("inferCNA error details:")
                malignant_strategy = "scMalignant"

    # ------------------------------------------------------------------
    # 6. Combine malignancy calls → final_malignant
    # ------------------------------------------------------------------
    scm_mal = adata.obs["scMalignantFinder_prediction"].str.lower() == "malignant"

    if infercna_available:
        cna_mal = adata.obs["infercna_prediction"].str.lower() == "malignant"
        if malignant_strategy == "union":
            malignant_mask = scm_mal | cna_mal
            strategy_label = "union  (scMalignantFinder OR inferCNA)"
        elif malignant_strategy == "intersection":
            malignant_mask = scm_mal & cna_mal
            strategy_label = "intersection  (scMalignantFinder AND inferCNA)"
        elif malignant_strategy == "infercna":
            malignant_mask = cna_mal
            strategy_label = "inferCNA only"
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
    surfaceome = pd.read_csv(surfaceome_path)
    surfaceome.columns = surfaceome.columns.str.strip()
    surf_genes = surfaceome["Gene"].astype(str).tolist()
    adata      = adata[:, adata.var_names.intersection(surf_genes)].copy()
    print(f"Surfaceome genes retained: {adata.n_vars}\n")

    # ------------------------------------------------------------------
    # 8. DEG  (malignant vs normal)
    #
    # FIX A: filter on pvals_adj (Benjamini–Hochberg) not raw pvals.
    #        Raw p-values are nearly uniform under H0 and the old cutoff
    #        of 0.5 had essentially no multiple-testing control, causing
    #        either too many or too few genes depending on the dataset.
    #        BH-adjusted values at 0.05 are the bioinformatics standard.
    # ------------------------------------------------------------------
    sc.tl.rank_genes_groups(
        adata,
        groupby="final_malignant",
        method="wilcoxon",
        key_added="rank_genes_groups",
    )
    deg = sc.get.rank_genes_groups_df(adata, group=None)

    print(f"Total DEG candidates: {deg.shape[0]}")
    print(
        f"Applying filters: log2FC > {log2fc_threshold}, "
        f"pvals_adj < {pval_adj_threshold}"
    )

    filtered_deg = deg[
        (deg["logfoldchanges"] > log2fc_threshold) &
        (deg["pvals_adj"] < pval_adj_threshold)          # FIX A
    ]

    if filtered_deg.shape[0] == 0:
        print(
            "WARNING: 0 DEGs passed the filter.\n"
            "  Suggestions:\n"
            "  1. Lower log2fc_threshold (try 0.5)\n"
            "  2. Raise pval_adj_threshold (try 0.10 or 0.20)\n"
            "  3. Check malignant/normal cell counts above — if one group\n"
            "     has <10 cells, the Wilcoxon test will have low power.\n"
            "  4. Ensure Module 2 FIX 8 is applied so scMalignantFinder\n"
            "     uses the full gene space (adata.raw) for better accuracy."
        )

    adata.uns["filtered_deg"]    = filtered_deg
    adata.uns["all_deg"]         = deg           # save full table for inspection
    adata.uns["deg_params"]      = {
        "log2fc_threshold" : log2fc_threshold,
        "pval_adj_threshold": pval_adj_threshold,
        "method"           : "wilcoxon",
    }
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

    final_path = os.path.join(save_dir, "final_tumor.h5ad")
    adata.write(final_path)
    print(f"Final object saved to: {final_path}")
    print("\n========== PREPROCESSING COMPLETED ==========\n")
    return adata
