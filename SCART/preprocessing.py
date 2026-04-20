"""
preprocessing.py
Module 3 — Preprocessing, malignancy detection, and surfaceome DEG

inferCNA step order (matches official tutorial exactly)
--------------------------------------------------------
  1. useGenome()     — set the reference genome for gene-coordinate lookup
  2. infercna()      — CNA inference on the COMBINED matrix (query + ref cells)
                       with refCells supplied so refCorrect() runs internally
  3. strip ref cols  — cnaM = cna[, query cells only]  (ref cells removed)
  4. findMalignant() — called on the FULL cna (query + ref), with
                         samples   = per-cell sample-name vector for query cols
                         excludeFromAvg = ref cell barcodes
                       fits bimodal Gaussians to cnaSignal × cnaCor and returns
                       list(nonmalignant=..., malignant=...)

Two malignancy methods run in sequence:
  1. scMalignantFinder — deep-learning expression classifier
  2. inferCNA          — chromosomal CNA inference (R package via rpy2)

Final label (final_malignant) is determined by malignant_strategy
  (union | intersection | scMalignant | infercna).
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
# Paths  (edit to match your installation)
# ---------------------------------------------------------------------------
SURFACEOME_PATH = (
    "/lustre/anas.a/Vinaya/scT-CAR_Designer/GESP/GESP_surfaceome_gene.csv"
)
SCMALIGNANT_MODEL = (
    "/home/igib/anaconda3/envs/scart/lib/python3.10/site-packages/"
    "SCART/external/scMalignantFinder/model"
)
SAVE_DIR = "/lustre/anas.a/Vinaya/scT-CAR_Designer/preprocessed_input"
os.makedirs(SAVE_DIR, exist_ok=True)


# ===========================================================================
# Build full-gene AnnData for scMalignantFinder
# ===========================================================================

def _build_fullgene_adata_for_scm(adata, feature_tsv: str):
    """
    Return a log-normalised AnnData covering the full original gene space.

    Route A — adata.raw   (populated by Module 2's _reattach_raw_slot fix)
    Route B — uns['full_var_names']  (SCART-specific, future-proofing)
    Route C — fallback: 4000-HVG adata as-is with a warning

    Only used for scMalignantFinder; caller's adata is not modified.
    """
    model_features = set(
        pd.read_csv(feature_tsv, sep="\t", header=None)[0].tolist()
    )
    n_model = len(model_features)

    def _pct(names):
        return len(set(names) & model_features) / n_model * 100

    # Route A
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

    # Route B
    if "full_var_names" in adata.uns:
        full_var = list(adata.uns["full_var_names"])
        ov = _pct(full_var)
        logger.info(f"Route B (uns['full_var_names']): {len(full_var)} genes, {ov:.1f}% overlap")
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
        f"Re-run Module 2 with the fixed popv_annotation.py to fix this."
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

    Tutorial step 1 — useGenome()
        Set the reference genome so infercna knows each gene's chromosomal
        position and can sort them for rolling-mean smoothing.

    Tutorial step 2 — infercna(m, refCells, n, noise, isLog)
        Input matrix m = genes × ALL cells (query + ref combined).
        refCells = named list of reference-cell barcodes (REF_* prefix).
        n        = keep the n most variable genes before CNA inference.
        noise    = exclude genes whose expression range < noise (reduces noise).
        isLog    = TRUE because we log-normalise before passing.
        Internally this runs:
          • filterGenes()  — removes genes absent from the genome annotation
          • orderGenes()   — sorts genes by chromosomal position (chr1→chrY)
          • rolling mean (runMean, window ≈ 101 genes) per cell
          • refCorrect()   — subtracts reference average → absolute CNA values

    Tutorial step 3 — strip reference columns
        cnaM = cna[, query cells only]
        (Reference cells served their purpose for refCorrect; drop them now.)

    Tutorial step 4 — findMalignant(cna, signal.threshold, samples, excludeFromAvg)
        Called on the FULL cna (query + ref columns still present).
        samples        = per-cell sample-name vector aligned to query columns,
                         used so cnaCor correlates each cell against its own
                         tumour average rather than a global average.
        excludeFromAvg = ref barcodes, so they don't bias the tumour average.
        Fits bimodal Gaussians to cnaSignal × cnaCor.
        Returns list(nonmalignant=..., malignant=...) or FALSE if unimodal.

    Parameters
    ----------
    adata_query : AnnData
        QC-filtered epithelial query cells (raw counts in scvi_counts layer).
    adata_ref : AnnData
        Normal reference (Tabula Sapiens); epithelial cells extracted inside.
    genome : str
        'hg19' (default, built-in) or 'hg38' (requires addGenome() in R).
    n : int
        Number of most-variable genes to keep before CNA inference (infercna n).
    noise : float
        Genes with expression range < noise are excluded (infercna noise).
    signal_threshold : float
        Top fraction of genes used for cnaSignal / cnaCor.  0.9 = top 10%.
        Passed to findMalignant(signal.threshold=...).

    Returns
    -------
    pd.Series
        Index = query barcodes (all of them), values:
          'malignant' | 'non-malignant' | 'not.defined'
    """
    # ------------------------------------------------------------------
    # rpy2 / R package availability check
    # ------------------------------------------------------------------
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
        numpy2ri.activate()
    except ImportError as exc:
        raise ImportError(
            "rpy2 is required to run inferCNA.\n"
            "Install: pip install rpy2\n"
            "R package: devtools::install_github('jlaffy/infercna')"
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

    # ------------------------------------------------------------------
    # STEP 0 — Prepare matrices
    #
    # inferCNA data requirements (from README):
    #   • genes × cells matrix
    #   • NOT centred
    #   • normalised for sequencing depth (CPM / TPM / RPKM)
    #   • optionally log-transformed (pass isLog=TRUE)
    #
    # We use log1p(CPM) and set isLog=TRUE.
    # ------------------------------------------------------------------

    def _to_log_cpm(adata_obj):
        X = _get_raw_matrix(adata_obj)          # cells × genes
        rs = X.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        return np.log1p(X / rs * 1e6).T         # → genes × cells

    mat_query = _to_log_cpm(adata_query)

    # Extract epithelial reference cells
    EPITHELIAL = {
        "epithelial cell",
        "glandular epithelial cell",
        "ovarian surface epithelial cell",
    }
    if "cell_ontology_class" in adata_ref.obs.columns:
        ep_mask = adata_ref.obs["cell_ontology_class"].str.lower().isin(EPITHELIAL)
        adata_ref_ep = adata_ref[ep_mask].copy() if ep_mask.any() else adata_ref
        logger.info(f"inferCNA reference: {ep_mask.sum()} epithelial reference cells")
    else:
        adata_ref_ep = adata_ref

    mat_ref = _to_log_cpm(adata_ref_ep)

    # ------------------------------------------------------------------
    # Align to common genes
    # ------------------------------------------------------------------
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
    sub_genes = q_genes[q_idx]

    mat_query_sub = mat_query[q_idx, :]
    mat_ref_sub   = mat_ref[r_idx,   :]

    # ------------------------------------------------------------------
    # Build combined genes × cells matrix  (query first, then REF_)
    # ------------------------------------------------------------------
    q_barcodes   = np.array(adata_query.obs_names)
    ref_barcodes = np.array(["REF_" + b for b in adata_ref_ep.obs_names])

    mat_combined = np.hstack([mat_query_sub, mat_ref_sub])
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

    # refCells: named list — inferCNA expects list(group1=c(...), group2=c(...))
    r_ref_cells = ro.ListVector({
        "normal_ref": ro.StrVector(ref_barcodes.tolist())
    })

    # ------------------------------------------------------------------
    # TUTORIAL STEP 1 — useGenome()
    # ------------------------------------------------------------------
    logger.info(f"inferCNA step 1: useGenome('{genome}')")
    infercna_r.useGenome(genome)

    # ------------------------------------------------------------------
    # TUTORIAL STEP 2 — infercna()
    #
    # Combined matrix (query + ref) passed as m.
    # refCells supplied → refCorrect() runs internally → absolute CNAs.
    # ------------------------------------------------------------------
    logger.info("inferCNA step 2: infercna() — CNA inference on combined matrix")
    cna = infercna_r.infercna(
        m        = r_mat,
        refCells = r_ref_cells,
        n        = n,
        noise    = noise,
        isLog    = True,
        verbose  = False,
    )

    # ------------------------------------------------------------------
    # TUTORIAL STEP 3 — strip reference columns from CNA result
    #
    # cnaM = cna[, !colnames(cna) %in% unlist(refCells)]
    # Reference cells served their purpose in refCorrect; drop them now.
    # We keep the full 'cna' object (query + ref) for findMalignant in step 4.
    # ------------------------------------------------------------------
    logger.info("inferCNA step 3: stripping reference columns → cnaM (query only)")
    r_ref_vec = ro.StrVector(ref_barcodes.tolist())
    # R: cnaM = cna[, !colnames(cna) %in% unlist(refCells)]
    cnaM = ro.r(
        """
        function(cna, ref_barcodes) {
            cna[, !colnames(cna) %in% ref_barcodes, drop=FALSE]
        }
        """
    )(cna, r_ref_vec)

    # ------------------------------------------------------------------
    # TUTORIAL STEP 4 — findMalignant()
    #
    # Called on the FULL cna (query + ref columns still present).
    # samples        = per-cell sample label vector for query columns only,
    #                  so cnaCor uses the correct tumour-average per sample.
    # excludeFromAvg = ref_barcodes, so they don't bias the tumour average.
    # signal.threshold = 0.9 → cnaHotspotGenes() selects top 10% of genes.
    # ------------------------------------------------------------------
    logger.info("inferCNA step 4: findMalignant() — bimodal Gaussian fitting")

    # Build per-cell sample-name vector aligned to ALL columns of cna
    # (tutorial: samples can be a vector of length == ncol matching columns)
    # For query cells we use "tumor"; for ref cells use "normal".
    sample_vec = ["tumor"] * len(q_barcodes) + ["normal"] * len(ref_barcodes)
    r_samples  = ro.StrVector(sample_vec)

    try:
        modes = infercna_r.findMalignant(
            cna             = cna,
            signal_threshold = signal_threshold,
            samples         = r_samples,
            excludeFromAvg  = r_ref_vec,
        )
        # findMalignant returns FALSE (not a list) if fitting fails
        if not hasattr(modes, "names") or modes.names is None:
            modes = None
    except Exception as exc:
        logger.warning(f"findMalignant() raised: {exc}")
        modes = None

    # ------------------------------------------------------------------
    # Parse R result → Python Series indexed by ALL query barcodes
    # ------------------------------------------------------------------
    if modes is None:
        logger.warning(
            "inferCNA findMalignant() returned FALSE or failed — "
            "bimodal fit did not converge (likely unimodal distribution). "
            "All query cells labelled 'not.defined'. "
            "Try lowering signal_threshold (e.g. 0.75) or check data quality."
        )
        return pd.Series(
            "not.defined",
            index=q_barcodes,
            name="infercna_prediction",
        )

    # modes is a named R list: names contain 'malignant' and 'nonmalignant'
    label_map = {}
    for key in list(modes.names):
        cells  = list(modes.rx2(key))
        label  = "malignant" if "malignant" in key.lower() else "non-malignant"
        for bc in cells:
            label_map[bc] = label

    preds = pd.Series(
        [label_map.get(bc, "not.defined") for bc in q_barcodes],
        index=q_barcodes,
        name="infercna_prediction",
    )
    # Guarantee full coverage via reindex (cells absent from modes → not.defined)
    preds = preds.reindex(q_barcodes, fill_value="not.defined")

    logger.info(
        "inferCNA predictions:\n" + preds.value_counts().to_string()
    )
    return preds


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_preprocessing_pipeline(
    adata=None,
    popv_path: str = None,
    min_genes: int = 200,
    max_mt: float = 40.0,
    log2fc_threshold: float = 2.0,
    pval_threshold: float = 0.5,
    reference_h5ad: str = None,
    malignant_strategy: str = "union",
    # ---- inferCNA parameters (user-editable) ----
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
        Log2 fold-change cutoff for DEG.
    pval_threshold : float
        P-value cutoff for DEG.
    reference_h5ad : str or None
        Path to the normal reference h5ad (Tabula Sapiens or equivalent).
        Required for inferCNA.  If None, inferCNA is skipped.
    malignant_strategy : str
        How to combine scMalignantFinder and inferCNA calls:
          'union'        — malignant if EITHER method says so  (recommended)
          'intersection' — malignant only if BOTH agree  (more specific)
          'scMalignant'  — use scMalignantFinder only
          'infercna'     — use inferCNA only  (requires reference_h5ad)

    inferCNA parameters (passed to R — edit these to tune the CNA inference)
    --------------------------------------------------------------------------
    infercna_genome : str
        Reference genome for gene ordering.
        'hg19' (default, built-in to infercna) or 'hg38'.
        For hg38 run in R first:
          library(infercna); addGenome(genome_df, name='hg38')
    infercna_n : int
        Number of most-variable genes to retain before CNA inference.
        Default 5000 (same as tutorial). Lower values (e.g. 3000) speed up
        inference; higher values (e.g. 8000) may capture more CNAs.
    infercna_noise : float
        Genes whose expression range across all cells is < noise are excluded.
        Default 0.1 (same as tutorial). Increase to 0.2 for noisier data.
    infercna_signal_threshold : float
        Top fraction of genes used to compute cnaSignal and cnaCor.
        Default 0.9 = top 10% of genes by CNA signal (same as tutorial).
        Lower to 0.75–0.8 if findMalignant() returns FALSE (unimodal fit).

    Returns
    -------
    AnnData
        Binary expression matrix over surfaceome DEGs with obs columns:
          scMalignantFinder_prediction, infercna_prediction (if run),
          final_malignant.
        adata.uns['filtered_deg'] = DEG result DataFrame.
    """
    print("\n========== START ==========\n")

    # ------------------------------------------------------------------
    # Auto-load adata
    # ------------------------------------------------------------------
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

    if adata.raw is not None:
        print(f"adata.raw: {adata.raw.n_vars} genes "
              f"(scMalignantFinder will use full gene space via Route A)")
    else:
        print("WARNING: adata.raw is None — scMalignantFinder will use "
              "4000 HVGs only (19% overlap). Re-run Module 2 with the "
              "fixed popv_annotation.py to resolve this.")

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
    adata.layers["raw_for_cna"] = adata.X.copy()   # raw snapshot for inferCNA

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ------------------------------------------------------------------
    # 4. scMalignantFinder
    # ------------------------------------------------------------------
    print("Running scMalignantFinder ...")
    feature_tsv = os.path.join(SCMALIGNANT_MODEL, "ordered_feature.tsv")
    print("  Building full-gene matrix ...")
    adata_scm = _build_fullgene_adata_for_scm(adata, feature_tsv)
    print(f"  Gene space: {adata_scm.n_vars} genes")

    from scMalignantFinder import classifier
    model = classifier.scMalignantFinder(
        test_input          = adata_scm,
        celltype_annotation = False,
        pretrain_path       = SCMALIGNANT_MODEL,
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
                "  Falling back to scMalignantFinder only."
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
    surfaceome = pd.read_csv(SURFACEOME_PATH)
    surfaceome.columns = surfaceome.columns.str.strip()
    surf_genes = surfaceome["Gene"].astype(str).tolist()
    adata      = adata[:, adata.var_names.intersection(surf_genes)].copy()
    print(f"Surfaceome genes retained: {adata.n_vars}\n")

    # ------------------------------------------------------------------
    # 8. DEG  (malignant vs normal)
    # ------------------------------------------------------------------
    sc.tl.rank_genes_groups(adata, groupby="final_malignant", method="wilcoxon")
    deg          = sc.get.rank_genes_groups_df(adata, group=None)
    filtered_deg = deg[
        (deg["logfoldchanges"] > log2fc_threshold) &
        (deg["pvals"] < pval_threshold)
    ]
    adata.uns["filtered_deg"] = filtered_deg
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

    final_path = os.path.join(SAVE_DIR, "final_tumor.h5ad")
    adata.write(final_path)
    print(f"Final object saved to:\n{final_path}")
    print("\n========== PREPROCESSING COMPLETED ==========\n")
    return adata
