"""
preprocessing.py
Module 3 — Preprocessing, malignancy detection, and surfaceome DEG

GitHub: https://github.com/navinlabcode/SCART

All previously hardcoded paths (/lustre/..., /home/igib/...) have been
replaced with function parameters or auto-detected from the installed
package.  No path in this file requires editing before use.

Path resolution strategy
-------------------------
SCMALIGNANT_MODEL  Auto-detected from SCART/external/scMalignantFinder/model/
SURFACEOME_PATH    Auto-detected from SCART/GESP/GESP_surfaceome_gene.csv
SAVE_DIR           Defaults to 'preprocessing_results/' in cwd

Fixes vs original
-----------------
  FIX A  DEG now filters on pvals_adj (BH-adjusted) not raw pvals.
  FIX B  _build_fullgene_adata_for_scm() checks layers['full_counts']
          first (written by Module 2 FIX 8) giving >=90% model overlap.
  FIX C  rpy2 symbol-lookup error (R/rpy2 version mismatch) now prints
          the exact conda reinstall command to fix it.
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

def _find_scart_resource(relative_path):
    try:
        import SCART as _scart
        pkg_root  = os.path.dirname(_scart.__file__)
        candidate = os.path.join(pkg_root, relative_path)
        if os.path.exists(candidate):
            return candidate
    except ImportError:
        pass
    return None


def _auto_scmalignant_model():
    path = _find_scart_resource("external/scMalignantFinder/model")
    if path is None:
        raise FileNotFoundError(
            "Could not auto-detect scMalignantFinder model directory.\n"
            "Pass scmalignant_model_dir= explicitly.\n"
            "Expected: <scart_root>/external/scMalignantFinder/model/"
        )
    return path


def _auto_surfaceome_path():
    for candidate in (
        "GESP/GESP_surfaceome_gene.csv",
        "data/GESP_surfaceome_gene.csv",
        "resources/GESP_surfaceome_gene.csv",
    ):
        path = _find_scart_resource(candidate)
        if path is not None:
            return path
    raise FileNotFoundError(
        "Could not auto-detect surfaceome CSV inside SCART package.\n"
        "Pass surfaceome_path= explicitly."
    )


# ===========================================================================
# FIX B — Build full-gene AnnData for scMalignantFinder
# ===========================================================================

def _build_fullgene_adata_for_scm(adata, feature_tsv):
    """
    Return a log-normalised AnnData covering the full original gene space.

    Route A-new  layers['full_counts'] + uns['full_counts_var_names']
                 Written by Module 2 FIX 8. PRIMARY route. >=90% overlap.
    Route A-old  adata.raw  (fallback for older Module 2 output)
    Route B      uns['full_var_names']  (SCART-specific fallback)
    Route C      4000-HVG adata as-is  (last resort, ~19% overlap)
    """
    model_features = set(
        pd.read_csv(feature_tsv, sep="\t", header=None)[0].tolist()
    )
    n_model = len(model_features)

    def _pct(names):
        return len(set(names) & model_features) / n_model * 100

    # Route A-new
    if "full_counts" in adata.layers:
        var_names = adata.uns.get("full_counts_var_names", None)
        if var_names is not None and len(var_names) == adata.layers["full_counts"].shape[1]:
            ov = _pct(var_names)
            logger.info(f"Route A-new (layers['full_counts']): {len(var_names)} genes, {ov:.1f}% overlap")
            if ov >= 50:
                X = adata.layers["full_counts"]
                if sp.issparse(X):
                    X = X.toarray()
                af = sc.AnnData(
                    X   = X.astype(np.float32),
                    obs = adata.obs.copy(),
                    var = pd.DataFrame(index=var_names),
                )
                sc.pp.normalize_total(af, target_sum=1e4)
                sc.pp.log1p(af)
                logger.info(f"scMalignantFinder using Route A-new ({len(var_names)} genes, {ov:.1f}% overlap).")
                return af
            logger.warning(f"Route A-new overlap only {ov:.1f}% — trying Route A-old.")
        else:
            logger.warning("'full_counts' present but 'full_counts_var_names' missing/mismatched — trying Route A-old.")

    # Route A-old
    if adata.raw is not None:
        ov = _pct(adata.raw.var_names)
        logger.info(f"Route A-old (adata.raw): {adata.raw.n_vars} genes, {ov:.1f}% overlap")
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
            logger.info("scMalignantFinder using Route A-old.")
            return af
        logger.warning(f"Route A-old overlap only {ov:.1f}% — trying Route B.")

    # Route B
    if "full_var_names" in adata.uns:
        full_var = list(adata.uns["full_var_names"])
        ov       = _pct(full_var)
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
                    logger.info(f"scMalignantFinder using Route B (layers['{lyr}']).")
                    return af

    # Route C — fallback
    ov_hvg = _pct(adata.var_names)
    logger.warning(
        f"Routes A-new, A-old, and B all failed.\n"
        f"Falling back to {adata.n_vars} HVGs ({ov_hvg:.1f}% overlap).\n"
        "Fix: re-run Module 2 with updated popv_annotation.py (FIX 8)."
    )
    return adata.copy()


# ===========================================================================
# Helper: extract raw count matrix
# ===========================================================================

def _get_raw_matrix(adata):
    """Return a dense float64 (cells x genes) array of raw integer counts."""
    for lyr in ("full_counts", "scvi_counts", "raw_counts", "counts"):
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
# inferCNA  (via rpy2)
# ===========================================================================

def _run_infercna(
    adata_query,
    adata_ref,
    genome="hg19",
    n=5000,
    noise=0.1,
    signal_threshold=0.9,
):
    """
    Run inferCNA following the official tutorial step order.
    Step 1: useGenome() — string key 'hg19' or 'hg38', NOT a file path.
    Step 2: infercna()  — CNA inference on combined query+ref matrix.
    Step 3: strip reference columns.
    Step 4: findMalignant() — bimodal Gaussian fitting.

    Returns pd.Series: index=query barcodes, values='malignant'|'non-malignant'|'not.defined'
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
        numpy2ri.activate()
    except Exception as exc:
        err_str = str(exc)
        # FIX C: specific R/rpy2 version mismatch detection
        if any(sym in err_str for sym in ("R_getVar", "undefined symbol", "R_ClosureEnv")):
            raise ImportError(
                "rpy2 failed because it was compiled against a DIFFERENT R version "
                "than the one active in your conda env.\n\n"
                f"Error: {err_str}\n\n"
                "Fix — reinstall rpy2 against the current R:\n\n"
                "  conda activate scart\n"
                "  conda remove rpy2 --force\n"
                "  conda install -c conda-forge rpy2\n\n"
                "If that still fails, build from source:\n"
                "  pip uninstall rpy2\n"
                "  pip install rpy2 --no-binary rpy2\n\n"
                "Verify:\n"
                "  python -c \"import rpy2.robjects as ro; print(ro.r('R.version.string'))\"\n"
            ) from exc
        raise ImportError(
            f"rpy2 failed to import: {exc}\n"
            "Install: conda install -c conda-forge rpy2\n"
            "R package: devtools::install_github('jlaffy/infercna')"
        ) from exc

    try:
        infercna_r = importr("infercna")
    except Exception as exc:
        raise ImportError(
            "R package 'infercna' not found.\n"
            "  install.packages('devtools')\n"
            "  devtools::install_github('jlaffy/infercna')"
        ) from exc

    def _to_log_cpm(adata_obj):
        X  = _get_raw_matrix(adata_obj)
        rs = X.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        return np.log1p(X / rs * 1e6).T   # genes x cells

    mat_query = _to_log_cpm(adata_query)

    EPITHELIAL = {"epithelial cell", "glandular epithelial cell", "ovarian surface epithelial cell"}
    if "cell_ontology_class" in adata_ref.obs.columns:
        ep_mask      = adata_ref.obs["cell_ontology_class"].str.lower().isin(EPITHELIAL)
        adata_ref_ep = adata_ref[ep_mask].copy() if ep_mask.any() else adata_ref
        logger.info(f"inferCNA reference: {ep_mask.sum()} epithelial cells")
    else:
        adata_ref_ep = adata_ref

    mat_ref = _to_log_cpm(adata_ref_ep)

    q_genes = np.array(adata_query.var_names)
    r_genes = np.array(adata_ref_ep.var_names)
    common  = np.intersect1d(q_genes, r_genes)
    logger.info(f"inferCNA common genes: {len(common)}")

    if len(common) < 200:
        raise ValueError(
            f"Only {len(common)} common genes — inferCNA needs >=200. "
            "Check that query and reference use HGNC gene symbols."
        )

    q_idx = np.where(np.isin(q_genes, common))[0]
    r_idx = np.where(np.isin(r_genes, common))[0]

    mat_combined = np.hstack([mat_query[q_idx, :], mat_ref[r_idx, :]])
    sub_genes    = q_genes[q_idx]
    q_barcodes   = np.array(adata_query.obs_names)
    ref_barcodes = np.array(["REF_" + b for b in adata_ref_ep.obs_names])
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
    r_ref_cells = ro.ListVector({"normal_ref": ro.StrVector(ref_barcodes.tolist())})
    r_ref_vec   = ro.StrVector(ref_barcodes.tolist())

    logger.info(f"inferCNA step 1: useGenome('{genome}')")
    infercna_r.useGenome(genome)

    logger.info("inferCNA step 2: infercna()")
    cna = infercna_r.infercna(
        m=r_mat, refCells=r_ref_cells, n=n, noise=noise, isLog=True, verbose=False,
    )

    logger.info("inferCNA step 3: strip reference columns")
    cnaM = ro.r("function(cna, ref) cna[, !colnames(cna) %in% ref, drop=FALSE]")(cna, r_ref_vec)

    logger.info("inferCNA step 4: findMalignant()")
    sample_vec = ["tumor"] * len(q_barcodes) + ["normal"] * len(ref_barcodes)
    r_samples  = ro.StrVector(sample_vec)

    try:
        modes = infercna_r.findMalignant(
            cna=cna, signal_threshold=signal_threshold,
            samples=r_samples, excludeFromAvg=r_ref_vec,
        )
        if not hasattr(modes, "names") or modes.names is None:
            modes = None
    except Exception as exc:
        logger.warning(f"findMalignant() raised: {exc}")
        modes = None

    if modes is None:
        logger.warning(
            "inferCNA findMalignant() returned FALSE — bimodal fit did not converge.\n"
            "All query cells labelled 'not.defined'.\n"
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
    popv_path=None,
    min_genes=200,
    max_mt=40.0,
    log2fc_threshold=1.0,
    pval_adj_threshold=0.05,
    reference_h5ad=None,
    save_dir=None,
    scmalignant_model_dir=None,
    surfaceome_path=None,
    malignant_strategy="union",
    infercna_genome="hg19",
    infercna_n=5000,
    infercna_noise=0.1,
    infercna_signal_threshold=0.9,
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
        Minimum genes per cell (QC filter). Default 200.
    max_mt : float
        Maximum mitochondrial % per cell (QC filter). Default 40.
    log2fc_threshold : float
        Log2 fold-change cutoff for DEG (default 1.0 = 2-fold).
    pval_adj_threshold : float
        BH-adjusted p-value cutoff for DEG (default 0.05).
        FIX A: replaces the original raw pval with cutoff 0.5.
        If 0 DEGs result, try pval_adj_threshold=0.10 or
        log2fc_threshold=0.5.
    reference_h5ad : str or None
        Path to the Tabula Sapiens reference h5ad for inferCNA.
        If None, inferCNA is skipped.
    save_dir : str or None
        Output directory. Defaults to 'preprocessing_results/' in cwd.
    scmalignant_model_dir : str or None
        Path to scMalignantFinder model dir.
        Auto-detected from SCART/external/scMalignantFinder/model/.
    surfaceome_path : str or None
        Path to surfaceome CSV with 'Gene' column.
        Auto-detected from SCART/GESP/GESP_surfaceome_gene.csv.
    malignant_strategy : str
        'union' | 'intersection' | 'scMalignant' | 'infercna'
    infercna_genome : str
        'hg19' (default, bundled in R package) or 'hg38'.
        This is a string KEY not a file path.
    infercna_n : int
        Most-variable genes to retain (default 5000).
    infercna_noise : float
        Exclude genes with expression range < noise (default 0.1).
    infercna_signal_threshold : float
        Top fraction for cnaSignal/cnaCor (default 0.9).
        Lower to 0.75 if findMalignant() returns not.defined for all.

    Returns
    -------
    AnnData  Binary expression matrix over surfaceome DEGs.
    """
    print("\n========== START ==========\n")

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

    if adata is None:
        for path in [popv_path,
                     "popv_results/final_popv_annotated.h5ad",
                     "final_popv_annotated.h5ad"]:
            if path and os.path.exists(path):
                print(f"Loading PopV output: {path}")
                adata = sc.read_h5ad(path)
                break
        if adata is None:
            raise FileNotFoundError(
                "Could not auto-detect PopV output. "
                "Pass adata= or popv_path= explicitly."
            )

    # FIX B: report gene-space availability
    if "full_counts" in adata.layers:
        n_full  = adata.layers["full_counts"].shape[1]
        n_names = len(adata.uns.get("full_counts_var_names", []))
        print(
            f"layers['full_counts'] detected: {n_full} genes "
            f"(uns['full_counts_var_names'] has {n_names} entries).\n"
            "scMalignantFinder will use full gene space via Route A-new."
        )
    elif adata.raw is not None:
        print(
            f"adata.raw detected: {adata.raw.n_vars} genes. "
            "scMalignantFinder will use full gene space via Route A-old."
        )
    else:
        print(
            "WARNING: Neither layers['full_counts'] nor adata.raw found.\n"
            "  scMalignantFinder will fall back to 4000 HVGs (~19% overlap).\n"
            "  Re-run Module 2 with updated popv_annotation.py (FIX 8)."
        )

    print(f"Initial cells: {adata.n_obs}")

    # 1. Select epithelial cells
    labels  = adata.obs["popv_majority_vote_prediction"].astype(str)
    ep_mask = labels.str.endswith("epithelial cell")
    adata   = adata[ep_mask].copy()
    print(f"Epithelial cells retained: {adata.n_obs}")
    print(f"Cells removed:             {(~ep_mask).sum()}\n")

    # 2. Quality control
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

    # 3. Route raw counts into .X; snapshot for inferCNA
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
    adata.layers["raw_for_cna"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 4. scMalignantFinder  (FIX B: Route A-new via full_counts)
    print("Running scMalignantFinder ...")
    feature_tsv = os.path.join(scmalignant_model_dir, "ordered_feature.tsv")
    print("  Building full-gene matrix ...")
    adata_scm   = _build_fullgene_adata_for_scm(adata, feature_tsv)
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

    # 5. inferCNA
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

    # 6. Combine malignancy calls
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

    adata.obs["final_malignant"] = malignant_mask.map({True: "malignant", False: "normal"})
    print(f"Malignancy strategy: {strategy_label}")
    print(f"  Malignant: {malignant_mask.sum()} | Normal: {(~malignant_mask).sum()}\n")

    # 7. Surfaceome filter
    surfaceome = pd.read_csv(surfaceome_path)
    surfaceome.columns = surfaceome.columns.str.strip()
    surf_genes = surfaceome["Gene"].astype(str).tolist()
    adata      = adata[:, adata.var_names.intersection(surf_genes)].copy()
    print(f"Surfaceome genes retained: {adata.n_vars}\n")

    # 8. DEG  (FIX A: pvals_adj not raw pvals)
    sc.tl.rank_genes_groups(
        adata, groupby="final_malignant", method="wilcoxon",
        key_added="rank_genes_groups",
    )
    deg = sc.get.rank_genes_groups_df(adata, group=None)

    print(f"Total DEG candidates: {deg.shape[0]}")
    print(f"Applying filters: log2FC > {log2fc_threshold}, pvals_adj < {pval_adj_threshold}")

    filtered_deg = deg[
        (deg["logfoldchanges"] > log2fc_threshold) &
        (deg["pvals_adj"] < pval_adj_threshold)        # FIX A
    ]

    if filtered_deg.shape[0] == 0:
        print(
            "WARNING: 0 DEGs passed the filter.\n"
            "  Try: lower log2fc_threshold (0.5) or raise pval_adj_threshold (0.10).\n"
            "  Also check malignant/normal cell counts — small groups reduce power.\n"
            "  Applying Module 2 FIX 8 (full gene space) improves classification."
        )

    adata.uns["filtered_deg"] = filtered_deg
    adata.uns["all_deg"]      = deg
    adata.uns["deg_params"]   = {
        "log2fc_threshold"  : log2fc_threshold,
        "pval_adj_threshold": pval_adj_threshold,
        "method"            : "wilcoxon",
    }
    print(f"Final DE genes retained: {filtered_deg.shape[0]}\n")

    # 9. Binarise
    adata.X = (adata.X > 0).astype(int)
    print("Expression converted to binary (0/1).\n")

    # 10. Save
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object:
            adata.obs[col] = adata.obs[col].astype(str)

    final_path = os.path.join(save_dir, "final_tumor.h5ad")
    adata.write(final_path)
    print(f"Final object saved to: {final_path}")
    print("\n========== PREPROCESSING COMPLETED ==========\n")
    return adata
