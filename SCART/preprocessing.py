"""
preprocessing.py
Module 3 — Preprocessing, malignancy detection, and surfaceome DEG

═══════════════════════════════════════════════════════════════════════════════
CORRECT BIOLOGICAL PIPELINE DESIGN
═══════════════════════════════════════════════════════════════════════════════

Step 1  Load the full PopV-annotated h5ad (all cell types, e.g. 15202 cells).
Step 2  Extract epithelial cells only → apply QC filters if set in Module 1.
Step 3  Run scMalignantFinder + CopyKAT on epithelial cells.
Step 4  Keep ONLY malignant epithelial cells.
Step 5  From adata_full, take all NON-EPITHELIAL cells as the "rest" group.
Step 6  DEG: malignant epithelial vs non-epithelial rest (surfaceome genes).
Step 7  Binarise ONLY the malignant epithelial AnnData, store DEG, save.

═══════════════════════════════════════════════════════════════════════════════
KEY FIXES
═══════════════════════════════════════════════════════════════════════════════

FIX 1   Full-gene route priority for scMalignantFinder.
FIX 2   CopyKAT reference subsampled to copykat_ref_max_cells (default 100).
FIX 3   DEG uses pvals_adj (BH-adjusted).
FIX 5   _get_raw_matrix prefers adata.raw over HVG layers.
FIX 6   scMalignantFinder predictions aligned by obs_names.
FIX 8   CopyKAT results saved in full detail in adata.uns['copykat_results'].
FIX 9   Correct DEG: malignant epithelial vs NON-EPITHELIAL cells.
FIX 10  CopyKAT runs via fresh Rscript subprocess (not rpy2).
        Rscript resolved via sys.executable bin dir FIRST — the only
        reliable signal inside a Jupyter kernel. CONDA_PREFIX and PATH
        are fallbacks only.
QC FIX  QC thresholds read from adata.uns['qc_params'] (set by Module 1).
        If absent, QC is SKIPPED ENTIRELY.
"""

import os
import sys
import glob
import shutil
import logging
import tempfile
import subprocess

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Auto-detect paths
# ═══════════════════════════════════════════════════════════════════════════

def _find_scart_resource(relative_path):
    try:
        import SCART as _scart
        candidate = os.path.join(os.path.dirname(_scart.__file__), relative_path)
        if os.path.exists(candidate):
            return candidate
    except ImportError:
        pass
    return None


def _auto_scmalignant_model():
    path = _find_scart_resource("external/scMalignantFinder/model")
    if path is None:
        raise FileNotFoundError(
            "Could not auto-detect scMalignantFinder model.\n"
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
        "Could not auto-detect GESP surfaceome CSV.\n"
        "Pass surfaceome_path= explicitly."
    )


def _auto_tumor_h5ad():
    """Auto-detect Module 1 tumor h5ad from cwd and GSE_data/."""
    patterns    = ["*_tumor.h5ad", "combined_tumor.h5ad", "input_tumor.h5ad"]
    search_dirs = [os.getcwd(), os.path.join(os.getcwd(), "GSE_data")]
    files = []
    for d in search_dirs:
        for pat in patterns:
            files.extend(glob.glob(os.path.join(d, pat)))
    files = list(set(files))
    if not files:
        return None
    found = max(files, key=os.path.getctime)
    logger.info(f"Auto-detected Module 1 tumor h5ad: {found}")
    return found


def _auto_popv_h5ad():
    """Auto-detect Module 2 PopV output h5ad."""
    for c in [
        os.path.join("popv_results", "final_popv_annotated.h5ad"),
        "final_popv_annotated.h5ad",
    ]:
        if os.path.exists(c):
            return c
    return None


# ═══════════════════════════════════════════════════════════════════════════
# QC parameter reader
# ═══════════════════════════════════════════════════════════════════════════

def _read_qc_params(adata):
    qc = adata.uns.get("qc_params", None)
    if qc is None:
        return (None, None, False,
                "SKIPPED — 'qc_params' not found in adata.uns.")
    min_genes = qc.get("min_genes", None)
    max_mt    = qc.get("max_mt",    None)
    if min_genes is not None:
        min_genes = int(min_genes)
    if max_mt is not None:
        max_mt = float(max_mt)
    qc_active = (min_genes is not None) or (max_mt is not None)
    parts = []
    if min_genes is not None:
        parts.append(f"min_genes={min_genes}")
    if max_mt is not None:
        parts.append(f"max_mt={max_mt}")
    source = (
        "adata.uns['qc_params'] — " + ", ".join(parts)
        if qc_active
        else "'qc_params' present but both values None — QC SKIPPED."
    )
    return min_genes, max_mt, qc_active, source


# ═══════════════════════════════════════════════════════════════════════════
# FIX 5 — raw count extractor
# ═══════════════════════════════════════════════════════════════════════════

def _get_raw_matrix(adata):
    """Return dense float64 (cells × genes) raw count matrix."""
    if adata.raw is not None:
        raw_n   = adata.raw.n_vars
        layer_n = max(
            (adata.layers[l].shape[1]
             for l in ("full_counts", "scvi_counts", "raw_counts", "counts")
             if l in adata.layers),
            default=0,
        )
        if raw_n > layer_n:
            logger.info(f"Raw counts: adata.raw ({raw_n} genes > best layer {layer_n})")
            X = adata.raw.X
            if sp.issparse(X):
                X = X.toarray()
            return np.array(X, dtype=np.float64)
    for lyr in ("full_counts", "scvi_counts", "raw_counts", "counts"):
        if lyr in adata.layers:
            logger.info(f"Raw counts: layers['{lyr}']")
            X = adata.layers[lyr]
            if sp.issparse(X):
                X = X.toarray()
            return np.array(X, dtype=np.float64)
    src = adata.raw.X if adata.raw is not None else adata.X
    logger.warning("Using adata.X as raw counts — may be log-normalised.")
    if sp.issparse(src):
        src = src.toarray()
    return np.array(src, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# FIX 1 — full-gene AnnData for scMalignantFinder
# ═══════════════════════════════════════════════════════════════════════════

def _build_fullgene_adata_for_scm(adata, feature_tsv, tumor_h5ad_path=None):
    model_features = set(
        pd.read_csv(feature_tsv, sep="\t", header=None)[0].tolist()
    )
    n_model = len(model_features)

    def _pct(names):
        return len(set(names) & model_features) / n_model * 100

    def _make_adata(X, obs, var_ref):
        if sp.issparse(X):
            X = X.toarray()
        X   = np.array(X, dtype=np.float32)
        var = var_ref if isinstance(var_ref, pd.DataFrame) \
              else pd.DataFrame(index=list(var_ref))
        af  = sc.AnnData(X=X, obs=obs.copy(), var=var)
        sc.pp.normalize_total(af, target_sum=1e4)
        sc.pp.log1p(af)
        af.X = sp.csr_matrix(af.X)
        return af

    # Route A-new
    if "full_counts" in adata.layers:
        vn = adata.uns.get("full_counts_var_names")
        if vn is not None and len(vn) == adata.layers["full_counts"].shape[1]:
            ov = _pct(vn)
            logger.info(f"Route A-new: {len(vn)} genes, {ov:.1f}% model overlap")
            if ov >= 50:
                af = _make_adata(adata.layers["full_counts"], adata.obs, vn)
                logger.info(f"scMalignantFinder → Route A-new ({af.n_vars} genes).")
                return af
            logger.warning(f"Route A-new overlap {ov:.1f}% < 50% — trying A-rescue.")
        else:
            logger.warning("full_counts gene names mismatch — trying A-rescue.")

    # Route A-rescue
    rescue = tumor_h5ad_path or _auto_tumor_h5ad()
    if rescue and os.path.exists(rescue):
        logger.info(f"Route A-rescue: {rescue}")
        try:
            m1     = sc.read_h5ad(rescue)
            shared = sorted(set(adata.obs_names) & set(m1.obs_names))
            if m1.raw is not None and len(shared) > 0:
                ov = _pct(m1.raw.var_names)
                logger.info(f"Route A-rescue (m1.raw): {m1.raw.n_vars} genes, {ov:.1f}%")
                if ov >= 50:
                    order = [c for c in adata.obs_names if c in set(shared)]
                    sub   = m1[order]
                    af    = _make_adata(sub.raw.X, adata.obs.loc[order], sub.raw.var)
                    logger.info(f"scMalignantFinder → Route A-rescue ({af.n_vars} genes).")
                    return af
            for lyr in ("counts", "raw_counts", "scvi_counts"):
                if lyr in m1.layers and len(shared) > 0:
                    ov = _pct(m1.var_names)
                    logger.info(
                        f"Route A-rescue (m1.layers['{lyr}']): {m1.n_vars} genes, {ov:.1f}%"
                    )
                    if ov >= 50:
                        order = [c for c in adata.obs_names if c in set(shared)]
                        sub   = m1[order]
                        af    = _make_adata(sub.layers[lyr], adata.obs.loc[order], sub.var)
                        logger.info(f"scMalignantFinder → Route A-rescue (layers['{lyr}']).")
                        return af
                    break
        except Exception as exc:
            logger.warning(f"Route A-rescue failed: {exc}")

    # Route A-old
    if adata.raw is not None:
        ov = _pct(adata.raw.var_names)
        logger.info(f"Route A-old (adata.raw): {adata.raw.n_vars} genes, {ov:.1f}%")
        if ov >= 50:
            af = _make_adata(adata.raw.X, adata.obs, adata.raw.var)
            logger.info("scMalignantFinder → Route A-old.")
            return af
        logger.warning(f"Route A-old overlap {ov:.1f}% < 50%.")

    # Route B
    if "full_var_names" in adata.uns:
        fv  = list(adata.uns["full_var_names"])
        ov  = _pct(fv)
        logger.info(f"Route B (uns): {len(fv)} genes, {ov:.1f}%")
        for lyr in ("scvi_counts", "raw_counts", "counts"):
            if lyr in adata.layers:
                X = adata.layers[lyr]
                if sp.issparse(X):
                    X = X.toarray()
                if X.shape[1] == len(fv) and ov >= 50:
                    af = _make_adata(X, adata.obs, fv)
                    logger.info(f"scMalignantFinder → Route B (layers['{lyr}']).")
                    return af

    # Route C — last resort
    ov_hvg = _pct(adata.var_names)
    logger.warning(
        f"All routes failed. Falling back to {adata.n_vars} HVGs ({ov_hvg:.1f}% overlap).\n"
        "Place GSE*_tumor.h5ad in cwd or re-run Module 2 with FIX 8."
    )
    return adata.copy()


# ═══════════════════════════════════════════════════════════════════════════
# FIX 10 — Rscript resolver (sys.executable first)
# ═══════════════════════════════════════════════════════════════════════════

def _find_rscript():
    """
    Return the Rscript binary for the active conda environment.

    Priority:
      1. dirname(sys.executable)/Rscript  ← MOST reliable in Jupyter kernels
         sys.executable always points to the kernel's env, even when
         CONDA_PREFIX points to a different (server) env.
      2. $CONDA_PREFIX/bin/Rscript        ← fallback; unreliable in Jupyter
      3. shutil.which("Rscript")          ← PATH fallback, last resort
    """
    # 1. Same bin/ as the active Python interpreter — always correct in Jupyter
    py_bin = os.path.dirname(os.path.abspath(sys.executable))
    cand   = os.path.join(py_bin, "Rscript")
    if os.path.isfile(cand):
        logger.info(f"Rscript found via sys.executable dir: {cand}")
        return cand

    # 2. CONDA_PREFIX — reliable only when launched from the correct env shell
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        cand = os.path.join(conda_prefix, "bin", "Rscript")
        if os.path.isfile(cand):
            logger.warning(
                f"Rscript found via CONDA_PREFIX (may be wrong env in Jupyter): {cand}\n"
                f"  CONDA_PREFIX  = {conda_prefix}\n"
                f"  sys.executable= {sys.executable}"
            )
            return cand

    # 3. PATH fallback
    cand = shutil.which("Rscript")
    if cand:
        logger.warning(f"Rscript found via PATH (may be wrong env): {cand}")
        return cand

    return None


def _get_r_home(rscript_bin):
    try:
        r_home = subprocess.check_output(
            [rscript_bin, "--vanilla", "-e", "cat(R.home())"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return r_home
    except Exception as exc:
        logger.warning(f"Could not determine R_HOME from {rscript_bin}: {exc}")
        return None


def _build_r_env(r_home):
    env = os.environ.copy()
    if r_home:
        env["R_HOME"] = r_home
        r_lib = os.path.join(r_home, "lib")
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            r_lib + (os.pathsep + existing_ld if existing_ld else "")
        )
        logger.info(f"Subprocess R_HOME → {r_home}")
    return env


# ═══════════════════════════════════════════════════════════════════════════
# FIX 10 — CopyKAT subprocess runner
# ═══════════════════════════════════════════════════════════════════════════

def _run_copykat(
    adata_query,
    adata_ref,
    genome="hg20",
    id_type="S",
    ngene_chr=10,
    win_size=50,
    ks_cut=0.1,
    distance="euclidean",
    n_cores=2,
    plot_genes=True,
    output_seg=False,
    ref_max_cells=100,
    sam_name="copykat_run",
    ref_epithelial_key="cell_ontology_class",
    ref_epithelial_values=None,
):
    """
    Run CopyKAT via a fresh Rscript subprocess (FIX 10).

    Rscript is resolved via sys.executable bin dir first, guaranteeing the
    correct conda environment R is used even when CONDA_PREFIX is wrong.

    Data flow:
      Python → numpy .npy + text files
             → prep.R (builds R matrix RDS)
             → run_copykat.R (runs copykat, writes prediction CSV)
             → pandas DataFrame

    Returns
    -------
    pd.DataFrame  columns: barcode, copykat_prediction
                  values:  "aneuploid" | "diploid" | "not.defined"
    """
    if ref_epithelial_values is None:
        ref_epithelial_values = ["epithelial cell"]

    q_barcodes   = np.array(adata_query.obs_names)
    empty_result = pd.DataFrame({
        "barcode"            : list(q_barcodes),
        "copykat_prediction" : "not.defined",
    })

    # Locate Rscript via sys.executable first (FIX 10)
    rscript_bin = _find_rscript()
    if rscript_bin is None:
        logger.error("Rscript not found. CopyKAT skipped.")
        return empty_result

    r_home  = _get_r_home(rscript_bin)
    sub_env = _build_r_env(r_home)

    # Verify copykat is installed in THIS R
    try:
        check = subprocess.run(
            [rscript_bin, "--vanilla", "-e",
             "if (!requireNamespace('copykat', quietly=TRUE)) "
             "{ cat('NOT_INSTALLED'); quit(status=1) } else { cat('OK') }"],
            capture_output=True, text=True, env=sub_env,
        )
        if "NOT_INSTALLED" in check.stdout or check.returncode != 0:
            raise ImportError(
                f"R package 'copykat' is not installed in the R used by:\n"
                f"  {rscript_bin}\n"
                f"R home: {r_home}\n"
                f"Install it:\n"
                f"  {rscript_bin} -e \"devtools::install_github('navinlabcode/copykat')\""
            )
        logger.info(f"copykat verified OK in {rscript_bin}")
    except ImportError:
        raise
    except Exception as exc:
        logger.error(f"copykat verification failed: {exc}")
        return empty_result

    # Locate bundled run_copykat.R
    runner_script = _find_scart_resource("external/run_copykat.R")
    if runner_script is None:
        runner_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "external", "run_copykat.R"
        )
    if not os.path.isfile(runner_script):
        raise FileNotFoundError(
            f"CopyKAT runner script not found.\n"
            f"Expected: SCART/external/run_copykat.R\n"
            f"Checked:  {runner_script}"
        )
    logger.info(f"CopyKAT runner script: {runner_script}")

    # FIX 2 — filter reference to normal epithelial cells and subsample
    if ref_epithelial_key in adata_ref.obs.columns:
        ref_vals_lower = [v.lower() for v in ref_epithelial_values]
        ep_mask        = adata_ref.obs[ref_epithelial_key].str.lower().isin(ref_vals_lower)
        adata_ref_ep   = adata_ref[ep_mask].copy() if ep_mask.any() else adata_ref.copy()
        logger.info(f"CopyKAT reference epithelial cells: {adata_ref_ep.n_obs}")
    else:
        logger.warning(
            f"ref_epithelial_key '{ref_epithelial_key}' not found. "
            "Using full reference."
        )
        adata_ref_ep = adata_ref.copy()

    if adata_ref_ep.n_obs > ref_max_cells:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(adata_ref_ep.n_obs, size=ref_max_cells, replace=False)
        adata_ref_ep = adata_ref_ep[np.sort(idx)].copy()
        logger.info(f"CopyKAT reference subsampled to {ref_max_cells} cells.")

    if adata_ref_ep.n_obs == 0:
        logger.warning("CopyKAT: no reference cells after filtering. Skipping.")
        return empty_result

    # Build combined genes × cells raw count matrix
    logger.info("CopyKAT: extracting raw counts ...")
    mat_query = _get_raw_matrix(adata_query).T
    mat_ref   = _get_raw_matrix(adata_ref_ep).T

    q_genes = np.array(adata_query.var_names)
    r_genes = (
        np.array(adata_ref_ep.raw.var_names)
        if adata_ref_ep.raw is not None and adata_ref_ep.raw.n_vars >= mat_ref.shape[0]
        else np.array(adata_ref_ep.var_names)
    )
    if mat_ref.shape[0] != len(r_genes):
        r_genes = np.array(adata_ref_ep.var_names)
        mat_ref = _get_raw_matrix(adata_ref_ep).T

    common_genes = np.intersect1d(q_genes, r_genes)
    logger.info(f"CopyKAT common genes: {len(common_genes)}")

    if len(common_genes) < 200:
        raise ValueError(
            f"Only {len(common_genes)} common genes. Need >= 200. "
            "Both datasets must use HGNC gene symbols."
        )
    if len(common_genes) < 2000:
        logger.warning(f"Only {len(common_genes)} common genes — CopyKAT prefers more.")

    q_idx = np.where(np.isin(q_genes, common_genes))[0]
    r_idx = np.where(np.isin(r_genes, common_genes))[0]

    q_barcodes_sub = np.array(adata_query.obs_names)
    r_barcodes_sub = np.array(["REF_" + b for b in adata_ref_ep.obs_names])
    mat_combined   = np.hstack([mat_query[q_idx, :], mat_ref[r_idx, :]])
    all_barcodes   = np.concatenate([q_barcodes_sub, r_barcodes_sub])
    normal_cells   = r_barcodes_sub.tolist()

    logger.info(
        f"CopyKAT combined matrix: {mat_combined.shape[0]} genes × "
        f"{mat_combined.shape[1]} cells "
        f"({len(q_barcodes_sub)} query + {len(r_barcodes_sub)} ref)"
    )

    tmpdir = tempfile.mkdtemp(prefix="scart_copykat_")
    try:
        mat_path      = os.path.join(tmpdir, "matrix.npy")
        genes_path    = os.path.join(tmpdir, "genes.txt")
        barcodes_path = os.path.join(tmpdir, "barcodes.txt")
        norm_path     = os.path.join(tmpdir, "norm_cells.txt")
        rds_path      = os.path.join(tmpdir, "copykat_input.rds")
        output_csv    = os.path.join(tmpdir, "copykat_pred.csv")
        prep_r        = os.path.join(tmpdir, "prep.R")

        n_genes = mat_combined.shape[0]
        n_cells = mat_combined.shape[1]

        np.save(mat_path, mat_combined.astype(np.float32))
        np.savetxt(genes_path,    common_genes, fmt="%s")
        np.savetxt(barcodes_path, all_barcodes, fmt="%s")
        np.savetxt(norm_path,     normal_cells, fmt="%s")

        # prep.R — reconstruct genes × cells matrix from numpy binary, save RDS
        # numpy saves float32 C-order (row-major) with 128-byte header.
        # byrow=TRUE in matrix() restores the genes × cells layout.
        with open(prep_r, "w") as f:
            f.write(f"""\
genes      <- readLines("{genes_path}")
barcodes   <- readLines("{barcodes_path}")
norm_cells <- readLines("{norm_path}")
n_genes <- {n_genes}
n_cells <- {n_cells}
raw_bytes  <- readBin("{mat_path}", what = "raw",
                      n = file.info("{mat_path}")$size)
data_bytes <- raw_bytes[-(1:128)]
vals       <- readBin(data_bytes, what = "numeric",
                      n = n_genes * n_cells, size = 4, endian = "little")
r_mat <- matrix(vals, nrow = n_genes, ncol = n_cells, byrow = TRUE)
rownames(r_mat) <- genes
colnames(r_mat) <- barcodes
saveRDS(list(mat = r_mat, norm_cells = norm_cells), "{rds_path}")
cat("[prep.R] RDS saved — dims:", nrow(r_mat), "x", ncol(r_mat), "\\n")
""")

        logger.info("CopyKAT: building RDS via prep.R ...")
        prep_result = subprocess.run(
            [rscript_bin, "--vanilla", prep_r],
            capture_output=True, text=True, env=sub_env,
        )
        for line in prep_result.stdout.strip().splitlines():
            logger.info(f"[prep.R] {line}")
        for line in prep_result.stderr.strip().splitlines():
            logger.debug(f"[prep.R stderr] {line}")
        if prep_result.returncode != 0:
            logger.error(f"prep.R failed.\n" + prep_result.stderr[-2000:])
            return empty_result
        if not os.path.exists(rds_path):
            logger.error("prep.R completed but RDS not created.")
            return empty_result
        logger.info("CopyKAT: RDS ready.")

        cmd = [
            rscript_bin, "--vanilla", runner_script,
            rds_path, output_csv, sam_name, genome, id_type,
            str(ngene_chr), str(win_size), str(ks_cut), distance,
            str(n_cores), str(plot_genes).upper(), str(output_seg).upper(),
        ]
        logger.info("CopyKAT: launching run_copykat.R ...")
        run_result = subprocess.run(
            cmd, capture_output=True, text=True, env=sub_env, cwd=tmpdir,
        )
        for line in run_result.stdout.strip().splitlines():
            logger.info(f"[run_copykat.R] {line}")
        for line in run_result.stderr.strip().splitlines():
            logger.debug(f"[run_copykat.R stderr] {line}")
        if run_result.returncode != 0:
            logger.error(
                f"run_copykat.R exited {run_result.returncode}.\n"
                + "\n".join(run_result.stderr.strip().splitlines()[-30:])
            )
            return empty_result

        if not os.path.exists(output_csv):
            logger.error(f"run_copykat.R completed but CSV not found: {output_csv}")
            return empty_result

        pred_df = pd.read_csv(output_csv)
        logger.info(f"CopyKAT raw prediction columns: {list(pred_df.columns)}")

        if "cell.names" in pred_df.columns:
            barcode_col = "cell.names"
        elif "barcodes" in pred_df.columns:
            barcode_col = "barcodes"
        else:
            pred_df     = pred_df.reset_index()
            barcode_col = pred_df.columns[0]

        pred_col = (
            "copykat.pred" if "copykat.pred" in pred_df.columns
            else pred_df.columns[-1]
        )

        pred_df = pred_df[
            ~pred_df[barcode_col].astype(str).str.startswith("REF_")
        ].copy()
        pred_df = pred_df.rename(columns={
            barcode_col : "barcode",
            pred_col    : "copykat_prediction",
        })[["barcode", "copykat_prediction"]]
        pred_df["copykat_prediction"] = (
            pred_df["copykat_prediction"]
            .astype(str).str.strip().str.lower()
            .replace("nan", "not.defined")
            .fillna("not.defined")
        )

        result_full = pd.DataFrame({"barcode": list(q_barcodes)})
        result_full = result_full.merge(pred_df, on="barcode", how="left")
        result_full["copykat_prediction"] = (
            result_full["copykat_prediction"].fillna("not.defined")
        )

        logger.info(
            "CopyKAT predictions:\n"
            + result_full["copykat_prediction"].value_counts().to_string()
        )
        return result_full

    except Exception as exc:
        logger.error(f"CopyKAT subprocess runner failed: {exc}")
        logger.exception("Full traceback:")
        return empty_result
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_preprocessing_pipeline(
    adata=None,
    popv_path=None,
    log2fc_threshold=1.0,
    pval_adj_threshold=0.05,
    reference_h5ad=None,
    tumor_h5ad=None,
    save_dir=None,
    scmalignant_model_dir=None,
    surfaceome_path=None,
    malignant_strategy="intersection",
    copykat_genome="hg20",
    copykat_id_type="S",
    copykat_ngene_chr=10,
    copykat_win_size=50,
    copykat_ks_cut=0.1,
    copykat_distance="euclidean",
    copykat_n_cores=2,
    copykat_plot_genes=True,
    copykat_output_seg=False,
    copykat_ref_max_cells=100,
    copykat_sam_name="copykat_run",
    copykat_ref_epithelial_key="cell_ontology_class",
    copykat_ref_epithelial_values=None,
):
    """
    Full preprocessing pipeline — see module docstring for details.

    QC thresholds (min_genes, max_mt) are read from adata.uns['qc_params']
    written by Module 1.  If absent, QC is skipped entirely.

    malignant_strategy: 'intersection' | 'scMalignant' | 'copykat'

    CopyKAT runs via subprocess (FIX 10) using the Rscript found in
    dirname(sys.executable) — the correct env in Jupyter kernels.
    """
    if copykat_ref_epithelial_values is None:
        copykat_ref_epithelial_values = ["epithelial cell"]

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
    logger.info(f"Surfaceome (GESP) path: {surfaceome_path}")

    # STEP 1 — Load full PopV h5ad
    if adata is None:
        auto_popv = _auto_popv_h5ad()
        for path in ([popv_path] if popv_path else []) + ([auto_popv] if auto_popv else []):
            if path and os.path.exists(path):
                print(f"Loading PopV output: {path}")
                adata = sc.read_h5ad(path)
                break
        if adata is None:
            raise FileNotFoundError(
                "Could not auto-detect PopV output.\n"
                "Expected: popv_results/final_popv_annotated.h5ad\n"
                "Pass adata= or popv_path= explicitly."
            )

    adata_full = adata.copy()
    print(f"Full dataset loaded: {adata_full.n_obs} cells × {adata_full.n_vars} genes")

    # STEP 2 — Read QC thresholds
    min_genes, max_mt, qc_active, qc_source = _read_qc_params(adata_full)

    # STEP 3 — Extract epithelial cells → QC
    print("\n--- Step 3: Epithelial selection" +
          (" + QC ---" if qc_active else " ---"))

    labels  = adata_full.obs["popv_majority_vote_prediction"].astype(str)
    ep_mask = labels.str.endswith("epithelial cell")
    print(f"Epithelial cells: {ep_mask.sum()} / {adata_full.n_obs} total")
    print(f"Non-epithelial cells (will be 'rest' group for DEG): {(~ep_mask).sum()}")

    adata_epi = adata_full[ep_mask].copy()
    before_qc = adata_epi.n_obs

    if qc_active:
        adata_epi.var["mt"] = adata_epi.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata_epi, qc_vars=["mt"], inplace=True)
        print(f"Mean MT% BEFORE QC: {adata_epi.obs['pct_counts_mt'].mean():.2f}")
        filters = np.ones(adata_epi.n_obs, dtype=bool)
        if min_genes is not None:
            filters &= adata_epi.obs["n_genes_by_counts"] > min_genes
        if max_mt is not None:
            filters &= adata_epi.obs["pct_counts_mt"] < max_mt
        adata_epi = adata_epi[filters].copy()
        filter_desc = "  ".join(
            ([f"min_genes>{min_genes}"] if min_genes is not None else []) +
            ([f"max_mt<{max_mt}"]       if max_mt    is not None else [])
        )
        print(f"Epithelial cells after QC: {adata_epi.n_obs}  "
              f"(removed {before_qc - adata_epi.n_obs}  |  {filter_desc})")
        print(f"Mean MT% AFTER QC:  {adata_epi.obs['pct_counts_mt'].mean():.2f}\n")
    else:
        print(f"QC filtering SKIPPED — all {adata_epi.n_obs} epithelial cells proceed.\n")

    print("Detecting raw count source for epithelial cells...")
    for lyr in ("scvi_counts", "raw_counts", "counts"):
        if lyr in adata_epi.layers:
            print(f"  Using layers['{lyr}'] as raw counts.")
            adata_epi.X = adata_epi.layers[lyr].copy()
            break
    else:
        if adata_epi.raw is not None:
            print("  Using adata.raw.X as raw counts.")
            adata_epi.X = adata_epi.raw.X.copy()
        else:
            print("  No raw layer — assuming .X is raw counts.")

    adata_epi.var_names_make_unique()
    adata_epi.layers["raw_for_cna"] = adata_epi.X.copy()
    sc.pp.normalize_total(adata_epi, target_sum=1e4)
    sc.pp.log1p(adata_epi)

    # STEP 4a — scMalignantFinder
    print("\n--- Step 4a: scMalignantFinder ---")
    feature_tsv = os.path.join(scmalignant_model_dir, "ordered_feature.tsv")

    if "full_counts" in adata_epi.layers and adata_epi.uns.get("full_counts_var_names"):
        print(f"Gene-space: Route A-new ({len(adata_epi.uns['full_counts_var_names'])} genes)")
    else:
        rescue = tumor_h5ad or _auto_tumor_h5ad()
        if rescue and os.path.exists(rescue):
            print(f"Gene-space: Route A-rescue ({rescue})")
        elif adata_epi.raw is not None:
            print(f"Gene-space: Route A-old (adata.raw, {adata_epi.raw.n_vars} genes)")
        else:
            print("Gene-space: Route C (4000-HVG fallback)")

    adata_scm = _build_fullgene_adata_for_scm(adata_epi, feature_tsv, tumor_h5ad)
    print(f"  Gene space used: {adata_scm.n_vars} genes")

    import importlib.util as _ilu

    _scm_pkg_dir      = os.path.dirname(scmalignant_model_dir)
    _scm_external_dir = os.path.dirname(_scm_pkg_dir)
    _classifier_py    = os.path.join(_scm_pkg_dir, "classifier.py")
    _init_py          = os.path.join(_scm_pkg_dir, "__init__.py")

    def _load_module_from_file(mod_name, filepath):
        if mod_name in sys.modules:
            return sys.modules[mod_name]
        _spec = _ilu.spec_from_file_location(mod_name, filepath)
        _mod  = _ilu.module_from_spec(_spec)
        sys.modules[mod_name] = _mod
        _spec.loader.exec_module(_mod)
        return _mod

    _clf_mod = None
    if os.path.isfile(_classifier_py):
        logger.info(f"scMalignantFinder: loading via classifier.py ({_classifier_py})")
        _clf_mod = _load_module_from_file("scMalignantFinder.classifier", _classifier_py)
    elif os.path.isfile(_init_py):
        logger.info(f"scMalignantFinder: loading via __init__.py ({_init_py})")
        _clf_mod = _load_module_from_file("scMalignantFinder", _init_py)
    else:
        logger.warning(f"scMalignantFinder not found in {_scm_pkg_dir} — sys.path fallback.")
        _inserted = _scm_external_dir not in sys.path
        if _inserted:
            sys.path.insert(0, _scm_external_dir)
        try:
            import scMalignantFinder as _scm_pkg
            _clf_mod = _scm_pkg
        except ImportError as _exc:
            raise ImportError(
                f"Could not import scMalignantFinder.\n"
                f"  classifier.py: {_classifier_py}\n"
                f"  __init__.py:   {_init_py}\n"
                f"  Error: {_exc}"
            ) from _exc
        finally:
            if _inserted and _scm_external_dir in sys.path:
                sys.path.remove(_scm_external_dir)

    # pretrain_dir = directory with model.joblib + ordered_feature.tsv
    # norm_type=False: adata_scm already log-normalised by _build_fullgene_adata_for_scm
    model = _clf_mod.scMalignantFinder(
        test_input          = adata_scm,
        celltype_annotation = False,
        pretrain_dir        = scmalignant_model_dir,
        feature_path        = feature_tsv,
        norm_type           = False,
    )
    model.load()
    result_scm = model.predict()

    # FIX 6 — align by obs_names
    scm_col = "scMalignantFinder_prediction"
    if result_scm.obs_names.equals(adata_epi.obs_names):
        adata_epi.obs[scm_col] = result_scm.obs[scm_col].values
    else:
        logger.warning("scMalignantFinder obs_names differ — aligning by index.")
        adata_epi.obs[scm_col] = (
            result_scm.obs[scm_col]
            .reindex(adata_epi.obs_names)
            .fillna("Unknown")
            .values
        )

    print("scMalignantFinder completed:")
    print(adata_epi.obs[scm_col].value_counts().to_string())

    # STEP 4b — CopyKAT (subprocess, FIX 10)
    copykat_available = False
    copykat_result_df = None

    if malignant_strategy in ("copykat", "intersection"):
        if reference_h5ad is None:
            print("\nWarning: CopyKAT skipped — no reference_h5ad provided.\n"
                  "  Falling back to scMalignantFinder only.")
            malignant_strategy = "scMalignant"
        else:
            _rs = _find_rscript()
            _rh = _get_r_home(_rs) if _rs else "NOT FOUND"
            print(
                f"\n--- Step 4b: CopyKAT (subprocess) ---\n"
                f"  Reference:           {reference_h5ad}\n"
                f"  Genome:              {copykat_genome}\n"
                f"  id.type:             {copykat_id_type}\n"
                f"  ngene.chr:           {copykat_ngene_chr}\n"
                f"  win.size:            {copykat_win_size}\n"
                f"  KS.cut:              {copykat_ks_cut}\n"
                f"  distance:            {copykat_distance}\n"
                f"  n.cores:             {copykat_n_cores}\n"
                f"  plot.genes:          {copykat_plot_genes}\n"
                f"  output.seg:          {copykat_output_seg}\n"
                f"  ref_max_cells:       {copykat_ref_max_cells}\n"
                f"  sam_name:            {copykat_sam_name}\n"
                f"  ref_epithelial_key:  {copykat_ref_epithelial_key}\n"
                f"  ref_epithelial_vals: {copykat_ref_epithelial_values}\n"
                f"  Rscript:             {_rs}\n"
                f"  R home:              {_rh}"
            )

            try:
                adata_raw_cna   = adata_epi.copy()
                adata_raw_cna.X = adata_epi.layers["raw_for_cna"]
                adata_ref_full  = sc.read_h5ad(reference_h5ad)

                copykat_result_df = _run_copykat(
                    adata_query           = adata_raw_cna,
                    adata_ref             = adata_ref_full,
                    genome                = copykat_genome,
                    id_type               = copykat_id_type,
                    ngene_chr             = copykat_ngene_chr,
                    win_size              = copykat_win_size,
                    ks_cut                = copykat_ks_cut,
                    distance              = copykat_distance,
                    n_cores               = copykat_n_cores,
                    plot_genes            = copykat_plot_genes,
                    output_seg            = copykat_output_seg,
                    ref_max_cells         = copykat_ref_max_cells,
                    sam_name              = copykat_sam_name,
                    ref_epithelial_key    = copykat_ref_epithelial_key,
                    ref_epithelial_values = copykat_ref_epithelial_values,
                )

                bc_to_pred = dict(zip(
                    copykat_result_df["barcode"],
                    copykat_result_df["copykat_prediction"],
                ))
                adata_epi.obs["copykat_prediction"] = [
                    bc_to_pred.get(b, "not.defined") for b in adata_epi.obs_names
                ]
                copykat_available = True
                print("\nCopyKAT completed.")
                print("  Prediction counts:")
                print(adata_epi.obs["copykat_prediction"].value_counts().to_string())

            except Exception as exc:
                print(f"\nWarning: CopyKAT failed — {type(exc).__name__}: {exc}\n"
                      "  Falling back to scMalignantFinder only.")
                logger.exception("CopyKAT error:")
                malignant_strategy = "scMalignant"

    # STEP 4c — Combine malignancy calls
    print("\n--- Step 4c: Combine malignancy calls ---")
    scm_mal = adata_epi.obs[scm_col].str.lower() == "malignant"

    if copykat_available:
        ck_mal = adata_epi.obs["copykat_prediction"].str.lower() == "aneuploid"
        if malignant_strategy == "intersection":
            malignant_mask = scm_mal & ck_mal
            strategy_label = "intersection (scMalignantFinder AND CopyKAT)"
        elif malignant_strategy == "copykat":
            malignant_mask = ck_mal
            strategy_label = "CopyKAT only"
        else:
            malignant_mask = scm_mal
            strategy_label = "scMalignantFinder only"
    else:
        malignant_mask = scm_mal
        strategy_label = "scMalignantFinder only"

    adata_epi.obs["final_malignant"] = malignant_mask.map(
        {True: "malignant", False: "non-malignant"}
    )
    print(f"Strategy: {strategy_label}")
    print(f"  Malignant epithelial:     {malignant_mask.sum()}")
    print(f"  Non-malignant epithelial: {(~malignant_mask).sum()}  ← these will be REMOVED")

    # STEP 5 — Keep ONLY malignant epithelial cells
    print("\n--- Step 5: Retain malignant epithelial cells only ---")
    adata_mal = adata_epi[malignant_mask].copy()
    print(f"Malignant epithelial cells retained: {adata_mal.n_obs}")
    print(f"Non-malignant epithelial cells removed: {(~malignant_mask).sum()}")

    if adata_mal.n_obs == 0:
        raise ValueError(
            "No malignant cells found after filtering.\n"
            "Check scMalignantFinder model path and gene overlap."
        )

    # STEP 6 — Non-epithelial "rest" group
    print("\n--- Step 6: Extract non-epithelial 'rest' group ---")
    rest_mask  = ~ep_mask
    adata_rest = adata_full[rest_mask].copy()
    print(f"Non-epithelial 'rest' cells: {adata_rest.n_obs}")

    for lyr in ("scvi_counts", "raw_counts", "counts"):
        if lyr in adata_rest.layers:
            adata_rest.X = adata_rest.layers[lyr].copy()
            break
    else:
        if adata_rest.raw is not None:
            adata_rest.X = adata_rest.raw.X.copy()
    sc.pp.normalize_total(adata_rest, target_sum=1e4)
    sc.pp.log1p(adata_rest)

    # STEP 7 — Surfaceome filter
    print("\n--- Step 7: Surfaceome filter (GESP file) ---")
    surfaceome        = pd.read_csv(surfaceome_path)
    surfaceome.columns = surfaceome.columns.str.strip()
    surf_genes        = surfaceome["Gene"].astype(str).tolist()
    print(f"Surfaceome genes in GESP file: {len(surf_genes)}")

    surf_in_mal  = adata_mal.var_names.intersection(surf_genes)
    adata_mal    = adata_mal[:, surf_in_mal].copy()
    print(f"Surfaceome genes in malignant cells: {len(surf_in_mal)}")

    surf_in_rest = adata_rest.var_names.intersection(surf_genes)
    adata_rest   = adata_rest[:, surf_in_rest].copy()
    print(f"Surfaceome genes in rest cells: {len(surf_in_rest)}")

    surf_common = surf_in_mal.intersection(surf_in_rest)
    adata_mal   = adata_mal[:, surf_common].copy()
    adata_rest  = adata_rest[:, surf_common].copy()
    print(f"Common surfaceome genes (used for DEG): {len(surf_common)}\n")

    # STEP 8 — DEG
    print("--- Step 8: DEG — malignant epithelial vs non-epithelial rest ---")

    adata_mal.obs["deg_group"]  = "malignant_epithelial"
    adata_rest.obs["deg_group"] = "non_epithelial_rest"

    adata_deg = sc.concat([adata_mal, adata_rest], join="outer", label=None)
    adata_deg.obs_names_make_unique()
    adata_deg.var = adata_mal.var.copy()

    if sp.issparse(adata_deg.X):
        adata_deg.X = adata_deg.X.toarray()
    adata_deg.X = np.nan_to_num(np.array(adata_deg.X, dtype=np.float32), nan=0.0)
    adata_deg.X = sp.csr_matrix(adata_deg.X)

    print(f"DEG AnnData: {adata_deg.n_obs} cells × {adata_deg.n_vars} genes")
    print(f"  malignant_epithelial: "
          f"{(adata_deg.obs['deg_group'] == 'malignant_epithelial').sum()}")
    print(f"  non_epithelial_rest:  "
          f"{(adata_deg.obs['deg_group'] == 'non_epithelial_rest').sum()}")

    sc.tl.rank_genes_groups(
        adata_deg,
        groupby="deg_group",
        groups=["malignant_epithelial"],
        reference="non_epithelial_rest",
        method="wilcoxon",
        key_added="rank_genes_groups",
    )
    deg = sc.get.rank_genes_groups_df(adata_deg, group="malignant_epithelial")
    print(f"\nTotal DEG candidates: {deg.shape[0]}")
    print(f"Applying filters: log2FC > {log2fc_threshold}, pvals_adj < {pval_adj_threshold}")

    filtered_deg = deg[
        (deg["logfoldchanges"] > log2fc_threshold) &
        (deg["pvals_adj"]      < pval_adj_threshold)
    ]

    if filtered_deg.shape[0] == 0:
        print("WARNING: 0 DEGs passed the filter.\n"
              "  Try: log2fc_threshold=0.5 or pval_adj_threshold=0.10")
    else:
        print(f"DEGs retained: {filtered_deg.shape[0]}")
        print("\nTop 10 DEGs (malignant epithelial vs non-epithelial rest):")
        print(filtered_deg.head(10).to_string(index=False))

    # STEP 9 — Binarise, store, save
    print("\n--- Step 9: Binarise malignant cells and save ---")

    adata_mal.X = (
        np.array(
            adata_mal.X.toarray() if sp.issparse(adata_mal.X) else adata_mal.X
        ) > 0
    ).astype(np.int8)
    adata_mal.X = sp.csr_matrix(adata_mal.X)
    print("Expression converted to binary (0/1).")

    adata_mal.uns["filtered_deg"] = filtered_deg.reset_index(drop=True)
    adata_mal.uns["all_deg"]      = deg.reset_index(drop=True)
    adata_mal.uns["deg_params"]   = {
        "comparison"         : "malignant_epithelial vs non_epithelial_rest",
        "log2fc_threshold"   : log2fc_threshold,
        "pval_adj_threshold" : pval_adj_threshold,
        "method"             : "wilcoxon",
        "n_malignant"        : int(adata_mal.n_obs),
        "n_rest"             : int(adata_rest.n_obs),
        "n_surfaceome_genes" : int(len(surf_common)),
        "n_filtered_deg"     : int(filtered_deg.shape[0]),
    }
    adata_mal.uns["qc_params"] = (
        {"min_genes": min_genes, "max_mt": max_mt} if qc_active else None
    )

    if copykat_result_df is not None:
        final_barcodes = set(adata_mal.obs_names)
        copykat_stored = copykat_result_df.copy()
        copykat_stored["in_final_output"] = copykat_stored["barcode"].isin(final_barcodes)
        adata_mal.uns["copykat_results"] = copykat_stored
        print(
            f"\nCopyKAT results stored in adata.uns['copykat_results']:\n"
            f"  Shape: {copykat_stored.shape[0]} rows × {copykat_stored.shape[1]} columns\n"
            f"  Columns: {list(copykat_stored.columns)}\n"
            f"  Prediction summary:\n"
            + copykat_stored["copykat_prediction"].value_counts().to_string()
            + f"\n  Cells in final output: {copykat_stored['in_final_output'].sum()}"
        )
    else:
        adata_mal.uns["copykat_results"] = None
        print("\nCopyKAT was not run — adata.uns['copykat_results'] = None")

    for col in adata_mal.obs.columns:
        if adata_mal.obs[col].dtype == object:
            adata_mal.obs[col] = adata_mal.obs[col].astype(str)

    final_path = os.path.join(save_dir, "final_tumor.h5ad")
    adata_mal.write(final_path)
    print(f"\nFinal object saved to: {final_path}")
    print("\n========== PREPROCESSING COMPLETED ==========\n")
    return adata_mal
