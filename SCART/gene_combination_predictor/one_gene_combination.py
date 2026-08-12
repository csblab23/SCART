#!/usr/bin/env python
# coding: utf-8
"""
one_gene_combination.py
Module 4a — Single-gene CAR-T target evaluation

Evaluates every surface gene individually against tumour and healthy cells.

Healthy reference atlases
--------------------------
Two ready-to-use healthy single-cell reference atlases are used to score
safety:

  "hpa"    -> hpa_alltissues_geosketch_10k.h5ad
  "tabula" -> tabula_sapiens_alltissues_10k.h5ad

These are NOT bundled in the SCART package or GitHub repo — they are
distributed separately via Zenodo (see the SCART documentation for the
record link). Download them yourself and tell SCART where you put them:

  1. Pass an explicit path per atlas:
       run(atlas="both", hpa_path="/path/to/hpa_alltissues_geosketch_10k.h5ad",
                          tabula_path="/path/to/tabula_sapiens_alltissues_10k.h5ad")
  2. OR place the files (using their original filenames, unchanged) in
     one of these auto-detected locations and omit hpa_path/tabula_path:
       <current working directory>/hpa_alltissues_geosketch_10k.h5ad
       <current working directory>/healthy_atlases/hpa_alltissues_geosketch_10k.h5ad
       (same pattern for the Tabula Sapiens file)

The user selects which atlas(es) to score safety against via the `atlas`
argument of run():

  atlas="hpa"    -> evaluate every gene against HPA only. Returns a single
                     results DataFrame (unchanged behaviour from before).
  atlas="tabula" -> evaluate every gene against Tabula Sapiens only.
                     Returns a single results DataFrame.
  atlas="both"   -> evaluate every gene against EACH atlas independently,
                     save individual per-atlas results, then combine the
                     two ranked gene lists with Robust Rank Aggregation
                     (RRA). Returns a dict with both individual results
                     plus the RRA table.

A custom healthy matrix source is still supported per-atlas via hpa_path=/
tabula_path= (.h5ad or .tsv/.tsv.gz — same priority order as before,
via _load_healthy_matrix).

Fix applied
-----------
_load_h5ad_subset: h5py dense-dataset fancy indexing requires col_indices in
STRICTLY INCREASING order.  The original code passed indices in the order of
target_genes (caller order), which is almost never sorted.  h5py raised:
  TypeError: Indexing elements must be in increasing order

Fix: sort col_indices before h5py read, then un-permute columns to restore
the caller's gene order.  Sparse (CSR/CSC) paths are unaffected — scipy
sparse accepts unsorted column indices — but they also explicitly use the
raw (unsorted) indices for correctness.

Fix applied (gene symbols)
---------------------------
_load_h5ad_subset previously always used var/_index (var_names) as the gene
identifier. That is correct for the HPA atlas (its var_names already are
HGNC gene symbols), but NOT for the Tabula Sapiens atlas, whose var/_index
holds a different identifier while the actual HGNC symbol lives in the
var['gene_symbol'] column. This caused a "No overlap between target genes
and h5ad var_names" ValueError for Tabula Sapiens even though the genes are
present under a different column.

Fix: a new _read_var_column() helper reads a var/ column (handling both a
plain array dataset and pandas' categorical group encoding — categories +
codes), and _load_h5ad_subset now prefers var['gene_symbol'] when present,
falling back to var/_index exactly as before when it isn't.

Fix applied (rpy2 / R environment — atlas="both" RRA step)
------------------------------------------------------------
_run_rra_via_r previously called `import rpy2.robjects` with no control
over which R installation rpy2 binds to. On a machine with more than one R
on the system (e.g. a system R alongside the conda env's own R), rpy2 can
resolve a libR.so that does NOT match the R the RobustRankAggreg package
was actually installed into, or can load it without the R_HOME/lib
directory on the loader's search path. Both produce hard-to-read failures
during `import rpy2.rinterface`, e.g.:
  ImportError: .../_rinterface_cffi_api.abi3.so: undefined symbol: R_ClosureEnv
  (falls back to ABI mode, which then also fails:)
  error: symbol 'R_getVar' not found in library '.../lib/R/lib/libR.so'

This is an environment-resolution problem, not a code bug in the RRA logic
itself — rpy2 found *a* libR.so but the wrong/partially-linked one relative
to LD_LIBRARY_PATH at import time (rpy2.situation's own diagnostics showed
"R library path" containing only an unrelated CUDA lib64 entry, with no
R lib directory in LD_LIBRARY_PATH at all).

Fix: a new _setup_r_environment() (backed by _find_r_home()) explicitly
resolves R_HOME — preferring the R that lives inside the active conda
environment (sys.prefix/lib/R) over PATH/system R — and prepends
<R_HOME>/lib to LD_LIBRARY_PATH *before* rpy2 is imported anywhere in the
process. This mirrors the R-environment resolution already used
successfully in Module 4b (two_gene_combination.py)'s RRA step, and prints
the same Rscript/R home diagnostic line for consistency:
  Rscript: <R_HOME>/bin/Rscript
  R home:  <R_HOME>

IMPORTANT: because Python caches failed imports in sys.modules, if rpy2 has
already been imported once in the current process/kernel and failed as
above, this fix will NOT retroactively repair it — restart the Python
kernel/process before re-running so the corrected environment variables are
in place *before* rpy2 is imported for the first time.
"""

import os
import zipfile
import urllib.request
import logging

import numpy as np
import pandas as pd
import scanpy as sc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HPA_ZIP_URL = "https://www.proteinatlas.org/download/tsv/rna_single_cell_read_count.zip"
HPA_CACHE   = os.path.join(os.getcwd(), "hpa_cache", "rna_single_cell_read_count.tsv")


def _auto_tumor_h5ad() -> str:
    search = [
        os.path.join(os.getcwd(), "preprocessing_results", "final_tumor.h5ad"),
        os.path.join(os.getcwd(), "final_tumor.h5ad"),
    ]
    for path in search:
        if os.path.exists(path):
            logger.info(f"Auto-detected tumour h5ad: {path}")
            return path
    raise FileNotFoundError(
        "Could not auto-detect final_tumor.h5ad.\n"
        "Expected:\n"
        "  <cwd>/preprocessing_results/final_tumor.h5ad\n"
        "  <cwd>/final_tumor.h5ad\n"
        "Pass tumor_path= explicitly if saved elsewhere."
    )


def _download_hpa(cache_path: str) -> str:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    zip_path = cache_path.replace(".tsv", ".zip")

    if os.path.exists(cache_path):
        logger.info(f"HPA cache found: {cache_path}")
        return cache_path

    print(f"Downloading HPA single-cell read counts from:\n  {HPA_ZIP_URL}")
    urllib.request.urlretrieve(HPA_ZIP_URL, zip_path)
    print("Download complete. Extracting...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        tsv_names = [n for n in zf.namelist() if n.endswith(".tsv")]
        if not tsv_names:
            raise FileNotFoundError("No TSV found inside HPA zip archive.")
        zf.extract(tsv_names[0], os.path.dirname(cache_path))
        extracted = os.path.join(os.path.dirname(cache_path), tsv_names[0])
        if extracted != cache_path:
            os.rename(extracted, cache_path)

    os.remove(zip_path)
    print(f"HPA TSV saved to: {cache_path}")
    return cache_path


def _hpa_tsv_to_binary_matrix(tsv_path: str):
    print(f"Reading HPA TSV: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    df.columns = df.columns.str.strip()
    col_map   = {c.lower(): c for c in df.columns}
    gene_col  = col_map.get("gene name", col_map.get("gene", None))
    cell_col  = col_map.get("cell type", col_map.get("cell_type", None))
    count_col = col_map.get("read count", col_map.get("tpm", col_map.get("ntpm", None)))

    missing = [n for n, c in [("Gene", gene_col), ("Cell type", cell_col),
                               ("Read count", count_col)] if c is None]
    if missing:
        raise ValueError(
            f"HPA TSV missing expected columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df[[gene_col, cell_col, count_col]].copy()
    df.columns = ["gene", "cell_type", "count"]
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)

    pivot  = df.pivot_table(index="cell_type", columns="gene", values="count",
                            aggfunc="sum", fill_value=0)
    matrix = (pivot.values > 0).astype(np.int8)
    genes  = list(pivot.columns)
    cells  = list(pivot.index)

    print(f"HPA matrix built: {len(cells)} cell types x {len(genes)} genes")
    return matrix, genes, cells


def _read_var_column(var_grp, key):
    """
    Read a single column out of an h5ad 'var' h5py group, handling both
    storage layouts AnnData/h5py can use:

      - Plain array dataset (var_grp[key] is an h5py.Dataset).
      - Pandas categorical column, stored as a sub-group with 'categories'
        and 'codes' datasets (var_grp[key] is an h5py.Group).

    Returns a list[str] (decoding bytes -> str as needed). Categorical
    codes of -1 (missing) map to None.
    """
    import h5py

    node = var_grp[key]
    if isinstance(node, h5py.Group):
        categories = [c.decode() if isinstance(c, bytes) else c
                      for c in node["categories"][:]]
        codes = node["codes"][:]
        return [categories[c] if c >= 0 else None for c in codes]
    else:
        return [g.decode() if isinstance(g, bytes) else g for g in node[:]]


def _load_h5ad_subset(h5ad_path: str, target_genes: list = None):
    """
    Fast, memory-safe h5ad loader.

    Reads only the required gene columns from the X matrix via h5py —
    avoids loading the full matrix into RAM.

    FIX: col_indices are sorted before h5py indexing (h5py requires strictly
    increasing order for dense datasets), then columns are un-permuted to
    restore the caller's gene order.

    FIX (gene symbols): gene identifiers are read from var['gene_symbol']
    when that column exists (HGNC symbols), falling back to var/_index
    otherwise — see module docstring "Fix applied (gene symbols)".

    Returns: matrix (int8 ndarray), genes (list[str])
    """
    import h5py
    import scipy.sparse as _sp

    if target_genes is None:
        adata = sc.read_h5ad(h5ad_path)
        X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
        if "gene_symbol" in adata.var.columns:
            genes = list(adata.var["gene_symbol"])
        else:
            genes = list(adata.var_names)
        return (X > 0).astype(np.int8), genes

    with h5py.File(h5ad_path, "r") as f:

        # Read var names — prefer the 'gene_symbol' column (HGNC symbols)
        # when present, since var_names/_index is not guaranteed to be a
        # gene symbol (e.g. Tabula Sapiens indexes on a different ID, while
        # HPA already indexes on gene symbols). Falls back to _index/first
        # column exactly as before when 'gene_symbol' isn't available.
        if "var" in f:
            var_grp = f["var"]
            if "gene_symbol" in var_grp:
                all_genes = _read_var_column(var_grp, "gene_symbol")
            elif "_index" in var_grp:
                all_genes = [g.decode() if isinstance(g, bytes) else g
                             for g in var_grp["_index"][:]]
            else:
                key = list(var_grp.keys())[0]
                all_genes = [g.decode() if isinstance(g, bytes) else g
                             for g in var_grp[key][:]]
        else:
            raise ValueError(f"No 'var' group found in {h5ad_path}")

        gene_set   = set(all_genes)
        gene_index = {g: i for i, g in enumerate(all_genes)}
        common     = [g for g in target_genes if g in gene_set]

        if len(common) == 0:
            raise ValueError(
                f"No overlap between target genes and h5ad var_names in {h5ad_path}.\n"
                "Check both datasets use HGNC gene symbols."
            )

        # Raw indices — in caller's (target_genes) order
        raw_col_indices = np.array([gene_index[g] for g in common], dtype=np.int32)

        # FIX: sort for h5py dense indexing; track permutation to restore order
        sort_order     = np.argsort(raw_col_indices)
        sorted_indices = raw_col_indices[sort_order]    # strictly increasing
        restore_order  = np.argsort(sort_order)         # inverse permutation

        print(f"  HPA h5ad: {len(all_genes)} genes total — "
              f"extracting {len(common)} overlapping genes directly via h5py.")

        x_grp = f["X"]

        if isinstance(x_grp, h5py.Dataset):
            # Dense dataset — MUST use sorted indices
            X_sorted = x_grp[:, sorted_indices]
            X_sub    = X_sorted[:, restore_order]       # restore caller order

        elif isinstance(x_grp, h5py.Group):
            encoding = x_grp.attrs.get("encoding-type", b"").decode() \
                if isinstance(x_grp.attrs.get("encoding-type", ""), bytes) \
                else x_grp.attrs.get("encoding-type", "")

            if "csr" in encoding or all(k in x_grp for k in ("data", "indices", "indptr")):
                data    = x_grp["data"][:]
                indices = x_grp["indices"][:]
                indptr  = x_grp["indptr"][:]
                shape   = tuple(x_grp.attrs["shape"])
                full    = _sp.csr_matrix((data, indices, indptr), shape=shape)
                # scipy sparse accepts unsorted column indices
                X_sub   = full[:, raw_col_indices].toarray()

            elif "csc" in encoding:
                data    = x_grp["data"][:]
                indices = x_grp["indices"][:]
                indptr  = x_grp["indptr"][:]
                shape   = tuple(x_grp.attrs["shape"])
                full    = _sp.csc_matrix((data, indices, indptr), shape=shape)
                X_sub   = full[:, raw_col_indices].toarray()

            else:
                logger.warning(
                    "Unknown X encoding — falling back to scanpy backed mode."
                )
                adata_backed = sc.read_h5ad(h5ad_path, backed="r")
                adata_sub    = adata_backed[:, common].to_memory()
                adata_backed.file.close()
                X_full = adata_sub.X
                X_sub  = X_full.toarray() if _sp.issparse(X_full) else np.asarray(X_full)
        else:
            raise ValueError(f"Unrecognised X format in {h5ad_path}")

    return (X_sub > 0).astype(np.int8), common


def _load_healthy_matrix(hpa_path=None, target_genes=None):
    """
    Load and binarise the healthy/normal expression matrix.

    target_genes: list[str] or None
        When provided, only these genes are loaded from h5ad files —
        avoids OOM on large reference files.

    Priority:
      1. hpa_path ends with .h5ad  -> memory-safe load
      2. hpa_path ends with .tsv   -> parse as HPA TSV
      3. hpa_path = None           -> auto-download HPA TSV
      4. Fallback                  -> legacy final_healthy.h5ad

    Returns: matrix (int8), genes (list[str]), source (str)
    """
    if hpa_path and hpa_path.endswith(".h5ad"):
        if not os.path.exists(hpa_path):
            raise FileNotFoundError(f"Provided HPA h5ad not found: {hpa_path}")
        print(f"Loading user-supplied healthy h5ad: {hpa_path}")
        matrix, genes = _load_h5ad_subset(hpa_path, target_genes)
        return matrix, genes, f"user h5ad: {hpa_path}"

    if hpa_path and (hpa_path.endswith(".tsv") or hpa_path.endswith(".tsv.gz")):
        if not os.path.exists(hpa_path):
            raise FileNotFoundError(f"Provided HPA TSV not found: {hpa_path}")
        matrix, genes, _ = _hpa_tsv_to_binary_matrix(hpa_path)
        return matrix, genes, f"user TSV: {hpa_path}"

    if hpa_path is None:
        print("No HPA file provided — downloading from proteinatlas.org ...")
        tsv    = _download_hpa(HPA_CACHE)
        matrix, genes, _ = _hpa_tsv_to_binary_matrix(tsv)
        return matrix, genes, "auto-downloaded HPA"

    for legacy in [
        os.path.join(os.getcwd(), "preprocessing_results", "final_healthy.h5ad"),
        os.path.join(os.getcwd(), "final_healthy.h5ad"),
    ]:
        if os.path.exists(legacy):
            print(f"Using legacy healthy h5ad: {legacy}")
            adata = sc.read_h5ad(legacy)
            X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
            return (X > 0).astype(np.int8), list(adata.var_names), f"legacy: {legacy}"

    raise FileNotFoundError(
        "No healthy/HPA matrix available.\n"
        "Provide hpa_path= or ensure final_healthy.h5ad exists."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Healthy reference atlases — distributed via Zenodo (NOT bundled with SCART)
# ─────────────────────────────────────────────────────────────────────────────

# Canonical filenames of the two Zenodo-hosted healthy reference atlases.
# Users download these themselves; SCART never ships them.
ATLAS_FILES = {
    "hpa":    "hpa_alltissues_geosketch_10k.h5ad",
    "tabula": "tabula_sapiens_alltissues_10k.h5ad",
}

ATLAS_LABELS = {
    "hpa":    "HPA (all-tissues, geosketch 10k)",
    "tabula": "Tabula Sapiens (all-tissues, 10k)",
}


def _default_atlas_search_dirs() -> list:
    """Local directories auto-searched for a Zenodo-downloaded atlas file
    when the user does not pass an explicit hpa_path=/tabula_path=."""
    cwd = os.getcwd()
    return [
        cwd,
        os.path.join(cwd, "healthy_atlases"),
    ]


def _resolve_atlas_path(atlas_key: str, explicit_path: str = None) -> str:
    """
    Resolve the local path to a healthy reference atlas.

      - If explicit_path is given, it is used as-is (must exist).
      - Otherwise, the canonical filename for `atlas_key` is searched for in
        _default_atlas_search_dirs().
      - If not found anywhere, raises FileNotFoundError with download +
        placement instructions (the files are distributed via Zenodo, not
        bundled with SCART).
    """
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(
                f"Provided {atlas_key} atlas path does not exist: {explicit_path}"
            )
        return explicit_path

    fname = ATLAS_FILES[atlas_key]
    for d in _default_atlas_search_dirs():
        candidate = os.path.join(d, fname)
        if os.path.exists(candidate):
            print(f"Auto-detected {atlas_key} atlas file: {candidate}")
            return candidate

    search_dirs_str = "\n".join(f"    {os.path.join(d, fname)}" for d in _default_atlas_search_dirs())
    raise FileNotFoundError(
        f"Could not find the '{atlas_key}' healthy reference atlas "
        f"('{fname}').\n\n"
        f"This file is not bundled with SCART — it is distributed "
        f"separately via Zenodo (see the SCART documentation for the "
        f"record link).\n\n"
        f"To use it:\n"
        f"  1. Download '{fname}' from Zenodo.\n"
        f"  2. Either:\n"
        f"       a) pass its path explicitly, e.g.:\n"
        f"            run(atlas=..., {atlas_key}_path='/path/to/{fname}')\n"
        f"       b) OR save it (keeping the exact filename above) into one "
        f"of these auto-detected locations:\n"
        f"{search_dirs_str}"
    )


def evaluate_single_gene(gene_idx, tumor_matrix, healthy_matrix):
    tumor_expr   = tumor_matrix[:, gene_idx]
    healthy_expr = healthy_matrix[:, gene_idx]
    efficacy = np.sum(tumor_expr)            / len(tumor_expr)
    safety   = np.sum(healthy_expr == 0)     / len(healthy_expr)
    return efficacy, safety


# ─────────────────────────────────────────────────────────────────────────────
# Single-atlas evaluation  (original run() body, factored out so it can be
# executed once per atlas when atlas="both"; unchanged logic otherwise)
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_single_atlas(
    atlas_label: str,
    healthy_path: str,
    adata_tumor,
    tumor_genes: list,
    safety_threshold: float,
    output_dir: str,
) -> pd.DataFrame:
    """
    Evaluate every gene against a single healthy atlas.

    Output CSV is suffixed with the atlas label so that atlas="both" runs
    do not overwrite each other: single_gene_results_<atlas_label>.csv

    Returns df_results (all genes, columns: Gene, Efficacy, Safety,
    ObjectiveScore) for this atlas.
    """
    print(f"\nLoading healthy matrix for atlas '{atlas_label}': {healthy_path}")
    healthy_matrix_full, healthy_genes, healthy_source = _load_healthy_matrix(
        healthy_path, target_genes=tumor_genes
    )
    print(f"Healthy matrix source ({atlas_label}): {healthy_source}")

    common_genes = sorted(set(tumor_genes) & set(healthy_genes))
    if len(common_genes) == 0:
        raise ValueError(
            f"No common genes between tumour and healthy ({atlas_label}) matrices.\n"
            "Check both datasets use HGNC gene symbols."
        )
    print(f"Common genes ({atlas_label}): {len(common_genes)}")

    # Tumour subset
    adata_sub    = adata_tumor[:, common_genes].copy()
    X_tumor      = adata_sub.X.toarray() if not isinstance(adata_sub.X, np.ndarray) else adata_sub.X
    tumor_matrix = (X_tumor > 0).astype(np.int8)

    # Healthy subset — reindex to common_genes order
    hg_idx         = {g: i for i, g in enumerate(healthy_genes)}
    col_idx        = np.array([hg_idx[g] for g in common_genes])
    healthy_matrix = healthy_matrix_full[:, col_idx]

    n_genes    = len(common_genes)
    gene_names = common_genes

    print(f"Tumour matrix  ({atlas_label}): {tumor_matrix.shape[0]} cells x {n_genes} genes")
    print(f"Healthy matrix ({atlas_label}): {healthy_matrix.shape[0]} samples x {n_genes} genes")
    print(f"Starting single-gene analysis ({atlas_label})...")

    results = []
    tick    = max(1, n_genes // 100)

    for idx in range(n_genes):
        efficacy, safety = evaluate_single_gene(idx, tumor_matrix, healthy_matrix)
        objective_score  = efficacy if safety >= safety_threshold else 0
        results.append([gene_names[idx], efficacy, safety, objective_score])
        if (idx + 1) % tick == 0:
            print(f"\r[{atlas_label}] Progress: {(idx+1)/n_genes*100:.1f}% completed", end="")

    print(f"\n[{atlas_label}] Analysis completed!")

    df_results  = pd.DataFrame(results, columns=["Gene", "Efficacy", "Safety", "ObjectiveScore"])
    output_file = os.path.join(output_dir, f"single_gene_results_{atlas_label}.csv")
    df_results[["Gene", "Efficacy", "Safety"]].to_csv(
        output_file, index=False, header=["gene", "efficacy", "safety"]
    )
    print(f"[{atlas_label}] Results saved to: {output_file}")

    df_top = (
        df_results[df_results["Safety"] >= safety_threshold]
        .sort_values(by="Efficacy", ascending=False)
        .head(10)
    )
    print(f"\n[{atlas_label}] Top 10 single-gene candidates (safety >= {safety_threshold}):")
    print(df_top[["Gene", "Efficacy", "Safety"]].to_string(index=False))

    return df_results


# ─────────────────────────────────────────────────────────────────────────────
# Robust Rank Aggregation across the two atlases
#
# Same concept as two_gene_combination.py's RRA step, reduced to a single
# gene identifier instead of a (geneA, geneB, gate) triple: candidates must
# pass efficacy/safety thresholds in BOTH atlases, each atlas ranks genes by
# a COMBINED score (efficacy * safety), and the actual RRA rho-scoring is
# delegated to R's RobustRankAggreg package (aggregateRanks(method="RRA"))
# via rpy2, so the statistics match the reference implementation used
# elsewhere in SCART.
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_rra_input_single_gene(df_results: pd.DataFrame, atlas_key: str) -> pd.DataFrame:
    """Convert a single-gene df_results table (Gene, Efficacy, Safety,
    ObjectiveScore) into the Gene / <atlas>_efficacy / <atlas>_safety
    layout used by the RRA logic."""
    df = df_results.copy()
    df = df.rename(columns={
        "Efficacy": f"{atlas_key}_efficacy",
        "Safety":   f"{atlas_key}_safety",
    })
    df = df[["Gene", f"{atlas_key}_efficacy", f"{atlas_key}_safety"]]
    df = df.sort_values(by=f"{atlas_key}_efficacy", ascending=False)
    df = df.drop_duplicates(subset=["Gene"], keep="first")
    return df.reset_index(drop=True)


def _find_r_home() -> str:
    """
    Locate the R installation rpy2 should bind to for RobustRankAggreg.

    Preference order:
      1. R_HOME already set in the environment (respected as-is, if valid).
      2. <conda env>/lib/R  — i.e. sys.prefix/lib/R — the R that lives
         *inside the same conda environment SCART/rpy2 are installed in*.
         This is almost always the right answer and avoids ever touching a
         system R that may be a different version/ABI.
      3. `Rscript` resolved from PATH, asked directly for R.home().

    Raises RuntimeError with actionable install instructions if none of the
    above resolve to a real R installation.
    """
    import sys
    import shutil
    import subprocess

    env_r_home = os.environ.get("R_HOME")
    if env_r_home and os.path.isdir(env_r_home):
        return env_r_home

    conda_r_home = os.path.join(sys.prefix, "lib", "R")
    if os.path.exists(os.path.join(conda_r_home, "bin", "R")):
        return conda_r_home

    rscript = shutil.which("Rscript")
    if rscript:
        try:
            r_home = subprocess.check_output(
                [rscript, "-e", "cat(R.home())"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            if r_home and os.path.isdir(r_home):
                return r_home
        except Exception:
            pass

    raise RuntimeError(
        "Could not locate an R installation for Robust Rank Aggregation "
        "(atlas='both').\n"
        "Install R + RobustRankAggreg into this conda environment, e.g.:\n"
        "  conda install -c conda-forge r-base r-robustrankaggreg -y\n"
        "or set the R_HOME environment variable to point at an existing "
        "R installation before calling run()."
    )


def _setup_r_environment() -> str:
    """
    Point rpy2 at the correct R installation *before* rpy2 is imported.

    Sets R_HOME and prepends <R_HOME>/lib (where libR.so lives) to
    LD_LIBRARY_PATH. This must happen before the first `import rpy2...` in
    the process — otherwise rpy2/cffi may already have resolved a
    different, ABI-mismatched libR.so via whatever LD_LIBRARY_PATH the
    process started with (e.g. one containing only an unrelated CUDA lib64
    entry and no R lib directory at all), producing:
        undefined symbol: R_ClosureEnv   (cffi API mode)
        undefined symbol: R_getVar       (ABI-mode fallback)

    Mirrors the R-environment resolution already used by Module 4b
    (two_gene_combination.py)'s RRA step.

    Returns the resolved R_HOME (for logging).
    """
    r_home = _find_r_home()
    os.environ["R_HOME"] = r_home

    r_lib_dir = os.path.join(r_home, "lib")
    existing  = os.environ.get("LD_LIBRARY_PATH", "")
    existing_paths = [p for p in existing.split(os.pathsep) if p]
    if r_lib_dir not in existing_paths:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([r_lib_dir] + existing_paths)

    rscript_path = os.path.join(r_home, "bin", "Rscript")
    print(f"  Rscript:                  {rscript_path}")
    print(f"  R home:                   {r_home}")

    return r_home


def _run_rra_via_r(*rank_lists) -> dict:
    """
    Call R's RobustRankAggreg::aggregateRanks(method="RRA") via rpy2 to
    combine the per-atlas rank lists into a single robust rank score per
    gene. Requires the R package 'RobustRankAggreg' (installed by
    `python -m SCART.install`, see install.py).

    FIX: _setup_r_environment() runs first to make sure rpy2 binds to the
    R installation living in this same conda environment — see module
    docstring "Fix applied (rpy2 / R environment — atlas='both' RRA step)".
    """
    _setup_r_environment()

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import StrVector
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        raise ImportError(
            "rpy2 is required for Robust Rank Aggregation (atlas='both'). "
            "It is one of SCART's core dependencies — if missing, run:\n"
            "  pip install rpy2>=3.5\n"
            "If rpy2 is installed but this import still fails with an "
            "'undefined symbol' error, the R it bound to earlier in this "
            "Python session doesn't match this environment's R — restart "
            "the kernel/process and re-run so the corrected R_HOME/"
            "LD_LIBRARY_PATH set by _setup_r_environment() take effect "
            "before rpy2 is imported."
        ) from exc

    try:
        rra_pkg = importr("RobustRankAggreg")
    except Exception as exc:
        raise RuntimeError(
            "The R package 'RobustRankAggreg' is required for Robust Rank "
            "Aggregation (atlas='both') but was not found in your R "
            "environment.\n"
            "Install it with:\n"
            "  conda install -c conda-forge r-robustrankaggreg -y\n"
            "or from an R console:\n"
            "  install.packages('RobustRankAggreg')\n"
            "This is also installed automatically by: python -m SCART.install"
        ) from exc

    r_glist = ro.ListVector({
        f"list{i + 1}": StrVector(rl) for i, rl in enumerate(rank_lists)
    })

    r_result = rra_pkg.aggregateRanks(glist=r_glist, method="RRA")

    names  = [str(n) for n in r_result.rx2("Name")]
    scores = [float(s) for s in r_result.rx2("Score")]
    return dict(zip(names, scores))


def _plot_top_rra_single_gene(df_ranked: pd.DataFrame, output_dir: str, top_n: int = 20):
    """Grouped bar chart of the top-N RRA-ranked genes (efficacy + per-atlas
    safety), mirroring two_gene_combination.py's RRA plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = df_ranked.sort_values(by="RRA_Rank").head(top_n).copy()
    if top.empty:
        print("No RRA-ranked genes to plot — skipping plot.")
        return

    metrics = {
        "Efficacy":      top["hpa_efficacy"],
        "HPA Safety":    top["hpa_safety"],
        "Tabula Safety": top["tabula_safety"],
    }
    colors = {
        "Efficacy":      "#0072B2",
        "HPA Safety":    "#009E73",
        "Tabula Safety": "#CC79A7",
    }

    x     = np.arange(len(top))
    width = 0.25
    fig, ax = plt.subplots(figsize=(16, 8))

    for i, (label, values) in enumerate(metrics.items()):
        bars = ax.bar(x + (i - 1) * width, values, width, label=label,
                       color=colors[label], edgecolor="black", linewidth=0.4)
        ax.bar_label(bars, fmt="%.2f", rotation=90, padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(top["Gene"], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Top 20 RRA-Ranked Single-Gene Candidates (HPA + Tabula Sapiens)")
    ax.legend()
    fig.tight_layout()

    pdf_path = os.path.join(output_dir, "Top20_RRA_Single_Gene_Candidates_HPA_Tabula.pdf")
    png_path = os.path.join(output_dir, "Top20_RRA_Single_Gene_Candidates_HPA_Tabula.png")
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)

    print(f"Top-20 RRA plot saved to:\n  {pdf_path}\n  {png_path}")


def _robust_rank_aggregation_single_gene(
    df_results_hpa: pd.DataFrame,
    df_results_tabula: pd.DataFrame,
    efficacy_threshold: float,
    safety_threshold: float,
    output_dir: str,
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  Robust Rank Aggregation — combining HPA + Tabula Sapiens results")
    print("=" * 70)

    hpa_df    = _prepare_rra_input_single_gene(df_results_hpa,    "hpa")
    tabula_df = _prepare_rra_input_single_gene(df_results_tabula, "tabula")

    hpa_candidates = hpa_df[
        (hpa_df["hpa_efficacy"] > efficacy_threshold) &
        (hpa_df["hpa_safety"]   > safety_threshold)
    ]
    tabula_candidates = tabula_df[
        (tabula_df["tabula_efficacy"] > efficacy_threshold) &
        (tabula_df["tabula_safety"]   > safety_threshold)
    ]

    candidate_genes = pd.concat([
        hpa_candidates[["Gene"]],
        tabula_candidates[["Gene"]],
    ]).drop_duplicates()

    combined = candidate_genes.merge(hpa_df,    on="Gene", how="left")
    combined = combined.merge(tabula_df, on="Gene", how="left")
    combined = combined.drop_duplicates().reset_index(drop=True)
    print(f"unique_candidates dim: {combined.shape}")

    # Sanity check — efficacy should be identical across atlases for the
    # same gene (efficacy depends only on the tumour matrix; only safety
    # differs by healthy atlas).
    both_present = combined["hpa_efficacy"].notna() & combined["tabula_efficacy"].notna()
    equal_mask   = combined.loc[both_present, "hpa_efficacy"] == combined.loc[both_present, "tabula_efficacy"]
    print(f"Matching rows: {int(equal_mask.sum())} / {int(both_present.sum())}")
    print(f"Mismatching rows: {int((~equal_mask).sum())}")
    print(f"Rows present in only one atlas: {int((~both_present).sum())}")

    # STRICT FILTER: keep only genes passing efficacy > threshold AND
    # safety > threshold in BOTH atlases.
    strict = combined[
        (combined["hpa_efficacy"]    > efficacy_threshold) & (combined["hpa_safety"]    > safety_threshold) &
        (combined["tabula_efficacy"] > efficacy_threshold) & (combined["tabula_safety"] > safety_threshold)
    ].copy()
    print(f"Genes passing efficacy>{efficacy_threshold} & safety>{safety_threshold} "
          f"in BOTH atlases: {len(strict)}")

    out_csv = os.path.join(output_dir, "final_single_gene_candidates_RRA_HPA_Tabula.csv")

    if strict.empty:
        print("No genes passed the strict dual-atlas filter — "
              "skipping RRA aggregation and plot.")
        strict.to_csv(out_csv, index=False)
        return strict

    strict["hpa_combined"]    = strict["hpa_efficacy"]    * strict["hpa_safety"]
    strict["tabula_combined"] = strict["tabula_efficacy"] * strict["tabula_safety"]

    hpa_rank    = strict.sort_values(by="hpa_combined",    ascending=False)["Gene"].tolist()
    tabula_rank = strict.sort_values(by="tabula_combined", ascending=False)["Gene"].tolist()

    rra_scores = _run_rra_via_r(hpa_rank, tabula_rank)

    strict["RRA_Score"] = strict["Gene"].map(rra_scores)
    n_missing = strict["RRA_Score"].isna().sum()
    if n_missing > 0:
        logger.warning(f"{n_missing} gene(s) missing an RRA score after aggregation.")

    strict = strict.sort_values(by="RRA_Score", ascending=True).reset_index(drop=True)
    strict["RRA_Rank"] = np.arange(1, len(strict) + 1)

    strict.to_csv(out_csv, index=False)
    print(f"\nRRA-ranked dual-atlas single-gene candidates saved to: {out_csv}")

    print("\nTop 10 RRA-ranked genes:")
    print(strict.head(10).to_string(index=False))

    _plot_top_rra_single_gene(strict, output_dir)

    return strict


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    atlas: str = "both",
    hpa_path: str = None,
    tabula_path: str = None,
    tumor_path: str = None,
    safety_threshold: float = 0.9,
    rra_efficacy_threshold: float = 0.7,
    rra_safety_threshold: float = 0.9,
):
    """
    Run single-gene CAR-T target evaluation against one or both healthy
    reference atlases.

    Parameters
    ----------
    atlas : str
        Which healthy reference atlas(es) to score safety against:
          "hpa"    - HPA all-tissues (geosketch 10k) only.
          "tabula" - Tabula Sapiens all-tissues (10k) only.
          "both"   - evaluate against EACH atlas independently, save
                     individual per-atlas results, then combine the two
                     ranked gene lists via Robust Rank Aggregation (RRA).
                     Default.
    hpa_path : str or None
        Local path to the Zenodo-downloaded hpa_alltissues_geosketch_10k.h5ad
        file. If None and atlas is "hpa" or "both", SCART auto-searches
        <cwd>/hpa_alltissues_geosketch_10k.h5ad and
        <cwd>/healthy_atlases/hpa_alltissues_geosketch_10k.h5ad; raises
        FileNotFoundError with download/placement instructions if not found.
        (A .tsv/.tsv.gz path is also accepted, per _load_healthy_matrix.)
    tabula_path : str or None
        Same as hpa_path, for tabula_sapiens_alltissues_10k.h5ad.
    tumor_path : str or None
        Path to tumour h5ad (Module 3 output). Auto-detected if None.
    safety_threshold : float
        Minimum fraction of healthy cells that must NOT express the gene —
        used as the per-atlas ObjectiveScore cutoff. Range 0-1. Default 0.9.
    rra_efficacy_threshold, rra_safety_threshold : float
        Per-atlas thresholds a gene must clear in BOTH atlases to be
        eligible for Robust Rank Aggregation. Only used when atlas="both".
        Defaults 0.7 / 0.9 (matches two_gene_combination.py's RRA step).

    Returns
    -------
    If atlas == "hpa" or "tabula":
        df_results — unchanged behaviour: a single DataFrame (columns Gene,
        Efficacy, Safety, ObjectiveScore) for that one atlas.
    If atlas == "both":
        dict with keys:
          "hpa"    -> df_results_hpa
          "tabula" -> df_results_tabula
          "rra"    -> df_rra   (RRA-combined, ranked gene table)
    """
    atlas = (atlas or "both").strip().lower()
    if atlas not in ("hpa", "tabula", "both"):
        raise ValueError(f"atlas must be one of 'hpa', 'tabula', 'both' — got {atlas!r}")

    output_dir = os.getcwd()

    t_path = tumor_path or _auto_tumor_h5ad()
    print(f"Loading tumour matrix: {t_path}")
    adata_tumor = sc.read_h5ad(t_path)
    tumor_genes = list(adata_tumor.var_names)

    if atlas == "hpa":
        healthy_path = _resolve_atlas_path("hpa", hpa_path)
        print(f"\nAtlas selection: HPA only ({ATLAS_LABELS['hpa']})")
        return _evaluate_single_atlas(
            "hpa", healthy_path, adata_tumor, tumor_genes, safety_threshold, output_dir
        )

    if atlas == "tabula":
        healthy_path = _resolve_atlas_path("tabula", tabula_path)
        print(f"\nAtlas selection: Tabula Sapiens only ({ATLAS_LABELS['tabula']})")
        return _evaluate_single_atlas(
            "tabula", healthy_path, adata_tumor, tumor_genes, safety_threshold, output_dir
        )

    # atlas == "both"
    hpa_healthy_path    = _resolve_atlas_path("hpa", hpa_path)
    tabula_healthy_path = _resolve_atlas_path("tabula", tabula_path)

    print("\nAtlas selection: BOTH (independent runs + Robust Rank Aggregation)")

    print("\n" + "=" * 70)
    print(f"  Evaluating genes — ATLAS 1/2: {ATLAS_LABELS['hpa']}")
    print("=" * 70)
    df_results_hpa = _evaluate_single_atlas(
        "hpa", hpa_healthy_path, adata_tumor, tumor_genes, safety_threshold, output_dir
    )

    print("\n" + "=" * 70)
    print(f"  Evaluating genes — ATLAS 2/2: {ATLAS_LABELS['tabula']}")
    print("=" * 70)
    df_results_tabula = _evaluate_single_atlas(
        "tabula", tabula_healthy_path, adata_tumor, tumor_genes, safety_threshold, output_dir
    )

    df_rra = _robust_rank_aggregation_single_gene(
        df_results_hpa, df_results_tabula,
        efficacy_threshold=rra_efficacy_threshold,
        safety_threshold=rra_safety_threshold,
        output_dir=output_dir,
    )

    return {
        "hpa":    df_results_hpa,
        "tabula": df_results_tabula,
        "rra":    df_rra,
    }
