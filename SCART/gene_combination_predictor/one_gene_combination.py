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

Fix applied (rpy2 / R environment — atlas="both" RRA step) [SUPERSEDED]
-------------------------------------------------------------------------
_run_rra_via_r originally called `import rpy2.robjects` with no control
over which R installation rpy2 binds to. On a machine with more than one R
on the system (e.g. a system R alongside the conda env's own R), rpy2 could
resolve a libR.so that did NOT match the R the RobustRankAggreg package was
actually installed into, or could load it without the R_HOME/lib directory
on the loader's search path. Both produced hard-to-read failures during
`import rpy2.rinterface`, e.g.:
  ImportError: .../_rinterface_cffi_api.abi3.so: undefined symbol: R_ClosureEnv
  (falls back to ABI mode, which then also fails:)
  error: symbol 'R_getVar' not found in library '.../lib/R/lib/libR.so'

A first fix attempt (_setup_r_environment(), resolving R_HOME and
prepending <R_HOME>/lib to LD_LIBRARY_PATH before importing rpy2) reduced
but did not eliminate this — in practice the failure persisted even with
R_HOME/LD_LIBRARY_PATH correctly resolved and logged, because rpy2's
in-process embedding of libR.so is sensitive to whatever else has already
been loaded/linked into the same Python process (e.g. other native
extensions pulling in a conflicting BLAS/LAPACK, or rpy2 having cached a
partially-failed import earlier in a long-lived kernel). Because Python
caches failed imports in sys.modules, once rpy2 fails once in a process it
cannot be retroactively repaired without restarting the kernel — which is
fragile for a long pipeline run.

Fix (current): the RRA step no longer embeds R in-process via rpy2 at all.
Instead — mirroring exactly how the SCART preprocessing module already
drives R successfully for scMalignantFinder and SCEVAN — it now:
  1. writes each per-atlas ranked gene list out to a small CSV file,
  2. writes a short auto-generated R driver script that loads
     RobustRankAggreg, reads those CSVs, calls aggregateRanks(), and
     writes the result back out to a CSV,
  3. launches that driver script as an external `Rscript` subprocess
     (with R_HOME / LD_LIBRARY_PATH set on the subprocess's own env, not
     the Python process's), and
  4. reads the resulting CSV back into a Python dict.
A fresh `Rscript` subprocess always loads its own libR.so cleanly, so this
sidesteps the in-process ABI/symbol-resolution problem entirely, and one
failed R call can no longer poison the rest of the Python session the way
a bad rpy2 import could.

IMPORTANT: rpy2 is no longer a dependency of this module. R itself (with
the RobustRankAggreg package installed) plus a working `Rscript` on PATH
or inside the active conda environment are still required for atlas="both".

Fix applied (RRA plot)
------------------------
The grouped-bar "Top 20 RRA candidates" plot has been replaced with
_plot_top10_rra_genes(): a dual-panel scatter plot (avg. safety vs HPA
efficacy) of the top 10 genes by RRA_Rank, same visual style as
two_gene_combination.py's RRA plots (grey background of every candidate,
dashed partition lines on both panels, a shaded/dashed zoom rectangle on
the left, colour-ranked circles with soft shadow + labelled "chips" on the
right). There is no per-gate top-3 plot here, since single genes have no
logic-gate dimension.

Unlike two_gene_combination.py, the left panel's axes are NOT fixed to
start at safety=90%/efficacy=70%: _robust_rank_aggregation_single_gene()'s
"strict" filter below only actually enforces `safety > safety_threshold`
in both atlases (default 0.9) — the `efficacy_threshold` parameter it
accepts is not applied anywhere in that filter, so candidate efficacy has
no guaranteed floor here. Hard-coding an efficacy axis minimum could clip
genuine data, so both axes are left to auto-scale to whatever the actual
candidates span.
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
# delegated to R's RobustRankAggreg package (aggregateRanks(method="RRA")).
#
# As of the fix described in the module docstring, this is now driven the
# same way the SCART preprocessing module drives scMalignantFinder/SCEVAN:
# write inputs to disk -> write an R driver script -> launch Rscript as a
# subprocess -> read the R-written output CSV back into Python. No rpy2.
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
    Locate the R installation Rscript should bind to for RobustRankAggreg.

    Preference order:
      1. R_HOME already set in the environment (respected as-is, if valid).
      2. <conda env>/lib/R  — i.e. sys.prefix/lib/R — the R that lives
         *inside the same conda environment SCART is installed in*.
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


def _find_rscript() -> str:
    """
    Locate the Rscript executable to launch as a subprocess for the RRA
    step. Mirrors the preprocessing module's own Rscript resolution
    (SCART.preprocessing "Rscript found via sys.executable dir" logic):
    prefer the Rscript living inside the same conda environment SCART is
    installed in, falling back to whatever Rscript is on PATH.
    """
    import sys
    import shutil

    conda_rscript = os.path.join(sys.prefix, "bin", "Rscript")
    if os.path.exists(conda_rscript):
        return conda_rscript

    rscript = shutil.which("Rscript")
    if rscript:
        return rscript

    raise RuntimeError(
        "Could not locate an 'Rscript' executable for Robust Rank "
        "Aggregation (atlas='both').\n"
        "Install R + RobustRankAggreg into this conda environment, e.g.:\n"
        "  conda install -c conda-forge r-base r-robustrankaggreg -y"
    )


def _build_r_subprocess_env(r_home: str) -> dict:
    """
    Build the environment dict passed to the Rscript subprocess: sets
    R_HOME and prepends <R_HOME>/lib (where libR.so lives) to
    LD_LIBRARY_PATH, on top of a copy of the current process environment.

    Unlike the earlier rpy2-based approach, this only affects the child
    subprocess's environment — it never mutates os.environ for the parent
    Python process, so it can't interact with anything else already
    running in this Python session.
    """
    env = os.environ.copy()
    env["R_HOME"] = r_home

    r_lib_dir      = os.path.join(r_home, "lib")
    existing       = env.get("LD_LIBRARY_PATH", "")
    existing_paths = [p for p in existing.split(os.pathsep) if p]
    if r_lib_dir not in existing_paths:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([r_lib_dir] + existing_paths)

    return env


def _run_rra_via_r(*rank_lists, output_dir: str) -> dict:
    """
    Run R's RobustRankAggreg::aggregateRanks(method="RRA") as an external
    Rscript SUBPROCESS (not via in-process rpy2 embedding — see module
    docstring "Fix applied (rpy2 / R environment ...) [SUPERSEDED]").

    This mirrors exactly how the SCART preprocessing module already calls
    R for scMalignantFinder and SCEVAN: write inputs to disk, write a small
    R driver script, launch `Rscript <driver>.R` as a subprocess, then read
    the R-written output CSV back into Python.

    Requires the R package 'RobustRankAggreg' to be installed in whichever
    R installation Rscript resolves to (installed automatically by
    `python -m SCART.install`, see install.py).

    Parameters
    ----------
    *rank_lists : list[str]
        One ranked gene list (best -> worst) per atlas.
    output_dir : str
        Directory to write the RRA driver script / input & output CSVs
        into (a "rra_rscript" subfolder is created here).

    Returns
    -------
    dict[str, float]  — gene -> RRA score (lower = better, matching
    RobustRankAggreg's convention, same as before).
    """
    import subprocess
    import csv

    rscript_path = _find_rscript()
    r_home       = _find_r_home()
    env          = _build_r_subprocess_env(r_home)

    print(f"  Rscript:                  {rscript_path}")
    print(f"  R home:                   {r_home}")

    rra_dir = os.path.join(output_dir, "rra_rscript")
    os.makedirs(rra_dir, exist_ok=True)

    # Write each rank list to its own single-column CSV (no header, one
    # gene per row, best -> worst), same idea as SCEVAN's counts/barcodes
    # CSVs handed off to its driver R script.
    input_paths = []
    for i, rl in enumerate(rank_lists):
        path = os.path.join(rra_dir, f"rank_list_{i + 1}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for gene in rl:
                writer.writerow([gene])
        input_paths.append(path)
        print(f"  RRA input list {i + 1} ({len(rl)} genes) written to: {path}")

    output_path   = os.path.join(rra_dir, "rra_result.csv")
    r_script_path = os.path.join(rra_dir, "run_rra.R")

    r_input_vector = ", ".join(f'"{p}"' for p in input_paths)
    r_code = f'''# Auto-generated driver script — Robust Rank Aggregation (SCART Module 4a)
# Mirrors the driver-script pattern used by SCEVAN's run_scevan.R.
suppressMessages(library(RobustRankAggreg))

input_files <- c({r_input_vector})
glist <- lapply(input_files, function(f) {{
  read.csv(f, header = FALSE, stringsAsFactors = FALSE)[[1]]
}})

result <- aggregateRanks(glist = glist, method = "RRA")
write.csv(result, file = "{output_path}", row.names = FALSE)
cat("RRA aggregation completed. Rows:", nrow(result), "\\n")
'''
    with open(r_script_path, "w") as f:
        f.write(r_code)
    print(f"  RRA driver script written: {r_script_path}")
    print("  Launching Rscript subprocess for RobustRankAggreg::aggregateRanks() ...")

    try:
        proc = subprocess.run(
            [rscript_path, r_script_path],
            env=env,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Failed to launch Rscript subprocess at {rscript_path}: {exc}"
        ) from exc

    if proc.stdout:
        print(proc.stdout.strip())

    if proc.returncode != 0:
        raise RuntimeError(
            "Rscript subprocess for Robust Rank Aggregation failed "
            f"(exit code {proc.returncode}).\n"
            f"--- Rscript stderr ---\n{proc.stderr}\n"
            "Make sure the R package 'RobustRankAggreg' is installed in "
            f"the R at {r_home}, e.g.:\n"
            "  conda install -c conda-forge r-robustrankaggreg -y\n"
            "or from an R console: install.packages('RobustRankAggreg')"
        )

    if proc.stderr:
        # R (and library()) often write benign package-load / startup
        # messages to stderr even on success — log, don't fail on these.
        logger.info(f"Rscript stderr (non-fatal):\n{proc.stderr.strip()}")

    if not os.path.exists(output_path):
        raise RuntimeError(
            "Rscript subprocess exited successfully but the expected RRA "
            f"output file was not found: {output_path}"
        )

    df_out = pd.read_csv(output_path)
    if not {"Name", "Score"}.issubset(df_out.columns):
        raise RuntimeError(
            f"Unexpected RRA output columns in {output_path}: "
            f"{list(df_out.columns)} (expected 'Name' and 'Score')"
        )

    print(f"  RRA result CSV read back from: {output_path}")

    return dict(zip(df_out["Name"].astype(str), df_out["Score"].astype(float)))


# ─────────────────────────────────────────────────────────────────────────────
# RRA result plot
#
# Replaces the earlier grouped-bar "Top 20 RRA candidates" plot with a
# single dual-panel scatter plot of the top 10 genes overall by RRA_Rank —
# same visual style as two_gene_combination.py's RRA plots (see that
# module's "RRA result plots" section for the full design rationale), just
# with no gate dimension to plot (single genes have no logic gate), so
# there is only the one "top 10" plot here, no per-gate top-3 plot.
#
# The left panel's axes are intentionally left to auto-scale (no fixed
# safety>=90%/efficacy>=70% floor) — see module docstring "Fix applied
# (RRA plot)" for why: this module's strict filter only truly enforces a
# safety floor, not an efficacy one.
# ─────────────────────────────────────────────────────────────────────────────

# Rank-ordered palette for the top-10-overall plot (rank 1 first, darkest).
_RANK_PALETTE = [
    "#264653", "#2A9D8F", "#8AB17D", "#E9C46A", "#F4A261",
    "#EE8959", "#E76F51", "#C1121F", "#780000", "#4A0404",
]


def _configure_matplotlib_backend() -> bool:
    """
    Headless-safe by default — mirrors the previous hard-coded
    `matplotlib.use("Agg")` so unattended/HPC script runs never try (and
    fail) to open a GUI window. Inside a Jupyter kernel, the backend is
    left alone instead, so the inline/widget backend already configured
    there is free to actually display the figure.

    Returns True if running inside a Jupyter kernel (safe to call
    plt.show()), False otherwise.
    """
    import matplotlib

    try:
        from IPython import get_ipython
        ip = get_ipython()
        in_notebook = ip is not None and "IPKernelApp" in ip.config
    except Exception:
        in_notebook = False

    if not in_notebook:
        matplotlib.use("Agg")

    return in_notebook


def _prep_rra_plot_df_single_gene(df_ranked: pd.DataFrame) -> pd.DataFrame:
    """Shared prep for the RRA plot: average safety across atlases and a
    display label (just the gene name) per candidate."""
    df = df_ranked.dropna(subset=["hpa_safety", "tabula_safety", "hpa_efficacy"]).copy()
    df["avg_safety"] = (df["hpa_safety"] + df["tabula_safety"]) / 2.0
    df["efficacy"]   = df["hpa_efficacy"]  # atlas-invariant by construction
    df["candidate"]  = df["Gene"]
    return df


def _nice_step(data_range: float, target_ticks: int = 6) -> float:
    """Pick a 'nice' round tick increment (1/2/2.5/5/10 x 10^n) for an
    axis span of this size, so tick marks land on clean, evenly-spaced
    values — matching the reference script's explicit `seq(..., by=2)`
    breaks — instead of matplotlib's default (sometimes irregular)
    auto-ticks."""
    import math

    if data_range <= 0:
        return 1.0
    raw_step  = data_range / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw_step))
    residual  = raw_step / magnitude
    if residual <= 1:
        nice = 1
    elif residual <= 2:
        nice = 2
    elif residual <= 2.5:
        nice = 2.5
    elif residual <= 5:
        nice = 5
    else:
        nice = 10
    return nice * magnitude


def _style_scatter_axes(ax, title, xlabel, ylabel):
    """
    Shared panel styling matching the reference script's
    theme_classic() + explicit panel.border/axis.line: a full black box
    border on all four sides, no panel grid, bold titles, and evenly
    spaced ('nice') tick marks on both axes. Must be called AFTER the
    panel's final xlim/ylim are set, since the tick step is computed from
    the current axis range.
    """
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12, color="black")
    ax.set_xlabel(xlabel, fontsize=12.5, fontweight="bold", color="black")
    ax.set_ylabel(ylabel, fontsize=12.5, fontweight="bold", color="black")
    ax.tick_params(labelsize=11, colors="black")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)

    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    ax.xaxis.set_major_locator(MultipleLocator(_nice_step(x_hi - x_lo)))
    ax.yaxis.set_major_locator(MultipleLocator(_nice_step(y_hi - y_lo)))


def _place_labels_near_points(ax, fig, xs, ys, labels, colors, offset_frac: float = 0.02,
                               max_iter: int = 300):
    """
    Place each label immediately next to its point — no leader line drawn
    — matching the reference script's tight ggrepel placement (whose
    `min.segment.length = 0` in practice renders no visible connector for
    well-separated points, since the label sits right at the point).

    Prefers the optional `adjustText` package when installed (moved with
    no arrowprops, so it repels overlaps without ever drawing a line);
    otherwise falls back to a dependency-free local nudge pass that
    separates any colliding label boxes (and label-over-point overlaps)
    using their real rendered extents.
    """
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    dx0 = offset_frac * (x_hi - x_lo)
    dy0 = offset_frac * (y_hi - y_lo)

    texts = []
    for x, y, label, color in zip(xs, ys, labels, colors):
        t = ax.text(
            x + dx0, y + dy0, label, fontsize=10.5, fontweight="bold",
            color=color, ha="left", va="bottom", zorder=6,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor=color, linewidth=1.4),
        )
        texts.append(t)

    try:
        from adjustText import adjust_text
        fig.canvas.draw()
        pts_xy = list(zip(xs, ys))
        adjust_text(texts, x=[p[0] for p in pts_xy], y=[p[1] for p in pts_xy], ax=ax)
        return texts
    except ImportError:
        pass

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Small display-space box around each point's own marker, so labels
    # also get nudged off of OTHER points' markers, not just each other.
    marker_half_px = 9

    def _point_box(px, py):
        from matplotlib.transforms import Bbox
        dx_, dy_ = ax.transData.transform((px, py))
        return Bbox.from_extents(dx_ - marker_half_px, dy_ - marker_half_px,
                                  dx_ + marker_half_px, dy_ + marker_half_px)

    point_boxes = [_point_box(px, py) for px, py in zip(xs, ys)]

    for _ in range(max_iter):
        moved = False
        boxes = [t.get_window_extent(renderer) for t in texts]

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if boxes[i].overlaps(boxes[j]):
                    moved = True
                    cxi = boxes[i].x0 + boxes[i].width / 2
                    cyi = boxes[i].y0 + boxes[i].height / 2
                    cxj = boxes[j].x0 + boxes[j].width / 2
                    cyj = boxes[j].y0 + boxes[j].height / 2
                    ddx, ddy = cxj - cxi, cyj - cyi
                    dist = max((ddx ** 2 + ddy ** 2) ** 0.5, 1e-6)
                    ux, uy = ddx / dist, ddy / dist
                    xi, yi = texts[i].get_position()
                    xj, yj = texts[j].get_position()
                    disp_i = ax.transData.transform((xi, yi))
                    disp_j = ax.transData.transform((xj, yj))
                    disp_i = (disp_i[0] - ux * 2.5, disp_i[1] - uy * 2.5)
                    disp_j = (disp_j[0] + ux * 2.5, disp_j[1] + uy * 2.5)
                    inv = ax.transData.inverted()
                    texts[i].set_position(inv.transform(disp_i))
                    texts[j].set_position(inv.transform(disp_j))

        # Nudge any label off of a point marker it now overlaps (its own
        # or another's), pushing it away from that point's centre.
        boxes = [t.get_window_extent(renderer) for t in texts]
        for i in range(len(texts)):
            for pbox, (px, py) in zip(point_boxes, zip(xs, ys)):
                if boxes[i].overlaps(pbox):
                    moved = True
                    cxi = boxes[i].x0 + boxes[i].width / 2
                    cyi = boxes[i].y0 + boxes[i].height / 2
                    pdx, pdy = ax.transData.transform((px, py))
                    ddx, ddy = cxi - pdx, cyi - pdy
                    dist = max((ddx ** 2 + ddy ** 2) ** 0.5, 1e-6)
                    ux, uy = ddx / dist, ddy / dist
                    xi, yi = texts[i].get_position()
                    disp_i = ax.transData.transform((xi, yi))
                    disp_i = (disp_i[0] + ux * 2.5, disp_i[1] + uy * 2.5)
                    inv = ax.transData.inverted()
                    texts[i].set_position(inv.transform(disp_i))
                    boxes[i] = texts[i].get_window_extent(renderer)

        if not moved:
            break
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    return texts


def _dual_panel_rra_figure(
    plot_df: pd.DataFrame,
    highlighted: pd.DataFrame,
    colors: list,
    out_stub: str,
    output_dir: str,
    suptitle: str,
    left_title: str,
    right_title: str,
):
    """
    Dual-panel figure matching the reference R/ggplot2+ggrepel script:
    left panel shows every candidate as background, highlighted
    candidates as colour-coded diamonds, dashed partition lines (lowest
    safety/efficacy among the highlighted set), and a shaded dashed-border
    zoom rectangle. Right panel is the same partition zoomed in on just
    the highlighted candidates, on a light grey background, with labels
    placed immediately next to each point and no leader lines.

    NOTE: unlike two_gene_combination.py's version of this function, the
    left panel's axes are NOT pinned to a fixed safety>=90%/efficacy>=70%
    floor — see module docstring "Fix applied (RRA plot)".
    """
    in_notebook = _configure_matplotlib_backend()
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
    })

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(17, 8.5))
    fig.suptitle(suptitle, fontsize=17, fontweight="bold", y=1.03, color="black")

    xs_all = (plot_df["avg_safety"] * 100).to_numpy()
    ys_all = (plot_df["efficacy"] * 100).to_numpy()
    xs_hi  = (highlighted["avg_safety"] * 100).to_numpy()
    ys_hi  = (highlighted["efficacy"] * 100).to_numpy()

    # Partition thresholds — the lowest safety / efficacy among the
    # highlighted candidates, drawn as full dashed cross-lines on both
    # panels. The shaded zoom rectangle uses a 1-point floor/ceil margin
    # around the highlighted candidates.
    threshold_x = float(xs_hi.min())
    threshold_y = float(ys_hi.min())
    x_lo = max(0.0, np.floor(xs_hi.min()) - 1)
    x_hi = min(100.0, np.ceil(xs_hi.max()) + 1)
    y_lo = max(0.0, np.floor(ys_hi.min()) - 1)
    y_hi = min(100.0, np.ceil(ys_hi.max()) + 1)

    # ---- left panel: full candidate universe, with partition -------------
    ax_l.scatter(xs_all, ys_all, s=26, color="#8c8c8c", alpha=0.6,
                 linewidths=0, zorder=2)
    ax_l.axvline(threshold_x, linestyle="--", linewidth=1.1, color="#4d4d4d", zorder=3)
    ax_l.axhline(threshold_y, linestyle="--", linewidth=1.1, color="#4d4d4d", zorder=3)
    ax_l.add_patch(plt.Rectangle(
        (x_lo, y_lo), max(x_hi - x_lo, 0.5), max(y_hi - y_lo, 0.5),
        fill=True, facecolor="#777777", alpha=0.15,
        edgecolor="black", linestyle="--", linewidth=1.4, zorder=3,
    ))
    ax_l.scatter(xs_hi, ys_hi, s=170, marker="D", color=colors,
                 edgecolor="black", linewidth=0.9, zorder=5)

    # No fixed axis floor here (unlike two_gene_combination.py) — see
    # module docstring "Fix applied (RRA plot)"; both axes auto-scale.
    ax_l.set_box_aspect(1)
    _style_scatter_axes(ax_l, left_title, "Average safety", "Efficacy")

    # ---- right panel: zoomed, labelled highlighted candidates -------------
    ax_r.set_facecolor("#F5F5F5")
    ax_r.axvline(threshold_x, linestyle="--", linewidth=1.1, color="#4d4d4d", zorder=3)
    ax_r.axhline(threshold_y, linestyle="--", linewidth=1.1, color="#4d4d4d", zorder=3)
    ax_r.scatter(xs_hi, ys_hi, s=190, marker="D", color=colors,
                 edgecolor="black", linewidth=0.9, zorder=5)

    pad_x = 0.10 * max(x_hi - x_lo, 1.0)
    pad_y = 0.12 * max(y_hi - y_lo, 1.0)
    ax_r.set_xlim(x_lo - pad_x, x_hi + pad_x)
    ax_r.set_ylim(y_lo - pad_y, y_hi + pad_y)
    ax_r.set_box_aspect(1)

    _place_labels_near_points(
        ax_r, fig, xs_hi, ys_hi, highlighted["candidate"].tolist(), colors,
    )

    _style_scatter_axes(ax_r, right_title, "Average safety", "HPA efficacy")

    fig.tight_layout()

    pdf_path = os.path.join(output_dir, f"{out_stub}_claude.pdf")
    png_path = os.path.join(output_dir, f"{out_stub}_claude.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"RRA plot saved to:\n  {pdf_path}\n  {png_path}")

    if in_notebook:
        plt.show()
    plt.close(fig)


def _plot_top10_rra_genes(df_ranked: pd.DataFrame, output_dir: str, top_n: int = 10):
    """
    Top `top_n` (default 10) genes overall by RRA_Rank. Same dual-panel
    look as two_gene_combination.py's top-10 plot, with points coloured by
    rank (best = darkest) and rank numbers baked into each label. There is
    no per-gate variant of this plot — single genes have no logic gate.
    """
    plot_df     = _prep_rra_plot_df_single_gene(df_ranked)
    highlighted = plot_df.sort_values("RRA_Rank").head(top_n).reset_index(drop=True)

    if highlighted.empty:
        print("No RRA-ranked genes available — skipping top-10 RRA plot.")
        return

    highlighted = highlighted.copy()
    highlighted["candidate"] = [
        f"#{i + 1}  {c}" for i, c in enumerate(highlighted["candidate"])
    ]
    colors = _RANK_PALETTE[:len(highlighted)]

    _dual_panel_rra_figure(
        plot_df, highlighted, colors,
        out_stub="Top10_RRA_Single_Gene_Candidates",
        output_dir=output_dir,
        suptitle="Top 10 Genes Overall — Robust Rank Aggregation",
        left_title="All candidate genes",
        right_title="Top 10 by RRA rank",
    )


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
        (hpa_df["hpa_safety"]   > safety_threshold)
    ]
    tabula_candidates = tabula_df[
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
        (combined["hpa_safety"]    > safety_threshold) &
        (combined["tabula_safety"] > safety_threshold)
    ].copy()
    print(f"Genes passing safety>{safety_threshold} "
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

    rra_scores = _run_rra_via_r(hpa_rank, tabula_rank, output_dir=output_dir)

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

    _plot_top10_rra_genes(strict, output_dir)

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
