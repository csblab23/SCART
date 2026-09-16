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

Fix applied (RRA plot v2 — three-panel highlighted-region design)
--------------------------------------------------------------------
_plot_top10_rra_genes()/_dual_panel_rra_figure() have been redesigned to
match a reference R/ggplot2 + ggrepel + cowplot script (single-gene
efficacy-vs-safety figure), replacing the earlier fixed "top 10" dual-panel
scatter with a three-panel layout:

  LEFT   = every candidate gene as a grey background scatter, with a
           dashed-border, light-shaded rectangle marking the highlighted
           region and the highlighted genes drawn as coloured diamonds.
  RIGHT  = zoomed into that rectangle, on a light-grey panel background,
           with every highlighted gene labelled via a white, black-bordered
           box connected back to its point with a thin leader line when the
           label has been nudged away from it.
  LEGEND = a third narrow panel listing every highlighted gene as a
           numbered, colour-swatched row ("Gene Ranks"), best (RRA_Rank 1)
           first.

Key behavioural change from v1: the highlighted set is no longer a fixed
top-N. As in the reference script, the top `top_n` genes by RRA_Rank
(default 20, matching the reference script's "Top 20") are used only to
define the safety/efficacy rectangle (its bounds are the min/max spanned by
those genes, floored/ceiled by 1 point); EVERY gene whose point then falls
inside that rectangle is plotted, labelled and ranked — so the number of
genes actually shown can be smaller or larger than `top_n`. Genes are
coloured with an extended Okabe-Ito colourblind-safe palette (matching the
reference script exactly), smoothly interpolated with extra hues if more
genes fall inside the rectangle than the base palette has colours for.
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
# RRA result plot — three-panel "highlighted region" design
#
# Redesigned to match a reference R/ggplot2 + ggrepel + cowplot script
# (single-gene efficacy-vs-safety figure) — see module docstring "Fix
# applied (RRA plot v2 ...)" for the full design rationale.
#
#   LEFT   = every candidate gene, grey background, with a dashed-border
#            shaded rectangle marking the highlighted region and the
#            highlighted genes drawn as coloured diamonds.
#   RIGHT  = zoomed into that rectangle, light-grey panel background,
#            every highlighted gene labelled with a white leader-lined box.
#   LEGEND = a third narrow panel listing every highlighted gene as a
#            numbered, colour-swatched row ("Gene Ranks"), best first.
#
# The rectangle's bounds come from the top `top_n` genes by RRA_Rank
# (default 20), but every gene whose point falls inside it — not just
# those top `top_n` — gets plotted, labelled and ranked (mirrors the
# reference script's "top 20 defines the box, then show everyone inside
# it" behaviour exactly). As before, axes are left to auto-scale rather
# than pinned to a fixed floor, since the "strict" filter upstream only
# truly enforces a safety floor, not an efficacy one.
# ─────────────────────────────────────────────────────────────────────────────

# Extended Okabe-Ito colourblind-safe palette, matching the reference
# R/ggplot2 script exactly (7 Okabe-Ito hues + 14 additional vivid hues).
_HIGHLIGHT_PALETTE = [
    "#E69F00", "#0072B2", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#D55E00",
    "#7B2CBF", "#B37400", "#00A896", "#E63946", "#3A86FF", "#8C4A6B", "#FFB400",
    "#1D3557", "#43AA8B", "#9D4EDD", "#F4A261", "#118AB2", "#6A994E", "#FF6B6B",
]


def _rra_highlight_colors(n: int) -> list:
    """One colour per highlighted gene, ordered by rank. Uses the extended
    Okabe-Ito colourblind-safe palette as-is when it's large enough;
    smoothly interpolates additional hues (matching the reference script's
    colorRampPalette(base_palette)) when more genes fall inside the
    highlighted region than the base palette has colours for."""
    if n <= len(_HIGHLIGHT_PALETTE):
        return _HIGHLIGHT_PALETTE[:n]
    import matplotlib.colors as mcolors
    base_rgb = [mcolors.to_rgb(c) for c in _HIGHLIGHT_PALETTE]
    cmap = mcolors.LinearSegmentedColormap.from_list("rra_extended", base_rgb)
    return [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


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


def _resolve_serif_font() -> str:
    """Times New Roman, matching the reference R script, when it's actually
    installed on this machine; falls back to a metrically-similar or
    generic serif font otherwise (Times New Roman is a Windows/Office font
    and is often absent on Linux render hosts, e.g. HPC nodes)."""
    import matplotlib.font_manager as fm
    preferred = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return "serif"


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
    values instead of matplotlib's default (sometimes irregular)
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


def _style_scatter_axes(ax, title, xlabel, ylabel, x_step=None, y_step=None):
    """
    Shared panel styling: a full black box border on all four sides, no
    panel grid, bold titles, and evenly spaced ('nice') tick marks on both
    axes. Must be called AFTER the panel's final xlim/ylim are set, since
    the default tick step is computed from the current axis range.
    x_step/y_step let a caller pin an explicit tick increment (used on the
    right panel's efficacy axis, matching the reference script's
    hard-coded `by=2` breaks) instead of the auto "nice" one.
    """
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    ax.set_title(title, fontsize=16, fontweight="bold", pad=12, color="black")
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold", color="black")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold", color="black")
    ax.tick_params(labelsize=11, colors="black")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)

    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    ax.xaxis.set_major_locator(MultipleLocator(x_step or _nice_step(x_hi - x_lo)))
    ax.yaxis.set_major_locator(MultipleLocator(y_step or _nice_step(y_hi - y_lo)))


def _place_labels_with_leaders(ax, fig, xs, ys, labels, offset_frac: float = 0.035,
                                max_iter: int = 400):
    """
    White-boxed, black-text, black-bordered labels next to each point,
    connected back to the point with a thin grey leader line whenever the
    label ends up more than a small distance away from it — matching
    ggrepel's geom_label_repel style in the reference R script
    (label fill="white", color="black", segment.color="grey30",
    min.segment.length effectively 0). Prefers the optional `adjustText`
    package (pip install adjustText) for genuine collision-aware
    repulsion when it's installed; otherwise falls back to a
    dependency-free local nudge pass that separates overlapping label
    boxes (and label-over-point overlaps) using their real rendered
    extents.
    """
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    dx0 = offset_frac * (x_hi - x_lo)
    dy0 = offset_frac * (y_hi - y_lo)
    leader_thresh = 0.6 * min(dx0, dy0)

    texts = []
    for x, y, label in zip(xs, ys, labels):
        t = ax.text(
            x + dx0, y + dy0, label, fontsize=10.5, fontweight="bold",
            color="black", ha="left", va="bottom", zorder=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="black", linewidth=0.9),
        )
        texts.append(t)

    fig.canvas.draw()

    try:
        from adjustText import adjust_text
        adjust_text(texts, x=list(xs), y=list(ys), ax=ax)
    except ImportError:
        renderer = fig.canvas.get_renderer()
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
                        for t_, sign in ((texts[i], -1), (texts[j], 1)):
                            xt, yt = t_.get_position()
                            disp = ax.transData.transform((xt, yt))
                            disp = (disp[0] + sign * ux * 2.5, disp[1] + sign * uy * 2.5)
                            t_.set_position(ax.transData.inverted().transform(disp))

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
                        xt, yt = texts[i].get_position()
                        disp = ax.transData.transform((xt, yt))
                        disp = (disp[0] + ux * 2.5, disp[1] + uy * 2.5)
                        texts[i].set_position(ax.transData.inverted().transform(disp))
                        boxes[i] = texts[i].get_window_extent(renderer)

            if not moved:
                break
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

    fig.canvas.draw()
    for t, px, py in zip(texts, xs, ys):
        tx, ty = t.get_position()
        if ((tx - px) ** 2 + (ty - py) ** 2) ** 0.5 > leader_thresh:
            ax.annotate(
                "", xy=(px, py), xytext=(tx, ty),
                arrowprops=dict(arrowstyle="-", color="#4D4D4D", linewidth=0.8),
                zorder=4,
            )

    return texts


def _dual_panel_rra_figure(
    plot_df: pd.DataFrame,
    highlighted: pd.DataFrame,
    threshold_x: float,
    threshold_y: float,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    out_stub: str,
    output_dir: str,
    left_title: str,
    right_title: str,
):
    """
    Three-panel figure matching the reference R/ggplot2 + ggrepel + cowplot
    script: LEFT = every candidate (grey background) with the highlighted
    genes as coloured diamonds and a dashed-border shaded rectangle marking
    the highlighted region; RIGHT = zoomed into that rectangle, every
    highlighted gene labelled with a white leader-lined box, on a light
    grey panel background; a third narrow LEGEND panel lists every
    highlighted gene as a numbered, colour-swatched row ("Gene Ranks"),
    ordered by RRA_Rank (best first).
    """
    in_notebook = _configure_matplotlib_backend()
    import matplotlib.pyplot as plt

    serif = _resolve_serif_font()
    plt.rcParams.update({
        "font.family":      serif,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
    })

    n_hi = len(highlighted)
    fig = plt.figure(figsize=(17, 8.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.34], wspace=0.35)
    ax_l   = fig.add_subplot(gs[0, 0])
    ax_r   = fig.add_subplot(gs[0, 1])
    ax_leg = fig.add_subplot(gs[0, 2])

    xs_all = (plot_df["avg_safety"] * 100).to_numpy()
    ys_all = (plot_df["efficacy"] * 100).to_numpy()
    xs_hi  = (highlighted["avg_safety"] * 100).to_numpy()
    ys_hi  = (highlighted["efficacy"] * 100).to_numpy()
    colors = highlighted["color"].tolist()

    hi_genes = set(highlighted["Gene"])
    bg_mask  = ~plot_df["Gene"].isin(hi_genes)
    xs_bg    = (plot_df.loc[bg_mask, "avg_safety"] * 100).to_numpy()
    ys_bg    = (plot_df.loc[bg_mask, "efficacy"] * 100).to_numpy()

    # ---- left panel: full candidate universe, with highlighted rectangle --
    ax_l.scatter(xs_bg, ys_bg, s=24, color="#999999", alpha=0.55,
                 linewidths=0, zorder=2)
    ax_l.add_patch(plt.Rectangle(
        (x_lo, y_lo), max(x_hi - x_lo, 0.5), max(y_hi - y_lo, 0.5),
        fill=True, facecolor="#EDEEF3", alpha=0.9,
        edgecolor="black", linestyle="--", linewidth=1.2, zorder=1,
    ))
    ax_l.axvline(threshold_x, linestyle="--", linewidth=0.9, color="#4D4D4D", zorder=3)
    ax_l.axhline(threshold_y, linestyle="--", linewidth=0.9, color="#4D4D4D", zorder=3)
    ax_l.scatter(xs_hi, ys_hi, s=140, marker="D", c=colors,
                 edgecolor="black", linewidth=0.7, zorder=5)

    # No fixed axis floor here (matches v1) — both axes auto-scale to the
    # actual candidate spread rather than being pinned to e.g. 90%/70%.
    ax_l.margins(0.04)
    ax_l.set_box_aspect(1)
    _style_scatter_axes(ax_l, left_title, "Safety", "Efficacy")

    # ---- right panel: zoomed, labelled highlighted candidates -------------
    ax_r.set_facecolor("#EDEEF3")
    ax_r.axvline(threshold_x, linestyle="--", linewidth=0.9, color="#4D4D4D", zorder=3)
    ax_r.axhline(threshold_y, linestyle="--", linewidth=0.9, color="#4D4D4D", zorder=3)
    ax_r.scatter(xs_hi, ys_hi, s=170, marker="D", c=colors,
                 edgecolor="black", linewidth=0.8, zorder=5)

    pad_x = 0.08 * max(x_hi - x_lo, 1.0)
    pad_y = 0.08 * max(y_hi - y_lo, 1.0)
    ax_r.set_xlim(x_lo - pad_x, x_hi + pad_x)
    ax_r.set_ylim(y_lo - pad_y, y_hi + pad_y)
    ax_r.set_box_aspect(1)

    _place_labels_with_leaders(ax_r, fig, xs_hi, ys_hi, highlighted["label"].tolist())

    # y-axis ticks every 2 points, matching the reference script's
    # hard-coded `seq(..., by=2)` breaks, unless the zoomed range is wide
    # enough that a 2-point step would be unreadably dense.
    y_step = 2 if (y_hi - y_lo) <= 40 else None
    _style_scatter_axes(ax_r, right_title, "Safety", "Efficacy", y_step=y_step)

    # ---- legend panel: "Gene Ranks" ---------------------------------------
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(-(n_hi + 1), 1)
    ax_leg.axis("off")
    ax_leg.set_title("Gene Ranks", fontsize=15, fontweight="bold", loc="left")
    for _, row in highlighted.iterrows():
        y_pos = -row["rank"]
        ax_leg.scatter([0.04], [y_pos], s=120, marker="s", c=[row["color"]],
                       edgecolor="black", linewidth=0.6, zorder=3)
        ax_leg.text(0.12, y_pos, f"{int(row['rank'])}.  {row['label']}",
                    fontsize=10.5, fontweight="bold", color="black",
                    va="center", ha="left")

    fig.tight_layout()

    pdf_path = os.path.join(output_dir, f"{out_stub}_claude.pdf")
    png_path = os.path.join(output_dir, f"{out_stub}_claude.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"RRA plot saved to:\n  {pdf_path}\n  {png_path}")

    if in_notebook:
        plt.show()
    plt.close(fig)


def _plot_top10_rra_genes(df_ranked: pd.DataFrame, output_dir: str, top_n: int = 20):
    """
    Highlighted-region RRA plot (see module docstring "Fix applied (RRA
    plot v2 ...)" for the full design). The top `top_n` genes by RRA_Rank
    (default 20, matching the reference script's "Top 20") define a
    safety/efficacy rectangle, but EVERY gene whose point falls inside
    that rectangle — not just those top `top_n` — is plotted, labelled and
    ranked in the output. Function name is kept as `_plot_top10_rra_genes`
    for call-site compatibility even though the default is now 20 and the
    highlighted count is no longer fixed.
    """
    plot_df = _prep_rra_plot_df_single_gene(df_ranked)
    if plot_df.empty:
        print("No RRA-ranked genes available — skipping RRA plot.")
        return

    top_n_df = plot_df.sort_values("RRA_Rank", ascending=True).head(top_n)
    if top_n_df.empty:
        print("No RRA-ranked genes available — skipping RRA plot.")
        return

    safety_pct   = plot_df["avg_safety"] * 100
    efficacy_pct = plot_df["efficacy"] * 100
    top_safety_pct   = top_n_df["avg_safety"] * 100
    top_efficacy_pct = top_n_df["efficacy"] * 100

    # Thresholds / rectangle: min safety and min efficacy among the top
    # `top_n` genes, floored/ceiled by 1 point — exactly the reference
    # script's rank20_safety / min_top20_efficacy + zoom_x/y_min/max logic.
    threshold_x = float(top_safety_pct.min())
    threshold_y = float(top_efficacy_pct.min())
    x_lo = max(0.0, np.floor(top_safety_pct.min()) - 1)
    x_hi = min(100.0, np.ceil(top_safety_pct.max()) + 1)
    y_lo = max(0.0, np.floor(top_efficacy_pct.min()) - 1)
    y_hi = min(100.0, np.ceil(top_efficacy_pct.max()) + 1)

    # ALL genes whose point falls inside the rectangle — not just the
    # top_n used to define it (mirrors the reference script's
    # "highlighted" step exactly).
    in_rect = (
        (safety_pct >= x_lo) & (safety_pct <= x_hi) &
        (efficacy_pct >= y_lo) & (efficacy_pct <= y_hi)
    )
    highlighted = plot_df[in_rect].sort_values("RRA_Rank", ascending=True).reset_index(drop=True)

    if highlighted.empty:
        print("No genes fall inside the highlighted region — skipping RRA plot.")
        return

    highlighted["rank"]  = np.arange(1, len(highlighted) + 1)
    highlighted["label"] = highlighted["candidate"]
    highlighted["color"] = _rra_highlight_colors(len(highlighted))

    print(f"\nGenes inside highlighted region (rectangle set by top {top_n}): {len(highlighted)}")
    print(highlighted[["rank", "Gene", "efficacy", "avg_safety"]].to_string(index=False))

    _dual_panel_rra_figure(
        plot_df, highlighted,
        threshold_x=threshold_x, threshold_y=threshold_y,
        x_lo=x_lo, x_hi=x_hi, y_lo=y_lo, y_hi=y_hi,
        out_stub="Single_Gene_RRA_Highlighted_Region",
        output_dir=output_dir,
        left_title="All single-gene candidates",
        right_title="Candidates in highlighted region",
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
