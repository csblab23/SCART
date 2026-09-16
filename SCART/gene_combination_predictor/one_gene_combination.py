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

Fix applied (RRA plot v3 — plotted natively in R)
-----------------------------------------------------
v2 was a matplotlib approximation of the reference R/ggplot2 + ggrepel +
cowplot script. It is now the reference script itself, embedded directly
in this module: the highlighted-region design (thresholds, rectangle,
Okabe-Ito palette, ggrepel labels, "Gene Ranks" legend panel,
cowplot::plot_grid combine) lives in _SINGLE_GENE_RRA_PLOT_R_TEMPLATE
below as an R source template — no separate .R file to ship or locate.
_plot_single_gene_rra_r() fills in the three values that vary per run
(input CSV path, output dir, top_n) via simple string substitution, writes
the result to a small driver script under output_dir, and launches it as
an Rscript subprocess — exactly the pattern _run_rra_via_r() above already
uses for RobustRankAggreg (_find_rscript()/_find_r_home()/
_build_r_subprocess_env() are reused as-is, unchanged). The only changes
from the reference script are: the rank column is "RRA_Rank" (what
_robust_rank_aggregation_single_gene() actually writes) instead of
"Final_Rank", and the output file carries a "_claude" suffix.

This removes the entire matplotlib plotting stack (_dual_panel_rra_figure,
_place_labels_with_leaders, etc.) from this module — matplotlib is no
longer imported or required here.

New R dependency: the R package 'cowplot' (ggplot2, dplyr and ggrepel were
already required for atlas="both"). See install.py's CONDA_R_BASE list,
which now includes r-cowplot for both the Linux/Mac and Windows setup
paths.
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
# RRA result plot — rendered natively in R (self-contained in this module)
#
# v3 shipped the R plotting code as a separate companion .R file next to
# this module. It is now fully self-contained here instead, following the
# exact same pattern _run_rra_via_r() above already uses for
# RobustRankAggreg: the R source lives in this file as a template string,
# _plot_single_gene_rra_r() fills in the three values that vary per run
# (input CSV path, output dir, top_n), writes the result to a small driver
# script under output_dir, and launches it as an Rscript subprocess. No
# separate file needs to be shipped or located at import time.
#
# Design (thresholds, highlighted-rectangle selection, Okabe-Ito palette,
# ggrepel labels, "Gene Ranks" legend panel, cowplot::plot_grid combine)
# matches a reference R/ggplot2 + ggrepel + cowplot script exactly — see
# module docstring "Fix applied (RRA plot v3 ...)".
#
# Requires the R packages ggplot2, dplyr, ggrepel and cowplot (cowplot is
# newly added for this — see install.py's CONDA_R_BASE).
# ─────────────────────────────────────────────────────────────────────────────

_SINGLE_GENE_RRA_PLOT_R_TEMPLATE = r"""
#!/usr/bin/env Rscript

library(ggplot2)
library(dplyr)
library(cowplot)
library(ggrepel)

# ============================================================
# SINGLE-GENE:
# EFFICACY vs SAFETY
#
# Auto-generated by one_gene_combination.py's
# _plot_single_gene_rra_r() every time it runs (mirrors exactly
# how _run_rra_via_r() already generates its own R driver script
# for RobustRankAggreg) — NOT meant to be hand-edited or launched
# with different arguments. The plotting logic below (thresholds,
# highlighted-rectangle selection, palette, panels, legend,
# combine, save) matches a reference R/ggplot2/ggrepel/cowplot
# script exactly. Only two things differ from that reference:
#   1. input_file / output_dir_arg / top_n below are literal
#      values baked in by _plot_single_gene_rra_r() at write time,
#      instead of a hard-coded path.
#   2. The rank column is "RRA_Rank" (what
#      _robust_rank_aggregation_single_gene() actually writes),
#      not "Final_Rank"; and the output file gets a "_claude"
#      suffix, per this project's naming convention.
#
# Output:
# <output_dir>/SCATTER_PLOTS_FIGURE3/Single_gene_efficacy_vs_safety_COMBINED_claude.png
#
#   Single combined figure:
#     LEFT  = all genes, highlighted rectangle
#     RIGHT = zoom into rectangle, EVERY gene inside the
#             rectangle is labeled (not just top N)
#
# Resolution:
# 800 DPI
# Format:
# PNG
# ============================================================


# ============================================================
# 1. INPUT FILE / OUTPUT DIRECTORY / TOP N
#
# Written in directly by _plot_single_gene_rra_r() when it
# generates this file.
# ============================================================

input_file <- "@@INPUT_CSV@@"
output_dir_arg <- "@@OUTPUT_DIR@@"
top_n <- @@TOP_N@@

# ============================================================
# 2. INPUT / OUTPUT DIRECTORIES
# ============================================================

output_dir <- file.path(
  output_dir_arg,
  "SCATTER_PLOTS_FIGURE3"
)

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)


# ============================================================
# 3. CHECK INPUT FILE
# ============================================================

cat("\n")
cat("============================================================\n")
cat("SINGLE-GENE EFFICACY vs SAFETY ANALYSIS\n")
cat("============================================================\n")

cat(
  "Input file:\n",
  input_file,
  "\n"
)

if (!file.exists(input_file)) {

  stop(
    paste0(
      "\nERROR: Input file does not exist:\n",
      input_file,
      "\n"
    )
  )
}


# ============================================================
# 4. OUTPUT FILE (single combined figure)
# ============================================================

combined_file <- file.path(
  output_dir,
  "Single_gene_efficacy_vs_safety_COMBINED_claude.png"
)

cat(
  "\nOutput directory:\n",
  output_dir,
  "\n"
)

cat(
  "\nCombined plot:\n",
  combined_file,
  "\n"
)


# ============================================================
# 5. READ DATA
# ============================================================

single_gene <- read.csv(
  input_file,
  stringsAsFactors = FALSE,
  check.names = FALSE
)


# ============================================================
# 6. CHECK REQUIRED COLUMNS
#
# RRA_Rank replaces the reference script's Final_Rank — this is
# the rank column _robust_rank_aggregation_single_gene() actually
# writes (1 = best, lower is better; same convention as Final_Rank).
# ============================================================

required_columns <- c(
  "Gene",
  "hpa_efficacy",
  "hpa_safety",
  "tabula_safety",
  "RRA_Rank"
)

missing_columns <- setdiff(
  required_columns,
  colnames(single_gene)
)

if (length(missing_columns) > 0) {

  stop(
    paste0(
      "\nERROR: Missing required column(s):\n",
      paste(
        missing_columns,
        collapse = ", "
      ),
      "\n\nAvailable columns are:\n",
      paste(
        colnames(single_gene),
        collapse = ", "
      ),
      "\n"
    )
  )
}


# ============================================================
# 7. PREPARE DATA
# ============================================================

plot_df <- single_gene %>%

  mutate(

    # --------------------------------------------------------
    # Efficacy: HPA efficacy directly
    # --------------------------------------------------------

    Efficacy = as.numeric(hpa_efficacy),

    # --------------------------------------------------------
    # Safety: average of HPA safety and Tabula safety
    # --------------------------------------------------------

    Safety = rowMeans(
      cbind(
        as.numeric(hpa_safety),
        as.numeric(tabula_safety)
      ),
      na.rm = TRUE
    ),

    # --------------------------------------------------------
    # ObjectiveScore: RRA_Rank column.
    # NOTE: this is a RANK, not a score — 1 = best gene.
    # So LOWER ObjectiveScore is better, and everywhere it is
    # used for ranking we sort ASCENDING (not desc()).
    # --------------------------------------------------------

    ObjectiveScore = as.numeric(RRA_Rank),

    # --------------------------------------------------------
    # Convert to percentages
    # --------------------------------------------------------

    efficacy_pct = Efficacy * 100,

    safety_pct = Safety * 100,

    # --------------------------------------------------------
    # Gene label
    # --------------------------------------------------------

    candidate = as.character(Gene)

  ) %>%

  filter(
    !is.na(safety_pct),
    !is.na(efficacy_pct),
    !is.na(ObjectiveScore),
    !is.na(candidate),
    candidate != ""
  )


# ============================================================
# 8. PRINT DATA CHECK
# ============================================================

cat("\n")
cat("Single-gene data prepared:\n")
cat("--------------------------------------------\n")

print(
  plot_df %>%
    select(
      Gene,
      Efficacy,
      Safety,
      ObjectiveScore,
      efficacy_pct,
      safety_pct
    ) %>%
    head(20)
)

cat(
  "--------------------------------------------\n"
)


# ============================================================
# 9. TOP N
#
# Top `top_n` by ObjectiveScore (RRA_Rank, 1 = best) is used ONLY
# to define the highlighted rectangle (thresholds). It is NOT
# the final set that gets plotted/labeled on the right panel.
# Sorted ASCENDING since lower Final_Rank/RRA_Rank = better gene.
# ============================================================

top20 <- plot_df %>%

  arrange(
    ObjectiveScore
  ) %>%

  slice_head(n = top_n)


# ============================================================
# 10. PRINT TOP N (for reference only)
# ============================================================

cat("\n")
cat(paste0("TOP ", top_n, " SINGLE-GENE CANDIDATES (used to define rectangle)\n"))
cat("--------------------------------------------\n")

print(
  top20 %>%
    select(
      Gene,
      Efficacy,
      Safety,
      ObjectiveScore,
      efficacy_pct,
      safety_pct
    )
)

cat(
  "--------------------------------------------\n"
)


# ============================================================
# 11. THRESHOLDS
#
# Vertical:
# Minimum safety among Top N
#
# Horizontal:
# Minimum efficacy among Top N
# ============================================================

rank20_safety <- min(
  top20$safety_pct,
  na.rm = TRUE
)

min_top20_efficacy <- min(
  top20$efficacy_pct,
  na.rm = TRUE
)

cat(
  paste0("Minimum Top-", top_n, " safety:"),
  round(
    rank20_safety,
    2
  ),
  "%\n"
)

cat(
  paste0("Minimum Top-", top_n, " efficacy:"),
  round(
    min_top20_efficacy,
    2
  ),
  "%\n"
)


# ============================================================
# 12. ZOOM / HIGHLIGHT RECTANGLE
# ============================================================

zoom_x_min <- max(
  0,
  floor(
    min(
      c(
        top20$safety_pct,
        rank20_safety
      ),
      na.rm = TRUE
    )
  ) - 1
)

zoom_x_max <- min(
  100,
  ceiling(
    max(
      c(
        top20$safety_pct,
        rank20_safety
      ),
      na.rm = TRUE
    )
  ) + 1
)

zoom_y_min <- max(
  0,
  floor(
    min(
      c(
        top20$efficacy_pct,
        min_top20_efficacy
      ),
      na.rm = TRUE
    )
  ) - 1
)

zoom_y_max <- min(
  100,
  ceiling(
    max(
      top20$efficacy_pct,
      na.rm = TRUE
    )
  ) + 1
)

cat("\n")
cat("Zoom / highlight rectangle:\n")

cat(
  "X:",
  zoom_x_min,
  "-",
  zoom_x_max,
  "%\n"
)

cat(
  "Y:",
  zoom_y_min,
  "-",
  zoom_y_max,
  "%\n"
)


# ============================================================
# 13. GENES INSIDE THE HIGHLIGHTED RECTANGLE
#
# This is the KEY behaviour: instead of only using the top N,
# we pull EVERY gene whose point falls inside the rectangle
# drawn on the left panel. These are the genes that get
# colored diamonds + labels on BOTH panels.
# ============================================================

highlighted <- plot_df %>%

  filter(
    safety_pct   >= zoom_x_min,
    safety_pct   <= zoom_x_max,
    efficacy_pct >= zoom_y_min,
    efficacy_pct <= zoom_y_max
  ) %>%

  arrange(
    ObjectiveScore
  ) %>%

  mutate(
    label = candidate,

    # --------------------------------------------------------
    # Rank by ObjectiveScore (RRA_Rank, 1 = best). Data is
    # already arranged ascending by ObjectiveScore above, so
    # row_number() gives the correct rank directly.
    # --------------------------------------------------------

    rank = row_number()
  )

n_highlighted <- nrow(highlighted)

cat("\n")
cat("Genes inside highlighted rectangle:", n_highlighted, "\n")
cat("--------------------------------------------\n")

print(
  highlighted %>%
    select(
      rank,
      Gene,
      Efficacy,
      Safety,
      ObjectiveScore,
      efficacy_pct,
      safety_pct
    )
)

cat(
  "--------------------------------------------\n"
)


# ============================================================
# 14. COLOR PALETTE FOR HIGHLIGHTED GENES
#
# Colorblind-friendly but still bright/vibrant. Built from the
# Okabe-Ito colorblind-safe hue set (orange, sky blue, bluish
# green, yellow, blue, vermillion, reddish purple), then
# extended with additional hues + lightness variants chosen so
# that genes remain distinguishable by BOTH hue and brightness
# — this keeps the palette usable for deuteranopia,
# protanopia, and tritanopia, not just standard vision. If more
# than 21 genes fall inside the rectangle, the palette is
# smoothly extended with colorRampPalette so every gene still
# gets a unique, consistent color across both panels.
# ============================================================

base_palette <- c(
  "#E69F00",  # orange            (Okabe-Ito)
  "#0072B2",  # blue              (Okabe-Ito)
  "#009E73",  # bluish green      (Okabe-Ito)
  "#CC79A7",  # reddish purple    (Okabe-Ito)
  "#F0E442",  # yellow            (Okabe-Ito)
  "#56B4E9",  # sky blue          (Okabe-Ito)
  "#D55E00",  # vermillion        (Okabe-Ito)

  "#7B2CBF",  # vivid violet
  "#B37400",  # deep amber
  "#00A896",  # vivid teal
  "#E63946",  # vivid coral red
  "#3A86FF",  # vivid azure
  "#8C4A6B",  # deep magenta
  "#FFB400",  # golden yellow
  "#1D3557",  # deep navy
  "#43AA8B",  # sea green
  "#9D4EDD",  # bright purple
  "#F4A261",  # warm sand orange
  "#118AB2",  # cerulean blue
  "#6A994E",  # olive green
  "#FF6B6B"   # bright coral
)

if (n_highlighted <= length(base_palette)) {

  highlight_colors <- base_palette[seq_len(n_highlighted)]

} else {

  highlight_colors <- colorRampPalette(base_palette)(n_highlighted)
}

highlighted <- highlighted %>%

  mutate(
    color = highlight_colors
  )


# ============================================================
# 14b. "NICE" Y-AXIS BREAK STEP (fixes overcrowded right-panel
# axis when the highlighted set spans a wide efficacy range)
#
# The right panel's y-axis previously used a hard-coded
# `by = 2` break step across the FULL zoomed range. That is
# fine when the highlighted genes are tightly clustered (a
# ~10-20 point spread), but when they span a much wider range
# (e.g. one gene at ~80% efficacy and another at ~26%), a fixed
# 2-point step produces 25+ bold, size-18 tick labels crammed
# onto one axis — they visibly overlap each other and the axis
# title. This picks a step from the same "nice round number"
# family (1/2/2.5/5/10 x 10^n) matplotlib-style tick locators
# use, targeting ~6 ticks regardless of how wide the range is,
# so the axis stays readable either way.
# ============================================================

nice_breaks <- function(lo, hi, target_n = 6) {

  data_range <- hi - lo

  if (data_range <= 0) {
    return(pretty(c(lo, hi)))
  }

  raw_step  <- data_range / target_n
  magnitude <- 10 ^ floor(log10(raw_step))
  residual  <- raw_step / magnitude

  nice <- if (residual <= 1) {
    1
  } else if (residual <= 2) {
    2
  } else if (residual <= 2.5) {
    2.5
  } else if (residual <= 5) {
    5
  } else {
    10
  }

  step <- nice * magnitude

  seq(
    ceiling(lo / step) * step,
    floor(hi / step) * step,
    by = step
  )
}


# ============================================================
# 15. LEFT PLOT — ALL SINGLE GENES
# ============================================================

p_left <- ggplot() +

  # ----------------------------------------------------------
  # Zoom / highlight rectangle (drawn first so points sit
  # on top of it, not behind it)
  # ----------------------------------------------------------

  annotate(

    "rect",

    xmin = zoom_x_min,

    xmax = zoom_x_max,

    ymin = zoom_y_min,

    ymax = zoom_y_max,

    fill = "#EDEEF3",

    alpha = 0.9,

    color = "black",

    linetype = "dashed",

    linewidth = 1.0
  ) +

  # ----------------------------------------------------------
  # Vertical threshold
  # ----------------------------------------------------------

  geom_vline(

    xintercept = rank20_safety,

    linetype = "dashed",

    linewidth = 0.8,

    color = "grey30"
  ) +

  # ----------------------------------------------------------
  # Horizontal threshold
  # ----------------------------------------------------------

  geom_hline(

    yintercept = min_top20_efficacy,

    linetype = "dashed",

    linewidth = 0.8,

    color = "grey30"
  ) +

  # ----------------------------------------------------------
  # Background genes (outside the highlighted rectangle)
  # ----------------------------------------------------------

  geom_point(

    data = plot_df %>%
      filter(
        !Gene %in% highlighted$Gene
      ),

    aes(
      x = safety_pct,
      y = efficacy_pct
    ),

    color = "grey60",

    alpha = 0.55,

    size = 2.6
  ) +

  # ----------------------------------------------------------
  # Genes inside the highlighted rectangle
  # ----------------------------------------------------------

  geom_point(

    data = highlighted,

    aes(
      x = safety_pct,
      y = efficacy_pct,
      fill = label
    ),

    shape = 23,

    color = "black",

    size = 4.2,

    stroke = 0.6
  ) +

  # ----------------------------------------------------------
  # Colors
  # ----------------------------------------------------------

  scale_fill_manual(

    values = setNames(
      highlighted$color,
      highlighted$label
    ),

    guide = "none"
  ) +

  # ----------------------------------------------------------
  # X AXIS
  # ----------------------------------------------------------

  scale_x_continuous(

    labels = function(x) {
      paste0(
        x,
        "%"
      )
    },

    expand = expansion(
      mult = 0.03
    )
  ) +

  # ----------------------------------------------------------
  # Y AXIS
  # ----------------------------------------------------------

  scale_y_continuous(

    labels = function(y) {
      paste0(
        y,
        "%"
      )
    },

    expand = expansion(
      mult = 0.03
    )
  ) +

  # ----------------------------------------------------------
  # LABELS
  # ----------------------------------------------------------

  labs(

    title = "All single-gene candidates",

    x = "Safety",

    y = "Efficacy"
  ) +

  # ----------------------------------------------------------
  # THEME
  # ----------------------------------------------------------

  theme_classic(

    base_size = 18,

    base_family = "Times New Roman"

  ) +

  theme(

    text = element_text(
      family = "Times New Roman",
      color = "black"
    ),

    plot.title = element_text(
      family = "Times New Roman",
      size = 23,
      face = "bold",
      color = "black",
      hjust = 0.5
    ),

    axis.title.x = element_text(
      family = "Times New Roman",
      size = 23,
      face = "bold",
      color = "black",
      margin = margin(t = 8)
    ),

    axis.title.y = element_text(
      family = "Times New Roman",
      size = 23,
      face = "bold",
      color = "black",
      margin = margin(r = 12)
    ),

    axis.text.x = element_text(
      family = "Times New Roman",
      size = 18,
      face = "bold",
      color = "black"
    ),

    axis.text.y = element_text(
      family = "Times New Roman",
      size = 18,
      face = "bold",
      color = "black"
    ),

    axis.line = element_line(
      linewidth = 1,
      color = "black"
    ),

    panel.border = element_rect(
      color = "black",
      fill = NA,
      linewidth = 1
    ),

    panel.background = element_rect(
      fill = "#FFFFFF",
      color = NA
    ),

    panel.grid = element_blank(),

    plot.background = element_rect(
      fill = "#FFFFFF",
      color = NA
    ),

    plot.margin = margin(
      10,
      6,
      10,
      10
    ),

    aspect.ratio = 1
  )


# ============================================================
# 16. RIGHT PLOT — ZOOM INTO RECTANGLE
#
# Every gene inside the rectangle is plotted AND labeled here,
# not just the top N.
# ============================================================

p_right <- ggplot(

  highlighted,

  aes(
    x = safety_pct,
    y = efficacy_pct
  )
) +

  # ----------------------------------------------------------
  # Vertical threshold
  # ----------------------------------------------------------

  geom_vline(

    xintercept = rank20_safety,

    linetype = "dashed",

    linewidth = 0.8,

    color = "grey30"
  ) +

  # ----------------------------------------------------------
  # Horizontal threshold
  # ----------------------------------------------------------

  geom_hline(

    yintercept = min_top20_efficacy,

    linetype = "dashed",

    linewidth = 0.8,

    color = "grey30"
  ) +

  # ----------------------------------------------------------
  # Points
  # ----------------------------------------------------------

  geom_point(

    aes(
      fill = label
    ),

    shape = 23,

    color = "black",

    size = 4.8,

    stroke = 0.7
  ) +

  # ----------------------------------------------------------
  # GENE LABELS — ALL genes inside the rectangle
  #
  # Plain white labels with just the gene name. Rank and
  # color are shown separately in the "Gene Ranks" legend
  # panel, so the plot stays uncluttered.
  # ----------------------------------------------------------

  geom_label_repel(

    aes(
      label = label
    ),

    color = "black",

    fill = "white",

    family = "Times New Roman",

    fontface = "bold",

    size = 4.5,

    label.size = 0.5,

    segment.color = "grey30",

    segment.size = 0.5,

    min.segment.length = 0,

    box.padding = 0.5,

    point.padding = 0.3,

    max.overlaps = Inf,

    seed = 42
  ) +

  # ----------------------------------------------------------
  # Fill colors
  # ----------------------------------------------------------

  scale_fill_manual(

    values = setNames(
      highlighted$color,
      highlighted$label
    ),

    guide = "none"
  ) +

  # ----------------------------------------------------------
  # X AXIS
  # ----------------------------------------------------------

  scale_x_continuous(

    limits = c(
      zoom_x_min,
      zoom_x_max
    ),

    labels = function(x) {
      paste0(
        x,
        "%"
      )
    },

    expand = expansion(
      mult = 0.08
    )
  ) +

  # ----------------------------------------------------------
  # Y AXIS
  # ----------------------------------------------------------

  scale_y_continuous(

    limits = c(
      zoom_y_min,
      zoom_y_max
    ),

    breaks = nice_breaks(zoom_y_min, zoom_y_max),

    labels = function(y) {
      paste0(
        y,
        "%"
      )
    },

    expand = expansion(
      mult = 0.08
    )
  ) +

  # ----------------------------------------------------------
  # LABELS
  # ----------------------------------------------------------

  labs(

    title = paste0(
      "Candidates in highlighted region"
    ),

    x = "Safety",

    y = "Efficacy"
  ) +

  # ----------------------------------------------------------
  # THEME
  # ----------------------------------------------------------

  theme_classic(

    base_size = 20,

    base_family = "Times New Roman"

  ) +

  theme(

    text = element_text(
      family = "Times New Roman",
      color = "black"
    ),

    plot.title = element_text(
      family = "Times New Roman",
      size = 23,
      face = "bold",
      color = "black",
      hjust = 0.5
    ),

    axis.title.x = element_text(
      family = "Times New Roman",
      size = 23,
      face = "bold",
      color = "black",
      margin = margin(t = 8)
    ),

    axis.title.y = element_text(
      family = "Times New Roman",
      size = 23,
      face = "bold",
      color = "black",
      margin = margin(r = 12)
    ),

    axis.text.x = element_text(
      family = "Times New Roman",
      size = 18,
      face = "bold",
      color = "black"
    ),

    axis.text.y = element_text(
      family = "Times New Roman",
      size = 18,
      face = "bold",
      color = "black"
    ),

    axis.line = element_line(
      linewidth = 1,
      color = "black"
    ),

    panel.border = element_rect(
      color = "black",
      fill = NA,
      linewidth = 1
    ),

    panel.background = element_rect(
      fill = "#EDEEF3",
      color = NA
    ),

    panel.grid = element_blank(),

    plot.background = element_rect(
      fill = "#FFFFFF",
      color = NA
    ),

    plot.margin = margin(
      10,
      4,
      10,
      6
    ),

    aspect.ratio = 1
  )


# ============================================================
# 17. RANK / COLOR LEGEND PANEL
#
# A standalone panel listing every highlighted gene as a
# color swatch next to its name (ordered by rank), so the
# color <-> gene mapping is explicit without cluttering the
# scatter plot with rank numbers.
# ============================================================

legend_df <- highlighted %>%

  arrange(
    rank
  ) %>%

  mutate(
    y_pos = -rank
  )

p_legend <- ggplot(

  legend_df,

  aes(
    x = 0,
    y = y_pos
  )
) +

  geom_point(

    aes(
      fill = label
    ),

    shape = 22,

    color = "black",

    size = 5.5,

    stroke = 0.5
  ) +

  geom_text(

    aes(
      label = paste0(
        rank,
        ".  ",
        label
      )
    ),

    hjust = 0,

    nudge_x = 0.18,

    family = "Times New Roman",

    fontface = "bold",

    size = 4.2,

    color = "black"
  ) +

  scale_fill_manual(

    values = setNames(
      legend_df$color,
      legend_df$label
    ),

    guide = "none"
  ) +

  xlim(
    -0.1,
    2.3
  ) +

  labs(
    title = "Gene Ranks"
  ) +

  theme_void(

    base_family = "Times New Roman"

  ) +

  theme(

    plot.title = element_text(
      family = "Times New Roman",
      size = 16,
      face = "bold",
      color = "black",
      hjust = 0
    ),

    plot.title.position = "plot",

    plot.margin = margin(
      10,
      4,
      10,
      0
    )
  )


# ============================================================
# 18. COMBINE INTO A SINGLE FIGURE
# ============================================================

p_combined <- cowplot::plot_grid(

  p_left,

  p_right,

  p_legend,

  ncol = 3,

  align = "h",

  axis = "tb",

  rel_widths = c(1, 1, 0.26)
)


# ============================================================
# 19. SAVE COMBINED PLOT
# ============================================================

cat("\n")
cat("Saving COMBINED plot...\n")

ggsave(

  filename = combined_file,

  plot = p_combined,

  width = 16.5,

  height = 8,

  units = "in",

  dpi = 800,

  bg = "#FFFFFF"
)

cat(
  "COMBINED plot saved successfully:\n",
  combined_file,
  "\n"
)


# ============================================================
# 20. VERIFY OUTPUT FILE
# ============================================================

cat("\n")
cat("============================================================\n")
cat("OUTPUT VERIFICATION\n")
cat("============================================================\n")

if (file.exists(combined_file)) {

  combined_size <- file.info(combined_file)$size / (1024^2)

  cat(
    "COMBINED plot: SUCCESS\n",
    "File: ",
    combined_file,
    "\n",
    "Size: ",
    round(combined_size, 2),
    " MB\n\n",
    sep = ""
  )

} else {

  cat(
    "COMBINED plot: FAILED\n\n"
  )

  quit(status = 1)
}


# ============================================================
# 21. FINAL REPORT
# ============================================================

cat("============================================================\n")
cat("SINGLE-GENE ANALYSIS COMPLETED\n")
cat("============================================================\n")

cat(
  "Input:\n",
  input_file,
  "\n\n"
)

cat(
  "Number of candidates:",
  nrow(plot_df),
  "\n\n"
)

cat(
  "Genes inside highlighted rectangle (labeled on right panel):",
  n_highlighted,
  "\n"
)

print(
  highlighted %>%
    select(
      rank,
      Gene,
      Efficacy,
      Safety,
      ObjectiveScore
    )
)

cat("\n")

cat(
  paste0("Minimum Top-", top_n, " safety (threshold):"),
  round(rank20_safety, 2),
  "%\n"
)

cat(
  paste0("Minimum Top-", top_n, " efficacy (threshold):"),
  round(min_top20_efficacy, 2),
  "%\n\n"
)

cat(
  "High-resolution output file:\n\n"
)

cat(
  combined_file,
  "\n\n"
)

cat(
  "Resolution: 800 DPI\n"
)

cat(
  "Dimensions: 16.5 x 8 inches (all genes | zoom | gene ranks)\n"
)

cat(
  "Format: PNG\n"
)

cat("============================================================\n")
"""


def _display_png_if_notebook(png_path: str) -> None:
    """
    Display a rendered PNG inline when running inside a Jupyter kernel —
    mirrors the inline-display behaviour the earlier matplotlib-based
    plotting had (via _configure_matplotlib_backend() + plt.show(), see
    module docstring "Fix applied (RRA plot v2 ...)"). Since the plot is
    now rendered by an Rscript subprocess rather than in-process, there is
    no matplotlib figure object to show — this displays the saved PNG
    file directly via IPython's rich display instead, which gives the
    same "just ran a cell and saw the plot" experience in a notebook.
    No-op outside a notebook (headless script runs), and if the file
    doesn't exist for some reason.
    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
        in_notebook = ip is not None and (
            "IPKernelApp" in ip.config or type(ip).__name__ == "ZMQInteractiveShell"
        )
    except Exception:
        in_notebook = False

    if not in_notebook:
        return

    if not os.path.exists(png_path):
        logger.warning(f"Expected plot PNG not found for inline display: {png_path}")
        return

    try:
        from IPython.display import display, Image
        display(Image(filename=png_path))
    except Exception as exc:
        logger.warning(f"Could not display plot inline: {exc}")


def _plot_single_gene_rra_r(rra_csv_path: str, output_dir: str, top_n: int = 20) -> None:
    """
    Render the single-gene RRA "highlighted region" figure natively in R.

    Fills _SINGLE_GENE_RRA_PLOT_R_TEMPLATE's three placeholders
    (@@INPUT_CSV@@, @@OUTPUT_DIR@@, @@TOP_N@@) with the actual values for
    this run, writes the result to a driver script under output_dir, and
    launches it as an Rscript subprocess — the same subprocess pattern
    _run_rra_via_r() already uses for RobustRankAggreg, reused here via
    _find_rscript()/_find_r_home()/_build_r_subprocess_env().

    The R script itself defines the highlighted-region logic: the top
    `top_n` genes by RRA_Rank (default 20) set a safety/efficacy rectangle,
    and every gene whose point falls inside that rectangle — not just
    those top `top_n` — gets plotted, labelled (via ggrepel) and listed in
    a "Gene Ranks" legend panel, combined with cowplot.

    Parameters
    ----------
    rra_csv_path : str
        Path to the RRA-ranked CSV — exactly what
        _robust_rank_aggregation_single_gene() already writes to
        final_single_gene_candidates_RRA_HPA_Tabula.csv. Must contain the
        columns Gene, hpa_efficacy, hpa_safety, tabula_safety, RRA_Rank.
    output_dir : str
        Directory the figure is written into. The R script creates a
        SCATTER_PLOTS_FIGURE3 subfolder here.
    top_n : int
        Number of top RRA_Rank genes used to define the highlighted
        rectangle (default 20).
    """
    import subprocess

    rscript_path = _find_rscript()
    r_home       = _find_r_home()
    env          = _build_r_subprocess_env(r_home)

    plot_dir = os.path.join(output_dir, "rra_plot_rscript")
    os.makedirs(plot_dir, exist_ok=True)
    r_script_path = os.path.join(plot_dir, "plot_single_gene_rra.R")

    r_code = (
        _SINGLE_GENE_RRA_PLOT_R_TEMPLATE
        .replace("@@INPUT_CSV@@", os.path.abspath(rra_csv_path).replace("\\", "/"))
        .replace("@@OUTPUT_DIR@@", os.path.abspath(output_dir).replace("\\", "/"))
        .replace("@@TOP_N@@", str(int(top_n)))
    )
    with open(r_script_path, "w") as f:
        f.write(r_code)

    print(f"  Rscript:                  {rscript_path}")
    print(f"  R home:                   {r_home}")
    print(f"  Plot driver script:       {r_script_path}")
    print("  Launching Rscript subprocess for the single-gene RRA plot ...")

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
            "Rscript subprocess for the single-gene RRA plot failed "
            f"(exit code {proc.returncode}).\n"
            f"--- Rscript stderr ---\n{proc.stderr}\n"
            "Make sure ggplot2, dplyr, cowplot and ggrepel are installed "
            f"in the R at {r_home}, e.g.:\n"
            "  conda install -c conda-forge r-ggplot2 r-dplyr r-cowplot r-ggrepel -y"
        )

    if proc.stderr:
        # R (and library()) often write benign package-load / startup
        # messages to stderr even on success — log, don't fail on these.
        logger.info(f"Rscript stderr (non-fatal):\n{proc.stderr.strip()}")

    print("  Single-gene RRA plot rendered via R.")

    png_path = os.path.join(
        output_dir, "SCATTER_PLOTS_FIGURE3",
        "Single_gene_efficacy_vs_safety_COMBINED_claude.png",
    )
    _display_png_if_notebook(png_path)

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

    _plot_single_gene_rra_r(out_csv, output_dir)

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
