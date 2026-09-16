#!/usr/bin/env python
# coding: utf-8
"""
two_gene_combination.py
Module 4b — Two-gene logic-gate CAR-T target evaluation (Genetic Algorithm)

Searches over all (geneA, geneB, logic_gate) combinations using a Genetic
Algorithm (DEAP) to find pairs that maximise tumour killing (efficacy) while
sparing healthy tissue (safety).

Logic gates:
  A & B   — both genes must be expressed  (AND)
  A | B   — either gene expressed         (OR)
  A & !B  — A expressed, B NOT expressed  (NOT-B gate)

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

If a file can't be found, run() raises a FileNotFoundError with these same
instructions.

The user selects which atlas(es) to score safety against via the `atlas`
argument of run():

  atlas="hpa"    -> GA search scored against HPA only
  atlas="tabula" -> GA search scored against Tabula Sapiens only
  atlas="both"   -> GA search run independently against EACH atlas,
                     individual per-atlas results are saved, and the two
                     ranked candidate lists are then combined with Robust
                     Rank Aggregation (RRA) into a single consensus ranking.

Genetic algorithm
-----------------
Pair search uses an island-model GA: gate-quota population seeding (a fixed
share of the starting population is pre-seeded as A&B / A&!B / open-gate
individuals), a multi-island model with periodic ring migration, gate-quota
tournament selection (guarantees a minimum share of each gate type survives
selection), SBX (simulated binary bounded) crossover, rare-gene immigrant
injection during the periodic diversity-injection step, and per-seed
parallelism via joblib. See _run_ga() below. There is no alternate GA mode.

Fix applied
-----------
_load_h5ad_subset: same h5py sorted-indices fix as one_gene_combination.py.

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
partially-failed import earlier in a long-lived kernel — the joblib
multiprocessing workers used by the GA make this worse, since a fork can
inherit an already-partially-initialised rpy2 state). Because Python caches
failed imports in sys.modules, once rpy2 fails once in a process it cannot
be retroactively repaired without restarting the kernel — which is fragile
for a long pipeline run.

Fix (current): the RRA step no longer embeds R in-process via rpy2 at all.
Instead — mirroring exactly how the SCART preprocessing module already
drives R successfully for scMalignantFinder and SCEVAN, and identical to
the fix applied in Module 4a (one_gene_combination.py) — it now:
  1. writes each per-atlas ranked candidate-ID list out to a small CSV file,
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

Fix applied (RRA plots)
------------------------
The single grouped-bar "Top 20 RRA candidates" plot has been replaced with
two dual-panel scatter plots, adapted from a reference R/ggplot2+ggrepel
script (avg. safety vs HPA efficacy, all candidates as background, selected
candidates as labelled diamonds with a dashed rectangle marking the zoomed
region on the left panel):

  _plot_top_gate_rra_candidates() — top 5 candidates by RRA_Rank within
      EACH gate type (OR, AND, NAND — A & !B), shown on a three-panel
      figure: all candidates with a zoom rectangle, a zoomed-in labelled
      panel, and a "Gene-Ranks" legend panel — ported from a newer
      reference R/ggplot2+ggrepel+cowplot dual-gene script. Colour is a
      light-to-dark gradient within each gate (rank 1 = darkest); shape is
      fixed per gate (diamond = OR, square = AND, triangle = NAND).
  _plot_top10_rra_candidates()    — top 10 candidates overall by RRA_Rank,
      regardless of gate type.

Both are shown inline (in addition to being saved to disk) when run inside
a Jupyter kernel; see _configure_matplotlib_backend().

Fix applied (top-gate RRA plot — plotted natively in R)
------------------------------------------------------------
_plot_top_gate_rra_candidates() was a matplotlib approximation of a
reference R/ggplot2 + ggrepel + cowplot dual-gene script. It is now that
reference script itself, embedded directly in this module: the design
(gate-label mapping, pair-label building with the original gate symbol
between the two gene names, top-N-per-gate highlighting, per-gate colour
gradients + shapes, thresholds, rectangle, labelled zoom panel, "Gene-
Ranks" legend panel grouped by gate, cowplot::plot_grid combine) lives in
_TOP_GATE_RRA_PLOT_R_TEMPLATE as an R source template — no separate .R
file to ship or locate, matching the same approach already used in Module
4a (one_gene_combination.py) for its single-gene plot.
_plot_top_gate_rra_candidates_r() fills in the three values that vary per
run (input CSV path, output dir, top_n_per_gate) via simple string
substitution, writes the result to a driver script under output_dir, and
launches it as an Rscript subprocess — reusing
_find_rscript()/_find_r_home()/_build_r_subprocess_env() as-is, unchanged.
The only changes from the reference script are: the rank column is
"RRA_Rank" (what _robust_rank_aggregation() actually writes) instead of
"RRA_rank", and the output file carries a "_claude" suffix.

_plot_top10_rra_candidates() ("top 10 overall regardless of gate") is
unchanged and still rendered in matplotlib via _dual_panel_rra_figure() —
no reference R script was given for that plot, so it was left as-is.

The now-unused matplotlib helpers specific to the old top-gate plot
(_dual_panel_with_legend_rra_figure, _GATE_HUE_ENDS, _GATE_MARKERS,
_color_ramp) have been removed. _GATE_TYPE_MAP and _RANK_PALETTE remain —
both are still used by the retained top-10-overall plot / its shared
_prep_rra_plot_df() prep step.

New R dependency: none beyond what Module 4a's single-gene plot already
added — ggplot2, dplyr, ggrepel and cowplot are all already required (see
install.py's CONDA_R_BASE, which already includes r-cowplot).
"""

import os
import zipfile
import urllib.request
import logging
import random

import numpy as np
import pandas as pd
import scanpy as sc
from deap import base, creator, tools, algorithms
from joblib import Parallel, delayed

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

    FIX: col_indices sorted before h5py dense indexing; un-permuted after.
    Scipy sparse paths use raw (unsorted) indices — they accept any order.

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

        raw_col_indices = np.array([gene_index[g] for g in common], dtype=np.int32)

        # FIX: sort for h5py; restore caller order after read
        sort_order     = np.argsort(raw_col_indices)
        sorted_indices = raw_col_indices[sort_order]
        restore_order  = np.argsort(sort_order)

        print(f"  HPA h5ad: {len(all_genes)} genes total — "
              f"extracting {len(common)} overlapping genes directly via h5py.")

        x_grp = f["X"]

        if isinstance(x_grp, h5py.Dataset):
            X_sorted = x_grp[:, sorted_indices]
            X_sub    = X_sorted[:, restore_order]

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
                X_sub   = full[:, raw_col_indices].toarray()

            elif "csc" in encoding:
                data    = x_grp["data"][:]
                indices = x_grp["indices"][:]
                indptr  = x_grp["indptr"][:]
                shape   = tuple(x_grp.attrs["shape"])
                full    = _sp.csc_matrix((data, indices, indptr), shape=shape)
                X_sub   = full[:, raw_col_indices].toarray()

            else:
                logger.warning("Unknown X encoding — falling back to scanpy backed mode.")
                adata_backed = sc.read_h5ad(h5ad_path, backed="r")
                adata_sub    = adata_backed[:, common].to_memory()
                adata_backed.file.close()
                X_full = adata_sub.X
                X_sub  = X_full.toarray() if _sp.issparse(X_full) else np.asarray(X_full)
        else:
            raise ValueError(f"Unrecognised X format in {h5ad_path}")

    return (X_sub > 0).astype(np.int8), common


def _load_healthy_matrix(hpa_path=None, target_genes=None):
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
            matrix, genes = _load_h5ad_subset(legacy, target_genes)
            return matrix, genes, f"legacy: {legacy}"

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


# ─────────────────────────────────────────────────────────────────────────────
# Logic gates
# ─────────────────────────────────────────────────────────────────────────────

LOGIC_GATES = ["A & B", "A | B", "A & !B"]


def evaluate_gate(expression: str, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if expression == "A & B":
        return A & B
    elif expression == "A | B":
        return A | B
    elif expression == "A & !B":
        return A & (~B.astype(bool))
    raise ValueError(f"Unsupported logic expression: {expression}")


# Module-level state (set inside _run_single_atlas() before GA runs start)
_gene_names     = None
_n_genes        = None
_safety_thresh  = 0.9
_logic_gates    = LOGIC_GATES


# ─────────────────────────────────────────────────────────────────────────────
# Genetic algorithm — island model with gate-quota seeding
#
# Ported directly from the user's reference CAR-T GA script: gate-quota
# population seeding, a multi-island model with ring migration, gate-quota
# tournament selection, SBX (simulated binary bounded) crossover, and
# rare-gene immigrant injection during diversity-injection steps. This is
# the module's only GA implementation — there is no alternate "simple" mode.
# ─────────────────────────────────────────────────────────────────────────────

toolbox = None


def _normalize_gene_pair(genes):
    return tuple(sorted(genes))


def _postprocess_results(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.iloc[0]["Genes"], str):
        df["Genes"] = df["Genes"].apply(eval)
    df = df[df["Genes"].apply(lambda g: g[0] != g[1])].copy()
    df["GenePairKey"] = df["Genes"].apply(_normalize_gene_pair)
    df = df.sort_values(by="Efficacy", ascending=False)
    df = df.drop_duplicates(subset=["GenePairKey"], keep="first")
    return df.drop(columns=["GenePairKey"]).reset_index(drop=True)


# Per-gene contiguous column caches. Precomputing these avoids the
# strided-slice cache-miss penalty seen on very large matrices when
# indexing a 2D matrix column-wise repeatedly inside the GA's hot loop.
_tumor_cols   = None
_healthy_cols = None


def _init_deap(n_genes: int):
    global toolbox

    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    tb = base.Toolbox()
    tb.register("geneA",      random.randrange, n_genes)
    tb.register("geneB",      random.randrange, n_genes)
    tb.register("gate",       random.randrange, len(LOGIC_GATES))
    tb.register("individual", tools.initCycle, creator.Individual,
                 (tb.geneA, tb.geneB, tb.gate), n=1)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate",   _evaluate_fitness)
    tb.register("mutate",     tools.mutUniformInt,
                 low=[0, 0, 0],
                 up=[n_genes - 1, n_genes - 1, len(LOGIC_GATES) - 1],
                 indpb=0.2)

    toolbox = tb
    return tb


def _evaluate_fitness(individual):
    geneA_idx, geneB_idx, gate_type_idx = individual
    gate_type = LOGIC_GATES[gate_type_idx]

    A_tumor   = _tumor_cols[geneA_idx]
    B_tumor   = _tumor_cols[geneB_idx]
    A_healthy = _healthy_cols[geneA_idx]
    B_healthy = _healthy_cols[geneB_idx]

    output_tumor   = evaluate_gate(gate_type, A_tumor,   B_tumor)
    output_healthy = evaluate_gate(gate_type, A_healthy, B_healthy)

    efficacy = np.sum(output_tumor)        / len(output_tumor)
    safety   = np.sum(output_healthy == 0) / len(output_healthy)

    individual.safety = safety
    return (efficacy if safety >= _safety_thresh else 0,)


def _evaluate_individual(ind):
    ind.fitness.values = toolbox.evaluate(ind)
    return ind


def _round_back(ind, low, up):
    """Clamp and round SBX's float outputs back to valid integer gene/gate
    indices, in-place."""
    for i in range(3):
        ind[i] = int(round(max(low[i], min(up[i], ind[i]))))


def _cx_simulated_binary_bounded(ind1, ind2, low, up, eta=2.0):
    """SBX crossover with hard index bounds, rounded back to valid ints."""
    tools.cxSimulatedBinaryBounded(ind1, ind2, eta=eta, low=low, up=up)
    _round_back(ind1, low, up)
    _round_back(ind2, low, up)
    return ind1, ind2


def _init_islands(n_genes, n_islands, and_quota, nand_quota, open_quota):
    """Gate-quota, region-seeded island population initializer: each island
    is seeded with a fixed share of A&B, A&!B, and open-gate individuals,
    with geneA drawn from that island's private gene-index region (so each
    island starts exploring a different slice of gene space)."""
    region_size  = n_genes // n_islands
    gene_regions = [
        range(i * region_size,
              (i + 1) * region_size if i < n_islands - 1 else n_genes)
        for i in range(n_islands)
    ]

    islands = []
    for isl_idx in range(n_islands):
        region     = list(gene_regions[isl_idx])
        island_pop = []

        def make_ind(gate_idx, region=region):
            ind = toolbox.individual()
            ind[0] = random.choice(region)
            ind[1] = random.randrange(n_genes)
            while ind[1] == ind[0]:
                ind[1] = random.randrange(n_genes)
            ind[2] = gate_idx
            return ind

        for _ in range(and_quota // n_islands):
            island_pop.append(make_ind(gate_idx=0))    # "A & B"
        for _ in range(nand_quota // n_islands):
            island_pop.append(make_ind(gate_idx=2))    # "A & !B"
        for _ in range(open_quota // n_islands):
            ind = toolbox.individual()
            ind[0] = random.choice(region)
            ind[1] = random.randrange(n_genes)
            while ind[1] == ind[0]:
                ind[1] = random.randrange(n_genes)
            island_pop.append(ind)

        random.shuffle(island_pop)
        islands.append(island_pop)

    return islands


def _select_with_gate_quota(population, k, tournsize=2, gate_min_frac=0.2):
    """Tournament selection that guarantees at least `gate_min_frac` of the
    `k` selected individuals carry each gate type (prevents any one gate
    type from being selected out of existence)."""
    min_per_gate = int(k * gate_min_frac)

    gate_pools = {i: [] for i in range(len(LOGIC_GATES))}
    for ind in population:
        gate_pools[ind[2]].append(ind)

    selected = []
    for gate_idx in range(len(LOGIC_GATES)):
        pool     = gate_pools[gate_idx]
        n_select = min(min_per_gate, len(pool))
        for _ in range(n_select):
            aspirants = random.choices(pool, k=min(tournsize, len(pool)))
            selected.append(max(aspirants, key=lambda x: x.fitness.values[0]))

    remaining = k - len(selected)
    for _ in range(remaining):
        aspirants = random.choices(population, k=tournsize)
        selected.append(max(aspirants, key=lambda x: x.fitness.values[0]))

    random.shuffle(selected)
    return selected


def _migrate_islands(islands, n_islands, migrate_k):
    """Ring-topology migration: the top `migrate_k` individuals from each
    island replace the bottom `migrate_k` of the next island in the ring."""
    migrants = []
    for island in islands:
        island.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        migrants.append([toolbox.clone(ind) for ind in island[:migrate_k]])

    for i, island in enumerate(islands):
        incoming = migrants[(i - 1) % n_islands]
        island[-migrate_k:] = incoming

    return islands


def _run_ga(
    seed, n_genes, island_size, n_islands,
    and_quota, nand_quota, open_quota,
    Gmax, Ggap, Rrep, patience,
    gate_min_frac, mutpb, sbx_eta,
    migrate_interval, migrate_k,
):
    """
    Single-seed island-model GA run: gate-quota init, ring migration,
    gate-quota selection, SBX crossover, rare-gene immigrant injection.

    Meant to be called once per seed — parallelised ACROSS seeds via joblib
    in _run_single_atlas() (each seed runs single-process; unlike the
    standard GA's per-generation multiprocessing, this is per-generation
    single-threaded, per-seed parallel).
    """
    random.seed(seed)
    np.random.seed(seed)

    low = [0, 0, 0]
    up  = [n_genes - 1, n_genes - 1, len(LOGIC_GATES) - 1]
    toolbox.register("mate", _cx_simulated_binary_bounded, low=low, up=up, eta=sbx_eta)

    islands = _init_islands(n_genes, n_islands, and_quota, nand_quota, open_quota)

    hof   = tools.HallOfFame(100)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    max_fitness                 = 0
    generations_without_improve = 0
    logbook                     = []
    all_results                 = []

    for gen in range(Gmax):

        for isl_idx, island in enumerate(islands):

            offspring = algorithms.varAnd(island, toolbox, cxpb=0.5, mutpb=mutpb)
            offspring = list(map(_evaluate_individual, offspring))

            for ind in offspring:
                gA, gB, gT = ind
                all_results.append([
                    gen, LOGIC_GATES[gT],
                    [_gene_names[gA], _gene_names[gB]],
                    ind.fitness.values[0],
                    getattr(ind, "safety", None),
                    seed,
                ])
                ind.generation = gen
                ind.seed_value = seed

            if gen > 0 and gen % Ggap == 0:
                num_replace = max(1, int(Rrep * island_size))
                offspring.sort(key=lambda ind: ind.fitness.values[0])

                hof_genes = set()
                for h in hof:
                    hof_genes.add(h[0]); hof_genes.add(h[1])

                rare_genes    = list(set(range(n_genes)) - hof_genes)
                hof_gene_list = list(hof_genes) if hof_genes else list(range(n_genes))

                num_rare = num_replace // 2

                for i in range(num_replace):
                    new_ind = toolbox.individual()

                    if i < num_rare and rare_genes:
                        forced_gene = random.choice(rare_genes)
                        gate_idx    = new_ind[2]
                        if gate_idx == 2:  # "A & !B" — forced gene goes in slot B
                            new_ind[1] = forced_gene
                            hof_a = random.choice(hof_gene_list)
                            while hof_a == forced_gene:
                                hof_a = random.choice(hof_gene_list)
                            new_ind[0] = hof_a
                        else:
                            if random.random() < 0.5:
                                new_ind[0] = forced_gene
                                new_ind[1] = random.randrange(n_genes)
                                while new_ind[1] == new_ind[0]:
                                    new_ind[1] = random.randrange(n_genes)
                            else:
                                new_ind[1] = forced_gene
                                new_ind[0] = random.randrange(n_genes)
                                while new_ind[0] == new_ind[1]:
                                    new_ind[0] = random.randrange(n_genes)

                    new_ind.fitness.values = toolbox.evaluate(new_ind)
                    gA, gB, gT = new_ind
                    all_results.append([
                        gen, LOGIC_GATES[gT],
                        [_gene_names[gA], _gene_names[gB]],
                        new_ind.fitness.values[0],
                        getattr(new_ind, "safety", None),
                        seed,
                    ])
                    new_ind.generation = gen
                    new_ind.seed_value = seed
                    offspring[i] = new_ind

            islands[isl_idx] = _select_with_gate_quota(
                offspring, k=island_size, tournsize=2, gate_min_frac=gate_min_frac
            )

        if gen > 0 and gen % migrate_interval == 0:
            islands = _migrate_islands(islands, n_islands, migrate_k)

        combined_pop = [ind for island in islands for ind in island]
        hof.update(combined_pop)
        record = stats.compile(combined_pop)
        logbook.append(record)

        current_best = record["max"]
        if current_best > max_fitness:
            max_fitness                 = current_best
            generations_without_improve = 0
        else:
            generations_without_improve += 1

        if generations_without_improve >= patience:
            print(f"  [island] Early stopping at generation {gen} for seed {seed}")
            break

    return hof, logbook, all_results


# ─────────────────────────────────────────────────────────────────────────────
# Single-atlas GA run  (factored out of run() so it can be executed once per
# atlas when atlas="both")
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_atlas(
    atlas_label: str,
    healthy_path: str,
    adata_tumor,
    tumor_genes: list,
    safety_threshold: float,
    pop_size: int,
    Gmax: int,
    Ggap: int,
    Rrep: float,
    patience: int,
    n_runs: int,
    output_dir: str,
    n_islands: int = 4,
    migrate_interval: int = 10,
    migrate_k: int = 10,
    and_quota_frac: float = 0.25,
    nand_quota_frac: float = 0.25,
    gate_min_frac: float = 0.20,
    mutpb: float = 0.30,
    sbx_eta: float = 2.0,
    n_jobs: int = None,
):
    """
    Run the full island-model GA search (all n_runs, one seed per run,
    parallelised across seeds via joblib) against a single healthy atlas.

    Output CSVs are suffixed with the atlas label so that atlas="both" runs
    do not overwrite each other:
      two_gene_complete_<atlas_label>.csv
      two_gene_hof_<atlas_label>.csv

    Returns (df_hof, df_all) for this atlas.
    """
    global _gene_names, _n_genes, _safety_thresh
    global _tumor_cols, _healthy_cols

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

    adata_sub = adata_tumor[:, common_genes].copy()
    X_tumor   = adata_sub.X.toarray() if not isinstance(adata_sub.X, np.ndarray) else adata_sub.X
    tumor_mat = (X_tumor > 0).astype(np.int8)

    hg_idx      = {g: i for i, g in enumerate(healthy_genes)}
    col_idx     = np.array([hg_idx[g] for g in common_genes])
    healthy_mat = healthy_matrix_full[:, col_idx]

    print(f"Tumour matrix  ({atlas_label}): {tumor_mat.shape[0]} cells x {len(common_genes)} genes")
    print(f"Healthy matrix ({atlas_label}): {healthy_mat.shape[0]} samples x {len(common_genes)} genes")

    _gene_names    = common_genes
    _n_genes       = len(common_genes)
    _safety_thresh = safety_threshold

    _tumor_cols   = [np.ascontiguousarray(tumor_mat[:, i])   for i in range(_n_genes)]
    _healthy_cols = [np.ascontiguousarray(healthy_mat[:, i]) for i in range(_n_genes)]
    _init_deap(_n_genes)

    island_size = pop_size // n_islands
    and_quota   = int(pop_size * and_quota_frac)
    nand_quota  = int(pop_size * nand_quota_frac)
    open_quota  = pop_size - and_quota - nand_quota

    seed_list      = [42 + i for i in range(n_runs)]
    resolved_n_jobs = n_jobs or n_runs

    print(f"\n[{atlas_label}] Running island-model GA across {n_runs} seed(s) "
          f"in parallel (n_jobs={resolved_n_jobs}, n_islands={n_islands}, "
          f"island_size={island_size}, pop_size={pop_size})")

    parallel_results = Parallel(n_jobs=resolved_n_jobs, backend="multiprocessing")(
        delayed(_run_ga)(
            seed=seed, n_genes=_n_genes, island_size=island_size,
            n_islands=n_islands, and_quota=and_quota, nand_quota=nand_quota,
            open_quota=open_quota, Gmax=Gmax, Ggap=Ggap, Rrep=Rrep,
            patience=patience, gate_min_frac=gate_min_frac,
            mutpb=mutpb, sbx_eta=sbx_eta,
            migrate_interval=migrate_interval, migrate_k=migrate_k,
        )
        for seed in seed_list
    )

    all_hof     = []
    all_results = []
    for hof, logbook, results in parallel_results:
        df_run = pd.DataFrame(
            results,
            columns=["generation", "LogicGates", "Genes", "Efficacy", "Safety", "seed_value"]
        )
        df_run = df_run[["seed_value", "generation", "LogicGates", "Genes", "Efficacy", "Safety"]]
        all_results.append(df_run)
        all_hof.extend(hof)

    df_all       = pd.concat(all_results, ignore_index=True)
    df_all       = _postprocess_results(df_all)
    complete_csv = os.path.join(output_dir, f"two_gene_complete_{atlas_label}.csv")
    df_all.to_csv(complete_csv, index=False)
    print(f"\n[{atlas_label}] Complete results saved to: {complete_csv}")

    hof_data = []
    for ind in all_hof:
        try:
            gA, gB, gT = ind
            hof_data.append([
                getattr(ind, "seed_value", None),
                getattr(ind, "generation", None),
                LOGIC_GATES[gT],
                [common_genes[gA], common_genes[gB]],
                ind.fitness.values[0],
                getattr(ind, "safety", None),
            ])
        except Exception as e:
            logger.warning(f"[{atlas_label}] Skipping HOF individual: {e}")

    df_hof  = pd.DataFrame(
        hof_data,
        columns=["seed_value", "generation", "LogicGates", "Genes", "Efficacy", "Safety"]
    )
    df_hof  = _postprocess_results(df_hof)
    hof_csv = os.path.join(output_dir, f"two_gene_hof_{atlas_label}.csv")
    df_hof.to_csv(hof_csv, index=False)
    print(f"[{atlas_label}] Hall of Fame saved to: {hof_csv}")

    print(f"\n[{atlas_label}] Top 10 from Hall of Fame:")
    print(df_hof.head(10).to_string(index=False))

    return df_hof, df_all


# ─────────────────────────────────────────────────────────────────────────────
# Robust Rank Aggregation across the two atlases
#
# Ported from the user's standalone RRA R script (Robust_Rank_Aggregation_
# Three_Atlas_FIXED.R), reduced from 3 atlases (HPA/HCA/Tabula) to 2
# (HPA/Tabula) since only two healthy atlases are used in this module.
# Same steps, same fixes:
#   - candidates must pass efficacy > threshold & safety > threshold in
#     BOTH atlases (strict "all atlases" filter)
#   - each atlas ranks candidates by a COMBINED score (efficacy * safety)
#     before aggregation, so both efficacy and safety inform the final rank
#   - "___" ID separator (avoids collision with the literal "|" inside the
#     "A | B" gate string)
#   - the actual RRA rho-scoring is delegated to R's RobustRankAggreg
#     package (aggregateRanks(method="RRA")), now launched as an external
#     Rscript subprocess rather than embedded in-process via rpy2 — see
#     module docstring "Fix applied (rpy2 / R environment) [SUPERSEDED]" —
#     so the statistics are still identical to the reference implementation.
# ─────────────────────────────────────────────────────────────────────────────

_SYMMETRIC_GATES = {"A | B", "A & B"}


def _prepare_rra_input(df_all: pd.DataFrame, atlas_key: str) -> pd.DataFrame:
    """
    Convert a two-gene GA df_all table (seed_value, generation, LogicGates,
    Genes, Efficacy, Safety) into the geneA / geneB / gate /
    <atlas>_efficacy / <atlas>_safety layout used by the RRA logic.
    """
    df = df_all.copy()
    if isinstance(df.iloc[0]["Genes"], str):
        df["Genes"] = df["Genes"].apply(eval)

    df["geneA"] = df["Genes"].apply(lambda g: g[0])
    df["geneB"] = df["Genes"].apply(lambda g: g[1])
    df = df.rename(columns={
        "LogicGates": "gate",
        "Efficacy":   f"{atlas_key}_efficacy",
        "Safety":     f"{atlas_key}_safety",
    })
    df = df[["geneA", "geneB", "gate", f"{atlas_key}_efficacy", f"{atlas_key}_safety"]]
    df = df.sort_values(by=f"{atlas_key}_efficacy", ascending=False)
    df = df.drop_duplicates(subset=["geneA", "geneB", "gate"], keep="first")
    return df.reset_index(drop=True)


def _normalize_gene_order_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sort geneA/geneB alphabetically for symmetric gates only (A|B, A&B),
    so the same unordered pair merges to one row regardless of GA draw order.
    Directional gates (A & !B) are left untouched."""
    df = df.copy()

    def _swap(row):
        if row["gate"] in _SYMMETRIC_GATES:
            a, b = sorted([row["geneA"], row["geneB"]])
            return pd.Series([a, b])
        return pd.Series([row["geneA"], row["geneB"]])

    df[["geneA", "geneB"]] = df.apply(_swap, axis=1)
    return df


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

    Identical to Module 4a (one_gene_combination.py) — see that module's
    docstring "Fix applied (rpy2 / R environment)" for the full rationale.
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
    (SCART.preprocessing "Rscript found via sys.executable dir" logic) and
    Module 4a's identical helper: prefer the Rscript living inside the same
    conda environment SCART is installed in, falling back to whatever
    Rscript is on PATH.
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
    running in this Python session (including the joblib multiprocessing
    workers used by the GA step above).
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
    R for scMalignantFinder and SCEVAN, and is identical to the fix applied
    in Module 4a (one_gene_combination.py): write inputs to disk, write a
    small R driver script, launch `Rscript <driver>.R` as a subprocess,
    then read the R-written output CSV back into Python.

    Requires the R package 'RobustRankAggreg' to be installed in whichever
    R installation Rscript resolves to (installed automatically by
    `python -m SCART.install`, see install.py).

    Parameters
    ----------
    *rank_lists : list[str]
        One ranked candidate-ID list (best -> worst) per atlas.
    output_dir : str
        Directory to write the RRA driver script / input & output CSVs
        into (a "rra_rscript" subfolder is created here).

    Returns
    -------
    dict[str, float]  — candidate ID -> RRA score (lower = better, matching
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
    # candidate ID per row, best -> worst), same idea as SCEVAN's
    # counts/barcodes CSVs handed off to its driver R script.
    input_paths = []
    for i, rl in enumerate(rank_lists):
        path = os.path.join(rra_dir, f"rank_list_{i + 1}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for candidate_id in rl:
                writer.writerow([candidate_id])
        input_paths.append(path)
        print(f"  RRA input list {i + 1} ({len(rl)} candidates) written to: {path}")

    output_path   = os.path.join(rra_dir, "rra_result.csv")
    r_script_path = os.path.join(rra_dir, "run_rra.R")

    r_input_vector = ", ".join(f'"{p}"' for p in input_paths)
    r_code = f'''# Auto-generated driver script — Robust Rank Aggregation (SCART Module 4b)
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
# RRA result plots
#
# Replaces the earlier single grouped-bar "Top 20 RRA candidates" plot with
# two dual-panel scatter plots, adapted from the user's reference
# R/ggplot2+ggrepel script (avg. safety vs HPA efficacy; every candidate
# shown as background; selected candidates highlighted as labelled diamonds
# with a dashed rectangle on the left panel marking the zoomed-in region
# shown on the right panel). Same core idea as the reference script, just
# ported to matplotlib and restyled:
#
#   _plot_top_gate_rra_candidates() — top 5 candidates by RRA_Rank within
#       EACH gate type (OR, AND, NAND — A & !B), shown on a three-panel
#       figure (all candidates + zoom rectangle, zoomed-in labelled panel,
#       and a "Gene-Ranks" legend panel), ported from a newer reference
#       R/ggplot2+ggrepel+cowplot dual-gene script. Colour is a
#       light-to-dark gradient within each gate (rank 1 = darkest); shape
#       is fixed per gate (diamond = OR, square = AND, triangle = NAND).
#   _plot_top10_rra_candidates()    — top 10 candidates overall by
#       RRA_Rank, regardless of gate type.
#
# Both save PDF + PNG (with a `_claude` suffix) and, when run inside a
# Jupyter kernel, also display inline via plt.show() — see
# _configure_matplotlib_backend().
# ─────────────────────────────────────────────────────────────────────────────

_GATE_TYPE_MAP = {"A & B": "AND", "A | B": "OR", "A & !B": "NAND"}

# Distinct rank-ordered palette for the top-10-overall plot (rank 1 first).
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


def _prep_rra_plot_df(df_ranked: pd.DataFrame) -> pd.DataFrame:
    """Shared prep for both RRA plots: average safety across atlases, a
    display label per candidate, and a coarse gate-type category."""
    df = df_ranked.dropna(subset=["hpa_safety", "tabula_safety", "hpa_efficacy"]).copy()
    df["avg_safety"] = (df["hpa_safety"] + df["tabula_safety"]) / 2.0
    df["efficacy"]   = df["hpa_efficacy"]  # atlas-invariant by construction
    df["gate_type"]  = df["gate"].map(_GATE_TYPE_MAP).fillna("OTHER")

    def _label(row):
        if row["gate"] == "A & !B":
            return f"{row['geneA']} & !{row['geneB']}"
        symbol = row["gate"].replace("A", "").replace("B", "").strip()
        return f"{row['geneA']} {symbol} {row['geneB']}"

    df["candidate"] = df.apply(_label, axis=1)
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
    separates any colliding label boxes using their real rendered extents.
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
    # highlighted candidates, exactly as in the reference script (its
    # `rank6_safety` / `min_highlight_efficacy`), drawn as full dashed
    # cross-lines on both panels. The shaded zoom rectangle uses a
    # 1-point floor/ceil margin around the highlighted candidates, also
    # matching the reference script's `zoom_x_min/max`, `zoom_y_min/max`.
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

    ax_l.set_xlim(left=90)
    ax_l.set_ylim(bottom=70)
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


# ─────────────────────────────────────────────────────────────────────────────
# Top-gate RRA plot — rendered natively in R (self-contained in this module)
#
# Adapted from a reference R/ggplot2 + ggrepel + cowplot dual-gene script,
# embedded directly here and launched as an Rscript subprocess — mirrors
# exactly the pattern _run_rra_via_r() above already uses for
# RobustRankAggreg, and the equivalent fix already applied to Module 4a's
# single-gene plot (one_gene_combination.py's _plot_single_gene_rra_r()).
# The other plot below (_plot_top10_rra_candidates, "top 10 overall
# regardless of gate") is unchanged and still rendered in matplotlib via
# _dual_panel_rra_figure() above.
#
# Requires the R packages ggplot2, dplyr, ggrepel and cowplot — already
# required/installed for Module 4a's single-gene plot (see install.py's
# CONDA_R_BASE); no further R dependency changes needed here.
# ─────────────────────────────────────────────────────────────────────────────

_TOP_GATE_RRA_PLOT_R_TEMPLATE = r"""
#!/usr/bin/env Rscript

library(ggplot2)
library(dplyr)
library(cowplot)
library(ggrepel)

# ============================================================
# DUAL-GENE:
# EFFICACY vs SAFETY
#
# Auto-generated by two_gene_combination.py's
# _plot_top_gate_rra_candidates_r() every time it runs (mirrors
# exactly how _run_rra_via_r() already generates its own R
# driver script for RobustRankAggreg, and how Module 4a's
# one_gene_combination.py embeds its own single-gene plot R
# source) — NOT meant to be hand-edited or launched with
# different arguments. The plotting logic below (gate-label
# mapping, pair-label building, top-N-per-gate highlighting,
# per-gate colour gradients + shapes, thresholds, rectangle,
# labelled zoom panel, "Gene-Ranks" legend panel grouped by
# gate, cowplot::plot_grid combine) matches a reference
# R/ggplot2/ggrepel/cowplot dual-gene script exactly. Only two
# things differ from that reference:
#   1. input_file / output_dir_arg / top_n_per_gate below are
#      literal values baked in by _plot_top_gate_rra_candidates_r()
#      at write time, instead of a hard-coded path and a
#      hard-coded "top 5".
#   2. The rank column is "RRA_Rank" (what
#      _robust_rank_aggregation() actually writes), not
#      "RRA_rank"; and the output file gets a "_claude" suffix.
#
# Highlighting logic (DIFFERENT from the single-gene script):
# Instead of "top 20 overall -> rectangle -> everyone inside
# the rectangle", this version highlights EXACTLY the top
# `top_n_per_gate` pairs (by RRA_Rank, 1 = best) WITHIN EACH
# GATE TYPE (OR / AND / NAND, taken from the `gate` column).
#
# Output:
# <output_dir>/SCATTER_PLOTS_FIGURE3/Dual_gene_efficacy_vs_safety_COMBINED_claude.png
#
#   Single combined figure:
#     LEFT   = all pairs, colored by gate for the highlighted
#              top-N-per-gate pairs, with highlighted rectangle
#     RIGHT  = zoom into rectangle, every highlighted pair is
#              labeled with the ORIGINAL gate symbol between
#              the two gene names (e.g. "FOLR1 | MSLN")
#     FAR    = rank / gate legend panel
#     RIGHT
#
# Resolution:
# 800 DPI
# Format:
# PNG
# ============================================================


# ============================================================
# 1. INPUT FILE / OUTPUT DIRECTORY / TOP N PER GATE
#
# Written in directly by _plot_top_gate_rra_candidates_r() when
# it generates this file.
# ============================================================

input_file <- "@@INPUT_CSV@@"
output_dir_arg <- "@@OUTPUT_DIR@@"
top_n_per_gate <- @@TOP_N_PER_GATE@@


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
cat("DUAL-GENE EFFICACY vs SAFETY ANALYSIS\n")
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
  "Dual_gene_efficacy_vs_safety_COMBINED_claude.png"
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

dual_gene <- read.csv(
  input_file,
  stringsAsFactors = FALSE,
  check.names = FALSE
)


# ============================================================
# 6. CHECK REQUIRED COLUMNS
# ============================================================

required_columns <- c(
  "geneA",
  "geneB",
  "gate",
  "hpa_efficacy",
  "hpa_safety",
  "tabula_safety",
  "RRA_Rank"
)

missing_columns <- setdiff(
  required_columns,
  colnames(dual_gene)
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
        colnames(dual_gene),
        collapse = ", "
      ),
      "\n"
    )
  )
}


# ============================================================
# 7. GATE LABEL MAPPING
#
# The `gate` column stores the raw logic-gate string (e.g.
# "A | B"). This maps it to a human-readable category:
# OR / AND / NAND. Falls back to the raw string if it does
# not match any known pattern, so nothing is silently dropped.
# ============================================================

map_gate_label <- function(g) {

  if (grepl("NAND", g, ignore.case = TRUE) || grepl("!", g, fixed = TRUE)) {
    return("NAND")
  }

  if (grepl("|", g, fixed = TRUE)) {
    return("OR")
  }

  if (grepl("&", g, fixed = TRUE)) {
    return("AND")
  }

  return(g)
}


# ============================================================
# 7b. PAIR LABEL BUILDER
#
# Builds the on-plot label using the ORIGINAL gate string
# (e.g. "A | B") with the placeholder letters "A" and "B"
# swapped for the real gene names, so the plot shows e.g.
# "FOLR1 | MSLN" instead of "FOLR1 + MSLN". Word-boundary
# matching means this is safe even for gate strings like
# "A NAND B", since the standalone "A"/"B" tokens are the
# only thing replaced (the "A" inside "NAND" is not a
# standalone word and is left untouched).
# ============================================================

build_candidate_label <- function(gA, gB, g) {

  lbl <- g

  lbl <- gsub("\\bA\\b", gA, lbl)
  lbl <- gsub("\\bB\\b", gB, lbl)

  return(lbl)
}


# ============================================================
# 8. PREPARE DATA
# ============================================================

plot_df <- dual_gene %>%

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
    # NOTE: this is a RANK, not a score — 1 = best pair.
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
    # Human-readable gate category
    # --------------------------------------------------------

    gate_group = vapply(
      gate,
      map_gate_label,
      character(1)
    ),

    # --------------------------------------------------------
    # Pair label — original gate symbol between the two genes
    # (e.g. "FOLR1 | MSLN"), not a generic "+"
    # --------------------------------------------------------

    candidate = mapply(
      build_candidate_label,
      as.character(geneA),
      as.character(geneB),
      as.character(gate)
    )

  ) %>%

  filter(
    !is.na(safety_pct),
    !is.na(efficacy_pct),
    !is.na(ObjectiveScore),
    !is.na(candidate),
    candidate != ""
  )


# ============================================================
# 8b. ORDER GATE CATEGORIES: OR, then AND, then NAND
#
# gate_group is converted to a factor with this fixed level
# order, so every downstream group_by()/arrange() (highlighted
# selection, plotting legends, the rank/gate legend panel)
# naturally follows OR -> AND -> NAND. Any gate category not
# in this preferred list (e.g. an unrecognized gate string)
# is appended afterward, alphabetically.
# ============================================================

preferred_gate_order <- c("OR", "AND", "NAND")

gates_present <- unique(plot_df$gate_group)

gate_level_order <- c(
  intersect(preferred_gate_order, gates_present),
  sort(setdiff(gates_present, preferred_gate_order))
)

plot_df <- plot_df %>%

  mutate(
    gate_group = factor(
      gate_group,
      levels = gate_level_order
    )
  )


# ============================================================
# 9. PRINT DATA CHECK
# ============================================================

cat("\n")
cat("Dual-gene data prepared:\n")
cat("--------------------------------------------\n")

print(
  plot_df %>%
    select(
      candidate,
      gate_group,
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

cat(
  "\nGate categories found:",
  paste(
    sort(unique(plot_df$gate_group)),
    collapse = ", "
  ),
  "\n"
)


# ============================================================
# 10. TOP 5 PER GATE TYPE
#
# For each gate category (OR / AND / NAND, or whatever is
# present in the data), take the top 5 pairs by ObjectiveScore
# (RRA_Rank, 1 = best). Sorted ASCENDING since lower RRA_Rank
# = better pair.
# ============================================================

highlighted <- plot_df %>%

  group_by(
    gate_group
  ) %>%

  arrange(
    ObjectiveScore,
    .by_group = TRUE
  ) %>%

  slice_head(
    n = top_n_per_gate
  ) %>%

  ungroup() %>%

  arrange(
    gate_group,
    ObjectiveScore
  ) %>%

  group_by(
    gate_group
  ) %>%

  mutate(
    rank_in_gate = row_number()
  ) %>%

  ungroup() %>%

  mutate(
    label = candidate
  )

n_highlighted <- nrow(highlighted)

cat("\n")
cat(paste0("TOP ", top_n_per_gate, " PER GATE (highlighted candidates): "), n_highlighted, "\n")
cat("--------------------------------------------\n")

print(
  highlighted %>%
    select(
      gate_group,
      rank_in_gate,
      candidate,
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
# 11. GRADIENT COLORS + SHAPES PER GATE TYPE
#
# Each gate gets its OWN hue family (blues for OR, oranges for
# AND, greens for NAND) so the three gate types stay clearly
# distinguishable at a glance. WITHIN a gate's 5 candidates, a
# light-to-dark gradient is used: rank 1 (best RRA_Rank) gets
# the darkest/most saturated shade, rank 5 gets the lightest —
# so within-gate ranking is visible from shade alone. Point
# SHAPE is still fixed per gate (diamond/square/triangle) as
# an additional, color-independent way to tell gates apart.
# ============================================================

gate_hue_ends <- list(
  OR   = c("#1E3A8A", "#60A5FA"),  # deep vivid blue    -> bright sky blue
  AND  = c("#9A3412", "#FB923C"),  # deep vivid orange  -> bright orange
  NAND = c("#065F46", "#34D399")   # deep vivid emerald -> bright emerald
)

get_gate_hues <- function(g) {

  if (g %in% names(gate_hue_ends)) {
    return(gate_hue_ends[[g]])
  }

  # Fallback ramp for any gate category outside OR/AND/NAND
  return(c("#581C87", "#C084FC"))  # deep vivid purple -> bright purple
}

highlighted <- highlighted %>%

  group_by(
    gate_group
  ) %>%

  mutate(
    color = colorRampPalette(
      get_gate_hues(as.character(gate_group)[1])
    )(n())[rank_in_gate]
  ) %>%

  ungroup()

gate_levels <- levels(droplevels(highlighted$gate_group))

# --------------------------------------------------------
# Shapes per gate type, for extra distinction beyond color
# --------------------------------------------------------

base_gate_shapes <- c(
  "OR"   = 23,
  "AND"  = 22,
  "NAND" = 24
)

missing_shapes <- setdiff(
  gate_levels,
  names(base_gate_shapes)
)

if (length(missing_shapes) > 0) {

  extra_shapes <- rep_len(
    c(21, 23, 22, 24, 25),
    length(missing_shapes)
  )

  names(extra_shapes) <- missing_shapes

  base_gate_shapes <- c(
    base_gate_shapes,
    extra_shapes
  )
}

gate_shapes <- base_gate_shapes[gate_levels]


# ============================================================
# 12. THRESHOLDS (based on the combined highlighted set)
# ============================================================

min_highlighted_safety <- min(
  highlighted$safety_pct,
  na.rm = TRUE
)

min_highlighted_efficacy <- min(
  highlighted$efficacy_pct,
  na.rm = TRUE
)

cat(
  "\nMinimum highlighted safety:",
  round(min_highlighted_safety, 2),
  "%\n"
)

cat(
  "Minimum highlighted efficacy:",
  round(min_highlighted_efficacy, 2),
  "%\n"
)


# ============================================================
# 13. ZOOM / HIGHLIGHT RECTANGLE
# ============================================================

zoom_x_min <- max(
  0,
  floor(
    min(
      highlighted$safety_pct,
      na.rm = TRUE
    )
  ) - 1
)

zoom_x_max <- min(
  100,
  ceiling(
    max(
      highlighted$safety_pct,
      na.rm = TRUE
    )
  ) + 1
)

zoom_y_min <- max(
  0,
  floor(
    min(
      highlighted$efficacy_pct,
      na.rm = TRUE
    )
  ) - 1
)

zoom_y_max <- min(
  100,
  ceiling(
    max(
      highlighted$efficacy_pct,
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
# 14. LEFT PLOT — ALL DUAL-GENE PAIRS
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
  # Threshold lines
  # ----------------------------------------------------------

  geom_vline(

    xintercept = min_highlighted_safety,

    linetype = "dashed",

    linewidth = 0.8,

    color = "grey30"
  ) +

  geom_hline(

    yintercept = min_highlighted_efficacy,

    linetype = "dashed",

    linewidth = 0.8,

    color = "grey30"
  ) +

  # ----------------------------------------------------------
  # Background pairs (not in the top 5 of any gate)
  # ----------------------------------------------------------

  geom_point(

    data = plot_df %>%
      filter(
        !candidate %in% highlighted$candidate
      ),

    aes(
      x = safety_pct,
      y = efficacy_pct
    ),

    color = "grey65",

    alpha = 0.6,

    size = 2.4
  ) +

  # ----------------------------------------------------------
  # Highlighted top-5-per-gate pairs
  # ----------------------------------------------------------

  geom_point(

    data = highlighted,

    aes(
      x = safety_pct,
      y = efficacy_pct,
      fill = color,
      shape = gate_group
    ),

    color = "black",

    size = 4,

    stroke = 0.5
  ) +

  # ----------------------------------------------------------
  # Fill uses the exact gradient hex color computed per pair
  # (identity scale, no separate legend needed for it — the
  # rank/gate legend panel on the right spells it out). Shape
  # still gets a real legend so the gate categories are
  # readable directly off this panel.
  # ----------------------------------------------------------

  scale_fill_identity() +

  scale_shape_manual(
    name = "Gate",
    values = gate_shapes
  ) +

  scale_x_continuous(

    labels = function(x) paste0(x, "%"),

    expand = expansion(mult = 0.03)
  ) +

  scale_y_continuous(

    labels = function(y) paste0(y, "%"),

    expand = expansion(mult = 0.03)
  ) +

  labs(

    title = "All dual-gene candidates",

    x = "Safety",

    y = "Efficacy"
  ) +

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
      color = "black"
    ),

    axis.title.y = element_text(
      family = "Times New Roman",
      size = 23,
      face = "bold",
      color = "black"
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

    legend.position = "bottom",

    legend.title = element_text(
      family = "Times New Roman",
      size = 16,
      face = "bold"
    ),

    legend.text = element_text(
      family = "Times New Roman",
      size = 14
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
# 15. RIGHT PLOT — ZOOM INTO RECTANGLE
#
# Every top-5-per-gate pair is plotted AND labeled here.
# ============================================================

p_right <- ggplot(

  highlighted,

  aes(
    x = safety_pct,
    y = efficacy_pct
  )
) +

  geom_vline(

    xintercept = min_highlighted_safety,

    linetype = "dashed",

    linewidth = 0.8,

    color = "grey30"
  ) +

  geom_hline(

    yintercept = min_highlighted_efficacy,

    linetype = "dashed",

    linewidth = 0.8,

    color = "grey30"
  ) +

  geom_point(

    aes(
      fill = color,
      shape = gate_group
    ),

    color = "black",

    size = 4.5,

    stroke = 0.6
  ) +

  # ----------------------------------------------------------
  # GENE-PAIR LABELS — ALL highlighted pairs
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

  scale_fill_identity() +

  scale_shape_manual(
    name = "Gate",
    values = gate_shapes,
    guide = "none"
  ) +

  scale_x_continuous(

    limits = c(zoom_x_min, zoom_x_max),

    labels = function(x) paste0(x, "%"),

    expand = expansion(mult = 0.08)
  ) +

  scale_y_continuous(

    limits = c(zoom_y_min, zoom_y_max),

    breaks = seq(
      ceiling(zoom_y_min / 2) * 2,
      floor(zoom_y_max / 2) * 2,
      by = 2
    ),

    labels = function(y) paste0(y, "%"),

    expand = expansion(mult = 0.08)
  ) +

  labs(

    title = paste0("Top ", top_n_per_gate, " per gate (OR / AND / NAND)"),

    x = "Safety",

    y = "Efficacy"
  ) +

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
      size = 21,
      face = "bold",
      color = "black",
      hjust = 0.5
    ),

    axis.title.x = element_text(
      family = "Times New Roman",
      size = 23,
      face = "bold",
      color = "black"
    ),

    axis.title.y = element_text(
      family = "Times New Roman",
      size = 23,
      face = "bold",
      color = "black"
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
# 16. RANK / GATE LEGEND PANEL
#
# A standalone panel listing every highlighted pair, grouped
# OR first, then AND, then NAND (via the gate_group factor
# order set in step 8b), and within each gate ordered best to
# worst. The number shown next to each pair is its ORIGINAL
# RRA_Rank value (not its 1-5 position within the gate), so
# it can be cross-referenced directly against the source CSV.
# ============================================================

# --------------------------------------------------------
# Vertical layout is built with an explicit cursor rather
# than a single continuous row_number(), so we can use a
# LARGER gap when moving to a new gate's header (visually
# separating it from the previous group's last row) and a
# SMALLER, fixed gap between a header and its own group's
# first row (visually attaching the header to its block).
# --------------------------------------------------------

row_spacing          <- 1.0   # spacing between rows within a group
header_to_row_gap    <- 0.55  # spacing between a header and its first row
inter_group_gap      <- 1.1   # extra spacing before a new group's header

groups_ordered <- levels(
  droplevels(
    highlighted$gate_group
  )
)

y_cursor <- 0

legend_rows_list  <- vector("list", length(groups_ordered))
gate_headers_list <- vector("list", length(groups_ordered))

for (i in seq_along(groups_ordered)) {

  g <- groups_ordered[i]

  grp_rows <- highlighted %>%
    filter(gate_group == g) %>%
    arrange(rank_in_gate)

  n_rows <- nrow(grp_rows)

  if (i > 1) {
    y_cursor <- y_cursor - inter_group_gap
  }

  gate_headers_list[[i]] <- data.frame(
    gate_group = g,
    y_pos = y_cursor
  )

  y_cursor <- y_cursor - header_to_row_gap

  grp_rows$y_pos <- y_cursor - (seq_len(n_rows) - 1) * row_spacing

  legend_rows_list[[i]] <- grp_rows

  y_cursor <- min(grp_rows$y_pos)
}

legend_df <- bind_rows(legend_rows_list)

gate_headers <- bind_rows(gate_headers_list)

p_legend <- ggplot() +

  geom_text(

    data = gate_headers,

    aes(
      x = 0,
      y = y_pos,
      label = gate_group
    ),

    hjust = 0,

    family = "Times New Roman",

    fontface = "bold",

    size = 5,

    color = "black"
  ) +

  geom_point(

    data = legend_df,

    aes(
      x = 0,
      y = y_pos,
      fill = color,
      shape = gate_group
    ),

    color = "black",

    size = 5,

    stroke = 0.5
  ) +

  geom_text(

    data = legend_df,

    aes(
      x = 0,
      y = y_pos,
      label = paste0(
        "Rank ",
        as.integer(round(ObjectiveScore)),
        ":  ",
        label
      )
    ),

    hjust = 0,

    nudge_x = 0.18,

    family = "Times New Roman",

    fontface = "bold",

    size = 4,

    color = "black"
  ) +

  scale_fill_identity() +

  scale_shape_manual(
    values = gate_shapes,
    guide = "none"
  ) +

  xlim(
    -0.1,
    3.4
  ) +

  labs(
    title = "Gene-Ranks"
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
# 17. COMBINE INTO A SINGLE FIGURE
# ============================================================

p_combined <- cowplot::plot_grid(

  p_left,

  p_right,

  p_legend,

  ncol = 3,

  align = "h",

  axis = "tb",

  rel_widths = c(1, 1, 0.38)
)


# ============================================================
# 18. SAVE COMBINED PLOT
# ============================================================

cat("\n")
cat("Saving COMBINED plot...\n")

ggsave(

  filename = combined_file,

  plot = p_combined,

  width = 18.5,

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
# 19. VERIFY OUTPUT FILE
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
# 20. FINAL REPORT
# ============================================================

cat("============================================================\n")
cat("DUAL-GENE ANALYSIS COMPLETED\n")
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
  paste0("Highlighted pairs (top ", top_n_per_gate, " per gate):"),
  n_highlighted,
  "\n"
)

print(
  highlighted %>%
    select(
      gate_group,
      rank_in_gate,
      candidate,
      Efficacy,
      Safety,
      ObjectiveScore
    )
)

cat("\n")

cat(
  "Minimum highlighted safety (threshold):",
  round(min_highlighted_safety, 2),
  "%\n"
)

cat(
  "Minimum highlighted efficacy (threshold):",
  round(min_highlighted_efficacy, 2),
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
  "Dimensions: 18.5 x 8 inches (all pairs | zoom | gate/rank)\n"
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
    top-gate plot had (via _configure_matplotlib_backend() + plt.show()).
    Since that plot is now rendered by an Rscript subprocess rather than
    in-process, there is no matplotlib figure object to show — this
    displays the saved PNG file directly via IPython's rich display
    instead, which gives the same "just ran a cell and saw the plot"
    experience in a notebook. No-op outside a notebook (headless script
    runs), and if the file doesn't exist for some reason. Identical to
    Module 4a's (one_gene_combination.py) helper of the same name.
    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
        in_notebook = ip is not None and "IPKernelApp" in ip.config
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


def _plot_top_gate_rra_candidates_r(rra_csv_path: str, output_dir: str, top_n_per_gate: int = 5) -> None:
    """
    Render the top-gate RRA "top N per gate" figure natively in R.

    Fills _TOP_GATE_RRA_PLOT_R_TEMPLATE's three placeholders
    (@@INPUT_CSV@@, @@OUTPUT_DIR@@, @@TOP_N_PER_GATE@@) with the actual
    values for this run, writes the result to a driver script under
    output_dir, and launches it as an Rscript subprocess — the same
    subprocess pattern _run_rra_via_r() already uses for RobustRankAggreg,
    reused here via _find_rscript()/_find_r_home()/_build_r_subprocess_env().

    The R script itself defines the highlighting logic: EXACTLY the top
    `top_n_per_gate` candidates by RRA_Rank (default 5) WITHIN EACH gate
    type (OR / AND / NAND) are highlighted — colour is a light-to-dark
    gradient within each gate (rank 1 = darkest), shape is fixed per gate
    (diamond = OR, square = AND, triangle = NAND) — and every highlighted
    pair is labelled (via ggrepel, using the original gate symbol between
    the two gene names, e.g. "FOLR1 | MSLN") and listed in a "Gene-Ranks"
    legend panel grouped by gate, combined with cowplot.

    Parameters
    ----------
    rra_csv_path : str
        Path to the RRA-ranked CSV — exactly what
        _robust_rank_aggregation() already writes to
        final_candidates_RRA_HPA_Tabula.csv. Must contain the columns
        geneA, geneB, gate, hpa_efficacy, hpa_safety, tabula_safety,
        RRA_Rank.
    output_dir : str
        Directory the figure is written into. The R script creates a
        SCATTER_PLOTS_FIGURE3 subfolder here.
    top_n_per_gate : int
        Number of top RRA_Rank candidates highlighted within each gate
        type (default 5).
    """
    import subprocess

    rscript_path = _find_rscript()
    r_home       = _find_r_home()
    env          = _build_r_subprocess_env(r_home)

    plot_dir = os.path.join(output_dir, "rra_plot_rscript")
    os.makedirs(plot_dir, exist_ok=True)
    r_script_path = os.path.join(plot_dir, "plot_top_gate_rra.R")

    r_code = (
        _TOP_GATE_RRA_PLOT_R_TEMPLATE
        .replace("@@INPUT_CSV@@", os.path.abspath(rra_csv_path).replace("\\", "/"))
        .replace("@@OUTPUT_DIR@@", os.path.abspath(output_dir).replace("\\", "/"))
        .replace("@@TOP_N_PER_GATE@@", str(int(top_n_per_gate)))
    )
    with open(r_script_path, "w") as f:
        f.write(r_code)

    print(f"  Rscript:                  {rscript_path}")
    print(f"  R home:                   {r_home}")
    print(f"  Plot driver script:       {r_script_path}")
    print("  Launching Rscript subprocess for the top-gate RRA plot ...")

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
            "Rscript subprocess for the top-gate RRA plot failed "
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

    print("  Top-gate RRA plot rendered via R.")

    png_path = os.path.join(
        output_dir, "SCATTER_PLOTS_FIGURE3",
        "Dual_gene_efficacy_vs_safety_COMBINED_claude.png",
    )
    _display_png_if_notebook(png_path)


def _plot_top10_rra_candidates(df_ranked: pd.DataFrame, output_dir: str, top_n: int = 10):
    """
    Top `top_n` (default 10) candidates overall by RRA_Rank, regardless of
    gate type. Same dual-panel look as _plot_top_gate_rra_candidates(),
    with points coloured by rank (best = darkest) and rank numbers baked
    into each label.
    """
    plot_df     = _prep_rra_plot_df(df_ranked)
    highlighted = plot_df.sort_values("RRA_Rank").head(top_n).reset_index(drop=True)

    if highlighted.empty:
        print("No RRA-ranked candidates available — skipping top-10 RRA plot.")
        return

    highlighted = highlighted.copy()
    highlighted["candidate"] = [
        f"#{i + 1}  {c}" for i, c in enumerate(highlighted["candidate"])
    ]
    colors = _RANK_PALETTE[:len(highlighted)]

    _dual_panel_rra_figure(
        plot_df, highlighted, colors,
        out_stub="Top10_RRA_Candidates",
        output_dir=output_dir,
        suptitle="Top 10 Candidates Overall — Robust Rank Aggregation",
        left_title="All candidate combinations",
        right_title="Top 10 by RRA rank",
    )


def _robust_rank_aggregation(
    df_all_hpa: pd.DataFrame,
    df_all_tabula: pd.DataFrame,
    efficacy_threshold: float,
    safety_threshold: float,
    output_dir: str,
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  Robust Rank Aggregation — combining HPA + Tabula Sapiens results")
    print("=" * 70)

    hpa_df    = _normalize_gene_order_df(_prepare_rra_input(df_all_hpa,    "hpa"))
    tabula_df = _normalize_gene_order_df(_prepare_rra_input(df_all_tabula, "tabula"))

    hpa_candidates = hpa_df[
        (hpa_df["hpa_efficacy"] > efficacy_threshold) &
        (hpa_df["hpa_safety"]   > safety_threshold)
    ]
    tabula_candidates = tabula_df[
        (tabula_df["tabula_efficacy"] > efficacy_threshold) &
        (tabula_df["tabula_safety"]   > safety_threshold)
    ]

    candidate_combos = pd.concat([
        hpa_candidates[["geneA", "geneB", "gate"]],
        tabula_candidates[["geneA", "geneB", "gate"]],
    ]).drop_duplicates()

    combined = candidate_combos.merge(hpa_df,    on=["geneA", "geneB", "gate"], how="left")
    combined = combined.merge(tabula_df, on=["geneA", "geneB", "gate"], how="left")
    combined = combined.drop_duplicates().reset_index(drop=True)
    print(f"unique_candidates dim: {combined.shape}")

    # Sanity check — efficacy should be identical across atlases for the
    # same combo (same GA fitness definition, different safety only).
    both_present = combined["hpa_efficacy"].notna() & combined["tabula_efficacy"].notna()
    equal_mask   = combined.loc[both_present, "hpa_efficacy"] == combined.loc[both_present, "tabula_efficacy"]
    print(f"Matching rows: {int(equal_mask.sum())} / {int(both_present.sum())}")
    print(f"Mismatching rows: {int((~equal_mask).sum())}")
    print(f"Rows present in only one atlas: {int((~both_present).sum())}")

    # STRICT FILTER: keep only candidates passing efficacy > threshold AND
    # safety > threshold in BOTH atlases (mirrors the reference script's
    # "ALL atlases" strict filter, reduced from 3 atlases to 2).
    strict = combined[
        (combined["hpa_efficacy"]    > efficacy_threshold) & (combined["hpa_safety"]    > safety_threshold) &
        (combined["tabula_efficacy"] > efficacy_threshold) & (combined["tabula_safety"] > safety_threshold)
    ].copy()
    print(f"Candidates passing efficacy>{efficacy_threshold} & safety>{safety_threshold} "
          f"in BOTH atlases: {len(strict)}")

    out_csv = os.path.join(output_dir, "final_candidates_RRA_HPA_Tabula.csv")

    if strict.empty:
        print("No candidates passed the strict dual-atlas filter — "
              "skipping RRA aggregation and plot.")
        strict.to_csv(out_csv, index=False)
        return strict

    ID_SEP = "___"
    strict["hpa_combined"]    = strict["hpa_efficacy"]    * strict["hpa_safety"]
    strict["tabula_combined"] = strict["tabula_efficacy"] * strict["tabula_safety"]
    strict["ID"] = strict["geneA"] + ID_SEP + strict["geneB"] + ID_SEP + strict["gate"]

    hpa_rank    = strict.sort_values(by="hpa_combined",    ascending=False)["ID"].tolist()
    tabula_rank = strict.sort_values(by="tabula_combined", ascending=False)["ID"].tolist()

    rra_scores = _run_rra_via_r(hpa_rank, tabula_rank, output_dir=output_dir)

    strict["RRA_Score"] = strict["ID"].map(rra_scores)
    n_missing = strict["RRA_Score"].isna().sum()
    if n_missing > 0:
        logger.warning(f"{n_missing} candidate(s) missing an RRA score after aggregation.")

    strict = strict.sort_values(by="RRA_Score", ascending=True).reset_index(drop=True)
    strict["RRA_Rank"] = np.arange(1, len(strict) + 1)
    strict = strict.drop(columns=["ID"])

    strict.to_csv(out_csv, index=False)
    print(f"\nRRA-ranked dual-atlas candidates saved to: {out_csv}")

    print("\nTop 10 RRA-ranked candidates:")
    print(strict.head(10).to_string(index=False))

    _plot_top_gate_rra_candidates_r(out_csv, output_dir)
    _plot_top10_rra_candidates(strict, output_dir)

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
    pop_size: int  = 1000,
    Gmax: int      = 100,
    Ggap: int      = 10,
    Rrep: float    = 0.1,
    patience: int  = 50,
    n_runs: int    = 10,
    n_islands: int = 4,
    migrate_interval: int = 10,
    migrate_k: int = 10,
    and_quota_frac: float = 0.25,
    nand_quota_frac: float = 0.25,
    gate_min_frac: float = 0.20,
    mutpb: float = 0.30,
    sbx_eta: float = 2.0,
    n_jobs: int = None,
):
    """
    Run two-gene logic-gate CAR-T target search via island-model Genetic
    Algorithm, against one or both healthy reference atlases.

    Parameters
    ----------
    atlas : str
        Which healthy reference atlas(es) to score safety against:
          "hpa"    - HPA all-tissues (geosketch 10k) only.
          "tabula" - Tabula Sapiens all-tissues (10k) only.
          "both"   - run independently against EACH atlas, save individual
                     per-atlas results, then combine the two ranked
                     candidate lists via Robust Rank Aggregation (RRA).
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
        Fraction of healthy samples NOT expressing the combination — used
        as the GA fitness cutoff *inside* each atlas's search. Default 0.9.
    rra_efficacy_threshold, rra_safety_threshold : float
        Per-atlas thresholds a candidate must clear in BOTH atlases to be
        eligible for Robust Rank Aggregation. Only used when atlas="both".
        Defaults 0.7 / 0.9 (matches the reference RRA script).
    pop_size : int
        Total population size per seed. Default 1000.
    Gmax : int
        Maximum generations per seed. Default 100.
    Ggap : int
        Interval (generations) between diversity-injection / rare-gene
        immigrant steps. Default 10.
    Rrep : float
        Fraction of each island replaced during diversity injection.
        Default 0.1.
    patience : int
        Early-stop a seed if its best fitness hasn't improved for this many
        consecutive generations. Default 50.
    n_runs : int
        Independent GA runs (one seed each, seeds 42, 43, 44, ...), run in
        parallel via joblib and combined afterward. Default 10.
    n_islands : int
        Number of islands the population is split across. Default 4.
    migrate_interval : int
        Generations between ring-migration events. Default 10.
    migrate_k : int
        Individuals migrated per island at each migration event. Default 10.
    and_quota_frac, nand_quota_frac : float
        Share of pop_size pre-seeded as "A & B" / "A & !B" individuals at
        initialisation (remainder is open/random-gate). Defaults (0.25 /
        0.25) match the reference script's 250/250/500 split for
        pop_size=1000.
    gate_min_frac : float
        Minimum fraction of each gate type guaranteed to survive gate-quota
        tournament selection. Default 0.20.
    mutpb : float
        Mutation probability applied by varAnd each generation. Default 0.30.
    sbx_eta : float
        Distribution index for SBX (simulated binary bounded) crossover.
        Default 2.0.
    n_jobs : int or None
        Parallel workers across seeds (joblib, multiprocessing backend).
        Defaults to n_runs (one worker per seed).

    Returns
    -------
    If atlas == "hpa" or "tabula":
        (df_hof, df_all) — for that single atlas.
    If atlas == "both":
        dict with keys:
          "hpa"    -> (df_hof_hpa, df_all_hpa)
          "tabula" -> (df_hof_tabula, df_all_tabula)
          "rra"    -> df_rra   (RRA-combined, ranked candidate table)
    """
    atlas = (atlas or "both").strip().lower()
    if atlas not in ("hpa", "tabula", "both"):
        raise ValueError(f"atlas must be one of 'hpa', 'tabula', 'both' — got {atlas!r}")

    output_dir = os.getcwd()

    t_path = tumor_path or _auto_tumor_h5ad()
    print(f"Loading tumour matrix: {t_path}")
    adata_tumor = sc.read_h5ad(t_path)
    tumor_genes = list(adata_tumor.var_names)

    ga_kwargs = dict(
        safety_threshold=safety_threshold, pop_size=pop_size, Gmax=Gmax,
        Ggap=Ggap, Rrep=Rrep, patience=patience, n_runs=n_runs,
        output_dir=output_dir, n_islands=n_islands,
        migrate_interval=migrate_interval, migrate_k=migrate_k,
        and_quota_frac=and_quota_frac, nand_quota_frac=nand_quota_frac,
        gate_min_frac=gate_min_frac, mutpb=mutpb,
        sbx_eta=sbx_eta, n_jobs=n_jobs,
    )

    if atlas == "hpa":
        healthy_path = _resolve_atlas_path("hpa", hpa_path)
        print(f"\nAtlas selection: HPA only ({ATLAS_LABELS['hpa']})")
        return _run_single_atlas("hpa", healthy_path, adata_tumor, tumor_genes, **ga_kwargs)

    if atlas == "tabula":
        healthy_path = _resolve_atlas_path("tabula", tabula_path)
        print(f"\nAtlas selection: Tabula Sapiens only ({ATLAS_LABELS['tabula']})")
        return _run_single_atlas("tabula", healthy_path, adata_tumor, tumor_genes, **ga_kwargs)

    # atlas == "both"
    hpa_healthy_path    = _resolve_atlas_path("hpa", hpa_path)
    tabula_healthy_path = _resolve_atlas_path("tabula", tabula_path)

    print("\nAtlas selection: BOTH (independent runs + Robust Rank Aggregation)")

    print("\n" + "=" * 70)
    print(f"  Running GA search — ATLAS 1/2: {ATLAS_LABELS['hpa']}")
    print("=" * 70)
    df_hof_hpa, df_all_hpa = _run_single_atlas(
        "hpa", hpa_healthy_path, adata_tumor, tumor_genes, **ga_kwargs
    )

    print("\n" + "=" * 70)
    print(f"  Running GA search — ATLAS 2/2: {ATLAS_LABELS['tabula']}")
    print("=" * 70)
    df_hof_tabula, df_all_tabula = _run_single_atlas(
        "tabula", tabula_healthy_path, adata_tumor, tumor_genes, **ga_kwargs
    )

    df_rra = _robust_rank_aggregation(
        df_all_hpa, df_all_tabula,
        efficacy_threshold=rra_efficacy_threshold,
        safety_threshold=rra_safety_threshold,
        output_dir=output_dir,
    )

    return {
        "hpa":    (df_hof_hpa,    df_all_hpa),
        "tabula": (df_hof_tabula, df_all_tabula),
        "rra":    df_rra,
    }
