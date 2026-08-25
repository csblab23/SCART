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

  _plot_top_gate_rra_candidates() — top 3 AND, top 3 OR, and top 3 NAND
      (A & !B) candidates by RRA_Rank, one colour per gate type. Unlike the
      reference script (which excludes NAND), NAND is included here.
  _plot_top10_rra_candidates()    — top 10 candidates overall by RRA_Rank,
      regardless of gate type.

Both are shown inline (in addition to being saved to disk) when run inside
a Jupyter kernel; see _configure_matplotlib_backend().

Fix applied (RRA plot styling)
--------------------------------
The two dual-panel RRA plots above have been restyled to match, as closely
as matplotlib allows, the look and feel of the reference R/ggplot2 +
cowplot + ggrepel scatter script (theme_classic, diamond-shaped highlighted
points, a colour-blind-friendly Okabe-Ito-derived highlight palette
assigned per-candidate rather than per-gate-type, white ggrepel-style label
chips, a light-grey "grey96" background on the zoomed right panel, and the
same axis-limit / zoom-rectangle logic — including the reference script's
somewhat unusual choice of leaving the left panel's x-axis unrestricted
while still clamping its y-axis to the zoom_y range). Only the plotting
helpers below were touched; the GA, RRA aggregation, and data-loading logic
are unchanged.
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
# Restyled to match, as closely as matplotlib allows, the look of the
# reference R/ggplot2 + cowplot + ggrepel scatter script (avg. safety vs
# HPA efficacy; every candidate shown as grey background on the left panel;
# selected candidates highlighted as black-outlined diamonds in a
# colour-blind-friendly per-candidate palette — the same six-colour
# Okabe-Ito-derived vector as the reference script, assigned in row order;
# a dashed rectangle on the left panel marking the zoomed-in region shown,
# without background points, on the right panel; white ggrepel-style label
# chips):
#
#   _plot_top_gate_rra_candidates() — top 3 AND, top 3 OR, and top 3 NAND
#       (A & !B) candidates by RRA_Rank. Unlike the reference script (which
#       only highlights AND/OR and excludes NAND), NAND is included here.
#   _plot_top10_rra_candidates()    — top 10 candidates overall by
#       RRA_Rank, regardless of gate type.
#
# Both save PDF + PNG (with a `_claude` suffix) and, when run inside a
# Jupyter kernel, also display inline via plt.show() — see
# _configure_matplotlib_backend().
# ─────────────────────────────────────────────────────────────────────────────

_GATE_TYPE_MAP = {"A & B": "AND", "A | B": "OR", "A & !B": "NAND"}

# Colour-blind-friendly (Okabe-Ito derived) palette, assigned one colour
# per highlighted CANDIDATE (not per gate type) in row order — exactly the
# same convention as the reference script's `highlight_colors[seq_len(n())]`.
# The first six entries are identical, in the same order, to the reference
# script's `highlight_colors` vector; entries 7-9 extend the palette so
# NAND candidates (highlighted here, unlike in the reference script) also
# get a distinct colour.
_HIGHLIGHT_COLORS = [
    "#0072B2",  # blue
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
    "#E69F00",  # orange
    "#000000",  # black
    "#F0E442",  # yellow
    "#999999",  # grey
]

# Same palette, extended to 10 entries for the top-10-overall plot.
_RANK_HIGHLIGHT_COLORS = _HIGHLIGHT_COLORS + ["#7F3C8D"]


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
    display label per candidate, and a coarse gate-type category. Label
    format mirrors the reference script's `candidate` column exactly
    (including its `geneA & ! geneB` NAND spacing)."""
    df = df_ranked.dropna(subset=["hpa_safety", "tabula_safety", "hpa_efficacy"]).copy()
    df["avg_safety"] = (df["hpa_safety"] + df["tabula_safety"]) / 2.0
    df["efficacy"]   = df["hpa_efficacy"]  # atlas-invariant by construction
    df["gate_type"]  = df["gate"].map(_GATE_TYPE_MAP).fillna("OTHER")

    def _label(row):
        if row["gate"] == "A & !B":
            return f"{row['geneA']} & ! {row['geneB']}"
        symbol = row["gate"].replace("A", "").replace("B", "").strip()
        return f"{row['geneA']} {symbol} {row['geneB']}"

    df["candidate"] = df.apply(_label, axis=1)
    return df


def _style_scatter_axes(ax, title, xlabel, ylabel):
    """
    Reproduces the reference script's `theme_classic(base_size = 18)` +
    explicit theme() overrides: bold centred title (size 20), bold axis
    titles (size 17), black axis text (size 13), a full black panel
    border/axis line (linewidth 1), no gridlines, and a square aspect
    ratio (handled by the caller via set_box_aspect(1)).
    """
    from matplotlib.ticker import FuncFormatter

    ax.set_title(title, fontsize=20, fontweight="bold", pad=14, color="black", loc="center")
    ax.set_xlabel(xlabel, fontsize=17, fontweight="bold", color="black")
    ax.set_ylabel(ylabel, fontsize=17, fontweight="bold", color="black")
    ax.tick_params(labelsize=13, colors="black")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # theme_classic draws only left/bottom axis lines, but the reference
    # script layers an explicit panel.border rectangle on top of that, so
    # the net visual effect is a full black box around each panel.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    ax.grid(False)
    ax.set_axisbelow(True)


def _repel_labels(ax, xs, ys, labels, colors, x_range, y_range, side="right"):
    """
    Lightweight, dependency-free label placement approximating
    ggrepel::geom_label_repel as configured in the reference script: white
    label fill, a border and text coloured to match each point's own
    highlight colour, bold text, and a thin leader line back to the point.
    Uses the optional `adjustText` package instead, when installed, for a
    tighter layout.
    """
    from matplotlib.colors import to_rgba

    try:
        from adjustText import adjust_text

        texts = [
            ax.text(
                x, y, label, fontsize=11, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=color, linewidth=1.4),
            )
            for x, y, label, color in zip(xs, ys, labels, colors)
        ]
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="#4D4D4D", lw=1.0))
        return
    except ImportError:
        pass

    order      = np.argsort(ys)[::-1]
    n          = len(order)
    y_lo, y_hi = y_range
    margin     = 0.06 * (y_hi - y_lo)
    slots      = (np.linspace(y_hi - margin, y_lo + margin, n)
                  if n > 1 else [ys[order[0]]])
    x_text     = (x_range[1] + 0.16 * (x_range[1] - x_range[0]) if side == "right"
                  else x_range[0] - 0.16 * (x_range[1] - x_range[0]))

    for slot_y, idx in zip(slots, order):
        x, y, label, color = xs[idx], ys[idx], labels[idx], colors[idx]
        ax.annotate(
            label, xy=(x, y), xytext=(x_text, slot_y),
            fontsize=11, fontweight="bold", color=color,
            ha="left" if side == "right" else "right", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=color, linewidth=1.4),
            arrowprops=dict(arrowstyle="-", color="#4D4D4D", lw=1.0,
                             shrinkA=7, shrinkB=7),
            annotation_clip=False,
        )

    if side == "right":
        ax.set_xlim(x_range[0], x_text + 0.06 * (x_range[1] - x_range[0]))
    else:
        ax.set_xlim(x_text - 0.06 * (x_range[1] - x_range[0]), x_range[1])


def _dual_panel_rra_figure(
    plot_df: pd.DataFrame,
    highlighted: pd.DataFrame,
    colors: list,
    out_stub: str,
    output_dir: str,
    suptitle: str,
    left_title: str,
    right_title: str,
    legend_handles=None,
):
    """
    Shared dual-panel figure builder used by both RRA plots below,
    reproducing the reference script's `p_left` / `p_right` + `plot_grid`
    layout as closely as matplotlib allows:

      Left panel  ("All candidate combinations"): every candidate as a
        grey background scatter, highlighted candidates as black-outlined
        coloured diamonds, dashed threshold cross-lines, and a dashed
        grey zoom rectangle. X-axis is left unrestricted (full candidate
        range, matching the reference script's un-limited scale_x); the
        Y-axis is clamped to the zoom_y range with even-numbered breaks,
        matching the reference script's scale_y_continuous(limits=...).

      Right panel (zoomed): only the highlighted diamonds (no grey
        background), on a light "grey96" panel background, both axes
        clamped to the zoom region, with ggrepel-style white label chips.
    """
    in_notebook = _configure_matplotlib_backend()
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
    })

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(suptitle, fontsize=17, fontweight="bold", y=1.03, color="black")

    xs_all = (plot_df["avg_safety"] * 100).to_numpy()
    ys_all = (plot_df["efficacy"] * 100).to_numpy()
    xs_hi  = (highlighted["avg_safety"] * 100).to_numpy()
    ys_hi  = (highlighted["efficacy"] * 100).to_numpy()

    # Threshold cross-lines and zoom rectangle — identical logic to the
    # reference script's `rank6_safety` / `min_highlight_efficacy` and
    # `zoom_x_min/max`, `zoom_y_min/max`.
    threshold_x = float(xs_hi.min())
    threshold_y = float(ys_hi.min())
    zoom_x_lo = max(0.0, np.floor(xs_hi.min()) - 1)
    zoom_x_hi = min(100.0, np.ceil(xs_hi.max()) + 1)
    zoom_y_lo = max(0.0, np.floor(ys_hi.min()) - 1)
    zoom_y_hi = min(100.0, np.ceil(ys_hi.max()) + 1)

    # ---- left panel: full candidate universe, with partition -------------
    ax_l.scatter(xs_all, ys_all, s=26, color="#A6A6A6", alpha=0.6,
                 linewidths=0, zorder=2)

    ax_l.axvline(threshold_x, linestyle="--", linewidth=0.8, color="#4D4D4D", zorder=3)
    ax_l.axhline(threshold_y, linestyle="--", linewidth=0.8, color="#4D4D4D", zorder=3)
    ax_l.add_patch(plt.Rectangle(
        (zoom_x_lo, zoom_y_lo), max(zoom_x_hi - zoom_x_lo, 0.5), max(zoom_y_hi - zoom_y_lo, 0.5),
        fill=True, facecolor="#808080", alpha=0.15,
        edgecolor="black", linestyle="--", linewidth=1.0, zorder=3,
    ))

    ax_l.scatter(xs_hi, ys_hi, s=190, marker="D", color=colors,
                 edgecolor="black", linewidth=1.2, zorder=5)

    ax_l.set_box_aspect(1)
    _style_scatter_axes(ax_l, left_title, "Average safety", "Efficacy")

    # X-axis: unrestricted (full range), small expansion margin — mirrors
    # the reference script leaving scale_x_continuous without limits.
    x_span = xs_all.max() - xs_all.min()
    x_pad  = 0.03 * (x_span if x_span > 0 else 1.0)
    ax_l.set_xlim(xs_all.min() - x_pad, xs_all.max() + x_pad)

    # Y-axis: clamped to the zoom_y range, even-numbered breaks — mirrors
    # the reference script's scale_y_continuous(limits = zoom_y range).
    y_break_lo = (zoom_y_lo // 2) * 2
    y_break_hi = -(-zoom_y_hi // 2) * 2  # ceil to nearest even
    y_pad = 0.08 * ((zoom_y_hi - zoom_y_lo) if zoom_y_hi > zoom_y_lo else 1.0)
    ax_l.set_ylim(zoom_y_lo - y_pad, zoom_y_hi + y_pad)
    ax_l.set_yticks(np.arange(y_break_lo, y_break_hi + 1, 2))

    # ---- right panel: zoomed, labelled highlighted candidates only -------
    # No grey background scatter here — matches the reference script's
    # p_right, which is built purely from `highlighted`.
    ax_r.set_facecolor("#F5F5F5")  # ggplot "grey96"

    ax_r.axvline(threshold_x, linestyle="--", linewidth=0.8, color="#4D4D4D", zorder=3)
    ax_r.axhline(threshold_y, linestyle="--", linewidth=0.8, color="#4D4D4D", zorder=3)

    ax_r.scatter(xs_hi, ys_hi, s=210, marker="D", color=colors,
                 edgecolor="black", linewidth=1.2, zorder=5)

    x_pad_r = 0.08 * ((zoom_x_hi - zoom_x_lo) if zoom_x_hi > zoom_x_lo else 1.0)
    y_pad_r = 0.08 * ((zoom_y_hi - zoom_y_lo) if zoom_y_hi > zoom_y_lo else 1.0)
    x_range = (zoom_x_lo - x_pad_r, zoom_x_hi + x_pad_r)
    y_range = (zoom_y_lo - y_pad_r, zoom_y_hi + y_pad_r)
    ax_r.set_ylim(*y_range)
    ax_r.set_xlim(*x_range)

    _repel_labels(
        ax_r, xs_hi, ys_hi, highlighted["candidate"].tolist(), colors,
        x_range=x_range, y_range=y_range,
    )

    ax_r.set_box_aspect(1)
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


def _plot_top_gate_rra_candidates(df_ranked: pd.DataFrame, output_dir: str, top_n_per_gate: int = 3):
    """
    Top `top_n_per_gate` (default 3) candidates by RRA_Rank within EACH
    gate type — AND, OR, and NAND (A & !B) — highlighted together on one
    dual-panel scatter plot. Styling mirrors the reference R/ggplot2
    scatter script (diamond markers, per-candidate colour palette, white
    ggrepel-style labels); unlike that script, NAND is included here
    rather than excluded, and each highlighted candidate gets its own
    colour (matching the reference script's row-order colour assignment)
    rather than one colour per gate type.
    """
    plot_df = _prep_rra_plot_df(df_ranked)

    pieces = []
    for gate_type in ("AND", "OR", "NAND"):
        sub = plot_df[plot_df["gate_type"] == gate_type].sort_values("RRA_Rank").head(top_n_per_gate)
        if sub.empty:
            print(f"No {gate_type}-gate candidates available for highlighting — skipping.")
            continue
        pieces.append(sub)

    if not pieces:
        print("No AND/OR/NAND candidates available — skipping top-gate RRA plot.")
        return

    highlighted = pd.concat(pieces).sort_values(["gate_type", "RRA_Rank"]).reset_index(drop=True)
    colors      = [_HIGHLIGHT_COLORS[i % len(_HIGHLIGHT_COLORS)] for i in range(len(highlighted))]

    _dual_panel_rra_figure(
        plot_df, highlighted, colors,
        out_stub="Top3_AND_OR_NAND_RRA_Candidates",
        output_dir=output_dir,
        suptitle="Top 3 AND / OR / NAND Candidates — Robust Rank Aggregation",
        left_title="All candidate combinations",
        right_title="Top 3 per gate (AND / OR / NAND)",
    )


def _plot_top10_rra_candidates(df_ranked: pd.DataFrame, output_dir: str, top_n: int = 10):
    """
    Top `top_n` (default 10) candidates overall by RRA_Rank, regardless of
    gate type. Same dual-panel look as _plot_top_gate_rra_candidates(),
    with each candidate assigned its own colour from the extended
    highlight palette and rank numbers baked into each label.
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
    colors = [_RANK_HIGHLIGHT_COLORS[i % len(_RANK_HIGHLIGHT_COLORS)] for i in range(len(highlighted))]

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

    _plot_top_gate_rra_candidates(strict, output_dir)
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
