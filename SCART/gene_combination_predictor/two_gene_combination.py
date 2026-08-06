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


def _load_h5ad_subset(h5ad_path: str, target_genes: list = None):
    """
    Fast, memory-safe h5ad loader.

    FIX: col_indices sorted before h5py dense indexing; un-permuted after.
    Scipy sparse paths use raw (unsorted) indices — they accept any order.

    Returns: matrix (int8 ndarray), genes (list[str])
    """
    import h5py
    import scipy.sparse as _sp

    if target_genes is None:
        adata = sc.read_h5ad(h5ad_path)
        X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
        return (X > 0).astype(np.int8), list(adata.var_names)

    with h5py.File(h5ad_path, "r") as f:

        if "var" in f:
            var_grp = f["var"]
            if "_index" in var_grp:
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
#     package (aggregateRanks(method="RRA")) via rpy2, so the statistics are
#     identical to the reference implementation.
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


def _run_rra_via_r(*rank_lists) -> dict:
    """
    Call R's RobustRankAggreg::aggregateRanks(method="RRA") via rpy2 to
    combine the per-atlas rank lists into a single robust rank score per
    candidate ID. Requires the R package 'RobustRankAggreg' (installed by
    `python -m SCART.install`, see install.py).
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import StrVector
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        raise ImportError(
            "rpy2 is required for Robust Rank Aggregation (atlas='both'). "
            "It is one of SCART's core dependencies — if missing, run:\n"
            "  pip install rpy2>=3.5"
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


def _plot_top_rra_candidates(df_ranked: pd.DataFrame, output_dir: str, top_n: int = 20):
    """Grouped bar chart of the top-N RRA-ranked candidates (efficacy +
    per-atlas safety), mirroring the ggplot2 plot in the reference R script."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = df_ranked.sort_values(by="RRA_Rank").head(top_n).copy()
    if top.empty:
        print("No RRA-ranked candidates to plot — skipping plot.")
        return

    def _combo_label(row):
        gate_symbol = row["gate"].replace("A", "").replace("B", "")
        if row["geneA"] == "MSLN":
            return f"{row['geneA']}{gate_symbol}{row['geneB']}"
        if row["geneB"] == "MSLN":
            return f"{row['geneB']}{gate_symbol}{row['geneA']}"
        return f"{row['geneA']}{gate_symbol}{row['geneB']}"

    top["Combo"] = top.apply(_combo_label, axis=1)

    # efficacy is atlas-invariant by construction (same GA fitness definition
    # in both runs), hpa_efficacy is used as the single "Efficacy" series.
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
    ax.set_xticklabels(top["Combo"], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Top 20 RRA-Ranked Dual Candidates (HPA + Tabula Sapiens)")
    ax.legend()
    fig.tight_layout()

    pdf_path = os.path.join(output_dir, "Top20_RRA_Dual_Candidates_HPA_Tabula.pdf")
    png_path = os.path.join(output_dir, "Top20_RRA_Dual_Candidates_HPA_Tabula.png")
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)

    print(f"Top-20 RRA plot saved to:\n  {pdf_path}\n  {png_path}")


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

    rra_scores = _run_rra_via_r(hpa_rank, tabula_rank)

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

    _plot_top_rra_candidates(strict, output_dir)

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
