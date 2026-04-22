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

Healthy matrix source (priority order):
  1. User-supplied HPA file  (hpa_path=)  .h5ad or .tsv/.tsv.gz
  2. Auto-downloaded HPA single-cell TSV from proteinatlas.org
  3. Legacy final_healthy.h5ad

Fix applied
-----------
_load_h5ad_subset: same h5py sorted-indices fix as one_gene_combination.py.
"""

import os
import zipfile
import urllib.request
import logging
import random
import multiprocessing as mp

import numpy as np
import pandas as pd
import scanpy as sc
from deap import base, creator, tools, algorithms

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


# Module-level matrices (set inside run() before multiprocessing starts)
_tumor_matrix   = None
_healthy_matrix = None
_gene_names     = None
_n_genes        = None
_safety_thresh  = 0.9
_logic_gates    = LOGIC_GATES

toolbox = None


def _init_deap(n_genes: int, safety_threshold: float):
    global toolbox

    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("geneA",      random.randrange, n_genes)
    toolbox.register("geneB",      random.randrange, n_genes)
    toolbox.register("gate",       random.randrange, len(LOGIC_GATES))
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.geneA, toolbox.geneB, toolbox.gate), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate",   _evaluate_fitness)
    toolbox.register("mate",       tools.cxOnePoint)
    toolbox.register("mutate",     tools.mutUniformInt,
                     low=[0, 0, 0],
                     up=[n_genes - 1, n_genes - 1, len(LOGIC_GATES) - 1],
                     indpb=0.2)
    toolbox.register("select",     tools.selTournament, tournsize=2)


def _evaluate_fitness(individual):
    geneA_idx, geneB_idx, gate_type_idx = individual
    gate_type = _logic_gates[gate_type_idx]

    A_tumor   = _tumor_matrix[:, geneA_idx]
    B_tumor   = _tumor_matrix[:, geneB_idx]
    A_healthy = _healthy_matrix[:, geneA_idx]
    B_healthy = _healthy_matrix[:, geneB_idx]

    output_tumor   = evaluate_gate(gate_type, A_tumor,   B_tumor)
    output_healthy = evaluate_gate(gate_type, A_healthy, B_healthy)

    efficacy = np.sum(output_tumor)        / len(output_tumor)
    safety   = np.sum(output_healthy == 0) / len(output_healthy)

    individual.safety = safety
    return (efficacy if safety >= _safety_thresh else 0,)


def _evaluate_individual_mp(ind):
    ind.fitness.values = toolbox.evaluate(ind)
    return ind


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


def _run_ga(seed, pop_size, Gmax, Ggap, Rrep, patience, n_cpus):
    random.seed(seed)
    np.random.seed(seed)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(100)

    max_fitness                 = 0
    generations_without_improve = 0
    all_results                 = []

    for gen in range(Gmax):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)

        with mp.Pool(processes=n_cpus) as pool:
            offspring = pool.map(_evaluate_individual_mp, offspring)

        if gen > 0 and gen % Ggap == 0:
            n_replace = int(Rrep * pop_size)
            offspring.sort(key=lambda ind: ind.fitness.values[0])
            for i in range(n_replace):
                new_ind = toolbox.individual()
                new_ind.fitness.values = toolbox.evaluate(new_ind)
                gA, gB, gT = new_ind
                new_ind.generation = gen
                new_ind.seed_value = seed
                all_results.append([
                    gen, LOGIC_GATES[gT],
                    [_gene_names[gA], _gene_names[gB]],
                    new_ind.fitness.values[0],
                    getattr(new_ind, "safety", None),
                    seed,
                ])
                offspring[i] = new_ind

        for ind in offspring:
            gA, gB, gT = ind
            ind.generation = gen
            ind.seed_value = seed
            all_results.append([
                gen, LOGIC_GATES[gT],
                [_gene_names[gA], _gene_names[gB]],
                ind.fitness.values[0],
                getattr(ind, "safety", None),
                seed,
            ])

        pop = toolbox.select(offspring, k=pop_size)
        hof.update(pop)

        print(f"\rProgress: {(gen+1)/Gmax*100:.1f}% completed", end="")

        current_best = max(ind.fitness.values[0] for ind in pop)
        if current_best > max_fitness:
            max_fitness                 = current_best
            generations_without_improve = 0
        else:
            generations_without_improve += 1

        if generations_without_improve >= patience:
            print(f"\nEarly stopping at generation {gen}")
            break

    return hof, all_results


def run(
    hpa_path: str = None,
    tumor_path: str = None,
    safety_threshold: float = 0.9,
    pop_size: int  = 1000,
    Gmax: int      = 100,
    Ggap: int      = 10,
    Rrep: float    = 0.1,
    patience: int  = 50,
    n_cpus: int    = 1,
    n_runs: int    = 10,
):
    """
    Run two-gene logic-gate CAR-T target search via Genetic Algorithm.

    Parameters
    ----------
    hpa_path : str or None
        Path to user-supplied healthy/HPA file (.h5ad or .tsv/.tsv.gz).
        If None, auto-downloaded from proteinatlas.org.
    tumor_path : str or None
        Path to tumour h5ad (Module 3 output). Auto-detected if None.
    safety_threshold : float
        Fraction of healthy samples NOT expressing the combination. Default 0.9.
    pop_size : int   GA population per generation. Default 1000.
    Gmax : int       Max generations. Default 100.
    Ggap : int       Interval for random immigrant injection. Default 10.
    Rrep : float     Fraction replaced at each Ggap. Default 0.1.
    patience : int   Early-stop if no improvement for N generations. Default 50.
    n_cpus : int     CPU cores for parallel evaluation. Default 1.
    n_runs : int     Independent GA runs. Default 10.

    Returns
    -------
    df_hof : pd.DataFrame   Hall-of-Fame results (best unique pairs)
    df_all : pd.DataFrame   Complete results across all runs
    """
    global _tumor_matrix, _healthy_matrix, _gene_names, _n_genes, _safety_thresh

    output_dir = os.getcwd()

    t_path = tumor_path or _auto_tumor_h5ad()
    print(f"Loading tumour matrix: {t_path}")
    adata_tumor = sc.read_h5ad(t_path)
    tumor_genes = list(adata_tumor.var_names)

    healthy_matrix_full, healthy_genes, healthy_source = _load_healthy_matrix(
        hpa_path, target_genes=tumor_genes
    )
    print(f"Healthy matrix source: {healthy_source}")

    common_genes = sorted(set(tumor_genes) & set(healthy_genes))
    if len(common_genes) == 0:
        raise ValueError(
            "No common genes between tumour and healthy matrices.\n"
            "Check both datasets use HGNC gene symbols."
        )
    print(f"Common genes: {len(common_genes)}")

    adata_tumor = adata_tumor[:, common_genes].copy()
    X_tumor     = adata_tumor.X.toarray() if not isinstance(adata_tumor.X, np.ndarray) else adata_tumor.X
    tumor_mat   = (X_tumor > 0).astype(np.int8)

    hg_idx      = {g: i for i, g in enumerate(healthy_genes)}
    col_idx     = np.array([hg_idx[g] for g in common_genes])
    healthy_mat = healthy_matrix_full[:, col_idx]

    print(f"Tumour matrix  : {tumor_mat.shape[0]} cells x {len(common_genes)} genes")
    print(f"Healthy matrix : {healthy_mat.shape[0]} samples x {len(common_genes)} genes")

    _tumor_matrix   = tumor_mat
    _healthy_matrix = healthy_mat
    _gene_names     = common_genes
    _n_genes        = len(common_genes)
    _safety_thresh  = safety_threshold

    _init_deap(_n_genes, safety_threshold)

    all_hof     = []
    all_results = []

    for run_id in range(n_runs):
        seed = 42 + run_id
        print(f"\nStarting run {run_id + 1}/{n_runs}  (seed={seed})")

        hof, results = _run_ga(
            seed=seed, pop_size=pop_size, Gmax=Gmax,
            Ggap=Ggap, Rrep=Rrep, patience=patience, n_cpus=n_cpus,
        )

        df_run = pd.DataFrame(
            results,
            columns=["generation", "LogicGates", "Genes", "Efficacy", "Safety", "seed_value"]
        )
        df_run = df_run[["seed_value", "generation", "LogicGates", "Genes", "Efficacy", "Safety"]]
        all_results.append(df_run)
        all_hof.extend(hof)

    df_all       = pd.concat(all_results, ignore_index=True)
    df_all       = _postprocess_results(df_all)
    complete_csv = os.path.join(output_dir, "two_gene_complete.csv")
    df_all.to_csv(complete_csv, index=False)
    print(f"\nComplete results saved to: {complete_csv}")

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
            logger.warning(f"Skipping HOF individual: {e}")

    df_hof   = pd.DataFrame(
        hof_data,
        columns=["seed_value", "generation", "LogicGates", "Genes", "Efficacy", "Safety"]
    )
    df_hof   = _postprocess_results(df_hof)
    hof_csv  = os.path.join(output_dir, "two_gene_hof.csv")
    df_hof.to_csv(hof_csv, index=False)
    print(f"Hall of Fame saved to: {hof_csv}")

    print("\nTop 10 from Hall of Fame:")
    print(df_hof.head(10).to_string(index=False))

    return df_hof, df_all
