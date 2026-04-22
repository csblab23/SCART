#!/usr/bin/env python
# coding: utf-8
"""
two_gene_combination.py
Module 4b — Two-gene logic-gate CAR-T target evaluation (Genetic Algorithm)

Searches over all (geneA, geneB, logic_gate) combinations using a Genetic
Algorithm (DEAP) to find pairs that maximise tumour killing (efficacy) while
sparing healthy tissue (safety).

Logic gates evaluated:
  A & B   — both genes must be expressed  (AND)
  A | B   — either gene expressed         (OR)
  A & !B  — A expressed, B NOT expressed  (NOT-B gate)

Healthy matrix source (in priority order):
  1. User-supplied HPA file  (hpa_path=)  → .h5ad  or  .tsv/.tsv.gz
  2. Auto-downloaded HPA single-cell read-count TSV from proteinatlas.org
  3. Legacy final_healthy.h5ad  (backward compatibility)

HPA matrix binarised: 0 = not expressed, 1 = any expression present.
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

# ──────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────
HPA_ZIP_URL = "https://www.proteinatlas.org/download/tsv/rna_single_cell_read_count.zip"
HPA_CACHE   = os.path.join(os.getcwd(), "hpa_cache", "rna_single_cell_read_count.tsv")


def _auto_tumor_h5ad() -> str:
    """
    Auto-detect final_tumor.h5ad produced by Module 3.
    Searches cwd and common output subdirectories.
    """
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
        "Expected locations:\n"
        "  <cwd>/preprocessing_results/final_tumor.h5ad\n"
        "  <cwd>/final_tumor.h5ad\n"
        "Pass tumor_path= explicitly if saved elsewhere."
    )

# ──────────────────────────────────────────────────────────────────────────
# HPA helpers  (shared with one_gene_combination)
# ──────────────────────────────────────────────────────────────────────────

def _download_hpa(cache_path: str) -> str:
    """Download and unzip the HPA single-cell TSV, return local path."""
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
    """
    Parse HPA single-cell TSV → binary (cell_types × genes) numpy array.

    Expected columns: Gene [name], Cell type, Read count
    Returns: matrix (int8), genes (list), cells (list)
    """
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

    print(f"HPA matrix built: {len(cells)} cell types × {len(genes)} genes")
    return matrix, genes, cells


def _load_h5ad_subset(h5ad_path: str, target_genes: list = None):
    """
    Memory-safe h5ad loader — loads only target_genes from disk.
    Avoids OOM on large reference files (e.g. HPA 664k × 19k = 98 GB).
    Returns: matrix (int8 ndarray), genes (list[str])
    """
    if target_genes is None:
        adata = sc.read_h5ad(h5ad_path)
        X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
        return (X > 0).astype(np.int8), list(adata.var_names)

    adata_backed = sc.read_h5ad(h5ad_path, backed="r")
    all_genes    = list(adata_backed.var_names)
    common       = [g for g in target_genes if g in set(all_genes)]

    if len(common) == 0:
        adata_backed.file.close()
        raise ValueError(
            f"No overlap between target genes and h5ad var_names in {h5ad_path}.\n"
            "Check that both datasets use the same gene symbol convention (HGNC)."
        )

    print(f"  HPA h5ad has {len(all_genes)} genes — "
          f"loading only {len(common)} that overlap with tumour matrix.")

    adata_sub = adata_backed[:, common].to_memory()
    adata_backed.file.close()

    X = adata_sub.X.toarray() if not isinstance(adata_sub.X, np.ndarray) else adata_sub.X
    return (X > 0).astype(np.int8), list(adata_sub.var_names)


def _load_healthy_matrix(hpa_path=None, target_genes=None):
    """
    Load and binarise the healthy/normal expression matrix.

    target_genes : list[str] or None
        When provided, only these genes are loaded from h5ad files —
        avoids OOM on large reference files like HPA (98 GB full size).

    Priority:
      1. hpa_path ends with .h5ad  → memory-safe backed-mode load
      2. hpa_path ends with .tsv   → parse as HPA TSV
      3. hpa_path = None           → auto-download HPA TSV
      4. Fallback                  → legacy final_healthy.h5ad

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
            matrix, genes = _load_h5ad_subset(legacy, target_genes)
            return matrix, genes, f"legacy: {legacy}"

    raise FileNotFoundError(
        "No healthy/HPA matrix available.\n"
        "Provide hpa_path= or ensure final_healthy.h5ad exists."
    )


# ──────────────────────────────────────────────────────────────────────────
# Logic gates
# ──────────────────────────────────────────────────────────────────────────

LOGIC_GATES = ["A & B", "A | B", "A & !B"]


def evaluate_gate(expression: str, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if expression == "A & B":
        return A & B
    elif expression == "A | B":
        return A | B
    elif expression == "A & !B":
        return A & (~B.astype(bool))
    raise ValueError(f"Unsupported logic expression: {expression}")


# ──────────────────────────────────────────────────────────────────────────
# Module-level matrices (set inside run() before multiprocessing starts)
# ──────────────────────────────────────────────────────────────────────────
_tumor_matrix   = None
_healthy_matrix = None
_gene_names     = None
_n_genes        = None
_safety_thresh  = 0.9
_logic_gates    = LOGIC_GATES

# DEAP toolbox — module-level so multiprocessing workers can access it
toolbox = None


def _init_deap(n_genes: int, safety_threshold: float):
    """Initialise DEAP creators and toolbox."""
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
    """Fitness function — uses module-level matrices."""
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
    """Top-level wrapper for multiprocessing pool.map."""
    ind.fitness.values = toolbox.evaluate(ind)
    return ind


# ──────────────────────────────────────────────────────────────────────────
# Post-processing helpers
# ──────────────────────────────────────────────────────────────────────────

def _normalize_gene_pair(genes):
    return tuple(sorted(genes))


def _postprocess_results(df: pd.DataFrame) -> pd.DataFrame:
    """Remove same-gene pairs, deduplicate by gene pair, sort by efficacy."""
    if isinstance(df.iloc[0]["Genes"], str):
        df["Genes"] = df["Genes"].apply(eval)

    df = df[df["Genes"].apply(lambda g: g[0] != g[1])].copy()
    df["GenePairKey"] = df["Genes"].apply(_normalize_gene_pair)
    df = df.sort_values(by="Efficacy", ascending=False)
    df = df.drop_duplicates(subset=["GenePairKey"], keep="first")
    return df.drop(columns=["GenePairKey"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────
# Single GA run
# ──────────────────────────────────────────────────────────────────────────

def _run_ga(
    seed: int,
    pop_size: int,
    Gmax: int,
    Ggap: int,
    Rrep: float,
    patience: int,
    n_cpus: int,
):
    random.seed(seed)
    np.random.seed(seed)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(100)

    max_fitness                  = 0
    generations_without_improve  = 0
    all_results                  = []

    for gen in range(Gmax):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)

        with mp.Pool(processes=n_cpus) as pool:
            offspring = pool.map(_evaluate_individual_mp, offspring)

        # Periodic random-immigrant injection
        if gen > 0 and gen % Ggap == 0:
            n_replace = int(Rrep * pop_size)
            offspring.sort(key=lambda ind: ind.fitness.values[0])
            for i in range(n_replace):
                new_ind = toolbox.individual()
                new_ind.fitness.values = toolbox.evaluate(new_ind)
                gA, gB, gT = new_ind
                new_ind.generation  = gen
                new_ind.seed_value  = seed
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

        progress = (gen + 1) / Gmax * 100
        print(f"\rProgress: {progress:.1f}% completed", end="")

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


# ──────────────────────────────────────────────────────────────────────────
# Public entry-point
# ──────────────────────────────────────────────────────────────────────────

def run(
    # Healthy matrix
    hpa_path: str = None,
    # Tumour matrix
    tumor_path: str = None,
    # Safety
    safety_threshold: float = 0.9,
    # GA parameters
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

    Results are saved as CSV files in the current working directory.

    Parameters
    ----------
    hpa_path : str or None
        Path to user-supplied healthy/HPA file (.h5ad or .tsv/.tsv.gz).
        If None, HPA data is auto-downloaded from proteinatlas.org.
    tumor_path : str or None
        Path to tumour h5ad (Module 3 output).
        If None, auto-detected from the SCART package directory.
    safety_threshold : float
        Fraction of healthy samples that must NOT express the gene combination.
        Range 0–1. Default 0.9.
        Higher → more stringent safety, fewer valid combinations.
        Lower  → more candidates, higher off-tumour risk.
    pop_size : int
        GA population size per generation. Default 1000.
        Larger = better exploration, slower per generation.
        Typical range: 200–5000.
    Gmax : int
        Maximum number of generations. Default 100.
        More generations = more thorough search, longer runtime.
        Typical range: 50–500.
    Ggap : int
        Every Ggap generations, Rrep fraction of worst individuals are
        replaced with fresh random ones (random immigrants).
        Prevents premature convergence. Default 10.
    Rrep : float
        Fraction of population replaced at each Ggap interval.
        Range 0.0–0.5. Default 0.1 (10%).
    patience : int
        Stop early if fitness does not improve for this many generations.
        Default 50. Set to Gmax to disable early stopping.
    n_cpus : int
        CPU cores used for parallel fitness evaluation. Default 1.
        Increase to speed up the GA (e.g. n_cpus=8 on a laptop,
        n_cpus=40 on an HPC node).
    n_runs : int
        Number of independent GA runs with different random seeds.
        Default 10. More runs = more diverse solutions but longer runtime.

    Returns
    -------
    df_hof : pd.DataFrame   Hall-of-Fame results (best unique pairs)
    df_all : pd.DataFrame   Complete results across all runs and generations
    """
    global _tumor_matrix, _healthy_matrix, _gene_names, _n_genes, _safety_thresh

    # ── Resolve output directory (always cwd) ─────────────────────────
    output_dir = os.getcwd()

    # ── Load tumour matrix ─────────────────────────────────────────────
    t_path = tumor_path or _auto_tumor_h5ad()
    print(f"Loading tumour matrix: {t_path}")
    adata_tumor  = sc.read_h5ad(t_path)
    tumor_genes  = list(adata_tumor.var_names)

    # ── Load healthy matrix (pass tumour genes for memory-safe slicing) ─
    healthy_matrix_full, healthy_genes, healthy_source = _load_healthy_matrix(
        hpa_path, target_genes=tumor_genes
    )
    print(f"Healthy matrix source: {healthy_source}")

    # ── Align gene spaces ──────────────────────────────────────────────
    common_genes = sorted(set(tumor_genes) & set(healthy_genes))

    if len(common_genes) == 0:
        raise ValueError(
            "No common genes between tumour and healthy matrices.\n"
            "Check that both use the same gene symbol convention (HGNC)."
        )
    print(f"Common genes: {len(common_genes)}")

    # Tumour subset
    adata_tumor = adata_tumor[:, common_genes].copy()
    X_tumor = adata_tumor.X.toarray() if not isinstance(adata_tumor.X, np.ndarray) else adata_tumor.X
    tumor_mat = (X_tumor > 0).astype(np.int8)

    # Healthy subset — reindex to common_genes order
    hg_idx  = {g: i for i, g in enumerate(healthy_genes)}
    col_idx = np.array([hg_idx[g] for g in common_genes])
    healthy_mat = healthy_matrix_full[:, col_idx]

    print(f"Tumour matrix  : {tumor_mat.shape[0]} cells × {len(common_genes)} genes")
    print(f"Healthy matrix : {healthy_mat.shape[0]} samples × {len(common_genes)} genes")

    # ── Set module-level state for multiprocessing workers ────────────
    _tumor_matrix   = tumor_mat
    _healthy_matrix = healthy_mat
    _gene_names     = common_genes
    _n_genes        = len(common_genes)
    _safety_thresh  = safety_threshold

    # ── Initialise DEAP ────────────────────────────────────────────────
    _init_deap(_n_genes, safety_threshold)

    # ── Run GA for each seed ───────────────────────────────────────────
    all_hof     = []
    all_results = []

    for run_id in range(n_runs):
        seed = 42 + run_id
        print(f"\nStarting run {run_id + 1}/{n_runs}  (seed={seed})")

        hof, results = _run_ga(
            seed=seed,
            pop_size=pop_size,
            Gmax=Gmax,
            Ggap=Ggap,
            Rrep=Rrep,
            patience=patience,
            n_cpus=n_cpus,
        )

        df_run = pd.DataFrame(
            results,
            columns=["generation", "LogicGates", "Genes", "Efficacy", "Safety", "seed_value"]
        )
        df_run = df_run[["seed_value", "generation", "LogicGates", "Genes", "Efficacy", "Safety"]]
        all_results.append(df_run)
        all_hof.extend(hof)

    # ── Post-process complete results ──────────────────────────────────
    df_all = pd.concat(all_results, ignore_index=True)
    df_all = _postprocess_results(df_all)
    complete_csv = os.path.join(output_dir, "two_gene_complete.csv")
    df_all.to_csv(complete_csv, index=False)
    print(f"\nComplete results saved to: {complete_csv}")

    # ── Post-process Hall of Fame ──────────────────────────────────────
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

    df_hof = pd.DataFrame(
        hof_data,
        columns=["seed_value", "generation", "LogicGates", "Genes", "Efficacy", "Safety"]
    )
    df_hof = _postprocess_results(df_hof)
    hof_csv = os.path.join(output_dir, "two_gene_hof.csv")
    df_hof.to_csv(hof_csv, index=False)
    print(f"Hall of Fame saved to: {hof_csv}")

    print("\nTop 10 from Hall of Fame:")
    print(df_hof.head(10).to_string(index=False))

    return df_hof, df_all
