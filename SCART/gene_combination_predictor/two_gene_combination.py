#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import pandas as pd
import scanpy as sc
from deap import base, creator, tools, algorithms
import multiprocessing as mp
import os

# =====================
# Top-level function for multiprocessing
# =====================
def evaluate_individual(ind):
    ind.fitness.values = toolbox.evaluate(ind)
    return ind

# =====================
# Main module function
# =====================
def run(
    safety_threshold=0.9,
    pop_size=1000,
    Gmax=100,
    Ggap=10,
    Rrep=0.1,
    patience=50,
    n_cpus=40
):

    # =====================
    # File paths
    # =====================
    PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))

    TUMOR_PATH = os.path.join(BASE_DIR, "preprocessed_input", "final_tumor.h5ad")
    HEALTHY_PATH = os.path.join(BASE_DIR, "preprocessed_input", "final_healthy.h5ad")

    OUTPUT_DIR = os.path.join(BASE_DIR, "scT-CAR_Designer", "tumor_h5ad_out")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =====================
    # Load matrices
    # =====================
    adata_tumor = sc.read_h5ad(TUMOR_PATH)
    adata_healthy = sc.read_h5ad(HEALTHY_PATH)

    tumor_matrix = adata_tumor.X.toarray() if not isinstance(adata_tumor.X, np.ndarray) else adata_tumor.X
    tumor_matrix = (tumor_matrix > 0).astype(int)

    healthy_matrix = adata_healthy.X.toarray() if not isinstance(adata_healthy.X, np.ndarray) else adata_healthy.X
    healthy_matrix = (healthy_matrix > 0).astype(int)

    gene_names = adata_tumor.var_names.tolist()
    n_genes = len(gene_names)

    # =====================
    # Logic Gates
    # =====================
    logic_gates = ['A & B', 'A | B', 'A & !B']

    def evaluate_gate(expression, A, B):
        if expression == 'A & B':
            return A & B
        elif expression == 'A | B':
            return A | B
        elif expression == 'A & !B':
            return A & (~B.astype(bool))
        else:
            raise ValueError(f"Unsupported logic expression: {expression}")

    def evaluate_fitness(individual, gate_list=logic_gates):
        geneA_idx, geneB_idx, gate_type_idx = individual
        gate_type = gate_list[gate_type_idx]

        A_tumor = tumor_matrix[:, geneA_idx]
        B_tumor = tumor_matrix[:, geneB_idx]
        A_healthy = healthy_matrix[:, geneA_idx]
        B_healthy = healthy_matrix[:, geneB_idx]

        output_tumor = evaluate_gate(gate_type, A_tumor, B_tumor)
        output_healthy = evaluate_gate(gate_type, A_healthy, B_healthy)

        efficacy = np.sum(output_tumor) / len(output_tumor)
        safety = np.sum(output_healthy == 0) / len(output_healthy)

        individual.safety = safety
        return (efficacy if safety >= safety_threshold else 0,)

    # =====================
    # DEAP setup
    # =====================
    global toolbox
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("geneA", random.randrange, n_genes)
    toolbox.register("geneB", random.randrange, n_genes)
    toolbox.register("gate", random.randrange, len(logic_gates))
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.geneA, toolbox.geneB, toolbox.gate), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_fitness)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt,
                     low=[0, 0, 0],
                     up=[n_genes - 1, n_genes - 1, len(logic_gates) - 1],
                     indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=2)

    # =====================
    # Run GA for one seed
    # =====================
    def run_ga(seed):
        random.seed(seed)
        np.random.seed(seed)

        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(100)

        max_fitness = 0
        generations_without_improvement = 0
        all_results = []

        for gen in range(Gmax):
            offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)

            with mp.Pool(processes=n_cpus) as pool:
                offspring = pool.map(evaluate_individual, offspring)

            for ind in offspring:
                gA, gB, gT = ind
                logic_gate = logic_gates[gT]
                gene_pair = [gene_names[gA], gene_names[gB]]
                efficacy = ind.fitness.values[0]
                safety = getattr(ind, "safety", None)
                all_results.append([gen, logic_gate, gene_pair, efficacy, safety, seed])
                ind.generation = gen
                ind.seed_value = seed

            pop = toolbox.select(offspring, k=pop_size)
            hof.update(pop)

            final_generation = gen
            progress = ((final_generation + 1) / Gmax) * 100
            print(f"\rProgress: {progress:.1f}% completed", end="")

            current_best = max(ind.fitness.values[0] for ind in pop)
            if current_best > max_fitness:
                max_fitness = current_best
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            if generations_without_improvement >= patience:
                print(f"Early stopping")
                break

        return hof, all_results

    # =====================
    # Execute 10 seeds
    # =====================
    all_hof = []
    all_results = []

    for run_id in range(10):
        print(f"\nStarting run {run_id + 1}/10")
        seed = 42 + run_id
        hof, results = run_ga(seed)

        df_results = pd.DataFrame(
            results,
            columns=["generation", "LogicGates", "Genes", "Efficacy", "Safety", "seed_value"]
        )
        df_results = df_results[["seed_value", "generation", "LogicGates", "Genes", "Efficacy", "Safety"]]
        all_results.append(df_results)
        all_hof.extend(hof)

    def normalize_gene_pair(genes):
        return tuple(sorted(genes))

    def postprocess_results(df):
        df = df[df["Genes"].apply(lambda g: g[0] != g[1])].copy()
        df["GenePairKey"] = df["Genes"].apply(normalize_gene_pair)
        df = df.sort_values(by="Efficacy", ascending=False)
        df = df.drop_duplicates(subset=["GenePairKey"], keep="first")
        return df.drop(columns=["GenePairKey"]).reset_index(drop=True)

    df_all = pd.concat(all_results, ignore_index=True)
    df_all = postprocess_results(df_all)
    df_all.to_csv(os.path.join(OUTPUT_DIR, "two_gene_complete.csv"), index=False)

    hof_data = []
    for ind in all_hof:
        gA, gB, gT = ind
        logic_gate = logic_gates[gT]
        gene_pair = [gene_names[gA], gene_names[gB]]
        efficacy = ind.fitness.values[0]
        safety = getattr(ind, "safety", None)
        generation = getattr(ind, "generation", None)
        seed_value = getattr(ind, "seed_value", None)
        hof_data.append([seed_value, generation, logic_gate, gene_pair, efficacy, safety])

    df_hof = pd.DataFrame(
        hof_data,
        columns=["seed_value", "generation", "LogicGates", "Genes", "Efficacy", "Safety"]
    )
    df_hof = postprocess_results(df_hof)
    df_hof.to_csv(os.path.join(OUTPUT_DIR, "two_gene_hof.csv"), index=False)

    print("\nTop 10 from Hall of Fame:")
    print(df_hof.head(10))