#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import scanpy as sc
import os
import random
import multiprocessing as mp

from deap import base, creator, tools, algorithms


# ======================================================
# MATRIX LOADER
# ======================================================

def load_matrices(tumor_path, healthy_path=None):

    PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))

    if healthy_path is None:
        healthy_path = os.path.join(
            BASE_DIR,
            "preprocessed_input",
            "final_healthy.h5ad"
        )

    print("Loading tumor matrix:", tumor_path)
    print("Loading healthy matrix:", healthy_path)

    adata_tumor = sc.read_h5ad(tumor_path)
    adata_healthy = sc.read_h5ad(healthy_path)

    common_genes = adata_tumor.var_names.intersection(
        adata_healthy.var_names
    )

    adata_tumor = adata_tumor[:, common_genes].copy()
    adata_healthy = adata_healthy[:, common_genes].copy()

    tumor_matrix = (
        adata_tumor.X.toarray()
        if not isinstance(adata_tumor.X, np.ndarray)
        else adata_tumor.X
    )

    healthy_matrix = (
        adata_healthy.X.toarray()
        if not isinstance(adata_healthy.X, np.ndarray)
        else adata_healthy.X
    )

    tumor_matrix = (tumor_matrix > 0).astype(int)
    healthy_matrix = (healthy_matrix > 0).astype(int)

    gene_names = common_genes.tolist()

    return tumor_matrix, healthy_matrix, gene_names


# ======================================================
# SINGLE GENE
# ======================================================

def run_one_gene(
    tumor_matrix,
    healthy_matrix,
    gene_names,
    safety_threshold=0.9,
    output_file="single_gene_results.csv"
):

    print("Starting single-gene analysis...")

    n_genes = len(gene_names)
    results = []

    for idx in range(n_genes):

        tumor_expr = tumor_matrix[:, idx]
        healthy_expr = healthy_matrix[:, idx]

        efficacy = np.sum(tumor_expr) / len(tumor_expr)
        safety = np.sum(healthy_expr == 0) / len(healthy_expr)

        objective_score = efficacy if safety >= safety_threshold else 0

        results.append([
            gene_names[idx],
            efficacy,
            safety,
            objective_score
        ])

        if idx % max(1, n_genes//100) == 0:
            print(f"\rProgress: {idx/n_genes*100:.1f}%", end="")

    print("\nCompleted")

    df = pd.DataFrame(
        results,
        columns=["Gene","Efficacy","Safety","ObjectiveScore"]
    )

    df.to_csv(output_file,index=False)

    print("\nTop 10:")
    print(
        df[df.Safety>=safety_threshold]
        .sort_values("Efficacy",ascending=False)
        .head(10)
    )

    return df


# ======================================================
# TWO GENE
# ======================================================

logic_gates = ['A & B','A | B','A & !B']


def evaluate_gate(expression,A,B):

    if expression=='A & B':
        return A & B

    if expression=='A | B':
        return A | B

    if expression=='A & !B':
        return A & (~B.astype(bool))


def evaluate_two_gene(ind):

    gA,gB,gT = ind

    A_t = tumor_matrix_global[:,gA]
    B_t = tumor_matrix_global[:,gB]

    A_h = healthy_matrix_global[:,gA]
    B_h = healthy_matrix_global[:,gB]

    gate = logic_gates[gT]

    out_t = evaluate_gate(gate,A_t,B_t)
    out_h = evaluate_gate(gate,A_h,B_h)

    efficacy = np.sum(out_t)/len(out_t)
    safety = np.sum(out_h==0)/len(out_h)

    ind.safety=safety

    return (efficacy if safety>=safety_threshold_global else 0,)


def evaluate_individual(ind):

    ind.fitness.values = toolbox.evaluate(ind)

    return ind



def run_two_gene(
    tumor_matrix,
    healthy_matrix,
    gene_names,
    safety_threshold=0.9,
    pop_size=1000,
    Gmax=100,
    patience=50,
    n_cpus=10
):

    global tumor_matrix_global
    global healthy_matrix_global
    global safety_threshold_global
    global toolbox

    tumor_matrix_global=tumor_matrix
    healthy_matrix_global=healthy_matrix
    safety_threshold_global=safety_threshold


    n_genes=len(gene_names)


    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax",base.Fitness,weights=(1.0,))

    if "Individual" not in creator.__dict__:
        creator.create("Individual",list,
        fitness=creator.FitnessMax)


    toolbox=base.Toolbox()

    toolbox.register("geneA",random.randrange,n_genes)
    toolbox.register("geneB",random.randrange,n_genes)
    toolbox.register("gate",random.randrange,3)


    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (toolbox.geneA,toolbox.geneB,toolbox.gate),
        n=1
    )


    toolbox.register("population",tools.initRepeat,list,toolbox.individual)

    toolbox.register("evaluate",evaluate_two_gene)

    toolbox.register("mate",tools.cxOnePoint)


    toolbox.register(
        "mutate",
        tools.mutUniformInt,
        low=[0,0,0],
        up=[n_genes-1,n_genes-1,2],
        indpb=0.2
    )


    toolbox.register("select",tools.selTournament,tournsize=2)



    # =====================
    # GA per seed
    # =====================

    def run_ga(seed):

        random.seed(seed)
        np.random.seed(seed)

        pop=toolbox.population(pop_size)

        hof=tools.HallOfFame(100)

        all_results=[]


        for gen in range(Gmax):

            offspring=algorithms.varAnd(pop,toolbox,0.5,0.2)


            with mp.Pool(n_cpus) as pool:

                offspring=pool.map(evaluate_individual,offspring)


            for ind in offspring:

                gA,gB,gT=ind

                logic_gate=logic_gates[gT]

                genes=[gene_names[gA],gene_names[gB]]

                efficacy=ind.fitness.values[0]

                safety=getattr(ind,"safety",None)


                all_results.append([
                    gen,
                    logic_gate,
                    genes,
                    efficacy,
                    safety,
                    seed
                ])


                ind.generation=gen
                ind.seed_value=seed


            pop=toolbox.select(offspring,pop_size)

            hof.update(pop)


            final_generation = gen
            progress = ((final_generation + 1) / Gmax) * 100
            print(f"\rProgress: {progress:.1f}% completed", end="")


        return hof,all_results



    # =====================
    # 10 runs
    # =====================

    all_hof=[]
    all_results=[]


    for run_id in range(10):

        print(f"\nStarting run {run_id+1}/10")

        seed=42+run_id

        hof,results=run_ga(seed)

        df=pd.DataFrame(
            results,
            columns=[
                "generation",
                "LogicGates",
                "Genes",
                "Efficacy",
                "Safety",
                "seed_value"
            ]
        )

        all_results.append(df)

        all_hof.extend(hof)



    df_all=pd.concat(all_results)


    hof_data=[]


    for ind in all_hof:

        gA,gB,gT=ind

        hof_data.append([

            ind.seed_value,

            ind.generation,

            logic_gates[gT],

            [gene_names[gA],gene_names[gB]],

            ind.fitness.values[0],

            getattr(ind,"safety",None)

        ])


    df_hof=pd.DataFrame(

        hof_data,

        columns=[

            "seed_value",

            "generation",

            "LogicGates",

            "Genes",

            "Efficacy",

            "Safety"

        ]

    )


    df_hof=df_hof.sort_values(
        by="Efficacy",
        ascending=False
    ).reset_index(drop=True)



    print("\nTop 10 from Hall of Fame:")

    print(df_hof.head(10))


    return df_hof


# ======================================================
# MAIN ENTRY
# ======================================================

def run(
    tumor_matrix_path,
    mode="one",
    healthy_matrix_path=None,
    safety_threshold=0.9,

    pop_size=1000,
    Gmax=100,
    patience=50,
    n_cpus=10
):

    tumor_matrix,healthy_matrix,gene_names=load_matrices(
        tumor_matrix_path,
        healthy_matrix_path
    )

    if mode=="one":

        return run_one_gene(
            tumor_matrix,
            healthy_matrix,
            gene_names,
            safety_threshold
        )


    if mode=="two":

        return run_two_gene(
            tumor_matrix,
            healthy_matrix,
            gene_names,
            safety_threshold,
            pop_size,
            Gmax,
            patience,
            n_cpus
        )


    raise ValueError("mode must be 'one' or 'two'")