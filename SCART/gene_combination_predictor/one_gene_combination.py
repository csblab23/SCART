#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import scanpy as sc
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))

TUMOR_PATH = os.path.join(BASE_DIR, "preprocessed_input", "final_tumor.h5ad")
HEALTHY_PATH = os.path.join(BASE_DIR, "preprocessed_input", "final_healthy.h5ad")

adata_tumor = sc.read_h5ad(TUMOR_PATH)
adata_healthy = sc.read_h5ad(HEALTHY_PATH)

common_genes = adata_tumor.var_names.intersection(adata_healthy.var_names)
adata_tumor = adata_tumor[:, common_genes].copy()
adata_healthy = adata_healthy[:, common_genes].copy()

tumor_matrix = adata_tumor.X.toarray() if not isinstance(adata_tumor.X, np.ndarray) else adata_tumor.X
healthy_matrix = adata_healthy.X.toarray() if not isinstance(adata_healthy.X, np.ndarray) else adata_healthy.X

tumor_matrix = (tumor_matrix > 0).astype(int)
healthy_matrix = (healthy_matrix > 0).astype(int)

gene_names = common_genes.tolist()
n_genes = len(gene_names)

def evaluate_single_gene(gene_idx):
    tumor_expr = tumor_matrix[:, gene_idx]
    healthy_expr = healthy_matrix[:, gene_idx]

    efficacy = np.sum(tumor_expr) / len(tumor_expr)
    safety = np.sum(healthy_expr == 0) / len(healthy_expr)
    return efficacy, safety

def run(safety_threshold=0.9, output_file="single_gene_results.csv"):
    results = []
    print("Starting single-gene analysis...")

    for idx, gene_idx in enumerate(range(n_genes), 1):
        efficacy, safety = evaluate_single_gene(gene_idx)
        objective_score = efficacy if safety >= safety_threshold else 0
        results.append([gene_names[gene_idx], efficacy, safety, objective_score])
        if idx % max(1, n_genes//100) == 0:
            print(f"\rProgress: {idx/n_genes*100:.1f}% completed", end="")

    print("\nAnalysis completed!")

    df_results = pd.DataFrame(results, columns=["Gene", "Efficacy", "Safety", "ObjectiveScore"])
    df_results[["Gene","Efficacy","Safety"]].to_csv(output_file, index=False, header=["gene","efficacy","safety"])

    # Print top 10
    df_top = df_results[df_results["Safety"] >= safety_threshold].sort_values(by="Efficacy", ascending=False).head(10)
    print("\nTop 10 single-gene combinations (Efficacy >= safety threshold):")
    print(df_top[["Gene","Efficacy","Safety"]])

    return df_results