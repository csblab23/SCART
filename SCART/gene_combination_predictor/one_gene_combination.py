#!/usr/bin/env python
# coding: utf-8
"""
one_gene_combination.py
Module 4a — Single-gene CAR-T target evaluation

Evaluates every surface gene individually against tumour and healthy cells.
Healthy matrix source (in priority order):
  1. User-supplied HPA TSV/h5ad file  (hpa_path=)
  2. Auto-downloaded HPA single-cell read-count TSV from proteinatlas.org
  3. Legacy final_healthy.h5ad  (kept for backward compatibility)

HPA matrix is binarised: 0 = not expressed, 1 = any expression detected.
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
# HPA helpers
# ──────────────────────────────────────────────────────────────────────────

def _download_hpa(cache_path: str) -> str:
    """Download and unzip the HPA single-cell TSV, return path to TSV."""
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
    Parse HPA single-cell TSV → binary (cells × genes) numpy array.

    Expected HPA TSV columns: Gene, Cell type, Read count
    Pivots to (cell_type × gene) then binarises: 0 → 0, >0 → 1.

    Returns
    -------
    matrix : np.ndarray  shape (n_cell_types, n_genes), dtype int8
    genes  : list[str]
    cells  : list[str]   cell-type labels used as pseudo-cells
    """
    print(f"Reading HPA TSV: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")

    # Normalise column names — HPA occasionally changes capitalisation
    df.columns = df.columns.str.strip()
    col_map = {c.lower(): c for c in df.columns}
    gene_col  = col_map.get("gene name", col_map.get("gene", None))
    cell_col  = col_map.get("cell type", col_map.get("cell_type", None))
    count_col = col_map.get("read count", col_map.get("tpm", col_map.get("ntpm", None)))

    missing = [n for n, c in [("Gene", gene_col), ("Cell type", cell_col), ("Read count", count_col)] if c is None]
    if missing:
        raise ValueError(
            f"HPA TSV missing expected columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df[[gene_col, cell_col, count_col]].copy()
    df.columns = ["gene", "cell_type", "count"]
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)

    pivot = df.pivot_table(index="cell_type", columns="gene", values="count",
                           aggfunc="sum", fill_value=0)
    matrix = (pivot.values > 0).astype(np.int8)
    genes  = list(pivot.columns)
    cells  = list(pivot.index)

    print(f"HPA matrix built: {len(cells)} cell types × {len(genes)} genes")
    return matrix, genes, cells


def _load_healthy_matrix(hpa_path=None):
    """
    Load healthy/normal expression matrix.

    Priority:
      1. hpa_path provided and ends with .h5ad → read as AnnData
      2. hpa_path provided and ends with .tsv / .tsv.gz → parse as HPA TSV
      3. hpa_path = None → auto-download HPA TSV
      4. Fallback → legacy final_healthy.h5ad

    Returns
    -------
    matrix : np.ndarray  int8, shape (n_samples, n_genes)
    genes  : list[str]
    source : str         human-readable description
    """
def _load_h5ad_subset(h5ad_path: str, target_genes: list = None):
    """
    Fast, memory-safe h5ad loader.

    Uses h5py directly to read only the required gene columns from the X
    matrix — avoids loading the full matrix into RAM and avoids the slow
    backed-mode AnnData slice.

    For a 664k × 19k HPA file this reduces load time from minutes to seconds
    by reading only 651 column indices instead of the full matrix.

    Returns: matrix (int8 ndarray), genes (list[str])
    """
    import h5py
    import scipy.sparse as _sp

    if target_genes is None:
        adata = sc.read_h5ad(h5ad_path)
        X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
        return (X > 0).astype(np.int8), list(adata.var_names)

    with h5py.File(h5ad_path, "r") as f:

        # ── Read var names ────────────────────────────────────────────
        if "var" in f:
            var_grp = f["var"]
            # var index may be stored as _index or the first dataset
            if "_index" in var_grp:
                all_genes = [g.decode() if isinstance(g, bytes) else g
                             for g in var_grp["_index"][:]]
            else:
                key = list(var_grp.keys())[0]
                all_genes = [g.decode() if isinstance(g, bytes) else g
                             for g in var_grp[key][:]]
        else:
            raise ValueError(f"No 'var' group found in {h5ad_path}")

        gene_set    = set(all_genes)
        gene_index  = {g: i for i, g in enumerate(all_genes)}
        common      = [g for g in target_genes if g in gene_set]

        if len(common) == 0:
            raise ValueError(
                f"No overlap between target genes and h5ad var_names in {h5ad_path}.\n"
                "Check that both datasets use the same gene symbol convention (HGNC)."
            )

        col_indices = np.array([gene_index[g] for g in common], dtype=np.int32)
        print(f"  HPA h5ad: {len(all_genes)} genes total — "
              f"extracting {len(common)} overlapping genes directly via h5py.")

        # ── Read X — handle both dense and CSR sparse formats ────────
        x_grp = f["X"]

        if isinstance(x_grp, h5py.Dataset):
            # Dense array — read only needed columns
            X_sub = x_grp[:, col_indices]

        elif isinstance(x_grp, h5py.Group):
            # Sparse CSR format (most common for h5ad)
            encoding = x_grp.attrs.get("encoding-type", b"").decode() \
                if isinstance(x_grp.attrs.get("encoding-type", ""), bytes) \
                else x_grp.attrs.get("encoding-type", "")

            if "csr" in encoding or all(k in x_grp for k in ("data", "indices", "indptr")):
                data    = x_grp["data"][:]
                indices = x_grp["indices"][:]   # column indices
                indptr  = x_grp["indptr"][:]
                shape   = tuple(x_grp.attrs["shape"])
                full    = _sp.csr_matrix((data, indices, indptr), shape=shape)
                X_sub   = full[:, col_indices].toarray()

            elif "csc" in encoding or all(k in x_grp for k in ("data", "indices", "indptr")):
                data    = x_grp["data"][:]
                indices = x_grp["indices"][:]
                indptr  = x_grp["indptr"][:]
                shape   = tuple(x_grp.attrs["shape"])
                full    = _sp.csc_matrix((data, indices, indptr), shape=shape)
                X_sub   = full[:, col_indices].toarray()

            else:
                # Unknown sparse format — fall back to scanpy backed slice
                logger.warning(
                    "Unknown X encoding in h5ad — falling back to scanpy backed mode."
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
    # Option 1 — user supplied h5ad (memory-safe)
    if hpa_path and hpa_path.endswith(".h5ad"):
        if not os.path.exists(hpa_path):
            raise FileNotFoundError(f"Provided HPA h5ad not found: {hpa_path}")
        print(f"Loading user-supplied healthy h5ad: {hpa_path}")
        matrix, genes = _load_h5ad_subset(hpa_path, target_genes)
        return matrix, genes, f"user h5ad: {hpa_path}"

    # Option 2 — user supplied TSV
    if hpa_path and (hpa_path.endswith(".tsv") or hpa_path.endswith(".tsv.gz")):
        if not os.path.exists(hpa_path):
            raise FileNotFoundError(f"Provided HPA TSV not found: {hpa_path}")
        matrix, genes, _ = _hpa_tsv_to_binary_matrix(hpa_path)
        return matrix, genes, f"user TSV: {hpa_path}"

    # Option 3 — auto-download HPA
    if hpa_path is None:
        print("No HPA file provided — downloading from proteinatlas.org ...")
        tsv = _download_hpa(HPA_CACHE)
        matrix, genes, _ = _hpa_tsv_to_binary_matrix(tsv)
        return matrix, genes, "auto-downloaded HPA"

    # Option 4 — legacy fallback: look for final_healthy.h5ad near cwd
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


# ──────────────────────────────────────────────────────────────────────────
# Single-gene evaluator
# ──────────────────────────────────────────────────────────────────────────

def evaluate_single_gene(gene_idx, tumor_matrix, healthy_matrix):
    tumor_expr   = tumor_matrix[:, gene_idx]
    healthy_expr = healthy_matrix[:, gene_idx]
    efficacy = np.sum(tumor_expr)   / len(tumor_expr)
    safety   = np.sum(healthy_expr == 0) / len(healthy_expr)
    return efficacy, safety


# ──────────────────────────────────────────────────────────────────────────
# Public entry-point
# ──────────────────────────────────────────────────────────────────────────

def run(
    safety_threshold: float = 0.9,
    hpa_path: str = None,
    tumor_path: str = None,
):
    """
    Run single-gene CAR-T target evaluation.

    Parameters
    ----------
    safety_threshold : float
        Minimum fraction of healthy cells that must NOT express the gene.
        Range 0–1. Default 0.9 (90% of healthy cells must be negative).
        Higher → safer but fewer candidates.
        Lower  → more candidates but higher off-tumour risk.
    hpa_path : str or None
        Path to a user-supplied HPA file (.h5ad or .tsv/.tsv.gz).
        If None, HPA data is auto-downloaded from proteinatlas.org.
    tumor_path : str or None
        Path to the tumour h5ad (Module 3 output).
        If None, auto-detected from the SCART package directory.

    Returns
    -------
    pd.DataFrame  columns: Gene, Efficacy, Safety, ObjectiveScore
    """
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
    common_genes  = sorted(set(tumor_genes) & set(healthy_genes))

    if len(common_genes) == 0:
        raise ValueError(
            "No common genes between tumour and healthy matrices.\n"
            "Check that both use the same gene symbol convention (HGNC)."
        )
    print(f"Common genes: {len(common_genes)}")

    # Tumour subset
    adata_tumor = adata_tumor[:, common_genes].copy()
    X_tumor = adata_tumor.X.toarray() if not isinstance(adata_tumor.X, np.ndarray) else adata_tumor.X
    tumor_matrix = (X_tumor > 0).astype(np.int8)

    # Healthy subset — reindex columns to match common_genes order
    hg_idx = {g: i for i, g in enumerate(healthy_genes)}
    col_idx = np.array([hg_idx[g] for g in common_genes])
    healthy_matrix = healthy_matrix_full[:, col_idx]

    n_genes    = len(common_genes)
    gene_names = common_genes

    print(f"Tumour matrix  : {tumor_matrix.shape[0]} cells × {n_genes} genes")
    print(f"Healthy matrix : {healthy_matrix.shape[0]} samples × {n_genes} genes")
    print("Starting single-gene analysis...")

    # ── Evaluate every gene ────────────────────────────────────────────
    results = []
    tick    = max(1, n_genes // 100)

    for idx in range(n_genes):
        efficacy, safety = evaluate_single_gene(idx, tumor_matrix, healthy_matrix)
        objective_score  = efficacy if safety >= safety_threshold else 0
        results.append([gene_names[idx], efficacy, safety, objective_score])
        if (idx + 1) % tick == 0:
            print(f"\rProgress: {(idx+1)/n_genes*100:.1f}% completed", end="")

    print("\nAnalysis completed!")

    # ── Build output DataFrame ─────────────────────────────────────────
    df_results = pd.DataFrame(
        results, columns=["Gene", "Efficacy", "Safety", "ObjectiveScore"]
    )
    output_file = os.path.join(os.getcwd(), "single_gene_results.csv")
    df_results[["Gene", "Efficacy", "Safety"]].to_csv(
        output_file, index=False, header=["gene", "efficacy", "safety"]
    )
    print(f"Results saved to: {output_file}")

    # ── Print top 10 ───────────────────────────────────────────────────
    df_top = (
        df_results[df_results["Safety"] >= safety_threshold]
        .sort_values(by="Efficacy", ascending=False)
        .head(10)
    )
    print(f"\nTop 10 single-gene candidates (safety >= {safety_threshold}):")
    print(df_top[["Gene", "Efficacy", "Safety"]].to_string(index=False))

    return df_results
