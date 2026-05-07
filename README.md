# SCART — Single-Cell CAR-T Target Discovery

**SCART** is an end-to-end computational pipeline for identifying tumour-specific surface protein targets for CAR-T cell therapy from single-cell RNA-seq data. Starting from raw GEO accession IDs or user-provided h5ad files, SCART automates cell-type annotation, malignant cell identification, surfaceome differential expression, and logic-gate gene combination scoring to rank candidate CAR-T targets by efficacy and safety.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Module 1 — Data Acquisition](#module-1--data-acquisition)
  - [Module 2 — Cell-Type Annotation](#module-2--cell-type-annotation)
  - [Module 3 — Preprocessing & Malignancy Detection](#module-3--preprocessing--malignancy-detection)
  - [Module 4a — Single-Gene Scoring](#module-4a--single-gene-scoring)
  - [Module 4b — Two-Gene Logic-Gate Scoring](#module-4b--two-gene-logic-gate-scoring)
- [Module Reference](#module-reference)
  - [Module 1 — Data Acquisition](#module-1--data-acquisition-geo_fetcherpy)
  - [Module 2 — Cell-Type Annotation](#module-2--cell-type-annotation-popv_annotationpy)
  - [Module 3 — Preprocessing & Malignancy Detection](#module-3--preprocessing--malignancy-detection-preprocessingpy)
  - [Module 4a — Single-Gene Scoring](#module-4a--single-gene-scoring-one_gene_combinationpy)
  - [Module 4b — Two-Gene Logic-Gate Scoring](#module-4b--two-gene-logic-gate-scoring-two_gene_combinationpy)
- [Manual Annotation — Skip PopV with Your Own Labels](#manual-annotation--skip-popv-with-your-own-labels)
- [Output Files](#output-files)
- [Parameter Reference](#parameter-reference)

---

## Overview

CAR-T therapy requires surface targets that are highly expressed on tumour cells and absent on healthy tissue. SCART automates this discovery by:

1. Downloading and parsing scRNA-seq datasets from GEO
2. Annotating cell types with PopV (multi-method consensus) — or using your own pre-existing annotations
3. Identifying malignant epithelial cells via scMalignantFinder and SCEVAN
4. Computing differentially expressed surfaceome genes (tumour vs stromal/immune)
5. Scoring every candidate gene — or gene pair with a logic gate — for efficacy (tumour coverage) and safety (healthy tissue sparing)

---

## Pipeline Architecture

To be added

---

## Installation

### 1. Create and activate the conda environment

```bash
conda create -n scart_env python=3.10 -y
conda activate scart_env
```

> **Windows only** — run these two lines before the next step:
> ```bash
> pip install "orbax-checkpoint<0.5" "flax<0.8"
> pip install scvi-tools==1.1.6.post2
> ```

### 2. Install SCART

```bash
pip install git+https://github.com/csblab23/SCART.git
```

### 3. Install R and SCEVAN dependencies

These are required for SCEVAN (Module 3). Run inside your activated conda environment:

```bash
conda install -c conda-forge -c bioconda \
  r-base r-devtools r-remotes r-ggplot2 r-data.table \
  r-igraph r-gdtools r-ragg r-dplyr \
  cairo freetype fontconfig harfbuzz fribidi \
  libpng libtiff libjpeg libwebp
```

```bash
conda install -c bioconda -c conda-forge \
  r-fgsea \
  bioconductor-genomeinfoDb \
  bioconductor-genomicranges \
  bioconductor-summarizedexperiment \
  bioconductor-singlecellexperiment \
  bioconductor-scuttle \
  bioconductor-scran \
  r-rcurl -y
```

Then inside R:

```r
library(devtools)
install_github("miccec/yaGST")
install_github("AntonioDeFalco/SCEVAN")

# Verify installation
library(SCEVAN)
```

### 4. Set up Jupyter Notebook

```bash
pip install notebook ipykernel
python -m ipykernel install --user --name=scart_env --display-name "Python (scart_env)"
jupyter notebook
```

---

## Quick Start

### Module 1 — Data Acquisition

```python
from SCART.geo_fetcher import SampleAnnotator

# ── Option 1: Single GEO ID, QC disabled (default) ───────────────────────
# QC step in Module 3 will be skipped entirely
annotator = SampleAnnotator("GSE158937")

# ── Option 2: Single GEO ID with both QC thresholds ──────────────────────
annotator = SampleAnnotator("GSE158937", min_genes=200, max_mt=40)

# ── Option 3: Single GEO ID with gene-count filter only ──────────────────
annotator = SampleAnnotator("GSE158937", min_genes=300)

# ── Option 4: Single GEO ID with MT filter only ───────────────────────────
annotator = SampleAnnotator("GSE158937", max_mt=25)

# ── Option 5: Multiple GEO IDs → saves combined_tumor.h5ad ───────────────
annotator = SampleAnnotator("GSE158937", "GSE184880", "GSE217517",
                             min_genes=200, max_mt=40)

# ── Option 6: User-supplied h5ad instead of GEO download ─────────────────
annotator = SampleAnnotator("/path/to/my_data.h5ad",
                             min_genes=200, max_mt=40)

# ── Option 7: Mixed — GEO ID + user h5ad combined ────────────────────────
annotator = SampleAnnotator("GSE158937", "/path/to/extra_data.h5ad",
                             min_genes=200, max_mt=40)

# ── Option 8: User h5ad WITH manual cell-type annotations ────────────────
# Skips Module 2 (PopV) entirely — go straight to Module 3 after this
annotator = SampleAnnotator(
    "my_data.h5ad",
    manual_annotation_col="cell_type",   # name of your obs column
    min_genes=200,
    max_mt=40,
)

# ── Run (same call for all options above) ────────────────────────────────
normal, tumor, unspecified, annotation_info, query_h5ad, cancer_type, results = annotator.run()

# ── What you get ──────────────────────────────────────────────────────────
# normal          → list of normal sample IDs
# tumor           → list of tumour sample IDs
# unspecified     → list of unclassified sample IDs
# annotation_info → dict mapping sample ID → "tumor" / "normal" / "unspecified"
# query_h5ad      → path to the saved tumour h5ad (input for Module 2)
# cancer_type     → detected cancer type string (e.g. "ovary_cancer")
# results         → full per-GSE result dictionary
```

> **Output files**
> - Single GEO run → `GSE*_tumor.h5ad`
> - Multiple inputs → `combined_tumor.h5ad`
> - User h5ad input → `input_tumor.h5ad`

> **Manual annotation note**
> When `manual_annotation_col` is provided, Module 1 stores `adata.uns['skip_popv'] = True`
> and copies your column into `popv_majority_vote_prediction`. Skip Module 2 entirely
> and proceed directly to Module 3. See [Manual Annotation](#manual-annotation--skip-popv-with-your-own-labels) for full details.

---

### Module 2 — Cell-Type Annotation

> **Skip this module** if you used `manual_annotation_col` in Module 1
> (`adata.uns['skip_popv'] == True`). Go directly to Module 3.

```python
from SCART import popv_annotation

# ── Option 1: User-supplied tissue reference (recommended) ───────────────
# Module 1 prints which Tabula Sapiens file to download for your cancer type
adata = popv_annotation.auto_run_popv(
    input_type     = "raw",
    nsamples       = 300,
    user_reference = "/path/to/Ovary_TSP1_30_version2d_10X_smartseq_scvi.h5ad"
)

# ── Option 2: Auto-download reference from Figshare ──────────────────────
# Downloads the matching Tabula Sapiens tissue file automatically
# (requires internet; large file ~2–5 GB)
adata = popv_annotation.auto_run_popv(
    input_type = "raw",
    nsamples   = 300,
)

# ── Option 3: Pre-log-normalised input (runs CELLTYPIST only) ────────────
# Use when your h5ad contains log-normalised counts, not raw integers
adata = popv_annotation.auto_run_popv(
    input_type     = "log1p",
    nsamples       = 300,
    user_reference = "/path/to/reference.h5ad"
)

# ── Option 4: Custom output directory ────────────────────────────────────
adata = popv_annotation.auto_run_popv(
    input_type     = "raw",
    nsamples       = 300,
    output_dir     = "/path/to/my_popv_results/",
    user_reference = "/path/to/reference.h5ad"
)

# ── Option 5: Keep Tabula Sapiens metadata columns in output ─────────────
# By default these are removed; set False to retain them
adata = popv_annotation.auto_run_popv(
    input_type             = "raw",
    nsamples               = 300,
    user_reference         = "/path/to/reference.h5ad",
    drop_reference_columns = False
)

# ── What you get ──────────────────────────────────────────────────────────
# adata → AnnData saved to popv_results/final_popv_annotated.h5ad
# Key column: adata.obs['popv_majority_vote_prediction']
```

> **Output files**
> - `popv_results/final_popv_annotated.h5ad`

---

### Module 3 — Preprocessing & Malignancy Detection

```python
from SCART import preprocessing

# ── Option 1: Standard run with SCEVAN + scMalignantFinder ───────────────
# Both tools used; a cell must be called malignant by BOTH (intersection)
adata_preprocessed = preprocessing.run_preprocessing_pipeline(
    reference_h5ad     = "/path/to/tissue_reference.h5ad",
    log2fc_threshold   = 2.0,
    pval_adj_threshold = 0.05,
    malignant_strategy = "intersection",
)

# ── Option 2: scMalignantFinder only (no SCEVAN) ─────────────────────────
# Use when no tissue reference is available, or to skip the R/SCEVAN step
adata_preprocessed = preprocessing.run_preprocessing_pipeline(
    malignant_strategy = "scMalignant",
    log2fc_threshold   = 2.0,
    pval_adj_threshold = 0.05,
)

# ── Option 3: SCEVAN only ─────────────────────────────────────────────────
# Use when you trust CNA profiling more than the ML classifier
adata_preprocessed = preprocessing.run_preprocessing_pipeline(
    reference_h5ad     = "/path/to/tissue_reference.h5ad",
    malignant_strategy = "scevan",
    log2fc_threshold   = 2.0,
    pval_adj_threshold = 0.05,
)

# ── Option 4: Relaxed DEG filters (if 0 DEGs with default settings) ───────
adata_preprocessed = preprocessing.run_preprocessing_pipeline(
    reference_h5ad     = "/path/to/tissue_reference.h5ad",
    malignant_strategy = "intersection",
    log2fc_threshold   = 0.5,    # lower fold-change cutoff
    pval_adj_threshold = 0.10,   # less strict p-value
)

# ── Option 5: Tune SCEVAN reference cell count ────────────────────────────
adata_preprocessed = preprocessing.run_preprocessing_pipeline(
    reference_h5ad       = "/path/to/tissue_reference.h5ad",
    malignant_strategy   = "intersection",
    scevan_ref_max_cells = 200,   # default 100; increase for more stable CNV calls
)

# ── Option 6: Explicit file paths (if auto-detection fails) ───────────────
adata_preprocessed = preprocessing.run_preprocessing_pipeline(
    popv_path      = "/path/to/final_popv_annotated.h5ad",
    tumor_h5ad     = "/path/to/GSE158937_tumor.h5ad",
    reference_h5ad = "/path/to/tissue_reference.h5ad",
    save_dir       = "/path/to/my_output_dir/",
)

# ── Option 7: Manual annotation path (after skipping PopV) ───────────────
# input_tumor.h5ad already has popv_majority_vote_prediction set by Module 1
adata_preprocessed = preprocessing.run_preprocessing_pipeline(
    adata              = sc.read_h5ad("input_tumor.h5ad"),
    reference_h5ad     = "/path/to/tissue_reference.h5ad",
    malignant_strategy = "intersection",
    log2fc_threshold   = 2.0,
    pval_adj_threshold = 0.05,
)

# ── What you get ──────────────────────────────────────────────────────────
# adata_preprocessed → malignant epithelial cells only, binarised expression
# adata_preprocessed.obs columns:
#   scMalignantFinder_prediction, scevan_prediction, final_malignant
# adata_preprocessed.uns keys:
#   filtered_deg, all_deg, deg_params, scevan_results, qc_params
print(adata_preprocessed.uns["filtered_deg"].head(10))
```

> **Output files**
> - `preprocessing_results/final_tumor.h5ad`

---

### Module 4a — Single-Gene Scoring

```python
from SCART.gene_combination_predictor import one_gene_combination

# ── Option 1: User-supplied HPA h5ad, default safety ─────────────────────
df = one_gene_combination.run(
    hpa_path         = "/path/to/HPA_updated.h5ad",
    safety_threshold = 0.9,
)

# ── Option 2: Stricter safety (fewer but safer candidates) ───────────────
df = one_gene_combination.run(
    hpa_path         = "/path/to/HPA_updated.h5ad",
    safety_threshold = 0.95,
)

# ── Option 3: Relaxed safety (more candidates) ───────────────────────────
df = one_gene_combination.run(
    hpa_path         = "/path/to/HPA_updated.h5ad",
    safety_threshold = 0.8,
)

# ── Option 4: User-supplied HPA as TSV ───────────────────────────────────
df = one_gene_combination.run(
    hpa_path         = "/path/to/rna_single_cell_read_count.tsv",
    safety_threshold = 0.9,
)

# ── Option 5: Auto-download HPA from proteinatlas.org ────────────────────
# Downloads once and caches in <cwd>/hpa_cache/
df = one_gene_combination.run(
    safety_threshold = 0.9,
)

# ── Option 6: Explicit tumour path (if auto-detection fails) ─────────────
df = one_gene_combination.run(
    hpa_path         = "/path/to/HPA_updated.h5ad",
    tumor_path       = "/path/to/preprocessing_results/final_tumor.h5ad",
    safety_threshold = 0.9,
)

# ── Inspect results ───────────────────────────────────────────────────────
# df columns: Gene, Efficacy, Safety, ObjectiveScore
print(df.head(20))

# Top candidates passing safety threshold
df_filtered = df[df["Safety"] >= 0.9].sort_values("Efficacy", ascending=False)
print(df_filtered.head(10))
```

> **Output files**
> - `single_gene_results.csv` — columns: `gene`, `efficacy`, `safety`

---

### Module 4b — Two-Gene Logic-Gate Scoring

```python
from SCART.gene_combination_predictor import two_gene_combination

# ── Option 1: Minimal — user HPA + default GA settings ───────────────────
df_hof, df_all = two_gene_combination.run(
    hpa_path         = "/path/to/HPA_updated.h5ad",
    safety_threshold = 0.9,
)

# ── Option 2: Quick test run (small pop, few generations) ────────────────
# Use to verify the pipeline runs before committing to a full search
df_hof, df_all = two_gene_combination.run(
    hpa_path         = "/path/to/HPA_updated.h5ad",
    safety_threshold = 0.9,
    pop_size         = 200,   # small population
    Gmax             = 20,    # few generations
    patience         = 10,
    n_cpus           = 4,
    n_runs           = 2,
)

# ── Option 3: Standard workstation run ───────────────────────────────────
df_hof, df_all = two_gene_combination.run(
    hpa_path         = "/path/to/HPA_updated.h5ad",
    safety_threshold = 0.9,
    pop_size         = 1000,
    Gmax             = 100,
    patience         = 50,
    n_cpus           = 8,     # set to your available CPU cores
    n_runs           = 10,
)

# ── Option 4: HPC / server run ───────────────────────────────────────────
df_hof, df_all = two_gene_combination.run(
    hpa_path         = "/path/to/HPA_updated.h5ad",
    safety_threshold = 0.9,
    pop_size         = 2000,
    Gmax             = 200,
    patience         = 100,
    n_cpus           = 40,
    n_runs           = 10,
)

# ── Option 5: Stricter safety threshold ──────────────────────────────────
df_hof, df_all = two_gene_combination.run(
    hpa_path         = "/path/to/HPA_updated.h5ad",
    safety_threshold = 0.95,  # 95% of healthy cell types must be negative
    pop_size         = 1000,
    Gmax             = 100,
    patience         = 50,
    n_cpus           = 8,
    n_runs           = 10,
)

# ── Option 6: Explicit tumour path (if auto-detection fails) ─────────────
df_hof, df_all = two_gene_combination.run(
    hpa_path         = "/path/to/HPA_updated.h5ad",
    tumor_path       = "/path/to/preprocessing_results/final_tumor.h5ad",
    safety_threshold = 0.9,
    n_cpus           = 8,
)

# ── Option 7: User-supplied HPA as TSV ───────────────────────────────────
df_hof, df_all = two_gene_combination.run(
    hpa_path         = "/path/to/rna_single_cell_read_count.tsv",
    safety_threshold = 0.9,
    n_cpus           = 8,
)

# ── Option 8: Auto-download HPA from proteinatlas.org ────────────────────
df_hof, df_all = two_gene_combination.run(
    safety_threshold = 0.9,
    n_cpus           = 8,
)

# ── Option 9: Use all available CPU cores automatically ──────────────────
import multiprocessing
df_hof, df_all = two_gene_combination.run(
    hpa_path         = "/path/to/HPA_updated.h5ad",
    safety_threshold = 0.9,
    n_cpus           = multiprocessing.cpu_count(),
)

# ── Inspect results ───────────────────────────────────────────────────────
# df_hof — Hall of Fame: best unique gene pairs across all runs
# Columns: seed_value, generation, LogicGates, Genes, Efficacy, Safety
print(df_hof.head(10))

# df_all — complete record of every evaluated pair across all runs
print(df_all.shape)

# Filter Hall of Fame by logic gate type
and_pairs = df_hof[df_hof["LogicGates"] == "A & B"]    # both genes expressed
or_pairs  = df_hof[df_hof["LogicGates"] == "A | B"]    # either gene expressed
not_pairs = df_hof[df_hof["LogicGates"] == "A & !B"]   # A on, B off

# Top AND-gate candidates
print(and_pairs.head(10))

# All pairs above a safety threshold
safe_pairs = df_hof[df_hof["Safety"] >= 0.95].sort_values("Efficacy", ascending=False)
print(safe_pairs.head(10))
```

> **Output files**
> - `two_gene_hof.csv` — Hall of Fame: best unique pairs. Columns: `seed_value`, `generation`, `LogicGates`, `Genes`, `Efficacy`, `Safety`
> - `two_gene_complete.csv` — all evaluated pairs across every generation and run

---

## Module Reference

### Module 1 — Data Acquisition (`geo_fetcher.py`)

Downloads GEO datasets or accepts existing h5ad files, classifies samples as tumour/normal/unspecified, and writes a tumour h5ad for downstream modules.

```python
from SCART.geo_fetcher import SampleAnnotator

annotator = SampleAnnotator(*inputs, min_genes=None, max_mt=None,
                             manual_annotation_col=None)
normal, tumor, unspecified, annotation_info, query_h5ad, cancer_type, results = annotator.run()
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `*inputs` | str | — | One or more GEO accession IDs (e.g. `"GSE158937"`) or paths to `.h5ad` files |
| `min_genes` | int or None | `None` | Minimum genes per cell for QC in Module 3. `None` = QC skipped |
| `max_mt` | float or None | `None` | Maximum mitochondrial % per cell. `None` = QC skipped |
| `manual_annotation_col` | str or None | `None` | Name of the obs column in your h5ad that contains cell-type labels. When set, Module 2 (PopV) is skipped. Only applies to h5ad inputs — ignored for GEO IDs |

**Usage examples**

```python
# Single GEO ID, no QC
annotator = SampleAnnotator("GSE158937")

# Single GEO ID with QC
annotator = SampleAnnotator("GSE158937", min_genes=200, max_mt=40)

# Multiple GEO IDs → saves combined_tumor.h5ad
annotator = SampleAnnotator("GSE158937", "GSE184880", min_genes=200, max_mt=40)

# User-supplied h5ad
annotator = SampleAnnotator("/path/to/my_data.h5ad", min_genes=200, max_mt=40)

# User h5ad with manual annotations — skips PopV
annotator = SampleAnnotator("/path/to/my_data.h5ad",
                             manual_annotation_col="cell_type",
                             min_genes=200, max_mt=40)

# Mixed GEO + h5ad
annotator = SampleAnnotator("GSE158937", "/path/to/extra.h5ad")
```

**Output files**

| File | Description |
|---|---|
| `GSE*_tumor.h5ad` | Single GEO run — tumour cells only |
| `combined_tumor.h5ad` | Multiple inputs merged |
| `input_tumor.h5ad` | User-supplied h5ad input |

**QC parameter flow**

QC thresholds set here are stored in `adata.uns['qc_params']` and automatically read by Module 3. If neither `min_genes` nor `max_mt` is provided, the QC step in Module 3 is skipped entirely — no defaults are silently applied.

**Manual annotation flow**

When `manual_annotation_col` is set on an h5ad input, Module 1 copies that column into `adata.obs['popv_majority_vote_prediction']` and sets `adata.uns['skip_popv'] = True`. Module 3 reads `popv_majority_vote_prediction` identically regardless of whether it came from PopV or manual annotation. See [Manual Annotation](#manual-annotation--skip-popv-with-your-own-labels) for label requirements.

---

### Module 2 — Cell-Type Annotation (`popv_annotation.py`)

Annotates cell types using PopV, a consensus framework that runs multiple methods (CELLTYPIST, KNN-BBKNN, KNN-SCVI, KNN-HARMONY, ONCLASS, SCANVI, Support Vector, Random Forest) and reports a majority-vote prediction.

> **This module can be skipped** when `manual_annotation_col` was provided in Module 1.
> The output h5ad will contain `adata.uns['skip_popv'] = True` — if you call
> `auto_run_popv()` on such a file it will exit immediately with a clear message.

```python
from SCART import popv_annotation

# Automatic mode — finds tumour h5ad from Module 1 automatically
adata = popv_annotation.auto_run_popv(
    input_type            = "raw",
    nsamples              = 300,
    output_dir            = "popv_results",
    user_reference        = "/path/to/reference.h5ad",
    drop_reference_columns = True,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_type` | str | `"raw"` | `"raw"` for raw counts; `"log1p"` for pre-normalised data (runs CELLTYPIST only) |
| `nsamples` | int | `300` | Cells sampled per cell-type label during reference subsampling |
| `output_dir` | str | `"popv_results"` | Directory where `final_popv_annotated.h5ad` is saved |
| `user_reference` | str or None | `None` | Path to Tabula Sapiens tissue h5ad. If `None`, auto-downloaded from Figshare |
| `drop_reference_columns` | bool | `True` | Remove Tabula Sapiens metadata columns from output |

**Reference files**

Tabula Sapiens tissue references are available at: https://doi.org/10.6084/m9.figshare.27921984

Module 1 prints the recommended reference file for the detected cancer type at the end of its run.

**Output files**

| File | Description |
|---|---|
| `popv_results/final_popv_annotated.h5ad` | Full dataset with `popv_majority_vote_prediction` column and `layers['counts']` for Module 3 |

---

### Module 3 — Preprocessing & Malignancy Detection (`preprocessing.py`)

Identifies malignant epithelial cells using scMalignantFinder and SCEVAN, performs surfaceome differential expression analysis, and outputs a binarised tumour matrix for Module 4.

Epithelial cells are selected by substring-matching `"epithelial cell"` (case-insensitive) in `popv_majority_vote_prediction` — this captures all variants such as `"ovarian surface epithelial cell"`, `"glandular epithelial cell"`, `"lung epithelial cell"`, etc.

When the input h5ad has `adata.uns['skip_popv'] = True` (manual annotation path), Module 3 detects this automatically, validates the label column, and proceeds identically — no code changes needed.

```python
from SCART import preprocessing

adata_preprocessed = preprocessing.run_preprocessing_pipeline(
    reference_h5ad     = "/path/to/tissue_reference.h5ad",
    log2fc_threshold   = 2.0,
    pval_adj_threshold = 0.05,
    malignant_strategy = "intersection",
    scevan_ref_max_cells = 100,
    scevan_sample_name   = "SCEVAN_run",
    scevan_organism      = "human",
    scevan_par_cores     = 1,
    scevan_subclones     = False,
    scevan_batch_size    = 3000,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adata` | AnnData or None | `None` | Module 2 output. Auto-loaded from `popv_results/` if None |
| `popv_path` | str or None | `None` | Explicit path to PopV h5ad |
| `log2fc_threshold` | float | `1.0` | DEG log2 fold-change cutoff |
| `pval_adj_threshold` | float | `0.05` | DEG BH-adjusted p-value cutoff |
| `reference_h5ad` | str or None | `None` | Tabula Sapiens h5ad for SCEVAN normal reference. SCEVAN skipped if None |
| `tumor_h5ad` | str or None | `None` | Module 1 h5ad for scMalignantFinder full-gene recovery. Auto-detected if None |
| `save_dir` | str or None | `None` | Output directory. Default: `<cwd>/preprocessing_results/` |
| `malignant_strategy` | str | `"intersection"` | `"intersection"` (both tools must agree), `"scMalignant"`, or `"scevan"` |
| `scevan_ref_max_cells` | int | `100` | Maximum normal reference cells subsampled for SCEVAN |
| `scevan_sample_name` | str | `"SCEVAN_run"` | Prefix for SCEVAN output files |
| `scevan_organism` | str | `"human"` | `"human"` or `"mouse"` |
| `scevan_par_cores` | int | `1` | CPU cores per SCEVAN batch |
| `scevan_subclones` | bool | `False` | Whether SCEVAN infers tumour subclones |
| `scevan_batch_size` | int | `3000` | Query cells per SCEVAN batch |

**QC thresholds** are not parameters here — they are read automatically from `adata.uns['qc_params']` set in Module 1.

**Pipeline steps**

1. Load full PopV-annotated dataset (or manual-annotation h5ad)
2. Detect manual annotation mode (`skip_popv`) — validate labels if present
3. Read QC thresholds from `adata.uns['qc_params']` (skipped if absent)
4. Extract epithelial cells via substring match on `"epithelial cell"` → apply QC filters if set
5. Run **scMalignantFinder** (machine learning classifier) on full gene space
6. Run **SCEVAN** (copy number alteration profiling) against normal reference cells selected by substring match on `"epithelial cell"` in `cell_ontology_class`
7. Combine predictions with chosen strategy → `final_malignant` column
8. Keep only malignant epithelial cells; extract non-epithelial cells as DEG reference
9. Filter both groups to surfaceome genes (GESP database)
10. Wilcoxon DEG: malignant epithelial vs non-epithelial rest
11. Binarise expression matrix (0/1), store DEG results, save

**Output files**

| File | Description |
|---|---|
| `preprocessing_results/final_tumor.h5ad` | Malignant epithelial cells, surfaceome-filtered, binarised. Contains `obs['final_malignant']`, DEG in `uns['filtered_deg']`, SCEVAN scores in `uns['scevan_results']` |

---

### Module 4a — Single-Gene Scoring (`one_gene_combination.py`)

Evaluates every surfaceome gene individually for CAR-T suitability by computing efficacy (fraction of tumour cells expressing the gene) and safety (fraction of healthy cells NOT expressing the gene).

```python
from SCART.gene_combination_predictor import one_gene_combination

df = one_gene_combination.run(
    hpa_path         = "/path/to/HPA_healthy_reference.h5ad",
    safety_threshold = 0.9,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `safety_threshold` | float | `0.9` | Minimum fraction of healthy cells that must NOT express the gene. Range 0–1 |
| `hpa_path` | str or None | `None` | Path to healthy reference (`.h5ad` or `.tsv`/`.tsv.gz`). If `None`, auto-downloaded from proteinatlas.org |
| `tumor_path` | str or None | `None` | Path to Module 3 output. Auto-detected from `preprocessing_results/` if None |

**HPA healthy reference options**

```python
# Option 1: User-supplied HPA h5ad (recommended — fastest)
df = one_gene_combination.run(hpa_path="/path/to/HPA.h5ad")

# Option 2: User-supplied HPA TSV
df = one_gene_combination.run(hpa_path="/path/to/rna_single_cell_read_count.tsv")

# Option 3: Auto-download from proteinatlas.org (cached after first run)
df = one_gene_combination.run()
```

Large HPA h5ad files (e.g. 664k × 19k genes) are loaded memory-safely — only the genes overlapping with the tumour matrix are read into RAM.

**Output files**

| File | Description |
|---|---|
| `single_gene_results.csv` | All genes scored. Columns: `gene`, `efficacy`, `safety` |

**Return value:** `pd.DataFrame` with columns `Gene`, `Efficacy`, `Safety`, `ObjectiveScore`

---

### Module 4b — Two-Gene Logic-Gate Scoring (`two_gene_combination.py`)

Uses a Genetic Algorithm (DEAP) to search over all `(geneA, geneB, logic_gate)` combinations and find pairs that maximise tumour killing while sparing healthy tissue.

**Logic gates evaluated:**

| Gate | Meaning | Use case |
|---|---|---|
| `A & B` | Both genes expressed | High specificity, moderate coverage |
| `A \| B` | Either gene expressed | High coverage, moderate specificity |
| `A & !B` | A expressed, B NOT expressed | Tumour-specific NOT gates |

```python
from SCART.gene_combination_predictor import two_gene_combination

df_hof, df_all = two_gene_combination.run(
    hpa_path         = "/path/to/HPA_healthy_reference.h5ad",
    safety_threshold = 0.9,
    pop_size         = 1000,
    Gmax             = 100,
    patience         = 50,
    n_cpus           = 8,
    n_runs           = 10,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hpa_path` | str or None | `None` | Healthy reference path (same options as Module 4a) |
| `tumor_path` | str or None | `None` | Module 3 output. Auto-detected if None |
| `safety_threshold` | float | `0.9` | Minimum healthy-cell safety. Range 0–1 |
| `pop_size` | int | `1000` | GA population size per generation. Larger = better search, slower runtime |
| `Gmax` | int | `100` | Maximum generations. More = more thorough search |
| `Ggap` | int | `10` | Random immigrant injection interval (prevents premature convergence) |
| `Rrep` | float | `0.1` | Fraction of population replaced at each Ggap. Range 0.0–0.5 |
| `patience` | int | `50` | Early stop if fitness does not improve for N generations |
| `n_cpus` | int | `1` | CPU cores for parallel fitness evaluation. Increase for faster runs |
| `n_runs` | int | `10` | Independent GA runs with different random seeds |

**Recommended settings by hardware**

```python
# Laptop (quick test)
two_gene_combination.run(pop_size=200, Gmax=20, patience=10, n_cpus=4, n_runs=2)

# Standard workstation
two_gene_combination.run(pop_size=1000, Gmax=100, patience=50, n_cpus=8, n_runs=10)

# HPC node
two_gene_combination.run(pop_size=2000, Gmax=200, patience=100, n_cpus=40, n_runs=10)
```

**Output files**

| File | Description |
|---|---|
| `two_gene_hof.csv` | Hall of Fame — best unique gene pairs across all runs. Columns: `seed_value`, `generation`, `LogicGates`, `Genes`, `Efficacy`, `Safety` |
| `two_gene_complete.csv` | All evaluated pairs across every generation and run |

**Return value:** `(df_hof, df_all)` — both as `pd.DataFrame`

---

## Manual Annotation — Skip PopV with Your Own Labels

If you already have cell-type annotations (from Seurat, Scanpy, CellTypist, or any other tool), you can skip Module 2 (PopV) entirely by providing your annotation column in Module 1.

### When to use this

Use `manual_annotation_col` when you supply your own `.h5ad` file and already have reliable cell-type labels. GEO ID inputs always run the full PopV pipeline regardless of this parameter.

### How to use it

```python
from SCART.geo_fetcher import SampleAnnotator

annotator = SampleAnnotator(
    "my_data.h5ad",
    manual_annotation_col="cell_type",   # name of your obs column
    min_genes=200,
    max_mt=40,
)
normal, tumor, unspecified, annotation_info, query_h5ad, cancer_type, results = annotator.run()
```

After `run()` completes, skip Module 2 entirely and go straight to Module 3:

```python
from SCART import preprocessing
adata_mal = preprocessing.run_preprocessing_pipeline(
    reference_h5ad     = "/path/to/tissue_reference.h5ad",
    malignant_strategy = "intersection",
)
```

### Requirements for your annotation column

**Column must exist in adata.obs.** A clear error listing all available columns is raised if it is missing.

**Epithelial labels must contain "epithelial cell".** Module 3 identifies epithelial cells by substring-matching `"epithelial cell"` (case-insensitive) anywhere in the label. Examples:

| Label | Valid? |
|---|---|
| `epithelial cell` | ✅ |
| `glandular epithelial cell` | ✅ |
| `ovarian surface epithelial cell` | ✅ |
| `luminal epithelial cell` | ✅ |
| `epithelial` | ❌ — does not contain "epithelial cell" |
| `Epithelial_cells` | ❌ — does not contain "epithelial cell" |
| `cancer cell` | ❌ — not recognised as epithelial |

Module 1 prints a warning immediately if no epithelial labels are detected, before any files are written. Module 3 raises a descriptive error with fix instructions if the column is missing or empty.

**All other labels are treated as non-epithelial** (the "rest" comparison group for DEG). Any string label is accepted for non-epithelial cells.

### What is stored in the output h5ad

| Key | Location | Value |
|---|---|---|
| `manual_annotation_col` | `adata.uns` | your column name |
| `skip_popv` | `adata.uns` | `True` |
| `popv_majority_vote_prediction` | `adata.obs` | copy of your annotation column |

### Multiple h5ad files

`manual_annotation_col` applies to all h5ad files passed. All files must share the same column name. The `popv_majority_vote_prediction` column is preserved through concatenation automatically.

```python
annotator = SampleAnnotator(
    "dataset1.h5ad",
    "dataset2.h5ad",
    manual_annotation_col="cell_type",
)
```

---

## Output Files

| Module | File | Contents |
|---|---|---|
| 1 | `GSE*_tumor.h5ad` / `combined_tumor.h5ad` / `input_tumor.h5ad` | Raw tumour counts, cancer type, optional QC params, optional manual annotation flags |
| 2 | `popv_results/final_popv_annotated.h5ad` | Cell-type labels, raw counts layer |
| 3 | `preprocessing_results/final_tumor.h5ad` | Malignant cells, binarised expression, DEG results, SCEVAN scores |
| 4a | `single_gene_results.csv` | Per-gene efficacy and safety scores |
| 4b | `two_gene_hof.csv` | Best gene pairs with logic gates |
| 4b | `two_gene_complete.csv` | Full GA search history |

---

## Parameter Reference

### SCEVAN parameters (Module 3)

| Parameter | Default | Range / Options | Effect of increasing | Effect of decreasing |
|---|---|---|---|---|
| `scevan_ref_max_cells` | `100` | `50–500` | More stable CNV baseline, slower | Faster, marginally less stable |
| `scevan_batch_size` | `3000` | `500–5000` | Fewer batches, more RAM per batch | More batches, less RAM per batch |
| `scevan_par_cores` | `1` | `1–N` | Faster per-batch classification | — |
| `scevan_subclones` | `False` | `True / False` | Infers tumour subclone structure | Runs faster, no subclone output |
| `scevan_organism` | `"human"` | `"human"` / `"mouse"` | — | — |
| `scevan_sample_name` | `"SCEVAN_run"` | any string | — | — |

### Safety threshold (Modules 4a, 4b)

| Value | Interpretation |
|---|---|
| `0.95` | 95% of healthy cell types must not express the target — very conservative |
| `0.90` | 90% must not express — recommended default |
| `0.80` | 80% must not express — more candidates, higher off-tumour risk |
