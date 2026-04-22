# SCART — Single-Cell CAR-T Target Discovery

**SCART** is an end-to-end computational pipeline for identifying tumour-specific surface protein targets for CAR-T cell therapy from single-cell RNA-seq data. Starting from raw GEO accession IDs or user-provided h5ad files, SCART automates cell-type annotation, malignant cell identification, surfaceome differential expression, and logic-gate gene combination scoring to rank candidate CAR-T targets by efficacy and safety.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation](#installation)
- [Quick Start](#SCART — Quick Start)
- [Module Reference](#module-reference)
  - [Module 1 — Data Acquisition](#module-1--data-acquisition-geo_fetcherpy)
  - [Module 2 — Cell-Type Annotation](#module-2--cell-type-annotation-popv_annotationpy)
  - [Module 3 — Preprocessing & Malignancy Detection](#module-3--preprocessing--malignancy-detection-preprocessingpy)
  - [Module 4a — Single-Gene Scoring](#module-4a--single-gene-scoring-one_gene_combinationpy)
  - [Module 4b — Two-Gene Logic-Gate Scoring](#module-4b--two-gene-logic-gate-scoring-two_gene_combinationpy)
- [Output Files](#output-files)
- [Parameter Reference](#parameter-reference)

---

## Overview

CAR-T therapy requires surface targets that are highly expressed on tumour cells and absent on healthy tissue. SCART automates this discovery by:

1. Downloading and parsing scRNA-seq datasets from GEO
2. Annotating cell types with PopV (multi-method consensus)
3. Identifying malignant epithelial cells via scMalignantFinder and inferCNA
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

### 3. Install R and Bioconductor dependencies

These are required for inferCNA (Module 3). Run inside your activated conda environment:

```bash
conda install -c conda-forge -c bioconda \
  r-base r-devtools r-remotes r-ggplot2 r-data.table \
  r-igraph r-gdtools r-ragg r-dplyr \
  cairo freetype fontconfig harfbuzz fribidi \
  libpng libtiff libjpeg libwebp
```

```bash
conda install -c bioconda \
  bioconductor-annotationdbi bioconductor-go.db \
  bioconductor-org.hs.eg.db bioconductor-biomart \
  bioconductor-scran bioconductor-genomicfeatures \
  bioconductor-rtracklayer \
  bioconductor-txdb.hsapiens.ucsc.hg19.knowngene \
  bioconductor-clusterprofiler bioconductor-enrichplot \
  bioconductor-ggtree bioconductor-homo.sapiens
```

Then inside R:

```r
remotes::install_github("hrbrmstr/hrbrthemes")
remotes::install_github("jlaffy/scalop")
remotes::install_github("jlaffy/infercna")
```

### 4. Set up Jupyter Notebook

```bash
pip install notebook ipykernel
python -m ipykernel install --user --name=scart_env --display-name "Python (scart_env)"
jupyter notebook
```

---

# SCART — Quick Start

---

## Module 1 — Data Acquisition

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

---

## Module 2 — Cell-Type Annotation

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

## Module 3 — Preprocessing & Malignancy Detection

```python
from SCART import preprocessing

# ── Option 1: Standard run with inferCNA + scMalignantFinder ─────────────
# Both tools used; a cell must be called malignant by BOTH (intersection)
adata_preprocessed = preprocessing.run_preprocessing_pipeline(
    reference_h5ad            = "/path/to/tissue_reference.h5ad",
    log2fc_threshold          = 2.0,
    pval_adj_threshold        = 0.05,
    malignant_strategy        = "intersection",
    infercna_genome           = "hg19",
    infercna_n                = 5000,
    infercna_noise            = 0.1,
    infercna_signal_threshold = 0.9,
    infercna_ref_max_cells    = 2000,
)

# ── Option 2: scMalignantFinder only (no inferCNA) ───────────────────────
# Use when no tissue reference is available, or to skip the R/rpy2 step
adata_preprocessed = preprocessing.run_preprocessing_pipeline(
    malignant_strategy = "scMalignant",
    log2fc_threshold   = 2.0,
    pval_adj_threshold = 0.05,
)

# ── Option 3: inferCNA only ───────────────────────────────────────────────
# Use when you trust CNA profiling more than the ML classifier
adata_preprocessed = preprocessing.run_preprocessing_pipeline(
    reference_h5ad     = "/path/to/tissue_reference.h5ad",
    malignant_strategy = "infercna",
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

# ── Option 5: hg38 genome (for newer datasets) ───────────────────────────
adata_preprocessed = preprocessing.run_preprocessing_pipeline(
    reference_h5ad     = "/path/to/tissue_reference.h5ad",
    malignant_strategy = "intersection",
    infercna_genome    = "hg38",
)

# ── Option 6: Explicit file paths (if auto-detection fails) ───────────────
adata_preprocessed = preprocessing.run_preprocessing_pipeline(
    popv_path      = "/path/to/final_popv_annotated.h5ad",
    tumor_h5ad     = "/path/to/GSE158937_tumor.h5ad",
    reference_h5ad = "/path/to/tissue_reference.h5ad",
    save_dir       = "/path/to/my_output_dir/",
)

# ── Option 7: Faster inferCNA (fewer reference cells, smaller n) ──────────
adata_preprocessed = preprocessing.run_preprocessing_pipeline(
    reference_h5ad         = "/path/to/tissue_reference.h5ad",
    infercna_n             = 2000,   # fewer genes → faster
    infercna_ref_max_cells = 500,    # fewer ref cells → faster
)

# ── What you get ──────────────────────────────────────────────────────────
# adata_preprocessed → malignant epithelial cells only, binarised expression
# adata_preprocessed.obs columns:
#   scMalignantFinder_prediction, infercna_prediction,
#   infercna_cna_signal, infercna_cna_cor, final_malignant
# adata_preprocessed.uns keys:
#   filtered_deg, all_deg, deg_params, infercna_results, qc_params
print(adata_preprocessed.uns["filtered_deg"].head(10))
```

> **Output files**
> - `preprocessing_results/final_tumor.h5ad`

---

## Module 4a — Single-Gene Scoring

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

## Module 4b — Two-Gene Logic-Gate Scoring

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

annotator = SampleAnnotator(*inputs, min_genes=None, max_mt=None)
normal, tumor, unspecified, annotation_info, query_h5ad, cancer_type, results = annotator.run()
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `*inputs` | str | — | One or more GEO accession IDs (e.g. `"GSE158937"`) or paths to `.h5ad` files |
| `min_genes` | int or None | `None` | Minimum genes per cell for QC in Module 3. `None` = QC skipped |
| `max_mt` | float or None | `None` | Maximum mitochondrial % per cell. `None` = QC skipped |

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

---

### Module 2 — Cell-Type Annotation (`popv_annotation.py`)

Annotates cell types using PopV, a consensus framework that runs multiple methods (CELLTYPIST, KNN-BBKNN, KNN-SCVI, KNN-HARMONY, ONCLASS, SCANVI, Support Vector, XGBoost) and reports a majority-vote prediction.

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
| `popv_results/final_popv_annotated.h5ad` | Full dataset with `popv_majority_vote_prediction` column and `layers['full_counts']` for Module 3 |

---

### Module 3 — Preprocessing & Malignancy Detection (`preprocessing.py`)

Identifies malignant epithelial cells using scMalignantFinder and inferCNA, performs surfaceome differential expression analysis, and outputs a binarised tumour matrix for Module 4.

```python
from SCART import preprocessing

adata_preprocessed = preprocessing.run_preprocessing_pipeline(
    reference_h5ad            = "/path/to/tissue_reference.h5ad",
    log2fc_threshold          = 2.0,
    pval_adj_threshold        = 0.05,
    malignant_strategy        = "intersection",
    infercna_genome           = "hg19",
    infercna_n                = 5000,
    infercna_noise            = 0.1,
    infercna_signal_threshold = 0.9,
    infercna_ref_max_cells    = 2000,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adata` | AnnData or None | `None` | Module 2 output. Auto-loaded from `popv_results/` if None |
| `popv_path` | str or None | `None` | Explicit path to PopV h5ad |
| `log2fc_threshold` | float | `1.0` | DEG log2 fold-change cutoff |
| `pval_adj_threshold` | float | `0.05` | DEG BH-adjusted p-value cutoff |
| `reference_h5ad` | str or None | `None` | Tabula Sapiens h5ad for inferCNA normal reference. inferCNA skipped if None |
| `tumor_h5ad` | str or None | `None` | Module 1 h5ad for scMalignantFinder full-gene recovery. Auto-detected if None |
| `save_dir` | str or None | `None` | Output directory. Default: `<cwd>/preprocessing_results/` |
| `malignant_strategy` | str | `"intersection"` | `"intersection"` (both tools must agree), `"scMalignant"`, or `"infercna"` |
| `infercna_genome` | str | `"hg19"` | `"hg19"` or `"hg38"` — must match your data's genome build |
| `infercna_n` | int | `5000` | Top variable genes for CNA profiling. Auto-capped to available common genes |
| `infercna_noise` | float | `0.1` | CNA noise floor. Range 0.05–0.5. Lower = more sensitive |
| `infercna_signal_threshold` | float | `0.9` | Top fraction of CNA values used for malignancy signal score |
| `infercna_ref_max_cells` | int | `2000` | Maximum reference cells subsampled for inferCNA |

**QC thresholds** are not parameters here — they are read automatically from `adata.uns['qc_params']` set in Module 1.

**Pipeline steps**

1. Load full PopV-annotated dataset
2. Read QC thresholds from `adata.uns['qc_params']` (skipped if absent)
3. Extract epithelial cells → apply QC filters if set
4. Run **scMalignantFinder** (machine learning classifier) on full gene space
5. Run **inferCNA** (copy number alteration profiling) against normal reference
6. Combine predictions with chosen strategy → `final_malignant` column
7. Keep only malignant epithelial cells; extract non-epithelial cells as DEG reference
8. Filter both groups to surfaceome genes (GESP database)
9. Wilcoxon DEG: malignant epithelial vs non-epithelial rest
10. Binarise expression matrix (0/1), store DEG results, save

**Output files**

| File | Description |
|---|---|
| `preprocessing_results/final_tumor.h5ad` | Malignant epithelial cells, surfaceome-filtered, binarised. Contains `obs['final_malignant']`, DEG in `uns['filtered_deg']`, inferCNA scores in `uns['infercna_results']` |

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

## Output Files

| Module | File | Contents |
|---|---|---|
| 1 | `GSE*_tumor.h5ad` / `combined_tumor.h5ad` | Raw tumour counts, cancer type, optional QC params |
| 2 | `popv_results/final_popv_annotated.h5ad` | Cell-type labels, full gene counts layer |
| 3 | `preprocessing_results/final_tumor.h5ad` | Malignant cells, binarised expression, DEG results, inferCNA scores |
| 4a | `single_gene_results.csv` | Per-gene efficacy and safety scores |
| 4b | `two_gene_hof.csv` | Best gene pairs with logic gates |
| 4b | `two_gene_complete.csv` | Full GA search history |

---

## Parameter Reference

### inferCNA parameters (Module 3)

| Parameter | Default | Range | Effect of increasing | Effect of decreasing |
|---|---|---|---|---|
| `infercna_genome` | `"hg19"` | `"hg19"` / `"hg38"` | — | — |
| `infercna_n` | `5000` | `500–20000` | More genes → smoother, slower | Fewer genes → faster, noisier |
| `infercna_noise` | `0.1` | `0.05–0.5` | More conservative, fewer false positives | More sensitive, more false positives |
| `infercna_signal_threshold` | `0.9` | `0.7–0.99` | Stricter — fewer cells called malignant | More permissive — more cells called malignant |
| `infercna_ref_max_cells` | `2000` | `500–5000` | More accurate, slower | Faster, marginally less accurate |

### Safety threshold (Modules 4a, 4b)

| Value | Interpretation |
|---|---|
| `0.95` | 95% of healthy cell types must not express the target — very conservative |
| `0.90` | 90% must not express — recommended default |
| `0.80` | 80% must not express — more candidates, higher off-tumour risk |
