from setuptools import setup, find_packages

# ─────────────────────────────────────────────────────────────────────────────
# SCART — Single-Cell Antigen Ranking Tool
#
# Compatibility matrix (Python 3.8 – 3.10 ONLY; 3.11+ breaks tensorflow 2.12):
#
#   popv 0.5.x        ← requires scvi-tools <1.0  (1.x broke the entire API)
#   scvi-tools 0.20.3 ← last stable 0.20 release; works with popv 0.5
#   anndata 0.9.2     ← upper-bounded by both popv and scvi-tools 0.20
#   numpy 1.23.4      ← numpy <1.24 required by tensorflow 2.12;
#                        also keeps C-extension ABI stable for scipy/scanpy
#   pandas 1.5.3      ← GEOparse 2.0.4 uses deprecated pandas 2.x APIs;
#                        pin to 1.5.x avoids FutureWarning / breakage
#   scipy 1.11.x      ← upper-bounded to stay compatible with numpy 1.23
#   pytorch 2.0.x     ← scvi-tools 0.20 supports torch >=1.13, <=2.0
#   pytorch-lightning ← scvi-tools 0.20 requires <2.0
#   tensorflow 2.12   ← ONCLASS (used inside popv) uses keras; 2.12 is last
#                        version that ships keras as a sub-module AND works
#                        with numpy 1.23
#   protobuf 3.20.3   ← tensorflow 2.12 is incompatible with protobuf >=4.0
# ─────────────────────────────────────────────────────────────────────────────

setup(
    name="SCART",
    version="0.1.0",
    description="Single-Cell Antigen Ranking Tool",
    author="CSB LAB",
    packages=find_packages(),
    python_requires=">=3.8, <3.11",   # tensorflow 2.12 hard ceiling at 3.10
    include_package_data=True,

    install_requires=[

        # ── core scientific stack ──────────────────────────────────────────
        "numpy==1.23.4",               # ceiling: tensorflow 2.12 + scipy ABI
        "scipy>=1.10,<1.12",           # compatible with numpy 1.23
        "pandas>=1.5,<2.0",            # ceiling: GEOparse 2.0.4 compatibility
        "h5py>=3.7,<3.10",             # anndata 0.9 h5ad I/O

        # ── AnnData / single-cell core ─────────────────────────────────────
        "anndata>=0.9,<0.10",          # pinned: popv 0.5 + scvi-tools 0.20
        "scanpy>=1.9,<1.10",           # scanpy 1.10 dropped Python <3.9 APIs

        # ── deep-learning backends ─────────────────────────────────────────
        "torch>=1.13,<2.1",            # scvi-tools 0.20 upper bound
        "pytorch-lightning>=1.8,<2.0", # scvi-tools 0.20 requirement
        "tensorflow==2.12.*",          # ONCLASS/keras dependency; numpy <1.24
        "protobuf>=3.20,<4.0",         # tensorflow 2.12 hard requirement

        # ── scVI / PopV ecosystem ──────────────────────────────────────────
        "scvi-tools==0.20.3",          # CRITICAL: popv 0.5 breaks on 1.x
        "popv>=0.5,<0.6",              # Module 2 core; 0.6 not tested
        "ml-dtypes==0.2.0",            # version lock: jax/tf shared dtype lib

        # ── PopV method backends ───────────────────────────────────────────
        "celltypist>=1.3,<1.8",        # CELLTYPIST method in Module 2
        "harmonypy>=0.0.9,<0.0.11",    # KNN_HARMONY method
        "bbknn>=1.5,<1.7",             # KNN_BBKNN method
        "scgen>=2.1,<2.2",             # SCGEN method (optional popv backend)
        "xgboost>=1.7,<2.1",           # XGboost method in Module 2
        "scanorama>=1.7",              # optional integration backend

        # ── GEO data ingestion (Module 1) ──────────────────────────────────
        "GEOparse==2.0.4",
        "geofetch>=0.12,<0.13",
        "requests>=2.28",
        "urllib3>=1.26,<2.0",          # urllib3 2.x changed retry API

        # ── graph / clustering ─────────────────────────────────────────────
        "networkx>=2.8,<3.3",          # ONCLASS ontology graph; 3.3 = numpy 2
        "igraph>=0.10,<0.12",
        "leidenalg>=0.9,<0.11",
        "louvain>=0.7,<0.9",
        "umap-learn>=0.5,<0.6",

        # ── ML utilities ──────────────────────────────────────────────────
        "scikit-learn>=1.2,<1.3",      # ceiling: ABI stable with numpy 1.23
        "scikit-image>=0.19,<0.22",
        "statsmodels>=0.13,<0.15",

        # ── scrublet (QC doublet detection, used by scanpy pipeline) ──────
        "scrublet>=0.2.3",

        # ── genetic algorithm (Module 4) ───────────────────────────────────
        "deap>=1.3,<1.5",

        # ── NLP / transformers (ONCLASS text embeddings) ───────────────────
        "transformers>=4.20,<4.36",    # ceiling: 4.36 requires torch >=2.1
        "sentence-transformers>=2.2,<3.0",

        # ── CLI / display ──────────────────────────────────────────────────
        "typer>=0.7",
        "rich>=12.0",
        "pydantic>=1.10,<2.0",         # scvi-tools 0.20 uses pydantic v1 API
    ],

    extras_require={
        # rpy2 + R package copykat are required for Module 3 CopyKAT step.
        # rpy2 is not listed in install_requires because it requires a working
        # R installation (>=4.1) to build native extensions.  Install manually:
        #   pip install "SCART[copykat]"
        # then inside R:
        #   devtools::install_github("navinlabcode/copykat")
        "copykat": [
            "rpy2>=3.5,<3.6",
        ],
    },
)
