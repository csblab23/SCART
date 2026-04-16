from setuptools import setup, find_packages

setup(
    name="SCART",
    version="0.1.0",
    description="Single-cell Antigen Ranking Tool",
    author="CSB LAB",
    packages=find_packages(),
    python_requires=">=3.8,<3.11",

    include_package_data=True,
    package_data={
        "SCART": [
            "**/*.json",
            "**/*.csv",
            "**/*.txt",
            "**/*.pkl",
            "**/*.h5ad",
            "**/*.yaml",
            "**/*.yml"
        ]
    },

    zip_safe=False,

    install_requires=[
        # Core scientific stack
        "numpy>=1.23,<1.25",
        "scipy==1.13.1",
        "anndata==0.9.1",

        # Single-cell ecosystem
        "scanpy>=1.9",
        "scanorama>=1.7",
        "bbknn>=1.6",
        "scgen>=2.1",
        "scvi-tools>=1.1",   # keep modern
        "scrublet>=0.2",
        "popv>=0.5,<0.6",    # 🔥 CRITICAL FIX
        "scmalignantfinder>=1.0,<1.1",
        "celltypist>=1.7",

        # Integration / embedding
        "umap-learn>=0.5",
        "harmonypy>=0.0.9",
        "harmony-pytorch>=0.1",

        # ML / DL
        "torch>=2.0",
        "pytorch-lightning>=2.5",
        "tensorflow>=2.12",

        # Analysis utilities
        "statsmodels>=0.14",
        "scikit-image>=0.24",
        "networkx>=3.2",
        "igraph>=0.11",
        "leidenalg>=0.10",
        "louvain>=0.8",
        "numba>=0.56,<0.57",
        "gseapy>=1.1",

        # GEO
        "geofetch==0.12.10",
        "GEOparse==2.0.4",

        # Transformers
        "transformers>=4.53",
        "sentence-transformers>=5.0",

        # Misc
        "deap>=1.4",
        "joblib",
        "rich",
        "typer",
        "pydantic",
    ]
)
