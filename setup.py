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
        # Loose ranges — real pins live in requirements.txt
        "numpy>=1.23,<1.25",
        "scipy==1.13.1",
        "anndata==0.9.1",
        "scanpy>=1.9",
        "scanorama>=1.7",
        "bbknn>=1.6",
        "scgen>=2.1",
        "scvi-tools>=1.1",
        "scrublet>=0.2",
        "popv>=0.5",
        "scmalignantfinder>=1.0,<1.1",   # loose bound — 1.0.1 satisfies this
        "celltypist>=1.7",
        "umap-learn>=0.5",
        "harmonypy>=0.0.9",
        "harmony-pytorch>=0.1",
        "torch>=2.0",
        "pytorch-lightning>=2.5",
        "tensorflow>=2.12",
        "jax>=0.4",
        "jaxlib>=0.4",
        "statsmodels>=0.14",
        "scikit-image>=0.24",
        "networkx>=3.2",
        "igraph>=0.11",
        "leidenalg>=0.10",
        "louvain>=0.8",
        "numba>=0.56,<0.57",
        "gseapy>=1.1",
        "geofetch==0.12.10",
        "GEOparse==2.0.4",
        "transformers>=4.53",
        "sentence-transformers>=5.0",
        "deap>=1.4",
        "joblib",
        "rich",
        "typer",
        "pydantic",
    ]
)
