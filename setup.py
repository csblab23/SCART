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
    # --- Numerical stack (flexible, lets resolver work) ---
    "numpy>=1.23.4,<1.25",      # satisfies tensorflow 2.12 + numba 0.56 + jax 0.4.23
    "scipy>=1.9,<1.13",         # satisfies jax 0.4.23 + fixes tril bug (gone in 1.13)

    # --- JAX ---
    "jax==0.4.23",
    "jaxlib==0.4.23",

    # --- Single-cell ecosystem ---
    "anndata==0.9.1",
    "scanpy==1.9.3",
    "scanorama==1.7.4",
    "bbknn==1.6.0",
    "scgen==2.1.0",
    "scvi-tools==1.1.6.post2",
    "scrublet==0.2.3",
    "popv==0.6.0",
    "scmalignantfinder==1.1.9",  # ← upgraded; 1.0.1 hard-pinned scipy==1.13.1
    "celltypist==1.7.1",
    "umap-learn==0.5.7",
    "harmonypy==0.0.10",
    "harmony-pytorch==0.1.8",

    # --- ML / DL ---
    "torch==2.6.0",
    "pytorch-lightning==2.5.2",
    "tensorflow==2.12.0",

    # --- Analysis utilities ---
    "statsmodels==0.14.5",
    "scikit-image==0.24.0",
    "networkx==3.2.1",
    "igraph==0.11.9",
    "leidenalg==0.10.2",
    "louvain==0.8.2",
    "numba==0.56.4",

    # --- Bio tools ---
    "gseapy==1.1.11",
    "geofetch==0.12.10",
    "GEOparse==2.0.4",

    # --- Transformers ---
    "transformers==4.53.2",
    "sentence-transformers==5.0.0",

    # --- Misc ---
    "deap==1.4.3",
    "joblib",
    "rich",
    "typer",
    "pydantic",
]
)
