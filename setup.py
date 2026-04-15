from setuptools import setup, find_packages
 
setup(
    name="SCART",
    version="0.1.0",
    description="Single-cell Antigen Ranking Tool",
    author="CSB LAB",
    packages=find_packages(),
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        "SCART": [
            "**/*.json",
            "**/*.csv",
            "**/*.txt",
            "**/*.pkl",
            "**/*.h5ad",
            "**/*.yaml",
            "**/*.yml",
        ]
    },
    zip_safe=False,
    install_requires=[
        "scanpy>=1.9",
        "geofetch==0.12.10",
        "GEOparse==2.0.4",
        "popv>=0.5",
        "scvi-tools>=1.1",
        "numpy>=1.23,<2",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "torch>=2.0",
        "tensorflow>=2.12",
        "deap>=1.4",
        "rpy2==3.5.16",
        "scmalignantfinder==1.0.1",
        "typer",
        "rich",
        # --- JAX stack pinned to avoid jax.core.Shape removal in jax>=0.4.24 ---
        # chex<0.1.86 uses jax.core.Shape which was removed in jax 0.4.24+.
        # These four packages must be upgraded together.
        "jax==0.4.23",
        "jaxlib==0.4.23",
        "chex==0.1.85",
        "optax==0.1.9",
        "flax==0.7.5",
    ],
)
