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
            "**/*.yml"
        ]
    },

    zip_safe=False,

install_requires=[
    # Core
    "scanpy==1.9.3",
    "scvi-tools==1.2.1",
    "scmalignantfinder==1.0.1",

    #  FIXED JAX stack (compatible)
    "jax==0.4.28",
    "jaxlib==0.4.28",
    "flax==0.10.2",
    "optax==0.2.1",
    "chex==0.1.7",

    # Numerical
    "numpy==1.23.4",
    "pandas>=1.5",
    "scikit-learn>=1.2",
    "scipy>=1.9",

    # ML
    "tensorflow==2.12.0",
    "torch>=2.0",

    # Bio
    "geofetch==0.12.10",
    "GEOparse==2.0.4",
    "popv>=0.5",

    # Utils
    "deap>=1.4",
    "rpy2==3.5.16",
    "typer",
    "rich",
    "ipywidgets"
]
)
