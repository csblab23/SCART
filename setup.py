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
    # core single-cell
    "scanpy==1.9.3",
    "anndata==0.9.1",

    # 🔴 scvi ecosystem (STRICT pinning)
    "scvi-tools==1.1.6.post2",
    "jax==0.4.23",
    "jaxlib==0.4.23",
    "numpyro==0.13.2",
    "chex==0.1.7",
    "optax==0.1.7",
    "flax==0.7.5",

    # data + utils
    "numpy>=1.23,<2",
    "pandas>=1.5",
    "scikit-learn>=1.2",

    # ML frameworks
    "torch==2.2.2",
    "tensorflow==2.12",

    # bio tools
    "popv>=0.5",
    "geofetch==0.12.10",
    "GEOparse==2.0.4",
    "scmalignantfinder==1.0.1",

    # misc
    "deap>=1.4",
    "rpy2==3.5.16",
    "typer",
    "rich"
]
)
