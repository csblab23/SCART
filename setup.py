from setuptools import setup, find_packages

setup(
    name="SCART",
    version="0.1.0",
    description="Single-cell Antigen Ranking Tool",
    author="CSB LAB",
    packages=find_packages(),

    python_requires=">=3.8",

    include_package_data=True,

 install_requires=[
    "scanpy>=1.9",
    "geofetch==0.12.10",
    "GEOparse==2.0.4",

    "popv>=0.5,<0.6",

    "scvi-tools==1.1.6.post2",   # 🔥 fixed
    "anndata==0.9.1",            # 🔥 lock
    "numpy>=1.23,<1.24",         # 🔥 lock

    "scikit-learn>=1.2,<1.3",
    "pandas>=1.5,<2.2",

    "torch>=2.0",
    "tensorflow>=2.12,<2.13",

    "deap>=1.4",
    "typer",
    "rich",

    
]
)
