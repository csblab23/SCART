from setuptools import setup, find_packages

setup(
    name="SCART",
    version="0.1.0",
    description="Single-cell Antigen Ranking Tool",
    author="CSB LAB",
    packages=find_packages(include=["SCART", "SCART.*"]),



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
        "scMalignantFinder==1.0.0",
        "typer",
        "rich",
        "onclass @ git+https://github.com/wangshenguiuc/OnClass.git"

    ]
)
