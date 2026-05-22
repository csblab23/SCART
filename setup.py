from setuptools import setup, find_packages
from setuptools.command.install import install
import urllib.request, os

class PostInstall(install):
    def run(self):
        super().run()
        obo_path = os.path.join(
            self.install_lib,
            "SCART", "PopV", "resources", "ontology", "cl.obo"
        )
        os.makedirs(os.path.dirname(obo_path), exist_ok=True)
        if not os.path.exists(obo_path) or open(obo_path).read(20).startswith("version https://git-lfs"):
            print("Downloading cl.obo ontology file...")
            urllib.request.urlretrieve("http://purl.obolibrary.org/obo/cl.obo", obo_path)
            print("cl.obo downloaded successfully.")

setup(
    name="SCART",
    version="0.1.0",
    description="Single-cell Antigen Ranking Tool",
    author="CSB LAB",
    packages=find_packages(include=["SCART", "SCART.*"]),
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
            "**/*.joblib",
            "**/*.gmt",
            "**/*.tsv"
        ]
    },
    zip_safe=False,
    install_requires=[
        "numpy>=1.24,<2.0",   
        "pandas>=1.5",
        "scikit-learn",
        "typer",
        "rich",
        "GEOparse",
        "geofetch",
        "popv==0.4.2",
        "deap==1.4",
        "scipy>=1.10",
        "scanpy>=1.9",
        "scvi-tools",
        "torch",
        "rpy2>=3.5"
    ],
    cmdclass={"install": PostInstall},
)
