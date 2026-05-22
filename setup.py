from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import urllib.request
import os
import sys


OBO_URL = "http://purl.obolibrary.org/obo/cl.obo"


def _get_obo_path(install_lib=None):
    """
    Resolve the correct cl.obo destination path.
    Falls back to the already-importable package location if install_lib is None.
    """
    if install_lib:
        return os.path.join(install_lib, "SCART", "PopV", "resources", "ontology", "cl.obo")

    # Fallback: find the installed package via importlib
    try:
        import importlib.util
        spec = importlib.util.find_spec("SCART")
        if spec and spec.submodule_search_locations:
            pkg_root = list(spec.submodule_search_locations)[0]
            return os.path.join(pkg_root, "PopV", "resources", "ontology", "cl.obo")
    except Exception:
        pass

    # Last resort: relative to this setup.py (editable / source install)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "SCART", "PopV", "resources", "ontology", "cl.obo")


def _is_lfs_stub(path):
    """Return True if the file is a Git-LFS pointer stub, not the real file."""
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read(30).startswith("version https://git-lfs")
    except OSError:
        return False


def download_cl_obo(install_lib=None):
    obo_path = _get_obo_path(install_lib)
    os.makedirs(os.path.dirname(obo_path), exist_ok=True)

    if os.path.exists(obo_path) and not _is_lfs_stub(obo_path):
        print(f"cl.obo already present at {obo_path}, skipping download.")
        return

    print(f"Downloading cl.obo ontology (~17 MB) → {obo_path} ...")
    try:
        urllib.request.urlretrieve(OBO_URL, obo_path)
        print("cl.obo downloaded successfully.")
    except Exception as exc:
        print(
            f"WARNING: Failed to download cl.obo: {exc}\n"
            f"You can download it manually with:\n"
            f"  wget {OBO_URL!r} -O {obo_path!r}",
            file=sys.stderr,
        )


class PostInstall(install):
    """Runs after a normal `pip install`."""
    def run(self):
        super().run()
        download_cl_obo(self.install_lib)


class PostDevelop(develop):
    """Runs after `pip install -e .` (editable / develop mode)."""
    def run(self):
        super().run()
        download_cl_obo()          # install_lib not needed; source tree IS the package


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
            "**/*.tsv",
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
        "rpy2>=3.5",
    ],
    cmdclass={
        "install": PostInstall,
        "develop": PostDevelop,
    },
)
