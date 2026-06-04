"""
SCART setup.py
==============
Handles Python package metadata and installation only.

Post-installation setup (R, SCEVAN, ontology, Windows pip fixes)
is handled by a separate interactive script. Run ONCE after pip install:

    python -m SCART.install

Full installation sequence:
    Step 0  (Windows only) : Install VC++ Redistributable manually
    Step 1  (all)          : conda create -n scart_env python=3.10 -y
                             conda activate scart_env
    Step 2  (Windows only) : Pre-install JAX stubs + python-annoy via conda
    Step 3  (all)          : pip install git+https://github.com/csblab23/SCART.git
    Step 4  (all)          : python -m SCART.install   <- interactive, asks your OS
"""

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop


# ---------------------------------------------------------------------------
# Post-install notice  (shown after pip install)
# ---------------------------------------------------------------------------

POSTINSTALL_MSG = """
================================================================
  SCART installed successfully!
================================================================

  NEXT STEP — run the interactive setup to install
  R, SCEVAN, and OS-specific dependencies:

      python -m SCART.install

  It will ask for your OS and tell you exactly what runs
  automatically and what (little) you need to do manually.
================================================================
"""


class PostInstall(install):
    """Triggered by: pip install ."""
    def run(self):
        super().run()
        print(POSTINSTALL_MSG)


class PostDevelop(develop):
    """Triggered by: pip install -e ."""
    def run(self):
        super().run()
        print(POSTINSTALL_MSG)


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------

setup(
    name="SCART",
    version="0.1.0",
    description="Single-cell Antigen Ranking Tool",
    author="CSB LAB",
    packages=find_packages(include=["SCART", "SCART.*"]),
    include_package_data=True,
    package_data={
        "SCART": [
            "**/*.json", "**/*.csv", "**/*.txt", "**/*.pkl", "**/*.h5ad",
            "**/*.yaml", "**/*.yml", "**/*.joblib", "**/*.gmt", "**/*.tsv",
            "PopV/resources/ontology/cl.obo",
            "PopV/resources/ontology/cl.ontology",
        ]
    },
    zip_safe=False,
    python_requires=">=3.9,<3.12",
    install_requires=[
        # Core scientific stack
        "numpy>=1.24,<2.0",
        "scipy==1.12.0",            # pinned: jax==0.4.23 breaks with scipy>=1.13
        "pandas>=1.5",
        "scikit-learn",

        # JAX ecosystem — pinned for scvi-tools 1.1.6 + Windows compat
        "jax[cpu]==0.4.23",
        "jaxlib==0.4.23",
        "optax==0.1.7",
        "flax==0.7.5",
        "orbax-checkpoint<0.5",
        "numpyro<=0.13.2",

        # Single-cell libraries
        "scanpy>=1.9",
        "scvi-tools==1.1.6.post2",
        "popv==0.4.2",

        # Deep-learning back-ends
        # torch: on Windows the install script replaces this with
        #        torch==2.2.2 from the PyTorch CPU wheel index
        "torch",
        # tensorflow: on Windows the install script pins to
        #             tensorflow-cpu==2.10.0 (last version without CUDA on Win)
        # Linux/Mac: user controls TF version (GPU builds etc.)

        # Genomics / enrichment
        "GEOparse",
        "geofetch",
        "deap==1.4",

        # R bridge
        "rpy2>=3.5",

        # CLI
        "typer",
        "rich",
    ],
    # Entry point so `python -m SCART.install` works
    cmdclass={
        "install": PostInstall,
        "develop": PostDevelop,
    },
)
