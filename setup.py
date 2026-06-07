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
    Step 2  (Windows only) : Pre-install JAX stubs + annoy via conda
    Step 3  (all)          : pip install git+https://github.com/csblab23/SCART.git
                             (PostInstall auto-fixes torch on Windows)
    Step 4  (all)          : python -m SCART.install   <- interactive, asks your OS
"""

import subprocess
import sys
import platform

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop


# ---------------------------------------------------------------------------
# Windows torch auto-fix
# ---------------------------------------------------------------------------

def _fix_torch_windows():
    """
    Fix the torch DLL error on Windows immediately after pip install SCART.

    Why here (PostInstall) and not in install.py
    --------------------------------------------
    `python -m SCART.install` imports SCART/__init__.py on startup, which
    chains into scanpy -> anndata -> torch.  If torch has the DLL error,
    the installer is completely unreachable.

    Running the fix here (in setup.py's PostInstall hook) solves it before
    the user ever tries to run `python -m SCART.install`.  No SCART import
    happens in this file — only subprocess calls.
    """
    print("\n[SCART:setup] Windows detected — auto-fixing PyTorch CPU wheel ...")
    print("[SCART:setup] Uninstalling existing torch ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "torch", "-y"],
        check=False,
    )
    print("[SCART:setup] Installing torch==2.2.2 (CPU wheel, no DLL issues) ...")
    result = subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "torch==2.2.2",
            "--index-url", "https://download.pytorch.org/whl/cpu",
        ],
        check=False,
    )
    if result.returncode == 0:
        print("[SCART:setup] torch==2.2.2 CPU wheel installed successfully.")
    else:
        print(
            "[SCART:setup] WARNING: torch auto-fix failed. Run manually:\n"
            "  pip uninstall torch -y\n"
            "  pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu",
            file=sys.stderr,
        )


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
        if platform.system() == "Windows":
            _fix_torch_windows()
        print(POSTINSTALL_MSG)


class PostDevelop(develop):
    """Triggered by: pip install -e ."""
    def run(self):
        super().run()
        if platform.system() == "Windows":
            _fix_torch_windows()
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

        # annoy — NOT listed here intentionally.
        # PyPI has no pre-built Windows wheel for annoy (source-only tarball).
        # Listing it here would cause pip to build from source on Windows,
        # requiring C++ Build Tools. Instead, install it via conda BEFORE
        # pip install SCART:
        #   conda install -c conda-forge python-annoy -y
        # Then register it with pip (see install.py _ensure_annoy_windows).
        # annoy is pulled in transitively via popv — pip will skip the build
        # once it sees annoy already registered.

        # Deep-learning back-ends
        # torch: PostInstall auto-replaces this with torch==2.2.2 CPU wheel on Windows.
        # Linux/Mac: default torch (GPU/CPU as available).
        "torch",
        # tensorflow: on Windows install.py pins to tensorflow-cpu==2.10.0
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
