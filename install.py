"""
SCART/install.py
================
Interactive post-installation setup for SCART.

Run this ONCE after installing SCART:
    python -m SCART.install

It will:
  - Ask you to choose your OS (Linux / Windows / Mac)
  - Show exactly what will run automatically
  - Show exactly what you must do manually (with commands)
  - Ask for confirmation before doing anything
  - Run all automatable steps for your OS
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import urllib.request


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBO_URL = "http://purl.obolibrary.org/obo/cl.obo"

SEP  = "=" * 62
SEP2 = "-" * 62

# ── Linux / Mac conda packages ───────────────────────────────────────────────

# Step 3a — R base + graphics libraries
CONDA_R_BASE = [
    "r-base", "r-devtools", "r-remotes",
    "r-ggplot2", "r-data.table", "r-igraph",
    "r-gdtools", "r-ragg", "r-dplyr",
    "cairo", "freetype", "fontconfig",
    "harfbuzz", "fribidi", "libpng",
    "libtiff", "libjpeg-turbo", "libwebp",
]

# Step 3b — Bioconductor + CRAN packages (available via conda on Linux/Mac only)
CONDA_R_BIO = [
    "bioconductor-scran", "bioconductor-fgsea", "bioconductor-ggtree",
    "r-paralleldist", "r-pheatmap", "r-forcats",
    "r-cluster", "r-rtsne", "r-ape", "r-tidytree", "r-ggrepel",
]

# ── Windows pip fixes ────────────────────────────────────────────────────────

WIN_TORCH_UNINSTALL = ["torch"]
WIN_TORCH_INSTALL   = ["torch==2.2.2", "--index-url", "https://download.pytorch.org/whl/cpu"]
WIN_TF_UNINSTALL    = ["tensorflow", "tensorflow-cpu", "tensorflow-intel"]
WIN_TF_INSTALL      = ["tensorflow-cpu==2.10.0"]


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _run(cmd: list, label: str = "") -> bool:
    tag = f"[SCART:{label}]" if label else "[SCART]"
    print(f"\n{tag} Running:\n  {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            f"{tag} WARNING: command exited with code {result.returncode}.\n"
            f"  If this is critical, run it manually:\n"
            f"    {' '.join(cmd)}",
            file=sys.stderr,
        )
        return False
    return True


def _pip(*args) -> bool:
    return _run([sys.executable, "-m", "pip", "install"] + list(args), label="pip")


def _pip_uninstall(*packages) -> bool:
    return _run(
        [sys.executable, "-m", "pip", "uninstall", "-y"] + list(packages),
        label="pip",
    )


def _conda(*args) -> bool:
    conda_exe = shutil.which("conda")
    if not conda_exe:
        print(
            "\n[SCART:conda] ERROR: 'conda' not found on PATH.\n"
            "  Make sure your conda environment is activated:\n"
            "    conda activate scart_env\n"
            "  Then re-run: python -m SCART.install",
            file=sys.stderr,
        )
        return False
    return _run([conda_exe] + list(args), label="conda")


def _rscript(r_code: str, label: str = "R") -> bool:
    rscript = shutil.which("Rscript")
    if not rscript:
        print(
            "\n[SCART:R] ERROR: 'Rscript' not found on PATH.\n"
            "  R does not appear to be installed yet.\n"
            "  Install R first (Step 3a/4a), then re-run: python -m SCART.install",
            file=sys.stderr,
        )
        return False
    return _run([rscript, "-e", r_code], label=label)


# ---------------------------------------------------------------------------
# Ontology download  (all platforms)
# ---------------------------------------------------------------------------

def _get_obo_path() -> str:
    try:
        spec = importlib.util.find_spec("SCART")
        if spec and spec.submodule_search_locations:
            pkg_root = list(spec.submodule_search_locations)[0]
            return os.path.join(pkg_root, "PopV", "resources", "ontology", "cl.obo")
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "PopV", "resources", "ontology", "cl.obo")


def _is_lfs_stub(path: str) -> bool:
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read(30).startswith("version https://git-lfs")
    except OSError:
        return False


def download_cl_obo() -> bool:
    obo_path = _get_obo_path()
    os.makedirs(os.path.dirname(obo_path), exist_ok=True)

    if os.path.exists(obo_path) and not _is_lfs_stub(obo_path):
        print(f"[SCART] cl.obo already present at:\n  {obo_path}\n  Skipping download.")
        return True

    print(f"[SCART] Downloading cl.obo ontology (~17 MB) ...\n  -> {obo_path}")
    try:
        urllib.request.urlretrieve(OBO_URL, obo_path)
        print("[SCART] cl.obo downloaded successfully.")
        return True
    except Exception as exc:
        print(
            f"[SCART] WARNING: Failed to download cl.obo: {exc}\n"
            f"  Download it manually:\n"
            f"    wget '{OBO_URL}' -O '{obo_path}'",
            file=sys.stderr,
        )
        return False


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def _verify_python():
    print(f"\n{SEP2}")
    print("Verifying Python packages ...")
    print(SEP2)
    _run(
        [
            sys.executable, "-c",
            (
                "import jax, flax, scvi, SCART; "
                "print('jax   :', jax.__version__); "
                "import numpy; print('numpy :', numpy.__version__); "
                "print('scvi  :', scvi.__version__); "
                "print('SCART : OK')"
            ),
        ],
        label="verify",
    )
    _run(
        [
            sys.executable, "-c",
            (
                "import SCART; "
                "from SCART.geo_fetcher import SampleAnnotator; "
                "from SCART import popv_annotation, preprocessing; "
                "from SCART.gene_combination_predictor import "
                "one_gene_combination, two_gene_combination; "
                "print('All SCART imports OK')"
            ),
        ],
        label="verify",
    )


def _verify_r():
    print(f"\n{SEP2}")
    print("Verifying R / SCEVAN ...")
    print(SEP2)
    _rscript("library(SCEVAN); cat('SCEVAN OK\\n')", label="verify")


# ---------------------------------------------------------------------------
# Linux / Mac automated steps
# ---------------------------------------------------------------------------

def _run_linux_mac(os_name: str):
    print(f"\n{SEP}")
    print(f"  Running automated steps for {os_name}")
    print(SEP)

    # --- Step 1: ontology ---
    print(f"\n[Step 1/4] Downloading cl.obo ontology ...")
    download_cl_obo()

    # --- Step 2: R base via conda ---
    print(f"\n[Step 2/4] Installing R base + graphics stack via conda ...")
    ok = _conda(
        "install",
        "--override-channels",
        "-c", "conda-forge",
        "-c", "bioconda",
        "-c", "defaults",
        "-y",
        *CONDA_R_BASE,
    )
    if not ok:
        print("[SCART] Step 2 failed. Fix the conda issue above and re-run.", file=sys.stderr)
        return

    # --- Step 3: channel priority + Bioconductor via conda ---
    print(f"\n[Step 3/4] Setting flexible channel priority ...")
    _conda("config", "--set", "channel_priority", "flexible")

    print(f"\n[Step 3/4] Installing Bioconductor + CRAN packages via conda ...")
    ok = _conda(
        "install",
        "-c", "conda-forge",
        "-c", "bioconda",
        "-y",
        *CONDA_R_BIO,
    )
    if not ok:
        print("[SCART] Step 3 failed. Fix the conda issue above and re-run.", file=sys.stderr)
        return

    # --- Step 4: SCEVAN via Rscript ---
    print(f"\n[Step 4/4] Installing SCEVAN via Rscript ...")
    ok = _rscript(
        "library(devtools); "
        "install_github('miccec/yaGST'); "
        "install_github('AntonioDeFalco/SCEVAN')"
    )
    if not ok:
        print("[SCART] SCEVAN install failed. See error above.", file=sys.stderr)
        return

    # --- Verification ---
    _verify_python()
    _verify_r()

    print(f"\n{SEP}")
    print(f"  {os_name} setup complete!")
    print(SEP)


# ---------------------------------------------------------------------------
# Windows automated steps
# ---------------------------------------------------------------------------

def _run_windows():
    print(f"\n{SEP}")
    print("  Running automated pip fixes for Windows")
    print(SEP)

    # --- Step 1: ontology ---
    print("\n[Step 1/5] Downloading cl.obo ontology ...")
    download_cl_obo()

    # --- Step 2: re-pin numpy + scipy ---
    print("\n[Step 2/5] Re-pinning numpy + scipy ...")
    _pip("numpy>=1.24,<2.0", "scipy==1.12.0", "--force-reinstall")

    # --- Step 3: re-pin full JAX stack ---
    print("\n[Step 3/5] Re-pinning full JAX stack ...")
    _pip(
        "jax[cpu]==0.4.23",
        "jaxlib==0.4.23",
        "optax==0.1.7",
        "flax==0.7.5",
        "orbax-checkpoint<0.5",
        "numpyro<=0.13.2",
        "numpy>=1.24,<2.0",
        "scipy==1.12.0",
        "--force-reinstall",
    )

    # --- Step 4: fix PyTorch DLL ---
    print("\n[Step 4/5] Fixing PyTorch (CPU wheel, no DLL issues) ...")
    _pip_uninstall(*WIN_TORCH_UNINSTALL)
    _pip(*WIN_TORCH_INSTALL)

    # --- Step 5: pin TensorFlow ---
    print("\n[Step 5/5] Pinning TensorFlow CPU to 2.10.0 ...")
    _pip_uninstall(*WIN_TF_UNINSTALL)
    _pip(*WIN_TF_INSTALL)

    # --- Verification ---
    _verify_python()

    print(f"\n{SEP}")
    print("  Windows automated pip fixes complete!")
    print(f"{SEP}")
    print("""
NEXT: Complete the remaining manual steps below
(open a new terminal for each conda command if needed):

  Step 4 — R + SCEVAN (run in order inside scart_env):

    4a. conda install -c conda-forge r-base r-devtools r-remotes \\
            r-ggplot2 r-data.table r-igraph r-gdtools r-ragg r-dplyr \\
            cairo freetype fontconfig harfbuzz fribidi \\
            libpng libtiff libjpeg-turbo libwebp \\
            --override-channels -c conda-forge -c bioconda -c defaults -y

    4b. conda config --set channel_priority flexible
        conda install -c conda-forge r-paralleldist r-pheatmap r-forcats \\
            r-cluster r-rtsne r-ape r-tidytree r-ggrepel -y

    4c. Rscript -e "install.packages('BiocManager', repos='https://cran.r-project.org')"
        Rscript -e "BiocManager::install(c('scran', 'fgsea', 'ggtree'))"

    4d. Rscript -e "library(devtools); install_github('miccec/yaGST'); install_github('AntonioDeFalco/SCEVAN')"

  Step 5 — Verify R/SCEVAN:
        Rscript -e "library(SCEVAN); cat('SCEVAN OK')"

  NOTE: On Windows use single-line commands in PowerShell.
        Backslash (\\) line continuation does NOT work in PowerShell.
""")


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

def _ask(prompt: str, valid: list) -> str:
    """Keep asking until the user gives a valid answer."""
    while True:
        ans = input(prompt).strip().lower()
        if ans in valid:
            return ans
        print(f"  Invalid input. Please enter one of: {', '.join(valid)}")


def _show_manual_steps(os_choice: str):
    """Print the manual pre-requisites for the chosen OS before confirming."""

    if os_choice == "1":   # Linux
        print(f"""
{SEP2}
 MANUAL STEPS REQUIRED — Linux
 (do these BEFORE or AFTER as indicated)
{SEP2}

 BEFORE running this script:
   [1] Create and activate conda environment (if not done yet):
         conda create -n scart_env python=3.10 -y
         conda activate scart_env

   [2] Install SCART (if not done yet):
         pip install git+https://github.com/csblab23/SCART.git

 AUTOMATED by this script (no action needed):
   [auto] Download cl.obo ontology
   [auto] conda install R base + graphics stack
   [auto] conda install Bioconductor + CRAN packages
   [auto] Rscript install_github SCEVAN
   [auto] Verify Python packages
   [auto] Verify SCEVAN

 NOTHING extra required on Linux — all steps are automated.
{SEP2}""")

    elif os_choice == "2":  # Windows
        print(f"""
{SEP2}
 MANUAL STEPS REQUIRED — Windows
 (do these BEFORE or AFTER as indicated)
{SEP2}

 BEFORE running this script:
   [1] Install Microsoft Visual C++ Redistributable
       (required for TensorFlow + PyTorch — needs admin rights)

       In PowerShell:
         Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile "vc_redist.exe"
         .\\vc_redist.exe /install /quiet /norestart
       Then RESTART your terminal.

   [2] Create and activate conda environment (if not done yet):
         conda create -n scart_env python=3.10 -y
         conda activate scart_env

   [3] Pre-install JAX-stack stubs (Windows only, before SCART):
         pip install "orbax-checkpoint<0.5" "flax<0.8"
         pip install scvi-tools==1.1.6.post2
         pip install "jax[cpu]==0.4.23" "jaxlib==0.4.23"

   [4] Install python-annoy via conda (no Windows wheel on PyPI):
         conda install -c conda-forge python-annoy -y

   [5] Install SCART (if not done yet):
         pip install git+https://github.com/csblab23/SCART.git

 AUTOMATED by this script (no action needed):
   [auto] Download cl.obo ontology
   [auto] Re-pin numpy + scipy
   [auto] Re-pin full JAX stack (force-reinstall)
   [auto] Fix PyTorch DLL issue (reinstall CPU wheel)
   [auto] Pin TensorFlow CPU to 2.10.0
   [auto] Verify Python packages

 AFTER this script — manual R + SCEVAN steps (printed at end):
   [manual] conda install R base + graphics stack
   [manual] conda install CRAN packages
   [manual] Rscript BiocManager install scran/fgsea/ggtree
   [manual] Rscript install_github SCEVAN
{SEP2}""")

    elif os_choice == "3":  # Mac
        print(f"""
{SEP2}
 MANUAL STEPS REQUIRED — Mac
 (do these BEFORE or AFTER as indicated)
{SEP2}

 BEFORE running this script:
   [1] Create and activate conda environment (if not done yet):
         conda create -n scart_env python=3.10 -y
         conda activate scart_env

   [2] Install SCART (if not done yet):
         pip install git+https://github.com/csblab23/SCART.git

 AUTOMATED by this script (no action needed):
   [auto] Download cl.obo ontology
   [auto] conda install R base + graphics stack
   [auto] conda install Bioconductor + CRAN packages
   [auto] Rscript install_github SCEVAN
   [auto] Verify Python packages
   [auto] Verify SCEVAN

 NOTHING extra required on Mac — all steps are automated.
{SEP2}""")


def main():
    print(f"""
{SEP}
  SCART Post-Installation Setup
{SEP}

This script will set up all dependencies for SCART.
It will ask for your OS, show you exactly what will run,
and ask for confirmation before doing anything.

Select your operating system:

  [1]  Linux
  [2]  Windows
  [3]  Mac

""")

    os_choice = _ask("Enter choice (1 / 2 / 3): ", ["1", "2", "3"])

    os_names = {"1": "Linux", "2": "Windows", "3": "Mac"}
    os_name  = os_names[os_choice]
    print(f"\n  You selected: {os_name}")

    # Show manual steps summary for chosen OS
    _show_manual_steps(os_choice)

    confirm = _ask(
        f"Proceed with automated setup for {os_name}? [y/n]: ",
        ["y", "n"],
    )

    if confirm == "n":
        print("\n  Setup cancelled. Run again when ready: python -m SCART.install\n")
        sys.exit(0)

    # Dispatch
    if os_choice == "1":
        _run_linux_mac("Linux")
    elif os_choice == "2":
        _run_windows()
    elif os_choice == "3":
        _run_linux_mac("Mac")


if __name__ == "__main__":
    main()
