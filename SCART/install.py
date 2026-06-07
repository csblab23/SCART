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

# Linux / Mac: Step 3a — R base + graphics libraries
CONDA_R_BASE = [
    "r-base", "r-devtools", "r-remotes",
    "r-ggplot2", "r-data.table", "r-igraph",
    "r-gdtools", "r-ragg", "r-dplyr",
    "cairo", "freetype", "fontconfig",
    "harfbuzz", "fribidi", "libpng",
    "libtiff", "libjpeg-turbo", "libwebp",
]

# Linux / Mac: Step 3b — Bioconductor + CRAN (available via conda on Linux/Mac only)
# NOTE: On Windows, bioconductor-scran/fgsea/ggtree are NOT available for
#       win-64 via conda — must use BiocManager/Rscript instead.
CONDA_R_BIO = [
    "bioconductor-scran", "bioconductor-fgsea", "bioconductor-ggtree",
    "r-paralleldist", "r-pheatmap", "r-forcats",
    "r-cluster", "r-rtsne", "r-ape", "r-tidytree", "r-ggrepel",
]

# Windows pip fixes (Step 3 of Windows guide — runs AFTER pip install SCART)
WIN_TORCH_UNINSTALL = ["torch"]
WIN_TORCH_INSTALL   = ["torch==2.2.2", "--index-url", "https://download.pytorch.org/whl/cpu"]
WIN_TF_UNINSTALL    = ["tensorflow", "tensorflow-cpu", "tensorflow-intel"]
WIN_TF_INSTALL      = ["tensorflow-cpu==2.10.0"]


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _run(cmd: list, label: str = "") -> bool:
    tag = f"[SCART:{label}]" if label else "[SCART]"
    print(f"\n{tag} Running:\n  {chr(32).join(cmd)}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            f"{tag} WARNING: command exited with code {result.returncode}.\n"
            f"  If this is critical, run it manually:\n"
            f"    {chr(32).join(cmd)}",
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
            "\n[SCART:conda] ERROR: conda not found on PATH.\n"
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
            "\n[SCART:R] ERROR: Rscript not found on PATH.\n"
            "  R does not appear to be installed yet.\n"
            "  Install R first (Step 4a), then re-run: python -m SCART.install",
            file=sys.stderr,
        )
        return False
    return _run([rscript, "-e", r_code], label=label)


# ---------------------------------------------------------------------------
# Ontology download (all platforms)
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
            f"    wget \"{OBO_URL}\" -O \"{obo_path}\"",
            file=sys.stderr,
        )
        return False


# ---------------------------------------------------------------------------
# Verification (matches exactly the verify commands in the original guide)
# ---------------------------------------------------------------------------

def _verify_python():
    print(f"\n{SEP2}")
    print("Verifying Python packages ...")
    print(SEP2)
    # Matches: python -c "import jax; import flax; import scvi; import SCART; ..."
    _run(
        [
            sys.executable, "-c",
            (
                "import jax, flax, scvi, SCART; "
                "print('jax:', jax.__version__); "
                "import numpy; print('numpy:', numpy.__version__); "
                "print('scvi:', scvi.__version__)"
            ),
        ],
        label="verify",
    )
    # Matches: python -c "import SCART; from SCART.geo_fetcher import ..."
    _run(
        [
            sys.executable, "-c",
            (
                "import SCART; "
                "from SCART.geo_fetcher import SampleAnnotator; "
                "from SCART import popv_annotation, preprocessing; "
                "from SCART.gene_combination_predictor import "
                "one_gene_combination, two_gene_combination; "
                "print('All imports OK')"
            ),
        ],
        label="verify",
    )


def _verify_r():
    print(f"\n{SEP2}")
    print("Verifying R / SCEVAN ...")
    print(SEP2)
    # Matches: Rscript -e "library(SCEVAN); cat('SCEVAN OK\n')"
    _rscript("library(SCEVAN); cat('SCEVAN OK\\n')", label="verify")


# ---------------------------------------------------------------------------
# Linux / Mac: automated steps  (mirrors original Linux guide exactly)
# ---------------------------------------------------------------------------

def _run_linux_mac(os_name: str):
    print(f"\n{SEP}")
    print(f"  Running automated steps for {os_name}")
    print(SEP)

    # Step 1: ontology
    print("\n[Step 1/5] Downloading cl.obo ontology ...")
    download_cl_obo()

    # Step 2: r-base alone first  (mirrors guide Step 4a)
    print("\n[Step 2/5] Installing r-base via conda ...")
    ok = _conda("install", "-c", "conda-forge", "r-base", "-y")
    if not ok:
        print("[SCART] Step 2 failed. Fix the conda issue above and re-run.", file=sys.stderr)
        return

    # Step 3: full R base + graphics stack  (mirrors guide Step 4b)
    print("\n[Step 3/5] Installing R base packages + graphics stack via conda ...")
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
        print("[SCART] Step 3 failed. Fix the conda issue above and re-run.", file=sys.stderr)
        return

    # Step 4: flexible channel priority + Bioconductor + CRAN  (mirrors guide Step 4c)
    print("\n[Step 4/5] Setting flexible channel priority ...")
    _conda("config", "--set", "channel_priority", "flexible")

    print("\n[Step 4/5] Installing Bioconductor + CRAN packages via conda ...")
    ok = _conda(
        "install",
        "-c", "conda-forge",
        "-c", "bioconda",
        "-y",
        *CONDA_R_BIO,
    )
    if not ok:
        print("[SCART] Step 4 failed. Fix the conda issue above and re-run.", file=sys.stderr)
        return

    # Step 5: SCEVAN via Rscript  (mirrors guide Step 4e)
    print("\n[Step 5/5] Installing SCEVAN via Rscript ...")
    ok = _rscript(
        "library(devtools); "
        "install_github('miccec/yaGST'); "
        "install_github('AntonioDeFalco/SCEVAN')"
    )
    if not ok:
        print("[SCART] SCEVAN install failed. See error above.", file=sys.stderr)
        return

    _verify_python()
    _verify_r()

    print(f"\n{SEP}")
    print(f"  {os_name} setup complete!")
    print(SEP)


# ---------------------------------------------------------------------------
# Windows: annoy pre-install / repair helper
# ---------------------------------------------------------------------------

def _ensure_annoy_windows() -> bool:
    """
    Ensure annoy is installed from a pre-built binary wheel (no C++ compiler needed).

    Why this matters
    ----------------
    annoy is a transitive dependency (SCART → popv → annoy).  When pip tries to
    install annoy it builds from source, which requires Microsoft C++ Build Tools
    (NOT just the VC++ Redistributable).  On machines without Build Tools, the
    build fails with "Microsoft Visual C++ 14.0 or greater is required."

    Fix: annoy ≥ 1.17.0 ships pre-built win_amd64 wheels on PyPI — no compiler
    needed.  We pin ≥ 1.17.0 and use --prefer-binary so pip always grabs the wheel.

    Strategy (tried in order)
    -------------------------
    1. pip install annoy>=1.17.0 --prefer-binary   ← uses PyPI win_amd64 wheel
    2. conda install -c conda-forge python-annoy   ← fallback, also pre-built

    Call this BEFORE pip install SCART so pip sees annoy as already satisfied
    and does not attempt a source build.
    """
    # Already importable → nothing to do
    try:
        import importlib
        importlib.import_module("annoy")
        print("[SCART:annoy] annoy already installed — OK")
        return True
    except ImportError:
        pass

    print(
        "\n[SCART:annoy] annoy not found (or not pip-visible).\n"
        "  Installing pre-built binary — no C++ compiler required ..."
    )

    # ── Try 1: pip binary wheel ──────────────────────────────────────────────
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "annoy>=1.17.0", "--prefer-binary"],
        check=False,
    )
    if result.returncode == 0:
        # Verify the import works in a fresh subprocess (avoids stale sys.modules)
        check = subprocess.run(
            [sys.executable, "-c", "import annoy; print('annoy', annoy.__version__)"],
            check=False,
        )
        if check.returncode == 0:
            print("[SCART:annoy] annoy installed via pip binary wheel — OK")
            return True

    print(
        "[SCART:annoy] pip binary wheel unavailable for this Python/platform.\n"
        "  Falling back to conda-forge ...",
        file=sys.stderr,
    )

    # ── Try 2: conda-forge ──────────────────────────────────────────────────
    conda_exe = shutil.which("conda")
    if conda_exe:
        result = subprocess.run(
            [conda_exe, "install", "-c", "conda-forge", "python-annoy", "-y"],
            check=False,
        )
        if result.returncode == 0:
            print("[SCART:annoy] annoy installed via conda-forge — OK")
            return True

    # ── Both failed ──────────────────────────────────────────────────────────
    print(
        "\n[SCART:annoy] ERROR: Could not install annoy automatically.\n"
        "  Run ONE of these manually, then retry:\n"
        "\n"
        "    pip install \"annoy>=1.17.0\" --prefer-binary\n"
        "\n"
        "  Or:\n"
        "\n"
        "    conda install -c conda-forge python-annoy -y\n"
        "\n"
        "  If both fail, install Microsoft C++ Build Tools from:\n"
        "    https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
        "  (choose 'Desktop development with C++', then restart your terminal)",
        file=sys.stderr,
    )
    return False


# ---------------------------------------------------------------------------
# Windows: automated pip fixes + R + SCEVAN
# ---------------------------------------------------------------------------

def _run_windows():
    print(f"\n{SEP}")
    print("  Running automated setup for Windows")
    print(SEP)

    # Step 0: ensure annoy binary wheel is installed (no C++ compiler needed)
    print("\n[Step 0/11] Ensuring annoy binary wheel is installed ...")
    _ensure_annoy_windows()

    # Step 1: ontology
    print("\n[Step 1/11] Downloading cl.obo ontology ...")
    download_cl_obo()

    # Step 2: re-pin numpy + scipy
    print("\n[Step 2/11] Re-pinning numpy + scipy ...")
    _pip("numpy>=1.24,<2.0", "scipy==1.12.0", "--force-reinstall")

    # Step 3: re-pin full JAX stack
    print("\n[Step 3/11] Re-pinning full JAX stack ...")
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

    # Step 4: fix PyTorch DLL issue
    print("\n[Step 4/11] Fixing PyTorch DLL issue (CPU wheel) ...")
    _pip_uninstall(*WIN_TORCH_UNINSTALL)
    _pip(*WIN_TORCH_INSTALL)

    # Step 5: pin TensorFlow
    print("\n[Step 5/11] Pinning TensorFlow CPU to 2.10.0 ...")
    _pip_uninstall(*WIN_TF_UNINSTALL)
    _pip(*WIN_TF_INSTALL)

    # Step 6: re-ensure annoy after all the force-reinstalls above
    print("\n[Step 6/11] Re-checking annoy after force-reinstalls ...")
    _ensure_annoy_windows()

    _verify_python()

    # ── R + SCEVAN (automated, same pattern as Linux) ────────────────────────

    # Step 7: r-base first
    print("\n[Step 7/11] Installing r-base via conda ...")
    ok = _conda("install", "-c", "conda-forge", "r-base", "-y")
    if not ok:
        print("[SCART] Step 7 failed. Fix conda issue above and re-run.", file=sys.stderr)
        return

    # Step 8: R base + graphics stack
    print("\n[Step 8/11] Installing R base packages + graphics stack via conda ...")
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
        print("[SCART] Step 8 failed. Fix conda issue above and re-run.", file=sys.stderr)
        return

    # Step 9: CRAN packages (no bioconductor — not available for win-64 via conda)
    print("\n[Step 9/11] Installing R CRAN packages via conda ...")
    _conda("config", "--set", "channel_priority", "flexible")
    ok = _conda(
        "install",
        "-c", "conda-forge",
        "-y",
        "r-paralleldist", "r-pheatmap", "r-forcats",
        "r-cluster", "r-rtsne", "r-ape", "r-tidytree", "r-ggrepel",
    )
    if not ok:
        print("[SCART] Step 9 failed. Fix conda issue above and re-run.", file=sys.stderr)
        return

    # Step 10: Bioconductor packages via Rscript + BiocManager
    # (bioconductor-scran/fgsea/ggtree are NOT available for win-64 via conda)
    print("\n[Step 10/11] Installing Bioconductor packages via BiocManager ...")
    ok = _rscript(
        "install.packages('BiocManager', repos='https://cran.r-project.org'); "
        "BiocManager::install(c('scran', 'fgsea', 'ggtree'))",
        label="BiocManager",
    )
    if not ok:
        print("[SCART] Step 10 failed. See error above.", file=sys.stderr)
        return

    # Step 11: SCEVAN
    print("\n[Step 11/11] Installing SCEVAN via Rscript ...")
    ok = _rscript(
        "library(devtools); "
        "install_github('miccec/yaGST'); "
        "install_github('AntonioDeFalco/SCEVAN')",
        label="SCEVAN",
    )
    if not ok:
        print("[SCART] Step 11 failed. See error above.", file=sys.stderr)
        return

    _verify_r()

    print(f"\n{SEP}")
    print("  Windows setup complete!")
    print(SEP)


# ---------------------------------------------------------------------------
# Manual steps summary shown BEFORE confirmation prompt
# ---------------------------------------------------------------------------

def _show_manual_steps(os_choice: str):

    if os_choice == "1":   # Linux
        print(f"""
{SEP2}
 MANUAL STEPS REQUIRED — Linux
{SEP2}

 BEFORE running this script (if not done yet):
   1. Create and activate conda environment:
        conda create -n scart_env python=3.10 -y
        conda activate scart_env

   2. Install SCART:
        pip install git+https://github.com/csblab23/SCART.git

 AUTOMATED by this script:
   [auto] Download cl.obo ontology
   [auto] conda install r-base
   [auto] conda install R base packages + graphics stack
   [auto] conda install Bioconductor + CRAN packages
   [auto] Rscript install_github SCEVAN
   [auto] Verify Python packages
   [auto] Verify SCEVAN

 Nothing else required on Linux — everything above is automated.
{SEP2}""")

    elif os_choice == "2":  # Windows
        print(f"""
{SEP2}
 MANUAL STEPS REQUIRED — Windows
{SEP2}

 BEFORE running this script (in this exact order):

   [0] Install Microsoft Visual C++ Redistributable
       (required for TensorFlow + PyTorch — needs admin rights)
       In PowerShell:
         Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile "vc_redist.exe"
         .\\vc_redist.exe /install /quiet /norestart
       Then RESTART your terminal.

   [1] Create and activate conda environment:
         conda create -n scart_env python=3.10 -y
         conda activate scart_env

   [2] Pre-install Python dependencies (Windows only — must be BEFORE pip install SCART):
         pip install "numpy>=1.24,<2.0"
         pip install "scipy==1.12.0"
         pip install "orbax-checkpoint<0.5" "flax<0.8"
         pip install scvi-tools==1.1.6.post2
         pip install "jax[cpu]==0.4.23" "jaxlib==0.4.23" "optax==0.1.7" "flax==0.7.5" "orbax-checkpoint<0.5" "numpyro<=0.13.2" "numpy>=1.24,<2.0" "scipy==1.12.0" --force-reinstall

   [3] Install annoy binary wheel — NO C++ compiler needed
       (annoy 1.17+ ships pre-built win_amd64 wheels on PyPI):
         pip install "annoy>=1.17.0" --prefer-binary
       If that fails, use conda instead:
         conda install -c conda-forge python-annoy -y
       NOTE: Do NOT skip this step. pip will try to build annoy from source
       when installing SCART (via popv dependency), which requires C++ Build Tools.
       Pre-installing the binary wheel prevents that build attempt entirely.

   [4] Install SCART (annoy is now pre-installed — no source build triggered):
         pip install git+https://github.com/csblab23/SCART.git

 AUTOMATED by this script:
   [auto] Ensure annoy binary wheel (repair if needed)
   [auto] Download cl.obo ontology
   [auto] Re-pin numpy + scipy
   [auto] Re-pin full JAX stack (force-reinstall)
   [auto] Fix PyTorch DLL issue (CPU wheel)
   [auto] Pin TensorFlow CPU to 2.10.0
   [auto] Re-check annoy after force-reinstalls
   [auto] Verify Python packages
   [auto] conda install r-base
   [auto] conda install R base packages + graphics stack
   [auto] conda install R CRAN packages
   [auto] Rscript BiocManager install scran + fgsea + ggtree
   [auto] Rscript install_github SCEVAN
   [auto] Verify R / SCEVAN

 Nothing else required — everything above is automated.
{SEP2}""")

    elif os_choice == "3":  # Mac
        print(f"""
{SEP2}
 MANUAL STEPS REQUIRED — Mac
{SEP2}

 BEFORE running this script (if not done yet):
   1. Create and activate conda environment:
        conda create -n scart_env python=3.10 -y
        conda activate scart_env

   2. Install SCART:
        pip install git+https://github.com/csblab23/SCART.git

 AUTOMATED by this script:
   [auto] Download cl.obo ontology
   [auto] conda install r-base
   [auto] conda install R base packages + graphics stack
   [auto] conda install Bioconductor + CRAN packages
   [auto] Rscript install_github SCEVAN
   [auto] Verify Python packages
   [auto] Verify SCEVAN

 Nothing else required on Mac — everything above is automated.
{SEP2}""")


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

def _ask(prompt: str, valid: list) -> str:
    """Keep prompting until user gives a valid answer."""
    while True:
        ans = input(prompt).strip().lower()
        if ans in valid:
            return ans
        print(f"  Invalid input. Please enter one of: {chr(44).join(valid)}")


def main():
    print(f"""
{SEP}
  SCART Post-Installation Setup
{SEP}

This script sets up all dependencies for SCART.
It will show you exactly what runs automatically and
what you need to do manually, then ask for confirmation.

Select your operating system:

  [1]  Linux
  [2]  Windows
  [3]  Mac

""")

    os_choice = _ask("Enter choice (1 / 2 / 3): ", ["1", "2", "3"])
    os_names  = {"1": "Linux", "2": "Windows", "3": "Mac"}
    os_name   = os_names[os_choice]
    print(f"\n  You selected: {os_name}")

    _show_manual_steps(os_choice)

    confirm = _ask(
        f"Proceed with automated setup for {os_name}? [y/n]: ",
        ["y", "n"],
    )

    if confirm == "n":
        print("\n  Setup cancelled. Run again when ready: python -m SCART.install\n")
        sys.exit(0)

    if os_choice == "1":
        _run_linux_mac("Linux")
    elif os_choice == "2":
        _run_windows()
    elif os_choice == "3":
        _run_linux_mac("Mac")


if __name__ == "__main__":
    main()
