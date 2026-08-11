#!/usr/bin/env python
"""
windows_bootstrap.py
=====================
One-shot Windows setup for SCART.

Standalone script — has ZERO dependency on the SCART package itself (stdlib
only), because it has to run BEFORE `pip install SCART` even exists in this
environment. Collapses the old manual Windows flow (Steps 2a-2f, plus
Step 3 R/SCEVAN/RobustRankAggreg, plus Step 5 verification — everything
after "create + activate the conda env") into a single command.

Usage
-----
    conda create -n scart_env python=3.10 -y
    conda activate scart_env
    python windows_bootstrap.py

That's the whole install, apart from the one thing that genuinely can't be
scripted: the Microsoft Visual C++ Redistributable, which needs an admin-
rights GUI installer. This script reminds you about it up front and exits
early if it looks like torch can't load because of it.

What this script does, in order
--------------------------------
  1. Checks you're on Windows and that conda is on PATH.
  2. Pre-pins numpy/scipy/the full JAX stack (old Step 2a) — must happen
     BEFORE pip install SCART or pip's resolver fights itself.
  3. Installs annoy via conda and registers it with pip (old Step 2b) —
     done as real Python function calls, not a copy-pasted `python -c`
     one-liner, so there is no risk of the cmd.exe multi-line-quoting
     failure mode (cmd.exe treats each line of a multi-line quoted string
     as a separate command — this is what silently broke the manual
     instructions when pasted into Anaconda Prompt / cmd.exe).
  4. pip installs SCART itself (old Step 2c).
  5. Runs `python -m SCART.install --os windows --yes` (old Steps
     2d/2e/2f + Step 3 R/SCEVAN/RobustRankAggreg + Step 5 verification —
     all of it, now that SCART.install supports a non-interactive
     --os/--yes mode).

Flags
-----
  --stop-before-install   Only do Steps 1-3 (everything that MUST happen
                           before pip install SCART) and stop there, if
                           you'd rather run steps 4-5 yourself.
  --skip-annoy            Skip the annoy step (only if you've already
                           handled it, e.g. re-running after a partial
                           failure downstream).
"""

import argparse
import os
import shutil
import subprocess
import sys


SEP  = "=" * 62
SEP2 = "-" * 62

PINNED_CORE  = ["numpy>=1.24,<2.0", "scipy==1.12.0"]
PINNED_FLAX  = ["orbax-checkpoint<0.5", "flax<0.8"]
PINNED_SCVI  = ["scvi-tools==1.1.6.post2"]
PINNED_JAX_STACK = [
    "jax[cpu]==0.4.23", "jaxlib==0.4.23", "optax==0.1.7", "flax==0.7.5",
    "orbax-checkpoint<0.5", "numpyro<=0.13.2",
    "numpy>=1.24,<2.0", "scipy==1.12.0",
]

ANNOY_VERSION = "1.17.3"


# ---------------------------------------------------------------------------
# Subprocess helpers (self-contained — same pattern as SCART/install.py,
# duplicated here on purpose since this script cannot import SCART)
# ---------------------------------------------------------------------------

def _run(cmd: list, label: str = "") -> bool:
    tag = f"[bootstrap:{label}]" if label else "[bootstrap]"
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
    return _run([sys.executable, "-m", "pip", "install", *args], label="pip")


def _conda(*args) -> bool:
    conda_exe = shutil.which("conda")
    if not conda_exe:
        print(
            "\n[bootstrap:conda] ERROR: conda not found on PATH.\n"
            "  Make sure your conda environment is activated:\n"
            "    conda activate scart_env\n"
            "  Then re-run: python windows_bootstrap.py",
            file=sys.stderr,
        )
        return False
    return _run([conda_exe, *args], label="conda")


# ---------------------------------------------------------------------------
# Step 0 — sanity checks
# ---------------------------------------------------------------------------

def _check_platform_and_conda() -> bool:
    if sys.platform != "win32":
        print(
            "[bootstrap] This script is Windows-specific (pip has no annoy\n"
            "  wheel for Windows, which is most of what it works around).\n"
            "  On Linux/Mac, just run: python -m SCART.install --os linux --yes\n"
            "  (or --os mac) after `pip install git+https://github.com/csblab23/SCART.git`.",
            file=sys.stderr,
        )
        return False

    if not shutil.which("conda"):
        print(
            "[bootstrap] ERROR: conda not found on PATH.\n"
            "  Run this INSIDE an activated conda environment:\n"
            "    conda create -n scart_env python=3.10 -y\n"
            "    conda activate scart_env\n"
            "    python windows_bootstrap.py",
            file=sys.stderr,
        )
        return False

    print(
        "\nREMINDER — one thing this script cannot do for you:\n"
        "  Microsoft Visual C++ Redistributable (needed by PyTorch/TensorFlow)\n"
        "  requires an admin-rights GUI installer. If you haven't already:\n"
        "    1. Download: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
        "    2. Double-click it, click through the installer\n"
        "    3. Restart your terminal, then re-run this script\n"
        "  If it's already installed, ignore this and read on.\n"
    )
    return True


# ---------------------------------------------------------------------------
# Step 1 — pre-pin numpy/scipy/JAX stack  (old Step 2a)
# ---------------------------------------------------------------------------

def _prepin_python_deps() -> bool:
    print(f"\n{SEP}\n  Step 1/5 — Pre-pinning numpy / scipy / JAX stack\n{SEP}")
    ok = True
    ok &= _pip(*PINNED_CORE)
    ok &= _pip(*PINNED_FLAX)
    ok &= _pip(*PINNED_SCVI)
    ok &= _pip(*PINNED_JAX_STACK, "--force-reinstall")
    return ok


# ---------------------------------------------------------------------------
# Step 2 — annoy: conda install + pip dist-info registration  (old Step 2b)
# ---------------------------------------------------------------------------

def _register_annoy_with_pip() -> bool:
    """
    Create a minimal pip dist-info record for the conda-installed annoy, so
    pip treats it as already satisfied and skips the source build (which
    needs a C++ compiler + Build Tools) when resolving SCART's deps.
    """
    try:
        import site
        for sp in site.getsitepackages():
            dist_info = os.path.join(sp, f"annoy-{ANNOY_VERSION}.dist-info")
            try:
                os.makedirs(dist_info, exist_ok=True)
                with open(os.path.join(dist_info, "METADATA"), "w") as f:
                    f.write(
                        f"Metadata-Version: 2.1\n"
                        f"Name: annoy\n"
                        f"Version: {ANNOY_VERSION}\n"
                    )
                with open(os.path.join(dist_info, "RECORD"), "w") as f:
                    f.write("")
                with open(os.path.join(dist_info, "INSTALLER"), "w") as f:
                    f.write("conda\n")
                print(f"[bootstrap:annoy] Registered annoy {ANNOY_VERSION} with pip at:\n  {dist_info}")
                return True
            except OSError:
                continue
    except Exception as exc:
        print(f"[bootstrap:annoy] WARNING: could not register annoy with pip: {exc}", file=sys.stderr)
    return False


def _install_annoy() -> bool:
    print(f"\n{SEP}\n  Step 2/5 — Installing annoy (no C++ compiler needed)\n{SEP}")

    check = subprocess.run(
        [sys.executable, "-m", "pip", "show", "annoy"],
        capture_output=True, check=False,
    )
    if check.returncode == 0:
        print("[bootstrap:annoy] annoy already pip-visible — OK")
        return True

    try:
        import importlib
        importlib.import_module("annoy")
        print("[bootstrap:annoy] annoy importable but not pip-registered. Registering ...")
        return _register_annoy_with_pip()
    except ImportError:
        pass

    ok = _conda("install", "-c", "conda-forge", "python-annoy", "-y")
    if not ok:
        print(
            "[bootstrap:annoy] ERROR: conda install failed. Run manually:\n"
            "  conda install -c conda-forge python-annoy -y\n"
            "  then re-run this script.",
            file=sys.stderr,
        )
        return False

    _register_annoy_with_pip()

    verify = subprocess.run(
        [sys.executable, "-c", "import annoy; print('annoy OK')"],
        check=False,
    )
    if verify.returncode == 0:
        print("[bootstrap:annoy] annoy installed and verified — OK")
        return True

    print(
        "[bootstrap:annoy] ERROR: annoy installed but import failed.\n"
        "  Try restarting your terminal and re-running this script.",
        file=sys.stderr,
    )
    return False


# ---------------------------------------------------------------------------
# Step 3 — pip install SCART itself  (old Step 2c)
# ---------------------------------------------------------------------------

def _install_scart() -> bool:
    print(f"\n{SEP}\n  Step 3/5 — Installing SCART\n{SEP}")
    return _run(
        [sys.executable, "-m", "pip", "install",
         "git+https://github.com/csblab23/SCART.git"],
        label="pip",
    )


# ---------------------------------------------------------------------------
# Step 4 — hand off to SCART's own installer, non-interactively
# (old Steps 2d/2e/2f + Step 3 R/SCEVAN/RobustRankAggreg + Step 5 verify)
# ---------------------------------------------------------------------------

def _run_scart_install() -> bool:
    print(f"\n{SEP}\n  Step 4/5 — Running SCART's own installer "
          f"(torch fix, TensorFlow pin, R + SCEVAN + RobustRankAggreg, verify)\n{SEP}")
    return _run(
        [sys.executable, "-m", "SCART.install", "--os", "windows", "--yes"],
        label="SCART.install",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="One-shot Windows setup for SCART — run inside an "
                     "activated scart_env, before pip install SCART.",
    )
    parser.add_argument(
        "--stop-before-install", action="store_true",
        help="Only do the pre-pin + annoy steps (must happen before "
             "pip install SCART); stop there instead of continuing.",
    )
    parser.add_argument(
        "--skip-annoy", action="store_true",
        help="Skip the annoy step (only if already handled).",
    )
    args = parser.parse_args()

    print(f"\n{SEP}\n  SCART Windows Bootstrap\n{SEP}")

    if not _check_platform_and_conda():
        sys.exit(1)

    if not _prepin_python_deps():
        print(
            "\n[bootstrap] Step 1 had failures — fix the error(s) above and "
            "re-run this script before continuing.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.skip_annoy:
        if not _install_annoy():
            print(
                "\n[bootstrap] Step 2 (annoy) failed — fix the error above and "
                "re-run. Do not skip this: pip will otherwise try to build "
                "annoy from source in Step 3.",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.stop_before_install:
        print(f"\n{SEP}\n  --stop-before-install given — steps 1-2 complete.\n"
              f"  Continue yourself with:\n"
              f"    pip install git+https://github.com/csblab23/SCART.git\n"
              f"    python -m SCART.install --os windows --yes\n{SEP}")
        return

    if not _install_scart():
        print(
            "\n[bootstrap] Step 3 (pip install SCART) failed — see the error "
            "above. Fix it, then re-run with --stop-before-install already "
            "done, or just re-run this whole script (steps 1-2 are safe to "
            "repeat).",
            file=sys.stderr,
        )
        sys.exit(1)

    _run_scart_install()

    print(f"\n{SEP}\n  SCART Windows bootstrap complete!\n"
          f"  Check the verification output above for anything that needs "
          f"attention (e.g. SCEVAN, which Windows doesn't auto-install —\n"
          f"  see the R Dependencies section of the install guide if so).\n{SEP}")


if __name__ == "__main__":
    main()
