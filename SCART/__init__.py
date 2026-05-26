# from .geo_fetcher import SampleAnnotator

# __all__ = ["SampleAnnotator"]

# ---------------------------------------------------------------------------
# Triton segfault guard — must be first, before any other import
#
# triton >= 3.x segfaults on CPU-only machines (no CUDA drivers) during
# torch._dynamo / transformers initialisation (illegal CPU instruction in
# triton/knobs.py on AVX-512 Xeon hosts without a GPU).
#
# This stub intercepts every triton submodule import before Python's import
# machinery tries to load the real .so, preventing the kernel crash.
#
# Safe to do: triton is only needed for GPU-accelerated torch.compile.
# SCART's pipeline is CPU-only and never calls torch.compile.
# ---------------------------------------------------------------------------
import sys as _sys


class _TritonStub:
    """
    Minimal no-op replacement for the triton package.
    Satisfies attribute lookups and call expressions so that any code
    that does `import triton; triton.something()` silently gets a no-op
    instead of a segfault.
    """
    __version__ = "0.0.0+stub"

    def __repr__(self):
        return "<triton stub — CPU-only host, triton disabled>"

    def __getattr__(self, name):
        # Return self for chained attribute access (triton.runtime.autotuner …)
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __iter__(self):
        return iter([])

    def __bool__(self):
        return False


_stub = _TritonStub()

# Remove any partially-loaded triton modules that may already be in sys.modules
# (e.g. if a previous import attempt started before the guard was in place).
for _mod_name in list(_sys.modules.keys()):
    if _mod_name == "triton" or _mod_name.startswith("triton."):
        del _sys.modules[_mod_name]

# Register the stub for every triton submodule that the crash trace references.
# Additional submodules can be added here if new triton versions add new paths.
for _mod_name in [
    "triton",
    "triton.knobs",
    "triton.language",
    "triton.runtime",
    "triton.runtime.autotuner",
    "triton.runtime.cache",
    "triton.runtime.driver",
    "triton.compiler",
    "triton.compiler.compiler",
    "triton.backends",
    "triton.backends.compiler",
    "triton.tools",
    "triton.tools.disasm",
]:
    _sys.modules[_mod_name] = _stub

# Clean up — don't leak stub or helper names into the SCART namespace
del _mod_name, _stub, _TritonStub, _sys
# ---------------------------------------------------------------------------

from .geo_fetcher import SampleAnnotator

__all__ = ["SampleAnnotator"]
