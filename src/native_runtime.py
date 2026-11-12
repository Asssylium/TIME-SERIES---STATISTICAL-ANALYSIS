"""Load native libraries bundled for the serverless Linux runtime."""

import ctypes
from pathlib import Path
import sys

_openmp_handle = None


def load_bundled_openmp() -> None:
    """Preload libgomp by absolute path before importing LightGBM."""
    global _openmp_handle
    if sys.platform != "linux" or _openmp_handle is not None:
        return
    library = Path(__file__).resolve().parents[1] / "native" / "libgomp.so.1"
    if library.is_file():
        _openmp_handle = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
