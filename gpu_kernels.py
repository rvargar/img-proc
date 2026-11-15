"""Loads and caches the CUDA kernels in cuda/ (compiled on first use)."""
import os

import pycuda.autoinit  # noqa: F401  (creates the CUDA context)
from pycuda.compiler import SourceModule

CUDA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda")
BLOCK = (32, 32, 1)

_modules = {}


def get_function(cu_file, name):
    """Returns kernel `name` from cuda/<cu_file>, compiling the file the first time."""
    if cu_file not in _modules:
        with open(os.path.join(CUDA_DIR, cu_file)) as f:
            _modules[cu_file] = SourceModule(f.read(), arch='sm_80')
    return _modules[cu_file].get_function(name)


def grid_for(w, h):
    """Grid size that covers a w x h image with BLOCK-sized thread blocks."""
    return (w + BLOCK[0] - 1) // BLOCK[0], (h + BLOCK[1] - 1) // BLOCK[1]
