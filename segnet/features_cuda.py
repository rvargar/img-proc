"""CUDA backend of the feature front-ends: the kernels in cuda/ run directly on PyTorch tensors.

PyCUDA attaches to PyTorch's (primary) CUDA context instead of creating its own, so a kernel
can read and write tensor memory through raw pointers: no host round trip. Same outputs as the
pure-PyTorch modules in features.py, several times faster (see segnet/check_features.py).

Do not import gpu_kernels / pycuda.autoinit in the same process: they create a second context.
"""
import atexit
import math
import os

import numpy as np
import torch
import torch.nn as nn

from structuring_elements import flat_bank

CUDA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cuda")
BLOCK = (16, 16, 1)

_state = {}


def _kernel(cu_file, name):
    """Kernel `name` of cuda/<cu_file>, compiled once inside PyTorch's CUDA context."""
    if 'ctx' not in _state:
        import pycuda.driver as drv
        torch.cuda.init()
        torch.zeros(1, device='cuda')                 # make sure the primary context exists
        drv.init()
        ctx = drv.Device(torch.cuda.current_device()).retain_primary_context()
        ctx.push()
        atexit.register(ctx.pop)
        _state['ctx'] = ctx
    if cu_file not in _state:
        from pycuda.compiler import SourceModule
        with open(os.path.join(CUDA_DIR, cu_file)) as f:
            _state[cu_file] = SourceModule(f.read())
    return _state[cu_file].get_function(name)


def _ptr(tensor, offset=0):
    """Device pointer to element `offset` of a contiguous CUDA tensor, as a kernel argument."""
    return np.uintp(tensor.data_ptr() + offset * tensor.element_size())


def _grid(w, h, depth=1):
    return (w + BLOCK[0] - 1) // BLOCK[0], (h + BLOCK[1] - 1) // BLOCK[1], depth


class RegulatedMorphFeaturesCuda(nn.Module):
    """Same as features.RegulatedMorphFeatures (and the flat part of MorphFeatureMap),
    computed by quantileBank in cuda/segnet_features.cu: the whole bank in one launch.
    Each channel is filtered on its own. Input (B, C, H, W) float in [0, 1] on the GPU."""

    def __init__(self, elements=None, quantiles=(0.1, 0.5, 0.9), include_range=True):
        super().__init__()
        self.elements = flat_bank() if elements is None else list(elements)
        self.quantiles = tuple(sorted(quantiles))
        self.include_range = include_range and len(self.quantiles) > 1
        self.planes_per_channel = len(self.elements) * (len(self.quantiles) + int(self.include_range))
        if any(e.size > 255 for e in self.elements):
            raise ValueError("the CUDA backend supports elements of at most 255 pixels")
        offsets = np.concatenate([e.offsets for e in self.elements])
        starts = np.cumsum([0] + [e.size for e in self.elements])
        ranks = [[1 + int(round(q * (e.size - 1))) for q in self.quantiles] for e in self.elements]  # 1-based
        self.register_buffer("offsets", torch.from_numpy(offsets.astype(np.int32)), persistent=False)
        self.register_buffer("starts", torch.tensor(starts, dtype=torch.int32), persistent=False)
        self.register_buffer("ranks", torch.tensor(ranks, dtype=torch.int32), persistent=False)

    @torch.no_grad()
    def forward(self, x):
        B, C, H, W = x.shape
        E, nq = len(self.elements), len(self.quantiles)
        q8 = (x * 255.0).round().clamp_(0, 255).to(torch.uint8).contiguous()
        out = torch.empty((B, C, E, nq, H, W), dtype=torch.float32, device=x.device)
        _kernel("segnet_features.cu", "quantileBank")(
            _ptr(out), _ptr(q8), _ptr(self.offsets), _ptr(self.starts), np.int32(E),
            _ptr(self.ranks), np.int32(nq), np.int32(W), np.int32(H),
            block=BLOCK, grid=_grid(W, H, B * C))
        out = out.permute(0, 2, 3, 1, 4, 5)                  # (B, E, nq, C, H, W)
        if self.include_range:
            out = torch.cat([out, out[:, :, -1:] - out[:, :, :1]], dim=2)
        return out.reshape(B, -1, H, W)


class SymmetryFeaturesCuda(nn.Module):
    """Same as features.SymmetryFeatures, computed by symmetryFeatures in cuda/segnet_features.cu.
    Input (B, 1, H, W) grey in [0, 1] on the GPU."""

    def __init__(self, angles_deg=(0.0, 45.0, 90.0), half_widths=(4, 12, 32)):
        super().__init__()
        self.angles_deg = tuple(angles_deg)
        self.half_widths = tuple(sorted(int(k) for k in half_widths))
        self.n_planes = 2 * len(self.angles_deg) * len(self.half_widths)
        dirs = []
        for a in self.angles_deg:                            # rotate((0, 1), angle)
            dx, dy = math.sin(math.radians(a)), math.cos(math.radians(a))
            dirs.append((0.0 if abs(dx) < 1e-6 else dx, 0.0 if abs(dy) < 1e-6 else dy))
        self.register_buffer("dirs", torch.tensor(dirs, dtype=torch.float32), persistent=False)
        self.register_buffer("hw", torch.tensor(self.half_widths, dtype=torch.int32), persistent=False)

    @torch.no_grad()
    def forward(self, grey):
        B, _, H, W = grey.shape
        grey = grey.float().contiguous()
        out = torch.empty((B, self.n_planes, H, W), dtype=torch.float32, device=grey.device)
        _kernel("segnet_features.cu", "symmetryFeatures")(
            _ptr(out), _ptr(grey), _ptr(self.dirs), np.int32(len(self.angles_deg)),
            _ptr(self.hw), np.int32(len(self.half_widths)), np.int32(W), np.int32(H),
            block=BLOCK, grid=_grid(W, H, B))
        return out


class DiffFilterBankCuda(nn.Module):
    """Bank of difference filters (segnet/diff_bank.py), computed by generalFilter in
    cuda/diff_filter.cu. Input (B, 1, H, W) grey in [0, 1] on the GPU; output (B, n, H, W)."""

    def __init__(self, bank='full'):
        super().__init__()
        from diff_filter import ALPHA, BETA, GAMMA, STAT
        from .diff_bank import diff_bank
        self.names_, self.calls = [], []
        for i, (name, offsets, kw, divisor) in enumerate(diff_bank(bank)):
            modulus = kw.get('modulus', 1)
            pairing = kw.get('pairing', 'neighbour')
            step = {'neighbour': 1, 'same_class': modulus, 'none': 0}.get(pairing, pairing)
            self.register_buffer(f"offsets{i}", torch.from_numpy(offsets), persistent=False)
            self.names_.append(name)
            self.calls.append((len(offsets), [modulus, kw.get('mod_offset', 1), kw.get('mod_count', modulus),
                                              step, kw.get('pair_stride', 1), GAMMA[kw.get('gamma', 'diff')],
                                              BETA[kw.get('beta', 'diff')], STAT[kw.get('stat', 'pooled')],
                                              ALPHA[kw.get('alpha', 'max')]], divisor))
        self.n_planes = len(self.calls)
        self.register_buffer("scale", torch.tensor([1.0 / c[2] for c in self.calls]).view(1, -1, 1, 1),
                             persistent=False)

    def names(self):
        return list(self.names_)

    @torch.no_grad()
    def forward(self, grey):
        B, _, H, W = grey.shape
        grey = grey.float().contiguous()
        kernel = _kernel("diff_filter.cu", "generalFilter")
        # filter-major, so that one launch (grid.z = batch) fills one block of B planes
        out = torch.empty((self.n_planes, B, H, W), dtype=torch.float32, device=grey.device)
        for i, (n_points, params, _) in enumerate(self.calls):
            kernel(_ptr(out, i * B * H * W), _ptr(grey), _ptr(getattr(self, f"offsets{i}")), np.int32(n_points),
                   *[np.int32(p) for p in params], np.int32(W), np.int32(H),
                   block=BLOCK, grid=_grid(W, H, B))
        return out.permute(1, 0, 2, 3) * self.scale
