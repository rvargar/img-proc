"""Checks LearnableDiffFilter against generalFilter (cuda/diff_filter.cu).

    python -m segnet.check_learnable_diff

With 4 directions and integer radii every sampling point is a pixel, and with hard_limit()
the soft pooling is a hard max / min, so the layer at its initial setting must reproduce the
kernel for the presets without pairs. (Presets with pairs differ by design: the layer wraps
pairs around the circle, the kernel pairs along the point sequence.) Also checks that every
parameter receives a finite, non-zero gradient.
"""
import cv2
import numpy as np
import torch

from diff_filter import ALPHA, BETA, GAMMA, STAT, sampling_pattern
from .diff_bank import FILTERS
from .features_cuda import BLOCK, _grid, _kernel, _ptr
from .learnable_diff import PRESETS, LearnableDiffFilter

IMAGE = "street_view.jpg"
CIRCLES, DIRECTIONS, DIST = 3, 4, 2


def kernel_filter(grey, offsets, kw):
    """general_filter() on a (B, 1, H, W) CUDA tensor, in PyTorch's CUDA context."""
    B, _, H, W = grey.shape
    modulus = kw.get('modulus', 1)
    step = {'neighbour': 1, 'same_class': modulus, 'none': 0}[kw.get('pairing', 'neighbour')]
    out = torch.empty((B, H, W), dtype=torch.float32, device=grey.device)
    off = torch.from_numpy(offsets).to(grey.device)
    _kernel("diff_filter.cu", "generalFilter")(
        _ptr(out), _ptr(grey.contiguous()), _ptr(off), np.int32(len(offsets)),
        *[np.int32(v) for v in (modulus, kw.get('mod_offset', 1), kw.get('mod_count', modulus), step,
                                kw.get('pair_stride', 1), GAMMA[kw.get('gamma', 'diff')],
                                BETA[kw.get('beta', 'diff')], STAT[kw.get('stat', 'pooled')],
                                ALPHA[kw.get('alpha', 'max')])],
        np.int32(W), np.int32(H), block=BLOCK, grid=_grid(W, H, B))
    return out


def main():
    device = torch.device('cuda')
    bgr = cv2.imread(IMAGE)
    grey = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)).float().div(255.0)[None, None].to(device)
    rgb = grey.expand(1, 3, -1, -1).contiguous()       # the colour mix sums to 1: z = grey

    names = [n for n, p in PRESETS.items() if p[1] is None]
    layer = LearnableDiffFilter(names, spacings=(float(DIST),), circles=CIRCLES, directions=DIRECTIONS,
                                stride=1, half=False).to(device).hard_limit()
    with torch.no_grad():
        got = layer(rgb)[0]
    offsets = np.ascontiguousarray(sampling_pattern(CIRCLES, DIST, DIRECTIONS).offsets())
    print("LearnableDiffFilter in the hard limit vs generalFilter:")
    for i, name in enumerate(names):
        kw, divisor = FILTERS[name]
        kw = dict(kw, modulus=DIRECTIONS) if 'modulus' in kw else kw
        ref = kernel_filter(grey, offsets, kw)[0] / divisor
        d = (ref - got[i]).abs()
        print(f"  {name:12s} max diff {d.max().item():.2e}   mean {d.mean().item():.1e}")

    print("gradients at the initial setting (8 presets, soft pooling):")
    layer = LearnableDiffFilter(spacings=(2.0,)).to(device)
    x = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().div(255.0).to(device)
    (layer(x) ** 2).mean().mul(65536).backward()
    for n, p in layer.named_parameters():
        ok = bool(torch.isfinite(p.grad).all()) and bool((p.grad != 0).any())
        print(f"  {n:13s} {'ok ' if ok else 'BAD'} mean |grad| {p.grad.abs().mean().item():.2e}")


if __name__ == '__main__':
    main()
