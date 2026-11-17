"""Checks the feature front-ends against each other.

    python -m segnet.check_features            # both checks, each in its own process
    python -m segnet.check_features backend    # PyTorch modules vs the CUDA backend, with timing
    python -m segnet.check_features kernels    # PyTorch modules vs the PyCUDA kernels of the repo

The two checks need separate processes: the repository kernels create their own CUDA context
(pycuda.autoinit), the CUDA backend shares PyTorch's. The difference-filter bank has no PyTorch
version: `backend` saves its planes to a temporary file and `kernels` compares them with
diff_filter.general_filter.
"""
import os
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import torch

IMAGE = "street_view.jpg"
DIFF_FILE = os.path.join(tempfile.gettempdir(), "segnet_diff_planes.npz")


def report(name, ref, got):
    d = np.abs(ref - got)
    print(f"  {name:28s} max diff {d.max():.2e}   mean {d.mean():.1e}   {100 * (d > 1e-2).mean():.4f}% of pixels > 0.01")


def check_kernels():
    import sym_measure_example as sym_cuda        # loads street_view.jpg, compiles the kernels
    from morph_features import MorphFeatureMap
    from segnet.features import RegulatedMorphFeatures, SymmetryFeatures

    print("PyTorch modules vs the PyCUDA kernels (MorphFeatureMap, symmetry_image):")
    bgr = cv2.imread(IMAGE)
    rgb = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255.0
    ref = MorphFeatureMap(weighted_elements=[], include_raw=False).compute(bgr)
    report(f"morphology, {ref.shape[0]} planes", ref, RegulatedMorphFeatures()(rgb)[0].numpy())

    module = SymmetryFeatures()
    got = module(torch.from_numpy(sym_cuda.input_img)[None, None].float() / 255.0)[0].numpy()
    i = 0
    for angle in module.angles_deg:
        for K in module.half_widths:
            sa, _, s = sym_cuda.symmetry_image(np.radians(angle), K)
            report(f"symmetry SA  a={angle:g} K={K}", sa, got[i])
            report(f"symmetry S   a={angle:g} K={K}", s, got[i + 1])
            i += 2

    if os.path.exists(DIFF_FILE):
        from diff_filter import general_filter
        from segnet.diff_bank import diff_bank
        saved = np.load(DIFF_FILE)
        ref = np.stack([general_filter(saved['grey'], offsets, **kw) / divisor
                        for _, offsets, kw, divisor in diff_bank()])
        report(f"difference bank, {len(ref)} planes", ref, saved['planes'])
        os.remove(DIFF_FILE)


def check_backend():
    from segnet.features import RegulatedMorphFeatures, SymmetryFeatures
    from segnet.features_cuda import RegulatedMorphFeaturesCuda, SymmetryFeaturesCuda

    def timed(module, x, n=10):
        module(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            module(x)
        torch.cuda.synchronize()
        return 1000 * (time.time() - t) / n

    print("PyTorch modules vs the CUDA backend (batch of 8 crops, 384 x 768):")
    img = cv2.cvtColor(cv2.imread(IMAGE), cv2.COLOR_BGR2RGB)[:384, :768]
    rgb = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255.0).cuda()
    rgb = torch.cat([rgb.roll(40 * i, dims=3) for i in range(8)])
    grey = rgb.mean(1, keepdim=True).mul(255).round().div(255)
    for name, a, b, x in (("morphology (luminance)", RegulatedMorphFeatures(), RegulatedMorphFeaturesCuda().cuda(), grey),
                          ("symmetry", SymmetryFeatures(), SymmetryFeaturesCuda().cuda(), grey)):
        report(name, a(x).cpu().numpy(), b(x).cpu().numpy())
        print(f"  {'':28s} PyTorch {timed(a, x, 3):.0f} ms, CUDA backend {timed(b, x):.1f} ms per batch")

    from segnet.features_cuda import DiffFilterBankCuda
    bank = DiffFilterBankCuda().cuda()
    np.savez(DIFF_FILE, grey=grey[0, 0].cpu().numpy(), planes=bank(grey)[0].cpu().numpy())
    print(f"  {'difference bank':28s} CUDA backend {timed(bank, grey):.1f} ms per batch (checked in the `kernels` step)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode == "kernels":
        check_kernels()
    elif mode == "backend":
        check_backend()
    else:
        for m in ("backend", "kernels"):
            subprocess.run([sys.executable, "-m", "segnet.check_features", m], check=True)
