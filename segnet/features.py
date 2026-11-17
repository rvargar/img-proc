"""Fixed (parameter-free) feature front-ends in PyTorch.

These are GPU re-implementations of the PyCUDA kernels of this repository, written with
tensor operations so that they run inside the training loop, on batches, after augmentation:

    RegulatedMorphFeatures  ==  MorphFeatureMap (flat elements)      cuda/morph_features.cu
    SymmetryFeatures        ==  symmetry_image()                     cuda/sym_measure.cu

Both are checked against the kernels in segnet/check_features.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from structuring_elements import flat_bank, haar_bank

SYM_EPS = 1e-6


class RegulatedMorphFeatures(nn.Module):
    """Order statistics under a bank of flat structuring elements, per colour channel.

    For every element and channel: one plane per quantile (low quantile = regulated erosion,
    0.5 = median, high quantile = regulated dilation) and their range (regulated
    morphological gradient). Borders replicate the nearest pixel.

    Input : (B, C, H, W) float in [0, 1]; it is quantised to 8 bits like the CUDA kernel.
    Output: (B, n_elements * (n_quantiles + 1) * C, H, W) float in [0, 1], ordered
            element -> quantile (then range) -> channel, as MorphFeatureMap does.
    """

    def __init__(self, elements=None, quantiles=(0.1, 0.5, 0.9), include_range=True):
        super().__init__()
        self.elements = flat_bank() if elements is None else list(elements)
        self.quantiles = tuple(sorted(quantiles))
        self.include_range = include_range and len(self.quantiles) > 1
        # 0-based ranks, same rounding as MorphFeatureMap.ranks_for
        self.ranks = [[int(round(q * (e.size - 1))) for q in self.quantiles] for e in self.elements]
        self.pad = max(e.radius for e in self.elements)
        self.planes_per_channel = len(self.elements) * (len(self.quantiles) + int(self.include_range))

    def names(self, channels=('r', 'g', 'b')):
        out = []
        for e in self.elements:
            for q in self.quantiles:
                out += [f"{e.name}/q{q:g}/{c}" for c in channels]
            if self.include_range:
                out += [f"{e.name}/range/{c}" for c in channels]
        return out

    @torch.no_grad()
    def forward(self, x):
        B, C, H, W = x.shape
        p = self.pad
        q8 = (x * 255.0).round().clamp_(0, 255).to(torch.uint8)
        # replicate padding by index clamping (F.pad has no uint8 replicate mode)
        iy = torch.arange(-p, H + p, device=x.device).clamp_(0, H - 1)
        ix = torch.arange(-p, W + p, device=x.device).clamp_(0, W - 1)
        padded = q8[:, :, iy][:, :, :, ix]

        out = []
        for e, ranks in zip(self.elements, self.ranks):
            # (B, C, H, W, n): the pixels under the element, one slice per offset
            nb = torch.stack([padded[:, :, p + dy:p + dy + H, p + dx:p + dx + W]
                              for dx, dy in e.offsets.tolist()], dim=-1)
            nb = nb.sort(dim=-1).values[..., ranks]                    # (B, C, H, W, nq)
            planes = nb.permute(0, 4, 1, 2, 3).float().div_(255.0)     # (B, nq, C, H, W)
            out.append(planes.reshape(B, -1, H, W))
            if self.include_range:
                out.append(planes[:, -1] - planes[:, 0])
        return torch.cat(out, dim=1)


class SymmetryFeatures(nn.Module):
    """Per-pixel intensity symmetry (Zielke et al., ECCV'92) for several scan angles and
    several maximal half-widths.

    For a scan line through the pixel in direction `angle` (0 = along y, 90 = along x) and
    growing intervals [-k, k], S compares the energy of the even and the odd part of the
    intensity profile; SA = w / (2 w_max) * (S + 1) weights it by the interval width
    w = 2k + 1. Per (angle, half-width K) the module returns
        SA: the largest SA over k <= K, in [0, 1]   (confidence of a symmetry axis here)
        S : the symmetry measure at that width, in [-1, 1]
    One pass up to max(K) serves all half-widths: the running maximum of w (S + 1) is
    snapshotted at each K.

    Input : (B, 1, H, W) grey image in [0, 1].
    Output: (B, n_angles * n_half_widths * 2, H, W), ordered angle -> half-width -> (SA, S).
    """

    def __init__(self, angles_deg=(0.0, 45.0, 90.0), half_widths=(4, 12, 32)):
        super().__init__()
        self.angles_deg = tuple(angles_deg)
        self.half_widths = tuple(sorted(int(k) for k in half_widths))
        self.n_planes = 2 * len(self.angles_deg) * len(self.half_widths)

    def names(self):
        return [f"sym/a{a:g}/hw{k}/{n}" for a in self.angles_deg for k in self.half_widths
                for n in ('SA', 'S')]

    @staticmethod
    def _shifted(padded, pad, fx, fy, H, W):
        """Image sampled at (x + fx, y + fy) with bilinear interpolation: a fractional shift
        that is the same for every pixel is a blend of four integer shifts."""
        x0, y0 = math.floor(fx), math.floor(fy)
        ax, ay = fx - x0, fy - y0

        def at(dx, dy):
            return padded[:, :, pad + dy:pad + dy + H, pad + dx:pad + dx + W]

        out = at(x0, y0) * ((1 - ax) * (1 - ay))
        if ax > 1e-6:
            out = out + at(x0 + 1, y0) * (ax * (1 - ay))
        if ay > 1e-6:
            out = out + at(x0, y0 + 1) * ((1 - ax) * ay)
        if ax > 1e-6 and ay > 1e-6:
            out = out + at(x0 + 1, y0 + 1) * (ax * ay)
        return out

    @torch.no_grad()
    def forward(self, grey):
        B, _, H, W = grey.shape
        grey = grey.float()
        K_max = self.half_widths[-1]
        pad = K_max + 2
        padded = F.pad(grey, (pad, pad, pad, pad), mode='replicate')
        # texel-centre coordinates, as in the kernel
        cy = torch.arange(H, device=grey.device, dtype=torch.float32).view(1, 1, H, 1) + 0.5
        cx = torch.arange(W, device=grey.device, dtype=torch.float32).view(1, 1, 1, W) + 0.5

        dirs = []
        for angle in self.angles_deg:
            t = math.radians(angle)
            dx, dy = math.sin(t), math.cos(t)          # rotate((0, 1), angle)
            dirs.append((0.0 if abs(dx) < 1e-6 else dx, 0.0 if abs(dy) < 1e-6 else dy))

        # all angles are processed together, as channels: (B, n_angles, H, W)
        A = len(dirs)
        shape = (B, A, H, W)
        sumE = grey.new_zeros(shape)
        sumE2 = grey.new_zeros(shape)
        sumO2 = grey.new_zeros(shape)
        best = grey.new_zeros(shape)                   # running max of w * (S + 1)
        bestS = grey.new_zeros(shape)
        snapshots = []
        for k in range(1, K_max + 1):
            # the whole interval must lie inside the image
            valid = torch.cat([((cx - k * dx >= 0) & (cx - k * dx < W) & (cx + k * dx >= 0) & (cx + k * dx < W)
                                & (cy - k * dy >= 0) & (cy - k * dy < H) & (cy + k * dy >= 0) & (cy + k * dy < H)
                                ).expand(1, 1, H, W) for dx, dy in dirs], dim=1)
            gn = torch.cat([self._shifted(padded, pad, -k * dx, -k * dy, H, W) for dx, dy in dirs], dim=1) - grey
            gp = torch.cat([self._shifted(padded, pad, k * dx, k * dy, H, W) for dx, dy in dirs], dim=1) - grey
            e, o = 0.5 * (gp + gn), 0.5 * (gp - gn)
            sumE += 2 * e
            sumE2 += 2 * e * e
            sumO2 += 2 * o * o
            w = 2.0 * k + 1.0
            En2 = (sumE2 - sumE * sumE / w).clamp_min_(0)
            den = En2 + sumO2
            S = ((En2 - sumO2) / den.clamp_min(SYM_EPS)) * (den > SYM_EPS)
            v = w * (S + 1)
            better = valid if k == 1 else valid & (v > best)
            best = torch.where(better, v, best)
            bestS = torch.where(better, S, bestS)
            if k in self.half_widths:
                snapshots.append(torch.stack([best / (2.0 * w), bestS], dim=2))   # (B, A, 2, H, W)
        return torch.stack(snapshots, dim=2).reshape(B, -1, H, W)           # (B, A, nK, 2, H, W)


class HaarFeatures(nn.Module):
    """Linear responses of a bank of weighted (Haar-like) structuring elements: mean of the
    positive region minus mean of the negative region. The same result as weightedFilter in
    cuda/morph_features.cu (the weighted part of MorphFeatureMap), computed as ONE fixed
    convolution whose kernels are the element masks. Borders replicate the nearest pixel.

    Input : (B, 1, H, W) grey image in [0, 1].
    Output: (B, n_elements, H, W) in [-1, 1].
    """

    def __init__(self, elements=None):
        super().__init__()
        self.elements = haar_bank() if elements is None else list(elements)
        self.pad = max(e.radius for e in self.elements)
        k = 2 * self.pad + 1
        weight = torch.zeros(len(self.elements), 1, k, k)
        for i, e in enumerate(self.elements):
            weight[i, 0, e.offsets[:, 1] + self.pad, e.offsets[:, 0] + self.pad] = torch.from_numpy(e.weights)
        self.register_buffer("weight", weight, persistent=False)
        self.n_planes = len(self.elements)

    def names(self):
        return [e.name for e in self.elements]

    @torch.no_grad()
    def forward(self, grey):
        p = self.pad
        return F.conv2d(F.pad(grey.float(), (p, p, p, p), mode='replicate'), self.weight)


GREY_WEIGHTS = (0.299, 0.587, 0.114)


class FeatureFrontEnd(nn.Module):
    """RGB batch -> stack of [raw RGB] + [regulated morphology] + [symmetry] + [Haar] +
    [difference filter] planes.

    All planes but RGB are computed on the luminance: they describe
    structure, the raw RGB planes carry the colour. This keeps the front-end 3x cheaper than
    filtering every colour channel.

    use_rgb / use_morph / use_sym / use_haar / use_diff switch the groups, for ablations.
    diff_bank:    name of the difference-filter bank (segnet/diff_bank.py); CUDA backend only.
    morph_planes: 'all' (quantiles and range per element) or 'range' (only the range, the
                  regulated morphological gradient).
    backend: 'cuda' runs the kernels in cuda/ on the tensors (fast, needs PyCUDA and a GPU),
             'torch' uses plain tensor operations (portable, also works on the CPU).
    """

    def __init__(self, use_rgb=True, use_morph=True, use_sym=True, use_haar=False, backend='cuda',
                 morph_planes='all', morph_kwargs=None, sym_kwargs=None, haar_kwargs=None,
                 use_diff=False, diff_bank='full'):
        super().__init__()
        if not (use_rgb or use_morph or use_sym or use_haar or use_diff):
            raise ValueError("at least one feature group is needed")
        if morph_planes not in ('all', 'range'):
            raise ValueError("morph_planes must be 'all' or 'range'")
        if backend == 'cuda':
            from .features_cuda import RegulatedMorphFeaturesCuda as Morph, SymmetryFeaturesCuda as Sym
        elif backend == 'torch':
            Morph, Sym = RegulatedMorphFeatures, SymmetryFeatures
        else:
            raise ValueError("backend must be 'cuda' or 'torch'")
        self.use_rgb = use_rgb
        self.morph = Morph(**(morph_kwargs or {})) if use_morph else None
        self.sym = Sym(**(sym_kwargs or {})) if use_sym else None
        self.haar = HaarFeatures(**(haar_kwargs or {})) if use_haar else None
        self.diff = None
        if use_diff:
            if backend != 'cuda':
                raise ValueError("the difference-filter bank needs backend='cuda'")
            from .features_cuda import DiffFilterBankCuda
            self.diff = DiffFilterBankCuda(diff_bank)
        self.morph_index = None
        n_morph = self.morph.planes_per_channel if self.morph else 0
        if self.morph and morph_planes == 'range':
            if not self.morph.include_range:
                raise ValueError("morph_planes='range' needs include_range")
            per_element = len(self.morph.quantiles) + 1
            self.morph_index = list(range(per_element - 1, n_morph, per_element))
            n_morph = len(self.morph_index)
        self.groups = {'rgb': 3 if use_rgb else 0, 'morph': n_morph,
                       'sym': self.sym.n_planes if self.sym else 0,
                       'haar': self.haar.n_planes if self.haar else 0,
                       'diff': self.diff.n_planes if self.diff else 0}
        self.n_features = sum(self.groups.values())

    @torch.no_grad()
    def forward(self, rgb):
        """rgb: (B, 3, H, W) float in [0, 1]."""
        with torch.cuda.amp.autocast(enabled=False):
            rgb = rgb.float()
            out = [rgb] if self.use_rgb else []
            if any(m is not None for m in (self.morph, self.sym, self.haar, self.diff)):
                w = GREY_WEIGHTS
                grey = w[0] * rgb[:, 0:1] + w[1] * rgb[:, 1:2] + w[2] * rgb[:, 2:3]
                # 8-bit luminance, as the kernels of this repository expect
                grey = (grey * 255.0).round().clamp_(0, 255) / 255.0
                if self.morph is not None:
                    m = self.morph(grey)
                    out.append(m if self.morph_index is None else m[:, self.morph_index])
                if self.sym is not None:
                    out.append(self.sym(grey))
                if self.haar is not None:
                    out.append(self.haar(grey))
                if self.diff is not None:
                    out.append(self.diff(grey))
            return torch.cat(out, dim=1)
