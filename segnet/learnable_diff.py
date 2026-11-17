"""generalFilter (cuda/diff_filter.cu) with learnable parameters, in PyTorch.

The kernel picks one of a few fixed options at every stage. Here each stage is a continuous
family that contains those options, and its coefficients are trained with the network:

    stage     kernel                              here, per filter
    -------   ---------------------------------   ------------------------------------------
    input     8-bit luminance                     colour mix  z = k . (r, g, b)
    points    integer offsets on circles          radius per circle, axis ratio and rotation
                                                  of the ellipse; bilinear sampling
    gamma     v-c, c-v, |v-c|, max(v-c,0),        g = w1 v + w2 c + w3 |v - c|
              2 min(v,c)
    beta      b-a, 2 min(a,b), or no pairs        t = u0 a + sum_k (u1k b_k + u2k |a - b_k|)
    pairing   step in the point sequence          over three partners b_k of a point: the next
                                                  direction on its circle, the next circle on
                                                  its ray, the opposite point
    stat      max, min, sum per class             soft pooling over the circles of a direction
    alpha     min, max, sum, mean, range          P(ta) - rho P(tb): soft pooling over the
              over the classes                    directions, twice

Soft pooling with temperature t is  sum_i softmax(t x)_i x_i : the maximum for t -> +inf, the
minimum for t -> -inf, the mean for t = 0. A sum is a mean up to a constant factor.

What is NOT relaxed is the structure a plain convolution lacks: values are referenced to the
centre pixel, a class is one direction (a ray), and the result is pooled over directions.

Every filter starts as one of the fixed filters of segnet/diff_bank.py (PRESETS), with soft
instead of hard pooling. `hard_limit()` sets the temperatures so that the layer reproduces
general_filter(); segnet/check_learnable_diff.py compares the two.

Differences to the kernel: pairs wrap around (the last direction is paired with the first one
of the SAME circle, the outermost circle with the innermost), and points lie at their exact
positions instead of the nearest pixel.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

GREY_WEIGHTS = (0.299, 0.587, 0.114)
PARTNERS = ('angular', 'radial', 'opposite')
T_SCALE = 20.0          # temperature = T_SCALE * parameter; intensities are in [0, 1]
T_INIT = 2.0            # initial |parameter| of a max or min pooling

GAMMAS = {'diff': (1.0, -1.0, 0.0), 'neg_diff': (-1.0, 1.0, 0.0), 'abs_diff': (0.0, 0.0, 1.0),
          'pos_diff': (0.5, -0.5, 0.5), 'min2': (1.0, 1.0, -1.0)}
POOL = {'max': 1.0, 'min': -1.0, 'mean': 0.0, 'sum': 0.0}

# name -> gamma, (beta, partner) or None, stat, alpha. The filters of segnet/diff_bank.py;
# the two blobs use no pairs, which gives the same result as the bank's pooled min2 pairs.
PRESETS = {
    'edge':        ('diff', ('diff', 'angular'), 'max', 'max'),
    'dirsum':      ('diff', ('diff', 'angular'), 'mean', 'range'),
    'dark_blob':   ('diff', None, 'min', 'min'),
    'bright_blob': ('neg_diff', None, 'min', 'min'),
    'ray_minmax':  ('abs_diff', None, 'min', 'max'),
    'ray_best':    ('abs_diff', None, 'mean', 'min'),
    'ray_range':   ('abs_diff', None, 'mean', 'range'),
    'pair_bright': ('pos_diff', ('min2', 'angular'), 'mean', 'max'),
}


def soft_pool(x, t, dim):
    """sum softmax(t x) x over `dim`: max for t -> +inf, min for t -> -inf, mean for t = 0."""
    return (torch.softmax(t * x, dim=dim) * x).sum(dim=dim)


class LearnableDiffFilter(nn.Module):
    """Bank of difference filters with learnable parameters.

    Input : (B, 3, H, W) RGB in [0, 1].
    Output: (B, n_filters, H / stride, W / stride); n_filters = len(presets) * len(spacings).

    presets:    names from PRESETS; the initial setting of the filters
    spacings:   initial distance between the circles in pixels; every preset is created once
                per spacing
    circles, directions: size of the sampling pattern (fixed)
    stride:     distance of the output pixels. The sampling points are always read from the
                full-resolution input.
    chunk:      0, or the number of filters computed at a time while training. The
                intermediate tensors of a chunk are then recomputed in the backward pass
                instead of being kept (gradient checkpointing): memory for one chunk, more time.
    half:       on the GPU, compute everything after the sampling in half precision: half
                the memory and about 1.7x faster. The sampling itself stays in single
                precision, half precision would move the points by up to 0.2 pixels.
    """

    def __init__(self, presets=tuple(PRESETS), spacings=(2.0, 4.0), circles=3, directions=8,
                 stride=2, chunk=0, half=True):
        super().__init__()
        if directions % 2:
            raise ValueError("directions must be even (opposite points are paired)")
        self.names_ = [f"{p}/d{s:g}" for s in spacings for p in presets]
        self.circles, self.directions, self.stride, self.chunk, self.half = circles, directions, stride, chunk, half
        M, K = len(self.names_), len(PARTNERS)
        self.n_planes = M

        colour = torch.tensor(GREY_WEIGHTS).repeat(M, 1)
        radius = torch.zeros(M, circles)
        gamma = torch.zeros(M, 3)
        beta_self = torch.ones(M)
        beta_partner, beta_abs = torch.zeros(M, K), torch.zeros(M, K)
        t_stat, t_alpha, t_alpha2, rho = torch.zeros(M), torch.zeros(M), torch.zeros(M), torch.zeros(M)
        i = 0
        for s in spacings:
            for p in presets:
                g, pair, stat, alpha = PRESETS[p]
                radius[i] = s * torch.arange(1, circles + 1)
                gamma[i] = torch.tensor(GAMMAS[g])
                if pair is not None:
                    k = PARTNERS.index(pair[1])
                    if pair[0] == 'diff':                       # b - a
                        beta_self[i], beta_partner[i, k] = -1.0, 1.0
                    else:                                       # a + b - |a - b| = 2 min(a, b)
                        beta_partner[i, k], beta_abs[i, k] = 1.0, -1.0
                t_stat[i] = T_INIT * POOL[stat]
                if alpha == 'range':
                    t_alpha[i], t_alpha2[i], rho[i] = T_INIT, -T_INIT, 1.0
                else:
                    t_alpha[i], t_alpha2[i] = T_INIT * POOL[alpha], -T_INIT * POOL[alpha]
                i += 1
        self.colour = nn.Parameter(colour)              # (M, 3)  input mix
        self.radius = nn.Parameter(radius)              # (M, C)  pixels
        self.log_ratio = nn.Parameter(torch.zeros(M))   # x axis * e^r, y axis * e^-r
        self.angle = nn.Parameter(torch.zeros(M))       # rotation of the pattern, radians
        self.gamma = nn.Parameter(gamma)                # (M, 3)  w1, w2, w3
        self.beta_self = nn.Parameter(beta_self)        # (M,)    u0
        self.beta_partner = nn.Parameter(beta_partner)  # (M, K)  u1k
        self.beta_abs = nn.Parameter(beta_abs)          # (M, K)  u2k
        self.t_stat = nn.Parameter(t_stat)              # temperatures / T_SCALE
        self.t_alpha = nn.Parameter(t_alpha)
        self.t_alpha2 = nn.Parameter(t_alpha2)
        self.rho = nn.Parameter(rho)
        theta = 2 * math.pi * torch.arange(directions) / directions
        self.register_buffer("unit", torch.stack([theta.cos(), theta.sin()], 1), persistent=False)   # (D, 2)

    def names(self):
        return list(self.names_)

    @torch.no_grad()
    def hard_limit(self, temperature=1e5):
        """Replaces every non-zero temperature by +-temperature / T_SCALE: hard max and min."""
        for t in (self.t_stat, self.t_alpha, self.t_alpha2):
            t.copy_(torch.sign(t) * temperature / T_SCALE)
        return self

    def points(self):
        """(M, C, D, 2) sampling positions (dx, dy) relative to the pixel."""
        r = self.radius.clamp_min(0.5)[:, :, None]                                   # (M, C, 1)
        ex = r * torch.exp(self.log_ratio)[:, None, None] * self.unit[:, 0]           # (M, C, D)
        ey = r * torch.exp(-self.log_ratio)[:, None, None] * self.unit[:, 1]
        cos, sin = self.angle.cos()[:, None, None], self.angle.sin()[:, None, None]
        return torch.stack([ex * cos + ey * sin, -ex * sin + ey * cos], dim=-1)

    def state(self):
        """The learned parameters as nested lists, for the training log."""
        return {k: v.detach().float().cpu().tolist() for k, v in self.named_parameters()}

    def _filter(self, z, points, gamma, beta_self, beta_partner, beta_abs, t_stat, t_alpha, t_alpha2, rho):
        """z: (B, m, H, W) mixed input of m filters; the other arguments are their parameters.
        Returns (B, m, h, w)."""
        B, m, H, W = z.shape
        C, D, s = self.circles, self.directions, self.stride
        h, w = H // s, W // s
        # centres of the output pixels in input pixel coordinates
        ys = torch.arange(h, device=z.device, dtype=z.dtype) * s + (s - 1) / 2
        xs = torch.arange(w, device=z.device, dtype=z.dtype) * s + (s - 1) / 2
        base = torch.stack([xs.view(1, w).expand(h, w), ys.view(h, 1).expand(h, w)], dim=-1)   # (h, w, 2)
        size = torch.tensor([W, H], device=z.device, dtype=z.dtype)
        zt = z.transpose(0, 1)                                                       # (m, B, H, W)

        # one grid_sample call: the batch is the channel axis, the points are stacked along y
        grid = base + points.reshape(m, C * D, 1, 1, 2)                               # (m, P, h, w, 2)
        grid = ((2 * grid + 1) / size - 1).reshape(m, C * D * h, w, 2)
        v = F.grid_sample(zt, grid, mode='bilinear', padding_mode='border', align_corners=False)
        v = v.view(m, B, C, D, h, w)
        if s == 1:
            c = zt
        else:
            cgrid = ((2 * base + 1) / size - 1).expand(m, h, w, 2)
            c = F.grid_sample(zt, cgrid, mode='bilinear', padding_mode='border', align_corners=False)
        c = c.view(m, B, 1, 1, h, w)
        if self.half and z.is_cuda:
            v, c = v.half(), c.half()
            gamma, beta_self, beta_partner, beta_abs, t_stat, t_alpha, t_alpha2, rho = (
                x.half() for x in (gamma, beta_self, beta_partner, beta_abs, t_stat, t_alpha, t_alpha2, rho))

        def p(x):
            return x.view(m, 1, 1, 1, 1, 1)

        g = p(gamma[:, 0]) * v + p(gamma[:, 1]) * c + p(gamma[:, 2]) * (v - c).abs()
        t = p(beta_self) * g
        for k, partner in enumerate((g.roll(-1, 3), g.roll(-1, 2), g.roll(D // 2, 3))):
            t = t + p(beta_partner[:, k]) * partner + p(beta_abs[:, k]) * (g - partner).abs()
        stat = soft_pool(t, T_SCALE * p(t_stat), dim=2)                              # (m, B, D, h, w)
        q = lambda x: x.view(m, 1, 1, 1, 1)
        out = soft_pool(stat, T_SCALE * q(t_alpha), dim=2) \
            - rho.view(m, 1, 1, 1) * soft_pool(stat, T_SCALE * q(t_alpha2), dim=2)
        return out.transpose(0, 1).float()

    def forward(self, rgb):
        with torch.cuda.amp.autocast(enabled=False):
            z = F.conv2d(rgb.float(), self.colour[:, :, None, None])                 # (B, M, H, W)
            params = (self.points(), self.gamma, self.beta_self, self.beta_partner, self.beta_abs,
                      self.t_stat, self.t_alpha, self.t_alpha2, self.rho)
            M = self.n_planes
            step = self.chunk or M
            out = []
            for i in range(0, M, step):
                args = (z[:, i:i + step],) + tuple(q[i:i + step] for q in params)
                if torch.is_grad_enabled() and z.requires_grad and step < M:
                    out.append(checkpoint(self._filter, *args))
                else:
                    out.append(self._filter(*args))
            return torch.cat(out, dim=1)
