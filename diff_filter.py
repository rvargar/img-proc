"""Difference filters on sampling points placed on concentric circles or ellipses.

Refactored from the MATLAB project in to_refactor/: createFilter.m (the sampling pattern),
diffFilter.m and the kernels generalFilter and convFilter of t.cu (now cuda/diff_filter.cu).

A SamplingPattern is an ORDERED list of points around a pixel. The points are split into
modulo classes by their position in that order, and general_filter() works in three steps:

    gamma   every point gets a value from its intensity and the centre pixel
    beta    points are paired and each pair is combined (difference or doubled minimum)
    alpha   the pair values are reduced, per class and then over the classes

    pattern = sampling_pattern(circles=3, dist=4, directions=16)
    out = general_filter(image, pattern, modulus=16, mod_count=16, stat='max', alpha='range')
"""
from dataclasses import dataclass
from itertools import permutations

import numpy as np

KERNELS_CU = "diff_filter.cu"

# names -> constants of cuda/diff_filter.cu
GAMMA = {'diff': 0, 'neg_diff': 1, 'abs_diff': 2, 'min2': 3, 'pos_diff': 4}
BETA = {'diff': 0, 'min2': 1}
STAT = {'pooled': 0, 'max': 1, 'min': 2, 'sum': 3}
ALPHA = {'min': 0, 'max': 1, 'sum': 2, 'mean': 3, 'range': 4, 'neg_range': 5}

# The `order` argument of createFilter.m: nesting of the three loops, outermost first.
# c = circles, r = rotations, d = directions.
LEGACY_ORDER = {1: 'crd', 2: 'cdr', 3: 'drc', 4: 'dcr', 5: 'rdc', 6: 'rcd'}


@dataclass(frozen=True)
class SamplingPattern:
    """Ordered sampling points around a pixel.

    points:    (n, 2) float64, exact (dx, dy) positions; x = column, y = row.
    circle, rotation, direction: (n,) indices of the circle (0 = innermost), the rotated
               copy and the direction each point was generated from.
    """
    points: np.ndarray
    circle: np.ndarray
    rotation: np.ndarray
    direction: np.ndarray

    def __len__(self):
        return len(self.points)

    def offsets(self, rounding='round'):
        """(n, 2) int32 pixel offsets. 'round' takes the nearest pixel; 'floor' reproduces
        the MATLAB kernels, which truncated the sampling coordinate."""
        if rounding == 'round':
            return np.rint(self.points).astype(np.int32)
        if rounding == 'floor':
            return np.floor(self.points).astype(np.int32)
        raise ValueError("rounding must be 'round' or 'floor'")

    def classes(self, modulus, mod_offset=1, mod_count=None):
        """Modulo class of every point: (position + 1 - mod_offset) mod modulus, with the
        position counted from 0. Classes >= mod_count are not used by the filter: -1."""
        if modulus < 1:
            raise ValueError("modulus must be at least 1")
        cls = (np.arange(len(self)) + 1 - mod_offset) % modulus
        if mod_count is not None:
            cls[cls >= mod_count] = -1
        return cls

    def render(self, modulus=None, mod_offset=1, mod_count=None, scale=8):
        """BGR picture of the pattern. Without modulus the grey level follows the order of
        the points; with it every used class gets its own colour and unused points are grey."""
        import cv2
        radius = int(np.ceil(np.abs(self.points).max())) + 2
        size = (2 * radius + 1) * scale
        img = np.zeros((size, size, 3), dtype=np.uint8)
        centre = np.array([size // 2, size // 2])
        cv2.drawMarker(img, tuple(int(v) for v in centre), (255, 255, 255), cv2.MARKER_CROSS, scale, 1)
        cls = None if modulus is None else self.classes(modulus, mod_offset, mod_count)
        n_classes = 1 if cls is None else max(int(cls.max()) + 1, 1)
        for i, p in enumerate(self.points):
            if cls is None:
                g = int(60 + 195 * i / max(len(self) - 1, 1))
                colour = (g, g, g)
            elif cls[i] < 0:
                colour = (70, 70, 70)
            else:
                hsv = np.uint8([[[int(180 * cls[i] / n_classes), 220, 255]]])
                colour = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
            xy = centre + np.rint(p * scale).astype(int)
            cv2.circle(img, (int(xy[0]), int(xy[1])), max(scale // 3, 2), colour, -1, cv2.LINE_AA)
        return img


def sampling_pattern(circles, dist, directions, rotations=1, x_ratio=1.0, y_ratio=1.0,
                     order='crd', swap_axes=False):
    """Sampling points on concentric circles or ellipses (createFilter.m).

    circles:     number of circles; circle k (1-based) has radius k * dist
    dist:        distance between neighbouring circles, in pixels
    directions:  points per circle, at the angles 2 pi j / directions
    rotations:   number of rotated copies of the whole pattern; copy r is rotated by
                 pi * r / rotations. 1 = no rotation. Only useful for ellipses.
    x_ratio, y_ratio: stretch of the x and y axis; equal values give circles
    order:       nesting of the generating loops, outermost first: a permutation of 'c'
                 (circles), 'r' (rotations) and 'd' (directions), or the number 1..6 that
                 createFilter.m used. It fixes the ORDER of the points and therefore which
                 points share a modulo class: with 'crd' consecutive points walk around a
                 circle, with 'drc' they walk outwards along a ray.
    swap_axes:   start the angle at the y axis instead of the x axis (createFilter's t = 0)

    The MATLAB code used its first coordinate as the image ROW; here dx is the column, so
    compared to MATLAB figures the pattern is mirrored about the diagonal.
    """
    order = LEGACY_ORDER.get(order, order)
    if order not in {''.join(p) for p in permutations('crd')}:
        raise ValueError(f"order must be a permutation of 'crd' or 1..6, got {order!r}")
    if min(circles, directions, rotations) < 1:
        raise ValueError("circles, directions and rotations must be at least 1")

    sizes = {'c': circles, 'r': rotations, 'd': directions}
    grids = np.meshgrid(*[np.arange(sizes[a]) for a in order], indexing='ij')   # last axis fastest
    index = {a: g.ravel() for a, g in zip(order, grids)}

    radius = (index['c'] + 1) * dist
    theta = 2 * np.pi * index['d'] / directions
    phi = np.pi * index['r'] / rotations
    ex, ey = radius * x_ratio * np.cos(theta), radius * y_ratio * np.sin(theta)
    if swap_axes:
        ex, ey = radius * y_ratio * np.sin(theta), radius * x_ratio * np.cos(theta)
    points = np.stack([ex * np.cos(phi) + ey * np.sin(phi), -ex * np.sin(phi) + ey * np.cos(phi)], axis=1)
    return SamplingPattern(points, index['c'], index['r'], index['d'])


def _run(kernel_name, image, make_args):
    """Uploads a 2D or (n, h, w) image as float32, runs one kernel, returns float32 output."""
    # imported here: gpu_kernels creates a CUDA context, and the pattern code above is also
    # used by segnet, which runs the kernels in PyTorch's context instead
    import pycuda.driver as cuda
    from gpu_kernels import BLOCK, get_function, grid_for
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError("image must be (h, w) or a stack (n, h, w)")
    img = np.ascontiguousarray(image, dtype=np.float32)
    h, w = img.shape[-2:]
    depth = 1 if img.ndim == 2 else img.shape[0]
    out = np.empty_like(img)
    out_gpu = cuda.mem_alloc(out.nbytes)
    try:
        get_function(KERNELS_CU, kernel_name)(out_gpu, cuda.In(img), *make_args(cuda, w, h),
                                              block=BLOCK, grid=grid_for(w, h) + (depth,))
        cuda.memcpy_dtoh(out, out_gpu)
    finally:
        out_gpu.free()
    return out


def _offsets(pattern, rounding):
    if isinstance(pattern, SamplingPattern):
        return np.ascontiguousarray(pattern.offsets(rounding))
    offsets = np.ascontiguousarray(pattern, dtype=np.int32)
    if offsets.ndim != 2 or offsets.shape[1] != 2 or len(offsets) == 0:
        raise ValueError("pattern must be a SamplingPattern or a non-empty (n, 2) offset array")
    return offsets


def general_filter(image, pattern, modulus=1, mod_offset=1, mod_count=None, pairing='neighbour',
                   gamma='diff', beta='diff', stat='pooled', alpha='max', rounding='round',
                   pair_stride=1):
    """Difference filter on a sampling pattern (generalFilter in t.cu).

    image:      (h, w) or (n, h, w), any numeric type; processed as float32
    pattern:    SamplingPattern, or an ordered (n, 2) array of integer (dx, dy) offsets
    modulus, mod_offset, mod_count:
                point i (0-based) is in class (i + 1 - mod_offset) mod modulus; the classes
                0 .. mod_count-1 take part (default: all of them)
    pairing:    'neighbour'  - a point is paired with the next point of the sequence
                'same_class' - with the next point of its own class (modulus points later)
                an integer k  - with the point k places later in the sequence
                'none'       - no pairs: beta is skipped and every point counts on its own
                A pair belongs to the class of its first point.
    pair_stride: only points whose position is a multiple of pair_stride start a pair;
                2 with 'neighbour' gives the disjoint pairs (0, 1), (2, 3), ...
    gamma:      value of a point with intensity v, for the centre intensity c:
                'diff' v - c, 'neg_diff' c - v, 'abs_diff' |v - c|, 'min2' 2 min(v, c),
                'pos_diff' max(v - c, 0)
    beta:       value of a pair: 'diff' second - first, 'min2' 2 min(first, second)
    stat:       per class, the 'max', 'min' or 'sum' of its pair values; 'pooled' skips the
                per-class step and hands all pair values to alpha
    alpha:      'min', 'max', 'sum', 'mean', 'range' (max - min) or 'neg_range' over the
                class statistics

    Returns float32 of the image's shape. Points outside the image use the nearest pixel.
    """
    offsets = _offsets(pattern, rounding)
    mod_count = modulus if mod_count is None else mod_count
    if modulus < 1 or not 0 <= mod_count <= modulus:
        raise ValueError("need modulus >= 1 and 0 <= mod_count <= modulus")
    steps = {'neighbour': 1, 'same_class': modulus, 'none': 0}
    if pairing in steps:
        step = steps[pairing]
    elif isinstance(pairing, (int, np.integer)) and not isinstance(pairing, bool) and pairing >= 1:
        step = int(pairing)
    else:
        raise ValueError("pairing must be 'neighbour', 'same_class', 'none' or a positive integer")
    if pair_stride < 1:
        raise ValueError("pair_stride must be at least 1")
    for name, value, table in (('gamma', gamma, GAMMA), ('beta', beta, BETA),
                               ('stat', stat, STAT), ('alpha', alpha, ALPHA)):
        if value not in table:
            raise ValueError(f"{name} must be one of {sorted(table)}, got {value!r}")
    return _run("generalFilter", image, lambda cuda, w, h: (
        cuda.In(offsets), np.int32(len(offsets)), np.int32(modulus), np.int32(mod_offset),
        np.int32(mod_count), np.int32(step), np.int32(pair_stride), np.int32(GAMMA[gamma]), np.int32(BETA[beta]),
        np.int32(STAT[stat]), np.int32(ALPHA[alpha]), np.int32(w), np.int32(h)))


def conv_filter(image, pattern, weights, centre_weight=0.0, rounding='round'):
    """Linear filter on a sampling pattern (convFilter in t.cu):
    sum_i weights[i] * value(point i) + centre_weight * value(centre)."""
    offsets = _offsets(pattern, rounding)
    weights = np.ascontiguousarray(np.broadcast_to(np.asarray(weights, dtype=np.float32), (len(offsets),)))
    return _run("convFilter", image, lambda cuda, w, h: (
        cuda.In(offsets), cuda.In(weights), np.float32(centre_weight), np.int32(len(offsets)),
        np.int32(w), np.int32(h)))


def legacy_modes(modeA, modeB, modeC):
    """Keyword arguments of general_filter() for the mode numbers of diffFilter.m.

    modeC (gamma) 1..5, modeB (beta) 1..9, modeA (alpha) 1..6. modeB = 17 selected
    convFilter and has no counterpart here: call conv_filter().
    """
    gammas = {1: 'diff', 2: 'neg_diff', 3: 'abs_diff', 4: 'abs_diff', 5: 'min2'}
    stats = ('pooled', 'min', 'max', 'sum')            # modeB 1..4, and again for 5..8
    alphas = {1: 'min', 2: 'max', 3: 'sum', 4: 'mean', 5: 'range', 6: 'neg_range'}
    if modeC not in gammas or modeA not in alphas or modeB not in range(1, 10):
        raise ValueError("unknown mode number")
    if modeB == 9:
        beta, stat, pairing = 'diff', 'pooled', 'same_class'
    else:
        beta, stat, pairing = ('diff' if modeB <= 4 else 'min2'), stats[(modeB - 1) % 4], 'neighbour'
    return {'gamma': gammas[modeC], 'beta': beta, 'stat': stat, 'pairing': pairing, 'alpha': alphas[modeA]}


# ---------------------------------------------------------------------------------------
# diffFilter1 .. diffFilter6 of t.cu: the fixed filters that preceded generalFilter. Each is
# one setting of general_filter(). They all rectify the point values: D = max(v - c, 0),
# except diff_filter5, which uses |v - c|.
# ---------------------------------------------------------------------------------------

def diff_filter1(image, pattern, offset, modulus=1, rounding='round'):
    """Range (max - min) of D[i] - D[i - offset] over the points i that are a multiple of
    modulus. Also diffFilterTest, which computed the same."""
    return general_filter(image, pattern, modulus, 1 - offset, 1, pairing=int(offset), gamma='pos_diff',
                          beta='diff', stat='pooled', alpha='range', rounding=rounding)


def diff_filter2(image, pattern, modulus, mod_offset=1, mod_count=None, rounding='round'):
    """Range of the neighbour differences D[i+1] - D[i] in the classes 0 .. mod_count-2
    (one class fewer than the other filters, as in the original)."""
    mod_count = modulus if mod_count is None else mod_count
    return general_filter(image, pattern, modulus, mod_offset, max(mod_count - 1, 0), gamma='pos_diff',
                          beta='diff', stat='pooled', alpha='range', rounding=rounding)


def diff_filter3(image, pattern, modulus, mod_offset=1, mod_count=None, rounding='round'):
    """Range of 2 min(D[i], D[i+1]) over the neighbour pairs in the classes 0 .. mod_count-1."""
    return general_filter(image, pattern, modulus, mod_offset, mod_count, gamma='pos_diff',
                          beta='min2', stat='pooled', alpha='range', rounding=rounding)


def diff_filter4(image, pattern, modulus, mod_offset=1, mod_count=None, rounding='round'):
    """Strongest class: per class the sum of 2 min(D[i], D[i+1]) over the disjoint pairs
    (0, 1), (2, 3), ..., then the largest sum divided by the points per class
    (n_points // modulus)."""
    n = len(_offsets(pattern, rounding))
    if n < modulus:
        raise ValueError("diff_filter4 needs at least `modulus` points")
    out = general_filter(image, pattern, modulus, mod_offset, mod_count, gamma='pos_diff', beta='min2',
                         stat='sum', alpha='max', rounding=rounding, pair_stride=2)
    return out / np.float32(n // modulus)


def diff_filter5(image, pattern, modulus, mod_offset=1, mod_count=None, rounding='round'):
    """Max over the classes of the smallest |v - c| in the class: large when ALL points of
    some class differ from the centre. No pairs. The original took the classes as two
    matrices (classX, classY); pattern_from_classes() converts that layout."""
    return general_filter(image, pattern, modulus, mod_offset, mod_count, pairing='none', gamma='abs_diff',
                          stat='min', alpha='max', rounding=rounding)


def diff_filter6(image, pattern, rounding='round'):
    """Largest D[i+1] - D[i] over the disjoint pairs (0, 1), (2, 3), ..."""
    return general_filter(image, pattern, gamma='pos_diff', beta='diff', stat='pooled', alpha='max',
                          rounding=rounding, pair_stride=2)


def pattern_from_classes(classes):
    """Ordered offsets for explicitly given classes, for diff_filter5.

    classes: sequence of n_classes arrays, each (n_elements, 2) integer (dx, dy) offsets, all
    with the same number of elements. Returns (offsets, modulus): the points interleaved so
    that point i belongs to class i mod n_classes, to be used with mod_offset = 1.
    """
    classes = [np.asarray(c, dtype=np.int32) for c in classes]
    if not classes or any(c.ndim != 2 or c.shape != classes[0].shape or c.shape[1] != 2 for c in classes):
        raise ValueError("classes must be equally sized (n_elements, 2) arrays")
    return np.stack(classes, axis=1).reshape(-1, 2), len(classes)
