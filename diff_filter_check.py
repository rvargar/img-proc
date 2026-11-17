"""Checks the CUDA kernels of cuda/diff_filter.cu.

    python diff_filter_check.py

1. generalFilter against a plain NumPy implementation, for every combination of gamma, beta,
   stat, alpha and pairing.
2. diff_filter1 .. diff_filter6 against line-by-line NumPy transcriptions of the original
   kernels in to_refactor/t.cu. The originals have quirks that the port does not keep (see
   the README); each comparison uses parameters for which the quirk has no effect, as noted.
"""
from itertools import product

import cv2
import numpy as np

from diff_filter import (ALPHA, BETA, GAMMA, STAT, conv_filter, diff_filter1, diff_filter2, diff_filter3,
                         diff_filter4, diff_filter5, diff_filter6, general_filter, pattern_from_classes,
                         sampling_pattern)


def samples(image, offsets):
    """(n, h, w): the image value at every sampling point, borders clamped."""
    h, w = image.shape
    ys, xs = np.mgrid[0:h, 0:w]
    return np.stack([image[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)] for dx, dy in offsets])


# ------------------------------------------------------------------ generalFilter reference

def general_filter_reference(image, offsets, modulus, mod_offset, mod_count, pairing, gamma, beta, stat,
                             alpha, pair_stride=1):
    image = image.astype(np.float32)
    v, c = samples(image, offsets), image[None]
    D = {'diff': v - c, 'neg_diff': c - v, 'abs_diff': np.abs(v - c), 'min2': 2 * np.minimum(v, c),
         'pos_diff': np.maximum(v - c, 0)}[gamma]

    n = len(offsets)
    step = {'neighbour': 1, 'same_class': modulus, 'none': 0}.get(pairing, pairing)
    cls = (np.arange(n) + 1 - mod_offset) % modulus
    first = np.array([i for i in range(n - step) if cls[i] < mod_count and i % pair_stride == 0], dtype=int)
    if len(first) == 0:
        return np.zeros_like(image)
    a, b = D[first], D[first + step]
    T = a if step == 0 else (b - a if beta == 'diff' else 2 * np.minimum(a, b))

    if stat == 'pooled':
        S = T
    else:
        reduce = {'max': np.max, 'min': np.min, 'sum': np.sum}[stat]
        S = np.stack([reduce(T[cls[first] == m], axis=0) for m in np.unique(cls[first])])
    return {'min': S.min(0), 'max': S.max(0), 'sum': S.sum(0), 'mean': S.mean(0),
            'range': S.max(0) - S.min(0), 'neg_range': S.min(0) - S.max(0)}[alpha]


# ------------------------------------------- transcriptions of diffFilter1..6 in t.cu
# Variable names follow the kernels. D is (n, h, w); the per-pixel scalars of the kernels are
# (h, w) arrays here. The kernels stored the result in an unsigned short; that is left out.

def relu_diff(image, offsets):
    return np.maximum(samples(image, offsets) - image[None], 0)      # D[index] = value-center < 0 ? 0 : ...


def k1_class(i, modOffs, modulus):
    """(unsigned short)((i - modOffs + 1) % modulus) with C's remainder (sign of the dividend)."""
    r = int(np.fmod(i - modOffs + 1, modulus))
    return r if r >= 0 else r + 65536


def max_minus_min(values, like):
    mx, mn = np.full_like(like, -1.0), np.full_like(like, 9e13)
    for s in values:
        mx, mn = np.maximum(mx, s), np.minimum(mn, s)
    return mx - mn


def t_diffFilter1(image, offsets, offset, modulus):
    D = relu_diff(image, offsets)
    S = [D[index] - D[index - offset] for index in range(len(D)) if index > offset and index % modulus == 0]
    return max_minus_min(S, image)


def t_diffFilter2(image, offsets, modulus, modOffs, modCount):
    D = relu_diff(image, offsets)
    zero = np.zeros_like(image)
    tmp1, tmp2, S = zero, zero, []
    for i in range(len(D) - 1):
        for m in range(modCount - 1):                 # every m overwrites: the last one decides
            tmp1, tmp2 = (D[i], D[i + 1]) if k1_class(i, modOffs, modulus) == m else (zero, zero)
        S.append(tmp2 - tmp1)
    return max_minus_min(S, image)


def t_diffFilter3(image, offsets, modulus, modOffs, modCount):
    D = relu_diff(image, offsets)
    tmp1, S = np.zeros_like(image), []
    for i in range(len(D) - 1):
        for m in range(modCount):
            tmp1 = D[i] + D[i + 1] - np.abs(D[i] - D[i + 1]) if k1_class(i, modOffs, modulus) == m else np.zeros_like(image)
        S.append(tmp1)
    return max_minus_min(S, image)


def t_diffFilter4(image, offsets, modulus, modOffs, modCount):
    D = relu_diff(image, offsets)
    count = len(D)
    S = [np.zeros_like(image) for _ in range(modCount)]
    for i in range(0, count - 1, 2):
        for m in range(modCount):
            if k1_class(i, modOffs, modulus) == m:
                S[m] = S[m] + D[i] + D[i + 1] - np.abs(D[i] - D[i + 1])
    mx = np.full_like(image, -1.0)
    for s in S:
        mx = np.maximum(mx, s)
    return np.trunc(mx / (count // modulus))          # (int)(max / (count / modulus))


def t_diffFilter5(image, dx, dy, elements, modCount):
    """dx, dy: flat class matrices; the kernel reads class i, element j at i*modCount + j."""
    h, w = image.shape
    ys, xs = np.mgrid[0:h, 0:w]
    mx = np.full_like(image, -1e8)
    for i in range(modCount):
        mn = np.full_like(image, 1e6)
        for j in range(elements):
            index = i * modCount + j
            value = image[np.clip(ys + dy[index], 0, h - 1), np.clip(xs + dx[index], 0, w - 1)]
            mn = np.minimum(mn, np.abs(value - image))
        mx = np.maximum(mx, mn)
    return mx


def t_diffFilter6(image, offsets):
    D = relu_diff(image, offsets)
    mx = np.full_like(image, -1.0)
    for index in range(1, len(D)):
        if index % 2 != 0:
            mx = np.maximum(mx, D[index] - D[index - 1])
    return mx


# --------------------------------------------------------------------------------- checks

def rel_err(got, ref):
    return np.abs(got - ref).max() / max(np.abs(ref).max(), 1.0)


def check_general(image):
    cases = [  # pattern, modulus, mod_offset, mod_count
        (sampling_pattern(3, 3, 8), 8, 1, 8),                                   # one class per direction
        (sampling_pattern(3, 3, 8, order='drc'), 3, 1, 2),                      # rays, 2 of 3 classes
        (sampling_pattern(2, 4, 6, rotations=3, x_ratio=1.5, order='rcd'), 5, 3, 4),   # ellipses, offset
        (sampling_pattern(4, 2, 12), 1, 1, 1),                                  # a single class
    ]
    worst, n = 0.0, 0
    for (pattern, modulus, mod_offset, mod_count), (pairing, stride), gamma, beta, stat, alpha in product(
            cases, (('neighbour', 1), ('same_class', 1), ('none', 1), (3, 1), ('neighbour', 2)),
            GAMMA, BETA, STAT, ALPHA):
        got = general_filter(image, pattern, modulus, mod_offset, mod_count, pairing, gamma, beta, stat,
                             alpha, pair_stride=stride)
        ref = general_filter_reference(image, pattern.offsets(), modulus, mod_offset, mod_count,
                                       pairing, gamma, beta, stat, alpha, stride)
        worst, n = max(worst, rel_err(got, ref)), n + 1
    print(f"generalFilter vs NumPy: {n} combinations, largest relative difference {worst:.1e}")

    pattern = sampling_pattern(3, 3, 8)
    weights = np.random.default_rng(1).normal(size=len(pattern)).astype(np.float32)
    ref = (samples(image, pattern.offsets()) * weights[:, None, None]).sum(0) + 0.5 * image
    print(f"convFilter vs NumPy: largest relative difference {rel_err(conv_filter(image, pattern, weights, 0.5), ref):.1e}")

    stack = np.stack([image, image[::-1].copy()])
    same = np.array_equal(general_filter(stack, pattern, 8)[1], general_filter(stack[1], pattern, 8))
    print(f"image stack gives the same result as single images: {same}")


def check_legacy(image):
    print("diff_filter1..6 vs transcriptions of the kernels in t.cu:")
    # The first circle has radius 0.4: its points all round to the centre pixel, so the
    # pattern contains pairs whose value is exactly 0.
    pattern = sampling_pattern(4, 0.4, 8).offsets()
    pattern = np.concatenate([pattern, sampling_pattern(3, 3, 8).offsets()])
    n = len(pattern)

    def report(name, got, ref, note):
        print(f"  {name}: largest relative difference {rel_err(got, ref):.1e}   ({note})")

    # original skips the pair that starts at point 0; it exists only if offset % modulus == 0
    for offset, modulus in ((3, 4), (5, 2), (1, 3)):
        report(f"diff_filter1 offset={offset} modulus={modulus}", diff_filter1(image, pattern, offset, modulus),
               t_diffFilter1(image, pattern, offset, modulus), "offset not a multiple of modulus")

    # original: only the LAST class counts, and the pairs of all other points count as 0.
    # mod_count = 2 leaves one class in both versions; the zero pairs of the pattern make the
    # extra zeros harmless.
    for modulus in (2, 3, 8):
        report(f"diff_filter2 modulus={modulus} mod_count=2", diff_filter2(image, pattern, modulus, 1, 2),
               t_diffFilter2(image, pattern, modulus, 1, 2), "one class, zero pairs present")
        report(f"diff_filter3 modulus={modulus} mod_count=1", diff_filter3(image, pattern, modulus, 1, 1),
               t_diffFilter3(image, pattern, modulus, 1, 1), "one class, zero pairs present")

    # original truncates the result to an integer
    for modulus, mod_count in ((8, 8), (4, 3), (3, 3), (5, 2)):
        report(f"diff_filter4 modulus={modulus} mod_count={mod_count}",
               np.trunc(diff_filter4(image, pattern, modulus, 1, mod_count) + 1e-4),
               t_diffFilter4(image, pattern, modulus, 1, mod_count), "port truncated like the original")

    # original indexes the class matrix with i*modCount + j: right only for a square matrix
    k = 6
    rng = np.random.default_rng(2)
    classes = [rng.integers(-9, 10, size=(k, 2)) for _ in range(k)]
    offsets, modulus = pattern_from_classes(classes)
    flat = np.concatenate(classes)                                   # class-major, as classX / classY
    report(f"diff_filter5 {k} classes x {k} points", diff_filter5(image, offsets, modulus),
           t_diffFilter5(image, flat[:, 0], flat[:, 1], k, k), "square class matrix")

    # original starts its maximum at -1; the zero pairs keep the true maximum >= 0
    report("diff_filter6", diff_filter6(image, pattern), t_diffFilter6(image, pattern), "zero pairs present")


def main():
    rng = np.random.default_rng(0)
    image = cv2.resize(cv2.imread("circuit.jpg", cv2.IMREAD_GRAYSCALE), (160, 120)).astype(np.float32)
    # integer intensities, as the 16-bit images of the original
    image = np.rint(image + rng.normal(0, 2, image.shape)).astype(np.float32)
    check_general(image)
    check_legacy(image)


if __name__ == "__main__":
    main()
