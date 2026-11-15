"""Weighted structuring elements: Haar-like rectangle features for linear filters.

Each element has a positive and a negative region. The weights are scaled so the positive
region sums to +1 and the negative region to -1: the filter response is
(mean of the positive region) - (mean of the negative region), in [-1, 1] for an image in [0, 1],
and 0 on a constant image.
"""
import numpy as np

from .base import StructuringElement, centred_grid


def _from_signs(name, dx, dy, sign):
    """sign: +1 / -1 / 0 per grid pixel."""
    weights = np.zeros(sign.shape, dtype=np.float32)
    weights[sign > 0] = 1.0 / np.count_nonzero(sign > 0)
    weights[sign < 0] = -1.0 / np.count_nonzero(sign < 0)
    used = sign != 0
    return StructuringElement(name, np.stack([dx[used], dy[used]], axis=1), weights[used])


def _oriented(dx, dy, vertical):
    """(across, along) coordinates: for vertical=True the pattern varies along x."""
    return (dx, dy) if vertical else (dy, dx)


def haar_edge(half, vertical=True):
    """Two-rectangle edge feature in a (2*half) x (2*half) window.

    vertical=True: left half negative, right half positive (responds to vertical edges).
    vertical=False: top half negative, bottom half positive (horizontal edges).
    """
    dy, dx = np.mgrid[-half:half, -half:half]
    across, _ = _oriented(dx, dy, vertical)
    return _from_signs(f"haar_edge_{'v' if vertical else 'h'}_{2 * half}", dx, dy,
                       np.where(across < 0, -1, 1))


def haar_line(band, vertical=True):
    """Three-rectangle line feature: a positive band of width `band` between two negative
    bands of the same width, in a (3*band) x (3*band) window. Responds to bars and ridges."""
    half = (3 * band) // 2
    lo, hi = -half, 3 * band - half
    dy, dx = np.mgrid[lo:hi, lo:hi]
    across, _ = _oriented(dx, dy, vertical)
    centre = (across >= lo + band) & (across < lo + 2 * band)
    return _from_signs(f"haar_line_{'v' if vertical else 'h'}_{3 * band}", dx, dy,
                       np.where(centre, 1, -1))


def haar_center_surround(inner_half, outer_half=None):
    """Positive inner square inside a negative surrounding frame. Responds to blobs."""
    outer_half = 2 * inner_half if outer_half is None else outer_half
    dx, dy = centred_grid(outer_half, outer_half)
    inner = (np.abs(dx) <= inner_half) & (np.abs(dy) <= inner_half)
    return _from_signs(f"haar_center_{2 * outer_half + 1}", dx, dy, np.where(inner, 1, -1))


def haar_checker(half):
    """Four-rectangle checkerboard feature in a (2*half) x (2*half) window. Responds to
    corners and diagonal structure."""
    dy, dx = np.mgrid[-half:half, -half:half]
    return _from_signs(f"haar_checker_{2 * half}", dx, dy, np.where((dx < 0) == (dy < 0), 1, -1))
