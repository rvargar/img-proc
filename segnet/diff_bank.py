"""Bank of difference filters (diff_filter.py) used as an input feature group of MorphSymNet.

Every filter is one setting of general_filter() on a pattern of 3 circles x 16 directions.
With the order 'crd' and modulus 16 a class is one DIRECTION: its three points lie on a ray.
"""
import numpy as np

from diff_filter import sampling_pattern

DIRECTIONS, CIRCLES = 16, 3

# name -> (general_filter arguments, divisor that brings the response to the intensity scale)
FILTERS = {
    # largest step between neighbouring points on the circles: an edge crosses the pattern
    'edge':        (dict(gamma='diff', beta='diff', stat='pooled', alpha='max'), 1.0),
    # angular differences summed per direction over the circles, range over the directions
    'dirsum':      (dict(modulus=DIRECTIONS, gamma='diff', beta='diff', stat='sum', alpha='range'), CIRCLES),
    # min over all points of (v - c): positive only if the whole surround is brighter
    'dark_blob':   (dict(gamma='diff', beta='min2', stat='pooled', alpha='min'), 2.0),
    'bright_blob': (dict(gamma='neg_diff', beta='min2', stat='pooled', alpha='min'), 2.0),
    # diff_filter5: some direction on which every point differs from the centre
    'ray_minmax':  (dict(modulus=DIRECTIONS, pairing='none', gamma='abs_diff', stat='min', alpha='max'), 1.0),
    # the direction most similar to the centre: small on lines and edges, large on corners and noise
    'ray_best':    (dict(modulus=DIRECTIONS, pairing='none', gamma='abs_diff', stat='sum', alpha='min'), CIRCLES),
    # anisotropy: most different direction minus most similar direction
    'ray_range':   (dict(modulus=DIRECTIONS, pairing='none', gamma='abs_diff', stat='sum', alpha='range'), CIRCLES),
    # diff_filter4: strongest direction pair that is brighter than the centre on all circles
    'pair_bright': (dict(modulus=DIRECTIONS, pair_stride=2, gamma='pos_diff', beta='min2', stat='sum',
                         alpha='max'), 2.0 * CIRCLES),
}
POOLED = ('edge', 'dark_blob', 'bright_blob')                       # filters without classes
BY_DIRECTION = ('dirsum', 'ray_minmax', 'ray_best', 'ray_range', 'pair_bright')

# bank name -> (distances between circles in pixels, filter names)
BANKS = {
    'full':      ((2, 4), tuple(FILTERS)),
    'small':     ((2,), tuple(FILTERS)),
    'large':     ((4,), tuple(FILTERS)),
    'pooled':    ((2, 4), POOLED),
    'direction': ((2, 4), BY_DIRECTION),
}


def diff_bank(name='full'):
    """List of (plane name, (n, 2) int32 offsets, general_filter arguments, divisor)."""
    if name not in BANKS:
        raise ValueError(f"diff bank must be one of {sorted(BANKS)}, got {name!r}")
    dists, filters = BANKS[name]
    bank = []
    for dist in dists:
        offsets = np.ascontiguousarray(sampling_pattern(CIRCLES, dist, DIRECTIONS).offsets())
        bank += [(f"diff/{f}/d{dist}", offsets, FILTERS[f][0], FILTERS[f][1]) for f in filters]
    return bank
