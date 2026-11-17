"""Sampling patterns and difference filters on circuit.jpg. Results go to diff_filter_out/.

    python diff_filter_example.py
"""
import os

import cv2
import numpy as np

from diff_filter import conv_filter, general_filter, sampling_pattern

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "diff_filter_out")


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def to_grey(values, symmetric=False):
    """Float response -> 8-bit BGR. symmetric: zero is mid-grey."""
    if symmetric:
        scale = max(np.percentile(np.abs(values), 99.5), 1e-6)
        g = np.clip(0.5 + 0.5 * values / scale, 0, 1)
    else:
        lo, hi = np.percentile(values, 0.5), np.percentile(values, 99.5)
        g = np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.cvtColor(np.uint8(g * 255), cv2.COLOR_GRAY2BGR)


def pattern_sheet():
    """The same geometry in different orders, and an ellipse pattern, coloured by class."""
    tiles = [
        ("circles, order crd, modulus 16", sampling_pattern(3, 4, 16), dict(modulus=16)),
        ("order crd, modulus 2", sampling_pattern(3, 4, 16), dict(modulus=2)),
        ("order drc, modulus 3", sampling_pattern(3, 4, 16, order='drc'), dict(modulus=3)),
        ("modulus 4, mod_count 2", sampling_pattern(3, 4, 16), dict(modulus=4, mod_count=2)),
        ("ellipses 1.5:0.6, 3 rotations", sampling_pattern(3, 4, 12, rotations=3, x_ratio=1.5, y_ratio=0.6,
                                                           order='rcd'), dict(modulus=3)),
    ]
    out = []
    for text, pattern, classes in tiles:
        img = pattern.render(scale=6, **classes)
        img = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT)
        out.append(label(cv2.resize(img, (260, 260), interpolation=cv2.INTER_AREA), text))
    return np.hstack(out)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    image = cv2.imread(os.path.join(BASE_DIR, "circuit.jpg"), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(OUT_DIR, "patterns.png"), pattern_sheet())

    rings = sampling_pattern(circles=3, dist=3, directions=16)                 # walks around each circle
    rays = sampling_pattern(circles=3, dist=3, directions=16, order='drc')     # walks out along each ray

    results = [
        ("input", cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)),
        # largest step between neighbours on a circle: strong where an edge crosses the pattern
        ("rings: max neighbour difference",
         to_grey(general_filter(image, rings, gamma='diff', beta='diff', stat='pooled', alpha='max'))),
        # one class per direction; a class holds the points of that direction on all circles
        ("rings: range of per-direction sums",
         to_grey(general_filter(image, rings, modulus=16, gamma='diff', beta='diff', stat='sum', alpha='range'))),
        # 2 min(a, b) of a pair is large only if BOTH points are brighter than the centre
        ("rings: dark blobs, min over pairs",
         to_grey(general_filter(image, rings, gamma='diff', beta='min2', stat='pooled', alpha='min'))),
        # radial profile: differences along each ray, the same class for each radial step
        ("rays: mean radial difference",
         to_grey(general_filter(image, rays, modulus=3, mod_count=2, gamma='diff', beta='diff',
                                stat='sum', alpha='mean'), symmetric=True)),
        # linear: mean of the outer circle minus the centre (a ring-shaped Laplacian)
        ("conv: outer circle mean - centre",
         to_grey(conv_filter(image, rings, np.where(rings.circle == 2, 1 / 16, 0.0), -1.0), symmetric=True)),
    ]
    for i, (text, img) in enumerate(results):
        cv2.imwrite(os.path.join(OUT_DIR, f"{i}_{text.split(':')[0].replace(' ', '_')}.png"), img)
    tiles = [label(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), text) for text, img in results]
    cv2.imwrite(os.path.join(OUT_DIR, "montage.png"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
