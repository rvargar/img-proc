"""Demonstrates greyscale regulated erosion (order-statistic filter) with different strictness.

Erodes circuit.jpg with a disc of radius RADIUS for each value in STRICTNESS and writes the
results as separate PNGs plus one labelled montage to regulated_erosion_gray_out/.
"""
import os

import cv2
import numpy as np

from regulated_morph import gen_morph_kernel, rank_filter, regulated_erosion_gray

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "regulated_erosion_gray_out")
RADIUS = 4
# 1 = ordinary greyscale erosion; each step ignores more of the darkest pixels under the disc
STRICTNESS = (1, 3, 8, 15, 25, 35)


def labelled(img, text):
    """Copy of img with a caption strip on top."""
    strip = np.zeros((28, img.shape[1]), dtype=np.uint8)
    cv2.putText(strip, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1, cv2.LINE_AA)
    return np.vstack([strip, img])


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    grey = cv2.imread(os.path.join(BASE_DIR, "circuit.jpg"), cv2.IMREAD_GRAYSCALE)
    kernel = gen_morph_kernel(RADIUS)
    ksize = len(kernel)
    print(f"disc radius {RADIUS}: {ksize} pixels; input mean intensity {grey.mean():.1f}")

    tiles = [labelled(grey, "input")]
    for s in STRICTNESS:
        eroded = regulated_erosion_gray(grey, kernel, strictness=s)
        print(f"strictness {s:3d}: mean intensity {eroded.mean():.1f}")
        cv2.imwrite(os.path.join(OUT_DIR, f"eroded_strictness_{s:03d}.png"), eroded)
        tiles.append(labelled(eroded, f"strictness {s} of {ksize}"))

    # the plain rank filter at the middle rank is a median filter
    median = rank_filter(grey, kernel, (ksize + 1) // 2)
    cv2.imwrite(os.path.join(OUT_DIR, "median.png"), median)
    tiles.append(labelled(median, f"rank filter, median (rank {(ksize + 1) // 2})"))

    # montage, 2 tiles per row
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    cv2.imwrite(os.path.join(OUT_DIR, "montage.png"), np.vstack(rows))
