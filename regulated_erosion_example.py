"""Demonstrates regulated erosion with different strictness values.

Thresholds circuit.jpg to a binary image, erodes it with a disc of radius RADIUS for each
value in STRICTNESS, and writes the results as separate PNGs plus one labelled montage to
regulated_erosion_out/.
"""
import os

import cv2
import numpy as np

from regulated_morph import gen_morph_kernel, regulated_erosion

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "regulated_erosion_out")
RADIUS = 4
# 1 = ordinary erosion; each step tolerates more background pixels under the disc
STRICTNESS = (1, 3, 8, 15, 25, 35)


def labelled(img, text):
    """8-bit copy of img with a caption strip on top."""
    strip = np.zeros((28, img.shape[1]), dtype=np.uint8)
    cv2.putText(strip, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1, cv2.LINE_AA)
    return np.vstack([strip, img])


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    grey = cv2.imread(os.path.join(BASE_DIR, "circuit.jpg"), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = gen_morph_kernel(RADIUS)
    ksize = len(kernel)
    print(f"disc radius {RADIUS}: {ksize} pixels; input foreground {np.count_nonzero(binary)} px")

    tiles = [labelled(binary, "input (Otsu threshold)")]
    cv2.imwrite(os.path.join(OUT_DIR, "input_binary.png"), binary)
    for s in STRICTNESS:
        eroded = np.uint8(regulated_erosion(binary, kernel, strictness=s) * 255)
        print(f"strictness {s:3d} (needs {ksize - s + 1:3d} of {ksize} px): "
              f"{np.count_nonzero(eroded)} foreground px left")
        cv2.imwrite(os.path.join(OUT_DIR, f"eroded_strictness_{s:03d}.png"), eroded)
        tiles.append(labelled(eroded, f"strictness {s} (>= {ksize - s + 1}/{ksize})"))

    # montage, 2 tiles per row
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    cv2.imwrite(os.path.join(OUT_DIR, "montage.png"), np.vstack(rows))
