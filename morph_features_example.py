"""Demonstrates colour regulated erosion, the structuring-element bank and MorphFeatureMap.

Reads bumble.jpg and writes to morph_features_out/:
  color_erosion_montage.png  colour regulated erosion at several strictness values
  se_bank.png                the structuring elements of the default bank
  feature_sheet.png          a contact sheet of feature planes
"""
import os

import cv2
import numpy as np

from morph_features import MorphFeatureMap
from regulated_morph import regulated_erosion_color
from structuring_elements import disc, flat_bank, haar_bank

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "morph_features_out")
STRICTNESS = (1, 8, 25)   # for a radius-4 disc (49 pixels)


def labelled(img, text):
    """Copy of a BGR image with a caption strip on top."""
    strip = np.zeros((26, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(strip, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def montage(tiles, per_row):
    """Grid of equally sized BGR tiles, padded with black tiles."""
    tiles = list(tiles) + [np.zeros_like(tiles[0])] * (-len(tiles) % per_row)
    return np.vstack([np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)])


def element_tile(element, cell=6, size=150):
    """Picture of a structuring element: white = in the element / positive weight,
    black = negative weight, grey = not in the element."""
    grey = np.uint8(np.clip(127 + 128 * np.sign(element.to_mask()), 0, 255))
    grey = cv2.resize(grey, None, fx=cell, fy=cell, interpolation=cv2.INTER_NEAREST)
    tile = np.full((size, size), 127, dtype=np.uint8)
    y, x = (size - grey.shape[0]) // 2, (size - grey.shape[1]) // 2
    tile[y:y + grey.shape[0], x:x + grey.shape[1]] = grey
    return labelled(cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR), element.name)


def feature_tile(plane, name, width=380):
    """Feature plane as an 8-bit tile; signed (Haar) planes are centred on mid-grey."""
    lo = -1.0 if plane.min() < 0 else 0.0
    grey = np.uint8(np.clip((plane - lo) / (1.0 - lo), 0, 1) * 255)
    grey = cv2.resize(grey, (width, round(plane.shape[0] * width / plane.shape[1])),
                      interpolation=cv2.INTER_AREA)
    return labelled(cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), name)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    image = cv2.imread(os.path.join(BASE_DIR, "bumble.jpg"))   # BGR

    # 1. colour regulated erosion: vector ordering by brightness vs. per-channel ordering
    element = disc(4)
    tiles = [labelled(image, "input")]
    for ordering in ("luminance", "marginal"):
        for s in STRICTNESS:
            eroded = regulated_erosion_color(image, element, strictness=s, ordering=ordering)
            tiles.append(labelled(eroded, f"{ordering} ordering, strictness {s} of {element.size}"))
    cv2.imwrite(os.path.join(OUT_DIR, "color_erosion_montage.png"), montage(tiles, per_row=2))

    # 2. the structuring elements of the default bank
    bank = flat_bank() + haar_bank()
    cv2.imwrite(os.path.join(OUT_DIR, "se_bank.png"),
                montage([element_tile(e) for e in bank], per_row=5))

    # 3. feature map
    feature_map = MorphFeatureMap()
    features = feature_map.compute(image)
    print(feature_map)
    print(f"feature stack: shape {features.shape}, dtype {features.dtype}, "
          f"{features.nbytes / 1e6:.0f} MB, range [{features.min():.2f}, {features.max():.2f}]")

    shown = ["raw/r", "disc_r5/q0.1/r", "disc_r5/q0.9/r", "disc_r5/range/r",
             "ring_r5/q0.5/g", "ellipse_5x2_a0/range/g", "ellipse_5x2_a90/range/g",
             "haar_edge_v_10/g", "haar_edge_h_10/g", "haar_line_v_15/b",
             "haar_center_21/b", "haar_checker_10/b"]
    tiles = [feature_tile(features[feature_map.feature_names.index(n)], n) for n in shown]
    cv2.imwrite(os.path.join(OUT_DIR, "feature_sheet.png"), montage(tiles, per_row=3))
