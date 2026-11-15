"""Structuring elements for morphological (flat) and linear (weighted) filters."""
from .bank import flat_bank, haar_bank
from .base import StructuringElement
from .haar import haar_center_surround, haar_checker, haar_edge, haar_line
from .shapes import cross, disc, ellipse, line, rectangle, ring, square

__all__ = [
    "StructuringElement",
    "disc", "ring", "ellipse", "rectangle", "square", "line", "cross",
    "haar_edge", "haar_line", "haar_center_surround", "haar_checker",
    "flat_bank", "haar_bank",
]
