"""The StructuringElement type shared by all shapes."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class StructuringElement:
    """A neighbourhood around a pixel, given as pixel offsets.

    offsets: (n, 2) int32 array of (dx, dy) offsets from the centre pixel.
    weights: None for a FLAT element (used by rank / morphological filters), or an (n,)
             float32 array for a WEIGHTED element (used by linear filters such as the
             Haar-like features).
    """
    name: str
    offsets: np.ndarray
    weights: Optional[np.ndarray] = None

    def __post_init__(self):
        offsets = np.ascontiguousarray(self.offsets, dtype=np.int32)
        if offsets.ndim != 2 or offsets.shape[1] != 2 or len(offsets) == 0:
            raise ValueError(f"{self.name}: offsets must be a non-empty (n, 2) array")
        object.__setattr__(self, "offsets", offsets)
        if self.weights is not None:
            weights = np.ascontiguousarray(self.weights, dtype=np.float32)
            if weights.shape != (len(offsets),):
                raise ValueError(f"{self.name}: weights must have one value per offset")
            object.__setattr__(self, "weights", weights)

    @property
    def size(self):
        """Number of pixels in the element."""
        return len(self.offsets)

    @property
    def is_flat(self):
        return self.weights is None

    @property
    def radius(self):
        """Largest |dx| or |dy|: how far the element reaches from the centre pixel."""
        return int(np.abs(self.offsets).max())

    def to_mask(self):
        """(2r+1, 2r+1) float32 image of the element: 1 inside for flat elements, the
        weights for weighted ones, 0 elsewhere. Useful for plotting."""
        r = self.radius
        mask = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.float32)
        values = 1.0 if self.is_flat else self.weights
        mask[self.offsets[:, 1] + r, self.offsets[:, 0] + r] = values
        return mask

    @classmethod
    def from_mask(cls, name, mask, weighted=False):
        """Builds an element from a 2D array centred on its middle pixel.

        Flat: every non-zero pixel belongs to the element.
        Weighted: non-zero pixels belong to it and their values become the weights.
        """
        mask = np.asarray(mask)
        cy, cx = mask.shape[0] // 2, mask.shape[1] // 2
        ys, xs = np.nonzero(mask)
        offsets = np.stack([xs - cx, ys - cy], axis=1)
        return cls(name, offsets, mask[ys, xs] if weighted else None)


def centred_grid(half_w, half_h):
    """(dx, dy) coordinate grids covering [-half_w, half_w] x [-half_h, half_h]."""
    dy, dx = np.mgrid[-half_h:half_h + 1, -half_w:half_w + 1]
    return dx, dy
