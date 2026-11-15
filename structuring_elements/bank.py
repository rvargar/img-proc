"""Ready-made filter banks: lists of structuring elements at several scales."""
from .haar import haar_center_surround, haar_checker, haar_edge, haar_line
from .shapes import disc, ellipse, ring


def flat_bank(scales=(2, 5)):
    """Flat elements per scale s: a disc, a ring, and a horizontal and a vertical ellipse."""
    bank = []
    for s in scales:
        minor = max(1, s // 2)
        bank += [disc(s), ring(s), ellipse(s, minor, 0), ellipse(s, minor, 90)]
    return bank


def haar_bank(scales=(2, 5)):
    """Haar-like elements per scale s: vertical and horizontal edge and line features,
    a centre-surround feature and a checkerboard feature."""
    bank = []
    for s in scales:
        bank += [haar_edge(s, True), haar_edge(s, False),
                 haar_line(s, True), haar_line(s, False),
                 haar_center_surround(s), haar_checker(s)]
    return bank
