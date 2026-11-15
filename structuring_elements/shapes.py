"""Flat structuring elements: geometric shapes for rank and morphological filters."""
import numpy as np

from .base import StructuringElement, centred_grid


def _from_condition(name, half_w, half_h, condition):
    dx, dy = centred_grid(half_w, half_h)
    inside = condition(dx, dy)
    inside[half_h, half_w] = True   # the centre pixel always belongs to the element
    return StructuringElement(name, np.stack([dx[inside], dy[inside]], axis=1))


def disc(radius):
    """All pixels within `radius` of the centre."""
    return _from_condition(f"disc_r{radius}", radius, radius,
                           lambda dx, dy: dx ** 2 + dy ** 2 <= radius ** 2)


def ring(radius, inner_radius=None):
    """Annulus: inner_radius < distance <= radius (default inner radius: radius / 2).

    The centre pixel is included, so the ring measures the pixel against its surround.
    """
    inner_radius = radius / 2.0 if inner_radius is None else inner_radius
    return _from_condition(f"ring_r{radius}", radius, radius,
                           lambda dx, dy: (dx ** 2 + dy ** 2 <= radius ** 2)
                           & (dx ** 2 + dy ** 2 > inner_radius ** 2))


def ellipse(semi_major, semi_minor, angle_deg=0.0):
    """Filled ellipse with the major axis rotated by angle_deg (0 = along x, 90 = along y)."""
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    half = int(np.ceil(max(semi_major, semi_minor)))

    def inside(dx, dy):
        u = dx * c + dy * s      # coordinate along the major axis
        v = -dx * s + dy * c     # coordinate along the minor axis
        return (u / semi_major) ** 2 + (v / semi_minor) ** 2 <= 1.0

    return _from_condition(f"ellipse_{semi_major}x{semi_minor}_a{angle_deg:g}", half, half, inside)


def rectangle(half_w, half_h):
    """Filled (2*half_w+1) x (2*half_h+1) rectangle."""
    return _from_condition(f"rect_{2 * half_w + 1}x{2 * half_h + 1}", half_w, half_h,
                           lambda dx, dy: np.ones_like(dx, dtype=bool))


def square(half):
    """Filled (2*half+1) x (2*half+1) square."""
    return rectangle(half, half)


def line(half_length, angle_deg=0.0):
    """One-pixel-wide line through the centre, 2*half_length+1 long, rotated by angle_deg."""
    t = np.radians(angle_deg)
    steps = np.arange(-half_length, half_length + 1)
    offsets = np.unique(np.stack([np.rint(steps * np.cos(t)), np.rint(steps * np.sin(t))], axis=1), axis=0)
    return StructuringElement(f"line_h{half_length}_a{angle_deg:g}", offsets)


def cross(half):
    """Horizontal plus vertical line through the centre."""
    return _from_condition(f"cross_{2 * half + 1}", half, half, lambda dx, dy: (dx == 0) | (dy == 0))
