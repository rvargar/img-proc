"""Regulated morphological erosion on the GPU (port of the MATLAB code in gen_morph/)."""
import numpy as np
import pycuda.driver as cuda

from gpu_kernels import BLOCK, get_function, grid_for

EROSION_CU = "regulated_morph_erosion.cu"
FEATURES_CU = "morph_features.cu"

# integer luma weights (sum 256) for the brightness ordering of colours, per channel order
LUMA_WEIGHTS = {'bgr': (29, 150, 77), 'rgb': (77, 150, 29)}


def _check_kernel(kernel, param_name, param):
    """Accepts an (n, 2) offset array or a StructuringElement; validates the rank."""
    kernel = np.ascontiguousarray(getattr(kernel, "offsets", kernel), dtype=np.int32)
    if kernel.ndim != 2 or kernel.shape[1] != 2:
        raise ValueError("kernel must be an (n, 2) array of (dx, dy) offsets")
    ksize = kernel.shape[0]
    if not 1 <= param <= ksize:
        raise ValueError(f"{param_name} must be between 1 and the kernel size {ksize}, got {param}")
    return kernel, ksize


def gen_morph_kernel(radius):
    """Disc-shaped structuring element (port of genMorphKer.m).

    Returns an (n, 2) int32 array of (dx, dy) offsets of all pixels with dx^2 + dy^2 <= radius^2.
    Unlike the MATLAB version it needs no image size: the offsets are 2D, not linear indices.
    """
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    inside = dx ** 2 + dy ** 2 <= radius ** 2
    return np.ascontiguousarray(np.stack([dx[inside], dy[inside]], axis=1), dtype=np.int32)


def regulated_erosion(image, kernel, strictness=1):
    """Regulated erosion of a binary image (port of rMorphErosion2D.m).

    image:      2D array, > 0 means foreground.
    kernel:     (n, 2) int32 (dx, dy) offsets, e.g. from gen_morph_kernel().
    strictness: 1 .. n. A foreground pixel survives if at most strictness - 1 pixels under
                the structuring element are background. 1 is ordinary erosion; larger
                values tolerate more missing pixels, so less is eroded.

    Returns a float32 array of the same shape with 1 where the pixel survives, 0 elsewhere.
    """
    kernel, ksize = _check_kernel(kernel, "strictness", strictness)
    image = np.ascontiguousarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("image must be 2D")
    h, w = image.shape

    min_count = ksize - strictness + 1
    out = np.empty_like(image)
    get_function(EROSION_CU, "regulatedMorphErosion2D")(
        cuda.Out(out), cuda.In(image), cuda.In(kernel),
        np.int32(ksize), np.int32(min_count), np.int32(w), np.int32(h),
        block=BLOCK, grid=grid_for(w, h))
    return out


def rank_filter(image, kernel, rank, clamp_to_centre=False):
    """Order-statistic filter of an 8-bit greyscale image.

    Each output pixel is the rank-th smallest intensity under the structuring element:
    rank 1 = minimum (greyscale erosion), (n + 1) // 2 = median, n = maximum (dilation).
    clamp_to_centre=True limits the result to the pixel's own value, so nothing gets brighter.
    Pixels of the structuring element outside the image count as intensity 0.

    image:  2D uint8 array.
    kernel: (n, 2) int32 (dx, dy) offsets, e.g. from gen_morph_kernel(), or a flat
            StructuringElement from the structuring_elements package.
    Returns a uint8 array of the same shape.
    """
    kernel, ksize = _check_kernel(kernel, "rank", rank)
    image = np.asarray(image)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("image must be a 2D uint8 array")
    image = np.ascontiguousarray(image)
    h, w = image.shape

    out = np.empty_like(image)
    get_function(EROSION_CU, "rankOrderFilter2D")(
        cuda.Out(out), cuda.In(image), cuda.In(kernel),
        np.int32(ksize), np.int32(rank), np.int32(clamp_to_centre), np.int32(w), np.int32(h),
        block=BLOCK, grid=grid_for(w, h))
    return out


def regulated_erosion_gray(image, kernel, strictness=1):
    """Greyscale regulated erosion: min(pixel, strictness-th smallest value under the kernel).

    strictness 1 is ordinary greyscale erosion; larger values ignore the strictness - 1
    darkest pixels under the structuring element, so less is eroded. On a 0/255 image the
    result equals regulated_erosion() with the same strictness (scaled to 0/255).
    """
    return rank_filter(image, kernel, strictness, clamp_to_centre=True)


def rank_filter_color(image, kernel, rank, ordering='luminance', clamp_to_centre=False,
                      channel_order='bgr'):
    """Order-statistic filter of an 8-bit 3-channel colour image.

    Colours have no natural order, so there are two ways to rank them:

    ordering='luminance': VECTOR ordering. The pixels under the structuring element are
        ordered by brightness and the whole colour of the rank-th darkest one is returned.
        Every output colour exists in the neighbourhood, so no false colours appear.
    ordering='marginal':  each channel is rank-filtered on its own (three greyscale
        filters). Simple and exactly the greyscale filter per channel, but it can mix
        channels from different pixels into colours that are not in the image.

    rank 1 = colour erosion, n = colour dilation. clamp_to_centre=True never brightens a
    pixel (per channel for 'marginal', by brightness for 'luminance').
    channel_order: 'bgr' (OpenCV) or 'rgb'; only used for the brightness weights.
    Pixels of the structuring element outside the image count as black.
    Returns a uint8 array of the same shape.
    """
    kernel, ksize = _check_kernel(kernel, "rank", rank)
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be an (h, w, 3) uint8 array")
    image = np.ascontiguousarray(image)

    if ordering == 'marginal':
        return np.dstack([rank_filter(image[..., c], kernel, rank, clamp_to_centre)
                          for c in range(3)])
    if ordering != 'luminance':
        raise ValueError(f"ordering must be 'luminance' or 'marginal', got {ordering!r}")

    h, w = image.shape[:2]
    out = np.empty_like(image)
    luma = np.array(LUMA_WEIGHTS[channel_order], dtype=np.int32)
    get_function(FEATURES_CU, "vectorRankFilter")(
        cuda.Out(out), cuda.In(image), cuda.In(kernel),
        np.int32(ksize), np.int32(rank), np.int32(clamp_to_centre), cuda.In(luma),
        np.int32(w), np.int32(h), np.int32(0),
        block=BLOCK, grid=grid_for(w, h))
    return out


def regulated_erosion_color(image, kernel, strictness=1, ordering='luminance', channel_order='bgr'):
    """Colour regulated erosion: rank_filter_color() that never brightens a pixel.

    strictness 1 is ordinary colour erosion; larger values ignore the strictness - 1
    darkest pixels under the structuring element, so less is eroded.
    """
    return rank_filter_color(image, kernel, strictness, ordering, clamp_to_centre=True,
                             channel_order=channel_order)
