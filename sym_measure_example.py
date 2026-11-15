import os

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401  (creates the CUDA context)
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_PATH = os.path.join(BASE_DIR, "cuda", "sym_measure.cu")
BLOCK = (32, 32, 1)
MAX_HALF_WIDTH = 20  # largest symmetric half-width tested per pixel, in pixels (w_max = 2K+1)

input_img = cv2.imread(os.path.join(BASE_DIR, "street_view.jpg"), cv2.IMREAD_GRAYSCALE)
rows, cols = input_img.shape

with open(KERNEL_PATH) as f:
    kernel_code = f.read()

mod = SourceModule(kernel_code, arch='sm_80')
image_tex = mod.get_texref('image')
cuda.matrix_to_texref(np.float32(input_img) / 255.0, image_tex, order='C')
image_tex.set_filter_mode(cuda.filter_mode.LINEAR)
image_tex.set_address_mode(0, cuda.address_mode.CLAMP)
image_tex.set_address_mode(1, cuda.address_mode.CLAMP)
func = mod.get_function("calcSymmetryGpuH")
hist_func = mod.get_function("calcSymmetryHistogram")
seed_func = mod.get_function("calcSeed")

# grid.x covers columns, grid.y covers rows (kernel: x = column, y = row)
grid = ((cols + BLOCK[0] - 1) // BLOCK[0], (rows + BLOCK[1] - 1) // BLOCK[1])

result = np.zeros((3, rows, cols), dtype=np.float32)
result_gpu = cuda.mem_alloc(result.nbytes)
angle_gpu = cuda.mem_alloc(rows * cols * 4)
half_width_gpu = cuda.mem_alloc(rows * cols * 4)


def symmetry_image(angle, max_half_width=MAX_HALF_WIDTH):
    """Zielke et al. (ECCV'92) intensity symmetry feature for EVERY pixel.

    angle:          scan-line direction in radians, scalar or (rows, cols) array.
                    0 -> scan along y (horizontal axis), pi/2 -> scan along x (vertical axis).
    max_half_width: largest half-width to test in pixels, scalar or (rows, cols) array.

    Returns (sa, width, s), each (rows, cols) float32:
      sa:    max over w of SA(xs, w), confidence of a symmetry axis at the pixel, [0, 1]
      width: interval width w (pixels) where that max occurs
      s:     symmetry measure S at that width, [-1, 1]
    """
    angles = np.broadcast_to(np.asarray(angle, dtype=np.float32), (rows, cols))
    widths = np.broadcast_to(np.asarray(max_half_width, dtype=np.float32), (rows, cols))
    cuda.memcpy_htod(angle_gpu, np.ascontiguousarray(angles))
    cuda.memcpy_htod(half_width_gpu, np.ascontiguousarray(widths))
    func(result_gpu, angle_gpu, half_width_gpu, np.int32(cols), np.int32(rows),
         block=BLOCK, grid=grid)
    out = np.empty_like(result)
    cuda.memcpy_dtoh(out, result_gpu)
    return out[0], out[1], out[2]


def symmetry_histogram(start=0, stop=None, max_half_width=MAX_HALF_WIDTH, horizontal=False):
    """Symmetry histogram (paper section 2.2, Fig. 2c).

    horizontal=False: VERTICAL axes. Sums SA over the image rows start <= y < stop,
                      for every axis column.
    horizontal=True:  HORIZONTAL axes. Sums SA over the image columns start <= x < stop,
                      for every axis row.

    Returns (hist2d, hist), with n_axis = cols (vertical) or rows (horizontal):
      hist2d: (max_half_width, n_axis) float32; hist2d[k-1, a] = summed SA(a, w = 2k+1)
      hist:   (n_axis,) float32, max over widths; its peaks are the symmetry-axis positions
    """
    n_axis, n_scan = (rows, cols) if horizontal else (cols, rows)
    stop = n_scan if stop is None else stop
    K = int(max_half_width)
    hist2d = np.zeros((K, n_axis), dtype=np.float32)
    hist_gpu = cuda.mem_alloc(hist2d.nbytes)
    hist_func(hist_gpu, np.int32(K), np.int32(start), np.int32(stop), np.int32(horizontal),
              np.int32(cols), np.int32(rows),
              block=(64, 1, 1), grid=((n_axis + 63) // 64, 1))
    cuda.memcpy_dtoh(hist2d, hist_gpu)
    hist_gpu.free()
    return hist2d, hist2d.max(axis=0)


SEED_N = 8  # must match SEED_N in the kernel
# compatibility weights per category of direction pair (paper section 3.2)
SEED_WEIGHTS = {'Z': -2.0, 'X': 2.0, 'M': 2.0, 'O': 0.0, 'Y': 1.0, 'N': 0.0}


def seed_category(a, b):
    """Category of a (left, right) pair of gradient directions, paper Fig. 3.

    a, b: direction indices, gradient pointing along a*45 and b*45 degrees.
    X ideal symmetry (mirror images), M inverse symmetry (mirrored contour, opposite
    contrast), Y / N within 45 degrees of X / M, Z antisymmetry, O non-symmetry.
    """
    s = (a + b) % SEED_N
    if s in (0, 4) and a % 4 == 2:   # both gradients vertical: neutral in the paper's table
        return 'O'
    return {4: 'X', 0: 'M', 3: 'Y', 5: 'Y', 1: 'N', 7: 'N', 2: 'Z', 6: 'Z'}[s]


def seed_weight_matrix(weights=SEED_WEIGHTS):
    """(W, k): W[i, j] for left direction i and right direction j, and the factor k (eq. 8).

    k is the largest element of W.A with a_ij = |cos(2*pi*(i-j)/n)|, as in the paper.
    """
    W = np.array([[weights[seed_category(a, b)] for b in range(SEED_N)] for a in range(SEED_N)],
                 dtype=np.float32)
    i, j = np.indices((SEED_N, SEED_N))
    A = np.abs(np.cos(2 * np.pi * (i - j) / SEED_N))
    return W, float((W @ A).max())


def default_seed_threshold():
    """T: half the average gradient magnitude at edges (paper's rule of thumb).

    "Edges" are taken as the 10 percent of pixels with the strongest Sobel response;
    units are grey levels per pixel, matching the kernel.
    """
    img = np.float32(input_img)
    mag = np.hypot(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3),
                   cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)) / 4.0
    return 0.5 * float(mag[mag >= np.percentile(mag, 90)].mean())


def seed_image(axis_pos, max_dist=0.0, T=None, weights=SEED_WEIGHTS, horizontal=False):
    """SEED edge image for a symmetry axis (paper eq. 7-8).

    horizontal=False: vertical axis at column axis_pos.
    horizontal=True:  horizontal axis at row axis_pos.
    Keeps edges that have a symmetric counterpart across the axis, suppresses the rest.
    max_dist: only pixels within this distance of the axis (0 = no limit).
    Returns a (rows, cols) float32 array in grey levels per pixel.
    """
    W, k = seed_weight_matrix(weights)
    T = default_seed_threshold() if T is None else T
    out = np.zeros((rows, cols), dtype=np.float32)
    out_gpu = cuda.mem_alloc(out.nbytes)
    w_gpu = cuda.mem_alloc(W.nbytes)
    cuda.memcpy_htod(w_gpu, W)
    # +0.5: the kernel works in texel-centre coordinates
    seed_func(out_gpu, w_gpu, np.float32(axis_pos + 0.5), np.float32(max_dist),
              np.float32(T), np.float32(k), np.int32(horizontal), np.int32(cols), np.int32(rows),
              block=BLOCK, grid=grid)
    cuda.memcpy_dtoh(out, out_gpu)
    out_gpu.free()
    w_gpu.free()
    return out


def symmetry_angle_series(n_steps=30, angle_from=0.0, angle_to=np.pi / 2.0, out_dir=None):
    """Greyscale series of the per-pixel axis confidence SA over n_steps scan angles.

    Writes one 8-bit PNG per angle to out_dir and, if Pillow is installed, an animated GIF
    next to them. All frames share one grey scale (SA = 1 -> white), so they are comparable.
    Returns (angles, frames) with frames a (n_steps, rows, cols) uint8 array.
    """
    out_dir = os.path.join(BASE_DIR, "sym_series") if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)
    angles = np.linspace(angle_from, angle_to, n_steps)
    frames = np.zeros((n_steps, rows, cols), dtype=np.uint8)
    for i, angle in enumerate(angles):
        sa, _, _ = symmetry_image(angle)
        frames[i] = np.uint8(np.clip(sa, 0, 1) * 255)
        cv2.imwrite(os.path.join(out_dir, f"sym_{i:02d}_{np.degrees(angle):05.1f}deg.png"), frames[i])

    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed: wrote PNG frames only, no GIF")
    else:
        images = [Image.fromarray(f) for f in frames]
        images[0].save(os.path.join(out_dir, "sym_series.gif"), save_all=True,
                       append_images=images[1:], duration=150, loop=0)
    return angles, frames


def _report_axis(horizontal):
    """Histogram -> strongest axis -> SEED, for one axis orientation; writes two PNGs."""
    name = "horizontal" if horizontal else "vertical"
    hist2d, hist = symmetry_histogram(horizontal=horizontal)
    axis_pos = int(np.argmax(hist))
    axis_k = int(np.argmax(hist2d[:, axis_pos])) + 1
    print(f"strongest {name} symmetry axis: {'row' if horizontal else 'column'} {axis_pos}, "
          f"interval width {2 * axis_k + 1} px")
    cv2.imwrite(os.path.join(BASE_DIR, f'result_hist_{name}.png'),
                np.uint8(hist2d / max(hist2d.max(), 1e-9) * 255))

    seed = seed_image(axis_pos, max_dist=axis_k, horizontal=horizontal)
    cv2.imwrite(os.path.join(BASE_DIR, f'result_seed_{name}.png'), np.uint8(np.clip(seed, 0, 255)))


if __name__ == "__main__":
    # 1. greyscale series of the per-pixel axis confidence SA, 30 scan angles in [0, pi/2]
    angles, frames = symmetry_angle_series(n_steps=30)
    for angle, frame in zip(angles, frames):
        print(f"angle {np.degrees(angle):5.1f} deg: SA max {frame.max() / 255:.3f}, mean {frame.mean() / 255:.3f}")

    # 2 + 3. row/column-summed symmetry histogram and SEED, for vertical and horizontal axes
    _report_axis(horizontal=False)
    _report_axis(horizontal=True)
