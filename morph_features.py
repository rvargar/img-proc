"""Feature maps for image segmentation: morphological and intensity features per colour channel.

MorphFeatureMap turns an image into a stack of per-pixel feature planes, computed on the GPU
with a bank of structuring elements. The stack is channel-first float32, ready to be the
input of a (shallow) convolutional network.
"""
import cv2
import numpy as np
import pycuda.driver as cuda

from gpu_kernels import BLOCK, get_function, grid_for
from structuring_elements import flat_bank, haar_bank

FEATURES_CU = "morph_features.cu"

# colour space -> (conversion from BGR, channel names)
COLOR_SPACES = {
    'bgr':  (None,               ('b', 'g', 'r')),
    'rgb':  (cv2.COLOR_BGR2RGB,  ('r', 'g', 'b')),
    'lab':  (cv2.COLOR_BGR2LAB,  ('L', 'a', 'b')),
    'hsv':  (cv2.COLOR_BGR2HSV,  ('h', 's', 'v')),
    'gray': (cv2.COLOR_BGR2GRAY, ('gray',)),
}


class MorphFeatureMap:
    """Per-pixel feature stack from a bank of structuring elements.

    For every colour channel the stack contains:

    * raw intensity (optional)
    * for each FLAT structuring element: one plane per quantile (order statistics of the
      intensities under the element: a low quantile is a noise-tolerant erosion, 0.5 the
      median, a high quantile a noise-tolerant dilation), and optionally their range
      (highest minus lowest quantile: a morphological gradient, i.e. local contrast)
    * for each WEIGHTED structuring element: its linear response (Haar-like features:
      mean of the positive region minus mean of the negative region)

    Value ranges: quantile, raw and range planes are in [0, 1], Haar planes in [-1, 1].

    Parameters
    ----------
    flat_elements, weighted_elements:
        Lists of StructuringElement. Defaults: structuring_elements.flat_bank() and
        haar_bank(). Pass an empty list to leave a group out.
    quantiles:
        Fractions in [0, 1] for the flat elements; 0 = minimum, 1 = maximum.
    color_space:
        'bgr', 'rgb', 'lab', 'hsv' or 'gray'. The features are computed on its channels.
    include_raw, include_range:
        Whether to add the raw channels and the quantile range.

    compute(image) takes an 8-bit BGR image (as cv2.imread returns) or a 2D grey image and
    returns a float32 array of shape (n_features, height, width). feature_names lists the
    planes in order.
    """

    def __init__(self, flat_elements=None, weighted_elements=None, quantiles=(0.1, 0.5, 0.9),
                 color_space='rgb', include_raw=True, include_range=True):
        if color_space not in COLOR_SPACES:
            raise ValueError(f"color_space must be one of {sorted(COLOR_SPACES)}, got {color_space!r}")
        self.flat_elements = flat_bank() if flat_elements is None else list(flat_elements)
        self.weighted_elements = haar_bank() if weighted_elements is None else list(weighted_elements)
        self.quantiles = tuple(sorted(quantiles))
        self.color_space = color_space
        self.include_raw = include_raw
        self.include_range = include_range and len(self.quantiles) > 1

        for e in self.flat_elements:
            if not e.is_flat:
                raise ValueError(f"{e.name} is weighted; put it in weighted_elements")
            if e.size > 65535:
                raise ValueError(f"{e.name} has more than 65535 pixels")
        for e in self.weighted_elements:
            if e.is_flat:
                raise ValueError(f"{e.name} is flat; put it in flat_elements")
        if self.flat_elements and not self.quantiles:
            raise ValueError("quantiles must not be empty when flat elements are used")
        if any(not 0.0 <= q <= 1.0 for q in self.quantiles):
            raise ValueError("quantiles must be in [0, 1]")

        self.channel_names = COLOR_SPACES[color_space][1]
        self.feature_names = self._build_names()

    # ------------------------------------------------------------------ layout

    def _build_names(self):
        ch = self.channel_names
        names = []
        if self.include_raw:
            names += [f"raw/{c}" for c in ch]
        for e in self.flat_elements:
            for q in self.quantiles:
                names += [f"{e.name}/q{q:g}/{c}" for c in ch]
            if self.include_range:
                names += [f"{e.name}/range/{c}" for c in ch]
        for e in self.weighted_elements:
            names += [f"{e.name}/{c}" for c in ch]
        return names

    @property
    def n_features(self):
        return len(self.feature_names)

    @property
    def n_channels(self):
        return len(self.channel_names)

    @staticmethod
    def ranks_for(element, quantiles):
        """1-based ranks of the quantiles within an element of element.size pixels."""
        return np.array([1 + int(round(q * (element.size - 1))) for q in quantiles], dtype=np.int32)

    # ----------------------------------------------------------------- compute

    def convert(self, image):
        """Converts an 8-bit BGR (or 2D grey) image to (h, w, n_channels) in the colour space."""
        image = np.asarray(image)
        if image.dtype != np.uint8:
            raise ValueError("image must be uint8")
        if image.ndim == 2:
            if self.n_channels != 1:
                raise ValueError(f"a 2D image needs color_space='gray', not {self.color_space!r}")
            return np.ascontiguousarray(image[..., None])
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be (h, w, 3) BGR or (h, w) grey")
        code = COLOR_SPACES[self.color_space][0]
        converted = image if code is None else cv2.cvtColor(image, code)
        return np.ascontiguousarray(converted.reshape(image.shape[0], image.shape[1], -1))

    def compute(self, image):
        """Returns the (n_features, height, width) float32 feature stack of the image."""
        img = self.convert(image)
        h, w, C = img.shape
        features = np.empty((self.n_features, h, w), dtype=np.float32)
        grid = grid_for(w, h)
        replicate = np.int32(1)   # border pixels reuse the nearest image pixel

        img_gpu = cuda.mem_alloc(img.nbytes)
        cuda.memcpy_htod(img_gpu, img)
        buffers = [img_gpu]
        try:
            pos = 0
            if self.include_raw:
                features[:C] = img.transpose(2, 0, 1) / np.float32(255.0)
                pos = C

            nq = len(self.quantiles)
            if self.flat_elements:
                quantile_filter = get_function(FEATURES_CU, "quantileFilter")
                out_gpu = cuda.mem_alloc(nq * C * h * w * 4)
                buffers.append(out_gpu)
            for e in self.flat_elements:
                ranks = self.ranks_for(e, self.quantiles)
                quantile_filter(out_gpu, img_gpu, cuda.In(e.offsets), np.int32(e.size),
                                cuda.In(ranks), np.int32(nq), np.int32(w), np.int32(h),
                                np.int32(C), replicate, block=BLOCK, grid=grid)
                block = features[pos:pos + nq * C]
                cuda.memcpy_dtoh(block, out_gpu)
                pos += nq * C
                if self.include_range:
                    features[pos:pos + C] = block[(nq - 1) * C:] - block[:C]
                    pos += C

            if self.weighted_elements:
                weighted_filter = get_function(FEATURES_CU, "weightedFilter")
                lin_gpu = cuda.mem_alloc(C * h * w * 4)
                buffers.append(lin_gpu)
            for e in self.weighted_elements:
                weighted_filter(lin_gpu, img_gpu, cuda.In(e.offsets), cuda.In(e.weights),
                                np.int32(e.size), np.int32(w), np.int32(h), np.int32(C), replicate,
                                block=BLOCK, grid=grid)
                cuda.memcpy_dtoh(features[pos:pos + C], lin_gpu)
                pos += C
        finally:
            for b in buffers:
                b.free()
        return features

    def __call__(self, image):
        return self.compute(image)

    def __repr__(self):
        return (f"MorphFeatureMap({self.n_features} features: {self.n_channels} channels "
                f"({self.color_space}), {len(self.flat_elements)} flat x "
                f"{len(self.quantiles)} quantiles, {len(self.weighted_elements)} weighted)")
