// Multi-channel morphological and linear filters over arbitrary structuring elements.
// Used by regulated_morph.py (colour rank filter) and morph_features.py (feature maps).
//
// Images are 8-bit, interleaved: img[(x + y*w)*C + c], x = column, y = row.
// A structuring element is ksize (dx, dy) offset pairs: offsets[2*i], offsets[2*i+1].
//
// Border handling, for element pixels that fall outside the image:
//   replicate != 0 : use the nearest image pixel
//   replicate == 0 : the pixel counts as intensity 0 (black), as in regulated_morph_erosion.cu

#define MAX_CHANNELS 4

// Reads channel c at (nx, ny) with the border rule above.
__device__ int read_px(const unsigned char* img, int nx, int ny, int c, int w, int h, int C,
                       int replicate)
{
    if (replicate)
    {
        nx = min(max(nx, 0), w - 1);
        ny = min(max(ny, 0), h - 1);
    }
    else if (nx < 0 || nx >= w || ny < 0 || ny >= h)
        return 0;
    return img[(nx + ny*w)*C + c];
}


// Several order statistics of every channel in one pass.
//
// For each pixel and channel, builds the 256-bin histogram of the intensities under the
// structuring element and reads the requested ranks off its cumulative sum. One histogram
// serves all ranks, so asking for more ranks costs almost nothing.
//
//   ranks : n_ranks ranks, 1-based (1 = minimum, ksize = maximum), in ASCENDING order
//   out   : n_ranks * C * h * w floats in [0, 1];  out[(r*C + c)*w*h + x + y*w]
__global__ void quantileFilter(float* out, const unsigned char* img, const int* offsets, int ksize,
                               const int* ranks, int n_ranks, int w, int h, int C, int replicate)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int plane = w*h;
    int idx   = x + y*w;
    unsigned short hist[256];

    for (int c = 0; c < C; c++)
    {
        for (int v = 0; v < 256; v++) hist[v] = 0;
        for (int i = 0; i < ksize; i++)
            hist[read_px(img, x + offsets[2*i], y + offsets[2*i + 1], c, w, h, C, replicate)]++;

        int cum = 0, r = 0;
        for (int v = 0; v < 256 && r < n_ranks; v++)
        {
            cum += hist[v];
            while (r < n_ranks && cum >= ranks[r])
            {
                out[(r*C + c)*plane + idx] = v / 255.0f;
                r++;
            }
        }
    }
}


// Linear filter with a weighted structuring element (e.g. Haar-like features, local mean).
//
//   weights : ksize floats, one per offset
//   out     : C * h * w floats;  out[c*w*h + x + y*w] = sum_i weights[i] * img(...)/255
__global__ void weightedFilter(float* out, const unsigned char* img, const int* offsets,
                               const float* weights, int ksize, int w, int h, int C, int replicate)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float sum[MAX_CHANNELS];
    for (int c = 0; c < C; c++) sum[c] = 0.0f;

    for (int i = 0; i < ksize; i++)
    {
        int nx = x + offsets[2*i], ny = y + offsets[2*i + 1];
        for (int c = 0; c < C; c++)
            sum[c] += weights[i] * read_px(img, nx, ny, c, w, h, C, replicate);
    }

    int plane = w*h;
    int idx   = x + y*w;
    for (int c = 0; c < C; c++)
        out[c*plane + idx] = sum[c] / 255.0f;
}


// Rank filter for 3-channel colour images with VECTOR ordering by brightness.
//
// Colours have no natural order, so the pixels under the structuring element are ordered by
// a scalar key, key = (lw[0]*c0 + lw[1]*c1 + lw[2]*c2) >> 8 (a luma with integer weights that
// sum to 256). The output is the whole colour of the pixel whose key has the requested rank,
// so every output colour is a colour that exists in the neighbourhood: no false colours.
// Ties in the key are resolved by the order of the offsets.
//
//   rank = 1 : darkest pixel   -> colour erosion
//   rank = ksize : brightest   -> colour dilation
// With clamp_to_centre != 0 the pixel keeps its own colour when it is darker than the
// selected one (regulated erosion: the filter never brightens a pixel).
//
//   img, out : w*h*3 unsigned chars, interleaved
__global__ void vectorRankFilter(unsigned char* out, const unsigned char* img, const int* offsets,
                                 int ksize, int rank, int clamp_to_centre, const int* lw,
                                 int w, int h, int replicate)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    unsigned short hist[256];
    for (int v = 0; v < 256; v++) hist[v] = 0;

    for (int i = 0; i < ksize; i++)
    {
        int nx = x + offsets[2*i], ny = y + offsets[2*i + 1];
        int key = (lw[0]*read_px(img, nx, ny, 0, w, h, 3, replicate)
                 + lw[1]*read_px(img, nx, ny, 1, w, h, 3, replicate)
                 + lw[2]*read_px(img, nx, ny, 2, w, h, 3, replicate)) >> 8;
        hist[key]++;
    }

    // key value of the rank-th darkest pixel
    int cum = 0, sel = 0;
    for (int v = 0; v < 256; v++)
    {
        cum += hist[v];
        if (cum >= rank) { sel = v; break; }
    }

    int o = (x + y*w)*3;
    int centre_key = (lw[0]*img[o] + lw[1]*img[o + 1] + lw[2]*img[o + 2]) >> 8;
    if (clamp_to_centre && centre_key <= sel)
    {
        out[o] = img[o]; out[o + 1] = img[o + 1]; out[o + 2] = img[o + 2];
        return;
    }

    // first pixel under the element with that key
    for (int i = 0; i < ksize; i++)
    {
        int nx = x + offsets[2*i], ny = y + offsets[2*i + 1];
        int c0 = read_px(img, nx, ny, 0, w, h, 3, replicate);
        int c1 = read_px(img, nx, ny, 1, w, h, 3, replicate);
        int c2 = read_px(img, nx, ny, 2, w, h, 3, replicate);
        if (((lw[0]*c0 + lw[1]*c1 + lw[2]*c2) >> 8) == sel)
        {
            out[o] = (unsigned char)c0; out[o + 1] = (unsigned char)c1; out[o + 2] = (unsigned char)c2;
            return;
        }
    }
}
