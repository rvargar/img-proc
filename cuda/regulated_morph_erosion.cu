// Regulated morphological erosion of a binary image (refactored from
// gen_morph/regulatedMorphErosion2D.cu).
//
// Ordinary erosion keeps a foreground pixel only if EVERY pixel under the structuring
// element is foreground. The regulated version keeps it if at least min_count of them are,
// so a few missing pixels (noise, small holes) no longer erase the pixel.
//
//   S         : input image, w*h floats, row-major (idx = x + y*w); > 0 means foreground
//   kernel    : structuring element as ksize (dx, dy) offset pairs: kernel[2*i], kernel[2*i+1]
//   min_count : number of foreground pixels required under the structuring element
//   out       : w*h floats, 1 where the pixel survives, 0 elsewhere
//
// Structuring-element pixels that fall outside the image count as background.
__global__ void regulatedMorphErosion2D(float* out, const float* S, const int* kernel,
                                        int ksize, int min_count, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // column
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // row
    if (x >= w || y >= h) return;

    int idx = x + y*w;
    out[idx] = 0.0f;
    if (S[idx] <= 0.0f) return;

    int count = 0;
    for (int i = 0; i < ksize; i++)
    {
        int nx = x + kernel[2*i];
        int ny = y + kernel[2*i + 1];
        if (nx >= 0 && nx < w && ny >= 0 && ny < h && S[nx + ny*w] > 0.0f)
            count++;
    }

    if (count >= min_count)
        out[idx] = 1.0f;
}


// Order-statistic (rank) filter for 8-bit greyscale images: the greyscale generalisation of
// the regulated erosion above.
//
// The output is the rank-th smallest intensity under the structuring element (rank = 1..ksize):
//   rank = 1            minimum  -> ordinary greyscale erosion
//   rank = (ksize+1)/2  median
//   rank = ksize        maximum  -> ordinary greyscale dilation
// With clamp_to_centre != 0 the result is additionally limited to the pixel's own value,
// out = min(S[centre], rank-th smallest), so the filter never brightens a pixel. On a 0/255
// image this is exactly regulatedMorphErosion2D with strictness = rank.
//
// The rank-th smallest value is found without sorting, by bisection on the intensity range:
// it is the smallest v with  #(neighbours <= v) >= rank.  8 counting passes for 8-bit data.
// Structuring-element pixels that fall outside the image count as intensity 0 (background),
// consistent with the binary kernel.
//
//   S, out : w*h unsigned chars, row-major (idx = x + y*w)
//   kernel : ksize (dx, dy) offset pairs
__global__ void rankOrderFilter2D(unsigned char* out, const unsigned char* S, const int* kernel,
                                  int ksize, int rank, int clamp_to_centre, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // column
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // row
    if (x >= w || y >= h) return;

    int lo = 0, hi = 255;
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        int count = 0;                       // neighbours with intensity <= mid
        for (int i = 0; i < ksize; i++)
        {
            int nx = x + kernel[2*i];
            int ny = y + kernel[2*i + 1];
            int val = (nx >= 0 && nx < w && ny >= 0 && ny < h) ? S[nx + ny*w] : 0;
            if (val <= mid) count++;
        }
        if (count >= rank) hi = mid;         // the rank-th smallest is <= mid
        else               lo = mid + 1;
    }

    int idx = x + y*w;
    int centre = S[idx];
    out[idx] = (unsigned char)((clamp_to_centre && centre < lo) ? centre : lo);
}
