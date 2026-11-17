// Difference filters on an ordered pattern of sampling points (diff_filter.py).
// Refactored from generalFilter and convFilter in the MATLAB project's t.cu.
//
// The pattern is an ORDERED list of n_points integer (dx, dy) offsets around the pixel,
// typically points on concentric circles or ellipses (diff_filter.sampling_pattern). The
// order matters: point i belongs to the modulo class
//       class(i) = (i + 1 - mod_offset) mod modulus          (i is 0-based)
// and only the classes 0 .. mod_count-1 take part in the filter.
//
// Images are float, one value per pixel: img[x + y*w], x = column, y = row. Sampling points
// outside the image use the nearest image pixel. Launch with grid.z = number of images; image
// b is read at img + b*w*h and written at out + b*w*h.

#define DF_INF 3.0e38f

// gamma: what a sampling point contributes, from its value v and the centre value c
#define GAMMA_DIFF      0   // v - c
#define GAMMA_NEG_DIFF  1   // c - v
#define GAMMA_ABS_DIFF  2   // |v - c|
#define GAMMA_MIN2      3   // v + c - |v - c|  =  2 min(v, c)
#define GAMMA_POS_DIFF  4   // max(v - c, 0)

// beta: how the two points of a pair are combined
#define BETA_DIFF       0   // second - first
#define BETA_MIN2       1   // first + second - |first - second|  =  2 min(first, second)

// stat: which statistic of the pair values is kept per class
#define STAT_POOLED     0   // no per-class statistic: all pairs of all classes in one set
#define STAT_MAX        1
#define STAT_MIN        2
#define STAT_SUM        3

// alpha: how the per-class statistics (or the pooled pair values) become the output
#define ALPHA_MIN       0
#define ALPHA_MAX       1
#define ALPHA_SUM       2
#define ALPHA_MEAN      3
#define ALPHA_RANGE     4   // max - min
#define ALPHA_NEG_RANGE 5   // min - max

__device__ float df_sample(const float* img, const int* offsets, int i, int x, int y, int w, int h)
{
    int sx = min(max(x + offsets[2*i], 0), w - 1);
    int sy = min(max(y + offsets[2*i + 1], 0), h - 1);
    return img[sx + sy*w];
}

__device__ float df_gamma(float v, float c, int gamma)
{
    switch (gamma)
    {
        case GAMMA_NEG_DIFF: return c - v;
        case GAMMA_ABS_DIFF: return fabsf(v - c);
        case GAMMA_MIN2:     return v + c - fabsf(v - c);
        case GAMMA_POS_DIFF: return fmaxf(v - c, 0.0f);
        default:             return v - c;
    }
}

// General difference filter: gamma -> beta -> alpha.
//
//   1. gamma: every sampling point i gets a value D[i] from its intensity and the centre pixel.
//   2. beta : points are paired, (i, i + step), for every i whose class is one of the first
//             mod_count classes and that is a multiple of pair_stride; the pair value is a
//             difference or a doubled minimum.
//             step = 1 pairs neighbours in the sequence; step = modulus pairs a point with
//             the next point of its own class; step = 0 uses no pairs: the "pair value" is
//             D[i] itself. pair_stride = 2 with step = 1 gives the disjoint pairs
//             (0, 1), (2, 3), ...
//   3. stat : per class, the maximum, minimum or sum of its pair values (or STAT_POOLED).
//   4. alpha: minimum, maximum, sum, mean or range over the classes (over all pair values
//             for STAT_POOLED).
// A pixel with no pair at all gives 0.
__global__ void generalFilter(float* out, const float* img, const int* offsets, int n_points,
                              int modulus, int mod_offset, int mod_count, int step, int pair_stride,
                              int gamma, int beta, int stat, int alpha, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const float* im = img + (size_t)blockIdx.z * w * h;
    float centre = im[x + y*w];

    // over the pair values of all classes
    float p_min = DF_INF, p_max = -DF_INF, p_sum = 0.0f;
    int   p_n = 0;
    // over the per-class statistics
    float a_min = DF_INF, a_max = -DF_INF, a_sum = 0.0f;
    int   a_n = 0;

    for (int m = 0; m < mod_count; m++)
    {
        float c_min = DF_INF, c_max = -DF_INF, c_sum = 0.0f;
        int   c_n = 0;

        // first point of class m, then every modulus-th point
        int first = ((m + mod_offset - 1) % modulus + modulus) % modulus;
        for (int i = first; i + step < n_points; i += modulus)
        {
            if (i % pair_stride != 0) continue;
            float t = df_gamma(df_sample(im, offsets, i, x, y, w, h), centre, gamma);
            if (step > 0)
            {
                float b = df_gamma(df_sample(im, offsets, i + step, x, y, w, h), centre, gamma);
                t = (beta == BETA_MIN2) ? t + b - fabsf(t - b) : b - t;
            }

            c_min = fminf(c_min, t);
            c_max = fmaxf(c_max, t);
            c_sum += t;
            c_n++;
        }
        if (c_n == 0) continue;

        p_min = fminf(p_min, c_min);
        p_max = fmaxf(p_max, c_max);
        p_sum += c_sum;
        p_n   += c_n;

        float s = (stat == STAT_MAX) ? c_max : (stat == STAT_MIN) ? c_min : c_sum;
        a_min = fminf(a_min, s);
        a_max = fmaxf(a_max, s);
        a_sum += s;
        a_n++;
    }

    if (stat == STAT_POOLED) { a_min = p_min; a_max = p_max; a_sum = p_sum; a_n = p_n; }

    float r = 0.0f;
    if (a_n > 0)
    {
        switch (alpha)
        {
            case ALPHA_MIN:       r = a_min;         break;
            case ALPHA_MAX:       r = a_max;         break;
            case ALPHA_SUM:       r = a_sum;         break;
            case ALPHA_MEAN:      r = a_sum / a_n;   break;
            case ALPHA_RANGE:     r = a_max - a_min; break;
            case ALPHA_NEG_RANGE: r = a_min - a_max; break;
        }
    }
    out[(size_t)blockIdx.z * w * h + x + y*w] = r;
}


// Linear filter on the pattern: sum_i weights[i] * value(point i) + centre_weight * centre.
__global__ void convFilter(float* out, const float* img, const int* offsets, const float* weights,
                           float centre_weight, int n_points, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const float* im = img + (size_t)blockIdx.z * w * h;
    float r = centre_weight * im[x + y*w];
    for (int i = 0; i < n_points; i++)
        r += weights[i] * df_sample(im, offsets, i, x, y, w, h);

    out[(size_t)blockIdx.z * w * h + x + y*w] = r;
}
