#include <pycuda-helpers.hpp>

// Intensity-based local symmetry measure after
//   Zielke, Brauckmann, von Seelen: "Intensity and Edge-Based Symmetry Detection
//   Applied to Car-Following", ECCV 1992, section 2 (eq. 2-6).
//
// For a 1D intensity function G along a scan line, a candidate axis position xs and an
// interval width w:
//   E(u)  = (G(xs+u) + G(xs-u)) / 2          even part            (eq. 2)
//   O(u)  = (G(xs+u) - G(xs-u)) / 2          odd part             (eq. 3)
//   En(u) = E(u) - mean(E)                   zero-mean even part  (eq. 4)
//   S     = (sum En^2 - sum O^2) / (sum En^2 + sum O^2)   in [-1, 1]   (eq. 5)
//   SA    = w / (2 * w_max) * (S + 1)                     in [ 0, 1]   (eq. 6)
// S = 1: ideal symmetry, 0: asymmetry, -1: antisymmetry. SA weights S by the interval
// size, so wide symmetric intervals give a confident axis.

#define SYM_EPS 1e-6f   // below this total energy the interval is flat: S is undefined, use 0

__device__ float2 rotate_f2(float2 dir, float theta)
{
    float tx =  dir.x*cosf(theta) + dir.y*sinf(theta);
    float ty = -dir.x*sinf(theta) + dir.y*cosf(theta);
    return make_float2(tx, ty);
}

texture<float, 2> image;   // intensity in [0, 1], linear filtering, x = column, y = row

// Adds the mirrored sample pair at distance k (values already offset by the centre value) to
// the running sums and returns S for the interval [-k, k] (eq. 5).
__device__ float sym_step(float gn, float gp, int k, float* sumE, float* sumE2, float* sumO2)
{
    float e = 0.5f*(gp + gn);
    float o = 0.5f*(gp - gn);
    *sumE  += 2.0f*e;
    *sumE2 += 2.0f*e*e;
    *sumO2 += 2.0f*o*o;

    float w   = 2.0f*k + 1.0f;                               // samples in the interval
    float En2 = fmaxf(*sumE2 - (*sumE)*(*sumE)/w, 0.0f);     // sum of (E - mean(E))^2
    float den = En2 + *sumO2;
    return (den > SYM_EPS) ? (En2 - *sumO2)/den : 0.0f;
}


// Computes the symmetry feature for EVERY pixel, treating the pixel as the axis position xs.
//
//   scan_angles[idx]    : direction of the scan line through the pixel, as a rotation of (0,1).
//                         0 -> scan along y (horizontal symmetry axis),
//                         pi/2 -> scan along x (vertical symmetry axis, the paper's car case).
//   max_half_width[idx] : largest half-width K to test, in pixels; w_max = 2K+1.
//
// Output, 3 planes of cols*rows floats (idx = x + y*cols):
//   plane 0: max over w of SA(xs, w)      confidence of a symmetry axis here, [0, 1]
//   plane 1: w (pixels) where that max occurs
//   plane 2: S(xs, w) at that w, [-1, 1]
__global__ void calcSymmetryGpuH(float* res, const float* scan_angles, const float* max_half_width,
                                 int cols, int rows)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= cols || y >= rows) return;

    int idx   = x + y*cols;
    int plane = cols*rows;

    // +0.5: sample texel centres (linear filtering)
    float cx = x + 0.5f, cy = y + 0.5f;
    // The measure is invariant to a constant intensity offset; subtracting the centre value
    // keeps the running sums small (less float cancellation in sumE2 - sumE^2/n).
    float centre = tex2D(image, cx, cy);

    float2 dir = rotate_f2(make_float2(0.0f, 1.0f), scan_angles[idx]);
    int   K     = (int)max_half_width[idx];
    float w_max = 2.0f*K + 1.0f;

    // Running sums over the interval [-k, k]. The centre sample contributes E=0 (after the
    // offset) and O=0; each step adds the two mirrored samples at -k and +k.
    float sumE = 0.0f, sumE2 = 0.0f, sumO2 = 0.0f;
    float bestSA = 0.0f, bestW = 0.0f, bestS = 0.0f;

    for (int k = 1; k <= K; k++)
    {
        float xn = cx - k*dir.x, yn = cy - k*dir.y;
        float xp = cx + k*dir.x, yp = cy + k*dir.y;
        // the interval must stay inside the image (w <= d in the paper)
        if (xn < 0.0f || xn >= cols || yn < 0.0f || yn >= rows ||
            xp < 0.0f || xp >= cols || yp < 0.0f || yp >= rows)
            break;

        float gn = tex2D(image, xn, yn) - centre;
        float gp = tex2D(image, xp, yp) - centre;
        float S  = sym_step(gn, gp, k, &sumE, &sumE2, &sumO2);
        float w  = 2.0f*k + 1.0f;
        float SA = w/(2.0f*w_max) * (S + 1.0f);

        if (k == 1 || SA > bestSA)
        {
            bestSA = SA;
            bestW  = w;
            bestS  = S;
        }
    }

    res[idx]           = bestSA;
    res[idx +   plane] = bestW;
    res[idx + 2*plane] = bestS;
}


// Samples the image at (a, s): a = coordinate ACROSS the symmetry axis, s = coordinate ALONG it.
// Vertical axis: a = x, s = y. Horizontal axis: a = y, s = x.
__device__ float tex_axis(float a, float s, int horizontal)
{
    return horizontal ? tex2D(image, s, a) : tex2D(image, a, s);
}

// Symmetry histogram (paper section 2.2, Fig. 2c): SA(a, w) summed over parallel scan lines,
// for every axis position a and every half-width k. One thread per axis position.
//
//   horizontal = 0: VERTICAL axes.   a = column, scan lines = image rows    s0 <= y < s1
//   horizontal = 1: HORIZONTAL axes. a = row,    scan lines = image columns s0 <= x < s1
//
//   hist[(k-1)*n_axis + a] = sum over scan lines of SA(a, w = 2k+1),  k = 1..K,  w_max = 2K+1
// with n_axis = cols (vertical) or rows (horizontal).
// Intervals that would leave the image contribute nothing.
__global__ void calcSymmetryHistogram(float* hist, int K, int s0, int s1, int horizontal,
                                      int cols, int rows)
{
    int n_axis = horizontal ? rows : cols;   // number of axis positions
    int n_scan = horizontal ? cols : rows;   // number of scan lines

    int a = threadIdx.x + blockDim.x * blockIdx.x;
    if (a >= n_axis) return;

    for (int k = 1; k <= K; k++)
        hist[(k-1)*n_axis + a] = 0.0f;

    float ca    = a + 0.5f;
    float w_max = 2.0f*K + 1.0f;
    // largest half-width that keeps [a-k, a+k] inside the image
    int k_end = min(K, min(a, n_axis - 1 - a));

    for (int sl = max(s0, 0); sl < min(s1, n_scan); sl++)
    {
        float cs     = sl + 0.5f;
        float centre = tex_axis(ca, cs, horizontal);
        float sumE = 0.0f, sumE2 = 0.0f, sumO2 = 0.0f;

        for (int k = 1; k <= k_end; k++)
        {
            float gn = tex_axis(ca - k, cs, horizontal) - centre;
            float gp = tex_axis(ca + k, cs, horizontal) - centre;
            float S  = sym_step(gn, gp, k, &sumE, &sumE2, &sumO2);
            float w  = 2.0f*k + 1.0f;
            hist[(k-1)*n_axis + a] += w/(2.0f*w_max) * (S + 1.0f);
        }
    }
}


// ---------------------------------------------------------------------------------------
// SEED: Symmetry-Enhancing Edge Detector (paper section 3.2, eq. 7-8).
// ---------------------------------------------------------------------------------------

#define SEED_N 8   // number of discrete directions, 45 degrees apart

// Directional edge responses at (px, py): Sobel gradient projected on 8 directions and
// half-wave rectified, so d[i] responds to gradients pointing along angle i*45 degrees
// (x right, y down) with cosine-shaped, overlapping angular sensitivity.
// Scaled to 0..255 intensity units per pixel so the sigmoid threshold T is in grey levels.
// swap_xy exchanges the roles of x and y (transposed image). A horizontal axis becomes a
// vertical one in that frame, so the same weight table W serves both axis orientations.
__device__ void seed_directional(float px, float py, int swap_xy, float* d)
{
    float tl = tex2D(image, px-1, py-1), t = tex2D(image, px, py-1), tr = tex2D(image, px+1, py-1);
    float l  = tex2D(image, px-1, py  ),                             r  = tex2D(image, px+1, py  );
    float bl = tex2D(image, px-1, py+1), b = tex2D(image, px, py+1), br = tex2D(image, px+1, py+1);

    float gx = ((tr + 2.0f*r + br) - (tl + 2.0f*l + bl)) * (255.0f/4.0f);
    float gy = ((bl + 2.0f*b + br) - (tl + 2.0f*t + tr)) * (255.0f/4.0f);

    if (swap_xy) { float tmp = gx; gx = gy; gy = tmp; }

    for (int i = 0; i < SEED_N; i++)
    {
        float a = i * (6.2831853f / SEED_N);
        d[i] = fmaxf(0.0f, gx*cosf(a) + gy*sinf(a));
    }
}

// For every pixel, the SEED output for a symmetry axis at position axis_pos:
//   horizontal = 0: VERTICAL axis at column axis_pos, pixels paired left <-> right
//   horizontal = 1: HORIZONTAL axis at row axis_pos,  pixels paired above <-> below
// The pixel is paired with its mirror image across the axis. An edge direction at the pixel
// is passed only to the degree that the directions present at the mirror point are
// compatible with it:
//   out = max_i [ d_own[i] * Phi( sum_j d_mirror[j] * W[i][j], T ) ]          (eq. 7)
//   Phi(u, T) = 1 / (1 + exp(T - u/k))                                          (eq. 8)
//
//   W        : SEED_N*SEED_N compatibility weights; W[i*SEED_N + j] relates direction i on
//              the LEFT (or ABOVE) side of the axis to direction j on the RIGHT (or BELOW)
//              side (built in sym_measure.py).
//   max_dist : pixels farther than this from the axis give 0 (<= 0: no limit).
//   T, k_norm: sigmoid threshold and normalisation factor k.
__global__ void calcSeed(float* out, const float* W, float axis_pos, float max_dist,
                         float T, float k_norm, int horizontal, int cols, int rows)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= cols || y >= rows) return;

    int   idx   = x + y*cols;
    float px    = x + 0.5f, py = y + 0.5f;
    float dist  = (horizontal ? py : px) - axis_pos;   // signed distance to the axis
    float m     = axis_pos - dist;                     // mirrored coordinate across the axis
    float limit = horizontal ? rows : cols;

    out[idx] = 0.0f;
    if (fabsf(dist) < 0.5f || m < 0.0f || m >= limit) return;
    if (max_dist > 0.0f && fabsf(dist) > max_dist) return;

    float d_own[SEED_N], d_mir[SEED_N];
    seed_directional(px, py, horizontal, d_own);
    if (horizontal) seed_directional(px, m, horizontal, d_mir);
    else            seed_directional(m, py, horizontal, d_mir);

    bool  own_is_first = dist < 0.0f;   // left of / above the axis
    float best = 0.0f;
    for (int i = 0; i < SEED_N; i++)
    {
        float u = 0.0f;
        for (int j = 0; j < SEED_N; j++)
            u += d_mir[j] * (own_is_first ? W[i*SEED_N + j] : W[j*SEED_N + i]);

        float phi = 1.0f / (1.0f + expf(T - u/k_norm));
        best = fmaxf(best, d_own[i] * phi);
    }
    out[idx] = best;
}
