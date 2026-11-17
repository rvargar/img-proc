// Batched feature kernels for segmentation networks (segnet/features_cuda.py). They work on
// plain buffers, so they can run directly on PyTorch tensors.
//
//   quantileBank     : order statistics under a whole bank of flat structuring elements;
//                      the same result as quantileFilter in morph_features.cu
//   symmetryFeatures : intensity symmetry for several directions and half-widths;
//                      the same measure as calcSymmetryGpuH in sym_measure.cu


// ---------------------------------------------------------------------------------------
// Regulated morphology: several order statistics for every element of a bank, one launch.
// ---------------------------------------------------------------------------------------
//
// Per pixel and element, a 256-bin histogram of the intensities under the element is kept in
// 32 64-bit words, one byte per bin (elements have at most 255 pixels, so a bin cannot
// overflow). The sum of the 8 bins of a word is one multiplication, so the cumulative scan
// that finds a rank visits at most 32 words and then the 8 bins of one word.
//
//   img        : n_planes * h * w unsigned chars (every plane is filtered on its own)
//   offsets    : (dx, dy) pairs of all elements, concatenated
//   elem_start : n_elem + 1 indices into offsets (element e owns [elem_start[e], elem_start[e+1]))
//   ranks      : n_elem * n_ranks ranks, 1-based, ASCENDING per element
//   out        : n_planes * n_elem * n_ranks * h * w floats in [0, 1]
// Element pixels outside the image reuse the nearest image pixel. Launch with grid.z = n_planes.
__global__ void quantileBank(float* out, const unsigned char* img, const int* offsets,
                             const int* elem_start, int n_elem, const int* ranks, int n_ranks,
                             int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int plane = w*h;
    const unsigned char* im = img + (size_t)blockIdx.z * plane;
    float* o = out + (size_t)blockIdx.z * n_elem * n_ranks * plane + x + y*w;
    unsigned long long hist[32];

    for (int e = 0; e < n_elem; e++)
    {
        for (int i = 0; i < 32; i++) hist[i] = 0ULL;
        for (int i = elem_start[e]; i < elem_start[e + 1]; i++)
        {
            int nx = min(max(x + offsets[2*i], 0), w - 1);
            int ny = min(max(y + offsets[2*i + 1], 0), h - 1);
            int v  = im[nx + ny*w];
            hist[v >> 3] += 1ULL << ((v & 7) * 8);
        }

        const int* rk = ranks + e*n_ranks;
        int cum = 0, r = 0;
        for (int i = 0; i < 32 && r < n_ranks; i++)
        {
            int s = (int)((hist[i] * 0x0101010101010101ULL) >> 56);   // sum of the 8 bins
            if (cum + s >= rk[r])
            {
                unsigned long long word = hist[i];
                int c = cum;
                for (int b = 0; b < 8 && r < n_ranks; b++)
                {
                    c += (int)(word & 0xffULL);
                    word >>= 8;
                    while (r < n_ranks && c >= rk[r])
                    {
                        o[(size_t)(e*n_ranks + r) * plane] = (i*8 + b) / 255.0f;
                        r++;
                    }
                }
            }
            cum += s;
        }
    }
}


// ---------------------------------------------------------------------------------------
// Intensity symmetry (Zielke et al., ECCV'92, eq. 2-6) for a batch, several scan directions
// and several maximal half-widths in ONE pass. Bilinear sampling is done by hand instead of
// with a texture.
//
// For every pixel and direction the interval [-k, k] grows up to the largest half-width; the
// running maximum of w*(S+1) is snapshotted at each requested half-width K, which gives
// max_{k<=K} SA(k) for w_max = 2K+1 without recomputation.
// ---------------------------------------------------------------------------------------

#define SYM_EPS 1e-6f

// Bilinear sample at texel-centre coordinates (px, py), clamped to the image.
__device__ float sample_bilinear(const float* img, float px, float py, int w, int h)
{
    float xb = px - 0.5f, yb = py - 0.5f;
    float xf = floorf(xb), yf = floorf(yb);
    float ax = xb - xf,    ay = yb - yf;
    int x0 = min(max((int)xf, 0), w - 1), x1 = min(max((int)xf + 1, 0), w - 1);
    int y0 = min(max((int)yf, 0), h - 1), y1 = min(max((int)yf + 1, 0), h - 1);
    return (1.0f - ay) * ((1.0f - ax) * img[x0 + y0*w] + ax * img[x1 + y0*w])
         +         ay  * ((1.0f - ax) * img[x0 + y1*w] + ax * img[x1 + y1*w]);
}

//   img         : batch * h * w floats in [0, 1]
//   dirs        : n_angles (dx, dy) unit scan directions
//   half_widths : n_hw maximal half-widths K, ASCENDING
//   out         : batch * (n_angles * n_hw * 2) * h * w floats; per image the planes are
//                 ordered angle -> half-width -> (SA in [0, 1], S in [-1, 1])
// Launch with grid.z = batch.
__global__ void symmetryFeatures(float* out, const float* img, const float* dirs, int n_angles,
                                 const int* half_widths, int n_hw, int w, int h)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= w || y >= h) return;

    int plane = w*h;
    int idx   = x + y*w;
    const float* im = img + (size_t)blockIdx.z * plane;
    float* o = out + (size_t)blockIdx.z * n_angles * n_hw * 2 * plane + idx;

    float cx = x + 0.5f, cy = y + 0.5f;
    float centre = im[idx];
    int   K_max  = half_widths[n_hw - 1];

    for (int a = 0; a < n_angles; a++)
    {
        float dx = dirs[2*a], dy = dirs[2*a + 1];
        float sumE = 0.0f, sumE2 = 0.0f, sumO2 = 0.0f;
        float best = 0.0f, bestS = 0.0f;     // best = running max of w*(S+1)
        int   j = 0;                         // next half-width to snapshot

        for (int k = 1; k <= K_max; k++)
        {
            float xn = cx - k*dx, yn = cy - k*dy;
            float xp = cx + k*dx, yp = cy + k*dy;
            // the interval must stay inside the image
            if (xn < 0.0f || xn >= w || yn < 0.0f || yn >= h ||
                xp < 0.0f || xp >= w || yp < 0.0f || yp >= h)
                break;

            float gn = sample_bilinear(im, xn, yn, w, h) - centre;
            float gp = sample_bilinear(im, xp, yp, w, h) - centre;
            float e = 0.5f*(gp + gn), od = 0.5f*(gp - gn);
            sumE  += 2.0f*e;
            sumE2 += 2.0f*e*e;
            sumO2 += 2.0f*od*od;

            float wk  = 2.0f*k + 1.0f;
            float En2 = fmaxf(sumE2 - sumE*sumE/wk, 0.0f);
            float den = En2 + sumO2;
            float S   = (den > SYM_EPS) ? (En2 - sumO2)/den : 0.0f;
            float v   = wk * (S + 1.0f);
            if (k == 1 || v > best) { best = v; bestS = S; }

            if (k == half_widths[j])
            {
                o[((a*n_hw + j)*2    )*plane] = best / (2.0f*wk);
                o[((a*n_hw + j)*2 + 1)*plane] = bestS;
                j++;
            }
        }
        // half-widths the interval could not reach (image border): w_max is still 2K+1
        for (; j < n_hw; j++)
        {
            o[((a*n_hw + j)*2    )*plane] = best / (2.0f*(2.0f*half_widths[j] + 1.0f));
            o[((a*n_hw + j)*2 + 1)*plane] = bestS;
        }
    }
}
