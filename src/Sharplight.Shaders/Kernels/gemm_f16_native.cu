// f16 GEMM written in CUDA C++ for the CUDA provider (DeviceFeatures.NativeGemmF16, compute capability 8.0 and later):
// y[t, n] = sum_k x[t, k] * W[n, k] with f16 activations and weights (the operands of the library GEMM: the generated GEMMs' f16
// copy of x and bf16 weights stored as f16), f32 accumulation and output. Compiled by NVRTC from this file: cp.async copies into
// a ring of 64-value K steps (128-byte rows, XOR swizzle) in dynamic shared memory, ldmatrix and mma.sync.m16n8k16.
//
// Tiles (H_CFG): 0 = 128 tokens x 256 weight rows, 2 x 4 warps of 64 x 64 outputs, two stages (96 KB); 1 = 128 x 128, 2 x 2 warps
// of 64 x 64, three stages (96 KB), for grids that the wide tile would fill in few waves; 2 = the decode form for batches of at
// most 64 rows: 64 tokens x H_DBN (default 64) weight rows, 2 x (H_DBN / 16) warps of 32 x 16 outputs, four stages, no K split (one
// launch where the generated GEMMs split K and reduce). Blocks are rasterized in groups of H_GROUP token tiles that walk all weight
// tiles, so a group's activations stay in L2 while the weights stream through.
//
// DUAL: two weight matrices of the same shape (w, w2; gate and up) share x; the B tile's 8-row groups alternate between them, each
// thread holds both results of an output and stores H_EPI2(a, b) (the generated epilogue); a block covers half its weight rows of
// outputs.
//
// MULTI (2 or 3, not with DUAL): the multi-output form over the concatenated rows of two or three matrices of one K (q / k / v),
// each with its own output (dense rows of its width) and, with H_BIAS, its own bias: rows [0, MULTI_O1) are w's into y, [MULTI_O1,
// MULTI_O2) w_1's into y_1, the rest w_2's into y_2. A block's weight tile lies in one matrix (MULTI_O1, MULTI_O2 multiples of the
// tile's weight rows), and each output sums its K steps in the order of the matrix's own launch.
//
// H_BIAS (single matrix): the store adds bias[n] (a projection's bias row). H_XRS: x is a row-ranged f16 copy whose scale words
// start at byte H_XRS of x (common.slang, f16_range_scale); the store multiplies each row's products by the inverse of its scale
// (before the bias; exact: a power of two), as the unscale launch (and the Add) would.
//
// H_SKIP_PAD (default 1): a warp skips the A fragments and MMAs of its 16-row fragments past the batch's last row (a partial last
// token tile costs about its rows); the other fragments compute the same products.
//
// Specialization macros: K, N (all rows with MULTI), Y_STRIDE, H_CFG, H_DBN (H_CFG 2), DUAL, H_EPI2 (DUAL), MULTI, MULTI_O1,
// MULTI_O2, H_BIAS, H_XRS. Requires K % 64 == 0, 16-byte aligned operands and an even Y_STRIDE (MULTI: even widths). Grid:
// x = token tiles, y = weight tiles.
#ifndef DUAL
#define DUAL 0
#endif
#ifndef H_CFG
#define H_CFG 0
#endif
#ifndef H_BIAS
#define H_BIAS 0
#endif
#ifndef H_SKIP_PAD
#define H_SKIP_PAD 1
#endif
#ifndef MULTI
#define MULTI 0
#endif
#if H_BIAS && DUAL
#error "the bias epilogue is for single matrices"
#endif
#if MULTI && DUAL
#error "gemm_f16_native: MULTI with DUAL"
#endif
#if H_CFG == 0
#define H_BM 128
#define H_BN 256
#define H_WN 4
#define H_STAGES 2
#elif H_CFG == 1
#define H_BM 128
#define H_BN 128
#define H_WN 2
#define H_STAGES 3
#else
#ifndef H_DBN
#define H_DBN 64
#endif
#define H_BM 64
#define H_BN H_DBN
#define H_WN (H_DBN / 16)
#define H_STAGES 4
#endif
#define H_WM 2
#define H_GROUP 16
#define H_THREADS (H_WM * H_WN * 32)
#define H_TM (H_BM / H_WM)
#define H_TN (H_BN / H_WN)
#define H_FM (H_TM / 16)
#define H_FN (H_TN / 8)
#define H_STAGE_BYTES ((H_BM + H_BN) * 128)
#define H_RS ((unsigned long long)K * 2)
#if DUAL
#define H_BOUT (H_BN / 2)
#else
#define H_BOUT H_BN
#endif

typedef unsigned int h_u32;
typedef unsigned long long h_u64;

struct HParams
{
    const unsigned char* x;
    const unsigned char* w;
#if DUAL
    const unsigned char* w2;
#endif
    float* y;
#if H_BIAS
    const float* bias;
#endif
#if MULTI
    const unsigned char* w_1;
#if MULTI > 2
    const unsigned char* w_2;
#endif
    float* y_1;
#if MULTI > 2
    float* y_2;
#endif
#if H_BIAS
    const float* bias_1;
#if MULTI > 2
    const float* bias_2;
#endif
#endif
#endif
    h_u32 T;
};

// The matrix a block's weight tile lies in (MULTI: its member's), with that member's output, bias, row count and first row.
struct HMember
{
    const unsigned char* w;
    float* y;
#if H_BIAS
    const float* bias;
#endif
    h_u32 rows;
};

static __device__ __forceinline__ void h_cp16(h_u32 dst, const void* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}

static __device__ __forceinline__ void h_ldsm4(h_u32 addr, h_u32& r0, h_u32& r1, h_u32& r2, h_u32& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

static __device__ __forceinline__ void h_mma(float* c, const h_u32* a, h_u32 b0, h_u32 b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Byte offset of 16-byte chunk c of row r in a stage tile of 128-byte rows (chunk c ^ (r % 8)).
static __device__ __forceinline__ h_u32 h_swz(h_u32 r, h_u32 c)
{
    return r * 128u + ((c ^ (r & 7u)) << 4);
}

// Weight row of B tile row r (rows past the matrix's are clamped: their outputs are never stored).
static __device__ __forceinline__ const unsigned char* h_wrow(const HParams& p, const HMember& m, h_u32 n0, h_u32 r)
{
#if DUAL
    const unsigned char* w = ((r >> 3) & 1) ? p.w2 : p.w;
    h_u32 n = min(n0 + (r >> 4) * 8 + (r & 7), m.rows - 1);
#else
    const unsigned char* w = m.w;
    h_u32 n = min(n0 + r, m.rows - 1);
#endif
    return w + (h_u64)n * H_RS;
}

// The member of weight row n0 (a tile's first), and n0 made relative to it.
static __device__ __forceinline__ HMember h_member(const HParams& p, h_u32& n0)
{
#if H_BIAS
    HMember m = { p.w, p.y, p.bias, (h_u32)N };
#else
    HMember m = { p.w, p.y, (h_u32)N };
#endif
#if MULTI
    m.rows = MULTI_O1;
    if (n0 >= MULTI_O1)
    {
#if MULTI > 2
        const bool third = n0 >= MULTI_O2;
        m.w = third ? p.w_2 : p.w_1;
        m.y = third ? p.y_2 : p.y_1;
#if H_BIAS
        m.bias = third ? p.bias_2 : p.bias_1;
#endif
        m.rows = third ? N - MULTI_O2 : MULTI_O2 - MULTI_O1;
        n0 -= third ? MULTI_O2 : MULTI_O1;
#else
        m.w = p.w_1;
        m.y = p.y_1;
#if H_BIAS
        m.bias = p.bias_1;
#endif
        m.rows = N - MULTI_O1;
        n0 -= MULTI_O1;
#endif
    }
#endif
    return m;
}

extern "C" __global__ void __launch_bounds__(H_THREADS, H_BM == 64 ? 2 : 1) gemm_f16_native(HParams p)
{
    extern __shared__ __align__(128) unsigned char smem[];
    const h_u32 tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const h_u32 wm = warp / H_WN, wn = warp % H_WN;
    const h_u32 id = blockIdx.y * gridDim.x + blockIdx.x, groupBlocks = H_GROUP * gridDim.y;
    const h_u32 first = id / groupBlocks * H_GROUP, groupRows = min(gridDim.x - first, (h_u32)H_GROUP);
    const h_u32 t0 = (first + id % groupBlocks % groupRows) * H_BM;
    h_u32 n0 = id % groupBlocks / groupRows * H_BOUT;
    const HMember m = h_member(p, n0);
    const h_u32 nk = K / 64;
#if H_SKIP_PAD
    // 16-row fragments of this warp that hold rows of x (warp-uniform): a partial last token tile skips the MMAs of rows past T.
    const int nfr = min(max(((int)p.T - (int)(t0 + wm * H_TM) + 15) / 16, 0), H_FM);
#else
    const int nfr = H_FM;
#endif

    auto load = [&](h_u32 s, h_u32 kb) {
        h_u32 base = (h_u32)__cvta_generic_to_shared(smem + s * H_STAGE_BYTES);
        const h_u32 k0 = kb * 128;
#pragma unroll
        for (h_u32 i = 0; i < H_BM * 8 / H_THREADS; i++)
        {
            h_u32 idx = tid + i * H_THREADS, r = idx >> 3, c = idx & 7;
            h_cp16(base + h_swz(r, c), p.x + (h_u64)min(t0 + r, p.T - 1) * H_RS + k0 + c * 16);
        }
#pragma unroll
        for (h_u32 i = 0; i < H_BN * 8 / H_THREADS; i++)
        {
            h_u32 idx = tid + i * H_THREADS, r = idx >> 3, c = idx & 7;
            h_cp16(base + H_BM * 128 + h_swz(r, c), h_wrow(p, m, n0, r) + k0 + c * 16);
        }
    };

    float acc[H_FM][H_FN][4];
#pragma unroll
    for (int i = 0; i < H_FM; i++)
    {
#pragma unroll
        for (int j = 0; j < H_FN; j++)
        {
#pragma unroll
            for (int e = 0; e < 4; e++)
            {
                acc[i][j][e] = 0.0f;
            }
        }
    }

#pragma unroll
    for (int s = 0; s < H_STAGES - 1; s++)
    {
        if (s < (int)nk)
        {
            load(s, s);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }

    for (h_u32 kb = 0; kb < nk; kb++)
    {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(H_STAGES - 2));
        __syncthreads();
        h_u32 next = kb + H_STAGES - 1;
        if (next < nk)
        {
            load(next % H_STAGES, next);
        }
        asm volatile("cp.async.commit_group;\n" ::);

        const h_u32 sa = (h_u32)__cvta_generic_to_shared(smem + (kb % H_STAGES) * H_STAGE_BYTES), sb = sa + H_BM * 128;
#pragma unroll
        for (h_u32 kk = 0; kk < 4; kk++)
        {
            h_u32 a[H_FM][4], b[H_FN][2];
#pragma unroll
            for (int i = 0; i < H_FM; i++)
            {
                if (i < nfr)
                {
                    h_u32 r = wm * H_TM + i * 16 + (lane & 15);
                    h_ldsm4(sa + h_swz(r, kk * 2 + (lane >> 4)), a[i][0], a[i][1], a[i][2], a[i][3]);
                }
            }
#pragma unroll
            for (int j = 0; j < H_FN; j += 2)
            {
                h_u32 r = wn * H_TN + j * 8 + (lane & 7) + ((lane >> 4) << 3);
                h_ldsm4(sb + h_swz(r, kk * 2 + ((lane >> 3) & 1)), b[j][0], b[j][1], b[j + 1][0], b[j + 1][1]);
            }
#pragma unroll
            for (int i = 0; i < H_FM; i++)
            {
                if (i < nfr)
                {
#pragma unroll
                    for (int j = 0; j < H_FN; j++)
                    {
                        h_mma(acc[i][j], a[i], b[j][0], b[j][1]);
                    }
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

    const h_u32 g = lane >> 2, c4 = lane & 3;
#pragma unroll
    for (int i = 0; i < H_FM; i++)
    {
#pragma unroll
        for (int h = 0; h < 2; h++)
        {
            h_u32 t = t0 + wm * H_TM + i * 16 + g + h * 8;
            if (t >= p.T)
            {
                continue;
            }

#if MULTI
            float* yr = m.y + (h_u64)t * m.rows;
#else
            float* yr = m.y + (h_u64)t * Y_STRIDE;
#endif
#ifdef H_XRS
            const h_u32 scaleWord = *(const h_u32*)(p.x + H_XRS + (h_u64)t * 4);
            const float rowInverse = scaleWord == 0 ? 1.0f : __uint_as_float(scaleWord);
#endif
#if DUAL
#pragma unroll
            for (int j = 0; j < H_FN; j += 2)
            {
                h_u32 n = n0 + (wn * H_TN) / 2 + (j / 2) * 8 + c4 * 2;
                float v0 = H_EPI2(acc[i][j][h * 2], acc[i][j + 1][h * 2]);
                float v1 = H_EPI2(acc[i][j][h * 2 + 1], acc[i][j + 1][h * 2 + 1]);
#else
#pragma unroll
            for (int j = 0; j < H_FN; j++)
            {
                h_u32 n = n0 + wn * H_TN + j * 8 + c4 * 2;
                float v0 = acc[i][j][h * 2], v1 = acc[i][j][h * 2 + 1];
#ifdef H_XRS
                v0 *= rowInverse;
                v1 *= rowInverse;
#endif
#if H_BIAS
                v0 += m.bias[min(n, m.rows - 1)];
                v1 += m.bias[min(n + 1, m.rows - 1)];
#endif
#endif
                if (n + 1 < m.rows)
                {
                    *(float2*)(yr + n) = make_float2(v0, v1);
                }
                else if (n < m.rows)
                {
                    yr[n] = v0;
                }
            }
        }
    }
}
