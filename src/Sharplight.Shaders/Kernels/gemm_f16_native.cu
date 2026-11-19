// f16 GEMM written in CUDA C++ for the CUDA provider (DeviceFeatures.NativeGemmF16, compute capability 8.0 and later):
// y[t, n] = sum_k x[t, k] * W[n, k] with f16 activations and weights (the operands of the library GEMM: the generated GEMMs' f16
// copy of x and bf16 weights stored as f16), f32 accumulation and output. Compiled by NVRTC from this file: cp.async copies into
// a ring of 64-value K steps (128-byte rows, XOR swizzle) in dynamic shared memory, ldmatrix and mma.sync.m16n8k16.
//
// Tiles (H_CFG): 0 = 128 tokens x 256 weight rows, 2 x 4 warps of 64 x 64 outputs, two stages (96 KB); 1 = 128 x 128, 2 x 2 warps
// of 64 x 64, three stages (96 KB), for grids that the wide tile would fill in few waves. Blocks are rasterized in groups of
// H_GROUP token tiles that walk all weight tiles, so a group's activations stay in L2 while the weights stream through.
//
// DUAL: two weight matrices of the same shape (w, w2; gate and up) share x; the B tile's 8-row groups alternate between them, each
// thread holds both results of an output and stores H_EPI2(a, b) (the generated epilogue); a block covers half its weight rows of
// outputs.
//
// Specialization macros: K, N, Y_STRIDE, H_CFG, DUAL, H_EPI2 (DUAL). Requires K % 64 == 0, 16-byte aligned operands and an even
// Y_STRIDE. Grid: x = token tiles, y = weight tiles.
#ifndef DUAL
#define DUAL 0
#endif
#ifndef H_CFG
#define H_CFG 0
#endif
#if H_CFG == 0
#define H_BN 256
#define H_WN 4
#define H_STAGES 2
#else
#define H_BN 128
#define H_WN 2
#define H_STAGES 3
#endif
#define H_BM 128
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
    h_u32 T;
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

// Weight row of B tile row r (rows past N are clamped: their outputs are never stored).
static __device__ __forceinline__ const unsigned char* h_wrow(const HParams& p, h_u32 n0, h_u32 r)
{
#if DUAL
    const unsigned char* w = ((r >> 3) & 1) ? p.w2 : p.w;
    h_u32 n = min(n0 + (r >> 4) * 8 + (r & 7), (h_u32)N - 1);
#else
    const unsigned char* w = p.w;
    h_u32 n = min(n0 + r, (h_u32)N - 1);
#endif
    return w + (h_u64)n * H_RS;
}

extern "C" __global__ void __launch_bounds__(H_THREADS, 1) gemm_f16_native(HParams p)
{
    extern __shared__ __align__(128) unsigned char smem[];
    const h_u32 tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const h_u32 wm = warp / H_WN, wn = warp % H_WN;
    const h_u32 id = blockIdx.y * gridDim.x + blockIdx.x, groupBlocks = H_GROUP * gridDim.y;
    const h_u32 first = id / groupBlocks * H_GROUP, groupRows = min(gridDim.x - first, (h_u32)H_GROUP);
    const h_u32 t0 = (first + id % groupBlocks % groupRows) * H_BM;
    const h_u32 n0 = id % groupBlocks / groupRows * H_BOUT;
    const h_u32 nk = K / 64;

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
            h_cp16(base + H_BM * 128 + h_swz(r, c), h_wrow(p, n0, r) + k0 + c * 16);
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
                h_u32 r = wm * H_TM + i * 16 + (lane & 15);
                h_ldsm4(sa + h_swz(r, kk * 2 + (lane >> 4)), a[i][0], a[i][1], a[i][2], a[i][3]);
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
#pragma unroll
                for (int j = 0; j < H_FN; j++)
                {
                    h_mma(acc[i][j], a[i], b[j][0], b[j][1]);
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

            float* yr = p.y + (h_u64)t * Y_STRIDE;
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
#endif
                if (n + 1 < N)
                {
                    *(float2*)(yr + n) = make_float2(v0, v1);
                }
                else if (n < N)
                {
                    yr[n] = v0;
                }
            }
        }
    }
}
