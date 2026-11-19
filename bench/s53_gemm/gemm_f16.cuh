// f16 GEMM prototype: y[t,n] f32 = sum_k x[t,k] w[n,k], f16 operands, rows of K halves (row bytes 2K).
#ifndef H_BM
#define H_BM 128
#endif
#ifndef H_BN
#define H_BN 256
#endif
#ifndef H_STAGES
#define H_STAGES 3
#endif
#ifndef H_WM
#define H_WM 2
#endif
#ifndef H_WN
#define H_WN 4
#endif
#ifndef H_MINB
#define H_MINB 1
#endif
#ifndef H_GROUP
#define H_GROUP 16
#endif
#define H_THREADS (H_WM * H_WN * 32)
#define H_TM (H_BM / H_WM)
#define H_TN (H_BN / H_WN)
#define H_FM (H_TM / 16)
#define H_FN (H_TN / 8)
#define H_STAGE_BYTES ((H_BM + H_BN) * 128)
#define H_SHARED (H_STAGES * H_STAGE_BYTES)

typedef unsigned int hu32;
typedef unsigned long long hu64;

struct HParams
{
    const unsigned short* x;
    const unsigned short* w;
    float* y;
    hu32 T;
    hu32 N;
    hu32 K;
    hu32 ldy;
};

static __device__ __forceinline__ void h_cp16(hu32 dst, const void* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}
static __device__ __forceinline__ void h_ldsm4(hu32 addr, hu32& r0, hu32& r1, hu32& r2, hu32& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
static __device__ __forceinline__ void h_mma(float* c, const hu32* a, hu32 b0, hu32 b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
static __device__ __forceinline__ hu32 h_swz(hu32 r, hu32 c)
{
    return r * 128u + ((c ^ (r & 7u)) << 4);
}

extern "C" __global__ void __launch_bounds__(H_THREADS, H_MINB) gemm_f16_native(HParams p)
{
    extern __shared__ __align__(128) unsigned char smem[];
    const hu32 tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const hu32 wm = warp / H_WN, wn = warp % H_WN;
    // Groups of H_GROUP token tiles walk all weight tiles: the group's activations stay in L2 while the weights stream.
    const hu32 tilesM = gridDim.x, id = blockIdx.y * gridDim.x + blockIdx.x;
    const hu32 groupSize = H_GROUP * gridDim.y, group = id / groupSize, first = group * H_GROUP;
    const hu32 rows = min(tilesM - first, (hu32)H_GROUP);
    const hu32 tm = first + (id % groupSize) % rows, tn = (id % groupSize) / rows;
    const hu32 t0 = tm * H_BM, n0 = tn * H_BN;
    const hu32 K = p.K, nk = K / 64;
    const unsigned char* xb = (const unsigned char*)p.x;
    const unsigned char* wbp = (const unsigned char*)p.w;
    const hu64 rs = (hu64)K * 2;

    auto load = [&](hu32 s, hu32 kb) {
        hu32 base = (hu32)__cvta_generic_to_shared(smem + s * H_STAGE_BYTES);
        hu32 k0 = kb * 128;
#pragma unroll
        for (hu32 i = 0; i < H_BM * 8 / H_THREADS; i++)
        {
            hu32 idx = tid + i * H_THREADS, r = idx >> 3, c = idx & 7;
            hu32 t = min(t0 + r, p.T - 1);
            h_cp16(base + h_swz(r, c), xb + (hu64)t * rs + k0 + c * 16);
        }
#pragma unroll
        for (hu32 i = 0; i < H_BN * 8 / H_THREADS; i++)
        {
            hu32 idx = tid + i * H_THREADS, r = idx >> 3, c = idx & 7;
            hu32 n = min(n0 + r, p.N - 1);
            h_cp16(base + H_BM * 128 + h_swz(r, c), wbp + (hu64)n * rs + k0 + c * 16);
        }
    };

    float acc[H_FM][H_FN][4];
#pragma unroll
    for (int i = 0; i < H_FM; i++)
#pragma unroll
        for (int j = 0; j < H_FN; j++)
#pragma unroll
            for (int e = 0; e < 4; e++)
                acc[i][j][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < H_STAGES - 1; s++)
    {
        if (s < (int)nk) load(s, s);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (hu32 kb = 0; kb < nk; kb++)
    {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(H_STAGES - 2));
        __syncthreads();
        hu32 next = kb + H_STAGES - 1;
        if (next < nk) load(next % H_STAGES, next);
        asm volatile("cp.async.commit_group;\n" ::);
        hu32 sa = (hu32)__cvta_generic_to_shared(smem + (kb % H_STAGES) * H_STAGE_BYTES), sb = sa + H_BM * 128;
#pragma unroll
        for (hu32 kk = 0; kk < 4; kk++)
        {
            hu32 a[H_FM][4], b[H_FN][2];
#pragma unroll
            for (int i = 0; i < H_FM; i++)
            {
                hu32 r = wm * H_TM + i * 16 + (lane & 15);
                h_ldsm4(sa + h_swz(r, kk * 2 + (lane >> 4)), a[i][0], a[i][1], a[i][2], a[i][3]);
            }
#pragma unroll
            for (int j = 0; j < H_FN; j += 2)
            {
                hu32 r = wn * H_TN + j * 8 + (lane & 7) + ((lane >> 4) << 3);
                h_ldsm4(sb + h_swz(r, kk * 2 + ((lane >> 3) & 1)), b[j][0], b[j][1], b[j + 1][0], b[j + 1][1]);
            }
#pragma unroll
            for (int i = 0; i < H_FM; i++)
#pragma unroll
                for (int j = 0; j < H_FN; j++)
                    h_mma(acc[i][j], a[i], b[j][0], b[j][1]);
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    const hu32 g = lane >> 2, c4 = lane & 3;
#pragma unroll
    for (int i = 0; i < H_FM; i++)
#pragma unroll
        for (int h = 0; h < 2; h++)
        {
            hu32 t = t0 + wm * H_TM + i * 16 + g + h * 8;
            if (t >= p.T) continue;
            float* yr = p.y + (hu64)t * p.ldy;
#pragma unroll
            for (int j = 0; j < H_FN; j++)
            {
                hu32 n = n0 + wn * H_TN + j * 8 + c4 * 2;
                if (n + 1 < p.N) *(float2*)(yr + n) = make_float2(acc[i][j][h * 2], acc[i][j][h * 2 + 1]);
                else if (n < p.N) yr[n] = acc[i][j][h * 2];
            }
        }
}
