// Native FP8 GEMM prototype B: per-row-fragment promotion (tmp holds one row fragment's 64 output columns).
#ifndef F8N_BM
#define F8N_BM 128
#endif
#ifndef F8N_BN
#define F8N_BN 256
#endif
#ifndef F8N_STAGES
#define F8N_STAGES 2
#endif
#ifndef F8N_WM
#define F8N_WM 2
#endif
#ifndef F8N_WN
#define F8N_WN 4
#endif
#ifndef F8N_DUAL
#define F8N_DUAL 0
#endif
#ifndef F8N_MINB
#define F8N_MINB 1
#endif
#define F8N_BK 128
#ifndef F8N_NOLOAD
#define F8N_NOLOAD 0
#endif
#ifndef F8N_NOPROMO
#define F8N_NOPROMO 0
#endif
#ifndef F8N_NOSTORE
#define F8N_NOSTORE 0
#endif
#define F8N_THREADS (F8N_WM * F8N_WN * 32)
#define F8N_TM (F8N_BM / F8N_WM)
#define F8N_TN (F8N_BN / F8N_WN)
#define F8N_FM (F8N_TM / 16)
#define F8N_FN (F8N_TN / 8)
#define F8N_CH 8
#define F8N_STAGE_BYTES ((F8N_BM + F8N_BN) * F8N_BK + (F8N_BM + F8N_BN) * 4)
#define F8N_SHARED (F8N_STAGES * F8N_STAGE_BYTES)

typedef unsigned char f8n_u8;
typedef unsigned int f8n_u32;
typedef unsigned long long f8n_u64;

struct F8nParams
{
    const f8n_u8* x;
    const f8n_u8* w;
#if F8N_DUAL
    const f8n_u8* w2;
#endif
    float* y;
    f8n_u32 T;
    f8n_u32 N;
    f8n_u32 K;
    f8n_u32 ldy;
};

static __device__ __forceinline__ f8n_u32 f8n_smem(const void* p)
{
    return (f8n_u32)__cvta_generic_to_shared(p);
}

static __device__ __forceinline__ void f8n_cp16(f8n_u32 dst, const void* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}

static __device__ __forceinline__ void f8n_cp4(f8n_u32 dst, const void* src)
{
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(dst), "l"(src));
}

static __device__ __forceinline__ void f8n_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
static __device__ __forceinline__ void f8n_wait()
{
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

static __device__ __forceinline__ void f8n_ldsm4(f8n_u32 addr, f8n_u32& r0, f8n_u32& r1, f8n_u32& r2, f8n_u32& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

static __device__ __forceinline__ void f8n_mma(float* c, const f8n_u32* a, f8n_u32 b0, f8n_u32 b1)
{
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

static __device__ __forceinline__ f8n_u32 f8n_swz(f8n_u32 r, f8n_u32 c)
{
    return r * 128u + ((c ^ (r & 7u)) << 4);
}

#ifndef F8N_EPI2
#define F8N_EPI2(a, b) ((a) / (1.0f + __expf(-(a))) * (b))
#endif

extern "C" __global__ void __launch_bounds__(F8N_THREADS, F8N_MINB) gemm_fp8_native(F8nParams p)
{
    extern __shared__ __align__(128) f8n_u8 smem[];
    const f8n_u32 tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const f8n_u32 wm = warp / F8N_WN, wn = warp % F8N_WN;
    const f8n_u32 t0 = blockIdx.x * F8N_BM;
#if F8N_DUAL
    const f8n_u32 n0 = blockIdx.y * (F8N_BN / 2);
#else
    const f8n_u32 n0 = blockIdx.y * F8N_BN;
#endif
    const f8n_u32 K = p.K;
    const f8n_u64 rs = (f8n_u64)K + K / 32;
    const f8n_u32 nk = K / F8N_BK;

    auto load = [&](f8n_u32 stage, f8n_u32 kb) {
        f8n_u32 sbase = f8n_smem(smem + stage * F8N_STAGE_BYTES);
        const f8n_u32 k0 = kb * F8N_BK;
#pragma unroll
        for (f8n_u32 i = 0; i < F8N_BM * F8N_CH / F8N_THREADS; i++)
        {
            f8n_u32 idx = tid + i * F8N_THREADS, r = idx >> 3, c = idx & 7;
            f8n_u32 t = min(t0 + r, p.T - 1);
            f8n_cp16(sbase + f8n_swz(r, c), p.x + (f8n_u64)t * rs + k0 + c * 16);
        }
#pragma unroll
        for (f8n_u32 i = 0; i < F8N_BN * F8N_CH / F8N_THREADS; i++)
        {
            f8n_u32 idx = tid + i * F8N_THREADS, r = idx >> 3, c = idx & 7;
#if F8N_DUAL
            const f8n_u8* wp = ((r >> 3) & 1) ? p.w2 : p.w;
            f8n_u32 n = min(n0 + (r >> 4) * 8 + (r & 7), p.N - 1);
#else
            const f8n_u8* wp = p.w;
            f8n_u32 n = min(n0 + r, p.N - 1);
#endif
            f8n_cp16(sbase + F8N_BM * 128 + f8n_swz(r, c), wp + (f8n_u64)n * rs + k0 + c * 16);
        }
        f8n_u32 sc = sbase + (F8N_BM + F8N_BN) * 128;
        for (f8n_u32 r = tid; r < F8N_BM + F8N_BN; r += F8N_THREADS)
        {
            if (r < F8N_BM)
            {
                f8n_u32 t = min(t0 + r, p.T - 1);
                f8n_cp4(sc + r * 4, p.x + (f8n_u64)t * rs + K + kb * 4);
            }
            else
            {
                f8n_u32 rr = r - F8N_BM;
#if F8N_DUAL
                const f8n_u8* wp = ((rr >> 3) & 1) ? p.w2 : p.w;
                f8n_u32 n = min(n0 + (rr >> 4) * 8 + (rr & 7), p.N - 1);
#else
                const f8n_u8* wp = p.w;
                f8n_u32 n = min(n0 + rr, p.N - 1);
#endif
                f8n_cp4(sc + r * 4, wp + (f8n_u64)n * rs + K + kb * 4);
            }
        }
    };

    float acc[F8N_FM][F8N_FN][4];
#pragma unroll
    for (int i = 0; i < F8N_FM; i++)
#pragma unroll
        for (int j = 0; j < F8N_FN; j++)
#pragma unroll
            for (int e = 0; e < 4; e++)
                acc[i][j][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < F8N_STAGES - 1; s++)
    {
        if (s < (int)nk)
        {
            load(s, s);
        }
        f8n_commit();
    }

    const f8n_u32 g = lane >> 2, c4 = lane & 3;
    for (f8n_u32 kb = 0; kb < nk; kb++)
    {
        f8n_wait<F8N_STAGES - 2>();
        __syncthreads();
        {
            f8n_u32 next = kb + F8N_STAGES - 1;
            if (next < nk && !F8N_NOLOAD)
            {
                load(next % F8N_STAGES, next);
            }
            f8n_commit();
        }
        const f8n_u8* base = smem + (kb % F8N_STAGES) * F8N_STAGE_BYTES;
#ifdef F8N_NOMMA
        if (kb < 1000000) { acc[0][0][0] += base[tid]; continue; }
#endif
        const f8n_u32 sa = f8n_smem(base), sb = sa + F8N_BM * 128;
        const float* ssa = (const float*)(base + (F8N_BM + F8N_BN) * 128);
        const float* ssb = ssa + F8N_BM;
        // B fragments of the whole K block.
        f8n_u32 b[4][F8N_FN][2];
#pragma unroll
        for (int kk = 0; kk < 4; kk++)
        {
#pragma unroll
            for (int j = 0; j < F8N_FN; j += 2)
            {
                f8n_u32 r = wn * F8N_TN + j * 8 + (lane & 7) + ((lane >> 4) << 3);
                f8n_ldsm4(sb + f8n_swz(r, kk * 2 + ((lane >> 3) & 1)), b[kk][j][0], b[kk][j][1], b[kk][j + 1][0], b[kk][j + 1][1]);
            }
        }
        float wb[F8N_FN][2];
#pragma unroll
        for (int j = 0; j < F8N_FN; j++)
        {
            wb[j][0] = ssb[wn * F8N_TN + j * 8 + c4 * 2];
            wb[j][1] = ssb[wn * F8N_TN + j * 8 + c4 * 2 + 1];
        }
#pragma unroll
        for (int i = 0; i < F8N_FM; i++)
        {
            float tmp[F8N_FN][4];
#pragma unroll
            for (int j = 0; j < F8N_FN; j++)
#pragma unroll
                for (int e = 0; e < 4; e++)
                    tmp[j][e] = 0.0f;
#pragma unroll
            for (int kk = 0; kk < 4; kk++)
            {
                f8n_u32 a[4];
                f8n_u32 r = wm * F8N_TM + i * 16 + (lane & 15);
                f8n_ldsm4(sa + f8n_swz(r, kk * 2 + (lane >> 4)), a[0], a[1], a[2], a[3]);
#pragma unroll
                for (int j = 0; j < F8N_FN; j++)
                    f8n_mma(F8N_NOPROMO ? acc[i][j] : tmp[j], a, b[kk][j][0], b[kk][j][1]);
            }
            if (F8N_NOPROMO) continue;
            float x0 = ssa[wm * F8N_TM + i * 16 + g], x1 = ssa[wm * F8N_TM + i * 16 + g + 8];
#pragma unroll
            for (int j = 0; j < F8N_FN; j++)
            {
                acc[i][j][0] = fmaf(tmp[j][0], x0 * wb[j][0], acc[i][j][0]);
                acc[i][j][1] = fmaf(tmp[j][1], x0 * wb[j][1], acc[i][j][1]);
                acc[i][j][2] = fmaf(tmp[j][2], x1 * wb[j][0], acc[i][j][2]);
                acc[i][j][3] = fmaf(tmp[j][3], x1 * wb[j][1], acc[i][j][3]);
            }
        }
    }
    f8n_wait<0>();

#pragma unroll
    for (int i = 0; i < F8N_FM; i++)
    {
#pragma unroll
        for (int h = 0; h < 2; h++)
        {
            f8n_u32 t = t0 + wm * F8N_TM + i * 16 + g + h * 8;
            if (t >= p.T || (F8N_NOSTORE && acc[i][0][0] != 12345.0f))
            {
                continue;
            }
            float* yr = p.y + (f8n_u64)t * p.ldy;
#if F8N_DUAL
#pragma unroll
            for (int j = 0; j < F8N_FN; j += 2)
            {
                f8n_u32 n = n0 + (wn * F8N_TN) / 2 + (j / 2) * 8 + c4 * 2;
                float v0 = F8N_EPI2(acc[i][j][h * 2], acc[i][j + 1][h * 2]);
                float v1 = F8N_EPI2(acc[i][j][h * 2 + 1], acc[i][j + 1][h * 2 + 1]);
                if (n + 1 < p.N)
                {
                    *(float2*)(yr + n) = make_float2(v0, v1);
                }
                else if (n < p.N)
                {
                    yr[n] = v0;
                }
            }
#else
#pragma unroll
            for (int j = 0; j < F8N_FN; j++)
            {
                f8n_u32 n = n0 + wn * F8N_TN + j * 8 + c4 * 2;
                if (n + 1 < p.N)
                {
                    *(float2*)(yr + n) = make_float2(acc[i][j][h * 2], acc[i][j][h * 2 + 1]);
                }
                else if (n < p.N)
                {
                    yr[n] = acc[i][j][h * 2];
                }
            }
#endif
        }
    }
}
