// FP4 activation rows for the FP4 GEMM (gemm_fp4_native.cu), in CUDA C++ for the CUDA provider (DeviceFeatures.BlockScaledFp4,
// compute capability 12.x, family-specific PTX for the E2M1 conversion): f32 rows X_STRIDE floats apart into rows of
// K / 256 * 160 + 32 bytes (a multiple of 32, so every record starts a 32-byte sector): for each 256 values of K a 160-byte record
// [128 code bytes][16 UE4M3 scales][16 bytes padding], then 32 bytes holding the row's f32 global scale gx.
// gx = amax(row) / (6 * 448); a 16-value group's scale is the E4M3 code of amax(group) / (6 * gx) (round to nearest), each value the
// E2M1 code of x / (scale * gx) (round to nearest even, saturating). The 8 code bytes of a group hold value i in the low nibble of
// byte i and value i + 8 in the high nibble (the NVFP4 weights' sub-block layout), so that the MMA pairs equal K positions.
// One block of 128 threads per row; requires K % 256 == 0 and 16-byte aligned rows (X_STRIDE % 4 == 0). Specialization macros: K,
// X_STRIDE. Grid: one block per row.
#define SL_FAMILY_ARCH 1

typedef unsigned char f4n_u8;
typedef unsigned short f4n_u16;
typedef unsigned int f4n_u32;
typedef unsigned long long f4n_u64;

#define F4_XREC 160
#define F4_XRB ((f4n_u64)(K / 256) * F4_XREC + 32)

struct Q4nParams
{
    const float* x;
    f4n_u8* q;
    f4n_u32 T;
};

// E4M3 code of a non-negative value (round to nearest even, saturating at 448).
static __device__ __forceinline__ f4n_u32 q4n_e4m3(float v)
{
    f4n_u16 r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.0f), "f"(v));
    return r & 0xFFu;
}

// The value of an E4M3 code.
static __device__ __forceinline__ float q4n_e4m3_value(f4n_u32 code)
{
    f4n_u32 e = (code >> 3) & 15u, m = code & 7u;
    return e == 0 ? (float)m * (1.0f / 512.0f) : __int_as_float((int)(((e + 120u) << 23) | (m << 20)));
}

// The value of the E2M1 code in the low nibble of c.
static __device__ __forceinline__ float q4n_e2m1_value(f4n_u32 c)
{
    // The doubled magnitudes 0, 1, 2, 3, 4, 6, 8, 12 as nibbles.
    float m = (float)((0xC8643210u >> ((c & 7u) * 4)) & 15u) * 0.5f;
    return (c & 8u) ? -m : m;
}

// The byte of two E2M1 codes: lo in the low nibble, hi in the high one.
static __device__ __forceinline__ f4n_u32 q4n_e2m1x2(float lo, float hi)
{
    f4n_u16 r;
    asm("{\n.reg .b8 t;\ncvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\ncvt.u16.u8 %0, t;\n}" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}

#define Q4N_THREADS 128

extern "C" __global__ void __launch_bounds__(Q4N_THREADS) quantize_fp4(Q4nParams p)
{
    __shared__ float amax[K / 16];
    __shared__ float warpMax[Q4N_THREADS / 32];
    const f4n_u32 row = blockIdx.x, tid = threadIdx.x;
    if (row >= p.T)
    {
        return;
    }

    const float* xr = p.x + (f4n_u64)row * X_STRIDE;
    float m = 0.0f;
    for (f4n_u32 g = tid; g < K / 16; g += Q4N_THREADS)
    {
        const float4* v = (const float4*)(xr + g * 16);
        float a = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float4 f = v[i];
            a = fmaxf(a, fmaxf(fmaxf(fabsf(f.x), fabsf(f.y)), fmaxf(fabsf(f.z), fabsf(f.w))));
        }
        amax[g] = a;
        m = fmaxf(m, a);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
    }
    if ((tid & 31) == 0)
    {
        warpMax[tid >> 5] = m;
    }
    __syncthreads();
    m = warpMax[0];
#pragma unroll
    for (int w = 1; w < Q4N_THREADS / 32; w++)
    {
        m = fmaxf(m, warpMax[w]);
    }

    // gx = m / 2688 (6 * 448): the largest group scale becomes 448, the largest E4M3 value.
    const float gx = m > 0.0f ? m * (1.0f / 2688.0f) : 1.0f;
    const float toScale = m > 0.0f ? 448.0f / m : 0.0f;
    f4n_u8* qr = p.q + (f4n_u64)row * F4_XRB;
    for (f4n_u32 g = tid; g < K / 16; g += Q4N_THREADS)
    {
        const float4* v4 = (const float4*)(xr + g * 16);
        float v[16];
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float4 f = v4[i];
            v[i * 4] = f.x;
            v[i * 4 + 1] = f.y;
            v[i * 4 + 2] = f.z;
            v[i * 4 + 3] = f.w;
        }
        // Two candidate scales: the group maximum at 6 (the largest E2M1 value) and at 4 (finer steps for the other values, the maximum
        // off by at most a third); the one with the smaller squared error is kept.
        f4n_u32 code = 0, lo = 0, hi = 0;
        float best = 3.4e38f;
#pragma unroll
        for (int c = 0; c < 2; c++)
        {
            f4n_u32 cand = q4n_e4m3(amax[g] * toScale * (c == 0 ? 1.0f : 1.5f));
            float scale = q4n_e4m3_value(cand) * gx;
            float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
            f4n_u32 l = 0, h = 0;
            float err = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                f4n_u32 b0 = q4n_e2m1x2(v[i] * inv, v[i + 8] * inv), b1 = q4n_e2m1x2(v[i + 4] * inv, v[i + 12] * inv);
                l |= b0 << (8 * i);
                h |= b1 << (8 * i);
                float d0 = v[i] - q4n_e2m1_value(b0) * scale, d1 = v[i + 8] - q4n_e2m1_value(b0 >> 4) * scale;
                float d2 = v[i + 4] - q4n_e2m1_value(b1) * scale, d3 = v[i + 12] - q4n_e2m1_value(b1 >> 4) * scale;
                err += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
            }
            if (err < best)
            {
                best = err;
                code = cand;
                lo = l;
                hi = h;
            }
        }
        f4n_u8* rec = qr + (f4n_u64)(g / 16) * F4_XREC;
        *(uint2*)(rec + (g % 16) * 8) = make_uint2(lo, hi);
        rec[128 + g % 16] = (f4n_u8)code;
    }
    if (tid == 0)
    {
        *(float*)(qr + (f4n_u64)(K / 256) * F4_XREC) = gx;
    }
}

