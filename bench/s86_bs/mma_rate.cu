// MMA throughput of the E4M3 forms of sm_120: legacy mma.sync e4m3 (f32 accumulate), kind::f8f6f4, kind::mxf8f6f4.block_scale.
// usage: mma_rate [iters]; build: nvcc -O3 -arch=sm_120a mma_rate.cu
#include <cstdio>
#include <cuda_runtime.h>
typedef unsigned int u32;
template <int FORM>
__device__ __forceinline__ void mma(float* c, const u32* a, u32 b0, u32 b1, u32 sa, u32 sb)
{
    if (FORM == 0)
    {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    else if (FORM == 1)
    {
        asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    else
    {
        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                       "r"(sa), "h"((unsigned short)0), "h"((unsigned short)0), "r"(sb), "h"((unsigned short)0), "h"((unsigned short)0));
    }
}
template <int FORM>
__global__ void k(float* out, int iters, u32 seed)
{
    u32 a[4] = {seed ^ threadIdx.x, seed * 3, seed * 5, seed * 7};
    u32 b[8][2];
    for (int j = 0; j < 8; j++) { b[j][0] = seed + j; b[j][1] = seed * j; }
    float c[8][4] = {};
    u32 sa = 127 + (threadIdx.x & 1), sb = 127;
    for (int i = 0; i < iters; i++)
    {
#pragma unroll
        for (int j = 0; j < 8; j++) { mma<FORM>(c[j], a, b[j][0], b[j][1], sa, sb); }
    }
    float s = 0;
    for (int j = 0; j < 8; j++) { s += c[j][0] + c[j][1] + c[j][2] + c[j][3]; }
    if (s == 1.2345f) { out[0] = s; }
}
template <int FORM>
void run(const char* name, int iters, int sms)
{
    float* out; cudaMalloc(&out, 4);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = sms * 4, threads = 256;
    k<FORM><<<blocks, threads>>>(out, iters / 10, 0x3c3c3c3c);
    float best = 1e9;
    for (int r = 0; r < 5; r++)
    {
        cudaEventRecord(e0); k<FORM><<<blocks, threads>>>(out, iters, 0x3c3c3c3c); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); if (ms < best) best = ms;
    }
    double flops = 2.0 * 16 * 8 * 32 * 8.0 * iters * (blocks * threads / 32);
    printf("%-28s %8.3f ms  %7.1f TFLOPS\n", name, best, flops / best / 1e9);
    cudaFree(out);
}
int main(int argc, char** argv)
{
    int iters = argc > 1 ? atoi(argv[1]) : 20000;
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    run<0>("e4m3 f32 (legacy)", iters, sms);
    run<1>("kind::f8f6f4 f32", iters, sms);
    run<2>("kind::mxf8f6f4.block_scale", iters, sms);
    run<0>("e4m3 f32 (legacy) again", iters, sms);
    return 0;
}
