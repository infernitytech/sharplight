#include <cstdio>
template <int NACC>
__global__ void k(float* out, int iters)
{
    float c[NACC][4] = {};
    unsigned a[4] = {threadIdx.x, threadIdx.x*3, threadIdx.x*5, threadIdx.x*7}, b0 = threadIdx.x*11, b1 = threadIdx.x*13;
    for (int i = 0; i < iters; i++)
#pragma unroll
        for (int j = 0; j < NACC; j++)
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    float s = 0; for (int j = 0; j < NACC; j++) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
template <int NACC> void run(int threads, int blocks)
{
    float* o; cudaMalloc(&o, 1 << 26);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int it = 20000 / NACC * 8;
    float best = 1e9;
    for (int r = 0; r < 5; r++) {
        cudaEventRecord(e0); k<NACC><<<blocks, threads>>>(o, it); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); if (ms < best) best = ms;
    }
    double flop = (double)blocks * threads / 32 * it * NACC * 8192;
    printf("acc %2d threads %4d blocks %4d: %.0f TFLOPS\n", NACC, threads, blocks, flop / best / 1e9);
}
int main()
{
    run<4>(256, 82); run<2>(256, 82); run<4>(512, 82); run<8>(128,82); run<4>(128,82);
}
