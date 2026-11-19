#include <cstdio>
template <int FM, int FN, int WARPS>
__global__ void k(float* out, int iters)
{
    __shared__ __align__(128) unsigned char sm[32768];
    for (int i = threadIdx.x; i < 32768 / 4; i += blockDim.x) ((unsigned*)sm)[i] = i * 2654435761u;
    __syncthreads();
    float c[FM][FN][4] = {};
    unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned base = (unsigned)__cvta_generic_to_shared(sm);
    for (int it = 0; it < iters; it++)
    {
#pragma unroll
        for (int kk = 0; kk < 4; kk++)
        {
            unsigned a[FM][4], b[FN][2];
#pragma unroll
            for (int i = 0; i < FM; i++)
            {
                unsigned r = (warp & 1) * 64 + i * 16 + (lane & 15);
                unsigned addr = base + r * 128 + (((kk * 2 + (lane >> 4)) ^ (r & 7)) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(a[i][0]), "=r"(a[i][1]), "=r"(a[i][2]), "=r"(a[i][3]) : "r"(addr));
            }
#pragma unroll
            for (int j = 0; j < FN; j += 2)
            {
                unsigned r = 128 + (warp >> 1) * 32 + j * 8 + (lane & 7) + ((lane >> 4) << 3);
                unsigned addr = base + r * 128 + (((kk * 2 + ((lane >> 3) & 1)) ^ (r & 7)) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(b[j][0]), "=r"(b[j][1]), "=r"(b[j+1][0]), "=r"(b[j+1][1]) : "r"(addr));
            }
#pragma unroll
            for (int i = 0; i < FM; i++)
#pragma unroll
                for (int j = 0; j < FN; j++)
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(c[i][j][0]), "+f"(c[i][j][1]), "+f"(c[i][j][2]), "+f"(c[i][j][3]) : "r"(a[i][0]), "r"(a[i][1]), "r"(a[i][2]), "r"(a[i][3]), "r"(b[j][0]), "r"(b[j][1]));
        }
    }
    float s = 0;
    for (int i = 0; i < FM; i++) for (int j = 0; j < FN; j++) s += c[i][j][0] + c[i][j][1] + c[i][j][2] + c[i][j][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
template <int FM, int FN, int WARPS> void run(int blocks)
{
    float* o; cudaMalloc(&o, 1 << 26);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int it = 2000;
    float best = 1e9;
    for (int r = 0; r < 5; r++) {
        cudaEventRecord(e0); k<FM, FN, WARPS><<<blocks, WARPS * 32>>>(o, it); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); if (ms < best) best = ms;
    }
    double flop = (double)blocks * WARPS * it * 4 * FM * FN * 8192;
    printf("FM %d FN %d warps %d blocks %d: %.0f TFLOPS\n", FM, FN, WARPS, blocks, flop / best / 1e9);
}
int main()
{
    run<4, 4, 8>(82); run<4, 4, 8>(164); run<4, 8, 4>(82); run<4, 8, 8>(82); run<2, 4, 8>(164);
}
