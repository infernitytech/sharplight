// Where a programmatic dependent's blocks land (docs/benchmarks.md section 99.1): kernel A (NA blocks of 256 threads, each spinning for
// A_US microseconds) then kernel B (NB blocks, each recording its SM and spinning B_US), B launched plain (MODE=0), as a programmatic
// dependent of A with A's trigger at its entry (MODE=1) or with no trigger (MODE=2). Prints B's blocks per SM histogram and the span.
// B_SMEM / A_SMEM: dynamic shared memory per block of B / A; B_US: B's FMA work (x 4 ns per iteration count); MEM_KB: B also reads that
// many KB per block. Section 99.1: B's blocks pile onto the few idle SMs when B needs shared memory and A's blocks run with none.
// nvcc -O3 -arch=sm_120 s99_pdl_placement.cu -o pdl && NA=108 NB=32 B_SMEM=4224 A_US=3 B_US=40 MODE=1 ./pdl
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

__device__ unsigned smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }

__device__ void spin(long long ns)
{
    long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (;;)
    {
        long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        if (t - t0 >= ns) { break; }
    }
}

__global__ void ka(long long ns, int trigger)
{
    extern __shared__ int aTile[];
    if (ns < 0) { aTile[threadIdx.x] = 0; }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (trigger) { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
    spin(ns);
}

__global__ void kb(long long ns, unsigned* sm, long long* times, const float4* buf, int kb4)
{
    extern __shared__ float4 tile[];
    asm volatile("griddepcontrol.wait;" ::: "memory");
    long long t0, t1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    if (threadIdx.x == 0) { sm[blockIdx.x] = smid(); }
    // B_US is a work count here: ns / 4 iterations of four independent FMA chains (issue-bound: blocks sharing an SM slow each other).
    float a = threadIdx.x, b = a + 1, c = a + 2, d = a + 3;
    for (long long i = 0; i < ns / 4; i++) { a = fmaf(a, 0.999f, 1.0f); b = fmaf(b, 0.999f, 1.0f); c = fmaf(c, 0.999f, 1.0f); d = fmaf(d, 0.999f, 1.0f); }
    // MEM_KB: also read that many KB per block (block-private region, 16-byte loads, through shared memory when B_SMEM is set).
    float4 acc = make_float4(0, 0, 0, 0);
    for (int i = threadIdx.x; i < kb4; i += blockDim.x)
    {
        float4 v = buf[(size_t)blockIdx.x * kb4 + i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    a += acc.x + acc.y + acc.z + acc.w;
    if (a + b + c + d == 0.123f) { sm[0] = 0; }
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
    if (threadIdx.x == 0) { times[2 * blockIdx.x] = t0; times[2 * blockIdx.x + 1] = t1; }
}

int main()
{
    int na = atoi(getenv("NA") ? getenv("NA") : "108"), nb = atoi(getenv("NB") ? getenv("NB") : "32"), mode = atoi(getenv("MODE") ? getenv("MODE") : "1");
    long long aNs = (long long)(1000 * atof(getenv("A_US") ? getenv("A_US") : "2")), bNs = (long long)(1000 * atof(getenv("B_US") ? getenv("B_US") : "5"));
    int smem = atoi(getenv("B_SMEM") ? getenv("B_SMEM") : "0"), aSmem = atoi(getenv("A_SMEM") ? getenv("A_SMEM") : "0");
    unsigned* sm; long long* times;
    cudaMalloc(&sm, nb * 4); cudaMalloc(&times, nb * 16);
    int kb4 = atoi(getenv("MEM_KB") ? getenv("MEM_KB") : "0") * 1024 / 16;
    float4* buf; cudaMalloc(&buf, (size_t)nb * kb4 * 16 + 16); cudaMemset(buf, 0, (size_t)nb * kb4 * 16 + 16);
    cudaStream_t s; cudaStreamCreate(&s);
    std::vector<unsigned> h(nb); std::vector<long long> ht(2 * nb);
    std::map<int, int> hist; double spanSum = 0; int reps = 50;
    for (int r = 0; r < reps + 5; r++)
    {
        ka<<<na, 256, aSmem, s>>>(aNs, mode == 1);
        cudaLaunchAttribute at; at.id = cudaLaunchAttributeProgrammaticStreamSerialization; at.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg = {}; cfg.gridDim = nb; cfg.blockDim = 256; cfg.dynamicSmemBytes = smem; cfg.stream = s; cfg.attrs = &at; cfg.numAttrs = mode ? 1 : 0;
        cudaLaunchKernelEx(&cfg, kb, bNs, sm, times, (const float4*)buf, kb4);
        cudaStreamSynchronize(s);
        if (r < 5) { continue; }
        cudaMemcpy(h.data(), sm, nb * 4, cudaMemcpyDeviceToHost); cudaMemcpy(ht.data(), times, nb * 16, cudaMemcpyDeviceToHost);
        std::map<unsigned, int> per; for (unsigned x : h) { per[x]++; }
        for (auto& [k, v] : per) { hist[v]++; }
        long long lo = ht[0], hi = ht[1];
        for (int i = 0; i < nb; i++) { lo = ht[2 * i] < lo ? ht[2 * i] : lo; hi = ht[2 * i + 1] > hi ? ht[2 * i + 1] : hi; }
        spanSum += (hi - lo) / 1e3;
    }
    printf("NA=%d NB=%d A_US=%.1f B_US=%.1f MODE=%d: B span %.2f us; SMs holding k blocks of B (per launch):", na, nb, aNs / 1e3, bNs / 1e3, mode, spanSum / reps);
    for (auto& [k, v] : hist) { printf(" %d:%.1f", k, (double)v / reps); }
    printf(" %s\n", cudaGetErrorString(cudaGetLastError()));
}
