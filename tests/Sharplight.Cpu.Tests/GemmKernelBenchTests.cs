using System.Diagnostics;

namespace Sharplight.Cpu.Tests;

/// <summary>Single-thread throughput of the f32 GEMM micro-kernel on cache-resident panels; run with SHARPLIGHT_BENCH=1.</summary>
public unsafe class GemmKernelBenchTests
{
    [Fact]
    public void MicroKernel()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH") == "1", "set SHARPLIGHT_BENCH=1 to run benchmarks");
        int mr = Gemm.MR;
        int nr = Gemm.NR;
        int kc = 256;
        float* a = (float*)Mem.Alloc((long)kc * mr * 4, true);
        float* b = (float*)Mem.Alloc((long)kc * nr * 4, true);
        float* c = (float*)Mem.Alloc((long)mr * nr * 4, true);
        for (int i = 0; i < kc * mr; i++)
        {
            a[i] = 0.001f * (i % 7);
        }

        for (int i = 0; i < kc * nr; i++)
        {
            b[i] = 0.001f * (i % 5);
        }

        int reps = 20000;
        double best = double.MaxValue;
        for (int round = 0; round < 5; round++)
        {
            long t0 = Stopwatch.GetTimestamp();
            for (int r = 0; r < reps; r++)
            {
                Gemm.Tile(a, b, nr, c, nr, kc, null, r > 0);
            }

            best = Math.Min(best, Stopwatch.GetElapsedTime(t0).TotalSeconds);
        }

        double flops = 2.0 * mr * nr * kc * reps;
        Console.Error.WriteLine($"micro-kernel {mr}x{nr} kc={kc}: {flops / best / 1e9:F1} GFLOP/s single thread");
        Mem.Free((byte*)a);
        Mem.Free((byte*)b);
        Mem.Free((byte*)c);
    }
}
