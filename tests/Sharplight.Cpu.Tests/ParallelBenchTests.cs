using System.Diagnostics;

namespace Sharplight.Cpu.Tests;

/// <summary>Dispatch overhead of <see cref="CpuParallel.For"/> on the dedicated pool; run with SHARPLIGHT_BENCH=1.</summary>
public class ParallelBenchTests
{
    [Fact]
    public void ForOverhead()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH") == "1", "set SHARPLIGHT_BENCH=1 to run benchmarks");
        if (int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_THREADS"), out var thr))
        {
            CpuParallel.Threads = thr;
        }

        var sink = new long[1024 * 16];
        Action<int> body = i => sink[(i & 1023) * 16]++;
        foreach (int n in new[] { 16, 64, 256, 1024, 4096 })
        {
            double best = double.MaxValue;
            for (int round = 0; round < 20; round++)
            {
                const int reps = 200;
                long t0 = Stopwatch.GetTimestamp();
                for (int r = 0; r < reps; r++)
                {
                    CpuParallel.For(n, body);
                }

                best = Math.Min(best, Stopwatch.GetElapsedTime(t0).TotalSeconds / reps);
            }

            Console.Error.WriteLine($"CpuParallel.For({n}) empty items: {best * 1e6:F2} us, threads {CpuParallel.Threads}");
        }
    }
}
