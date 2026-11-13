using System.Diagnostics;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>Throughput measurements; run with SHARPLIGHT_BENCH=1 (results go to stderr and $SHARPLIGHT_BENCH_OUT if set).</summary>
public class BenchTests
{
    private static void Report(string s)
    {
        Console.Error.WriteLine(s);
        if (Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH_OUT") is { Length: > 0 } f)
        {
            File.AppendAllText(f, s + Environment.NewLine);
        }
    }

    private static byte[] RandomBlocks(DataType t, long rows, long k, int seed)
    {
        var rng = new Random(seed);
        var data = new byte[rows * t.ByteSize(k)];
        rng.NextBytes(data);
        int bb = t.BlockBytes;
        long nb = data.Length / bb;
        for (long b = 0; b < nb; b++)
        {
            BitConverter.TryWriteBytes(data.AsSpan((int)(b * bb), 2), (Half)(0.01f + 0.01f * (b % 5)));
        }

        if (t.BlockSize == 256)
        {
            for (long b = 0; b < nb; b++) { if (t == DataType.Q6_K)
                {
                    BitConverter.TryWriteBytes(data.AsSpan((int)(b * bb + 208), 2), (Half)0.02f);
                }
                else
                {
                    BitConverter.TryWriteBytes(data.AsSpan((int)(b * bb + 2), 2), (Half)0.01f);
                }
            }
        }

        return data;
    }

    [Theory]
    [InlineData("q8_0", 1), InlineData("q4_0", 1), InlineData("q4_k", 1), InlineData("q5_k", 1), InlineData("q6_k", 1), InlineData("f16", 1), InlineData("q8_0", 128), InlineData("q4_k", 128)]
    public void MatMulThroughput(string type, int tokens)
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH") == "1", "set SHARPLIGHT_BENCH=1 to run benchmarks");
        const int N = 17408;
        const int K = 5120;
        if (int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_THREADS"), out var thr))
        {
            CpuParallel.Threads = thr;
        }

        var dt = DataType.Get(type);
        var data = type == "f16" ? new byte[(long)N * K * 2] : RandomBlocks(dt, N, K, 1);
        if (type == "f16") { new Random(1).NextBytes(data); for (long i = 1; i < data.Length; i += 2)
            {
                data[i] &= 0x3B;
            }
        }
        var tg = new TestGraph();
        var w = tg.W("w", dt, new Shape(N, K), data);
        var x = tg.InF("x", TestGraph.T, K);
        tg.B.Output("y", tg.B.MatMul(x, w));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: tokens);
        Kit.Set(m, x, Kit.Rand(tokens * K, 2));
        var p = Kit.P(tokens);
        for (int i = 0; i < 5; i++)
        {
            m.Run(p);
        }

        int iters = tokens == 1 ? 300 : 10;
        var times = new double[iters];
        for (int i = 0; i < iters; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            m.Run(p);
            times[i] = Stopwatch.GetElapsedTime(t0).TotalSeconds;
        }
        Array.Sort(times);
        double sec = times[Math.Max(0, iters / 10)]; // 10th percentile: robust against other processes sharing the machine
        double gbps = data.Length / sec / 1e9;
        double gflops = 2.0 * tokens * N * K / sec / 1e9;
        Report($"MatMul {type,-5} {N}x{K} T={tokens,-4} {sec * 1e3,8:F3} ms  weights {data.Length / 1e6:F1} MB  {gbps,7:F1} GB/s  {gflops,8:F1} GFLOP/s  (p10 of {iters} runs, threads {CpuParallel.Threads})");
    }
}
