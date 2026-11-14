using System.Diagnostics;
using Sharplight.Graph;
using Sharplight.Models;
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
    [InlineData("q8_0", 1), InlineData("q4_0", 1), InlineData("q4_k", 1), InlineData("q5_k", 1), InlineData("q6_k", 1), InlineData("f16", 1)]
    [InlineData("q8_0", 4), InlineData("q4_k", 4), InlineData("q8_0", 16), InlineData("q4_k", 16), InlineData("q8_0", 32), InlineData("q4_k", 32), InlineData("q6_k", 32), InlineData("q8_0", 128), InlineData("q4_k", 128), InlineData("q6_k", 128)]
    [InlineData("f32", 128), InlineData("q8_0", 512), InlineData("q4_k", 512)]
    [InlineData("q8_0", 1, 1024, 1024), InlineData("q8_0", 1, 3072, 1024), InlineData("q4_k", 1, 3072, 1024)]
    [InlineData("f32", 197, 3072, 768), InlineData("f32", 197, 768, 3072), InlineData("f32", 197, 2304, 768), InlineData("f32", 1576, 3072, 768)]
    public void MatMulThroughput(string type, int tokens, int N = 17408, int K = 5120)
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH") == "1", "set SHARPLIGHT_BENCH=1 to run benchmarks");
        if (int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_THREADS"), out var thr))
        {
            CpuParallel.Threads = thr;
        }

        var dt = DataType.Get(type);
        var data = type == "f16" ? new byte[(long)N * K * 2] : type == "f32" ? new byte[(long)N * K * 4] : RandomBlocks(dt, N, K, 1);
        if (type == "f32")
        {
            Kit.Rand(N * K, 1).AsSpan().CopyTo(System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(data.AsSpan()));
        }
        if (type == "f16") { new Random(1).NextBytes(data); for (long i = 1; i < data.Length; i += 2)
            {
                data[i] &= 0x3B;
            }
        }
        var tg = new TestGraph();
        var w = tg.W("w", dt, new Shape(N, K), data);
        var x = tg.InF("x", TestGraph.T, K);
        tg.B.Output("y", tg.B.MatMul(x, w));
        // SHARPLIGHT_BENCH_INTDOT=0 measures the f32 matrix-vector path instead of the integer-dot one.
        bool intDot = Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH_INTDOT") != "0";
        using var m = Kit.Compile(tg.B.Build(), maxTokens: tokens, intDot: intDot);
        Kit.Set(m, x, Kit.Rand(tokens * K, 2));
        var p = Kit.P(tokens);
        for (int i = 0; i < 5; i++)
        {
            m.Run(p);
        }

        int iters = tokens <= 16 ? (N * (long)K < 1 << 24 ? 3000 : 300) : 2.0 * tokens * N * K < 1e10 ? 300 : 10;
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
        Report($"MatMul {type,-5} {N}x{K} T={tokens,-4} {sec * 1e3,8:F3} ms  weights {data.Length / 1e6:F1} MB  {gbps,7:F1} GB/s  {gflops,8:F1} GFLOP/s  (p10 of {iters} runs, threads {CpuParallel.Threads}, {CpuIsa.Name(CpuIsa.Level)}{(intDot ? "" : ", no int-dot")})");
    }
    /// <summary>Cached attention (Qwen3-0.6B shape: 16 query heads, 8 KV heads, head size 128, f16 cache) after <paramref name="past"/> tokens.</summary>
    [Theory]
    [InlineData(1, 512), InlineData(1, 4096), InlineData(512, 0)]
    public void AttentionThroughput(int tokens, int past)
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH") == "1", "set SHARPLIGHT_BENCH=1 to run benchmarks");
        const int Hq = 16;
        const int Hkv = 8;
        const int D = 128;
        int ctx = past + tokens;
        var tg = new TestGraph();
        var T = TestGraph.T;
        var qi = tg.InF("q", T, Hq, D);
        var kc = tg.State("kc", DataType.F16, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D);
        var vc = tg.State("vc", DataType.F16, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
        tg.B.Output("o", tg.B.CachedAttention(qi, kc, vc, 1f / MathF.Sqrt(D)));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: tokens, ctx: ctx);
        Kit.Set(m, qi, Kit.Rand(tokens * Hq * D, 1));
        var p = Kit.P(tokens, past);
        for (int i = 0; i < 5; i++)
        {
            m.Run(p);
        }

        int iters = tokens == 1 ? 2000 : 20;
        var times = new double[iters];
        for (int i = 0; i < iters; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            m.Run(p);
            times[i] = Stopwatch.GetElapsedTime(t0).TotalSeconds;
        }

        Array.Sort(times);
        double sec = times[Math.Max(0, iters / 10)];
        double keys = tokens == 1 ? past + 1 : (tokens + 1) / 2.0;
        double gflops = 4.0 * tokens * Hq * D * keys / sec / 1e9;
        Report($"Attention T={tokens,-4} past={past,-5} {sec * 1e6,9:F1} us  {gflops,8:F1} GFLOP/s  (p10 of {iters} runs, threads {CpuParallel.Threads}, {CpuIsa.Name(CpuIsa.Level)})");
    }

    /// <summary>Bidirectional encoder attention over f32 Q/K/V: whisper-tiny (1500 rows, 6 heads) and ViT-B (197 rows, 12 heads).</summary>
    [Theory]
    [InlineData(1500, 6, 64), InlineData(197, 12, 64)]
    public void EncoderAttentionThroughput(int rows, int heads, int d)
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH") == "1", "set SHARPLIGHT_BENCH=1 to run benchmarks");
        var tg = new TestGraph();
        var qi = tg.InF("q", TestGraph.T, heads, d);
        var ki = tg.InF("k", TestGraph.T, heads, d);
        var vi = tg.InF("v", TestGraph.T, heads, d);
        tg.B.Output("o", tg.B.Emit1("Attention", [qi, ki, vi], Attributes.From(("scale", 1f / MathF.Sqrt(d)))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: rows);
        Kit.Set(m, qi, Kit.Rand(rows * heads * d, 1));
        Kit.Set(m, ki, Kit.Rand(rows * heads * d, 2));
        Kit.Set(m, vi, Kit.Rand(rows * heads * d, 3));
        var p = Kit.P(rows);
        for (int i = 0; i < 5; i++)
        {
            m.Run(p);
        }

        const int Iters = 50;
        var times = new double[Iters];
        for (int i = 0; i < Iters; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            m.Run(p);
            times[i] = Stopwatch.GetElapsedTime(t0).TotalSeconds;
        }

        Array.Sort(times);
        double sec = times[Iters / 10];
        double gflops = 4.0 * rows * rows * heads * d / sec / 1e9;
        Report($"EncoderAttention rows={rows,-5} heads={heads,-3} {sec * 1e6,9:F1} us  {gflops,8:F1} GFLOP/s  (p10 of {Iters} runs, threads {CpuParallel.Threads}, {CpuIsa.Name(CpuIsa.Level)})");
    }
}
