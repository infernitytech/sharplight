using System.Diagnostics;
using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Cpu.Tests;

/// <summary>
/// Winograd F(4x4, 3x3) for channels-last 3x3 stride-1 convs (<see cref="WinogradConvOp"/>): results against the
/// f64-accumulated reference (edge tiles, asymmetric padding, batches, every epilogue form), its error against the direct
/// channels-last conv's, and (with SHARPLIGHT_BENCH=1) its throughput against the direct conv. The tests of this class toggle
/// <see cref="WinogradConvOp.Policy"/> and run one after another.
/// </summary>
public class WinogradConvTests
{
    public static IEnumerable<object[]> Cases()
    {
        // C, H, W, O, pads (t, l, b, r), N, epilogue
        yield return [32, 16, 16, 32, new[] { 1, 1, 1, 1 }, 1, "none"];
        yield return [64, 14, 14, 64, new[] { 1, 1, 1, 1 }, 2, "relu"];
        yield return [48, 15, 17, 80, new[] { 1, 1, 1, 1 }, 3, "residual-clip"];
        yield return [32, 19, 21, 48, new[] { 0, 0, 0, 0 }, 1, "silu"];
        yield return [64, 18, 14, 32, new[] { 1, 2, 0, 1 }, 2, "none"];
        yield return [272, 13, 13, 96, new[] { 1, 1, 1, 1 }, 1, "relu"];
        yield return [128, 28, 28, 128, new[] { 1, 1, 1, 1 }, 1, "residual-clip"];
    }

    [Theory, MemberData(nameof(Cases))]
    public void MatchesReference(int C, int H, int W, int O, int[] pads, int N, string epilogue)
    {
        var (exp, act) = Run(C, H, W, O, pads, N, epilogue, true, 0.5f / MathF.Sqrt(9 * C));
        Kit.AssertClose(exp, act, 2e-5, 2e-4, $"C={C} {H}x{W} O={O} N={N} {epilogue}");
    }

    /// <summary>The Winograd error is a small multiple of the direct conv's (RMS against the f64 reference).</summary>
    [Theory]
    [InlineData(64, 56, 56, 64)]
    [InlineData(128, 28, 28, 128)]
    [InlineData(256, 14, 14, 256)]
    [InlineData(96, 21, 19, 160)]
    public void ErrorIsNearTheDirectConv(int C, int H, int W, int O)
    {
        // Below 256-bit vectors WinogradConvOp.Applies is false and both runs take the direct conv.
        Assert.SkipUnless(CpuIsa.Use256, "Winograd runs only with 256-bit vectors");
        float scale = 1f / MathF.Sqrt(9 * C);
        var (exp, wino) = Run(C, H, W, O, [1, 1, 1, 1], 1, "none", true, scale);
        var (_, direct) = Run(C, H, W, O, [1, 1, 1, 1], 1, "none", false, scale);
        double rmsWino = Rms(exp, wino);
        double rmsDirect = Rms(exp, direct);
        double rmsRef = Math.Sqrt(exp.Average(v => (double)v * v));
        TestContext.Current.SendDiagnosticMessage($"C={C} {H}x{W}: rms error winograd {rmsWino:E2}, direct {rmsDirect:E2}, output rms {rmsRef:E2}");
        Assert.NotEqual(rmsWino, rmsDirect);   // the Winograd path did run
        Assert.True(rmsWino < 5 * rmsDirect + 1e-7, $"winograd rms error {rmsWino:E2} against direct {rmsDirect:E2}");
        Assert.True(rmsWino < 1e-5 * rmsRef, $"winograd rms error {rmsWino:E2} against output rms {rmsRef:E2}");
    }

    /// <summary>Shapes the direct conv runs faster: too few tiles per image, transformed weights too large, other kernels.</summary>
    [Fact]
    public void AppliesOnlyWhereItPays()
    {
        var prev = WinogradConvOp.Policy;
        WinogradConvOp.Policy = "1";
        try
        {
            AssertApplies();
        }
        finally
        {
            WinogradConvOp.Policy = prev;
        }
    }

    /// <summary>
    /// One compiled model over a symbolic batch with a minimum batch of 2: batch 1 runs the kept direct conv, batch 3
    /// Winograd, both against the reference.
    /// </summary>
    [Fact]
    public void SmallBatchesRunTheDirectConv()
    {
        Assert.SkipUnless(CpuIsa.Use256, "needs 256-bit vectors");
        const int C = 64;
        const int H = 16;
        var w = Kit.Rand(C * C * 9, 21, 0.05f);
        var bias = Kit.Rand(C, 22, 0.1f);
        var prev = WinogradConvOp.Policy;
        WinogradConvOp.Policy = "2";
        CompiledModel m;
        Value xin;
        try
        {
            var tg = new TestGraph();
            xin = tg.InF("x", TestGraph.T, H, H, C);
            var conv = tg.B.Emit1("Conv", [tg.B.Permute(xin, 0, 3, 1, 2), tg.WF("w", w, C, C, 3, 3), tg.WF("b", bias, C)], Attributes.From(("pads", new long[] { 1, 1, 1, 1 })));
            tg.B.Output("y", tg.B.Emit1("Pool", [tg.B.Unary("Relu", conv)], OnePool()));
            m = Kit.Compile(tg.B.Build(), maxTokens: 3);
        }
        finally
        {
            WinogradConvOp.Policy = prev;
        }

        using (m)
        {
            double[] errors = new double[2];
            int k = 0;
            foreach (int n in new[] { 1, 3 })
            {
                var x = Kit.Rand(3 * H * H * C, 23)[..(n * H * H * C)];
                Kit.Set(m, xin, x);
                var p = Kit.P(n);
                m.Run(p);
                var act = Kit.Out(m, "y", p);
                int[] one = [1, 1];
                var exp = SRef.Conv(Nchw(x, n, C, H, H), n, C, H, H, w, C, 3, 3, bias, one, [1, 1, 1, 1], one, 1, out _, out _).Select(v => Math.Max(0, v)).ToArray();
                Kit.AssertClose(exp, act, 2e-5, 2e-4, $"N={n}");
                errors[k++] = Rms(exp[..(H * H * C)], act[..(H * H * C)]);
            }

            // Same first image: the direct conv's error is several times smaller than Winograd's.
            Assert.True(errors[0] < errors[1], $"batch 1 rms {errors[0]:E2} should be the direct conv's, below Winograd's {errors[1]:E2}");
        }
    }

    private static Attributes OnePool() => Attributes.From(
        ("mode", "max"), ("kernel", new long[] { 1, 1 }), ("strides", new long[] { 1, 1 }), ("pads", new long[] { 0, 0, 0, 0 }),
        ("dilations", new long[] { 1, 1 }), ("ceil_mode", false), ("count_include_pad", false));

    private static void AssertApplies()
    {
        int[] one = [1, 1];
        Assert.True(WinogradConvOp.Applies(64, 64, 56, 56, 3, 3, one, one, null) || !CpuIsa.Use256);
        Assert.False(WinogradConvOp.Applies(512, 512, 7, 7, 3, 3, one, one, null));
        Assert.False(WinogradConvOp.Applies(512, 512, 28, 28, 3, 3, one, one, null));
        Assert.False(WinogradConvOp.Applies(64, 64, 56, 56, 3, 3, [2, 2], one, null));
        Assert.False(WinogradConvOp.Applies(64, 64, 56, 56, 3, 3, one, [2, 2], null));
        Assert.False(WinogradConvOp.Applies(64, 64, 56, 56, 5, 5, one, one, null));
        Assert.False(WinogradConvOp.Applies(24, 64, 56, 56, 3, 3, one, one, null));
        Assert.False(WinogradConvOp.Applies(64, 40, 56, 56, 3, 3, one, one, null));
    }

    /// <summary>ResNet-50's 3x3 convs at batch 1 and 8: Winograd against the direct channels-last conv.</summary>
    [Theory]
    [InlineData(64, 56, 1), InlineData(128, 28, 1), InlineData(256, 14, 1), InlineData(512, 7, 1)]
    [InlineData(64, 56, 8), InlineData(128, 28, 8), InlineData(256, 14, 8), InlineData(512, 7, 8)]
    public void Throughput(int C, int H, int N)
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH") == "1", "set SHARPLIGHT_BENCH=1 to run benchmarks");
        if (int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_THREADS"), out var thr))
        {
            CpuParallel.Threads = thr;
        }

        double direct = Time(C, H, N, false);
        double wino = Time(C, H, N, true);
        double flops = 2.0 * N * H * H * C * C * 9;
        string line = $"Conv3x3 C={C,-4} {H,2}x{H,-2} N={N}  direct {direct * 1e6,8:F1} us {flops / direct / 1e12,5:F2} TFLOPS   winograd {wino * 1e6,8:F1} us {flops / wino / 1e12,5:F2} TFLOPS eq.  {direct / wino:F2}x  (threads {CpuParallel.Threads}, {CpuIsa.Name(CpuIsa.Level)})";
        Console.Error.WriteLine(line);
        if (Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH_OUT") is { Length: > 0 } path)
        {
            File.AppendAllText(path, line + Environment.NewLine);
        }
    }

    /// <summary>A ResNet bottleneck block (1x1 → 3x3 → 1x1 + residual, 4C outer channels), Winograd against the direct conv.</summary>
    [Theory]
    [InlineData(64, 56, 1), InlineData(128, 28, 1), InlineData(256, 14, 1)]
    [InlineData(64, 56, 8), InlineData(128, 28, 8), InlineData(256, 14, 8)]
    public void BottleneckThroughput(int C, int H, int N)
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH") == "1", "set SHARPLIGHT_BENCH=1 to run benchmarks");
        if (int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_THREADS"), out var thr))
        {
            CpuParallel.Threads = thr;
        }

        double direct = Time(C, H, N, false, true);
        double wino = Time(C, H, N, true, true);
        string line = $"Bottleneck C={C,-4} {H,2}x{H,-2} N={N}  direct {direct * 1e6,8:F1} us   winograd {wino * 1e6,8:F1} us  {direct / wino:F2}x  (threads {CpuParallel.Threads}, {CpuIsa.Name(CpuIsa.Level)})";
        Console.Error.WriteLine(line);
    }

    private static double Time(int C, int H, int N, bool winograd, bool bottleneck = false)
    {
        int outer = bottleneck ? 4 * C : C;
        var x = Kit.Rand(N * H * H * outer, 1);
        var w = Kit.Rand(C * C * 9, 2, 0.05f);
        var prev = WinogradConvOp.Policy;
        WinogradConvOp.Policy = winograd ? "1" : "0";
        try
        {
            var tg = new TestGraph();
            var xin = tg.InF("x", TestGraph.T, H, H, outer);
            var xv = tg.B.Permute(xin, 0, 3, 1, 2);
            var a = bottleneck ? tg.B.Unary("Relu", tg.B.Emit1("Conv", [xv, tg.WF("w1", Kit.Rand(C * outer, 3, 0.05f), C, outer, 1, 1), tg.WF("b1", Kit.Rand(C, 4, 0.1f), C)])) : xv;
            var conv = tg.B.Unary("Relu", tg.B.Emit1("Conv", [a, tg.WF("w", w, C, C, 3, 3), tg.WF("b", Kit.Rand(C, 5, 0.1f), C)], Attributes.From(("pads", new long[] { 1, 1, 1, 1 }))));
            var y = bottleneck ? tg.B.Unary("Relu", tg.B.Add(tg.B.Emit1("Conv", [conv, tg.WF("w3", Kit.Rand(C * outer, 6, 0.05f), outer, C, 1, 1)]), xv)) : conv;
            tg.B.Output("y", tg.B.Emit1("Pool", [y], Attributes.From(
                ("mode", "max"), ("kernel", new long[] { 1, 1 }), ("strides", new long[] { 1, 1 }), ("pads", new long[] { 0, 0, 0, 0 }),
                ("dilations", new long[] { 1, 1 }), ("ceil_mode", false), ("count_include_pad", false))));
            using var m = Kit.Compile(tg.B.Build(), maxTokens: N);
            Kit.Set(m, xin, x);
            var p = Kit.P(N);
            for (int i = 0; i < 5; i++)
            {
                m.Run(p);
            }

            const int Iters = 40;
            var times = new double[Iters];
            // SHARPLIGHT_BENCH_COLD=1 evicts the caches before every run (128 MB written), as the layers of a model see them.
            var sweep = Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH_COLD") == "1" ? new byte[128 << 20] : null;
            for (int i = 0; i < Iters; i++)
            {
                if (sweep != null)
                {
                    sweep.AsSpan().Fill((byte)i);
                }

                long t0 = Stopwatch.GetTimestamp();
                m.Run(p);
                times[i] = Stopwatch.GetElapsedTime(t0).TotalSeconds;
            }

            Array.Sort(times);
            return times[Iters / 10];
        }
        finally
        {
            WinogradConvOp.Policy = prev;
        }
    }

    private static (float[] Expected, float[] Actual) Run(int C, int H, int W, int O, int[] pads, int N, string epilogue, bool winograd, float wScale)
    {
        var w = Kit.Rand(O * C * 9, 11, wScale);
        var x = Kit.Rand(N * H * W * C, 12);
        var bias = Kit.Rand(O, 13, 0.1f);
        int ho = H + pads[0] + pads[2] - 2;
        int wo = W + pads[1] + pads[3] - 2;
        var res = Kit.Rand(N * ho * wo * O, 14);
        var prev = WinogradConvOp.Policy;
        WinogradConvOp.Policy = winograd ? "1" : "0";
        try
        {
            var tg = new TestGraph();
            var xin = tg.InF("x", TestGraph.T, H, W, C);
            var rin = tg.InF("r", TestGraph.T, ho, wo, O);
            var conv = tg.B.Emit1("Conv", [tg.B.Permute(xin, 0, 3, 1, 2), tg.WF("w", w, O, C, 3, 3), tg.WF("b", bias, O)], Attributes.From(
                ("pads", pads.Select(v => (long)v).ToArray())));
            var y = epilogue switch
            {
                "relu" => tg.B.Unary("Relu", conv),
                "residual-clip" => tg.B.Emit1("Clip", [tg.B.Add(conv, tg.B.Permute(rin, 0, 3, 1, 2))], Attributes.From(("min", 0f), ("max", 6f))),
                "silu" => tg.B.Mul(conv, tg.B.Unary("Sigmoid", conv)),
                _ => conv,
            };
            // A 1x1 max pool keeps the output channels-last up to the pool, so the conv takes the channels-last lowering.
            tg.B.Output("y", tg.B.Emit1("Pool", [y], Attributes.From(
                ("mode", "max"), ("kernel", new long[] { 1, 1 }), ("strides", new long[] { 1, 1 }), ("pads", new long[] { 0, 0, 0, 0 }),
                ("dilations", new long[] { 1, 1 }), ("ceil_mode", false), ("count_include_pad", false))));
            using var m = Kit.Compile(tg.B.Build(), maxTokens: N);
            Kit.Set(m, xin, x);
            Kit.Set(m, rin, res);
            var p = Kit.P(N);
            m.Run(p);
            var act = Kit.Out(m, "y", p);

            int[] one = [1, 1];
            var exp = SRef.Conv(Nchw(x, N, C, H, W), N, C, H, W, w, O, 3, 3, bias, one, pads, one, 1, out _, out _);
            var resNchw = Nchw(res, N, O, ho, wo);
            for (int i = 0; i < exp.Length; i++)
            {
                exp[i] = epilogue switch
                {
                    "relu" => Math.Max(0, exp[i]),
                    "residual-clip" => Math.Clamp(exp[i] + resNchw[i], 0f, 6f),
                    "silu" => exp[i] / (1f + MathF.Exp(-exp[i])),
                    _ => exp[i],
                };
            }

            return (exp, act);
        }
        finally
        {
            WinogradConvOp.Policy = prev;
        }
    }

    private static double Rms(float[] a, float[] b) => Math.Sqrt(a.Zip(b, (x, y) => ((double)x - y) * ((double)x - y)).Average());

    private static float[] Nchw(float[] nhwc, int n, int c, int h, int w)
    {
        var r = new float[nhwc.Length];
        for (int i = 0; i < n; i++)
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    for (int k = 0; k < c; k++)
                    {
                        r[((i * c + k) * h + y) * w + x] = nhwc[((i * h + y) * w + x) * c + k];
                    }
                }
            }
        }

        return r;
    }
}
